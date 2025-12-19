from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _compute_row_penalty(row: np.ndarray) -> int:
    """Replicates Take5Env._compute_row_penalty as a standalone helper."""
    penalty = 0
    for card in row:
        if card == 0:
            continue
        if card == 55:
            penalty += 7
        elif card % 11 == 0:
            penalty += 5
        elif card % 10 == 0:
            penalty += 3
        elif card % 5 == 0:
            penalty += 2
        else:
            penalty += 1
    return penalty


def _resolve_card(
    card: int, row_to_take: int, table: np.ndarray
) -> tuple[int, np.ndarray]:
    """Apply a single card to the table, returning the incurred penalty and new table."""
    end_cards = table.max(axis=-1)
    diffs = np.maximum(0, card - end_cards)

    if np.all(diffs == 0):
        # row_to_take = np.argmin([_compute_row_penalty(row) for row in table])
        return _take_row(card, row_to_take, table)

    row_to_place = np.argmin(np.where(diffs == 0, np.inf, diffs))
    num_cards = (table[row_to_place] != 0).astype(np.int64).sum()

    if num_cards == 5:
        return _take_row(card, row_to_place, table)

    table[row_to_place][num_cards] = card
    return 0, table


def _take_row(card: int, row_to_take: int, table: np.ndarray) -> tuple[int, np.ndarray]:
    """Take a row and restart it with the provided card."""
    penalty = _compute_row_penalty(table[row_to_take])
    table[row_to_take][0] = card
    table[row_to_take][1:] = 0
    return penalty, table


def _best_row_choices(table: np.ndarray) -> np.ndarray:
    """Return rows ordered from lowest to highest penalty."""
    penalties = np.asarray([_compute_row_penalty(row) for row in table], dtype=np.int64)
    return np.argsort(penalties, kind='stable')


@dataclass(frozen=True)
class GameState:
    """Lightweight, immutable snapshot of the current game for search."""

    table: np.ndarray
    hands: np.ndarray

    def available_cards(self, player_index: int) -> np.ndarray:
        return np.nonzero(self.hands[player_index])[0]

    def is_terminal(self) -> bool:
        return not np.any(self.hands != 0)

    def copy(self) -> "GameState":
        return GameState(table=self.table.copy(), hands=self.hands.copy())


class MinimaxAgent:
    """Two-player minimax planner with alpha-beta pruning for Take5Env."""

    def __init__(self, depth: int = 2, player_index: int = 0) -> None:
        if depth < 1:
            raise ValueError("Search depth must be at least 1.")
        self._depth = depth
        self._player_index = player_index
        self._cache: dict[tuple, tuple[float, tuple[int, int] | None]] = {}

    def choose_action(self, observation: dict[str, np.ndarray]) -> tuple[int, int]:
        """Return (card_index, row_index) for the controlled player.

        Designed for two-player lookahead; for 3+ players it still returns a move
        but only models a single opposing player during search. Row index is chosen
        alongside the card, mirroring the environment's simultaneous action
        submission.
        """
        num_players = int(observation["num_players"])
        if num_players < 2:
            raise ValueError(
                f"MinimaxAgent requires at least 2 players (got {num_players})."
            )

        table = np.asarray(observation["table"], dtype=np.int64).copy()
        hands = np.asarray(observation["hands"], dtype=np.int64).copy()
        state = GameState(table=table, hands=hands)

        # Reset per-search cache.
        self._cache.clear()

        _, action = self._max_node(
            state=state, depth=self._depth, alpha=-np.inf, beta=np.inf
        )
        if action is None:
            raise RuntimeError("No valid action available for minimax search.")

        return action

    def _max_node(
        self, state: GameState, depth: int, alpha: float, beta: float
    ) -> Tuple[float, tuple[int, int] | None]:
        if depth == 0 or state.is_terminal():
            return 0.0, None

        my_cards = state.available_cards(self._player_index)
        if my_cards.size == 0:
            return 0.0, None

        cache_key = ("max", depth, state.table.tobytes(), state.hands.tobytes())
        if cache_key in self._cache:
            return self._cache[cache_key]

        best_score = -np.inf
        best_action: tuple[int, int] | None = None
        rows = _best_row_choices(state.table)

        # Pre-evaluate own penalty for move ordering and pruning.
        action_candidates: list[tuple[float, int, int]] = []
        for card_idx in my_cards:
            for row_choice in rows:
                _, penalties = self._play_round(
                    state=state,
                    my_card_idx=int(card_idx),
                    my_row=int(row_choice),
                    opp_card_idx=None,
                    opp_row=None,
                )
                my_penalty = float(penalties[self._player_index])
                action_candidates.append((my_penalty, int(card_idx), int(row_choice)))

        # Sort by increasing self-penalty.
        action_candidates.sort(key=lambda x: x[0])

        if action_candidates:
            zero_penalty = [(p, c, r) for (p, c, r) in action_candidates if p == 0.0]
            if zero_penalty:
                ordered_actions = zero_penalty
            else:
                limit = max(1, len(action_candidates) // 2)  # cap at ~50% of moves
                ordered_actions = action_candidates[:limit]
        else:
            ordered_actions = []

        for _, card_idx, row_choice in ordered_actions:
            score = self._min_node(
                state=state,
                depth=depth,
                my_card_idx=card_idx,
                my_row=row_choice,
                alpha=alpha,
                beta=beta,
            )
            if score > best_score:
                best_score = score
                best_action = (card_idx, row_choice)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                result = (best_score, best_action)
                self._cache[cache_key] = result
                return result

        result = (best_score, best_action)
        self._cache[cache_key] = result
        return result

    def _min_node(
        self,
        state: GameState,
        depth: int,
        my_card_idx: int,
        my_row: int,
        alpha: float,
        beta: float,
    ) -> float:
        cache_key = (
            "min",
            depth,
            my_card_idx,
            my_row,
            state.table.tobytes(),
            state.hands.tobytes(),
        )
        if cache_key in self._cache:
            return self._cache[cache_key][0]

        opponent = (self._player_index + 1) % state.hands.shape[0]
        opp_cards = state.available_cards(opponent)

        if opp_cards.size == 0:
            # Only we move; no opposition this round.
            next_state, penalties = self._play_round(
                state=state,
                my_card_idx=my_card_idx,
                my_row=my_row,
                opp_card_idx=None,
                opp_row=None,
            )
            step_score = self._score(penalties)
            if depth > 1 and not next_state.is_terminal():
                child_score, _ = self._max_node(
                    state=next_state, depth=depth - 1, alpha=alpha, beta=beta
                )
                return step_score + child_score
            return step_score

        worst_for_me = np.inf
        rows = _best_row_choices(state.table)

        for opp_card_idx in opp_cards:
            for opp_row in rows:
                next_state, penalties = self._play_round(
                    state=state,
                    my_card_idx=my_card_idx,
                    my_row=my_row,
                    opp_card_idx=int(opp_card_idx),
                    opp_row=int(opp_row),
                )

                step_score = self._score(penalties)
                if depth > 1 and not next_state.is_terminal():
                    child_score, _ = self._max_node(
                        state=next_state, depth=depth - 1, alpha=alpha, beta=beta
                    )
                    total_score = step_score + child_score
                else:
                    total_score = step_score

                if total_score < worst_for_me:
                    worst_for_me = total_score
                beta = min(beta, worst_for_me)
                if beta <= alpha:
                    self._cache[cache_key] = (worst_for_me, None)
                    return worst_for_me

        self._cache[cache_key] = (worst_for_me, None)
        return worst_for_me

    def _play_round(
        self,
        state: GameState,
        my_card_idx: int,
        my_row: int,
        opp_card_idx: int | None,
        opp_row: int | None,
    ) -> tuple[GameState, np.ndarray]:
        table = state.table.copy()
        hands = state.hands.copy()
        num_players = hands.shape[0]

        cards = np.zeros((num_players,), dtype=np.int64)
        rows = np.zeros((num_players,), dtype=np.int64)

        cards[self._player_index] = hands[self._player_index, my_card_idx]
        rows[self._player_index] = my_row
        hands[self._player_index, my_card_idx] = 0

        if opp_card_idx is not None and opp_row is not None:
            opponent = 1 - self._player_index
            cards[opponent] = hands[opponent, opp_card_idx]
            rows[opponent] = opp_row
            hands[opponent, opp_card_idx] = 0

        penalties = np.zeros((num_players,), dtype=np.float32)

        active_players = [p for p in range(num_players) if cards[p] != 0]
        if active_players:
            resolution_order = sorted(active_players, key=lambda p: cards[p])
            for player in resolution_order:
                penalty, table = _resolve_card(
                    int(cards[player]), int(rows[player]), table
                )
                penalties[player] = penalty

        return GameState(table=table, hands=hands), penalties

    def _score(self, penalties: np.ndarray) -> float:
        """Positive is good for the agent (opponent penalty - self penalty)."""
        opponent = 1 - self._player_index
        return float(penalties[opponent] - penalties[self._player_index])
