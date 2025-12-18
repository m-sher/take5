from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np


class Take5Env(py_environment.PyEnvironment):
    def __init__(self, num_players: int, seed: int | None = None) -> None:
        self._num_players = np.array(num_players, dtype=np.int64)

        if seed is not None:
            np.random.seed(seed)

        self._table = np.zeros((4, 5), dtype=np.int64)
        self._hands = np.zeros((num_players, 10), dtype=np.int64)

        self._observation_spec = {
            "num_players": array_spec.BoundedArraySpec(
                shape=(), dtype=np.int64, minimum=2, maximum=10, name="num_players"
            ),
            "num_cards": array_spec.BoundedArraySpec(
                shape=(num_players,),
                dtype=np.int64,
                minimum=1,
                maximum=10,
                name="num_cards",
            ),
            "hands": array_spec.BoundedArraySpec(
                shape=(num_players, 10),
                dtype=np.int64,
                minimum=0,
                maximum=104,
                name="hands",
            ),
            "table": array_spec.BoundedArraySpec(
                shape=(4, 5), dtype=np.int64, minimum=0, maximum=104, name="table"
            ),
            "row_penalties": array_spec.BoundedArraySpec(
                shape=(4,), dtype=np.int64, minimum=0, maximum=27, name="row_penalties"
            ),
        }

        self._action_spec = {
            "card_inds_to_play": array_spec.BoundedArraySpec(
                shape=(num_players,),
                dtype=np.int64,
                minimum=0,
                maximum=9,
                name="cards",
            ),
            "rows_to_take": array_spec.BoundedArraySpec(
                shape=(num_players,),
                dtype=np.int64,
                minimum=0,
                maximum=3,
                name="rows_to_take",
            ),
        }

        self._reward_spec = array_spec.BoundedArraySpec(
            shape=(num_players,),
            dtype=np.float32,
            minimum=-27.0,
            maximum=0.0,
            name="self_penalty",
        )

        self.reset()

    def action_spec(self) -> array_spec.BoundedArraySpec:
        return self._action_spec

    def observation_spec(self) -> dict[str, array_spec.ArraySpec]:
        return self._observation_spec

    def reward_spec(self) -> dict[str, array_spec.ArraySpec]:
        return self._reward_spec

    def _create_observation(self):
        num_players = self._num_players
        num_cards = (self._hands != 0).astype(np.int64).sum(axis=-1)
        hands = self._hands
        table = self._table
        row_penalties = self._compute_row_penalties(self._table)
        return {
            "num_players": num_players,
            "num_cards": num_cards,
            "hands": hands,
            "table": table,
            "row_penalties": row_penalties,
        }

    def _reset(self) -> ts.TimeStep:
        self._episode_ended = False

        self._used_cards = np.random.choice(
            np.arange(1, 105, dtype=np.int64),
            size=(self._num_players * 10 + 4,),
            replace=False,
        )

        self._table = np.zeros((4, 5), dtype=np.int64)

        self._table[:, 0] = self._used_cards[:4]
        self._hands[:, :] = np.reshape(self._used_cards[4:], (self._num_players, 10))

        observation = self._create_observation()

        return ts.restart(observation=observation, batch_size=self._num_players)

    def _step(self, action: dict[str, np.ndarray]) -> ts.TimeStep:
        if self._episode_ended:
            return self.reset()

        card_inds_to_play = action["card_inds_to_play"]
        rows_to_take = action["rows_to_take"]

        cards = self._hands[np.arange(self._num_players), card_inds_to_play]
        self._hands = np.where(
            self._hands == cards[..., None], np.zeros_like(self._hands), self._hands
        )

        resolution_order = np.argsort(cards)

        table = self._table.copy()
        reward = np.zeros((self._num_players,), dtype=np.float32)
        for player in resolution_order:
            penalty, table = self._resolve_card(
                cards[player], rows_to_take[player], table
            )
            reward[player] = -penalty
        self._table = table

        num_cards = (self._hands != 0).astype(np.int64).sum(axis=-1)

        observation = self._create_observation()

        self._episode_ended = np.all(num_cards == 0)

        if self._episode_ended:
            return ts.termination(observation=observation, reward=reward)
        else:
            return ts.transition(observation=observation, reward=reward)

    def _resolve_card(
        self, card: int, row_to_take: int, table: np.ndarray
    ) -> tuple[int, np.ndarray]:
        end_cards = table.max(axis=-1)

        diffs = np.maximum(0, card - end_cards)
        if np.all(diffs == 0):
            # row_to_take = np.argmin(self._compute_row_penalties(table))
            return self._take_row(card, row_to_take, table)

        row_to_place = np.argmin(np.where(diffs == 0, np.inf, diffs))
        num_cards = (table[row_to_place] != 0).astype(np.int64).sum()
        if num_cards == 5:
            return self._take_row(card, row_to_place, table)

        table[row_to_place][num_cards] = card
        return 0, table

    def _take_row(
        self, card: int, row_to_take: int, table: np.ndarray
    ) -> tuple[int, np.ndarray]:
        penalty = self._compute_row_penalty(table[row_to_take])
        table[row_to_take][0] = card
        table[row_to_take][1:] = 0
        return penalty, table

    def _compute_row_penalties(self, table: np.ndarray) -> np.ndarray:
        row_penalties = np.zeros((table.shape[0]), dtype=np.int64)
        for i, row in enumerate(table):
            row_penalties[i] = self._compute_row_penalty(row)
        return row_penalties

    def _compute_row_penalty(self, row: np.ndarray) -> int:
        penalty = 0
        for card in row:
            if card != 0:
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
