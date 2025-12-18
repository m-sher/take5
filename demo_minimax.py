import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np

from minimax_agent import MinimaxAgent
from take5_env import Take5Env


@dataclass
class StepSnapshot:
    step_index: int
    episode_index: int
    hands: np.ndarray
    table: np.ndarray
    row_penalties: np.ndarray
    total_penalties: np.ndarray
    win_counts: np.ndarray
    action_cards: np.ndarray
    action_rows: np.ndarray
    model_card_probs: np.ndarray
    model_row_probs: np.ndarray
    reward: np.ndarray

    def to_serializable(self) -> dict[str, Any]:
        return {
            "stepIndex": int(self.step_index),
            "episodeIndex": int(self.episode_index),
            "hands": self.hands.tolist(),
            "table": self.table.tolist(),
            "rowPenalties": self.row_penalties.tolist(),
            "totalPenalties": self.total_penalties.tolist(),
            "winCounts": self.win_counts.tolist(),
            "actionCards": self.action_cards.tolist(),
            "actionRows": self.action_rows.tolist(),
            "modelCardProbs": self.model_card_probs.tolist(),
            "modelRowProbs": self.model_row_probs.tolist(),
            "reward": self.reward.tolist(),
        }


class MinimaxRollout:
    """Generates gameplay snapshots using a minimax agent versus random players."""

    def __init__(
        self,
        num_players: int,
        num_steps: int,
        depth: int,
        seed: int | None,
    ) -> None:
        if num_players != 2:
            raise ValueError("MinimaxRollout currently supports exactly 2 players.")
        self._num_players = num_players
        self._num_steps = num_steps
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._agent = MinimaxAgent(depth=depth, player_index=0)

    def run(self) -> List[StepSnapshot]:
        env = Take5Env(num_players=self._num_players, seed=self._seed)
        time_step = env.reset()

        current_penalties = np.zeros(self._num_players, dtype=np.float32)
        win_counts = np.zeros(self._num_players, dtype=np.int32)
        snapshots: List[StepSnapshot] = []

        episode_index = 0
        steps_collected = 0

        last_reward = np.zeros((self._num_players,), dtype=np.float32)

        while steps_collected < self._num_steps:
            observation = time_step.observation
            (
                action,
                action_cards,
                action_rows,
                model_card_probs,
                model_row_probs,
            ) = self._build_action(time_step)

            snapshot = StepSnapshot(
                step_index=steps_collected,
                episode_index=episode_index,
                hands=observation["hands"].copy(),
                table=observation["table"].copy(),
                row_penalties=observation["row_penalties"].copy(),
                total_penalties=current_penalties.copy(),
                win_counts=win_counts.copy(),
                action_cards=action_cards.copy(),
                action_rows=action_rows.copy(),
                model_card_probs=model_card_probs.copy(),
                model_row_probs=model_row_probs.copy(),
                reward=last_reward.copy(),
            )
            snapshots.append(snapshot)

            next_time_step = env.step(action)
            reward = np.asarray(next_time_step.reward, dtype=np.float32)
            penalties = -reward
            current_penalties += penalties
            last_reward = reward

            is_last = bool(np.all(next_time_step.is_last()))
            if is_last:
                winners = self._determine_winners(current_penalties)
                win_counts[winners] += 1

            steps_collected += 1

            if is_last:
                episode_index += 1
                current_penalties = np.zeros(self._num_players, dtype=np.float32)
                time_step = env.reset()
            else:
                time_step = next_time_step

        if snapshots:
            snapshots[-1].win_counts = win_counts.copy()

        return snapshots

    def _build_action(
        self, time_step
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        observation = time_step.observation
        hands = observation["hands"]
        mask = hands != 0

        card_idx, row_idx = self._agent.choose_action(observation)

        card_choices = [int(card_idx)]
        row_choices = [int(row_idx)]

        card_values = [int(hands[0, card_idx])]

        for player_idx in range(1, self._num_players):
            valid_cards = np.flatnonzero(mask[player_idx])
            if valid_cards.size == 0:
                card_choice = 0
            else:
                card_choice = int(self._rng.choice(valid_cards))

            row_choice = int(self._rng.integers(0, 4))

            card_choices.append(card_choice)
            row_choices.append(row_choice)
            card_values.append(int(hands[player_idx, card_choice]))

        action = {
            "card_inds_to_play": np.asarray(card_choices, dtype=np.int64),
            "rows_to_take": np.asarray(row_choices, dtype=np.int64),
        }

        model_card_probs = np.zeros((hands.shape[1],), dtype=np.float32)
        model_card_probs[card_idx] = 1.0
        model_row_probs = np.zeros((4,), dtype=np.float32)
        model_row_probs[row_idx] = 1.0

        return (
            action,
            np.asarray(card_values, dtype=np.int64),
            np.asarray(row_choices, dtype=np.int64),
            model_card_probs,
            model_row_probs,
        )

    def _determine_winners(self, penalties: np.ndarray) -> np.ndarray:
        min_penalty = np.min(penalties)
        return np.where(penalties == min_penalty)[0]


def _serialize_replay(
    snapshots: List[StepSnapshot],
    num_players: int,
    num_steps: int,
    seed: int | None,
) -> dict[str, Any]:
    return {
        "metadata": {
            "numPlayers": num_players,
            "numSteps": num_steps,
            "seed": seed,
        },
        "snapshots": [snapshot.to_serializable() for snapshot in snapshots],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Take5 replay file using the minimax agent."
    )
    parser.add_argument("--num-players", type=int, default=2, help="Number of players.")
    parser.add_argument("--num-steps", type=int, default=30, help="Steps to simulate.")
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Search depth for the minimax agent (player 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible random opponents and environment init.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./web-demo/public/replay_minimax.json",
        help="Path to write the replay JSON file consumed by the web viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout = MinimaxRollout(
        num_players=args.num_players,
        num_steps=args.num_steps,
        depth=args.depth,
        seed=args.seed,
    )
    snapshots = rollout.run()

    replay_payload = _serialize_replay(
        snapshots=snapshots,
        num_players=args.num_players,
        num_steps=args.num_steps,
        seed=args.seed,
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(replay_payload, fp, indent=2)

    print(
        f"Wrote {len(snapshots)} steps to {output_path}. "
        + "Run `npm run dev` inside web-demo to view the replay."
    )


if __name__ == "__main__":
    main()
