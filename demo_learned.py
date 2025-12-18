import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import tensorflow as tf

from take5_env import Take5Env
from take5_model import Take5Model


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


class DemoRollout:
    """Generates gameplay snapshots by mixing a trained agent with random players."""

    def __init__(
        self,
        model: Take5Model,
        num_players: int,
        num_steps: int,
        seed: int | None,
    ) -> None:
        self._model = model
        self._num_players = num_players
        self._num_steps = num_steps
        self._seed = seed
        self._rng = np.random.default_rng(seed)

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

        (
            model_card_value,
            model_row,
            model_card_probs,
            model_row_probs,
        ) = self._model_move(observation)

        # Map value-space probabilities (length 105) to hand-slot probabilities (length 10).
        hand_card_probs = np.zeros_like(hands[0], dtype=np.float32)
        for idx, value in enumerate(hands[0]):
            hand_card_probs[idx] = model_card_probs[value] if value != 0 else 0.0

        # Convert chosen card value to an index in the player's hand.
        model_card_matches = np.flatnonzero(hands[0] == model_card_value)
        model_card_index = int(model_card_matches[0]) if model_card_matches.size else 0

        card_choices = [model_card_index]
        row_choices = [model_row]
        card_values = [int(model_card_value)]

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
        return (
            action,
            np.asarray(card_values, dtype=np.int64),
            np.asarray(row_choices, dtype=np.int64),
            hand_card_probs,
            model_row_probs,
        )

    def _model_move(
        self, observation: dict[str, np.ndarray]
    ) -> tuple[int, int, np.ndarray, np.ndarray]:
        num_players = tf.convert_to_tensor(observation["num_players"], dtype=tf.int64)
        hands = tf.convert_to_tensor(observation["hands"], dtype=tf.int64)
        table = tf.convert_to_tensor(observation["table"], dtype=tf.int64)
        row_penalties = tf.convert_to_tensor(
            observation["row_penalties"], dtype=tf.float32
        )

        card_input = tf.concat(
            [
                tf.reshape(num_players, (1, 1)),  # 1, 1
                tf.reshape(hands[0], (1, -1)),  # 1, 10
                tf.reshape(table, (1, -1)),  # 1, 20
            ],
            axis=-1,
        )  # 1, 31

        row_penalty_input = tf.reshape(row_penalties, (1, -1)) * 0.1  # 1, 4

        mask = tf.concat(
            [
                tf.zeros((1, 1), tf.bool),
                tf.reduce_any(
                    tf.equal(
                        tf.range(1, 105, dtype=tf.int64)[None, None, :],  # 1, 1, 104
                        tf.reshape(hands[0], (1, -1, 1)),  # 1, 10, 1
                    ),
                    axis=1,
                ),
            ],
            axis=-1,
        )  # 1, 105

        model_out = self._model(
            (card_input, row_penalty_input, mask),
            training=False,
        )
        card_logits = model_out["card_logits"]
        row_logits = model_out["row_logits"]

        card_probs = tf.nn.softmax(card_logits, axis=-1)[0].numpy()
        row_probs = tf.nn.softmax(row_logits, axis=-1)[0].numpy()

        card_value = int(tf.argmax(card_logits, axis=-1, output_type=tf.int64).numpy()[0])
        row = int(tf.argmax(row_logits, axis=-1, output_type=tf.int64).numpy()[0])

        return card_value, row, card_probs, row_probs

    def _determine_winners(self, penalties: np.ndarray) -> np.ndarray:
        min_penalty = np.min(penalties)
        return np.where(penalties == min_penalty)[0]


def _load_model(model_width: int, embedding_dim: int, checkpoint_dir: str) -> Take5Model:
    model = Take5Model(embedding_dim=embedding_dim, width=model_width)
    model(
        (
            tf.zeros((1, 31), dtype=tf.int64),
            tf.zeros((1, 4), dtype=tf.float32),
            tf.zeros((1, 105), dtype=tf.bool),
        )
    )
    checkpoint = tf.train.Checkpoint(model=model)
    latest_path = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_path is None:
        msg = f"No checkpoint found in '{checkpoint_dir}'. Train the model before running the demo."
        raise FileNotFoundError(msg)
    checkpoint.restore(latest_path).expect_partial()
    return model


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
        description="Generate a Take5 replay file for the web visualizer."
    )
    parser.add_argument("--num-players", type=int, default=5, help="Number of players.")
    parser.add_argument("--num-steps", type=int, default=30, help="Steps to simulate.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory containing the trained model checkpoints.",
    )
    parser.add_argument(
        "--model-width", type=int, default=256, help="Hidden width used for Take5Model."
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=4, help="Embedding dimension for cards."
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
        default="./web-demo/public/replay.json",
        help="Path to write the replay JSON file consumed by the web viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = _load_model(args.model_width, args.embedding_dim, args.checkpoint_dir)
    rollout = DemoRollout(
        model=model,
        num_players=args.num_players,
        num_steps=args.num_steps,
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
