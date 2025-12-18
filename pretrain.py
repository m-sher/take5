from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from minimax_agent import MinimaxAgent
from take5_env import Take5Env
from take5_model import Take5Model


def _build_inputs_for_player0(
    observation: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct PPO-aligned inputs and a card-value mask (0..104) for player 0."""
    num_players = observation["num_players"]  # scalar
    hands = observation["hands"]  # num_players, 10
    table = observation["table"]  # 4, 5
    row_penalties = observation["row_penalties"]  # 4

    card_input = np.concatenate(
        [
            np.reshape(num_players, (1, 1)),  # 1, 1
            hands[0, None, :],  # 1, 10
            np.reshape(table, (1, -1)),  # 1, 20
        ],
        axis=-1,
    ).astype(np.int64)  # 1, 31

    row_penalty_input = (row_penalties[None, :].astype(np.float32)) * 0.1  # 1, 4

    mask_values = np.zeros((1, 104), dtype=np.bool_)  # card values 1..104
    valid_cards = hands[0][hands[0] != 0]  # card values in hand
    mask_values[0, valid_cards - 1] = True
    mask = np.concatenate([np.zeros((1, 1), dtype=np.bool_), mask_values], axis=-1)
    return card_input, row_penalty_input, mask


def collect_dataset(
    num_steps_per_player: int, depth: int, seed: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out minimax (player 0) vs random others across player counts 2..10."""
    rng = np.random.default_rng(seed)
    player_counts = list(range(2, 11))

    card_inputs = []
    row_penalty_inputs = []
    masks = []
    card_labels = []
    row_labels = []
    value_targets = []

    for num_players in tqdm(player_counts, desc="Collecting data per player count"):
        env_seed = None if seed is None else int(rng.integers(0, 1_000_000_000))
        env = Take5Env(num_players=num_players, seed=env_seed)
        agent = MinimaxAgent(depth=depth, player_index=0)
        time_step = env.reset()
        
        # We use a nested progress bar if desired, but a single outer one per player count is cleaner.
        # Alternatively, we could tqdm the inner loop. Let's tqdm the inner loop for better granularity.
        pbar = tqdm(total=num_steps_per_player, desc=f"Players: {num_players}", leave=False)
        steps = 0

        while steps < num_steps_per_player:
            obs = time_step.observation
            card_input, row_penalty_input, mask = _build_inputs_for_player0(obs)

            card_idx, row_idx = agent.choose_action(obs)
            card_value = int(obs["hands"][0, card_idx])

            hands = obs["hands"]
            mask_all = hands != 0

            card_choices = [card_idx]
            row_choices = [row_idx]

            for player_idx in range(1, num_players):
                valid_cards = np.flatnonzero(mask_all[player_idx])
                if valid_cards.size == 0:
                    opp_card_idx = 0
                else:
                    opp_card_idx = int(rng.choice(valid_cards))
                opp_row_idx = int(rng.integers(0, 4))
                card_choices.append(opp_card_idx)
                row_choices.append(opp_row_idx)

            action = {
                "card_inds_to_play": np.asarray(card_choices, dtype=np.int64),
                "rows_to_take": np.asarray(row_choices, dtype=np.int64),
            }

            next_time_step = env.step(action)

            reward = float(next_time_step.reward[0]) * 0.1  # align with PPO scaling

            card_inputs.append(card_input[0])  # drop batch dim
            row_penalty_inputs.append(row_penalty_input[0])
            masks.append(mask[0])
            # Label should align with logits indices: position equals card value (1..104).
            card_labels.append(card_value)
            row_labels.append(int(row_idx))
            value_targets.append(reward)

            steps += 1
            pbar.update(1)

            if np.all(next_time_step.is_last()):
                time_step = env.reset()
            else:
                time_step = next_time_step
        
        pbar.close()

    return (
        np.asarray(card_inputs, dtype=np.int64),
        np.asarray(row_penalty_inputs, dtype=np.float32),
        np.asarray(masks, dtype=np.bool_),
        np.asarray(card_labels, dtype=np.int64),
        np.asarray(row_labels, dtype=np.int64),
        np.asarray(value_targets, dtype=np.float32)[:, None],
    )


def save_shard(
    output_dir: str,
    card_inputs: np.ndarray,
    row_penalty_inputs: np.ndarray,
    masks: np.ndarray,
    card_labels: np.ndarray,
    row_labels: np.ndarray,
    value_targets: np.ndarray,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shard_id = int(time.time())
    file_path = str(Path(output_dir) / f"shard_{shard_id}.tfrecord")

    with tf.io.TFRecordWriter(file_path) as writer:
        for i in range(len(card_inputs)):
            feature = {
                "card_input": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=card_inputs[i])
                ),
                "row_penalty_input": tf.train.Feature(
                    float_list=tf.train.FloatList(value=row_penalty_inputs[i])
                ),
                "mask": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=masks[i].astype(int))
                ),
                "card_label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[card_labels[i]])
                ),
                "row_label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[row_labels[i]])
                ),
                "value_target": tf.train.Feature(
                    float_list=tf.train.FloatList(value=value_targets[i])
                ),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    print(f"Saved {len(card_inputs)} samples to {file_path}")


def load_dataset(data_dir: str, batch_size: int) -> tf.data.Dataset:
    file_pattern = str(Path(data_dir) / "shard_*.tfrecord")
    files = tf.data.Dataset.list_files(file_pattern)

    def _parse_function(proto):
        feature_description = {
            "card_input": tf.io.FixedLenFeature([31], tf.int64),
            "row_penalty_input": tf.io.FixedLenFeature([4], tf.float32),
            "mask": tf.io.FixedLenFeature([105], tf.int64),
            "card_label": tf.io.FixedLenFeature([1], tf.int64),
            "row_label": tf.io.FixedLenFeature([1], tf.int64),
            "value_target": tf.io.FixedLenFeature([1], tf.float32),
        }
        parsed = tf.io.parse_single_example(proto, feature_description)
        return {
            "card_input": parsed["card_input"],
            "row_penalty_input": parsed["row_penalty_input"],
            "mask": tf.cast(parsed["mask"], tf.bool),
            "card_label": parsed["card_label"][0],
            "row_label": parsed["row_label"][0],
            "value_target": parsed["value_target"][0],
        }

    dataset = (
        files.interleave(
            lambda x: tf.data.TFRecordDataset(x).map(_parse_function),
            cycle_length=4,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .shuffle(buffer_size=10000)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def train_supervised(
    model: Take5Model,
    dataset: tf.data.Dataset,
    epochs: int,
    learning_rate: float,
    value_coef: float,
) -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        card_loss_meter = tf.keras.metrics.Mean()
        row_loss_meter = tf.keras.metrics.Mean()
        value_loss_meter = tf.keras.metrics.Mean()
        total_loss_meter = tf.keras.metrics.Mean()
        correct_cards = 0
        correct_rows = 0
        num_samples = 0

        for batch in dataset:
            card_input = tf.cast(batch["card_input"], tf.int64)
            row_penalty_input = tf.cast(batch["row_penalty_input"], tf.float32)
            mask = tf.cast(batch["mask"], tf.bool)
            card_label = tf.cast(batch["card_label"], tf.int64)
            row_label = tf.cast(batch["row_label"], tf.int64)
            value_target = tf.cast(batch["value_target"], tf.float32)

            batch_size_actual = tf.shape(card_input)[0]
            num_samples += int(batch_size_actual)

            with tf.GradientTape() as tape:
                model_out = model(
                    (card_input, row_penalty_input, mask),
                    training=True,
                )
                card_logits = model_out["card_logits"]
                row_logits = model_out["row_logits"]
                value_pred = model_out["value"]

                card_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=card_label, logits=card_logits
                    )
                )
                row_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=row_label, logits=row_logits
                    )
                )

                value_loss = tf.reduce_mean(tf.square(value_pred - value_target))

                total_loss = card_loss + row_loss + value_coef * value_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            card_loss_meter.update_state(card_loss)
            row_loss_meter.update_state(row_loss)
            value_loss_meter.update_state(value_loss)
            total_loss_meter.update_state(total_loss)

            card_pred = tf.argmax(card_logits, axis=-1, output_type=tf.int64)
            row_pred = tf.argmax(row_logits, axis=-1, output_type=tf.int64)
            correct_cards += int(
                tf.reduce_sum(tf.cast(card_pred == card_label, tf.int32))
            )
            correct_rows += int(tf.reduce_sum(tf.cast(row_pred == row_label, tf.int32)))

        epoch_time = time.perf_counter() - epoch_start
        card_acc = correct_cards / num_samples if num_samples > 0 else 0
        row_acc = correct_rows / num_samples if num_samples > 0 else 0
        samples_per_sec = num_samples / max(epoch_time, 1e-6)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"samples {num_samples} | "
            f"card {card_loss_meter.result():.4f} (acc {card_acc:.3f}) | "
            f"row {row_loss_meter.result():.4f} (acc {row_acc:.3f}) | "
            f"value {value_loss_meter.result():.4f} | "
            f"total {total_loss_meter.result():.4f} | "
            f"time {epoch_time:.2f}s | "
            f"{samples_per_sec:.1f} samples/s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised pretraining using minimax-generated actions."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to save/load dataset shards.",
    )
    parser.add_argument(
        "--num-steps-per-player",
        type=int,
        default=4000,
        help="Samples to collect for each player count in [2, 10]. Set to 0 to skip collection.",
    )
    parser.add_argument("--depth", type=int, default=2, help="Minimax search depth.")
    parser.add_argument("--embedding-dim", type=int, default=4, help="Embedding dim.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Adam learning rate."
    )
    parser.add_argument(
        "--value-coef", type=float, default=0.5, help="Weight for value MSE in loss."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints_pretrain",
        help="Directory to save pretrained checkpoints.",
    )
    parser.add_argument(
        "--model-width", type=int, default=256, help="Model hidden width."
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_steps_per_player > 0:
        (
            card_inputs,
            row_penalty_inputs,
            masks,
            card_labels,
            row_labels,
            value_targets,
        ) = collect_dataset(
            num_steps_per_player=args.num_steps_per_player,
            depth=args.depth,
            seed=args.seed,
        )

        save_shard(
            output_dir=args.data_dir,
            card_inputs=card_inputs,
            row_penalty_inputs=row_penalty_inputs,
            masks=masks,
            card_labels=card_labels,
            row_labels=row_labels,
            value_targets=value_targets,
        )

    dataset = load_dataset(data_dir=args.data_dir, batch_size=args.batch_size)

    model = Take5Model(embedding_dim=args.embedding_dim, width=args.model_width)
    model(
        (
            tf.zeros((1, 31), dtype=tf.int64),
            tf.zeros((1, 4), dtype=tf.float32),
            tf.zeros((1, 105), dtype=tf.bool),
        )
    )

    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_dir = Path(args.checkpoint_dir)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, str(ckpt_dir), max_to_keep=3
    )
    checkpoint.restore(checkpoint_manager.latest_checkpoint)

    train_supervised(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        value_coef=args.value_coef,
    )

    ckpt_path = checkpoint_manager.save()
    print(
        f"Saved pretrained model to {ckpt_path}. "
        "Use checkpoint_dir when starting PPO to initialize from this model."
    )


if __name__ == "__main__":
    main()
