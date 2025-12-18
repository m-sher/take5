import tensorflow as tf
import keras
from tensorflow_probability import distributions
from keras import layers


class Take5Model(keras.Model):
    def __init__(self, embedding_dim, width):
        super().__init__()

        self._card_embedding = layers.Embedding(input_dim=105, output_dim=embedding_dim)

        self._flatten = layers.Flatten()

        self._body = keras.Sequential(
            [
                layers.Dense(width // 2, activation="relu"),
                layers.Dense(width, activation="relu"),
                layers.Dense(width, activation="relu"),
                layers.Dense(width // 2, activation="relu"),
                layers.Dense(width // 4, activation="relu"),
            ],
            name="body",
        )

        self._card_head = layers.Dense(embedding_dim, name="card_head")
        self._row_head = layers.Dense(4, name="row_head")
        self._value_head = layers.Dense(1, name="value_head")

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        card_input, row_penalty_input, output_mask = inputs
        embedded_cards = self._flatten(self._card_embedding(card_input))
        body_in = tf.concat([embedded_cards, row_penalty_input], axis=-1)
        body_out = self._body(body_in, training=training)
        card_projection = self._card_head(body_out, training=training)
        card_logits = tf.linalg.matmul(
            card_projection, self._card_embedding.embeddings, transpose_b=True
        )
        masked_card_logits = tf.where(
            output_mask, card_logits, tf.constant(-1e9, dtype=tf.float32)
        )
        row_logits = self._row_head(body_out, training=training)
        value = self._value_head(body_out, training=training)

        return {
            "card_logits": masked_card_logits,
            "row_logits": row_logits,
            "value": value,
        }

    @tf.function(jit_compile=True)
    def act(self, inputs, greedy=False):
        call_out = self.call(inputs, training=False)

        card_dist = distributions.Categorical(
            logits=call_out["card_logits"], dtype=tf.int64
        )
        row_dist = distributions.Categorical(
            logits=call_out["row_logits"], dtype=tf.int64
        )

        if greedy:
            card_action = tf.argmax(
                call_out["card_logits"], axis=-1, output_type=tf.int64
            )
            row_action = tf.argmax(
                call_out["row_logits"], axis=-1, output_type=tf.int64
            )
        else:
            card_action = card_dist.sample()
            row_action = row_dist.sample()

        card_log_prob = card_dist.log_prob(card_action)
        row_log_prob = row_dist.log_prob(row_action)

        return {
            "card_action": card_action,
            "card_log_prob": card_log_prob,
            "row_action": row_action,
            "row_log_prob": row_log_prob,
            "value": call_out["value"],
        }
