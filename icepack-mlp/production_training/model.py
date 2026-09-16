"""Lazy TensorFlow model construction and deterministic epoch batching."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def configure_tensorflow(model_seed: int):
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(model_seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass
    return tf


def build_model(input_count: int, lambda_l2: float, model_seed: int):
    tf = configure_tensorflow(model_seed)
    inputs = tf.keras.Input(shape=(int(input_count),), name="predictors")
    value = inputs
    regularizer = tf.keras.regularizers.L2(float(lambda_l2))
    for index in range(10):
        value = tf.keras.layers.Dense(
            200, kernel_regularizer=regularizer, bias_regularizer=None,
            name=f"hidden_dense_{index + 1:02d}",
        )(value)
        value = tf.keras.layers.BatchNormalization(name=f"hidden_bn_{index + 1:02d}")(value)
        value = tf.keras.layers.Activation("silu", name=f"hidden_silu_{index + 1:02d}")(value)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="scaled_log_C")(value)
    model = tf.keras.Model(inputs, outputs, name="jog_log_C_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-3),
        loss=tf.keras.losses.MeanSquaredError(name="data_mse_loss"),
        metrics=[tf.keras.metrics.MeanSquaredError(name="data_mse")],
    )
    return tf, model


def deterministic_sequence(tf, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle_seed: int):
    class EpochSequence(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.epoch = 0
            self.order = self._order_for_epoch(0)

        def _order_for_epoch(self, epoch: int) -> np.ndarray:
            return np.random.default_rng(np.random.SeedSequence([int(shuffle_seed), int(epoch)])).permutation(len(x))

        def __len__(self):
            return (len(x) + int(batch_size) - 1) // int(batch_size)

        def __getitem__(self, index):
            selected = self.order[index * int(batch_size):(index + 1) * int(batch_size)]
            return x[selected], y[selected]

        def on_epoch_end(self):
            self.epoch += 1
            self.order = self._order_for_epoch(self.epoch)

    return EpochSequence()


def training_callbacks(tf, output: Path):
    class LearningRateHistory(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.values = []

        def on_epoch_end(self, epoch, logs=None):
            self.values.append(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))

    lr_history = LearningRateHistory()
    return [
        tf.keras.callbacks.ModelCheckpoint(
            output / "best_model.keras", monitor="val_data_mse", mode="min", save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_data_mse", mode="min", factor=0.2, patience=15,
            min_delta=1.0e-4, min_lr=1.0e-6,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_data_mse", mode="min", patience=45, min_delta=1.0e-4,
            restore_best_weights=True,
        ),
        lr_history,
    ], lr_history


def reload_exact_best_model(tf, output: Path):
    """Reload the sole retained checkpoint at the exact validation minimum."""
    return tf.keras.models.load_model(output / "best_model.keras")
