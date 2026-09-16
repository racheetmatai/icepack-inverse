"""Frozen feature, architecture, and optimizer specifications."""

from __future__ import annotations


FEATURE_CONFIGURATIONS = {
    "CFG01": ["s", "h", "mag_s", "mag_h", "driving_stress", "surface_air_temp"],
    "CFG02": ["s", "h", "mag_s", "mag_h", "surface_air_temp"],
    "CFG03": ["b", "mag_b", "heatflux", "gravity_disturbance", "mag_anomaly"],
    "CFG04": ["b", "mag_b", "heatflux"],
    "CFG05": ["s", "h", "mag_s", "mag_h", "surface_air_temp", "b", "mag_b", "heatflux"],
    "CFG06": ["s", "h", "mag_s", "mag_h", "surface_air_temp", "b", "mag_b", "heatflux", "cos_theta_bs"],
}

ALL_PREDICTORS = [
    "s", "h", "mag_s", "mag_h", "driving_stress", "surface_air_temp",
    "b", "mag_b", "heatflux", "gravity_disturbance", "mag_anomaly",
    "cos_theta_bs",
]

FROZEN_POLICY = {
    "schema": "jog-portable-training-policy-v1",
    "target": "reference_log_C",
    "input_scaler": "sklearn.preprocessing.RobustScaler fitted on training rows only",
    "target_scaler": "sklearn.preprocessing.RobustScaler fitted on training rows only",
    "architecture": {
        "hidden_layers": 10,
        "hidden_units": 200,
        "hidden_order": ["Dense", "BatchNormalization", "SiLU"],
        "hidden_dense_kernel_regularizer": "L2(lambda_L2)",
        "regularized_parameters": "hidden Dense kernels only",
        "output": "one-unit linear Dense without regularization",
    },
    "optimizer": {"name": "Adam", "initial_learning_rate": 1.0e-3},
    "loss": "mean_squared_error including model regularization losses",
    "data_metric": "unpenalized mean_squared_error named data_mse",
    "batch_size": 1024,
    "max_epochs": 1500,
    "shuffle": "deterministic member-specific epoch permutation",
    "reduce_lr": {
        "monitor": "val_data_mse", "mode": "min", "factor": 0.2,
        "patience": 15, "min_delta": 1.0e-4, "min_lr": 1.0e-6,
    },
    "early_stopping": {
        "monitor": "val_data_mse", "mode": "min", "patience": 45,
        "min_delta": 1.0e-4, "restore_best_weights": True,
    },
    "checkpoint": {"monitor": "val_data_mse", "mode": "min", "save_best_only": True},
    "acceptance": "finite completed runs are retained; held-out performance never controls acceptance or reruns",
}


def validate_lambda_l2(value: float) -> float:
    result = float(value)
    if result < 0 or not result < float("inf"):
        raise ValueError("lambda_L2 must be finite and non-negative")
    return result


def resolved_run_spec(configuration: str, lambda_l2: float, *, max_epochs: int | None = None) -> dict:
    if configuration not in FEATURE_CONFIGURATIONS:
        raise ValueError(f"Unknown feature configuration: {configuration}")
    epochs = FROZEN_POLICY["max_epochs"] if max_epochs is None else int(max_epochs)
    if epochs < 1 or epochs > FROZEN_POLICY["max_epochs"]:
        raise ValueError("max_epochs override must be between 1 and the frozen 1500-epoch cap")
    return {
        "schema": "jog-resolved-training-spec-v1",
        "configuration": configuration,
        "features": FEATURE_CONFIGURATIONS[configuration],
        "lambda_L2": validate_lambda_l2(lambda_l2),
        "target": FROZEN_POLICY["target"],
        "policy": {**FROZEN_POLICY, "max_epochs": epochs},
    }
