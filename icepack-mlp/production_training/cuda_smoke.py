"""Bounded end-to-end CUDA rehearsal of one registered production job."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from .data import load_prepared_job
from .integrity import canonical_manifest_id, file_sha256
from .train import run as run_training
from .verify_run import verify as verify_training_run


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty smoke output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    environment_path = Path(args.cuda_environment).resolve()
    environment = _read_json(environment_path)
    if environment.get("passed") is not True:
        raise ValueError("CUDA environment manifest is not passing")
    if float(args.lambda_l2) != 0.0:
        raise ValueError("The frozen production lambda_L2 is 0")
    if not 1 <= int(args.epochs) <= 5:
        raise ValueError("CUDA smoke must use between 1 and 5 epochs")

    run_dir = output / "training_run"
    run_training(SimpleNamespace(
        dataset_dir=args.dataset_dir,
        split_bundle=args.split_bundle,
        job_id=args.job_id,
        lambda_l2=0.0,
        output=str(run_dir),
        max_epochs=int(args.epochs),
        prepare_only=False,
        allow_cpu_training=False,
        skip_full_hash_check=False,
    ))
    verification = verify_training_run(run_dir)

    # Reload the accepted split and saved scalers independently.
    prepared = load_prepared_job(args.dataset_dir, args.split_bundle, args.job_id, full_hash_check=True)
    input_scaler = joblib.load(run_dir / "input_scaler.joblib")
    target_scaler = joblib.load(run_dir / "target_scaler.joblib")
    scaler_match = bool(
        np.array_equal(input_scaler.center_, prepared.input_scaler.center_)
        and np.array_equal(input_scaler.scale_, prepared.input_scaler.scale_)
        and np.array_equal(target_scaler.center_, prepared.target_scaler.center_)
        and np.array_equal(target_scaler.scale_, prepared.target_scaler.scale_)
    )

    import tensorflow as tf

    count = min(int(args.prediction_rows), len(prepared.x_validation))
    fixed_x = prepared.x_validation[:count]
    exact_model_a = tf.keras.models.load_model(run_dir / "best_model.keras")
    exact_model_b = tf.keras.models.load_model(run_dir / "best_model.keras")
    prediction_a = exact_model_a.predict(fixed_x, batch_size=1024, verbose=0)
    prediction_repeat = exact_model_a.predict(fixed_x, batch_size=1024, verbose=0)
    prediction_reload = exact_model_b.predict(fixed_x, batch_size=1024, verbose=0)

    deterministic_repeat = bool(np.array_equal(prediction_a, prediction_repeat))
    deterministic_reload = bool(np.array_equal(prediction_a, prediction_reload))
    finite = bool(np.isfinite(prediction_a).all())

    saved = pd.read_csv(run_dir / "validation_predictions.csv.gz", nrows=count)
    predicted_log_c = target_scaler.inverse_transform(prediction_a).reshape(-1)
    sample_path = output / "fixed_validation_prediction_sample.csv.gz"
    pd.DataFrame({
        "row_id": prepared.validation_row_ids[:count],
        "predicted_log_C": predicted_log_c,
    }).to_csv(sample_path, index=False, compression={"method": "gzip", "mtime": 0})
    # Compare like with like after both arrays have passed through the persisted
    # CSV representation. Comparing parsed CSV floats with pre-serialization
    # in-memory floats can fail at an irrelevant parser/formatting roundoff.
    persisted_sample = pd.read_csv(sample_path)
    saved_prediction_match = bool(
        saved["row_id"].astype(str).tolist() == persisted_sample["row_id"].astype(str).tolist()
        and np.array_equal(
            saved["predicted_log_C"].to_numpy(),
            persisted_sample["predicted_log_C"].to_numpy(),
        )
    )

    checks = {
        "cuda_environment_passed": True,
        "training_run_verified": bool(verification["passed"]),
        "selected_lambda_L2_is_zero": True,
        "production_job_registry_used": not str(args.job_id).startswith("L2CAL_"),
        "bounded_epoch_count": True,
        "saved_scalers_match_train_only_refit": scaler_match,
        "predictions_finite": finite,
        "same_loaded_model_repeat_is_exact": deterministic_repeat,
        "independent_reload_is_exact": deterministic_reload,
        "sole_exact_checkpoint_contract": bool(
            (run_dir / "best_model.keras").is_file()
            and not (run_dir / "restored_best_model.keras").exists()
        ),
        "saved_validation_predictions_match_exact_checkpoint": saved_prediction_match,
        "held_out_test_accessed": False,
    }
    passed = all(value for key, value in checks.items() if key != "held_out_test_accessed") and not checks["held_out_test_accessed"]
    manifest = {
        "schema": "jog-production-cuda-smoke-v1",
        "status": "complete" if passed else "failed",
        "created_utc": _utc(),
        "purpose": "bounded operational rehearsal only; not an accepted production model",
        "job_id": args.job_id,
        "lambda_L2": 0.0,
        "epochs": int(args.epochs),
        "fixed_validation_rows": count,
        "checks": checks,
        "training_run_manifest_id": _read_json(run_dir / "run_manifest.json")["manifest_id"],
        "training_run_manifest_sha256": file_sha256(run_dir / "run_manifest.json"),
        "cuda_environment_manifest_id": environment.get("manifest_id"),
        "cuda_environment_sha256": file_sha256(environment_path),
        "runtime": {"python": sys.version, "platform": platform.platform(), "tensorflow": tf.__version__},
        "output_sha256": {sample_path.name: file_sha256(sample_path)},
    }
    manifest["manifest_id"] = canonical_manifest_id(manifest)
    _write_json(output / "cuda_smoke_manifest.json", manifest)
    if not passed:
        raise RuntimeError(json.dumps(manifest, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split-bundle", required=True)
    parser.add_argument("--cuda-environment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-id", default="SQ01_CFG06_M01")
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--prediction-rows", type=int, default=4096)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
