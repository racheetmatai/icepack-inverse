"""CLI for one immutable frozen production-training job."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import socket
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data import load_prepared_job, save_scalers
from .integrity import canonical_manifest_id, file_sha256
from .model import build_model, deterministic_sequence, reload_exact_best_model, training_callbacks
from .spec import FROZEN_POLICY, resolved_run_spec, validate_lambda_l2


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_json(temporary, value)
    os.replace(temporary, path)


class _ActiveProcessHeartbeat:
    """Process-owned evidence that survives loss of a notebook parent."""

    def __init__(self, output: Path, job_id: str):
        self.path = output / "active_process.json"
        self.job_id = job_id
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _write(self) -> None:
        _atomic_json(self.path, {
            "schema": "jog-training-active-process-v1", "status": "running",
            "job_id": self.job_id, "hostname": socket.gethostname(), "pid": os.getpid(),
            "heartbeat_utc": _utc(),
        })

    def _run(self) -> None:
        while not self.stop.wait(60):
            self._write()

    def __enter__(self):
        self._write(); self.thread.start(); return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set(); self.thread.join(timeout=5); self.path.unlink(missing_ok=True)


def _environment(tf=None) -> dict:
    result = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    if tf is not None:
        result["tensorflow"] = tf.__version__
        result["physical_gpus"] = [device.name for device in tf.config.list_physical_devices("GPU")]
        result["visible_gpus"] = [device.name for device in tf.config.get_visible_devices("GPU")]
        try:
            result["build_info"] = tf.sysconfig.get_build_info()
        except Exception:
            result["build_info"] = {}
    return result


def _finalize_manifest(output: Path, manifest: dict) -> None:
    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {"run_manifest.json", "active_process.json"}
    )
    manifest["output_sha256"] = {path.relative_to(output).as_posix(): file_sha256(path) for path in files}
    manifest["manifest_id"] = canonical_manifest_id(manifest)
    _write_json(output / "run_manifest.json", manifest)


def _scaler_record(scaler) -> dict:
    return {
        "class": type(scaler).__name__,
        "center": np.asarray(scaler.center_, dtype=float).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "n_features_in": int(scaler.n_features_in_),
    }


def _snapshot_source(output: Path) -> None:
    destination = output / "source_snapshot"
    destination.mkdir()
    package = Path(__file__).resolve().parent
    for name in (
        "__init__.py", "spec.py", "integrity.py", "data.py", "model.py",
        "train.py", "verify_run.py", "campaign.py", "concurrency_benchmark.py",
    ):
        source = package / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def _run_initialized(args: argparse.Namespace, output: Path) -> Path:
    started = _utc()
    prepared = load_prepared_job(args.dataset_dir, args.split_bundle, args.job_id, full_hash_check=not args.skip_full_hash_check)
    if "registered_lambda_L2" in prepared.job:
        registered = float(prepared.job["registered_lambda_L2"])
        if registered != validate_lambda_l2(args.lambda_l2):
            raise ValueError(f"Pilot registry freezes lambda_L2={registered}; received {args.lambda_l2}")
    spec = resolved_run_spec(prepared.job["configuration"], args.lambda_l2, max_epochs=args.max_epochs)
    _snapshot_source(output)
    save_scalers(prepared, output)
    _write_json(output / "resolved_spec.json", spec)
    _write_json(output / "scaler_parameters.json", {
        "fit_population": "training rows only", "input": _scaler_record(prepared.input_scaler),
        "target": _scaler_record(prepared.target_scaler),
    })
    identity = {
        "job": prepared.job, "split": prepared.split,
        "dataset_manifest_id": prepared.dataset_manifest["manifest_id"],
        "dataset_sha256": prepared.split_manifest["dataset_sha256"],
        "split_bundle_manifest_id": prepared.split_manifest["manifest_id"],
        "row_index_sha256": prepared.split_manifest["row_index_sha256"],
        "train_membership_id": prepared.split["train_membership_id"],
        "validation_membership_id": prepared.split["validation_membership_id"],
        "features": prepared.features,
    }
    _write_json(output / "data_identity.json", identity)

    manifest = {
        "schema": "jog-portable-training-run-v1", "status": "running", "mode": "data-smoke" if args.prepare_only else "model-training",
        "started_utc": started, "finished_utc": None, "job_id": args.job_id,
        "lambda_L2": validate_lambda_l2(args.lambda_l2), "data_identity": identity,
        "row_counts": {"training": len(prepared.train_row_ids), "validation": len(prepared.validation_row_ids)},
        "environment": _environment(),
    }
    campaign = {
        "campaign_id": getattr(args, "campaign_id", None),
        "shard_id": getattr(args, "shard_id", None),
        "production_package_manifest_id": getattr(args, "production_package_manifest_id", None),
    }
    if any(value is not None for value in campaign.values()):
        if not all(value is not None for value in campaign.values()):
            raise ValueError("Campaign provenance requires campaign_id, shard_id, and production package manifest ID together")
        manifest["campaign"] = campaign
    if args.prepare_only:
        manifest.update({
            "status": "complete", "finished_utc": _utc(),
            "purpose": "accepted-data preparation smoke only; no TensorFlow model was constructed or trained",
            "finite": bool(np.isfinite(prepared.x_train).all() and np.isfinite(prepared.y_train).all()
                           and np.isfinite(prepared.x_validation).all() and np.isfinite(prepared.y_validation).all()),
            "scaled_training_summary": {
                "input_shape": list(prepared.x_train.shape), "target_shape": list(prepared.y_train.shape),
                "validation_input_shape": list(prepared.x_validation.shape),
                "validation_target_shape": list(prepared.y_validation.shape),
            },
        })
        _finalize_manifest(output, manifest)
        return output

    tf, model = build_model(len(prepared.features), args.lambda_l2, int(prepared.job["model_seed"]))
    if not tf.config.list_physical_devices("GPU") and not args.allow_cpu_training:
        raise RuntimeError("No TensorFlow GPU detected; use --allow-cpu-training only for an explicitly bounded smoke test")
    batches = deterministic_sequence(
        tf, prepared.x_train, prepared.y_train, FROZEN_POLICY["batch_size"], int(prepared.job["shuffle_seed"]),
    )
    callbacks, lr_recorder = training_callbacks(tf, output)
    history = model.fit(
        batches, validation_data=(prepared.x_validation, prepared.y_validation),
        epochs=spec["policy"]["max_epochs"], callbacks=callbacks, verbose=2,
    )
    # ModelCheckpoint tracks the exact minimum val_data_mse. EarlyStopping uses
    # the frozen practical min_delta, so its in-memory "best" weights can be a
    # nearby epoch rather than the exact validation minimum. Reload the exact
    # checkpoint before predictions. best_model.keras is also the sole portable
    # production model; retaining a second identical model wastes ~4.4 MiB/run.
    model = reload_exact_best_model(tf, output)
    pd.DataFrame(history.history).assign(epoch=np.arange(1, len(history.epoch) + 1)).to_csv(output / "history.csv", index=False)
    pd.DataFrame({"epoch": np.arange(1, len(lr_recorder.values) + 1), "learning_rate": lr_recorder.values}).to_csv(
        output / "learning_rate_history.csv", index=False,
    )
    scaled_prediction = model.predict(prepared.x_validation, batch_size=FROZEN_POLICY["batch_size"], verbose=0)
    predicted = prepared.target_scaler.inverse_transform(scaled_prediction).reshape(-1)
    reference = prepared.target_scaler.inverse_transform(prepared.y_validation).reshape(-1)
    pd.DataFrame({
        "row_id": prepared.validation_row_ids, "reference_log_C": reference,
        "predicted_log_C": predicted,
    }).to_csv(output / "validation_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    val_series = np.asarray(history.history["val_data_mse"], dtype=float)
    best = int(np.nanargmin(val_series))
    _write_json(output / "training_summary.json", {
        "epochs_completed": len(history.epoch), "best_epoch_one_based": best + 1,
        "best_val_data_mse_scaled": float(val_series[best]),
        "prediction_and_export_source": "sole exact best_model.keras checkpoint reloaded after training",
        "retained_model_artifacts": ["best_model.keras"],
        "termination": "early_stopping" if len(history.epoch) < spec["policy"]["max_epochs"] else "maximum_epochs",
        "all_history_finite": bool(all(np.isfinite(np.asarray(values, dtype=float)).all() for values in history.history.values())),
    })
    manifest.update({"status": "complete", "finished_utc": _utc(), "environment": _environment(tf)})
    _finalize_manifest(output, manifest)
    return output


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with _ActiveProcessHeartbeat(output, args.job_id):
        return _run_initialized(args, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split-bundle", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--lambda-l2", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-cpu-training", action="store_true")
    parser.add_argument("--skip-full-hash-check", action="store_true", help="diagnostic only; never use for accepted runs")
    parser.add_argument("--campaign-id")
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--production-package-manifest-id")
    args = parser.parse_args()
    if args.skip_full_hash_check and not args.prepare_only:
        parser.error("--skip-full-hash-check is forbidden for model training")
    print(run(args))


if __name__ == "__main__":
    main()
