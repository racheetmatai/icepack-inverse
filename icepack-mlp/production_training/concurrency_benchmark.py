"""Source-matched single-versus-two-process T4 throughput benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .campaign import FROZEN_L2, _package_root, _read_json, _verify_package
from .integrity import canonical_manifest_id, file_sha256
from .verify_run import verify as verify_training_run


SINGLE_JOB = "SQ01_CFG06_M01"
SECOND_JOB = "SQ02_CFG06_M01"
EPOCHS = 3
MINIMUM_AGGREGATE_SPEEDUP = 1.65
MAXIMUM_MEMORY_FRACTION = 0.90


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class _GpuSampler:
    def __init__(self):
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        command = [
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            utilization, used, total = [float(value.strip()) for value in result.stdout.strip().split(",")]
            self.samples.append({"elapsed_seconds": time.monotonic() - self.started,
                                 "utilization_percent": utilization,
                                 "memory_used_MiB": used, "memory_total_MiB": total})

    def _run(self) -> None:
        while not self.stop.wait(2):
            self._sample()

    def __enter__(self):
        self.started = time.monotonic(); self._sample(); self.thread.start(); return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set(); self.thread.join(timeout=5); self._sample()

    def summary(self) -> dict:
        if not self.samples:
            return {"sample_count": 0}
        return {
            "sample_count": len(self.samples),
            "mean_utilization_percent": sum(value["utilization_percent"] for value in self.samples) / len(self.samples),
            "peak_utilization_percent": max(value["utilization_percent"] for value in self.samples),
            "peak_memory_used_MiB": max(value["memory_used_MiB"] for value in self.samples),
            "memory_total_MiB": max(value["memory_total_MiB"] for value in self.samples),
            "peak_memory_fraction": max(value["memory_used_MiB"] / value["memory_total_MiB"] for value in self.samples),
        }


def _run_diagnostic(
    *, label: str, job_id: str, package_root: Path, bundle_root: Path,
    output_root: Path, package_id: str, xla_flags: str,
) -> dict:
    output = output_root / label
    if output.exists():
        if (output / "run_manifest.json").is_file():
            verification = verify_training_run(output)
            record_path = output / "benchmark_timing.json"
            if verification["passed"] and record_path.is_file():
                return _read_json(record_path)
        resolved = output.resolve()
        if resolved.parent != output_root.resolve():
            raise RuntimeError("Refusing unsafe benchmark cleanup")
        shutil.rmtree(resolved)
    log_path = output_root / f"{label}.log"
    command = [
        sys.executable, "-m", "production_training.train",
        "--dataset-dir", str(bundle_root / "dataset"), "--split-bundle", str(bundle_root / "splits"),
        "--job-id", job_id, "--lambda-l2", "0", "--output", str(output),
        "--max-epochs", str(EPOCHS), "--campaign-id", "jog-concurrency-benchmark-v1",
        "--shard-id", "0", "--production-package-manifest-id", package_id,
    ]
    environment = os.environ.copy(); environment["XLA_FLAGS"] = xla_flags
    mlp = package_root / "icepack-mlp"
    environment["PYTHONPATH"] = str(mlp) + os.pathsep + environment.get("PYTHONPATH", "")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=mlp, env=environment, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(f"Benchmark job {label} failed; see {log_path}")
    verification = verify_training_run(output)
    summary = _read_json(output / "training_summary.json")
    if summary.get("retained_model_artifacts") != ["best_model.keras"] or (output / "restored_best_model.keras").exists():
        raise ValueError("Benchmark run violated the sole-checkpoint contract")
    record = {
        "schema": "jog-concurrency-benchmark-job-v1", "label": label, "job_id": job_id,
        "elapsed_seconds": elapsed, "verification_passed": verification["passed"],
        "run_manifest_id": _read_json(output / "run_manifest.json")["manifest_id"],
        "validation_predictions_sha256": file_sha256(output / "validation_predictions.csv.gz"),
        "epochs": EPOCHS,
    }
    # This timing file is benchmark metadata outside the immutable training-run
    # manifest; it is deliberately written only after independent run verification.
    _write_json(output / "benchmark_timing.json", record)
    return record


def _benchmark_evidence_hashes(output: Path) -> dict:
    relative_paths = [
        "single_SQ01_CFG06_M01_E3/run_manifest.json",
        "single_SQ01_CFG06_M01_E3/validation_predictions.csv.gz",
        "single_SQ01_CFG06_M01_E3/benchmark_timing.json",
        "concurrent_SQ01_CFG06_M01_E3/run_manifest.json",
        "concurrent_SQ01_CFG06_M01_E3/validation_predictions.csv.gz",
        "concurrent_SQ01_CFG06_M01_E3/benchmark_timing.json",
        "concurrent_SQ02_CFG06_M01_E3/run_manifest.json",
        "concurrent_SQ02_CFG06_M01_E3/validation_predictions.csv.gz",
        "concurrent_SQ02_CFG06_M01_E3/benchmark_timing.json",
    ]
    return {relative: file_sha256(output / relative) for relative in relative_paths}


def _verify_existing_benchmark(output: Path, manifest: dict) -> None:
    if canonical_manifest_id(manifest) != manifest.get("manifest_id"):
        raise ValueError("Existing concurrency benchmark manifest ID mismatch")
    labels = ["single_SQ01_CFG06_M01_E3", "concurrent_SQ01_CFG06_M01_E3", "concurrent_SQ02_CFG06_M01_E3"]
    if not all(verify_training_run(output / label)["passed"] for label in labels):
        raise ValueError("Existing concurrency benchmark contains an invalid training run")
    if _benchmark_evidence_hashes(output) != manifest.get("evidence_sha256"):
        raise ValueError("Existing concurrency benchmark evidence hash mismatch")


def run_concurrency_benchmark(
    *, bundle_root: str | Path, output_root: str | Path,
    cuda_environment: str | Path, runtime_override: str | Path,
    package_root: str | Path | None = None,
) -> dict:
    package = Path(package_root).resolve() if package_root else _package_root()
    bundle = Path(bundle_root).resolve(); output = Path(output_root).resolve()
    if (output / "concurrency_benchmark_manifest.json").is_file():
        existing = _read_json(output / "concurrency_benchmark_manifest.json")
        _verify_existing_benchmark(output, existing)
        return existing
    output.mkdir(parents=True, exist_ok=True)
    package_manifest = _verify_package(package)
    environment = _read_json(Path(cuda_environment).resolve())
    if environment.get("passed") is not True:
        raise ValueError("CUDA environment manifest is not passing")
    override = _read_json(Path(runtime_override).resolve())
    xla_flags = override["XLA_FLAGS"]

    with _GpuSampler() as single_sampler:
        single = _run_diagnostic(
            label="single_SQ01_CFG06_M01_E3", job_id=SINGLE_JOB, package_root=package,
            bundle_root=bundle, output_root=output, package_id=package_manifest["manifest_id"], xla_flags=xla_flags,
        )
    concurrent_started = time.monotonic()
    with _GpuSampler() as concurrent_sampler:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="jog-benchmark") as executor:
            first = executor.submit(
                _run_diagnostic, label="concurrent_SQ01_CFG06_M01_E3", job_id=SINGLE_JOB,
                package_root=package, bundle_root=bundle, output_root=output,
                package_id=package_manifest["manifest_id"], xla_flags=xla_flags,
            )
            second = executor.submit(
                _run_diagnostic, label="concurrent_SQ02_CFG06_M01_E3", job_id=SECOND_JOB,
                package_root=package, bundle_root=bundle, output_root=output,
                package_id=package_manifest["manifest_id"], xla_flags=xla_flags,
            )
            concurrent_a, concurrent_b = first.result(), second.result()
    concurrent_wall = time.monotonic() - concurrent_started
    speedup = (2.0 * single["elapsed_seconds"]) / concurrent_wall
    deterministic = single["validation_predictions_sha256"] == concurrent_a["validation_predictions_sha256"]
    gpu_single = single_sampler.summary(); gpu_concurrent = concurrent_sampler.summary()
    memory_safe = bool(gpu_concurrent.get("peak_memory_fraction", 1.0) <= MAXIMUM_MEMORY_FRACTION)
    recommend_two = bool(
        single["verification_passed"] and concurrent_a["verification_passed"]
        and concurrent_b["verification_passed"] and deterministic
        and speedup >= MINIMUM_AGGREGATE_SPEEDUP and memory_safe
    )
    manifest = {
        "schema": "jog-t4-concurrency-benchmark-v1", "status": "complete", "created_utc": _utc(),
        "hostname": socket.gethostname(), "production_package_manifest_id": package_manifest["manifest_id"],
        "cuda_environment_manifest_id": environment.get("manifest_id"), "lambda_L2": FROZEN_L2,
        "epochs_per_diagnostic": EPOCHS, "held_out_test_accessed": False,
        "single": single, "concurrent": [concurrent_a, concurrent_b],
        "concurrent_pair_wall_seconds": concurrent_wall, "aggregate_speedup": speedup,
        "minimum_required_speedup": MINIMUM_AGGREGATE_SPEEDUP,
        "same_job_predictions_exact_across_single_and_concurrent": deterministic,
        "single_gpu_samples": gpu_single, "concurrent_gpu_samples": gpu_concurrent,
        "maximum_allowed_memory_fraction": MAXIMUM_MEMORY_FRACTION,
        "memory_safe": memory_safe, "two_workers_recommended": recommend_two,
        "decision_rule": "two workers only if all runs verify, repeated-job predictions are exact, speedup >=1.65, and peak memory <=90%",
        "evidence_sha256": _benchmark_evidence_hashes(output),
    }
    manifest["manifest_id"] = canonical_manifest_id(manifest)
    _write_json(output / "concurrency_benchmark_manifest.json", manifest)
    print(json.dumps({"aggregate_speedup": speedup, "memory_safe": memory_safe,
                      "deterministic": deterministic, "two_workers_recommended": recommend_two,
                      "manifest_id": manifest["manifest_id"]}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cuda-environment", required=True)
    parser.add_argument("--runtime-override", required=True)
    parser.add_argument("--package-root")
    args = parser.parse_args()
    run_concurrency_benchmark(
        bundle_root=args.bundle_root, output_root=args.output_root,
        cuda_environment=args.cuda_environment, runtime_override=args.runtime_override,
        package_root=args.package_root,
    )


if __name__ == "__main__":
    main()
