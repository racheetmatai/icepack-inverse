"""Resume-safe multi-shard production campaign for LEAP JupyterHub workers."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .integrity import canonical_manifest_id, file_sha256, verify_declared_outputs
from .verify_run import verify as verify_training_run


BASELINE_BUNDLE_ID = "sha256-json-v1-303df5600ae81d7a21dd66b333fa76929669ce0af880bbe4a22bc8d9aed4a625"
SHARD_COUNT = 24
FROZEN_L2 = 0.0
STALE_HEARTBEAT_SECONDS = 15 * 60
MINIMUM_FREE_BYTES = 8 * 1024**3


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _verify_package(root: Path) -> dict:
    manifest = _read_json(root / "production_package_manifest.json")
    if manifest.get("schema") != "jog-production-launcher-package-v1":
        raise ValueError("Production package schema mismatch")
    if canonical_manifest_id(manifest) != manifest.get("manifest_id"):
        raise ValueError("Production package manifest ID mismatch")
    verify_declared_outputs(root, manifest)
    return manifest


def _nvidia_identity() -> dict:
    command = [
        "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    records = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(records) != 1:
        raise RuntimeError(f"Expected exactly one visible GPU; found {len(records)}")
    index, name, uuid, driver, memory = [value.strip() for value in records[0].split(",", 4)]
    return {"index": int(index), "name": name, "uuid": uuid, "driver": driver, "memory_total_MiB": int(memory)}


def _load_shards(package_root: Path) -> tuple[dict[int, list[str]], dict[str, dict]]:
    rows = _rows(package_root / "campaign" / "production_shards.csv")
    shards: dict[int, list[str]] = {index: [] for index in range(SHARD_COUNT)}
    records: dict[str, dict] = {}
    for row in rows:
        shard = int(row["shard_id"])
        if shard not in shards or row["job_id"] in records:
            raise ValueError("Invalid or duplicate production shard record")
        shards[shard].append(row["job_id"])
        records[row["job_id"]] = row
    if len(records) != 660 or any(len(value) not in {27, 28} for value in shards.values()):
        raise ValueError("Frozen shard registry is incomplete or imbalanced")
    return shards, records


def _validate_requested_shards(shard_ids: Iterable[int], workers: int) -> list[int]:
    selected = [int(value) for value in shard_ids]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("shard_ids must be a non-empty unique sequence")
    if any(value < 0 or value >= SHARD_COUNT for value in selected):
        raise ValueError(f"shard_ids must be in [0,{SHARD_COUNT - 1}]")
    if workers not in {1, 2} or workers > len(selected):
        raise ValueError("workers must be 1 or 2 and cannot exceed assigned shard count")
    return selected


def preflight(
    *, bundle_root: str | Path, runs_root: str | Path,
    cuda_environment: str | Path, runtime_override: str | Path,
    shard_ids: Iterable[int], workers: int, package_root: str | Path | None = None,
) -> dict:
    package = Path(package_root).resolve() if package_root else _package_root()
    bundle = Path(bundle_root).resolve()
    runs = Path(runs_root).resolve()
    selected = _validate_requested_shards(shard_ids, workers)
    package_manifest = _verify_package(package)
    baseline = _read_json(bundle / "bundle_manifest.json")
    if baseline.get("manifest_id") != BASELINE_BUNDLE_ID:
        raise ValueError("Baseline CUDA bundle identity mismatch")
    environment_path = Path(cuda_environment).resolve()
    environment = _read_json(environment_path)
    if environment.get("passed") is not True:
        raise ValueError("CUDA environment manifest is not passing")
    override_path = Path(runtime_override).resolve()
    override = _read_json(override_path)
    if override.get("schema") != "jog-cuda-runtime-override-v1" or not override.get("XLA_FLAGS"):
        raise ValueError("CUDA runtime override is missing or invalid")
    libdevice = Path(override["libdevice_path"])
    if not libdevice.is_file() or file_sha256(libdevice) != override["libdevice_sha256"]:
        raise ValueError("Configured CUDA libdevice is missing or has the wrong hash")
    shards, _ = _load_shards(package)
    if any(not shards[value] for value in selected):
        raise ValueError("Requested an empty shard")
    runs.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(runs).free
    if free < MINIMUM_FREE_BYTES:
        raise RuntimeError(f"Only {free / 1024**3:.2f} GiB free; at least 8 GiB is required")
    result = {
        "schema": "jog-production-worker-preflight-v1", "status": "passed", "created_utc": _utc(),
        "hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
        "gpu": _nvidia_identity(), "free_bytes": free, "shard_ids": selected, "workers": workers,
        "baseline_bundle_manifest_id": baseline["manifest_id"],
        "production_package_manifest_id": package_manifest["manifest_id"],
        "cuda_environment_manifest_id": environment.get("manifest_id"),
        "cuda_environment_sha256": file_sha256(environment_path),
        "runtime_override_sha256": file_sha256(override_path),
        "XLA_FLAGS": override["XLA_FLAGS"],
    }
    result["manifest_id"] = canonical_manifest_id(result)
    return result


class _ShardHeartbeat:
    def __init__(self, path: Path, state: dict):
        self.path = path
        self.state = state
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def write(self, **updates) -> None:
        with self.lock:
            self.state.update(updates)
            self.state["heartbeat_utc"] = _utc()
            _atomic_json(self.path, self.state)

    def _run(self) -> None:
        while not self.stop.wait(60):
            self.write()

    def __enter__(self):
        self.write()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join(timeout=5)


def _heartbeat_age_seconds(state: dict) -> float:
    value = datetime.fromisoformat(state["heartbeat_utc"])
    return max(0.0, (datetime.now(timezone.utc) - value).total_seconds())


def _claim_shard(path: Path, shard_id: int, campaign_id: str, package_id: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = path.with_suffix(path.suffix + ".claim")
    try:
        descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age = time.time() - guard.stat().st_mtime
        if age < STALE_HEARTBEAT_SECONDS:
            raise RuntimeError(f"Shard {shard_id:02d} is currently being claimed")
        guard.unlink()
        descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        previous = _read_json(path) if path.exists() else None
        if previous and previous.get("status") == "running" and _heartbeat_age_seconds(previous) < STALE_HEARTBEAT_SECONDS:
            raise RuntimeError(
                f"Shard {shard_id:02d} has a fresh running heartbeat from "
                f"{previous.get('hostname')} pid {previous.get('pid')}"
            )
        generation = int(previous.get("generation", 0)) + 1 if previous else 1
        state = {
            "schema": "jog-production-shard-state-v1", "status": "running", "campaign_id": campaign_id,
            "production_package_manifest_id": package_id, "shard_id": shard_id, "generation": generation,
            "hostname": socket.gethostname(), "pid": os.getpid(), "started_utc": _utc(),
            "current_job_id": None, "completed_in_shard": 0, "skipped_verified": 0,
        }
        _atomic_json(path, {**state, "heartbeat_utc": _utc()})
        return state
    finally:
        guard.unlink(missing_ok=True)


def _validate_final_run(path: Path, job_id: str, shard_id: int, campaign_id: str, package_id: str) -> dict:
    verification = verify_training_run(path)
    manifest = _read_json(path / "run_manifest.json")
    campaign = manifest.get("campaign", {})
    summary = _read_json(path / "training_summary.json")
    checks = {
        "verification": verification["passed"], "job_id": manifest.get("job_id") == job_id,
        "lambda_L2": float(manifest.get("lambda_L2", -1)) == FROZEN_L2,
        "campaign_id": campaign.get("campaign_id") == campaign_id,
        "shard_id": int(campaign.get("shard_id", -1)) == shard_id,
        "package_id": campaign.get("production_package_manifest_id") == package_id,
        "sole_model": (path / "best_model.keras").is_file() and not (path / "restored_best_model.keras").exists(),
        "exact_checkpoint_source": summary.get("retained_model_artifacts") == ["best_model.keras"],
        "full_epoch_cap": _read_json(path / "resolved_spec.json")["policy"]["max_epochs"] == 1500,
    }
    if not all(checks.values()):
        raise ValueError(f"Final run validation failed for {job_id}: {checks}")
    return {"run_manifest_id": manifest["manifest_id"], "checks": checks}


def _archive_interrupted_work(work: Path, failed_root: Path, job_id: str) -> None:
    if not work.exists():
        return
    files = [path for path in work.rglob("*") if path.is_file()]
    record = {
        "schema": "jog-interrupted-production-attempt-v1", "created_utc": _utc(), "job_id": job_id,
        "reason": "incomplete work directory found during resume; large partial artifacts removed",
        "file_count": len(files), "bytes_removed": sum(path.stat().st_size for path in files),
        "partial_manifest_present": (work / "run_manifest.json").is_file(),
    }
    destination = failed_root / job_id / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_interrupted")
    destination.mkdir(parents=True, exist_ok=False)
    _atomic_json(destination / "interrupted_attempt.json", record)
    resolved = work.resolve()
    if resolved.parent != work.parent.resolve() or work.name != job_id:
        raise RuntimeError("Refusing unsafe incomplete-work cleanup")
    shutil.rmtree(resolved)


def _training_process_is_active(work: Path) -> bool:
    marker = work / "active_process.json"
    if not marker.is_file():
        return False
    try:
        state = _read_json(marker)
        heartbeat = datetime.fromisoformat(state["heartbeat_utc"])
        age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
        return state.get("status") == "running" and age < STALE_HEARTBEAT_SECONDS
    except Exception:
        return False


def _adopt_or_wait_for_work(
    *, work: Path, final: Path, job_id: str, shard_id: int,
    campaign_id: str, package_id: str,
) -> dict | None:
    if not work.exists():
        return None
    # A TensorFlow child can outlive a disconnected notebook kernel. Attach by
    # waiting on the process-owned heartbeat instead of deleting active work.
    while _training_process_is_active(work):
        time.sleep(60)
    if (work / "run_manifest.json").is_file():
        evidence = _validate_final_run(work, job_id, shard_id, campaign_id, package_id)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work, final)
        return {"job_id": job_id, "disposition": "adopted_orphan_completion", **evidence}
    return None


def _run_job(
    *, package_root: Path, bundle_root: Path, campaign_root: Path,
    job_id: str, shard_id: int, campaign_id: str, package_id: str,
    xla_flags: str,
) -> dict:
    final = campaign_root / "runs" / job_id
    if final.exists():
        return {"job_id": job_id, "disposition": "skipped_verified",
                **_validate_final_run(final, job_id, shard_id, campaign_id, package_id)}
    work = campaign_root / "in_progress" / job_id
    adopted = _adopt_or_wait_for_work(
        work=work, final=final, job_id=job_id, shard_id=shard_id,
        campaign_id=campaign_id, package_id=package_id,
    )
    if adopted is not None:
        return adopted
    _archive_interrupted_work(work, campaign_root / "failed_attempts", job_id)
    work.parent.mkdir(parents=True, exist_ok=True)
    logs = campaign_root / "logs"; logs.mkdir(parents=True, exist_ok=True)
    attempts = (campaign_root / "failed_attempts" / job_id)
    index = (len(list(attempts.iterdir())) if attempts.exists() else 0) + 1
    log_path = logs / f"{job_id}.attempt-{index:02d}.log"
    command = [
        sys.executable, "-m", "production_training.train",
        "--dataset-dir", str(bundle_root / "dataset"),
        "--split-bundle", str(bundle_root / "splits"),
        "--job-id", job_id, "--lambda-l2", "0", "--output", str(work),
        "--campaign-id", campaign_id, "--shard-id", str(shard_id),
        "--production-package-manifest-id", package_id,
    ]
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = xla_flags
    mlp = package_root / "icepack-mlp"
    environment["PYTHONPATH"] = str(mlp) + os.pathsep + environment.get("PYTHONPATH", "")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=mlp, env=environment, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        failure = {
            "schema": "jog-production-job-failure-v1", "created_utc": _utc(), "job_id": job_id,
            "shard_id": shard_id, "returncode": process.returncode, "elapsed_seconds": elapsed,
            "log_path": str(log_path),
        }
        destination = campaign_root / "failed_attempts" / job_id / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination.mkdir(parents=True, exist_ok=False)
        _atomic_json(destination / "failure.json", failure)
        _archive_interrupted_work(work, campaign_root / "failed_attempts", job_id)
        raise RuntimeError(f"Production job {job_id} failed; see {log_path}")
    evidence = _validate_final_run(work, job_id, shard_id, campaign_id, package_id)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(work, final)
    return {"job_id": job_id, "disposition": "completed", "elapsed_seconds": elapsed,
            "log_path": str(log_path), **evidence}


def _run_shard(
    *, shard_id: int, jobs: list[str], package_root: Path, bundle_root: Path,
    campaign_root: Path, campaign_id: str, package_id: str, xla_flags: str,
    stop: threading.Event,
) -> dict:
    state_path = campaign_root / "shard_states" / f"shard_{shard_id:02d}.json"
    state = _claim_shard(state_path, shard_id, campaign_id, package_id)
    records = []
    try:
        with _ShardHeartbeat(state_path, state) as heartbeat:
            for position, job_id in enumerate(jobs, start=1):
                if stop.is_set():
                    heartbeat.write(status="stopped_after_peer_failure", current_job_id=None)
                    break
                heartbeat.write(current_job_id=job_id, position=position, jobs_in_shard=len(jobs))
                record = _run_job(
                    package_root=package_root, bundle_root=bundle_root, campaign_root=campaign_root,
                    job_id=job_id, shard_id=shard_id, campaign_id=campaign_id, package_id=package_id,
                    xla_flags=xla_flags,
                )
                records.append(record)
                heartbeat.write(
                    current_job_id=None,
                    completed_in_shard=sum(
                        item["disposition"] in {"completed", "adopted_orphan_completion"}
                        for item in records
                    ),
                    skipped_verified=sum(item["disposition"] == "skipped_verified" for item in records),
                )
            if not stop.is_set():
                heartbeat.write(status="complete", finished_utc=_utc(), current_job_id=None)
        return {"shard_id": shard_id, "status": state.get("status"), "records": records}
    except Exception as error:
        stop.set()
        _atomic_json(state_path, {**state, "status": "failed", "failed_utc": _utc(), "error": repr(error),
                                  "heartbeat_utc": _utc()})
        raise


def run_production_campaign(
    *, bundle_root: str | Path, runs_root: str | Path, cuda_environment: str | Path,
    runtime_override: str | Path, shard_ids: Iterable[int], workers: int = 1,
    package_root: str | Path | None = None,
) -> dict:
    """Run assigned immutable shards; safe to call again after interruption."""
    package = Path(package_root).resolve() if package_root else _package_root()
    bundle = Path(bundle_root).resolve(); root = Path(runs_root).resolve()
    selected = _validate_requested_shards(shard_ids, workers)
    preflight_record = preflight(
        bundle_root=bundle, runs_root=root, cuda_environment=cuda_environment,
        runtime_override=runtime_override, shard_ids=selected, workers=workers, package_root=package,
    )
    package_id = preflight_record["production_package_manifest_id"]
    campaign_id = _read_json(package / "campaign" / "shard_manifest.json")["campaign_id"]
    campaign_root = root / campaign_id
    campaign_root.mkdir(parents=True, exist_ok=True)
    preflight_path = campaign_root / "worker_preflights" / f"{socket.gethostname()}_{os.getpid()}.json"
    _atomic_json(preflight_path, preflight_record)
    override = _read_json(Path(runtime_override).resolve())
    shards, _ = _load_shards(package)
    stop = threading.Event(); results = []; failures = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jog-production") as executor:
        futures = {
            executor.submit(
                _run_shard, shard_id=shard, jobs=shards[shard], package_root=package,
                bundle_root=bundle, campaign_root=campaign_root, campaign_id=campaign_id,
                package_id=package_id, xla_flags=override["XLA_FLAGS"], stop=stop,
            ): shard for shard in selected
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"shard_id": futures[future], "error": repr(error)})
    summary = {
        "schema": "jog-production-worker-summary-v1", "created_utc": _utc(),
        "status": "complete" if not failures else "failed", "campaign_id": campaign_id,
        "production_package_manifest_id": package_id, "hostname": socket.gethostname(),
        "pid": os.getpid(), "assigned_shards": selected, "workers": workers,
        "results": results, "failures": failures,
    }
    summary["manifest_id"] = canonical_manifest_id(summary)
    _atomic_json(campaign_root / "worker_summaries" / f"{socket.gethostname()}_{os.getpid()}.json", summary)
    if failures:
        raise RuntimeError(json.dumps(summary, indent=2))
    return summary
