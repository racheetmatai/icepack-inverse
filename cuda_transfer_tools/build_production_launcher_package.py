"""Build the source-only restartable production launcher for LEAP Pangeo."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAME = "JOG_PRODUCTION_LAUNCHER_20260820_E"
BASELINE = ROOT / "cuda_transfer" / "JOG_CUDA_BUNDLE_20260820_C"
BASELINE_ID = "sha256-json-v1-303df5600ae81d7a21dd66b333fa76929669ce0af880bbe4a22bc8d9aed4a625"
L2_ID = "sha256-json-v1-e447c03282015a3978dd4c3d240cb8bf07fc89f87fa75d58737e25e29f19bd02"
SMOKE_ID = "sha256-json-v1-d2a4286a44849b078a1d1770e2c6c32561b941b4e0794c1a113e0896dc78019d"
SHARDS = 24


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_shards(destination: Path) -> dict:
    source = BASELINE / "splits" / "job_registry.csv"
    rows = sorted(read_rows(source), key=lambda row: row["job_id"])
    if len(rows) != 660 or len({row["job_id"] for row in rows}) != 660:
        raise ValueError("Baseline registry is not the frozen 660-job registry")
    totals = [0] * SHARDS
    regional = [0] * SHARDS
    configuration = [Counter() for _ in range(SHARDS)]
    assignments: dict[str, int] = {}
    # Assign regional jobs first so each shard receives 2--3, then all primary
    # jobs. The deterministic greedy score also keeps each feature configuration
    # at 4--5 jobs/shard while retaining 27--28 total jobs/shard.
    assignment_order = sorted(
        rows,
        key=lambda row: (
            0 if row["experiment"].startswith("REG_") else 1,
            row["configuration"], row["job_id"],
        ),
    )
    for row in assignment_order:
        is_regional = row["experiment"].startswith("REG_")
        feature = row["configuration"]
        shard = min(
            range(SHARDS),
            key=lambda value: (
                totals[value], configuration[value][feature],
                regional[value] if is_regional else 0, value,
            ),
        )
        assignments[row["job_id"]] = shard
        totals[shard] += 1
        regional[shard] += int(is_regional)
        configuration[shard][feature] += 1
    positions = Counter()
    output_rows = []
    for row in rows:
        shard = assignments[row["job_id"]]
        positions[shard] += 1
        output_rows.append({**row, "shard_id": str(shard), "shard_position": str(positions[shard])})
    path = destination / "campaign" / "production_shards.csv"
    path.parent.mkdir(parents=True)
    fields = list(rows[0]) + ["shard_id", "shard_position"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(output_rows)
    memberships = {}
    for shard in range(SHARDS):
        jobs = [row["job_id"] for row in output_rows if int(row["shard_id"]) == shard]
        memberships[str(shard)] = {
            "job_count": len(jobs),
            "job_ids_sha256": hashlib.sha256(("\n".join(jobs) + "\n").encode()).hexdigest(),
            "first_job_id": jobs[0], "last_job_id": jobs[-1],
        }
    membership_text = "".join(f"{row['shard_id']}\t{row['job_id']}\n" for row in output_rows)
    campaign_basis = {
        "schema": "jog-production-campaign-identity-v1", "baseline_job_registry_sha256": sha256(source),
        "selected_lambda_L2": 0.0, "shard_count": SHARDS,
        "ordered_job_ids_sha256": hashlib.sha256(("\n".join(row["job_id"] for row in rows) + "\n").encode()).hexdigest(),
        "shard_membership_sha256": hashlib.sha256(membership_text.encode()).hexdigest(),
    }
    campaign_id = canonical_id(campaign_basis)
    manifest = {
        "schema": "jog-production-shard-manifest-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id, "baseline_bundle_manifest_id": BASELINE_ID,
        "baseline_job_registry_sha256": sha256(source), "selected_lambda_L2": 0.0,
        "l2_selection_manifest_id": L2_ID, "smoke_acceptance_manifest_id": SMOKE_ID,
        "job_count": len(rows), "primary_job_count": sum(not row["experiment"].startswith("REG_") for row in rows),
        "regional_job_count": sum(row["experiment"].startswith("REG_") for row in rows),
        "shard_count": SHARDS,
        "assignment": "deterministic greedy balance by total count, configuration count, regional count, then shard ID",
        "shards": memberships,
    }
    manifest["manifest_id"] = canonical_id(manifest)
    write_json(destination / "campaign" / "shard_manifest.json", manifest)
    return manifest


def main() -> None:
    destination = ROOT / "cuda_transfer" / NAME
    archive = ROOT / "cuda_transfer" / f"{NAME}.zip"
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if destination.exists() or archive.exists() or sidecar.exists():
        raise FileExistsError("Refusing to replace an existing production launcher package")
    destination.mkdir(parents=True)
    mlp = destination / "icepack-mlp"; package = mlp / "production_training"; package.mkdir(parents=True)
    source_package = ROOT / "icepack-mlp" / "production_training"
    for name in (
        "__init__.py", "spec.py", "integrity.py", "data.py", "model.py",
        "train.py", "verify_run.py", "select_l2.py", "campaign.py", "concurrency_benchmark.py",
    ):
        shutil.copy2(source_package / name, package / name)
    tests = mlp / "tests"; tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    for name in ("test_exact_checkpoint_reload.py", "test_production_campaign.py", "test_production_training.py"):
        shutil.copy2(ROOT / "icepack-mlp" / "tests" / name, tests / name)
    tools = destination / "tools"; tools.mkdir()
    shutil.copy2(ROOT / "cuda_transfer_tools" / "verify_production_launcher_package.py",
                 tools / "verify_production_launcher_package.py")
    shutil.copy2(ROOT / "cuda_transfer_tools" / "PRODUCTION_LAUNCHER_README.md", destination / "README.md")

    provenance = destination / "provenance"; provenance.mkdir()
    l2_root = ROOT / "cuda_results" / "JOG_L2_RESULTS_20260820" / "JOG_CUDA_RUNS" / "l2_calibration"
    smoke_root = (ROOT / "cuda_results" / "JOG_CUDA_SMOKE_RESULTS_20260820" / "JOG_CUDA_RUNS"
                  / "production_smoke" / "SQ01_CFG06_M01_E3")
    shutil.copy2(l2_root / "global_l2_selection.json", provenance / "global_l2_selection.json")
    shutil.copy2(l2_root / "calibration_test" / "test_manifest.json", provenance / "calibration_test_manifest.json")
    shutil.copy2(smoke_root / "cuda_smoke_acceptance_manifest.json", provenance / "cuda_smoke_acceptance_manifest.json")
    policy = {
        "schema": "jog-production-execution-policy-v1", "selected_lambda_L2": 0.0,
        "retained_model_artifact": "best_model.keras only",
        "checkpoint_policy": "one fixed save_best_only path overwritten on improvement; no per-epoch accumulation",
        "restart_boundary": "completed verified job; interrupted current job restarts from epoch zero",
        "orphan_process_policy": "training-process heartbeat prevents cleanup; verified orphan completion is adopted",
        "partial_artifact_policy": "preserve small failure record/log and remove incomplete large model artifacts",
        "workers_per_T4": "freeze from source-matched concurrency benchmark; allowed values 1 or 2",
        "held_out_test_accessed_during_training_or_benchmark": False,
        "estimated_final_storage_GiB": 5.4,
    }
    write_json(provenance / "production_execution_policy.json", policy)
    shard_manifest = build_shards(destination)

    outputs = {
        path.relative_to(destination).as_posix(): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "production_package_manifest.json"
    }
    manifest = {
        "schema": "jog-production-launcher-package-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_bundle_manifest_id": BASELINE_ID, "campaign_id": shard_manifest["campaign_id"],
        "shard_manifest_id": shard_manifest["manifest_id"], "selected_lambda_L2": 0.0,
        "l2_selection_manifest_id": L2_ID, "smoke_acceptance_manifest_id": SMOKE_ID,
        "job_count": 660, "primary_job_count": 600, "regional_job_count": 60, "shard_count": SHARDS,
        "purpose": "source-only restartable production launcher and T4 concurrency benchmark",
        "output_sha256": outputs,
    }
    manifest["manifest_id"] = canonical_id(manifest)
    write_json(destination / "production_package_manifest.json", manifest)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                bundle.write(path, f"{NAME}/{path.relative_to(destination).as_posix()}")
    sidecar.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"package": str(destination), "archive": str(archive),
                      "archive_sha256": sha256(archive), "manifest_id": manifest["manifest_id"],
                      "file_count": len(outputs), "archive_bytes": archive.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
