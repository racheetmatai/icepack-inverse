"""Standard-library verification for the source-only production launcher."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


BASELINE_ID = "sha256-json-v1-303df5600ae81d7a21dd66b333fa76929669ce0af880bbe4a22bc8d9aed4a625"


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


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(package_root: Path, baseline_root: Path) -> dict:
    package_root = package_root.resolve(); baseline_root = baseline_root.resolve()
    manifest = json.loads((package_root / "production_package_manifest.json").read_text(encoding="utf-8"))
    actual_files = {
        path.relative_to(package_root).as_posix(): sha256(path)
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path.name != "production_package_manifest.json"
    }
    baseline = json.loads((baseline_root / "bundle_manifest.json").read_text(encoding="utf-8"))
    original = rows(baseline_root / "splits" / "job_registry.csv")
    sharded = rows(package_root / "campaign" / "production_shards.csv")
    original_by_id = {row["job_id"]: row for row in original}
    sharded_by_id = {row["job_id"]: row for row in sharded}
    identity_columns = [
        "job_id", "experiment", "configuration", "member", "split_id",
        "split_seed", "model_seed", "shuffle_seed", "split_file",
    ]
    registry_match = bool(
        len(original) == len(original_by_id) == len(sharded) == len(sharded_by_id) == 660
        and all(
            all(original_by_id[job][column] == sharded_by_id[job][column] for column in identity_columns)
            for job in original_by_id
        )
    )
    shard_counts = Counter(int(row["shard_id"]) for row in sharded)
    configuration_counts = {
        configuration: Counter(int(row["shard_id"]) for row in sharded if row["configuration"] == configuration)
        for configuration in sorted({row["configuration"] for row in sharded})
    }
    regional_counts = Counter(int(row["shard_id"]) for row in sharded if row["experiment"].startswith("REG_"))
    experiment_counts = Counter("regional" if row["experiment"].startswith("REG_") else "primary" for row in sharded)
    shard_manifest = json.loads((package_root / "campaign" / "shard_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "schema": manifest.get("schema") == "jog-production-launcher-package-v1",
        "manifest_id": canonical_id(manifest) == manifest.get("manifest_id"),
        "file_inventory_and_hashes": actual_files == manifest.get("output_sha256"),
        "baseline_bundle_id": baseline.get("manifest_id") == manifest.get("baseline_bundle_manifest_id") == BASELINE_ID,
        "baseline_registry_hash": sha256(baseline_root / "splits" / "job_registry.csv") == shard_manifest.get("baseline_job_registry_sha256"),
        "registry_exact_match": registry_match,
        "shards_complete": set(shard_counts) == set(range(24)) and set(shard_counts.values()).issubset({27, 28}),
        "configuration_balance": all(
            set(counts) == set(range(24)) and set(counts.values()).issubset({4, 5})
            for counts in configuration_counts.values()
        ),
        "regional_balance": set(regional_counts) == set(range(24)) and set(regional_counts.values()).issubset({2, 3}),
        "job_categories": experiment_counts == {"primary": 600, "regional": 60},
        "shard_manifest_id": canonical_id(shard_manifest) == shard_manifest.get("manifest_id"),
        "frozen_L2": manifest.get("selected_lambda_L2") == shard_manifest.get("selected_lambda_L2") == 0.0,
        "smoke_gate": manifest.get("smoke_acceptance_manifest_id")
        == "sha256-json-v1-d2a4286a44849b078a1d1770e2c6c32561b941b4e0794c1a113e0896dc78019d",
    }
    return {
        "schema": "jog-production-launcher-package-verification-v1",
        "package_manifest_id": manifest.get("manifest_id"), "checks": checks,
        "counts": {"jobs": len(sharded), "primary": experiment_counts["primary"],
                   "regional": experiment_counts["regional"], "shards": len(shard_counts)},
        "passed": all(checks.values()),
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_production_launcher_package.py PACKAGE_ROOT BASELINE_BUNDLE_ROOT")
    result = verify(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
