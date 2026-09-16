#!/usr/bin/env python3
"""Assemble the immutable Gate-3 CUDA transfer directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


DATASET_ID = "sha256-json-v1-496a391df29fc4d64ba1b134fc8e12fd808b2bb1194935e60981d980767dfd8e"
SPLIT_ID = "sha256-json-v1-b838631dfde2849f84ee2165749fe9ce44c01e71513e53fb66afe1d8785281b8"
CALIBRATION_ID = "sha256-json-v1-327c336b9afa88a5fe9306d0bec61a73543a384ff5d40f652d7317f99fa39899"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(args.project_root).resolve(); output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset_source = project / "production_workflow/gate2_results/gate2_canonical_dataset_20260820_c"
    split_source = project / "production_workflow/gate2_results/gate2_split_manifests_20260820_a"
    calibration_source = project / "production_workflow/gate2_results/l2_calibration_splits_20260820_a"
    dataset_manifest = read_json(dataset_source / "dataset_manifest.json")
    split_manifest = read_json(split_source / "split_bundle_manifest.json")
    calibration_manifest = read_json(calibration_source / "split_bundle_manifest.json")
    if (dataset_manifest.get("manifest_id") != DATASET_ID or split_manifest.get("manifest_id") != SPLIT_ID
            or calibration_manifest.get("manifest_id") != CALIBRATION_ID):
        raise ValueError("Accepted Gate 2 input identity mismatch")

    shutil.copytree(dataset_source, output / "dataset")
    shutil.copytree(split_source, output / "splits")
    shutil.copytree(calibration_source, output / "l2_calibration")
    mlp = output / "icepack-mlp"; mlp.mkdir()
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(project / "icepack-mlp/production_training", mlp / "production_training", ignore=ignored)
    shutil.copytree(project / "icepack-mlp/tests", mlp / "tests", ignore=ignored)
    for name in ("production_training_config.json", "README_PRODUCTION_TRAINING.md", "LEGACY_NOTEBOOKS.md"):
        shutil.copy2(project / "icepack-mlp" / name, mlp / name)
    tools = output / "tools"; tools.mkdir()
    for name in ("capture_cuda_environment.py", "verify_cuda_bundle.py", "verify_l2_calibration_splits.py",
                 "run_l2_calibration.py"):
        shutil.copy2(Path(__file__).resolve().parent / name, tools / name)
    shutil.copy2(Path(__file__).resolve().parent / "CUDA_BUNDLE_README.md", output / "README.md")

    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema": "jog-cuda-transfer-bundle-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest_id": DATASET_ID, "split_bundle_manifest_id": SPLIT_ID,
        "l2_calibration_split_manifest_id": CALIBRATION_ID,
        "purpose": "Gate 3 transfer, environment capture, L2 pilot, CUDA smoke, and production training",
        "file_count": len(files), "total_bytes": sum(path.stat().st_size for path in files),
        "output_sha256": {path.relative_to(output).as_posix(): sha256(path) for path in files},
    }
    manifest["manifest_id"] = canonical_id(manifest)
    (output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "manifest_id": manifest["manifest_id"],
                      "file_count": manifest["file_count"], "total_bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
