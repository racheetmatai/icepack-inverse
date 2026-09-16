#!/usr/bin/env python3
"""Sequentially execute/resume the frozen four-fit L2 calibration on CUDA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    return "sha256-json-v1-" + hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--runs-root", required=True); parser.add_argument("--cuda-environment", required=True)
    args = parser.parse_args(); bundle = Path(args.bundle_root).resolve(); runs = Path(args.runs_root).resolve()
    environment = json.loads(Path(args.cuda_environment).resolve().read_text())
    if canonical_id(environment) != environment.get("manifest_id") or not environment.get("passed"):
        raise ValueError("CUDA environment gate is absent, invalid, or failed")
    calibration = bundle / "l2_calibration"; mlp = bundle / "icepack-mlp"; runs.mkdir(parents=True, exist_ok=True)
    with (calibration / "l2_pilot_registry.csv").open(newline="", encoding="utf-8") as stream:
        jobs = list(csv.DictReader(stream))
    if len(jobs) != 4:
        raise ValueError("Expected exactly four frozen L2 calibration jobs")
    for job in jobs:
        output = runs / job["pilot_job_id"]
        if (output / "run_manifest.json").is_file():
            subprocess.run([sys.executable, "-m", "production_training.verify_run", str(output)], cwd=mlp, check=True)
            continue
        command = [
            sys.executable, "-m", "production_training.train",
            "--dataset-dir", str(bundle / "dataset"), "--split-bundle", str(calibration),
            "--job-id", job["pilot_job_id"], "--lambda-l2", job["lambda_L2"], "--output", str(output),
        ]
        subprocess.run(command, cwd=mlp, check=True)
        subprocess.run([sys.executable, "-m", "production_training.verify_run", str(output)], cwd=mlp, check=True)
    selection = runs / "global_l2_selection.json"
    subprocess.run([
        sys.executable, "-m", "production_training.select_l2", "--calibration-bundle", str(calibration),
        "--runs-root", str(runs), "--output", str(selection),
    ], cwd=mlp, check=True)
    print(selection)


if __name__ == "__main__":
    main()
