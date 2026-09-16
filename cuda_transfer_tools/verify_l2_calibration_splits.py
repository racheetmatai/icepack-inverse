#!/usr/bin/env python3
"""Independently verify the frozen four-fit L2 calibration split bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    return "sha256-json-v1-" + hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def unpack(value: np.ndarray, count: int) -> np.ndarray:
    return np.unpackbits(value, bitorder="little")[:count].astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--bundle", required=True); args = parser.parse_args()
    dataset = Path(args.dataset_dir).resolve(); root = Path(args.bundle).resolve()
    manifest = json.loads((root / "split_bundle_manifest.json").read_text())
    dataset_manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    checks = {}
    checks["schema_status"] = manifest.get("schema") == "jog-l2-calibration-split-bundle-v1" and manifest.get("status") == "complete"
    checks["manifest_id"] = canonical_id(manifest) == manifest.get("manifest_id")
    checks["dataset_identity"] = manifest.get("dataset_manifest_id") == dataset_manifest.get("manifest_id")
    data_path = dataset / "canonical_master_dataset.csv.gz"
    checks["dataset_sha256"] = sha256(data_path) == manifest.get("dataset_sha256") == dataset_manifest["output_sha256"][data_path.name]
    failures = [rel for rel, expected in manifest["output_sha256"].items() if not (root / rel).is_file() or sha256(root / rel) != expected]
    checks["declared_hashes"] = not failures
    row_path = root / "sorted_common_eligible_row_ids.txt.gz"
    checks["row_index_sha256"] = sha256(row_path) == manifest.get("row_index_sha256")
    row_ids = pd.read_csv(row_path, header=None, names=["row_id"], dtype=str)["row_id"].to_numpy(object)
    raw = pd.read_csv(data_path, usecols=["row_id", "common_eligible", "square_test_id"], low_memory=False)
    eligible = raw.loc[raw["common_eligible"].astype(bool), ["row_id", "square_test_id"]]
    checks["exact_row_index"] = not eligible["row_id"].duplicated().any() and np.array_equal(np.sort(eligible["row_id"].to_numpy(str)), row_ids)
    frame = eligible.set_index("row_id").reindex(row_ids)
    expected_excluded = frame["square_test_id"].fillna("").astype(str).str.fullmatch(r"SQ(?:0[1-9]|10)").to_numpy()
    count = len(row_ids)
    with np.load(root / "population_masks/CAL_GLOBAL.npz", allow_pickle=False) as archive:
        population = {name: unpack(archive[name], count) for name in (
            "calibration_population", "excluded_central_50km", "train", "validation", "test")}
    checks["outcome_blind_exclusion"] = np.array_equal(population["excluded_central_50km"], expected_excluded)
    checks["calibration_complement"] = np.array_equal(population["calibration_population"], ~expected_excluded)
    checks["disjoint_complete_70_20_10"] = (
        not np.any(population["train"] & population["validation"])
        and not np.any(population["train"] & population["test"])
        and not np.any(population["validation"] & population["test"])
        and np.array_equal(population["train"] | population["validation"] | population["test"], ~expected_excluded)
    )
    positions = np.flatnonzero(~expected_excluded); permutation = np.random.default_rng(20260811).permutation(positions)
    n_train = math.floor(0.70 * len(positions)); n_validation = math.floor(0.20 * len(positions))
    exact_train = np.zeros(count, bool); exact_validation = np.zeros(count, bool); exact_test = np.zeros(count, bool)
    exact_train[permutation[:n_train]] = True
    exact_validation[permutation[n_train:n_train+n_validation]] = True
    exact_test[permutation[n_train+n_validation:]] = True
    checks["exact_seeded_permutation"] = all(np.array_equal(population[name], expected) for name, expected in (
        ("train", exact_train), ("validation", exact_validation), ("test", exact_test)))
    checks["manifest_counts"] = (
        int(population["calibration_population"].sum()) == manifest["calibration_rows"]
        and int(expected_excluded.sum()) == manifest["excluded_central_50km_rows"]
        and int(exact_train.sum()) == manifest["train_rows"]
        and int(exact_validation.sum()) == manifest["validation_rows"]
        and int(exact_test.sum()) == manifest["test_rows"]
    )
    with (root / "l2_pilot_registry.csv").open(newline="", encoding="utf-8") as stream:
        pilots = list(csv.DictReader(stream))
    checks["four_cfg06_jobs"] = (
        len(pilots) == 4 and {row["configuration"] for row in pilots} == {"CFG06"}
        and {float(row["lambda_L2"]) for row in pilots} == {0.0, 1e-6, 1e-5, 1e-4}
        and {row["split_id"] for row in pilots} == {"CAL_GLOBAL_M01"}
        and {row["model_seed"] for row in pilots} == {"30260811"}
        and {row["shuffle_seed"] for row in pilots} == {"40260811"}
    )
    result = {"schema": "jog-l2-calibration-verification-v1", "manifest_id": manifest.get("manifest_id"),
              "checks": checks, "hash_failures": failures[:20], "passed": all(checks.values())}
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
