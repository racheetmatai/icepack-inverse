#!/usr/bin/env python3
"""Create the frozen CFG06 L2 calibration population and 70/20/10 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_SEED = 20260811
MODEL_SEED = 30260811
SHUFFLE_SEED = 40260811
L2_VALUES = [("0e+00", 0.0), ("1e-06", 1e-6), ("1e-05", 1e-5), ("1e-04", 1e-4)]
DATASET_ID = "sha256-json-v1-496a391df29fc4d64ba1b134fc8e12fd808b2bb1194935e60981d980767dfd8e"


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


def membership_id(index_sha: str, packed: np.ndarray) -> str:
    return "sha256-row-membership-v1-" + hashlib.sha256(bytes.fromhex(index_sha) + packed.tobytes()).hexdigest()


def pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.astype(np.uint8), bitorder="little")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True); parser.add_argument("--base-splits", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); dataset = Path(args.dataset_dir).resolve(); base = Path(args.base_splits).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {output}")
    (output / "member_splits").mkdir(parents=True, exist_ok=True)
    (output / "population_masks").mkdir(parents=True, exist_ok=True)
    (output / "source_snapshot").mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    base_manifest = json.loads((base / "split_bundle_manifest.json").read_text())
    if dataset_manifest.get("manifest_id") != DATASET_ID or base_manifest.get("dataset_manifest_id") != DATASET_ID:
        raise ValueError("Accepted dataset identity mismatch")
    data_path = dataset / "canonical_master_dataset.csv.gz"
    if sha256(data_path) != dataset_manifest["output_sha256"][data_path.name]:
        raise ValueError("Dataset SHA256 mismatch")
    index_source = base / "sorted_common_eligible_row_ids.txt.gz"
    if sha256(index_source) != base_manifest["row_index_sha256"]:
        raise ValueError("Base sorted row-index SHA256 mismatch")
    shutil.copy2(index_source, output / index_source.name)
    row_ids = pd.read_csv(index_source, header=None, names=["row_id"], dtype=str)["row_id"].to_numpy(object)
    raw = pd.read_csv(data_path, usecols=["row_id", "common_eligible", "square_test_id"], low_memory=False)
    eligible = raw.loc[raw["common_eligible"].astype(bool), ["row_id", "square_test_id"]]
    if eligible["row_id"].duplicated().any() or not np.array_equal(np.sort(eligible["row_id"].to_numpy(str)), row_ids):
        raise ValueError("Eligible row IDs do not exactly equal the frozen sorted index")
    frame = eligible.set_index("row_id").reindex(row_ids)
    if len(frame) != int(base_manifest["eligible_rows"]):
        raise ValueError("Eligible row-index reconstruction failure")
    excluded = frame["square_test_id"].fillna("").astype(str).str.fullmatch(r"SQ(?:0[1-9]|10)").to_numpy()
    calibration = ~excluded
    positions = np.flatnonzero(calibration)
    permutation = np.random.default_rng(SPLIT_SEED).permutation(positions)
    n = len(positions); n_train = math.floor(0.70 * n); n_validation = math.floor(0.20 * n)
    train = np.zeros(len(frame), bool); validation = np.zeros(len(frame), bool); test = np.zeros(len(frame), bool)
    train[permutation[:n_train]] = True
    validation[permutation[n_train:n_train + n_validation]] = True
    test[permutation[n_train + n_validation:]] = True
    if np.any(train & validation) or np.any(train & test) or np.any(validation & test):
        raise AssertionError("Calibration partitions overlap")
    if not np.array_equal(train | validation | test, calibration):
        raise AssertionError("Calibration partitions do not exactly cover the population")

    arrays = {name: pack(mask) for name, mask in {
        "calibration_population": calibration, "excluded_central_50km": excluded,
        "train": train, "validation": validation, "test": test,
    }.items()}
    population_path = output / "population_masks/CAL_GLOBAL.npz"
    np.savez_compressed(population_path, row_count=np.int64(len(frame)), bitorder=np.asarray("little"), **arrays)
    split_path = output / "member_splits/CAL_GLOBAL.npz"
    np.savez_compressed(
        split_path, row_count=np.int64(len(frame)), bitorder=np.asarray("little"), members=np.asarray([1], dtype=np.int8),
        train=np.asarray([arrays["train"]]), validation=np.asarray([arrays["validation"]]),
        split_seeds=np.asarray([SPLIT_SEED]), model_seeds=np.asarray([MODEL_SEED]),
        shuffle_seeds=np.asarray([SHUFFLE_SEED]),
    )
    row_index_sha = sha256(output / index_source.name)
    split_id = "CAL_GLOBAL_M01"
    member = {
        "split_id": split_id, "experiment": "CAL_GLOBAL", "member": 1, "offset": 0,
        "split_seed": SPLIT_SEED, "model_seed": MODEL_SEED, "shuffle_seed": SHUFFLE_SEED,
        "train_rows": int(train.sum()), "validation_rows": int(validation.sum()),
        "train_membership_id": membership_id(row_index_sha, arrays["train"]),
        "validation_membership_id": membership_id(row_index_sha, arrays["validation"]),
        "split_file": "member_splits/CAL_GLOBAL.npz", "split_file_sha256": sha256(split_path),
    }
    write_csv(output / "member_splits.csv", list(member), [member])
    write_csv(output / "job_registry.csv", ["job_id", "experiment", "configuration", "member", "split_id",
              "split_seed", "model_seed", "shuffle_seed", "split_file"], [])
    pilots = [{
        "pilot_job_id": f"L2CAL_CFG06_L2_{label}", "experiment": "CAL_GLOBAL", "configuration": "CFG06",
        "member": 1, "lambda_L2": label, "split_id": split_id, "split_seed": SPLIT_SEED,
        "model_seed": MODEL_SEED, "shuffle_seed": SHUFFLE_SEED,
    } for label, _ in L2_VALUES]
    write_csv(output / "l2_pilot_registry.csv", list(pilots[0]), pilots)
    population = {
        "experiment": "CAL_GLOBAL", "eligible_rows": len(frame), "excluded_central_50km_rows": int(excluded.sum()),
        "calibration_rows": int(calibration.sum()), "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()), "test_rows": int(test.sum()),
        "population_mask_file": "population_masks/CAL_GLOBAL.npz",
        "calibration_membership_id": membership_id(row_index_sha, arrays["calibration_population"]),
        "excluded_membership_id": membership_id(row_index_sha, arrays["excluded_central_50km"]),
        "test_membership_id": membership_id(row_index_sha, arrays["test"]),
    }
    write_csv(output / "experiments.csv", list(population), [population])
    methods = {
        "schema": "jog-l2-calibration-method-v1", "features": "CFG06", "candidate_lambda_L2": [v for _, v in L2_VALUES],
        "population": "common eligible rows excluding only the ten central 50x50 km square evaluation masks",
        "retained_regions": "all square buffers, PIG, and both inter-catchment corridors",
        "split": {"algorithm": "numpy.default_rng(seed).permutation", "seed": SPLIT_SEED,
                  "train": "floor(0.70*N)", "validation": "floor(0.20*N)", "test": "remainder"},
        "selection": "minimum validation data_mse; nonzero must improve zero by at least 1e-4; exact tie smaller lambda",
        "test_access": "test mask remains unavailable to fitting and selection; evaluate once only after lambda is frozen",
    }
    (output / "methods.json").write_text(json.dumps(methods, indent=2, sort_keys=True) + "\n")
    shutil.copy2(Path(__file__).resolve(), output / "source_snapshot/create_l2_calibration_splits.py")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema": "jog-l2-calibration-split-bundle-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_manifest_id": DATASET_ID,
        "dataset_sha256": sha256(data_path), "eligible_rows": len(frame), "row_index_sha256": row_index_sha,
        "calibration_rows": int(calibration.sum()), "excluded_central_50km_rows": int(excluded.sum()),
        "train_rows": int(train.sum()), "validation_rows": int(validation.sum()), "test_rows": int(test.sum()),
        "pilot_job_count": 4,
        "output_sha256": {path.relative_to(output).as_posix(): sha256(path) for path in files},
    }
    manifest["manifest_id"] = canonical_id(manifest)
    (output / "split_bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: manifest[key] for key in ("manifest_id", "calibration_rows", "excluded_central_50km_rows",
                                                     "train_rows", "validation_rows", "test_rows")}, indent=2))


if __name__ == "__main__":
    main()
