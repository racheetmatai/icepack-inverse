"""Verified loading and train-only scaling for one frozen training job."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .integrity import canonical_manifest_id, file_sha256, verify_declared_outputs
from .spec import FEATURE_CONFIGURATIONS, FROZEN_POLICY


@dataclass
class PreparedJob:
    job: dict
    split: dict
    dataset_manifest: dict
    split_manifest: dict
    features: list[str]
    train_row_ids: np.ndarray
    validation_row_ids: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    input_scaler: RobustScaler
    target_scaler: RobustScaler


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _one_csv_record(path: Path, key: str, value: str) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        matches = [row for row in csv.DictReader(stream) if row[key] == value]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {key}={value!r} in {path}; found {len(matches)}")
    return matches[0]


def _resolve_job(split_bundle: Path, job_id: str) -> tuple[dict, bool]:
    try:
        return _one_csv_record(split_bundle / "job_registry.csv", "job_id", job_id), False
    except ValueError:
        pilot = _one_csv_record(split_bundle / "l2_pilot_registry.csv", "pilot_job_id", job_id)
        return {
            "job_id": pilot["pilot_job_id"], "experiment": pilot["experiment"],
            "configuration": pilot["configuration"], "member": pilot["member"],
            "split_id": pilot["split_id"], "split_seed": pilot["split_seed"],
            "model_seed": pilot["model_seed"], "shuffle_seed": pilot["shuffle_seed"],
            "registered_lambda_L2": pilot["lambda_L2"],
        }, True


def verify_inputs(dataset_dir: Path, split_bundle: Path, *, full_hash_check: bool = True) -> tuple[dict, dict]:
    dataset_dir = dataset_dir.resolve()
    split_bundle = split_bundle.resolve()
    dataset_manifest = _read_json(dataset_dir / "dataset_manifest.json")
    split_manifest = _read_json(split_bundle / "split_bundle_manifest.json")
    if dataset_manifest.get("schema") != "jog-canonical-master-dataset-v1" or dataset_manifest.get("status") != "complete":
        raise ValueError("Canonical dataset manifest is not a complete v1 dataset")
    if canonical_manifest_id(dataset_manifest) != dataset_manifest.get("manifest_id"):
        raise ValueError("Canonical dataset manifest ID mismatch")
    allowed_split_schemas = {"jog-training-split-bundle-v1", "jog-l2-calibration-split-bundle-v1"}
    if split_manifest.get("schema") not in allowed_split_schemas or split_manifest.get("status") != "complete":
        raise ValueError("Split bundle manifest is not a complete accepted v1 bundle")
    if canonical_manifest_id(split_manifest) != split_manifest.get("manifest_id"):
        raise ValueError("Split bundle manifest ID mismatch")
    if split_manifest.get("dataset_manifest_id") != dataset_manifest.get("manifest_id"):
        raise ValueError("Split bundle was not made from this canonical dataset")
    dataset_path = dataset_dir / "canonical_master_dataset.csv.gz"
    if full_hash_check:
        expected = dataset_manifest["output_sha256"][dataset_path.name]
        if file_sha256(dataset_path) != expected or expected != split_manifest.get("dataset_sha256"):
            raise ValueError("Canonical dataset SHA256 mismatch")
        verify_declared_outputs(split_bundle, split_manifest)
    return dataset_manifest, split_manifest


def _sorted_ids(path: Path, expected_count: int) -> np.ndarray:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        values = np.asarray([line.rstrip("\n") for line in stream], dtype=object)
    if len(values) != expected_count or len(np.unique(values)) != expected_count:
        raise ValueError("Frozen eligible row index count/uniqueness failure")
    return values


def _unpack(value: np.ndarray, count: int) -> np.ndarray:
    return np.unpackbits(value, bitorder="little")[:count].astype(bool)


def load_prepared_job(
    dataset_dir: str | Path,
    split_bundle: str | Path,
    job_id: str,
    *,
    full_hash_check: bool = True,
) -> PreparedJob:
    dataset_dir = Path(dataset_dir).resolve()
    split_bundle = Path(split_bundle).resolve()
    dataset_manifest, split_manifest = verify_inputs(dataset_dir, split_bundle, full_hash_check=full_hash_check)
    job, is_pilot = _resolve_job(split_bundle, job_id)
    configuration = job["configuration"]
    if configuration not in FEATURE_CONFIGURATIONS:
        raise ValueError(f"Unknown configuration in job registry: {configuration}")
    split = _one_csv_record(split_bundle / "member_splits.csv", "split_id", job["split_id"])
    if is_pilot:
        job["split_file"] = split["split_file"]
    for name in ("split_seed", "model_seed", "shuffle_seed", "split_file"):
        if job[name] != split[name]:
            raise ValueError(f"Job/split registry disagreement for {name}")

    expected_count = int(split_manifest["eligible_rows"])
    index_path = split_bundle / "sorted_common_eligible_row_ids.txt.gz"
    if file_sha256(index_path) != split_manifest["row_index_sha256"]:
        raise ValueError("Frozen eligible row index SHA256 mismatch")
    row_ids = _sorted_ids(index_path, expected_count)
    split_path = split_bundle / job["split_file"]
    if file_sha256(split_path) != split["split_file_sha256"]:
        raise ValueError("Member split archive SHA256 mismatch")
    with np.load(split_path, allow_pickle=False) as archive:
        if int(archive["row_count"]) != expected_count or str(archive["bitorder"]) != "little":
            raise ValueError("Member split encoding disagreement")
        positions = np.flatnonzero(archive["members"] == int(job["member"]))
        if len(positions) != 1:
            raise ValueError("Requested member is absent or duplicated in split archive")
        index = int(positions[0])
        train_mask = _unpack(archive["train"][index], expected_count)
        validation_mask = _unpack(archive["validation"][index], expected_count)
        for name in ("split_seeds", "model_seeds", "shuffle_seeds"):
            registry_name = name[:-1] if name != "split_seeds" else "split_seed"
            if int(archive[name][index]) != int(job[registry_name]):
                raise ValueError(f"Archive/registry disagreement for {name}")
    if np.any(train_mask & validation_mask):
        raise ValueError("Training and validation memberships overlap")
    if int(train_mask.sum()) != int(split["train_rows"]) or int(validation_mask.sum()) != int(split["validation_rows"]):
        raise ValueError("Training/validation row counts disagree with registry")

    features = list(FEATURE_CONFIGURATIONS[configuration])
    columns = ["row_id", "common_eligible", *features, FROZEN_POLICY["target"]]
    raw = pd.read_csv(dataset_dir / "canonical_master_dataset.csv.gz", usecols=columns, low_memory=False)
    frame = raw.loc[raw["common_eligible"].astype(bool)].drop(columns="common_eligible")
    if len(frame) != expected_count or frame["row_id"].duplicated().any():
        raise ValueError("Canonical eligible rows fail count/uniqueness checks")
    frame = frame.set_index("row_id").reindex(row_ids)
    if frame.index.hasnans or frame.isna().any(axis=None):
        raise ValueError("Frozen row index is not an exact finite view of the canonical eligible data")
    matrix = frame[features].to_numpy(dtype=np.float64, copy=True)
    target = frame[[FROZEN_POLICY["target"]]].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite training inputs or target")

    input_scaler = RobustScaler().fit(matrix[train_mask])
    target_scaler = RobustScaler().fit(target[train_mask])
    return PreparedJob(
        job=job, split=split, dataset_manifest=dataset_manifest, split_manifest=split_manifest,
        features=features,
        train_row_ids=row_ids[train_mask], validation_row_ids=row_ids[validation_mask],
        x_train=input_scaler.transform(matrix[train_mask]).astype(np.float32),
        y_train=target_scaler.transform(target[train_mask]).astype(np.float32),
        x_validation=input_scaler.transform(matrix[validation_mask]).astype(np.float32),
        y_validation=target_scaler.transform(target[validation_mask]).astype(np.float32),
        input_scaler=input_scaler, target_scaler=target_scaler,
    )


def save_scalers(prepared: PreparedJob, output: Path) -> None:
    joblib.dump(prepared.input_scaler, output / "input_scaler.joblib")
    joblib.dump(prepared.target_scaler, output / "target_scaler.joblib")
