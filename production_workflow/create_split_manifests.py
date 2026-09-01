#!/usr/bin/env python3
"""Materialize exact deterministic row partitions for all frozen experiments."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_BASE = 20260811
MODEL_BASE = 30260811
SHUFFLE_BASE = 40260811
SQUARE_CONFIGS = [1, 2, 3, 4, 5, 6]
REGIONAL_CONFIGS = {"REG_INTER": [4, 5, 6], "REG_PIG": [2, 1, 3]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"No records for {path}")
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(records)


def write_deterministic_gzip_lines(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for value in values:
                    text.write(str(value)); text.write("\n")


def pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.astype(np.uint8), bitorder="little")


def membership_id(row_index_sha256: str, packed: np.ndarray) -> str:
    return "sha256-row-membership-v1-" + bytes_sha256(
        bytes.fromhex(row_index_sha256) + packed.tobytes()
    )


def experiment_specs(frame: pd.DataFrame) -> list[dict]:
    specs = []
    for square in range(1, 11):
        experiment = f"SQ{square:02d}"
        footprint = frame["square_footprint_id"].eq(experiment).to_numpy()
        central = frame["square_test_id"].eq(experiment).to_numpy()
        specs.append({
            "experiment": experiment, "kind": "square", "order": square,
            "development": ~footprint, "heldout": footprint,
            "populations": {
                "central_50km": central,
                "exclusion_annulus": footprint & ~central,
                "full_130km": footprint,
            },
            "configs": SQUARE_CONFIGS,
        })
    region = frame["region_code"].to_numpy(np.int8)
    inter = np.isin(region, [4, 5])
    specs.extend([
        {"experiment": "REG_INTER", "kind": "regional", "order": 1,
         "development": np.isin(region, [1, 2, 3]), "heldout": inter,
         "populations": {"both_corridors": inter,
                         "pig_thwaites_corridor": region == 4,
                         "thwaites_dotson_corridor": region == 5},
         "configs": REGIONAL_CONFIGS["REG_INTER"]},
        {"experiment": "REG_PIG", "kind": "regional", "order": 2,
         "development": region != 1, "heldout": region == 1,
         "populations": {"PIG": region == 1},
         "configs": REGIONAL_CONFIGS["REG_PIG"]},
    ])
    return specs


def offset_for(spec: dict, member: int) -> int:
    if spec["kind"] == "square":
        return 100 * (spec["order"] - 1) + (member - 1)
    return 10000 + 100 * (spec["order"] - 1) + (member - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    dataset_path = dataset_dir / "canonical_master_dataset.csv.gz"
    source_manifest_path = dataset_dir / "dataset_manifest.json"
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    (output / "population_masks").mkdir(parents=True, exist_ok=True)
    (output / "member_splits").mkdir(parents=True, exist_ok=True)
    (output / "source_snapshot").mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest["status"] != "complete":
        raise ValueError("Canonical dataset is not complete")
    if sha256(dataset_path) != source_manifest["output_sha256"][dataset_path.name]:
        raise ValueError("Canonical dataset hash mismatch")
    columns = ["row_id", "common_eligible", "region_code", "square_test_id", "square_footprint_id"]
    raw = pd.read_csv(dataset_path, usecols=columns, low_memory=False)
    frame = raw.loc[raw["common_eligible"].astype(bool)].copy()
    if len(frame) != int(source_manifest["common_eligible_count"]):
        raise AssertionError("Common eligible count mismatch")
    if frame["row_id"].duplicated().any():
        raise AssertionError("Eligible row IDs are not unique")
    order = np.argsort(frame["row_id"].to_numpy(str), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    sorted_ids = frame["row_id"].to_numpy(str)
    row_index_path = output / "sorted_common_eligible_row_ids.txt.gz"
    write_deterministic_gzip_lines(row_index_path, sorted_ids)
    row_index_sha = sha256(row_index_path)
    row_count = len(frame); packed_length = math.ceil(row_count / 8)

    experiments, members, jobs = [], [], []
    for spec in experiment_specs(frame):
        experiment = spec["experiment"]
        development = spec["development"].astype(bool)
        heldout = spec["heldout"].astype(bool)
        if np.any(development & heldout) or not np.all(development | heldout):
            raise AssertionError(f"Population partition failure: {experiment}")
        population_arrays = {"development": pack(development), "heldout": pack(heldout)}
        for name, mask in spec["populations"].items():
            population_arrays[name] = pack(mask)
        population_path = output / "population_masks" / f"{experiment}.npz"
        np.savez_compressed(population_path, row_count=np.int64(row_count), bitorder=np.asarray("little"), **population_arrays)
        experiment_record = {
            "experiment": experiment, "kind": spec["kind"],
            "development_rows": int(development.sum()), "heldout_rows": int(heldout.sum()),
            "population_mask_file": population_path.relative_to(output).as_posix(),
            "population_mask_sha256": sha256(population_path),
            "development_membership_id": membership_id(row_index_sha, population_arrays["development"]),
            "heldout_membership_id": membership_id(row_index_sha, population_arrays["heldout"]),
            "configurations": ";".join(f"CFG{x:02d}" for x in spec["configs"]),
        }
        for name, mask in spec["populations"].items():
            experiment_record[f"{name}_rows"] = int(mask.sum())
            experiment_record[f"{name}_membership_id"] = membership_id(row_index_sha, population_arrays[name])
        experiments.append(experiment_record)

        development_positions = np.flatnonzero(development)
        validation_count = int(math.ceil(0.10 * len(development_positions)))
        train_count = len(development_positions) - validation_count
        train_packed = np.empty((10, packed_length), dtype=np.uint8)
        validation_packed = np.empty_like(train_packed)
        member_metadata = []
        for member in range(1, 11):
            offset = offset_for(spec, member)
            split_seed = SPLIT_BASE + offset
            model_seed = MODEL_BASE + offset
            shuffle_seed = SHUFFLE_BASE + offset
            permutation = np.random.default_rng(split_seed).permutation(len(development_positions))
            validation_positions = development_positions[permutation[:validation_count]]
            training_positions = development_positions[permutation[validation_count:]]
            validation_mask = np.zeros(row_count, dtype=bool); validation_mask[validation_positions] = True
            training_mask = np.zeros(row_count, dtype=bool); training_mask[training_positions] = True
            if np.any(training_mask & validation_mask) or not np.array_equal(training_mask | validation_mask, development):
                raise AssertionError(f"Split failure: {experiment}/M{member:02d}")
            train_packed[member - 1] = pack(training_mask)
            validation_packed[member - 1] = pack(validation_mask)
            metadata = {
                "member": member, "offset": offset, "split_seed": split_seed,
                "model_seed": model_seed, "shuffle_seed": shuffle_seed,
                "train_rows": train_count, "validation_rows": validation_count,
                "train_membership_id": membership_id(row_index_sha, train_packed[member - 1]),
                "validation_membership_id": membership_id(row_index_sha, validation_packed[member - 1]),
            }
            member_metadata.append(metadata)
        split_path = output / "member_splits" / f"{experiment}.npz"
        np.savez_compressed(
            split_path, row_count=np.int64(row_count), bitorder=np.asarray("little"),
            members=np.arange(1, 11, dtype=np.int8), train=train_packed,
            validation=validation_packed,
            split_seeds=np.asarray([x["split_seed"] for x in member_metadata], dtype=np.int64),
            model_seeds=np.asarray([x["model_seed"] for x in member_metadata], dtype=np.int64),
            shuffle_seeds=np.asarray([x["shuffle_seed"] for x in member_metadata], dtype=np.int64),
        )
        split_sha = sha256(split_path)
        for metadata in member_metadata:
            member = metadata["member"]
            split_id = f"{experiment}_M{member:02d}"
            members.append({
                "split_id": split_id, "experiment": experiment,
                **metadata,
                "split_file": split_path.relative_to(output).as_posix(),
                "split_file_sha256": split_sha,
            })
            for config in spec["configs"]:
                jobs.append({
                    "job_id": f"{experiment}_CFG{config:02d}_M{member:02d}",
                    "experiment": experiment, "configuration": f"CFG{config:02d}",
                    "member": member, "split_id": split_id,
                    "split_seed": metadata["split_seed"],
                    "model_seed": metadata["model_seed"],
                    "shuffle_seed": metadata["shuffle_seed"],
                    "split_file": split_path.relative_to(output).as_posix(),
                })

    write_csv(output / "experiments.csv", experiments)
    write_csv(output / "member_splits.csv", members)
    write_csv(output / "job_registry.csv", jobs)
    pilot_jobs = []
    member_lookup = {(x["experiment"], x["member"]): x for x in members}
    for experiment in ("SQ01", "SQ05", "SQ10"):
        member = member_lookup[(experiment, 1)]
        for config in (1, 3, 6):
            for l2 in (0.0, 1e-6, 1e-5, 1e-4):
                pilot_jobs.append({
                    "pilot_job_id": f"L2_{experiment}_CFG{config:02d}_M01_L2_{l2:.0e}",
                    "experiment": experiment, "configuration": f"CFG{config:02d}",
                    "member": 1, "lambda_L2": f"{l2:.0e}",
                    "split_id": member["split_id"], "split_seed": member["split_seed"],
                    "model_seed": member["model_seed"], "shuffle_seed": member["shuffle_seed"],
                })
    write_csv(output / "l2_pilot_registry.csv", pilot_jobs)

    methods = {
        "schema": "jog-training-split-method-v1",
        "dataset_manifest_id": source_manifest["manifest_id"],
        "common_population": "common_eligible: grounded phi>0.1 and finite union of all twelve predictors",
        "row_index": "lexicographically sorted stable row IDs; bit-packed masks use this exact order",
        "mask_encoding": {"function": "numpy.packbits", "bitorder": "little", "row_count": row_count,
                          "unused_terminal_bits": 8 * packed_length - row_count},
        "split_algorithm": "numpy.default_rng(split_seed).permutation(N); first ceil(0.10*N) development positions are validation; remainder training",
        "split_seeds": "20260811 + frozen offset", "model_seeds": "30260811 + frozen offset",
        "shuffle_seeds": "40260811 + frozen offset",
        "square_offset": "100*(square_order-1)+(member-1)",
        "regional_offset": "10000+100*(regional_order-1)+(member-1); REG_INTER=1, REG_PIG=2",
        "matched_configuration_policy": "one experiment/member split is referenced unchanged by every approved configuration",
        "validation_overlap_policy": "validation subsets may overlap across different ensemble members",
        "feature_specific_removal": False,
    }
    (output / "methods.json").write_text(json.dumps(methods, indent=2) + "\n")
    shutil.copy2(Path(__file__).resolve(), output / "source_snapshot" / Path(__file__).name)

    output_hashes = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    result = {
        "schema": "jog-training-split-bundle-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest_id": source_manifest["manifest_id"],
        "dataset_sha256": sha256(dataset_path), "eligible_rows": row_count,
        "experiment_count": len(experiments), "member_split_count": len(members),
        "production_job_count": len(jobs), "l2_pilot_job_count": len(pilot_jobs),
        "row_index_sha256": row_index_sha, "output_sha256": output_hashes,
    }
    result["manifest_id"] = canonical_id(result)
    (output / "split_bundle_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("manifest_id", "eligible_rows", "experiment_count", "member_split_count", "production_job_count", "l2_pilot_job_count")}, indent=2))


if __name__ == "__main__":
    main()
