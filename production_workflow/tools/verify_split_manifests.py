#!/usr/bin/env python3
"""Independently verify frozen row partitions and deterministic split bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024): digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def unpack(value: np.ndarray, count: int) -> np.ndarray:
    return np.unpackbits(value, bitorder="little")[:count].astype(bool)


def membership_id(index_sha: str, packed: np.ndarray) -> str:
    digest = hashlib.sha256(bytes.fromhex(index_sha) + packed.tobytes()).hexdigest()
    return "sha256-row-membership-v1-" + digest


def expected_offset(experiment: str, member: int) -> int:
    if experiment.startswith("SQ"):
        return 100 * (int(experiment[2:]) - 1) + member - 1
    return 10000 + (0 if experiment == "REG_INTER" else 100) + member - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True); parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve(); dataset_dir = Path(args.dataset_dir).resolve()
    manifest = json.loads((bundle / "split_bundle_manifest.json").read_text())
    source_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    dataset_path = dataset_dir / "canonical_master_dataset.csv.gz"
    checks = {
        "manifest_id": manifest["manifest_id"] == canonical_id(manifest),
        "status": manifest["status"] == "complete",
        "declared_output_hashes": all(sha256(bundle / p) == h for p, h in manifest["output_sha256"].items()),
        "dataset_identity": manifest["dataset_manifest_id"] == source_manifest["manifest_id"] and manifest["dataset_sha256"] == sha256(dataset_path),
    }

    columns = ["row_id", "common_eligible", "region_code", "square_test_id", "square_footprint_id"]
    raw = pd.read_csv(dataset_path, usecols=columns, low_memory=False)
    frame = raw.loc[raw["common_eligible"].astype(bool)].copy()
    frame = frame.iloc[np.argsort(frame["row_id"].to_numpy(str), kind="stable")].reset_index(drop=True)
    expected_ids = frame["row_id"].to_numpy(str); count = len(frame)
    with gzip.open(bundle / "sorted_common_eligible_row_ids.txt.gz", "rt", encoding="utf-8") as stream:
        saved_ids = np.asarray([line.rstrip("\n") for line in stream], dtype=str)
    checks["sorted_row_index"] = len(saved_ids) == count and np.array_equal(saved_ids, expected_ids)
    checks["row_index_hash"] = manifest["row_index_sha256"] == sha256(bundle / "sorted_common_eligible_row_ids.txt.gz")

    experiments = table(bundle / "experiments.csv"); members = table(bundle / "member_splits.csv")
    jobs = table(bundle / "job_registry.csv"); pilots = table(bundle / "l2_pilot_registry.csv")
    checks["inventory_counts"] = len(experiments) == 12 and len(members) == 120 and len(jobs) == 660 and len(pilots) == 36
    expected_configs = {**{f"SQ{i:02d}": [1,2,3,4,5,6] for i in range(1,11)},
                        "REG_INTER": [4,5,6], "REG_PIG": [2,1,3]}
    member_lookup = {(r["experiment"], int(r["member"])): r for r in members}
    population_ok = split_ok = seed_ok = permutation_ok = membership_ok = distinct_ok = True
    recomputed_masks = {}
    region = frame["region_code"].to_numpy(np.int8)
    for row in experiments:
        experiment = row["experiment"]
        if experiment.startswith("SQ"):
            heldout_expected = frame["square_footprint_id"].eq(experiment).to_numpy()
            central = frame["square_test_id"].eq(experiment).to_numpy()
            population_expected = {"central_50km": central,
                                   "exclusion_annulus": heldout_expected & ~central,
                                   "full_130km": heldout_expected}
        elif experiment == "REG_INTER":
            heldout_expected = np.isin(region, [4,5])
            population_expected = {"both_corridors": heldout_expected,
                                   "pig_thwaites_corridor": region == 4,
                                   "thwaites_dotson_corridor": region == 5}
        else:
            heldout_expected = region == 1; population_expected = {"PIG": heldout_expected}
        development_expected = ~heldout_expected
        pop_file = bundle / row["population_mask_file"]
        archive = np.load(pop_file, allow_pickle=False)
        development = unpack(archive["development"], count); heldout = unpack(archive["heldout"], count)
        population_ok &= np.array_equal(development, development_expected) and np.array_equal(heldout, heldout_expected)
        population_ok &= int(row["development_rows"]) == int(development.sum()) and int(row["heldout_rows"]) == int(heldout.sum())
        membership_ok &= row["development_membership_id"] == membership_id(manifest["row_index_sha256"], archive["development"])
        membership_ok &= row["heldout_membership_id"] == membership_id(manifest["row_index_sha256"], archive["heldout"])
        for name, expected in population_expected.items():
            actual = unpack(archive[name], count)
            population_ok &= np.array_equal(actual, expected) and int(row[f"{name}_rows"]) == int(expected.sum())
            membership_ok &= row[f"{name}_membership_id"] == membership_id(manifest["row_index_sha256"], archive[name])
        recomputed_masks[experiment] = development

        member_rows = sorted((r for r in members if r["experiment"] == experiment), key=lambda r: int(r["member"]))
        split_file = bundle / member_rows[0]["split_file"]; split_archive = np.load(split_file, allow_pickle=False)
        validation_memberships = []
        for index, member_row in enumerate(member_rows):
            member = int(member_row["member"]); offset = expected_offset(experiment, member)
            train = unpack(split_archive["train"][index], count); validation = unpack(split_archive["validation"][index], count)
            expected_validation = math.ceil(0.10 * development.sum())
            split_ok &= not np.any(train & validation) and np.array_equal(train | validation, development)
            split_ok &= int(train.sum()) == int(member_row["train_rows"]) and int(validation.sum()) == expected_validation == int(member_row["validation_rows"])
            seed_ok &= int(member_row["offset"]) == offset
            seed_ok &= int(member_row["split_seed"]) == 20260811 + offset and int(member_row["model_seed"]) == 30260811 + offset and int(member_row["shuffle_seed"]) == 40260811 + offset
            development_positions = np.flatnonzero(development)
            permutation = np.random.default_rng(20260811 + offset).permutation(len(development_positions))
            expected_validation_mask = np.zeros(count, dtype=bool)
            expected_validation_mask[development_positions[permutation[:expected_validation]]] = True
            permutation_ok &= np.array_equal(validation, expected_validation_mask)
            membership_ok &= member_row["train_membership_id"] == membership_id(manifest["row_index_sha256"], split_archive["train"][index])
            membership_ok &= member_row["validation_membership_id"] == membership_id(manifest["row_index_sha256"], split_archive["validation"][index])
            validation_memberships.append(member_row["validation_membership_id"])
        distinct_ok &= len(set(validation_memberships)) == 10
    checks["population_masks_recomputed"] = bool(population_ok)
    checks["split_disjointness_and_coverage"] = bool(split_ok)
    checks["seed_formulas"] = bool(seed_ok)
    checks["seeded_permutations_recomputed"] = bool(permutation_ok)
    checks["membership_identifiers"] = bool(membership_ok)
    checks["ten_distinct_member_splits"] = bool(distinct_ok)

    job_ids = [r["job_id"] for r in jobs]
    job_ok = len(job_ids) == len(set(job_ids))
    grouped = {}
    for row in jobs:
        key = (row["experiment"], int(row["member"])); grouped.setdefault(key, []).append(row)
        member = member_lookup[key]
        job_ok &= row["split_id"] == member["split_id"] and row["split_file"] == member["split_file"]
        job_ok &= all(row[x] == member[x] for x in ("split_seed", "model_seed", "shuffle_seed"))
    for (experiment, member), rows_ in grouped.items():
        job_ok &= [int(r["configuration"][3:]) for r in rows_] == expected_configs[experiment]
        job_ok &= len({r["split_id"] for r in rows_}) == 1
    checks["production_registry_unique_and_matched"] = bool(job_ok and len(grouped) == 120)

    pilot_expected = {(e, f"CFG{c:02d}", l) for e in ("SQ01","SQ05","SQ10") for c in (1,3,6) for l in ("0e+00","1e-06","1e-05","1e-04")}
    pilot_actual = {(r["experiment"], r["configuration"], r["lambda_L2"]) for r in pilots}
    pilot_ok = pilot_actual == pilot_expected and len({r["pilot_job_id"] for r in pilots}) == 36
    pilot_ok &= all(int(r["member"]) == 1 and r["split_id"] == member_lookup[(r["experiment"],1)]["split_id"] for r in pilots)
    checks["l2_pilot_registry"] = bool(pilot_ok)

    passed = all(checks.values())
    result = {"schema": "jog-training-split-verification-v1", "passed": passed,
              "bundle_manifest_id": manifest["manifest_id"], "checks": checks,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__, "pandas": pd.__version__}}
    if args.output: Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
