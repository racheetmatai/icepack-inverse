#!/usr/bin/env python3
"""Independent structural/semantic verifier for Gate-2 diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve(); dataset_dir = Path(args.dataset_dir).resolve()
    manifest = json.loads((bundle / "diagnostics_manifest.json").read_text())
    checks: dict[str, bool] = {}
    checks["manifest_id"] = manifest["manifest_id"] == canonical_id(manifest)
    checks["status"] = manifest["status"] == "complete"
    checks["all_output_hashes"] = all(sha256(bundle / p) == h for p, h in manifest["output_sha256"].items())
    source_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    dataset_path = dataset_dir / "canonical_master_dataset.csv.gz"
    checks["dataset_identity"] = (
        manifest["dataset_manifest_id"] == source_manifest["manifest_id"]
        and manifest["dataset_sha256"] == sha256(dataset_path)
    )

    populations = rows(bundle / "population_summary.csv")
    support = rows(bundle / "support_categories.csv")
    marginal = rows(bundle / "marginal_support.csv")
    distribution = rows(bundle / "distribution_quantiles.csv")
    checks["expected_table_sizes"] = (
        len(populations) == 34 and len(support) == 192
        and len(distribution) == 34 * 13 * 2
    )
    pop_keys = {(r["experiment"], r["population"]): int(r["heldout_rows"]) for r in populations}
    checks["unique_population_keys"] = len(pop_keys) == len(populations)

    marginal_lookup: dict[tuple[str, str, str], list[dict]] = {}
    for row in marginal:
        marginal_lookup.setdefault((row["experiment"], row["population"], row["configuration"]), []).append(row)
    fraction_ok = count_ok = marginal_ok = True
    for row in support:
        key = (row["experiment"], row["population"], row["configuration"])
        fractions = np.asarray([float(row[name]) for name in
            ("neither_fraction", "marginal_only_fraction", "joint_only_fraction", "both_fraction")])
        fraction_ok &= bool(np.all((fractions >= 0) & (fractions <= 1)) and abs(fractions.sum() - 1) < 2e-12)
        count_ok &= int(row["heldout_rows"]) == pop_keys[key[:2]]
        details = marginal_lookup[key]
        minimum = min(float(x["heldout_coverage"]) for x in details)
        limiting = [x["feature"] for x in details if abs(float(x["heldout_coverage"]) - minimum) < 1e-14][0]
        marginal_ok &= abs(minimum - float(row["minimum_marginal_coverage"])) < 2e-12 and limiting == row["limiting_feature"]
        marginal_ok &= (row["population_passes_marginal_95"] == str(minimum >= 0.95))
        marginal_ok &= (row["population_passes_joint_95"] == str(float(row["joint_coverage"]) >= 0.95))
    checks["support_fractions"] = fraction_ok
    checks["support_population_counts"] = count_ok
    checks["marginal_summary_consistency"] = marginal_ok

    needed = ["common_eligible", "region_code", "square_test_id", "square_footprint_id"]
    frame = pd.read_csv(dataset_path, usecols=needed, low_memory=False)
    frame = frame.loc[frame["common_eligible"].astype(bool)].reset_index(drop=True)
    checks["eligible_count"] = len(frame) == int(source_manifest["common_eligible_count"])
    expected_counts = {}
    for number in range(1, 11):
        sid = f"SQ{number:02d}"
        central = frame["square_test_id"].eq(sid).to_numpy()
        footprint = frame["square_footprint_id"].eq(sid).to_numpy()
        expected_counts[(sid, "central_50km")] = int(central.sum())
        expected_counts[(sid, "exclusion_annulus")] = int((footprint & ~central).sum())
        expected_counts[(sid, "full_130km")] = int(footprint.sum())
    region = frame["region_code"].to_numpy(np.int8)
    expected_counts[("REG_INTER", "both_corridors")] = int(np.isin(region, [4, 5]).sum())
    expected_counts[("REG_INTER", "pig_thwaites_corridor")] = int((region == 4).sum())
    expected_counts[("REG_INTER", "thwaites_dotson_corridor")] = int((region == 5).sum())
    expected_counts[("REG_PIG", "PIG")] = int((region == 1).sum())
    checks["population_masks_recomputed"] = all(pop_keys[k] == v for k, v in expected_counts.items())

    archive = np.load(bundle / "point_support_categories.npz", allow_pickle=False)
    archive_ok = True
    support_lookup = {(r["experiment"], r["population"], r["configuration"]): r for r in support}
    for experiment in [f"SQ{i:02d}" for i in range(1, 11)] + ["REG_INTER", "REG_PIG"]:
        index = archive[f"{experiment}__row_index"].astype(int)
        archive_ok &= bool(np.all(index[:-1] < index[1:]))
        if experiment.startswith("SQ"):
            population_masks = {
                "central_50km": frame.loc[index, "square_test_id"].eq(experiment).to_numpy(),
                "full_130km": np.ones(len(index), dtype=bool),
            }
            population_masks["exclusion_annulus"] = ~population_masks["central_50km"]
        elif experiment == "REG_INTER":
            codes = frame.loc[index, "region_code"].to_numpy(np.int8)
            population_masks = {"both_corridors": np.ones(len(index), bool),
                                "pig_thwaites_corridor": codes == 4,
                                "thwaites_dotson_corridor": codes == 5}
        else:
            population_masks = {"PIG": np.ones(len(index), bool)}
        for key in archive.files:
            prefix = experiment + "__CFG"
            if not key.startswith(prefix):
                continue
            config = key.split("__", 1)[1]
            categories = archive[key]
            archive_ok &= len(categories) == len(index) and bool(np.all(categories <= 3))
            for population, mask in population_masks.items():
                counts = np.bincount(categories[mask], minlength=4) / int(mask.sum())
                row = support_lookup[(experiment, population, config)]
                recorded = np.asarray([float(row[x]) for x in
                    ("neither_fraction", "marginal_only_fraction", "joint_only_fraction", "both_fraction")])
                archive_ok &= bool(np.max(np.abs(counts - recorded)) < 2e-12)
    checks["point_archive_recomputed"] = archive_ok

    passed = all(checks.values())
    result = {"schema": "jog-heldout-distribution-verification-v1", "passed": passed,
              "bundle_manifest_id": manifest["manifest_id"], "checks": checks}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
