#!/usr/bin/env python3
"""Build controls for the spatially restricted replacement experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from export_canonical_dataset import assign_regions
from production_amundsen import sha256_file


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def load_squares(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10:
        raise RuntimeError("Expected exactly ten frozen square definitions")
    return {row["square_id"]: row for row in rows}


def write_control(
    output: Path,
    job_id: str,
    experiment: str,
    configuration: str | None,
    kind: str,
    coordinates: np.ndarray,
    reference: np.ndarray,
    eligible: np.ndarray,
    geography: np.ndarray,
    intended: np.ndarray,
    source: dict,
) -> dict:
    replacement = np.asarray(eligible & geography, dtype=bool)
    if replacement.shape != reference.shape or not replacement.any():
        raise RuntimeError(f"Empty or invalid replacement mask for {job_id}")
    control = reference.copy()
    control[replacement] = intended[replacement]
    if not np.array_equal(control[~replacement], reference[~replacement]):
        raise RuntimeError(f"Outside-mask control changed for {job_id}")
    if not np.array_equal(control[replacement], intended[replacement]):
        raise RuntimeError(f"Inside-mask control differs from intended values for {job_id}")
    if not (np.isfinite(control).all() and np.isfinite(reference).all()):
        raise RuntimeError(f"Nonfinite control for {job_id}")
    path = output / "controls" / f"{job_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            coordinates=coordinates,
            eligible_mask=eligible,
            geography_mask=geography,
            replacement_mask=replacement,
            reference_C=reference,
            control_C=control,
        )
    os.replace(temporary, path)
    return {
        "job_id": job_id,
        "experiment": experiment,
        "configuration": configuration,
        "control_kind": kind,
        "control_path": str(path.relative_to(output)),
        "control_sha256": sha256_file(path),
        "coordinates_sha256": array_hash(coordinates),
        "reference_sha256": array_hash(reference),
        "eligible_mask_sha256": array_hash(eligible),
        "geography_mask_sha256": array_hash(geography),
        "replacement_mask_sha256": array_hash(replacement),
        "control_values_sha256": array_hash(control),
        "replacement_dofs": int(replacement.sum()),
        "geography_dofs": int(geography.sum()),
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--squares", type=Path, required=True)
    parser.add_argument("--pig-mesh", type=Path, required=True)
    parser.add_argument("--thwaites-mesh", type=Path, required=True)
    parser.add_argument("--dotson-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    first = args.predictions / "SQ01_CFG01.npz"
    with np.load(first, allow_pickle=False) as archive:
        coordinates = archive["coordinates"].astype(np.float64)
        eligible = archive["eligible_mask"].astype(bool)
        reference = archive["reference_log_C"].astype(np.float64)
    if coordinates.shape != (35797, 2) or eligible.shape != (35797,) or reference.shape != (35797,):
        raise RuntimeError("Unexpected accepted prediction geometry")

    squares = load_squares(args.squares)
    x, y = coordinates[:, 0], coordinates[:, 1]
    geography_masks: dict[str, np.ndarray] = {}
    for square, row in squares.items():
        geography_masks[square] = (
            (x >= float(row["footprint_xmin_m"]))
            & (x < float(row["footprint_xmax_m"]))
            & (y >= float(row["footprint_ymin_m"]))
            & (y < float(row["footprint_ymax_m"]))
        )
    region_codes, _ = assign_regions(
        coordinates,
        {"PIG": args.pig_mesh, "Thwaites": args.thwaites_mesh, "Dotson": args.dotson_mesh},
    )
    geography_masks["REG_PIG"] = region_codes == 1

    jobs: list[dict] = []
    for experiment in [f"SQ{i:02d}" for i in range(1, 11)]:
        geography = geography_masks[experiment]
        baseline_path = args.baselines / "controls" / f"{experiment}_UNIFORM_C.npz"
        with np.load(baseline_path, allow_pickle=False) as archive:
            baseline_coordinates = archive["coordinates"]
            baseline_eligible = archive["eligible_mask"].astype(bool)
            baseline_reference = archive["reference_C"]
            uniform_value = float(archive["uniform_C"])
        if not (np.array_equal(coordinates, baseline_coordinates)
                and np.array_equal(eligible, baseline_eligible)
                and np.array_equal(reference, baseline_reference)):
            raise RuntimeError(f"Accepted uniform input disagrees for {experiment}")
        intended_uniform = np.full(reference.shape, uniform_value, dtype=np.float64)
        jobs.append(write_control(
            args.output, f"CR_{experiment}_UNIFORM", experiment, None, "uniform",
            coordinates, reference, eligible, geography, intended_uniform,
            {"path": str(baseline_path), "sha256": sha256_file(baseline_path), "uniform_C": uniform_value},
        ))
        for cfg_index in range(1, 7):
            configuration = f"CFG{cfg_index:02d}"
            prediction_path = args.predictions / f"{experiment}_{configuration}.npz"
            with np.load(prediction_path, allow_pickle=False) as archive:
                local_coordinates = archive["coordinates"]
                local_eligible = archive["eligible_mask"].astype(bool)
                local_reference = archive["reference_log_C"]
                median = archive["median_log_C"].astype(np.float64)
            if not (np.array_equal(coordinates, local_coordinates)
                    and np.array_equal(eligible, local_eligible)
                    and np.array_equal(reference, local_reference)):
                raise RuntimeError(f"Accepted prediction input disagrees for {experiment}/{configuration}")
            jobs.append(write_control(
                args.output, f"CR_{experiment}_{configuration}_MEDIAN", experiment,
                configuration, "median", coordinates, reference, eligible, geography, median,
                {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
            ))

    experiment = "REG_PIG"
    geography = geography_masks[experiment]
    baseline_path = args.baselines / "controls" / "REG_PIG_UNIFORM_C.npz"
    with np.load(baseline_path, allow_pickle=False) as archive:
        if not (np.array_equal(coordinates, archive["coordinates"])
                and np.array_equal(eligible, archive["eligible_mask"].astype(bool))
                and np.array_equal(reference, archive["reference_C"])):
            raise RuntimeError("Accepted uniform input disagrees for REG_PIG")
        uniform_value = float(archive["uniform_C"])
    jobs.append(write_control(
        args.output, "CR_REG_PIG_UNIFORM", experiment, None, "uniform",
        coordinates, reference, eligible, geography,
        np.full(reference.shape, uniform_value, dtype=np.float64),
        {"path": str(baseline_path), "sha256": sha256_file(baseline_path), "uniform_C": uniform_value},
    ))
    configuration = "CFG02"
    prediction_path = args.predictions / f"{experiment}_{configuration}.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        if not (np.array_equal(coordinates, archive["coordinates"])
                and np.array_equal(eligible, archive["eligible_mask"].astype(bool))
                and np.array_equal(reference, archive["reference_log_C"])):
            raise RuntimeError("Accepted prediction input disagrees for REG_PIG/CFG02")
        median = archive["median_log_C"].astype(np.float64)
    jobs.append(write_control(
        args.output, "CR_REG_PIG_CFG02_MEDIAN", experiment, configuration, "median",
        coordinates, reference, eligible, geography, median,
        {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
    ))

    if len(jobs) != 72 or len({job["job_id"] for job in jobs}) != 72:
        raise RuntimeError("Controlled registry is not exactly 72 unique jobs")
    for experiment in [f"SQ{i:02d}" for i in range(1, 11)] + ["REG_PIG"]:
        masks = {job["replacement_mask_sha256"] for job in jobs if job["experiment"] == experiment}
        if len(masks) != 1:
            raise RuntimeError(f"ML and uniform replacement masks differ for {experiment}")

    registry = {
        "schema": "jog-controlled-replacement-registry-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": (
            "reference C retained outside the complete withheld geography intersected "
            "with the accepted common eligible grounded CG2 replacement mask"
        ),
        "counts": {"jobs": 72, "square_median": 60, "square_uniform": 10,
                   "pig_median": 1, "pig_uniform": 1},
        "inputs": {
            "predictions_root": str(args.predictions),
            "baselines_root": str(args.baselines),
            "squares": {"path": str(args.squares), "sha256": sha256_file(args.squares)},
            "region_meshes": {
                "PIG": {"path": str(args.pig_mesh), "sha256": sha256_file(args.pig_mesh)},
                "Thwaites": {"path": str(args.thwaites_mesh), "sha256": sha256_file(args.thwaites_mesh)},
                "Dotson": {"path": str(args.dotson_mesh), "sha256": sha256_file(args.dotson_mesh)},
            },
        },
        "jobs": jobs,
    }
    atomic_json(args.output / "job_registry.json", registry)
    print(json.dumps(registry["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
