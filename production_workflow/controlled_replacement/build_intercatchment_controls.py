#!/usr/bin/env python3
"""Build four controlled-replacement controls for the combined corridors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shapely

from production_amundsen import sha256_file


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def membership_and_boundary(mesh_path: Path, coordinates: np.ndarray):
    import firedrake
    mesh = firedrake.Mesh(str(mesh_path))
    vertices = np.asarray(mesh.coordinates.dat.data_ro[:, :2], dtype=np.float64)
    cells = np.asarray(mesh.coordinates.cell_node_map().values, dtype=np.int64)
    membership = mesh.locate_cells_ref_coords_and_dists(coordinates)[0] >= 0
    edges = np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = shapely.MultiLineString(vertices[unique[counts == 1]].tolist())
    return membership, boundary


def assign_regions(coordinates: np.ndarray, mesh_paths: dict[str, Path]) -> np.ndarray:
    order = ("PIG", "Thwaites", "Dotson")
    membership = np.zeros((len(coordinates), 3), dtype=bool)
    boundaries = []
    for column, name in enumerate(order):
        membership[:, column], boundary = membership_and_boundary(mesh_paths[name], coordinates)
        boundaries.append(boundary)
    overlap = np.flatnonzero(membership.sum(axis=1) > 1)
    if len(overlap):
        points = shapely.points(coordinates[overlap, 0], coordinates[overlap, 1])
        clearances = np.full((len(overlap), 3), -np.inf)
        for column, boundary in enumerate(boundaries):
            candidate = membership[overlap, column]
            if np.any(candidate):
                clearances[candidate, column] = shapely.distance(boundary, points[candidate])
        selected = np.argmax(clearances, axis=1)
        membership[overlap, :] = False
        membership[overlap, selected] = True
    codes = np.zeros(len(coordinates), dtype=np.int8)
    codes[membership[:, 0]] = 1
    codes[membership[:, 1]] = 2
    codes[membership[:, 2]] = 3
    outside = membership.sum(axis=1) == 0
    codes[outside & (coordinates[:, 1] >= -400_000.0)] = 4
    codes[outside & (coordinates[:, 1] < -400_000.0)] = 5
    if np.any(codes == 0):
        raise RuntimeError("Five-region assignment left control points unlabeled")
    return codes


def write_control(
    output: Path, job_id: str, experiment: str, configuration: str | None,
    kind: str, coordinates: np.ndarray, reference: np.ndarray,
    eligible: np.ndarray, geography: np.ndarray, intended: np.ndarray, source: dict,
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
    if not np.isfinite(control).all():
        raise RuntimeError(f"Nonfinite control for {job_id}")
    path = output / "controls" / f"{job_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, coordinates=coordinates, eligible_mask=eligible,
                            geography_mask=geography, replacement_mask=replacement,
                            reference_C=reference, control_C=control)
    os.replace(temporary, path)
    return {
        "job_id": job_id, "experiment": experiment, "configuration": configuration,
        "control_kind": kind, "control_path": str(path.relative_to(output)),
        "control_sha256": sha256_file(path), "coordinates_sha256": array_hash(coordinates),
        "reference_sha256": array_hash(reference), "eligible_mask_sha256": array_hash(eligible),
        "geography_mask_sha256": array_hash(geography),
        "replacement_mask_sha256": array_hash(replacement),
        "control_values_sha256": array_hash(control), "replacement_dofs": int(replacement.sum()),
        "geography_dofs": int(geography.sum()), "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--pig-mesh", type=Path, required=True)
    parser.add_argument("--thwaites-mesh", type=Path, required=True)
    parser.add_argument("--dotson-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    first = args.predictions / "REG_INTER_CFG04.npz"
    with np.load(first, allow_pickle=False) as archive:
        coordinates = archive["coordinates"].astype(np.float64)
        eligible = archive["eligible_mask"].astype(bool)
        reference = archive["reference_log_C"].astype(np.float64)
    if coordinates.shape != (35797, 2) or eligible.shape != (35797,) or reference.shape != (35797,):
        raise RuntimeError("Unexpected accepted prediction geometry")

    region_codes = assign_regions(
        coordinates,
        {"PIG": args.pig_mesh, "Thwaites": args.thwaites_mesh, "Dotson": args.dotson_mesh},
    )
    geography = np.isin(region_codes, [4, 5])
    experiment = "REG_INTER"
    jobs: list[dict] = []

    baseline_path = args.baselines / "controls" / "REG_INTER_UNIFORM_C.npz"
    with np.load(baseline_path, allow_pickle=False) as archive:
        if not (
            np.array_equal(coordinates, archive["coordinates"])
            and np.array_equal(eligible, archive["eligible_mask"].astype(bool))
            and np.array_equal(reference, archive["reference_C"])
        ):
            raise RuntimeError("Accepted uniform input disagrees for REG_INTER")
        uniform_value = float(archive["uniform_C"])
    jobs.append(
        write_control(
            args.output,
            "CR_REG_INTER_UNIFORM",
            experiment,
            None,
            "uniform",
            coordinates,
            reference,
            eligible,
            geography,
            np.full(reference.shape, uniform_value, dtype=np.float64),
            {"path": str(baseline_path), "sha256": sha256_file(baseline_path), "uniform_C": uniform_value},
        )
    )

    for configuration in ("CFG04", "CFG05", "CFG06"):
        prediction_path = args.predictions / f"{experiment}_{configuration}.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            if not (
                np.array_equal(coordinates, archive["coordinates"])
                and np.array_equal(eligible, archive["eligible_mask"].astype(bool))
                and np.array_equal(reference, archive["reference_log_C"])
            ):
                raise RuntimeError(f"Accepted prediction input disagrees for {experiment}/{configuration}")
            median = archive["median_log_C"].astype(np.float64)
        jobs.append(
            write_control(
                args.output,
                f"CR_{experiment}_{configuration}_MEDIAN",
                experiment,
                configuration,
                "median",
                coordinates,
                reference,
                eligible,
                geography,
                median,
                {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
            )
        )

    if len(jobs) != 4 or len({job["job_id"] for job in jobs}) != 4:
        raise RuntimeError("Inter-catchment registry is not exactly four unique jobs")
    masks = {job["replacement_mask_sha256"] for job in jobs}
    if len(masks) != 1:
        raise RuntimeError("Inter-catchment ML and uniform replacement masks differ")
    registry = {
        "schema": "jog-controlled-replacement-registry-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": (
            "reference C retained outside the combined PIG-Thwaites and "
            "Thwaites-Dotson corridor geography intersected with the accepted "
            "common eligible grounded CG2 replacement mask"
        ),
        "counts": {"jobs": 4, "intercatchment_median": 3, "intercatchment_uniform": 1},
        "inputs": {
            "predictions_root": str(args.predictions),
            "baselines_root": str(args.baselines),
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
