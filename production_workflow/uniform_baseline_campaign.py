"""Build and solve the twelve frozen development-only uniform-C baselines.

The scalar for each spatial experiment is the finite-element-area-weighted
mean of the adopted inversion control C over that experiment's eligible
development region.  Cell averages and cell areas are used for the integral;
geographic membership is classified at the cell barycentre.  The resulting
scalar replaces C on every eligible grounded CG2 degree of freedom, while the
adopted inversion is retained outside the common eligible grounded mask, as in
the ML hybrid controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from export_canonical_dataset import assign_regions
from forward_solve_campaign import configure_solver
from full_mesh_ensemble_predictions import full_mesh_context
from production_amundsen import manifest_identifier, sha256_file


SCHEMA = "jog-uniform-c-baseline-campaign-v1"
EXPERIMENTS = [*(f"SQ{i:02d}" for i in range(1, 11)), "REG_INTER", "REG_PIG"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cell_geometry(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return barycentres and positive areas in mesh-cell order."""
    vertices = np.asarray(mesh.coordinates.dat.data_ro[:, :2], dtype=np.float64)
    cell_vertices = np.asarray(mesh.coordinates.cell_node_map().values, dtype=np.int64)
    triangles = vertices[cell_vertices]
    centres = triangles.mean(axis=1)
    cross = (
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    areas = 0.5 * np.abs(cross)
    if not np.isfinite(centres).all() or not np.isfinite(areas).all() or np.any(areas <= 0):
        raise RuntimeError("Invalid finite-element cell geometry")
    return centres, areas


def square_rows(path: Path) -> dict[str, dict]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = {row["square_id"]: row for row in csv.DictReader(stream)}
    if set(rows) != {f"SQ{i:02d}" for i in range(1, 11)}:
        raise RuntimeError("Frozen square table is incomplete")
    return rows


def development_mask(
    experiment: str, centres: np.ndarray, region_codes: np.ndarray, squares: dict[str, dict]
) -> np.ndarray:
    if experiment.startswith("SQ"):
        row = squares[experiment]
        heldout = (
            (centres[:, 0] >= float(row["footprint_xmin_m"]))
            & (centres[:, 0] < float(row["footprint_xmax_m"]))
            & (centres[:, 1] >= float(row["footprint_ymin_m"]))
            & (centres[:, 1] < float(row["footprint_ymax_m"]))
        )
        return ~heldout
    if experiment == "REG_INTER":
        return np.isin(region_codes, [1, 2, 3])
    if experiment == "REG_PIG":
        return region_codes != 1
    raise ValueError(experiment)


def build_registry(args, object_, eligible_dofs, coordinates, reference_c, config) -> tuple[list[dict], dict]:
    import firedrake
    from icepack.constants import ice_density, water_density

    centres, areas = cell_geometry(object_.mesh)
    dg0 = firedrake.FunctionSpace(object_.mesh, "DG", 0)
    cell_map = np.asarray(dg0.cell_node_map().values[:, 0], dtype=np.int64)
    cell_c_function = firedrake.project(object_.C, dg0)
    cell_h_function = firedrake.interpolate(object_.h, dg0)
    cell_s_function = firedrake.interpolate(object_.s, dg0)
    cell_c = np.asarray(cell_c_function.dat.data_ro, dtype=np.float64)[cell_map]
    cell_h = np.asarray(cell_h_function.dat.data_ro, dtype=np.float64)[cell_map]
    cell_s = np.asarray(cell_s_function.dat.data_ro, dtype=np.float64)[cell_map]
    water_ratio = np.divide(
        float(water_density) * np.maximum(0.0, cell_h - cell_s),
        float(ice_density) * cell_h,
        out=np.ones_like(cell_h), where=cell_h > 0.0,
    )
    eligible_cells = (
        (np.maximum(1.0 - water_ratio, 0.0) > 0.1)
        & (cell_h > 0.0) & np.isfinite(cell_c)
    )

    partition_audit = read_json(
        args.repo_root / "production_workflow"
        / config["frozen_design"]["five_region_support"]["path"]
    )
    mesh_paths = {
        name: Path(partition_audit["meshes"][name]["path"])
        for name in ("PIG", "Thwaites", "Dotson")
    }
    region_codes, overlap_count = assign_regions(centres, mesh_paths)
    squares_path = (
        args.repo_root / "production_workflow"
        / config["frozen_design"]["selected_squares"]["path"]
    )
    squares = square_rows(squares_path)

    controls_dir = args.output / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    registry = []
    for experiment in EXPERIMENTS:
        development = eligible_cells & development_mask(experiment, centres, region_codes, squares)
        selected_area = float(np.sum(areas[development]))
        if not np.any(development) or not np.isfinite(selected_area) or selected_area <= 0:
            raise RuntimeError(f"Empty development area for {experiment}")
        mean_c = float(np.sum(areas[development] * cell_c[development]) / selected_area)
        values = reference_c.copy()
        values[eligible_dofs] = mean_c
        if not np.isfinite(values).all() or not np.array_equal(values[~eligible_dofs], reference_c[~eligible_dofs]):
            raise RuntimeError(f"Invalid hybrid uniform control for {experiment}")
        control_path = controls_dir / f"{experiment}_UNIFORM_C.npz"
        temporary = control_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, coordinates=coordinates, eligible_mask=eligible_dofs,
                reference_C=reference_c, uniform_C=np.float64(mean_c), control_C=values,
            )
        os.replace(temporary, control_path)
        registry.append({
            "control_id": f"{experiment}_UNIFORM_C", "experiment": experiment,
            "uniform_C": mean_c, "development_cell_count": int(development.sum()),
            "development_area_m2": selected_area,
            "eligible_cell_count": int(eligible_cells.sum()),
            "cell_count": int(len(areas)), "control_path": control_path.relative_to(args.output).as_posix(),
            "control_sha256": sha256_file(control_path),
            "control_values_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        })
    metadata = {
        "integration": "DG0 L2 projection gives the adopted-C mean on each finite-element cell; cell means are weighted by exact triangle area",
        "membership": "eligible grounding and geographic development membership evaluated at finite-element cell barycentres",
        "uniform_application": "experiment scalar replaces C at every common eligible grounded CG2 DOF; adopted inversion C retained elsewhere",
        "region_overlap_cell_count": int(np.sum(overlap_count > 1)),
        "selected_squares_sha256": sha256_file(squares_path),
    }
    return registry, metadata


def existing_complete(root: Path, record: dict, campaign_id: str) -> bool:
    path = root / "solves" / record["control_id"] / "forward_manifest.json"
    if not path.is_file():
        return False
    manifest = read_json(path)
    velocity = path.parent / str(manifest.get("velocity_path", ""))
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("baseline_campaign_manifest_id") == campaign_id
        and manifest_identifier(manifest) == manifest.get("manifest_id")
        and velocity.is_file() and sha256_file(velocity) == manifest.get("velocity_sha256")
    )


def solve_one(args, object_, adoption: dict, record: dict, campaign_id: str) -> dict:
    import firedrake
    import firedrake.adjoint

    destination = args.output / "solves" / record["control_id"]
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(args.output / record["control_path"], allow_pickle=False) as archive:
        values = archive["control_C"].copy()
    control = firedrake.Function(object_.Q, name="uniform_C")
    control.dat.data[:] = values
    attempts = []
    status, exception, velocity_values = "failed", None, None
    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    for limit in dict.fromkeys([args.primary_max_it, args.retry_max_it]):
        configure_solver(object_, int(limit))
        firedrake.adjoint.get_working_tape().clear_tape()
        attempt_start = time.perf_counter()
        attempt_exception = None
        try:
            velocity = object_.simulation_C(control)
            velocity_values = np.asarray(velocity.dat.data_ro, dtype=np.float64).copy()
            if velocity_values.shape != (35797, 2) or not np.isfinite(velocity_values).all():
                raise RuntimeError("Forward velocity is nonfinite or has the wrong shape")
            status = "complete"
        except Exception:
            attempt_exception = traceback.format_exc()
            exception = attempt_exception
            print(attempt_exception, flush=True)
        attempts.append({
            "snes_max_it": int(limit), "status": status,
            "elapsed_seconds": float(time.perf_counter() - attempt_start),
            "exception": attempt_exception,
        })
        if status == "complete":
            exception = None
            break
        if "DIVERGED_MAX_IT" not in str(attempt_exception):
            break
    manifest = {
        "schema": "jog-icepack-uniform-c-forward-control-v1", "status": status,
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "control_id": record["control_id"], "experiment": record["experiment"],
        "uniform_C": record["uniform_C"], "control_path": record["control_path"],
        "control_sha256": record["control_sha256"],
        "control_values_sha256": record["control_values_sha256"],
        "baseline_campaign_manifest_id": campaign_id,
        "adoption_manifest_id": adoption["adoption_manifest_id"],
        "definitive_point_manifest_id": adoption["point_manifest_id"],
        "reg_C": float(adoption["reg_c"]),
        "solver_policy": {
            "initial_velocity": "frozen object_.u_initial for every independent solve",
            "algorithm": "petsc newtontr / gmres / lu / mumps",
            "primary_snes_max_it": int(args.primary_max_it),
            "retry_snes_max_it": int(args.retry_max_it),
            "retry_trigger": "DIVERGED_MAX_IT only",
        },
        "attempts": attempts, "exception": exception,
        "environment": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version},
        "velocity_path": None, "velocity_sha256": None, "velocity_summary": None,
    }
    if status == "complete":
        velocity_path = destination / "velocity.npy"
        temporary = velocity_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, velocity_values, allow_pickle=False)
        os.replace(temporary, velocity_path)
        speed = np.linalg.norm(velocity_values, axis=1)
        manifest.update({
            "velocity_path": velocity_path.name, "velocity_sha256": sha256_file(velocity_path),
            "velocity_summary": {"finite": True, "minimum_speed": float(speed.min()),
                                 "median_speed": float(np.median(speed)), "maximum_speed": float(speed.max())},
        })
    manifest["manifest_id"] = manifest_identifier(manifest)
    atomic_json(destination / "forward_manifest.json", manifest)
    print(json.dumps({"control_id": record["control_id"], "status": status,
                      "seconds": manifest["elapsed_seconds"]}), flush=True)
    return manifest


def run(args) -> dict:
    args.repo_root = args.repo_root.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    object_, _, eligible, coordinates, reference_c, adoption = full_mesh_context(
        args.config.resolve(), args.repo_root, args.adoption_record.resolve()
    )
    config = read_json(args.config.resolve())
    registry, method = build_registry(args, object_, eligible, coordinates, reference_c, config)
    campaign = {
        "schema": SCHEMA, "status": "controls_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": EXPERIMENTS, "count": len(registry), "controls": registry,
        "method": method, "config_sha256": sha256_file(args.config.resolve()),
        "adoption_manifest_id": adoption["adoption_manifest_id"],
        "definitive_point_manifest_id": adoption["point_manifest_id"],
        "source_sha256": sha256_file(Path(__file__)),
    }
    campaign["manifest_id"] = manifest_identifier(campaign)
    atomic_json(args.output / "baseline_campaign_manifest.json", campaign)
    completed, failed = 0, 0
    for record in registry:
        if existing_complete(args.output, record, campaign["manifest_id"]):
            completed += 1
            continue
        result = solve_one(args, object_, adoption, record, campaign["manifest_id"])
        completed += result["status"] == "complete"
        failed += result["status"] != "complete"
        if failed and args.stop_on_failure:
            break
    summary = {
        "schema": "jog-uniform-c-baseline-campaign-summary-v1",
        "status": "complete" if completed == 12 and failed == 0 else "incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_campaign_manifest_id": campaign["manifest_id"],
        "completed": int(completed), "failed": int(failed), "expected": 12,
    }
    summary["manifest_id"] = manifest_identifier(summary)
    atomic_json(args.output / "campaign_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adoption-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--primary-max-it", type=int, default=50)
    parser.add_argument("--retry-max-it", type=int, default=100)
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
