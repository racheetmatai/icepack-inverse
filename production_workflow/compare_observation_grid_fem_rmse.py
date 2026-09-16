"""Compare observation-row RMSE with an area-integrated FEM diagnostic.

This script only reads saved velocity fields.  It does not run an Icepack
diagnostic solve.  The FEM population is the set of production-mesh triangles
whose barycentre falls in a 450 m pixel represented by a common-eligible
observation row and in the requested test geography.  Thus the integral is
exact for the FE residual on the retained whole triangles, while the mapping
of observation coverage and test boundaries to triangles is explicitly a
cell-barycentre approximation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_forward_campaign import build_object, build_observation_alignment, interpolate_velocity
from production_amundsen import manifest_identifier, sha256_file
from uniform_baseline_campaign import assign_regions, cell_geometry, read_json, square_rows


CONFIGS = [f"CFG{i:02d}" for i in range(1, 7)]
EXPERIMENTS = [f"SQ{i:02d}" for i in range(1, 11)]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def field_from_values(firedrake, space, values, name):
    result = firedrake.Function(space, name=name)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != result.dat.data_ro.shape or not np.isfinite(values).all():
        raise ValueError(f"Invalid values for {name}: {values.shape}")
    result.dat.data[:] = values
    return result


def all_metric(record: dict, population: str) -> dict:
    candidates = [m for m in record["metrics"]
                  if m["population"] == population and m["support_stratum"] == "all"]
    if len(candidates) != 1:
        raise ValueError(f"Missing authoritative metric: {record['control_id']} {population}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    import firedrake
    import icepack
    from firedrake import as_vector, assemble, dx, inner

    config_path = root / "production_workflow/amundsen_production_config.json"
    adoption_path = root / "production_workflow/gate1_results/gate1_definitive_inversion_adoption_20260819_a.json"
    dataset_path = root / "production_workflow/gate2_results/gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz"
    corrected_root = root / "production_workflow/gate4_forward_evaluation_support_aligned_20260910"
    solve_root = root / "production_workflow/gate4_forward_solve_campaign_20260828_a/solves"
    baseline_root = root / "production_workflow/gate4_uniform_c_baselines_20260829_a"
    squares_path = root / "production_workflow/frozen_design/selected_squares.csv"

    # docker cp nests a source directory when its destination already exists.
    # Accept that staging detail without changing the identities of input files.
    if not (corrected_root / "evaluation_manifest.json").is_file():
        corrected_root = corrected_root / corrected_root.name
    solve_parent = solve_root.parent
    if not (solve_root / "SQ01_CFG02_MEDIAN/velocity.npy").is_file():
        solve_root = solve_parent / solve_parent.name / "solves"
    if not (baseline_root / "solves/SQ01_UNIFORM_C/velocity.npy").is_file():
        baseline_root = baseline_root / baseline_root.name

    # SHAP belongs to legacy exploratory methods imported by Invert and is not
    # used by this saved-field diagnostic; the production image omits it.
    sys.modules.setdefault("shap", types.ModuleType("shap"))
    object_, adoption, config = build_object(config_path, root, adoption_path)
    frame, lookup = build_observation_alignment(object_, dataset_path)
    observed_rows = frame[["observed_vx", "observed_vy"]].to_numpy(np.float64)
    inversion_rows = interpolate_velocity(object_, np.asarray(adoption["velocity"].dat.data_ro), lookup)

    # Icepack's existing CG2 representation of MEaSUREs on the production mesh.
    raw_observed_fe_rows = np.column_stack((
        np.asarray(icepack.interpolate(object_.u_exp_x, object_.Δ).dat.data_ro)[lookup],
        np.asarray(icepack.interpolate(object_.u_exp_y, object_.Δ).dat.data_ro)[lookup],
    ))
    roundtrip_finite = np.isfinite(raw_observed_fe_rows).all(axis=1)
    if not np.any(roundtrip_finite):
        raise RuntimeError("Mesh-represented MEaSUREs velocity is nowhere finite on retained rows")
    observation_roundtrip_rmse = float(np.sqrt(np.mean(np.sum(
        (raw_observed_fe_rows[roundtrip_finite]-observed_rows[roundtrip_finite])**2, axis=1))))

    centres, areas = cell_geometry(object_.mesh)
    dg0 = firedrake.FunctionSpace(object_.mesh, "DG", 0)
    cell_map = np.asarray(dg0.cell_node_map().values[:, 0], dtype=np.int64)
    if len(centres) != object_.mesh.num_cells() or len(np.unique(cell_map)) != len(cell_map):
        raise RuntimeError("Unexpected production cell/DG0 mapping")

    # Linear raster interpolation propagates missing pixels to some CG2 DOFs.
    # Retain only triangles on which both observed components are finite at
    # every local CG2 node.  Non-finite values are replaced only in a copy used
    # outside that zero-masked domain; no retained observation is filled.
    q_cell_nodes = np.asarray(object_.Q.cell_node_map().values, dtype=np.int64)
    raw_ux = np.asarray(object_.u_exp_x.dat.data_ro, dtype=np.float64)
    raw_uy = np.asarray(object_.u_exp_y.dat.data_ro, dtype=np.float64)
    observed_cell_finite = (np.isfinite(raw_ux[q_cell_nodes]).all(axis=1)
                            & np.isfinite(raw_uy[q_cell_nodes]).all(axis=1))
    safe_ux = object_.u_exp_x.copy(deepcopy=True); safe_uy = object_.u_exp_y.copy(deepcopy=True)
    safe_ux.dat.data[:] = np.nan_to_num(raw_ux, nan=0.0, posinf=0.0, neginf=0.0)
    safe_uy.dat.data[:] = np.nan_to_num(raw_uy, nan=0.0, posinf=0.0, neginf=0.0)
    observed_fe = as_vector((safe_ux, safe_uy))

    # The observation population represents regular 450 m pixels.  Map each FE
    # cell barycentre to its nearest pixel centre and require that exact row ID
    # to be in the frozen common-eligible population.
    x0 = float(frame.x.min()); y0 = float(frame.y.min()); spacing = 450.0
    sx = x0 + np.rint((centres[:, 0] - x0) / spacing) * spacing
    sy = y0 + np.rint((centres[:, 1] - y0) / spacing) * spacing
    if np.max(np.abs(centres[:, 0]-sx)) > spacing/2 + 1e-7 or np.max(np.abs(centres[:, 1]-sy)) > spacing/2 + 1e-7:
        raise RuntimeError("Observation-pixel mapping failed")
    from src.data_preprocessing import stable_xy_row_ids
    snapped_ids = np.asarray(stable_xy_row_ids(sx, sy), dtype=str)
    eligible_ids = pd.Index(frame.row_id.astype(str))
    pixel_supported = eligible_ids.get_indexer(snapped_ids) >= 0

    partition = read_json(root / "production_workflow" /
                          config["frozen_design"]["five_region_support"]["path"])
    mesh_paths = {name: Path(partition["meshes"][name]["path"])
                  for name in ("PIG", "Thwaites", "Dotson")}
    region_codes, overlap = assign_regions(centres, mesh_paths)
    region_overlap_cell_count = int(np.sum(overlap > 1))
    squares = square_rows(squares_path)

    def geography(experiment: str) -> np.ndarray:
        if experiment.startswith("SQ"):
            row = squares[experiment]
            return ((centres[:, 0] >= float(row["test_xmin_m"])) & (centres[:, 0] < float(row["test_xmax_m"]))
                    & (centres[:, 1] >= float(row["test_ymin_m"])) & (centres[:, 1] < float(row["test_ymax_m"])))
        if experiment == "REG_PIG":
            return region_codes == 1
        raise ValueError(experiment)

    masks = {}
    mask_metadata = {}
    for experiment in EXPERIMENTS + ["REG_PIG"]:
        selected = pixel_supported & observed_cell_finite & geography(experiment)
        if not np.any(selected):
            raise RuntimeError(f"Empty FEM population for {experiment}")
        mask = firedrake.Function(dg0, name=f"mask_{experiment}")
        mask.dat.data[:] = 0.0
        mask.dat.data[cell_map[selected]] = 1.0
        area_sum = float(np.sum(areas[selected]))
        assembled_area = float(assemble(mask * dx))
        if not np.isclose(area_sum, assembled_area, rtol=1e-12, atol=1e-3):
            raise RuntimeError(f"Area assembly mismatch for {experiment}")
        masks[experiment] = mask
        row_mask = (frame.square_test_id.astype(str).to_numpy() == experiment) if experiment.startswith("SQ") else (frame.region_code.to_numpy() == 1)
        row_count = int(row_mask.sum())
        mask_metadata[experiment] = {
            "fem_cells": int(selected.sum()), "fem_area_m2": assembled_area,
            "observation_rows": row_count, "row_pixel_area_m2": row_count * spacing**2,
            "fem_to_row_pixel_area_ratio": assembled_area/(row_count*spacing**2),
        }

    def fem_rmse(field, experiment: str, degree: int) -> float:
        residual = field - observed_fe
        numerator = float(assemble(masks[experiment] * inner(residual, residual) * dx(degree=degree)))
        denominator = float(assemble(masks[experiment] * dx))
        if numerator < -1e-8 or denominator <= 0:
            raise RuntimeError("Invalid FEM integral")
        return float(np.sqrt(max(numerator, 0.0)/denominator))

    # Basic mathematical checks on the same masks.
    zero = firedrake.Function(object_.V)
    constant = firedrake.Function(object_.V)
    constant.interpolate(as_vector((3.0, 4.0)))
    synthetic_zero = fem_rmse(observed_fe, "SQ01", 6)
    synthetic_five = float(np.sqrt(assemble(masks["SQ01"] * inner(constant, constant) * dx(degree=6)) /
                                   assemble(masks["SQ01"] * dx)))
    if abs(synthetic_zero) > 1e-10 or not np.isclose(synthetic_five, 5.0, atol=1e-10):
        raise RuntimeError("Synthetic FEM checks failed")

    inversion_field = field_from_values(firedrake, object_.V, adoption["velocity"].dat.data_ro, "inversion_velocity")
    baseline_manifest = read_json(baseline_root / "baseline_campaign_manifest.json")
    baseline_fields = {}; baseline_rows = {}
    for experiment in EXPERIMENTS + ["REG_PIG"]:
        manifest_path = baseline_root / "solves" / f"{experiment}_UNIFORM_C" / "forward_manifest.json"
        manifest = read_json(manifest_path)
        values = np.load(manifest_path.parent / manifest["velocity_path"], allow_pickle=False)
        baseline_fields[experiment] = field_from_values(firedrake, object_.V, values, f"uniform_{experiment}")
        baseline_rows[experiment] = interpolate_velocity(object_, values, lookup)

    cases = [(sq, cfg) for sq in EXPERIMENTS for cfg in CONFIGS] + [("REG_PIG", "CFG02")]
    rows = []
    first_case_timing = None
    for index, (experiment, cfg) in enumerate(cases):
        stamp = time.perf_counter()
        control_id = f"{experiment}_{cfg}_MEDIAN"
        forward_manifest_path = solve_root / control_id / "forward_manifest.json"
        forward_manifest = read_json(forward_manifest_path)
        values = np.load(forward_manifest_path.parent / forward_manifest["velocity_path"], allow_pickle=False)
        model = field_from_values(firedrake, object_.V, values, control_id)
        model_rows = interpolate_velocity(object_, values, lookup)
        population = "central_50km" if experiment.startswith("SQ") else "PIG"
        auth_record = read_json(corrected_root / "control_metrics" / f"{control_id}.json")
        auth = all_metric(auth_record, population)
        row_mask = (frame.square_test_id.astype(str).to_numpy() == experiment) if experiment.startswith("SQ") else (frame.region_code.to_numpy() == 1)
        grid_model = float(np.sqrt(np.mean(np.sum((model_rows[row_mask]-observed_rows[row_mask])**2, axis=1))))
        grid_uniform = float(np.sqrt(np.mean(np.sum((baseline_rows[experiment][row_mask]-observed_rows[row_mask])**2, axis=1))))
        grid_inversion = float(np.sqrt(np.mean(np.sum((inversion_rows[row_mask]-observed_rows[row_mask])**2, axis=1))))
        for got, expected, label in ((grid_model, auth["vector_rmse_m_per_a"], "ML"),
                                     (grid_uniform, auth["uniform_vector_rmse_m_per_a"], "uniform"),
                                     (grid_inversion, auth["inversion_vector_rmse_m_per_a"], "inversion")):
            if not np.isclose(got, expected, rtol=1e-12, atol=1e-10):
                raise RuntimeError(f"Authoritative {label} grid RMSE mismatch: {control_id}")
        fem_model_4 = fem_rmse(model, experiment, 4); fem_model_6 = fem_rmse(model, experiment, 6)
        fem_uniform = fem_rmse(baseline_fields[experiment], experiment, 6)
        fem_inversion = fem_rmse(inversion_field, experiment, 6)
        row = {
            "experiment": experiment, "configuration": cfg, "control_id": control_id,
            **mask_metadata[experiment],
            "grid_ml_rmse_m_per_a": grid_model, "fem_ml_rmse_m_per_a": fem_model_6,
            "grid_uniform_rmse_m_per_a": grid_uniform, "fem_uniform_rmse_m_per_a": fem_uniform,
            "grid_inversion_rmse_m_per_a": grid_inversion, "fem_inversion_rmse_m_per_a": fem_inversion,
            "grid_relative_rmse": grid_model/grid_uniform, "fem_relative_rmse": fem_model_6/fem_uniform,
            "ml_rmse_difference_fem_minus_grid": fem_model_6-grid_model,
            "ml_rmse_percent_difference": 100*(fem_model_6/grid_model-1),
            "relative_rmse_difference_fem_minus_grid": fem_model_6/fem_uniform-grid_model/grid_uniform,
            "grid_improves_uniform": grid_model < grid_uniform,
            "fem_improves_uniform": fem_model_6 < fem_uniform,
            "quadrature4_vs6_ml_abs_difference": abs(fem_model_4-fem_model_6),
        }
        rows.append(row)
        if index == 0:
            first_case_timing = time.perf_counter()-stamp
            print(json.dumps({"small_test": control_id, "seconds": first_case_timing,
                              "grid_rmse": grid_model, "fem_rmse": fem_model_6}), flush=True)
        if (index+1) % 10 == 0 or index+1 == len(cases):
            print(json.dumps({"completed": index+1, "total": len(cases)}), flush=True)

    table = pd.DataFrame(rows)
    table_path = output / "rmse_comparison.csv"
    table.to_csv(table_path, index=False)
    squares_table = table[table.experiment.str.startswith("SQ")]
    changed = squares_table[squares_table.grid_improves_uniform != squares_table.fem_improves_uniform]
    by_config = []
    for cfg, group in squares_table.groupby("configuration", sort=True):
        by_config.append({
            "configuration": cfg,
            "grid_improves_count": int(group.grid_improves_uniform.sum()),
            "fem_improves_count": int(group.fem_improves_uniform.sum()),
            "median_grid_rmse": float(group.grid_ml_rmse_m_per_a.median()),
            "median_fem_rmse": float(group.fem_ml_rmse_m_per_a.median()),
            "median_grid_relative_rmse": float(group.grid_relative_rmse.median()),
            "median_fem_relative_rmse": float(group.fem_relative_rmse.median()),
        })
    summary = {
        "schema": "jog-grid-fem-rmse-comparison-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "cases": len(table),
        "observation_roundtrip_rmse_m_per_a": observation_roundtrip_rmse,
        "observation_roundtrip_finite_rows": int(roundtrip_finite.sum()),
        "observation_roundtrip_total_rows": int(len(roundtrip_finite)),
        "observation_roundtrip_finite_fraction": float(roundtrip_finite.mean()),
        "synthetic_zero_rmse": synthetic_zero, "synthetic_3_4_vector_rmse": synthetic_five,
        "maximum_quadrature4_vs6_ml_difference": float(table.quadrature4_vs6_ml_abs_difference.max()),
        "first_case_seconds": first_case_timing, "elapsed_seconds": time.perf_counter()-started,
        "square_interpretation_changes": changed[["experiment","configuration","grid_relative_rmse","fem_relative_rmse"]].to_dict("records"),
        "by_configuration": by_config,
        "maximum_absolute_ml_rmse_difference": float(table.ml_rmse_difference_fem_minus_grid.abs().max()),
        "median_absolute_ml_rmse_percent_difference": float(table.ml_rmse_percent_difference.abs().median()),
        "maximum_absolute_ml_rmse_percent_difference": float(table.ml_rmse_percent_difference.abs().max()),
        "inputs": {
            "corrected_evaluation_manifest_id": read_json(corrected_root/"evaluation_manifest.json")["manifest_id"],
            "adoption_manifest_id": adoption["adoption_manifest_id"],
            "dataset_sha256": sha256_file(dataset_path), "mesh_path": config["domain"]["mesh_path"],
            "mesh_sha256": sha256_file(root/config["domain"]["mesh_path"]),
            "baseline_manifest_id": baseline_manifest["manifest_id"],
        },
        "method": {
            "observed_velocity": "existing Icepack CG2 MEaSUREs velocity fields u_exp_x/u_exp_y on production mesh",
            "coverage": "whole production triangles whose barycentre maps to a frozen common-eligible 450 m observation pixel and whose local CG2 nodes have finite MEaSUREs vx and vy",
            "geography": "central-square bounds or PIG region applied at triangle barycentres",
            "integration": "Firedrake assembly of DG0 cell mask times squared CG2 vector residual, divided by assembled masked area",
            "quadrature_degree": 6,
            "region_overlap_cell_count": region_overlap_cell_count,
            "limitation": "coverage and geographic boundaries are represented by whole triangles selected at barycentres; triangles touched by raster missing-data interpolation are conservatively excluded, so this differs from the exact retained observation-row population",
        },
        "software": {"python": sys.version, "platform": platform.platform(),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "firedrake": getattr(firedrake, "__version__", "unknown")},
        "outputs": {"rmse_comparison.csv": sha256_file(table_path)},
    }
    summary["manifest_id"] = manifest_identifier(summary)
    atomic_json(output / "manifest.json", summary)
    print(json.dumps({k: summary[k] for k in ("status","cases","maximum_absolute_ml_rmse_difference","manifest_id")}, indent=2))


if __name__ == "__main__":
    main()
