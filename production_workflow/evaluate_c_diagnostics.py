"""Exact FE-area-weighted agreement with the adopted inversion C field.

The inversion control called C in this project is the logarithmic field used
inside the friction-law exponential.  Velocity agreement remains the primary
scientific test; these C metrics are secondary non-uniqueness diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from export_canonical_dataset import assign_regions
from full_mesh_ensemble_predictions import full_mesh_context
from production_amundsen import load_config, manifest_identifier, resolve_design_path, sha256_file


def masks_for(experiment: str, xy: np.ndarray, regions: np.ndarray, squares: list[dict]) -> dict:
    x, y = xy[:, 0], xy[:, 1]
    if experiment.startswith("SQ"):
        row = next(item for item in squares if item["square_id"] == experiment)
        central = ((x >= float(row["test_xmin_m"])) & (x < float(row["test_xmax_m"])) &
                   (y >= float(row["test_ymin_m"])) & (y < float(row["test_ymax_m"])))
        full = ((x >= float(row["footprint_xmin_m"])) & (x < float(row["footprint_xmax_m"])) &
                (y >= float(row["footprint_ymin_m"])) & (y < float(row["footprint_ymax_m"])))
        return {"central_50km": central, "exclusion_annulus": full & ~central, "full_130km": full}
    if experiment == "REG_INTER":
        return {"both_corridors": np.isin(regions, [4, 5]),
                "pig_thwaites_corridor": regions == 4,
                "thwaites_dotson_corridor": regions == 5}
    if experiment == "REG_PIG":
        return {"PIG": regions == 1}
    raise ValueError(experiment)


def run(args) -> dict:
    import firedrake

    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    object_, _, eligible, coordinates, reference_values, adoption = full_mesh_context(
        args.config.resolve(), args.repo_root.resolve(), args.adoption_record.resolve()
    )
    config = load_config(args.config)
    prediction_set = json.loads((args.prediction_root / "prediction_set_manifest.json").read_text())
    if prediction_set.get("counts", {}).get("ensembles") != 66:
        raise RuntimeError("Prediction set is not the complete 66-ensemble set")

    q_cell_nodes = object_.Q.cell_node_map().values
    completely_replaced_cell = eligible[q_cell_nodes].all(axis=1)
    dg0 = firedrake.FunctionSpace(object_.mesh, "DG", 0)
    vdg0 = firedrake.VectorFunctionSpace(object_.mesh, "DG", 0)
    center_fn = firedrake.interpolate(object_.mesh.coordinates, vdg0)
    cell_xy = np.asarray(center_fn.dat.data_ro[:, :2], dtype=float)
    dg0_cell_nodes = dg0.cell_node_map().values[:, 0]
    if len(dg0_cell_nodes) != len(cell_xy):
        raise RuntimeError("DG0 cell mapping mismatch")

    partition = json.loads(resolve_design_path(config["frozen_design"]["five_region_support"]["path"]).read_text())
    mesh_paths = {name: Path(partition["meshes"][name]["path"]) for name in ("PIG", "Thwaites", "Dotson")}
    regions, _ = assign_regions(cell_xy, mesh_paths)
    with resolve_design_path(config["frozen_design"]["selected_squares"]["path"]).open(newline="") as stream:
        squares = list(csv.DictReader(stream))

    reference = object_.C
    control = firedrake.Function(object_.Q, name="predicted_C")
    indicator = firedrake.Function(dg0, name="evaluation_indicator")
    dx = firedrake.dx(domain=object_.mesh)
    rows = []

    declared = prediction_set["ensemble_manifests"]
    for sequence, item in enumerate(declared, start=1):
        manifest = json.loads((args.prediction_root / item["path"]).read_text())
        ensemble_id = manifest["ensemble_id"]
        experiment = manifest["experiment"]
        configuration = manifest["configuration"]
        npz_path = args.prediction_root / manifest["npz_path"]
        if sha256_file(npz_path) != manifest["npz_sha256"]:
            raise RuntimeError(f"Prediction hash mismatch: {ensemble_id}")
        geographic = masks_for(experiment, cell_xy, regions, squares)
        population_cache = {}
        for population, raw_mask in geographic.items():
            cell_mask = raw_mask & completely_replaced_cell
            indicator.dat.data[:] = 0.0
            indicator.dat.data[dg0_cell_nodes] = cell_mask.astype(float)
            area = float(firedrake.assemble(indicator * dx))
            if area <= 0:
                raise RuntimeError(f"Empty C population: {ensemble_id}/{population}")
            mean = float(firedrake.assemble(indicator * reference * dx) / area)
            variance = float(firedrake.assemble(indicator * (reference - mean) ** 2 * dx) / area)
            population_cache[population] = (indicator.copy(deepcopy=True), area, mean, variance, int(cell_mask.sum()))

        with np.load(npz_path, allow_pickle=False) as archive:
            members = archive["member_log_C"]
            median = archive["median_log_C"]
            controls = [(f"{ensemble_id}_M{i:02d}", "member", values)
                        for i, values in enumerate(members, start=1)]
            controls.append((f"{ensemble_id}_MEDIAN", "median", median))
            for control_id, kind, values in controls:
                control.dat.data[:] = values
                for population, (pop_indicator, area, mean, variance, cell_count) in population_cache.items():
                    difference = control - reference
                    mse = float(firedrake.assemble(pop_indicator * difference ** 2 * dx) / area)
                    bias = float(firedrake.assemble(pop_indicator * difference * dx) / area)
                    r2 = None if variance <= 1e-15 else float(1.0 - mse / variance)
                    rows.append({
                        "control_id": control_id, "ensemble_id": ensemble_id,
                        "control_kind": kind, "experiment": experiment,
                        "configuration": configuration, "population": population,
                        "cells": cell_count, "area_km2": area / 1e6,
                        "C_rmse": float(np.sqrt(max(mse, 0.0))), "C_bias": bias,
                        "C_reference_mean": mean, "C_reference_variance": variance,
                        "C_R2": r2,
                    })
        print(f"C diagnostics {sequence}/66: {ensemble_id}", flush=True)

    table = output / "control_population_c_metrics.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    manifest = {
        "schema": "jog-fe-area-weighted-c-diagnostics-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "secondary diagnostic only; observed-velocity agreement is primary because inversion C may be non-unique",
        "field": "inversion control C (logarithmic friction parameter)",
        "integration": "exact Firedrake finite-element integration over cells whose complete CG2 node set is ML-eligible",
        "ensembles": 66, "controls": 726, "metric_rows": len(rows),
        "prediction_set_manifest_id": prediction_set["manifest_id"],
        "adoption_manifest_id": adoption["adoption_manifest_id"],
        "output_sha256": {table.name: sha256_file(table)},
        "source_sha256": sha256_file(Path(__file__)),
    }
    manifest["manifest_id"] = manifest_identifier(manifest)
    (output / "c_diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adoption-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
