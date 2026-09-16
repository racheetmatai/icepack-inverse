"""Interpolate saved median controls to the canonical observation grid.

This is interpolation only.  It loads the adopted inversion and the already
saved vertex-wise median CG2 controls; it does not train a model or solve the
ice-flow equations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import icepack
import numpy as np
import pandas as pd

# The inversion module imports SHAP for optional explanatory plots.  It is not
# used while reconstructing the adopted state or interpolating saved fields,
# and the production Firedrake environment intentionally does not include it.
sys.modules.setdefault("shap", types.ModuleType("shap"))

from full_mesh_ensemble_predictions import full_mesh_context
from production_amundsen import sha256_file
from src.data_preprocessing import stable_xy_row_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--canonical-dataset", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--adoption-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    queries = pd.read_csv(args.queries, dtype={"row_id": str})
    required = {"experiment", "configuration", "row_id"}
    if not required.issubset(queries.columns):
        raise ValueError(f"Queries lack columns: {sorted(required - set(queries.columns))}")
    if queries.duplicated(list(required)).any():
        raise ValueError("Duplicate experiment/configuration/row_id query")

    object_, _, eligible, coordinates, reference_c, adoption = full_mesh_context(
        args.config.resolve(), args.repo_root.resolve(), args.adoption_record.resolve()
    )
    obs_xy = np.asarray(object_.Δ.mesh().coordinates.dat.data_ro[:, :2], dtype=np.float64)
    obs_ids = np.asarray(stable_xy_row_ids(obs_xy[:, 0], obs_xy[:, 1]), dtype=str)
    if len(np.unique(obs_ids)) != len(obs_ids):
        raise RuntimeError("Observation-grid row IDs are not unique")
    obs_lookup = pd.Series(np.arange(len(obs_ids), dtype=np.int64), index=obs_ids)

    reference_on_obs = np.asarray(icepack.interpolate(object_.C, object_.Δ).dat.data_ro, dtype=np.float64).copy()
    canonical = pd.read_csv(
        args.canonical_dataset,
        usecols=["row_id", "reference_log_C"],
        dtype={"row_id": str},
    ).set_index("row_id")
    canonical_reference = canonical["reference_log_C"].reindex(obs_ids).to_numpy(float)
    if not np.isfinite(canonical_reference).all():
        raise RuntimeError("Canonical reference C does not cover the observation mesh")
    reference_max_abs = float(np.max(np.abs(reference_on_obs - canonical_reference)))
    if reference_max_abs > 1e-12:
        raise RuntimeError(f"Reference-C interpolation disagrees with canonical export: {reference_max_abs}")

    control = object_.C.copy(deepcopy=True)
    rows: list[pd.DataFrame] = []
    source_files: dict[str, str] = {}
    for (experiment, configuration), group in queries.groupby(["experiment", "configuration"], sort=True):
        ensemble_id = f"{experiment}_{configuration}"
        npz_path = args.prediction_root / f"{ensemble_id}.npz"
        manifest_path = args.prediction_root / f"{ensemble_id}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_file(npz_path) != manifest["npz_sha256"]:
            raise RuntimeError(f"Prediction hash mismatch: {ensemble_id}")
        with np.load(npz_path, allow_pickle=False) as archive:
            if not np.array_equal(archive["coordinates"], coordinates):
                raise RuntimeError(f"CG2 coordinate mismatch: {ensemble_id}")
            if not np.array_equal(archive["eligible_mask"].astype(bool), eligible):
                raise RuntimeError(f"Eligible-mask mismatch: {ensemble_id}")
            if not np.array_equal(archive["reference_log_C"], reference_c):
                raise RuntimeError(f"Reference-control mismatch: {ensemble_id}")
            median = archive["median_log_C"].astype(np.float64)
            if not np.array_equal(median, np.median(archive["member_log_C"], axis=0)):
                raise RuntimeError(f"Saved field is not the exact vertex-wise median: {ensemble_id}")
        control.dat.data[:] = median
        predicted_on_obs = np.asarray(icepack.interpolate(control, object_.Δ).dat.data_ro, dtype=np.float64).copy()

        indices = obs_lookup.reindex(group["row_id"].astype(str)).to_numpy()
        if pd.isna(indices).any():
            raise RuntimeError(f"Unmatched observation-grid row ID: {ensemble_id}")
        indices = indices.astype(np.int64)
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"Repeated observation-grid query index: {ensemble_id}")
        rows.append(pd.DataFrame({
            "experiment": experiment,
            "configuration": configuration,
            "row_id": group["row_id"].astype(str).to_numpy(),
            "C_ref": reference_on_obs[indices],
            "C_ML": predicted_on_obs[indices],
        }))
        source_files[npz_path.name] = sha256_file(npz_path)

    result = pd.concat(rows, ignore_index=True)
    if len(result) != len(queries) or not np.isfinite(result[["C_ref", "C_ML"]].to_numpy()).all():
        raise RuntimeError("Incomplete or nonfinite interpolated C output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, compression="gzip")
    record = {
        "schema": "jog-observation-grid-median-c-export-v1",
        "status": "complete",
        "method": "icepack.interpolate(CG2 control, canonical observation mesh Delta)",
        "target": "dimensionless inversion control C",
        "rows": int(len(result)),
        "cases": int(result.groupby(["experiment", "configuration"]).ngroups),
        "reference_max_abs_difference_vs_canonical": reference_max_abs,
        "adoption_manifest_id": adoption["adoption_manifest_id"],
        "input_sha256": {
            "queries": sha256_file(args.queries),
            "canonical_dataset": sha256_file(args.canonical_dataset),
            **source_files,
        },
        "output_sha256": sha256_file(args.output),
        "source_sha256": sha256_file(Path(__file__)),
    }
    record_path = args.output.with_name("observation_grid_c_export.json")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
