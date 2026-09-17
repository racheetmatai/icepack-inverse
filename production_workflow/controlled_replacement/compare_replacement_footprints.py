#!/usr/bin/env python3
"""Compare sector-wide and footprint-only replacement on complete footprints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_controlled_campaign import interpolate_velocity, relative_rmse, rmse, velocity_path
from evaluate_forward_campaign import build_object, build_observation_alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corrected-observations", type=Path, required=True)
    parser.add_argument("--old-model-solves", type=Path, required=True)
    parser.add_argument("--old-baseline-solves", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    object_, _, _ = build_object(args.config, args.repo_root, args.adoption)
    frame, lookup = build_observation_alignment(object_, args.dataset)
    corrected = pd.read_csv(args.corrected_observations)
    aligned = corrected.set_index("row_id").reindex(frame["row_id"].astype(str))
    if corrected["row_id"].duplicated().any() or aligned.isna().any().any():
        raise RuntimeError("Corrected observations do not align uniquely")
    observed = aligned[["observed_vx_raw", "observed_vy_raw"]].to_numpy(np.float64)

    rows = []
    for number in range(1, 11):
        experiment = f"SQ{number:02d}"
        mask = frame["square_footprint_id"].eq(experiment).to_numpy()
        if not mask.any():
            raise RuntimeError(f"Empty footprint: {experiment}")
        old_uniform = interpolate_velocity(
            object_, np.load(velocity_path(args.old_baseline_solves, f"{experiment}_UNIFORM_C"), allow_pickle=False), lookup
        )
        new_uniform = interpolate_velocity(
            object_, np.load(velocity_path(args.new_root, f"CR_{experiment}_UNIFORM"), allow_pickle=False), lookup
        )
        for configuration in ("CFG02", "CFG04"):
            old_model = interpolate_velocity(
                object_, np.load(velocity_path(args.old_model_solves, f"{experiment}_{configuration}_MEDIAN"), allow_pickle=False), lookup
            )
            new_model = interpolate_velocity(
                object_, np.load(velocity_path(args.new_root, f"CR_{experiment}_{configuration}_MEDIAN"), allow_pickle=False), lookup
            )
            for scenario, model, uniform in (
                ("whole_sector_original_measures", old_model, old_uniform),
                ("controlled_original_measures", new_model, new_uniform),
            ):
                model_value = rmse(model, observed, mask)
                uniform_value = rmse(uniform, observed, mask)
                rows.append({
                    "scenario": scenario, "experiment": experiment,
                    "configuration": configuration, "rows": int(mask.sum()),
                    "ml_velocity_rmse_m_per_a": model_value,
                    "uniform_velocity_rmse_m_per_a": uniform_value,
                    "relative_rmse": relative_rmse(model_value, uniform_value),
                    "ml_improves_uniform": bool(model_value < uniform_value),
                })
    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)
    print(table.groupby(["scenario", "configuration"]).agg(
        median_relative_rmse=("relative_rmse", "median"),
        improved_footprints=("ml_improves_uniform", "sum"),
    ).to_string())


if __name__ == "__main__":
    main()
