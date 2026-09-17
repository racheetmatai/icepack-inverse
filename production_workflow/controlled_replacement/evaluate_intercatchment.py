#!/usr/bin/env python3
"""Evaluate controlled and earlier sector-wide inter-catchment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_controlled_campaign import (
    interpolate_velocity,
    read_json,
    relative_rmse,
    rmse,
    velocity_path,
)
from evaluate_forward_campaign import build_object, build_observation_alignment
from production_amundsen import sha256_file


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
    args.output.mkdir(parents=True, exist_ok=True)

    object_, adoption, _ = build_object(args.config, args.repo_root, args.adoption)
    frame, lookup = build_observation_alignment(object_, args.dataset)
    corrected = pd.read_csv(args.corrected_observations)
    if corrected["row_id"].duplicated().any():
        raise RuntimeError("Corrected observations have duplicated row IDs")
    aligned = corrected.set_index("row_id").reindex(frame["row_id"].astype(str))
    if aligned.isna().any().any():
        raise RuntimeError("Corrected observations do not cover every evaluated row")
    observed = aligned[["observed_vx_raw", "observed_vy_raw"]].to_numpy(np.float64)
    mask = np.isin(frame["region_code"].to_numpy(np.int8), [4, 5])
    if not mask.any() or not np.isfinite(observed[mask]).all():
        raise RuntimeError("Invalid combined-corridor evaluation population")

    inversion = interpolate_velocity(
        object_, np.asarray(adoption["velocity"].dat.data_ro, dtype=np.float64), lookup
    )
    inversion_value = rmse(inversion, observed, mask)
    old_uniform = interpolate_velocity(
        object_, np.load(velocity_path(args.old_baseline_solves, "REG_INTER_UNIFORM_C"), allow_pickle=False), lookup
    )
    new_uniform = interpolate_velocity(
        object_, np.load(velocity_path(args.new_root, "CR_REG_INTER_UNIFORM"), allow_pickle=False), lookup
    )

    rows = []
    for configuration in ("CFG04", "CFG05", "CFG06"):
        old = interpolate_velocity(
            object_,
            np.load(velocity_path(args.old_model_solves, f"REG_INTER_{configuration}_MEDIAN"), allow_pickle=False),
            lookup,
        )
        new = interpolate_velocity(
            object_,
            np.load(velocity_path(args.new_root, f"CR_REG_INTER_{configuration}_MEDIAN"), allow_pickle=False),
            lookup,
        )
        for scenario, predicted, uniform in (
            ("whole_sector_original_measures", old, old_uniform),
            ("controlled_original_measures", new, new_uniform),
        ):
            ml_value = rmse(predicted, observed, mask)
            uniform_value = rmse(uniform, observed, mask)
            rows.append({
                "scenario": scenario,
                "experiment": "REG_INTER",
                "configuration": configuration,
                "rows": int(mask.sum()),
                "ml_velocity_rmse_m_per_a": ml_value,
                "uniform_velocity_rmse_m_per_a": uniform_value,
                "inversion_velocity_rmse_m_per_a": inversion_value,
                "relative_rmse": relative_rmse(ml_value, uniform_value),
                "ml_improves_uniform": bool(ml_value < uniform_value),
            })
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "intercatchment_before_after.csv", index=False)
    summary = {
        "schema": "jog-controlled-intercatchment-evaluation-v1",
        "status": "complete",
        "rows": int(mask.sum()),
        "all_finite": bool(np.isfinite(table[["ml_velocity_rmse_m_per_a", "uniform_velocity_rmse_m_per_a", "inversion_velocity_rmse_m_per_a", "relative_rmse"]]).all().all()),
        "improvement_counts": {
            scenario: int(group["ml_improves_uniform"].sum())
            for scenario, group in table.groupby("scenario")
        },
        "input_sha256": {
            "config": sha256_file(args.config),
            "adoption": sha256_file(args.adoption),
            "dataset": sha256_file(args.dataset),
            "corrected_observations": sha256_file(args.corrected_observations),
        },
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
