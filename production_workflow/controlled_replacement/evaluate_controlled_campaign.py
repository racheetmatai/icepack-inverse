#!/usr/bin/env python3
"""Evaluate observational-reference and replacement-region changes separately."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from evaluate_forward_campaign import build_object, build_observation_alignment, interpolate_velocity
from production_amundsen import manifest_identifier, sha256_file


CONFIGS = [f"CFG{i:02d}" for i in range(1, 7)]
SQUARES = [f"SQ{i:02d}" for i in range(1, 11)]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def interpolate_control(object_, values: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    import firedrake
    import icepack
    function = firedrake.Function(object_.Q, name="evaluated_C")
    function.dat.data[:] = values
    result = np.asarray(icepack.interpolate(function, getattr(object_, "Δ")).dat.data_ro,
                        dtype=np.float64)[lookup]
    if result.shape != (len(lookup),) or not np.isfinite(result).all():
        raise RuntimeError("Observation-mesh control interpolation failed")
    return result


def rmse(predicted: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> float:
    delta = predicted[mask] - observed[mask]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def relative_rmse(numerator: float, denominator: float, tolerance: float = 1.0e-12) -> float:
    """Return the RMSE ratio, or NaN for a numerically negligible baseline."""
    return float("nan") if abs(denominator) <= tolerance else float(numerator / denominator)


def population_mask(frame: pd.DataFrame, experiment: str) -> np.ndarray:
    if experiment.startswith("SQ"):
        mask = frame["square_test_id"].eq(experiment).to_numpy()
    elif experiment == "REG_PIG":
        mask = frame["region_code"].to_numpy(np.int8) == 1
    else:
        raise ValueError(experiment)
    if not mask.any():
        raise RuntimeError(f"Empty population: {experiment}")
    return mask


def velocity_path(root: Path, control_id: str) -> Path:
    path = root / "solves" / control_id / "velocity.npy"
    manifest_path = path.parent / "forward_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or sha256_file(path) != manifest.get("velocity_sha256"):
        raise RuntimeError(f"Unverified velocity: {control_id}")
    return path


def holm_adjust(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["sign_test_p_unadjusted"])
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * rows[index]["sign_test_p_unadjusted"])
        running = max(running, adjusted)
        rows[index]["sign_test_p_holm"] = running


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corrected-observations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--old-model-solves", type=Path, required=True)
    parser.add_argument("--old-baseline-solves", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--accepted-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    object_, adoption, _ = build_object(args.config, args.repo_root, args.adoption)
    frame, lookup = build_observation_alignment(object_, args.dataset)
    stored_observed = frame[["observed_vx", "observed_vy"]].to_numpy(np.float64)
    corrected = pd.read_csv(args.corrected_observations)
    if corrected["row_id"].duplicated().any():
        raise RuntimeError("Corrected observations have duplicated row IDs")
    aligned = corrected.set_index("row_id").reindex(frame["row_id"].astype(str))
    if aligned.isna().any().any():
        raise RuntimeError("Corrected observations do not cover every evaluated row")
    raw_observed = aligned[["observed_vx_raw", "observed_vy_raw"]].to_numpy(np.float64)
    if not np.isfinite(raw_observed).all():
        raise RuntimeError("Corrected observations contain nonfinite values")

    inversion_mesh = np.asarray(adoption["velocity"].dat.data_ro, dtype=np.float64)
    inversion = interpolate_velocity(object_, inversion_mesh, lookup)
    reference_c_mesh = np.asarray(adoption["C"].dat.data_ro, dtype=np.float64)
    reference_c = interpolate_control(object_, reference_c_mesh, lookup)

    old_baselines: dict[str, np.ndarray] = {}
    new_baselines: dict[str, np.ndarray] = {}
    for experiment in SQUARES + ["REG_PIG"]:
        old_id = f"{experiment}_UNIFORM_C"
        new_id = f"CR_{experiment}_UNIFORM"
        old_baselines[experiment] = interpolate_velocity(
            object_, np.load(velocity_path(args.old_baseline_solves, old_id), allow_pickle=False), lookup
        )
        new_baselines[experiment] = interpolate_velocity(
            object_, np.load(velocity_path(args.new_root, new_id), allow_pickle=False), lookup
        )

    rows: list[dict] = []
    correlations: list[dict] = []
    bedmachine_correlations: list[dict] = []
    footprint_comparisons: list[dict] = []
    for experiment in SQUARES + ["REG_PIG"]:
        mask = population_mask(frame, experiment)
        configs = CONFIGS if experiment in SQUARES else ["CFG02"]
        for configuration in configs:
            old_control_id = f"{experiment}_{configuration}_MEDIAN"
            new_control_id = f"CR_{experiment}_{configuration}_MEDIAN"
            old_velocity = interpolate_velocity(
                object_, np.load(velocity_path(args.old_model_solves, old_control_id), allow_pickle=False), lookup
            )
            new_velocity = interpolate_velocity(
                object_, np.load(velocity_path(args.new_root, new_control_id), allow_pickle=False), lookup
            )
            scenarios = [
                ("whole_sector_stored_observation", old_velocity, old_baselines[experiment], stored_observed),
                ("whole_sector_original_measures", old_velocity, old_baselines[experiment], raw_observed),
                ("controlled_original_measures", new_velocity, new_baselines[experiment], raw_observed),
            ]
            for scenario, predicted, baseline, observed in scenarios:
                model_value = rmse(predicted, observed, mask)
                baseline_value = rmse(baseline, observed, mask)
                inversion_value = rmse(inversion, observed, mask)
                rows.append({
                    "scenario": scenario, "experiment": experiment,
                    "configuration": configuration, "rows": int(mask.sum()),
                    "ml_velocity_rmse_m_per_a": model_value,
                    "uniform_velocity_rmse_m_per_a": baseline_value,
                    "inversion_velocity_rmse_m_per_a": inversion_value,
                    "relative_rmse": relative_rmse(model_value, baseline_value),
                    "ml_improves_uniform": bool(model_value < baseline_value),
                })

            prediction_path = args.predictions / f"{experiment}_{configuration}.npz"
            with np.load(prediction_path, allow_pickle=False) as archive:
                median_c_mesh = archive["median_log_C"].astype(np.float64)
            predicted_c = interpolate_control(object_, median_c_mesh, lookup)
            c_error = np.abs(predicted_c - reference_c)
            for scenario, predicted in (
                ("whole_sector_original_measures", old_velocity),
                ("controlled_original_measures", new_velocity),
            ):
                velocity_error = np.linalg.norm(predicted - raw_observed, axis=1)
                correlation = spearmanr(c_error[mask], velocity_error[mask]).statistic
                correlations.append({
                    "scenario": scenario, "experiment": experiment,
                    "configuration": configuration, "rows": int(mask.sum()),
                    "spearman_abs_C_error_vs_local_velocity_error": float(correlation),
                    "sampling": "all retained 450 m observation rows; no spatial averaging",
                })
            local_error = np.linalg.norm(new_velocity - raw_observed, axis=1)
            bed_error = frame["bedmachine_errbed"].to_numpy(np.float64)
            bedmachine_correlations.append({
                "experiment": experiment,
                "configuration": configuration,
                "rows": int(mask.sum()),
                "spearman_bedmachine_errbed_vs_local_velocity_error": float(
                    spearmanr(bed_error[mask], local_error[mask], nan_policy="omit").statistic
                ),
                "scenario": "controlled_original_measures",
            })
            if experiment in SQUARES and configuration in ("CFG02", "CFG04"):
                footprint = frame["square_footprint_id"].eq(experiment).to_numpy()
                for scenario, predicted, baseline in (
                    ("whole_sector_original_measures", old_velocity, old_baselines[experiment]),
                    ("controlled_original_measures", new_velocity, new_baselines[experiment]),
                ):
                    model_value = rmse(predicted, raw_observed, footprint)
                    baseline_value = rmse(baseline, raw_observed, footprint)
                    footprint_comparisons.append({
                        "scenario": scenario, "experiment": experiment,
                        "configuration": configuration, "rows": int(footprint.sum()),
                        "ml_velocity_rmse_m_per_a": model_value,
                        "uniform_velocity_rmse_m_per_a": baseline_value,
                        "relative_rmse": relative_rmse(model_value, baseline_value),
                        "ml_improves_uniform": bool(model_value < baseline_value),
                    })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "velocity_metrics_before_after.csv", index=False)
    correlation_table = pd.DataFrame(correlations)
    correlation_table.to_csv(args.output / "C_velocity_correlations_comparable.csv", index=False)
    pd.DataFrame(bedmachine_correlations).to_csv(
        args.output / "bedmachine_errbed_correlations_controlled.csv", index=False
    )
    pd.DataFrame(footprint_comparisons).to_csv(
        args.output / "footprint_replacement_before_after.csv", index=False
    )

    # Reproduce the current accepted whole-population values before accepting any correction.
    accepted = pd.read_csv(args.accepted_metrics)
    accepted = accepted.loc[
        accepted["support_stratum"].eq("all")
        & accepted["control_kind"].eq("median")
        & (
            (accepted["experiment"].isin(SQUARES) & accepted["population"].eq("central_50km"))
            | (accepted["experiment"].eq("REG_PIG") & accepted["population"].eq("PIG")
               & accepted["configuration"].eq("CFG02"))
        )
    ].copy()
    old = metrics.loc[metrics["scenario"].eq("whole_sector_stored_observation")]
    merged = old.merge(accepted, on=["experiment", "configuration"], how="outer", validate="one_to_one")
    if len(merged) != 61 or merged[["ml_velocity_rmse_m_per_a", "vector_rmse_m_per_a"]].isna().any().any():
        raise RuntimeError("Accepted metric reproduction population is incomplete")
    np.testing.assert_allclose(merged["ml_velocity_rmse_m_per_a"], merged["vector_rmse_m_per_a"],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(merged["uniform_velocity_rmse_m_per_a"], merged["uniform_vector_rmse_m_per_a"],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(merged["inversion_velocity_rmse_m_per_a"], merged["inversion_vector_rmse_m_per_a"],
                               rtol=0, atol=1e-10)

    comparisons = metrics.pivot(index=["experiment", "configuration"], columns="scenario",
                                values=["ml_velocity_rmse_m_per_a", "uniform_velocity_rmse_m_per_a",
                                        "relative_rmse", "ml_improves_uniform"]).reset_index()
    comparisons.columns = ["_".join(item).strip("_") if isinstance(item, tuple) else item
                           for item in comparisons.columns]
    comparisons.to_csv(args.output / "scenario_comparisons.csv", index=False)

    paired_rows: list[dict] = []
    controlled = metrics.loc[metrics["scenario"].eq("controlled_original_measures")
                             & metrics["experiment"].isin(SQUARES)]
    pivot = controlled.pivot(index="experiment", columns="configuration",
                             values="ml_velocity_rmse_m_per_a").loc[SQUARES, CONFIGS]
    for left, right in combinations(CONFIGS, 2):
        delta = pivot[left] - pivot[right]
        nonzero = delta[delta != 0.0]
        p_value = float(binomtest(int((nonzero < 0).sum()), len(nonzero), 0.5,
                                  alternative="two-sided").pvalue) if len(nonzero) else 1.0
        paired_rows.append({
            "configuration_a": left, "configuration_b": right,
            "median_rmse_a_minus_b_m_per_a": float(np.median(delta)),
            "a_lower_rmse_squares": int((delta < 0).sum()),
            "b_lower_rmse_squares": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "sign_test_p_unadjusted": p_value,
        })
    holm_adjust(paired_rows)
    pd.DataFrame(paired_rows).to_csv(args.output / "controlled_configuration_pairwise_tests.csv", index=False)

    square_summary = []
    for scenario in metrics["scenario"].unique():
        subset = metrics.loc[metrics["scenario"].eq(scenario) & metrics["experiment"].isin(SQUARES)]
        for configuration in CONFIGS:
            local = subset.loc[subset["configuration"].eq(configuration)]
            square_summary.append({
                "scenario": scenario, "configuration": configuration,
                "improved_squares": int(local["ml_improves_uniform"].sum()),
                "median_velocity_rmse_m_per_a": float(local["ml_velocity_rmse_m_per_a"].median()),
                "median_relative_rmse": float(local["relative_rmse"].median()),
            })
    pd.DataFrame(square_summary).to_csv(args.output / "square_summary.csv", index=False)

    comparison = metrics.set_index(["scenario", "experiment", "configuration"])
    obs_delta = []
    replacement_delta = []
    flips_obs = []
    flips_replacement = []
    for experiment, configuration in metrics[["experiment", "configuration"]].drop_duplicates().itertuples(index=False):
        a = comparison.loc[("whole_sector_stored_observation", experiment, configuration)]
        b = comparison.loc[("whole_sector_original_measures", experiment, configuration)]
        c = comparison.loc[("controlled_original_measures", experiment, configuration)]
        obs_delta.append(float(b.ml_velocity_rmse_m_per_a - a.ml_velocity_rmse_m_per_a))
        replacement_delta.append(float(c.ml_velocity_rmse_m_per_a - b.ml_velocity_rmse_m_per_a))
        if bool(a.ml_improves_uniform) != bool(b.ml_improves_uniform):
            flips_obs.append(f"{experiment}_{configuration}")
        if bool(b.ml_improves_uniform) != bool(c.ml_improves_uniform):
            flips_replacement.append(f"{experiment}_{configuration}")

    correlation_summary = []
    for (scenario, configuration), group in correlation_table.loc[
        correlation_table["experiment"].isin(SQUARES)
    ].groupby(["scenario", "configuration"]):
        values = group["spearman_abs_C_error_vs_local_velocity_error"].to_numpy(float)
        correlation_summary.append({"scenario": scenario, "configuration": configuration,
                                    "median": float(np.median(values)),
                                    "minimum": float(values.min()), "maximum": float(values.max())})
    pd.DataFrame(correlation_summary).to_csv(args.output / "square_correlation_summary.csv", index=False)

    summary = {
        "schema": "jog-controlled-replacement-evaluation-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_metric_reproduction": {"passed": True, "cases": 61, "absolute_tolerance": 1e-10},
        "observation_correction": {
            "maximum_absolute_ml_rmse_change_m_per_a": float(np.max(np.abs(obs_delta))),
            "median_absolute_ml_rmse_change_m_per_a": float(np.median(np.abs(obs_delta))),
            "improvement_classification_flips": flips_obs,
        },
        "replacement_restriction": {
            "maximum_absolute_ml_rmse_change_m_per_a": float(np.max(np.abs(replacement_delta))),
            "median_absolute_ml_rmse_change_m_per_a": float(np.median(np.abs(replacement_delta))),
            "improvement_classification_flips": flips_replacement,
        },
        "definitions": {
            "velocity_error": "sqrt((vx_model-vx_original_MEaSUREs)^2+(vy_model-vy_original_MEaSUREs)^2)",
            "relative_rmse": "ML RMSE / uniform-C RMSE within the same scenario and population",
            "correlation": "Spearman association between absolute C difference and local vector velocity-error magnitude",
        },
        "input_hashes": {name: sha256_file(path) for name, path in {
            "config": args.config, "adoption": args.adoption, "dataset": args.dataset,
            "corrected_observations": args.corrected_observations,
            "accepted_metrics": args.accepted_metrics,
        }.items()},
    }
    summary["manifest_id"] = manifest_identifier(summary)
    atomic_json(args.output / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
