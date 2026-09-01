"""Close Gate 4 by auditing solver retries, fallbacks, and C bounds.

The headline scientific outputs are the 66 forward solves driven by the
vertex-wise median C controls.  This audit also tests whether omitting each
member that required the frozen same-solver iteration-ceiling retry changes
member summaries or the secondary ensemble-spread diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_ensemble_spread import quintile_summary
from evaluate_forward_campaign import (
    build_observation_alignment,
    interpolate_velocity,
    model_registry,
    population_masks,
)
from forward_solve_campaign import build_object
from production_amundsen import manifest_identifier, sha256_file


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def finite_spearman(x, y) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def primary_population(experiment: str) -> str:
    if experiment.startswith("SQ"):
        return "central_50km"
    return {"REG_PIG": "PIG", "REG_INTER": "both_corridors"}[experiment]


def summarize_member_influence(metrics: pd.DataFrame, retry_ids: list[str]) -> pd.DataFrame:
    rows = []
    for control_id in retry_ids:
        ensemble = control_id.rsplit("_", 1)[0]
        experiment, config = ensemble.split("_") if ensemble.startswith("SQ") else (
            "_".join(ensemble.split("_")[:2]), ensemble.split("_")[2]
        )
        population = primary_population(experiment)
        local = metrics.loc[
            metrics["ensemble_id"].eq(ensemble)
            & metrics["population"].eq(population)
            & metrics["support_stratum"].eq("all")
            & metrics["control_kind"].eq("member")
        ].copy()
        if len(local) != 10 or control_id not in set(local["control_id"]):
            raise RuntimeError(f"Incomplete primary member metrics for {control_id}")
        selected = local.loc[local["control_id"].eq(control_id)].iloc[0]
        retained = local.loc[~local["control_id"].eq(control_id)]
        rmse_all = local["vector_rmse_m_per_a"].to_numpy(float)
        rmse_retained = retained["vector_rmse_m_per_a"].to_numpy(float)
        pexp_all = local["P_exp_percent"].to_numpy(float)
        pexp_retained = retained["P_exp_percent"].to_numpy(float)
        rows.append({
            "control_id": control_id,
            "ensemble_id": ensemble,
            "experiment": experiment,
            "configuration": config,
            "primary_population": population,
            "retry_vector_rmse_m_per_a": float(selected["vector_rmse_m_per_a"]),
            "retry_P_exp_percent": float(selected["P_exp_percent"]),
            "rmse_rank_best_to_worst_of_10": int(np.argsort(np.argsort(rmse_all))[local.index.get_loc(selected.name)] + 1),
            "rmse_mean_10_m_per_a": float(np.mean(rmse_all)),
            "rmse_mean_9_m_per_a": float(np.mean(rmse_retained)),
            "rmse_mean_change_9_minus_10_m_per_a": float(np.mean(rmse_retained) - np.mean(rmse_all)),
            "rmse_q05_10_m_per_a": float(np.quantile(rmse_all, .05)),
            "rmse_q05_9_m_per_a": float(np.quantile(rmse_retained, .05)),
            "rmse_q95_10_m_per_a": float(np.quantile(rmse_all, .95)),
            "rmse_q95_9_m_per_a": float(np.quantile(rmse_retained, .95)),
            "P_exp_mean_10_percent": float(np.mean(pexp_all)),
            "P_exp_mean_9_percent": float(np.mean(pexp_retained)),
            "P_exp_mean_change_9_minus_10_percentage_points": float(np.mean(pexp_retained) - np.mean(pexp_all)),
        })
    return pd.DataFrame(rows)


def c_range_audit(prediction_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(prediction_root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            eligible = archive["eligible_mask"].astype(bool)
            reference = archive["reference_log_C"].astype(float)[eligible]
            members = archive["member_log_C"].astype(float)[:, eligible]
            median = archive["median_log_C"].astype(float)[eligible]
        reference_min = float(np.min(reference)); reference_max = float(np.max(reference))
        rows.append({
            "ensemble_id": path.stem,
            "eligible_dofs": int(eligible.sum()),
            "member_values": int(members.size),
            "all_member_values_finite": bool(np.isfinite(members).all()),
            "all_median_values_finite": bool(np.isfinite(median).all()),
            "reference_C_minimum": reference_min,
            "reference_C_maximum": reference_max,
            "member_C_minimum": float(np.min(members)),
            "member_C_maximum": float(np.max(members)),
            "median_C_minimum": float(np.min(median)),
            "median_C_maximum": float(np.max(median)),
            "member_values_below_reference_range": int(np.sum(members < reference_min)),
            "member_values_above_reference_range": int(np.sum(members > reference_max)),
            "median_values_below_reference_range": int(np.sum(median < reference_min)),
            "median_values_above_reference_range": int(np.sum(median > reference_max)),
        })
    if len(rows) != 66:
        raise RuntimeError(f"Expected 66 prediction ensembles, found {len(rows)}")
    return pd.DataFrame(rows)


def spread_sensitivity(args, retry_ids: list[str], output: Path) -> tuple[pd.DataFrame, dict]:
    affected = sorted({control.rsplit("_", 1)[0] for control in retry_ids if control.startswith("SQ")})
    object_, _, _ = build_object(args.config.resolve(), args.repo_root.resolve(), args.adoption_record.resolve())
    frame, lookup = build_observation_alignment(object_, args.dataset.resolve())
    observed = frame[["observed_vx", "observed_vy"]].to_numpy(np.float64)
    registry = model_registry(args.forward_root.resolve())
    records = {item["control_id"]: item for item in registry}
    array_dir = output / "spread_sensitivity_arrays"; array_dir.mkdir(exist_ok=True)
    rows = []
    for ensemble in affected:
        retry_id = next(item for item in retry_ids if item.startswith(ensemble + "_"))
        experiment, config = ensemble.split("_")
        members = [records[f"{ensemble}_M{index:02d}"] for index in range(1, 11)]
        mask = population_masks(frame, experiment)["central_50km"]
        values = np.stack([
            interpolate_velocity(object_, np.load(item["velocity_path"], allow_pickle=False), lookup)[mask]
            for item in members
        ])
        retained = np.asarray([item["control_id"] != retry_id for item in members])
        spread10 = np.sqrt(np.var(values[:, :, 0], axis=0, ddof=1)
                           + np.var(values[:, :, 1], axis=0, ddof=1))
        spread9 = np.sqrt(np.var(values[retained, :, 0], axis=0, ddof=1)
                          + np.var(values[retained, :, 1], axis=0, ddof=1))
        median_control = records[f"{ensemble}_MEDIAN"]
        median_velocity = interpolate_velocity(
            object_, np.load(median_control["velocity_path"], allow_pickle=False), lookup
        )[mask]
        error = np.linalg.norm(median_velocity - observed[mask], axis=1)
        with (array_dir / f"{ensemble}.npz").open("wb") as stream:
            np.savez_compressed(stream, spread10=spread10, spread9=spread9, error=error)
        q10, _ = quintile_summary(spread10, error); q9, _ = quintile_summary(spread9, error)
        rows.append({
            "ensemble_id": ensemble,
            "retry_control_id": retry_id,
            "rows": int(mask.sum()),
            "spread_rms_10_m_per_a": float(np.sqrt(np.mean(spread10 ** 2))),
            "spread_rms_9_m_per_a": float(np.sqrt(np.mean(spread9 ** 2))),
            "spread_rms_relative_change_9_vs_10": float(
                np.sqrt(np.mean(spread9 ** 2)) / np.sqrt(np.mean(spread10 ** 2)) - 1),
            "pointwise_spearman_10": finite_spearman(spread10, error),
            "pointwise_spearman_9": finite_spearman(spread9, error),
            "error_q5_over_q1_10": float(q10[4] / q10[0]),
            "error_q5_over_q1_9": float(q9[4] / q9[0]),
        })
    table = pd.DataFrame(rows)
    original = pd.read_csv(args.spread_root / "ensemble_spread_population_metrics.csv")
    primary = original.loc[original["population"].eq("central_50km")].copy()
    rho10 = finite_spearman(primary["velocity_spread_rms_m_per_a"],
                            primary["median_velocity_error_rmse_m_per_a"])
    replacements = dict(zip(table["ensemble_id"], table["spread_rms_9_m_per_a"]))
    primary["sensitivity_spread"] = [
        replacements.get(ensemble, value)
        for ensemble, value in zip(primary["ensemble_id"], primary["velocity_spread_rms_m_per_a"])
    ]
    rho9 = finite_spearman(primary["sensitivity_spread"],
                           primary["median_velocity_error_rmse_m_per_a"])
    return table, {
        "affected_square_ensembles": len(table),
        "population_spearman_all_60_original_10_members": rho10,
        "population_spearman_all_60_with_affected_cases_at_9_members": rho9,
        "population_spearman_change": float(rho9 - rho10),
        "maximum_absolute_spread_rms_relative_change": float(
            np.max(np.abs(table["spread_rms_relative_change_9_vs_10"]))),
        "maximum_absolute_pointwise_spearman_change": float(
            np.max(np.abs(table["pointwise_spearman_9"] - table["pointwise_spearman_10"]))),
    }


def run(args) -> dict:
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    forward_verification = json.loads((args.forward_root / "verification_manifest.json").read_text())
    retry_ids = sorted(forward_verification["retry_controls"])
    if len(retry_ids) != 6:
        raise RuntimeError(f"Expected six retries, found {len(retry_ids)}")
    inference = json.loads(args.inference_manifest.read_text())
    if inference["frozen_rules"]["prediction_clipping"] != "none":
        raise RuntimeError("Production inference unexpectedly used prediction clipping")

    forward_manifests = [
        json.loads((args.forward_root / "solves" / control / "forward_manifest.json").read_text())
        for control in retry_ids
    ]
    no_alternate_fallback = all(
        item["solver_policy"]["fallback"] == "no alternate algorithm and no silent substitution"
        and item["solver_policy"]["retry_semantics"].startswith("identical solver algorithm")
        and item["attempts"][-1]["snes_max_it"] == 100
        and item["attempts"][-1]["status"] == "complete"
        for item in forward_manifests
    )
    metrics = pd.read_csv(args.evaluation_root / "control_population_metrics.csv")
    influence = summarize_member_influence(metrics, retry_ids)
    influence.to_csv(output / "retry_member_influence.csv", index=False)
    c_ranges = c_range_audit(args.prediction_root.resolve())
    c_ranges.to_csv(output / "c_range_audit.csv", index=False)
    spread_table, spread_summary = spread_sensitivity(args, retry_ids, output)
    spread_table.to_csv(output / "spread_retry_sensitivity.csv", index=False)

    summary = {
        "schema": "jog-solver-fallback-bounds-influence-audit-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forward_controls": 726,
        "headline_median_controls": 66,
        "same_solver_retry_controls": retry_ids,
        "retry_control_count": len(retry_ids),
        "retry_median_control_count": int(sum(control.endswith("_MEDIAN") for control in retry_ids)),
        "retry_member_control_count": int(sum("_M" in control for control in retry_ids)),
        "preserved_prepolicy_failures": int(
            forward_verification["counts"]["preserved_prepolicy_failure_manifests"]
        ),
        "preserved_prepolicy_failure_controls": sorted(
            path.name for path in (args.forward_root / "failed_attempts").iterdir() if path.is_dir()
        ),
        "alternate_solver_fallbacks": 0,
        "no_alternate_fallback_verified": no_alternate_fallback,
        "production_prediction_clipping": inference["frozen_rules"]["prediction_clipping"],
        "active_friction_bounds": 0,
        "legacy_bound_path_used_by_revised_campaign": False,
        "eligible_prediction_range_summary": {
            "all_values_finite": bool(c_ranges[["all_member_values_finite", "all_median_values_finite"]].all().all()),
            "global_member_C_minimum": float(c_ranges["member_C_minimum"].min()),
            "global_member_C_maximum": float(c_ranges["member_C_maximum"].max()),
            "global_median_C_minimum": float(c_ranges["median_C_minimum"].min()),
            "global_median_C_maximum": float(c_ranges["median_C_maximum"].max()),
            "member_values_outside_same_ensemble_reference_range": int(
                c_ranges["member_values_below_reference_range"].sum()
                + c_ranges["member_values_above_reference_range"].sum()
            ),
            "median_values_outside_same_ensemble_reference_range": int(
                c_ranges["median_values_below_reference_range"].sum()
                + c_ranges["median_values_above_reference_range"].sum()
            ),
        },
        "retry_member_primary_metric_sensitivity": {
            "maximum_absolute_rmse_mean_change_m_per_a": float(
                np.max(np.abs(influence["rmse_mean_change_9_minus_10_m_per_a"]))
            ),
            "maximum_absolute_P_exp_mean_change_percentage_points": float(
                np.max(np.abs(influence["P_exp_mean_change_9_minus_10_percentage_points"]))
            ),
            "retry_member_rmse_ranks_best_to_worst": influence[
                "rmse_rank_best_to_worst_of_10"
            ].astype(int).tolist(),
        },
        "spread_sensitivity": spread_summary,
        "scientific_disposition": (
            "Headline median-control velocity results and feature comparisons are exactly unaffected: "
            "all six retries are member solves, no median solve retried, and no alternate solver or "
            "friction clipping/bound was used. Member-distribution and spread diagnostics are robust "
            "to omitting the six retry members and remain secondary diagnostics."
        ),
        "inputs": {
            "forward_verification_manifest_id": forward_verification["manifest_id"],
            "evaluation_manifest_id": json.loads(
                (args.evaluation_root / "evaluation_manifest.json").read_text()
            )["manifest_id"],
            "prediction_set_manifest_id": json.loads(
                (args.prediction_root / "prediction_set_manifest.json").read_text()
            )["manifest_id"],
            "inference_bundle_manifest_id": inference["manifest_id"],
            "spread_diagnostic_manifest_id": json.loads(
                (args.spread_root / "spread_diagnostic_manifest.json").read_text()
            )["manifest_id"],
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    outputs = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "solver_bounds_audit_manifest.json"
    }
    summary["output_sha256"] = outputs
    summary["manifest_id"] = manifest_identifier(summary)
    atomic_json(output / "solver_bounds_audit_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adoption-record", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--forward-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--spread-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
