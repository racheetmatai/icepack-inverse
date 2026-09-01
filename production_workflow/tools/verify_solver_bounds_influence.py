"""Independent verifier for the Gate-4 solver/fallback/bounds influence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_id(payload: dict) -> str:
    unsigned = dict(payload); unsigned.pop("manifest_id", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(canonical).hexdigest()


def close(a, b, tolerance=1e-10) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=tolerance, atol=tolerance, equal_nan=True))


def rho(x, y) -> float:
    return float(spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic)


def run(args) -> dict:
    root = args.audit_root.resolve()
    manifest = json.loads((root / "solver_bounds_audit_manifest.json").read_text())
    checks = {
        "schema_status": manifest.get("schema") == "jog-solver-fallback-bounds-influence-audit-v1"
                         and manifest.get("status") == "complete",
        "manifest_id": manifest_id(manifest) == manifest.get("manifest_id"),
        "inventory_hashes": True,
        "forward_retry_lineage": True,
        "retry_metric_recomputation": True,
        "C_range_recomputation": True,
        "spread_recomputation": True,
        "scientific_scope": manifest.get("retry_median_control_count") == 0
                            and manifest.get("retry_member_control_count") == 6,
    }
    declared = manifest.get("output_sha256", {})
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
        and path.name not in {"solver_bounds_audit_manifest.json", "verification_manifest.json"}
    }
    checks["inventory_hashes"] &= actual == set(declared)
    for relative, expected in declared.items():
        path = root / relative
        checks["inventory_hashes"] &= path.is_file() and sha256_file(path) == expected

    retry_ids = sorted(manifest["same_solver_retry_controls"])
    checks["forward_retry_lineage"] &= len(retry_ids) == 6
    for control in retry_ids:
        item = json.loads((args.forward_root / "solves" / control / "forward_manifest.json").read_text())
        policy = item.get("solver_policy", {})
        checks["forward_retry_lineage"] &= (
            item.get("status") == "complete"
            and item.get("control_kind") == "member"
            and item["attempts"][-1].get("snes_max_it") == 100
            and item["attempts"][-1].get("status") == "complete"
            and policy.get("fallback") == "no alternate algorithm and no silent substitution"
        )

    metrics = pd.read_csv(args.evaluation_root / "control_population_metrics.csv")
    influence = pd.read_csv(root / "retry_member_influence.csv")
    for _, row in influence.iterrows():
        local = metrics.loc[
            metrics["ensemble_id"].eq(row["ensemble_id"])
            & metrics["population"].eq(row["primary_population"])
            & metrics["support_stratum"].eq("all")
            & metrics["control_kind"].eq("member")
        ]
        kept = local.loc[~local["control_id"].eq(row["control_id"])]
        checks["retry_metric_recomputation"] &= len(local) == 10 and len(kept) == 9
        checks["retry_metric_recomputation"] &= close(
            kept["vector_rmse_m_per_a"].mean() - local["vector_rmse_m_per_a"].mean(),
            row["rmse_mean_change_9_minus_10_m_per_a"])
        checks["retry_metric_recomputation"] &= close(
            kept["P_exp_percent"].mean() - local["P_exp_percent"].mean(),
            row["P_exp_mean_change_9_minus_10_percentage_points"])

    c_table = pd.read_csv(root / "c_range_audit.csv").set_index("ensemble_id")
    checks["C_range_recomputation"] &= len(c_table) == 66
    for path in sorted(args.prediction_root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            eligible = archive["eligible_mask"].astype(bool)
            reference = archive["reference_log_C"].astype(float)[eligible]
            members = archive["member_log_C"].astype(float)[:, eligible]
            median = archive["median_log_C"].astype(float)[eligible]
        row = c_table.loc[path.stem]
        values = {
            "reference_C_minimum": np.min(reference), "reference_C_maximum": np.max(reference),
            "member_C_minimum": np.min(members), "member_C_maximum": np.max(members),
            "median_C_minimum": np.min(median), "median_C_maximum": np.max(median),
        }
        checks["C_range_recomputation"] &= all(close(row[key], value) for key, value in values.items())
        checks["C_range_recomputation"] &= int(row["member_values_below_reference_range"]) == int(
            np.sum(members < np.min(reference)))
        checks["C_range_recomputation"] &= int(row["member_values_above_reference_range"]) == int(
            np.sum(members > np.max(reference)))

    spread_table = pd.read_csv(root / "spread_retry_sensitivity.csv").set_index("ensemble_id")
    for path in sorted((root / "spread_sensitivity_arrays").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            spread10 = archive["spread10"].astype(float)
            spread9 = archive["spread9"].astype(float)
            error = archive["error"].astype(float)
        row = spread_table.loc[path.stem]
        checks["spread_recomputation"] &= len(spread10) == len(spread9) == len(error)
        checks["spread_recomputation"] &= close(row["spread_rms_10_m_per_a"], np.sqrt(np.mean(spread10**2)))
        checks["spread_recomputation"] &= close(row["spread_rms_9_m_per_a"], np.sqrt(np.mean(spread9**2)))
        checks["spread_recomputation"] &= close(row["pointwise_spearman_10"], rho(spread10, error))
        checks["spread_recomputation"] &= close(row["pointwise_spearman_9"], rho(spread9, error))
    original = pd.read_csv(args.spread_root / "ensemble_spread_population_metrics.csv")
    primary = original.loc[original["population"].eq("central_50km")].copy()
    replacements = spread_table["spread_rms_9_m_per_a"].to_dict()
    sensitivity = [replacements.get(e, x) for e, x in zip(
        primary["ensemble_id"], primary["velocity_spread_rms_m_per_a"])]
    summary = manifest["spread_sensitivity"]
    checks["spread_recomputation"] &= close(
        rho(primary["velocity_spread_rms_m_per_a"], primary["median_velocity_error_rmse_m_per_a"]),
        summary["population_spearman_all_60_original_10_members"])
    checks["spread_recomputation"] &= close(
        rho(sensitivity, primary["median_velocity_error_rmse_m_per_a"]),
        summary["population_spearman_all_60_with_affected_cases_at_9_members"])

    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "schema": "jog-solver-fallback-bounds-influence-verification-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "parent_manifest_id": manifest["manifest_id"],
        "retry_controls": len(retry_ids),
        "C_ensembles": len(c_table),
        "spread_sensitivity_ensembles": len(spread_table),
    }
    result["manifest_id"] = manifest_id(result)
    (root / "verification_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--forward-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--spread-root", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
