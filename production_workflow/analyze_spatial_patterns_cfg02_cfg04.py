"""Evidence audit for the manuscript's spatial-patterns subsection.

The script does not edit manuscript sources. It joins the finalized forward,
C-diagnostic, support, regional-partition, and footprint-error evidence for
CFG02 and CFG04 and writes a compact machine-readable audit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PW = ROOT / "production_workflow"
DATASET = PW / "gate2_results/gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz"
SUPPORT_ROOT = PW / "gate2_results/gate2_distribution_diagnostics_20260820_c"
FORWARD_ROOT = PW / "gate4_forward_evaluation_20260829_a"
C_ROOT = PW / "gate4_c_diagnostics_20260829_a"
ERROR_ROOT = PW / "gate4_square_footprint_error_maps_20260830_b"
SQUARES = PW / "frozen_design/selected_squares.csv"
OUT = ROOT / "output" / "analysis"
CONFIGS = ("CFG02", "CFG04")
CONFIG_SUPPORT = {
    "CFG02": "CFG02_best_ice",
    "CFG04": "CFG04_best_geophysical",
}
REGIONS = (
    "PIG",
    "Thwaites",
    "Dotson",
    "PIG-Thwaites inter-catchment",
    "Thwaites-Dotson inter-catchment",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_manifest(path: Path, *, passed: bool = False) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in (None, "complete"):
        raise RuntimeError(f"Incomplete manifest: {path}")
    if passed and not payload.get("passed", False):
        raise RuntimeError(f"Verification did not pass: {path}")
    return payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    forward_verify = require_manifest(FORWARD_ROOT / "verification_manifest.json", passed=True)
    c_manifest = require_manifest(C_ROOT / "c_diagnostic_manifest.json")
    support_manifest = require_manifest(SUPPORT_ROOT / "diagnostics_manifest.json")
    error_manifest = require_manifest(ERROR_ROOT / "footprint_error_export_manifest.json")
    error_verify = require_manifest(ERROR_ROOT / "verification_manifest.json", passed=True)

    declared = {
        C_ROOT / "control_population_c_metrics.csv": c_manifest["output_sha256"]["control_population_c_metrics.csv"],
        SUPPORT_ROOT / "support_categories.csv": support_manifest["output_sha256"]["support_categories.csv"],
    }
    for config in CONFIGS:
        path = ERROR_ROOT / f"{config}_ten_square_footprint_errors.npz"
        declared[path] = error_manifest["output_sha256"][path.name]
    for path, expected in declared.items():
        if sha(path) != expected:
            raise RuntimeError(f"Hash mismatch: {path}")

    squares = pd.read_csv(SQUARES)
    square_ids = squares["square_id"].tolist()

    usecols = [
        "common_eligible", "observed_speed", "region_name",
        "square_test_id", "square_footprint_id",
    ]
    canonical = pd.read_csv(DATASET, usecols=usecols, low_memory=False)
    canonical = canonical.loc[canonical["common_eligible"].astype(bool)].copy()
    sector_quantiles = canonical["observed_speed"].quantile([0.5, 0.9, 0.99]).to_dict()
    fast_threshold = float(sector_quantiles[0.9])

    geographic_rows: list[dict] = []
    for square_id in square_ids:
        for population, column in (
            ("central_50km", "square_test_id"),
            ("full_130km", "square_footprint_id"),
        ):
            subset = canonical.loc[canonical[column] == square_id]
            if subset.empty:
                raise RuntimeError(f"Empty canonical population: {square_id} {population}")
            counts = subset["region_name"].value_counts(normalize=True)
            row = {
                "experiment": square_id,
                "population": population,
                "canonical_rows": int(len(subset)),
                "observed_speed_median_m_per_a": float(subset["observed_speed"].median()),
                "observed_speed_q90_m_per_a": float(subset["observed_speed"].quantile(0.9)),
                "observed_speed_q99_m_per_a": float(subset["observed_speed"].quantile(0.99)),
                "fraction_above_sector_speed_q90": float((subset["observed_speed"] >= fast_threshold).mean()),
                "dominant_region": str(counts.index[0]),
                "dominant_region_fraction": float(counts.iloc[0]),
            }
            for region in REGIONS:
                row[f"region_fraction__{region}"] = float(counts.get(region, 0.0))
            geographic_rows.append(row)
    geography = pd.DataFrame(geographic_rows)

    forward = pd.read_csv(FORWARD_ROOT / "control_population_metrics.csv")
    forward = forward.loc[
        (forward["control_kind"] == "median")
        & forward["experiment"].isin(square_ids)
        & forward["configuration"].isin(CONFIGS)
        & forward["population"].isin(["central_50km", "full_130km"])
        & (forward["support_stratum"] == "all")
    ].copy()
    forward["relative_rmse_uniform"] = (
        forward["vector_rmse_m_per_a"] / forward["uniform_vector_rmse_m_per_a"]
    )
    forward = forward[[
        "experiment", "population", "configuration", "rows",
        "observed_vector_rms_m_per_a", "inversion_vector_rmse_m_per_a",
        "uniform_vector_rmse_m_per_a", "vector_mae_m_per_a",
        "vector_rmse_m_per_a", "relative_rmse_uniform",
    ]]
    if len(forward) != 40:
        raise RuntimeError(f"Expected 40 finalized forward rows, found {len(forward)}")

    c_metrics = pd.read_csv(C_ROOT / "control_population_c_metrics.csv")
    c_metrics = c_metrics.loc[
        (c_metrics["control_kind"] == "median")
        & c_metrics["experiment"].isin(square_ids)
        & c_metrics["configuration"].isin(CONFIGS)
        & c_metrics["population"].isin(["central_50km", "full_130km"])
    ][[
        "experiment", "population", "configuration", "cells", "area_km2",
        "C_rmse", "C_bias", "C_reference_variance", "C_R2",
    ]]
    if len(c_metrics) != 40:
        raise RuntimeError(f"Expected 40 finalized C rows, found {len(c_metrics)}")

    support = pd.read_csv(SUPPORT_ROOT / "support_categories.csv")
    support["configuration_short"] = support["configuration"].str.slice(0, 5)
    support = support.loc[
        support["experiment"].isin(square_ids)
        & support["configuration_short"].isin(CONFIGS)
        & support["population"].isin(["central_50km", "full_130km"])
    ][[
        "experiment", "population", "configuration_short",
        "minimum_marginal_coverage", "joint_coverage", "both_fraction",
        "limiting_feature",
    ]].rename(columns={"configuration_short": "configuration"})
    if len(support) != 40:
        raise RuntimeError(f"Expected 40 finalized support rows, found {len(support)}")

    local_rows: list[dict] = []
    for config in CONFIGS:
        with np.load(ERROR_ROOT / f"{config}_ten_square_footprint_errors.npz", allow_pickle=False) as z:
            number = z["square_number"].astype(int)
            central = z["central"].astype(bool)
            speed = z["observed_speed"].astype(float)
            model_sq = z["model_squared_error"].astype(float)
            uniform_sq = z["uniform_squared_error"].astype(float)
        model_error = np.sqrt(model_sq)
        uniform_error = np.sqrt(uniform_sq)
        delta = model_error - uniform_error
        for square_number, square_id in enumerate(square_ids, start=1):
            for population, pop_mask in (
                ("central_50km", central),
                ("full_130km", np.ones_like(central, dtype=bool)),
            ):
                mask = (number == square_number) & pop_mask
                fast = mask & (speed >= fast_threshold)
                nonfast = mask & (speed < fast_threshold)
                def ratio(selection: np.ndarray) -> float:
                    if not np.any(selection):
                        return float("nan")
                    return float(np.sqrt(np.mean(model_sq[selection])) / np.sqrt(np.mean(uniform_sq[selection])))
                local_rows.append({
                    "experiment": square_id,
                    "population": population,
                    "configuration": config,
                    "mapped_rows": int(mask.sum()),
                    "fraction_local_error_below_uniform": float(np.mean(delta[mask] < 0)),
                    "local_error_difference_median_m_per_a": float(np.median(delta[mask])),
                    "local_error_difference_q01_m_per_a": float(np.quantile(delta[mask], 0.01)),
                    "local_error_difference_q99_m_per_a": float(np.quantile(delta[mask], 0.99)),
                    "relative_rmse_uniform_sector_fastest_decile": ratio(fast),
                    "relative_rmse_uniform_below_sector_fastest_decile": ratio(nonfast),
                    "fraction_model_squared_error_in_sector_fastest_decile": (
                        float(np.sum(model_sq[fast]) / np.sum(model_sq[mask])) if np.any(fast) else 0.0
                    ),
                })
    local = pd.DataFrame(local_rows)

    evidence = forward.merge(c_metrics, on=["experiment", "population", "configuration"], validate="one_to_one")
    evidence = evidence.merge(support, on=["experiment", "population", "configuration"], validate="one_to_one")
    evidence = evidence.merge(local, on=["experiment", "population", "configuration"], validate="one_to_one")
    evidence = evidence.merge(geography, on=["experiment", "population"], validate="many_to_one")
    evidence = evidence.sort_values(["population", "experiment", "configuration"]).reset_index(drop=True)

    # Descriptive square-level associations only; n=10 per population/config.
    correlations: dict[str, dict[str, float]] = {}
    for population in ("central_50km", "full_130km"):
        for config in CONFIGS:
            group = evidence.loc[(evidence["population"] == population) & (evidence["configuration"] == config)]
            key = f"{population}__{config}"
            correlations[key] = {
                "spearman_C_rmse_vs_velocity_rmse": float(group["C_rmse"].rank().corr(group["vector_rmse_m_per_a"].rank())),
                "spearman_C_rmse_vs_relative_rmse": float(group["C_rmse"].rank().corr(group["relative_rmse_uniform"].rank())),
                "spearman_observed_rms_vs_velocity_rmse": float(group["observed_vector_rms_m_per_a"].rank().corr(group["vector_rmse_m_per_a"].rank())),
                "spearman_inversion_rmse_vs_model_rmse": float(group["inversion_vector_rmse_m_per_a"].rank().corr(group["vector_rmse_m_per_a"].rank())),
            }

    evidence_path = OUT / "spatial_patterns_cfg02_cfg04_evidence.csv"
    evidence.to_csv(evidence_path, index=False)
    manifest = {
        "schema": "jog-spatial-patterns-cfg02-cfg04-audit-v1",
        "status": "complete",
        "rows": int(len(evidence)),
        "configurations": list(CONFIGS),
        "populations": ["central_50km", "full_130km"],
        "sector_observed_speed_quantiles_m_per_a": {str(k): float(v) for k, v in sector_quantiles.items()},
        "sector_fast_threshold_definition": "observed_speed >= whole-sector q90",
        "correlations": correlations,
        "source_manifest_ids": {
            "forward_verification": forward_verify["manifest_id"],
            "c_diagnostics": c_manifest["manifest_id"],
            "support_diagnostics": support_manifest["manifest_id"],
            "footprint_export": error_manifest["manifest_id"],
            "footprint_verification": error_verify["manifest_id"],
        },
        "evidence_csv": evidence_path.name,
        "evidence_csv_sha256": sha(evidence_path),
        "notes": [
            "Finalized FE-area-weighted RMSE and C metrics are used for manuscript inference.",
            "Regional fractions and speed quantiles use the common eligible 450 m canonical grid and are descriptive.",
            "No area-weighted C MAE exists in the finalized C-diagnostic artifact; C RMSE and bias are used instead.",
        ],
    }
    manifest_path = OUT / "spatial_patterns_cfg02_cfg04_audit.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "rows": len(evidence),
        "evidence": str(evidence_path),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
