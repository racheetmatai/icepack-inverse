#!/usr/bin/env python3
"""Summarize corrected controlled-replacement map fields."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
MAPS = ROOT / "map_fields"
DATASET = WORKSPACE / "production_workflow/gate2_results/gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz"
SUPPORT = WORKSPACE / "production_workflow/gate2_results/gate2_distribution_diagnostics_20260820_c/point_support_categories.npz"
REPRESENTATION = WORKSPACE / "production_workflow/training_representation_diagnostic_20260909_a/point_diagnostics.csv.gz"

# Matches the manuscript's PIG spatial-concentration statement: the fraction
# of PIG area where inversion-reference velocity error is at least this
# value, and the fraction of CFG02 squared error contained within it.
INVERSION_CONTOUR_LEVEL_M_PER_A = 100.0


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(float)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def row_summary(experiment: str, configuration: str, population: str,
                model: np.ndarray, uniform: np.ndarray, inversion: np.ndarray,
                c_difference: np.ndarray) -> dict:
    return {
        "experiment": experiment,
        "configuration": configuration,
        "population": population,
        "rows": int(len(model)),
        "ml_rmse_m_per_a": rmse(model),
        "uniform_rmse_m_per_a": rmse(uniform),
        "inversion_rmse_m_per_a": rmse(inversion),
        "relative_rmse": rmse(model) / rmse(uniform),
        "fraction_ml_local_error_lower": float(np.mean(model < uniform)),
        "median_ml_minus_uniform_error_m_per_a": float(np.median(model - uniform)),
        "fraction_ml_error_ge_100": float(np.mean(model >= 100.0)),
        "fraction_inversion_error_ge_100": float(np.mean(inversion >= 100.0)),
        "C_rmse": rmse(c_difference),
        "spearman_abs_C_error_vs_local_velocity_error": spearman(
            np.abs(c_difference), model
        ),
    }


rows = []
for configuration in ("CFG02", "CFG04"):
    with np.load(MAPS / f"{configuration}_ten_square_controlled_fields.npz", allow_pickle=False) as data:
        for number in range(1, 11):
            footprint = data["square_number"] == number
            central = footprint & data["central_square"].astype(bool)
            for population, mask in (("central_50km", central), ("complete_130km", footprint)):
                rows.append(row_summary(
                    f"SQ{number:02d}", configuration, population,
                    data["model_error"][mask], data["uniform_error"][mask],
                    data["inversion_error"][mask], data["C_difference"][mask],
                ))

summary = pd.DataFrame(rows)
summary.to_csv(ROOT / "corrected_spatial_summary.csv", index=False)

# Recompute the velocity portion of the existing training-representation
# diagnostic without repeating any neighbor searches.
representation = pd.read_csv(
    REPRESENTATION,
    usecols=["experiment", "configuration", "row_id", "representation_category"],
    low_memory=False,
)
representation_rows = []
for configuration in ("CFG02", "CFG04"):
    with np.load(MAPS / f"{configuration}_ten_square_controlled_fields.npz", allow_pickle=True) as data:
        central = data["central_square"].astype(bool)
        local = pd.DataFrame({
            "row_id": data["row_id"][central].astype(str),
            "experiment": np.array([f"SQ{number:02d}" for number in data["square_number"][central]], dtype=str),
            "configuration": configuration,
            "model_error": data["model_error"][central].astype(float),
            "uniform_error": data["uniform_error"][central].astype(float),
        })
    joined = representation.loc[representation["configuration"].eq(configuration)].merge(
        local, on=["row_id", "experiment", "configuration"], how="inner", validate="one_to_one"
    )
    if len(joined) != len(local):
        raise RuntimeError(f"Representation rows do not cover corrected central squares: {configuration}")
    for (experiment, category), group in joined.groupby(["experiment", "representation_category"], sort=True):
        representation_rows.append({
            "experiment": experiment,
            "configuration": configuration,
            "representation_category": category,
            "rows": int(len(group)),
            "ml_rmse_m_per_a": rmse(group["model_error"].to_numpy(float)),
            "uniform_rmse_m_per_a": rmse(group["uniform_error"].to_numpy(float)),
            "relative_rmse": rmse(group["model_error"].to_numpy(float)) / rmse(group["uniform_error"].to_numpy(float)),
        })
pd.DataFrame(representation_rows).to_csv(
    ROOT / "corrected_representation_velocity_categories.csv", index=False
)

with np.load(MAPS / "REG_PIG_CFG02_controlled_fields.npz", allow_pickle=True) as data:
    pig = {name: data[name] for name in data.files}

pig_overall = row_summary(
    "REG_PIG", "CFG02", "PIG", pig["model_error"], pig["uniform_error"],
    pig["inversion_error"], pig["C_difference"],
)

speed_rows = []
classes = [
    ("lt_100", 0.0, 100.0),
    ("100_500", 100.0, 500.0),
    ("500_1000", 500.0, 1000.0),
    ("ge_1000", 1000.0, np.inf),
]
for label, lower, upper in classes:
    mask = (pig["observed_speed"] >= lower) & (pig["observed_speed"] < upper)
    speed_rows.append({
        "speed_class": label,
        "rows": int(mask.sum()),
        "ml_rmse_m_per_a": rmse(pig["model_error"][mask]),
        "uniform_rmse_m_per_a": rmse(pig["uniform_error"][mask]),
        "relative_rmse": rmse(pig["model_error"][mask]) / rmse(pig["uniform_error"][mask]),
        "fraction_ml_local_error_lower": float(np.mean(pig["model_error"][mask] < pig["uniform_error"][mask])),
    })
pd.DataFrame(speed_rows).to_csv(ROOT / "corrected_pig_speed_classes.csv", index=False)

threshold = np.quantile(pig["observed_speed"], 0.9)
fast = pig["observed_speed"] >= threshold
reduction = np.square(pig["uniform_error"]) - np.square(pig["model_error"])
positive_total = reduction.sum()

# Stable-row-ID support alignment, reusing the corrected support-category mapping.
canonical = pd.read_csv(DATASET, usecols=["row_id", "common_eligible"], low_memory=False)
eligible_ids = canonical.loc[canonical["common_eligible"].astype(bool), "row_id"].astype(str).reset_index(drop=True)
with np.load(SUPPORT, allow_pickle=False) as support:
    index = support["REG_PIG__row_index"].astype(np.int64)
    category = support["REG_PIG__CFG02_best_ice"].astype(np.uint8)
lookup = pd.Series(category, index=eligible_ids.iloc[index].to_numpy(str))
aligned_support = lookup.reindex(pd.Index(pig["row_id"].astype(str))).to_numpy()
if np.any(pd.isna(aligned_support)):
    raise RuntimeError("PIG support categories did not align completely")
both_supported = aligned_support.astype(np.uint8) == 3

pig_details = {
    "overall": pig_overall,
    "fastest_10_percent_speed_threshold_m_per_a": float(threshold),
    "fastest_10_percent_fraction_total_squared_error_reduction": float(reduction[fast].sum() / positive_total),
    "fastest_10_percent_fraction_remaining_ml_squared_error": float(np.square(pig["model_error"])[fast].sum() / np.square(pig["model_error"]).sum()),
    "fastest_10_percent_ml_rmse_m_per_a": rmse(pig["model_error"][fast]),
    "fastest_10_percent_uniform_rmse_m_per_a": rmse(pig["uniform_error"][fast]),
    "both_supported_area_fraction": float(np.mean(both_supported)),
    "supported_fraction_total_squared_error_reduction": float(reduction[both_supported].sum() / positive_total),
    "high_inversion_error_area_fraction": float(np.mean(pig["inversion_error"] >= INVERSION_CONTOUR_LEVEL_M_PER_A)),
    "high_inversion_error_fraction_of_ml_squared_error": float(
        np.sum(np.square(pig["model_error"])[pig["inversion_error"] >= INVERSION_CONTOUR_LEVEL_M_PER_A])
        / np.sum(np.square(pig["model_error"]))
    ),
}
(ROOT / "corrected_pig_details.json").write_text(
    json.dumps(pig_details, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

print(json.dumps(pig_details, indent=2, sort_keys=True))
