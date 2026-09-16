"""Relate training-representation percentiles to direct errors in control C."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parent
REP_LABELS = ["<=50", "50-95", ">95"]
REP_DISPLAY = ["<=50th", "50th--95th", ">95th"]
SPEED_LABELS = ["<100", "100-500", "500-1000", ">=1000"]
CONFIG_LABELS = {"CFG02": "CFG02 (selected ice)", "CFG04": "CFG04 (selected geophysical)"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_id(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "manifest_id"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def rho(x: pd.Series, y: pd.Series) -> float:
    mask = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    return float(spearmanr(x.to_numpy(float)[mask], y.to_numpy(float)[mask]).statistic) if mask.sum() >= 3 else np.nan


def summaries(points: pd.DataFrame, group_columns: list[str], fraction_within: list[str]) -> pd.DataFrame:
    totals = points.groupby(fraction_within, observed=True).size().rename("total_rows")
    rows = []
    for keys, group in points.groupby(group_columns, sort=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        denominator_key = tuple(row[name] for name in fraction_within)
        if len(denominator_key) == 1:
            denominator_key = denominator_key[0]
        row.update({
            "rows": int(len(group)),
            "area_fraction": float(len(group) / totals.loc[denominator_key]),
            "C_rmse": float(np.sqrt(np.mean(group["C_squared_difference"]))),
            "C_bias": float(np.mean(group["C_signed_difference"])),
            "C_mae": float(np.mean(group["C_absolute_difference"])),
            "velocity_rmse_m_per_a": float(np.sqrt(np.mean(group["ml_error"] ** 2))),
            "uniform_C_velocity_rmse_m_per_a": float(np.sqrt(np.mean(group["uniform_error"] ** 2))),
        })
        denominator = row["uniform_C_velocity_rmse_m_per_a"]
        row["velocity_rmse_ratio"] = np.nan if denominator <= 1e-15 else row["velocity_rmse_m_per_a"] / denominator
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(category: pd.DataFrame) -> tuple[Path, Path]:
    square_colors = dict(zip([f"SQ{i:02d}" for i in range(1, 11)], plt.get_cmap("tab10").colors))
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), gridspec_kw={"width_ratios": [1, 1, 0.72]})
    for ax, config in zip(axes[:2], ["CFG02", "CFG04"]):
        subset = category.loc[category.configuration.eq(config) & category.experiment.str.startswith("SQ")]
        pivot = subset.pivot(index="experiment", columns="representation_category", values="C_rmse").reindex(columns=REP_LABELS)
        for square, values in pivot.iterrows():
            ax.plot(x, values, color=square_colors[square], alpha=0.72, linewidth=1.2, marker="o", markersize=3.5)
        med = pivot.median(axis=0)
        ax.plot(x, med, color="black", linewidth=2.7, marker="o", markersize=5, label="median across squares")
        ax.set_title(CONFIG_LABELS[config], fontsize=12)
        ax.set_xticks(x, REP_DISPLAY)
        ax.set_xlabel("Training-representation percentile category")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", frameon=True, fontsize=8)
    pig = category.loc[category.experiment.eq("REG_PIG") & category.configuration.eq("CFG02")].set_index("representation_category").reindex(REP_LABELS)
    axes[2].plot(x, pig.C_rmse, color="#5E3C99", linewidth=2.4, marker="o")
    axes[2].set_title("PIG: CFG02", fontsize=12)
    axes[2].set_xticks(x, REP_DISPLAY)
    axes[2].set_xlabel("Training-representation\npercentile category")
    axes[2].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_ylabel("C RMSE (dimensionless)")
        ax.tick_params(labelsize=9)
    handles = [plt.Line2D([], [], color=square_colors[square], marker="o", linewidth=1.4, markersize=4, label=square)
               for square in square_colors]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.40, -0.03), frameon=True, fontsize=8)
    fig.suptitle("Direct inversion-control error versus training representation", fontsize=14, y=1.01)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    png = HERE / "figure_C_C_rmse_by_representation.png"
    pdf = HERE / "figure_C_C_rmse_by_representation.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    point_path = PARENT / "point_diagnostics.csv.gz"
    c_path = HERE / "observation_grid_c.csv.gz"
    parent_manifest_path = PARENT / "manifest.json"
    export_manifest_path = HERE / "observation_grid_c_export.json"
    fe_path = ROOT / "gate4_c_diagnostics_20260829_a" / "control_population_c_metrics.csv"

    points = pd.read_csv(point_path, dtype={"row_id": str})
    controls = pd.read_csv(c_path, dtype={"row_id": str})
    keys = ["experiment", "configuration", "row_id"]
    if points.duplicated(keys).any() or controls.duplicated(keys).any():
        raise RuntimeError("Diagnostic keys are not unique")
    merged = points.merge(controls, on=keys, validate="one_to_one", how="left", indicator=True)
    if not merged["_merge"].eq("both").all() or len(merged) != len(points):
        raise RuntimeError("C and representation populations do not align exactly")
    merged = merged.drop(columns="_merge")
    if not np.isfinite(merged[["C_ref", "C_ML"]].to_numpy()).all():
        raise RuntimeError("Nonfinite C values after stable-row-ID alignment")
    merged["C_signed_difference"] = merged["C_ML"] - merged["C_ref"]
    merged["C_absolute_difference"] = merged["C_signed_difference"].abs()
    merged["C_squared_difference"] = merged["C_signed_difference"] ** 2

    category = summaries(
        merged,
        ["experiment", "configuration", "representation_category"],
        ["experiment", "configuration"],
    )
    category["representation_category"] = pd.Categorical(category["representation_category"], REP_LABELS, ordered=True)
    category = category.sort_values(["experiment", "configuration", "representation_category"])
    category.to_csv(HERE / "c_category_metrics.csv", index=False)

    speed = summaries(
        merged,
        ["experiment", "configuration", "speed_class", "representation_category"],
        ["experiment", "configuration", "speed_class"],
    )
    speed.to_csv(HERE / "c_speed_stratified_metrics.csv", index=False)

    support = summaries(
        merged,
        ["experiment", "configuration", "both_support", "representation_category"],
        ["experiment", "configuration", "both_support"],
    )
    support.to_csv(HERE / "c_support_stratified_metrics.csv", index=False)

    association_rows = []
    for (experiment, configuration), group in merged.groupby(["experiment", "configuration"], sort=True):
        association_rows.append({
            "experiment": experiment,
            "configuration": configuration,
            "speed_class": "all",
            "rows": len(group),
            "spearman_representation_vs_absolute_C_error": rho(group.representation_percentile, group.C_absolute_difference),
            "spearman_representation_vs_signed_C_error": rho(group.representation_percentile, group.C_signed_difference),
        })
        for speed_class in SPEED_LABELS:
            subset = group.loc[group.speed_class.eq(speed_class)]
            association_rows.append({
                "experiment": experiment,
                "configuration": configuration,
                "speed_class": speed_class,
                "rows": len(subset),
                "spearman_representation_vs_absolute_C_error": rho(subset.representation_percentile, subset.C_absolute_difference),
                "spearman_representation_vs_signed_C_error": rho(subset.representation_percentile, subset.C_signed_difference),
            })
    associations = pd.DataFrame(association_rows)
    associations.to_csv(HERE / "c_associations.csv", index=False)

    c_velocity_rows = []
    for (experiment, configuration), group in merged.groupby(["experiment", "configuration"], sort=True):
        for speed_class, subset in [("all", group)] + [(label, group.loc[group.speed_class.eq(label)]) for label in SPEED_LABELS]:
            c_velocity_rows.append({
                "experiment": experiment,
                "configuration": configuration,
                "speed_class": speed_class,
                "rows": len(subset),
                "spearman_absolute_C_error_vs_ML_velocity_error": rho(subset.C_absolute_difference, subset.ml_error),
                "spearman_absolute_C_error_vs_ML_minus_uniform_local_error": rho(subset.C_absolute_difference, subset.ml_minus_uniform_error),
            })
    pd.DataFrame(c_velocity_rows).to_csv(HERE / "c_velocity_associations.csv", index=False)

    speed_overall = summaries(
        merged,
        ["experiment", "configuration", "speed_class"],
        ["experiment", "configuration", "speed_class"],
    )
    speed_overall.to_csv(HERE / "c_speed_overall_metrics.csv", index=False)

    overall = summaries(merged, ["experiment", "configuration"], ["experiment", "configuration"])
    overall.to_csv(HERE / "c_overall_metrics.csv", index=False)
    fe = pd.read_csv(fe_path)
    fe = fe.loc[fe.control_kind.eq("median") & fe.configuration.isin(["CFG02", "CFG04"])].copy()
    fe["analysis_population"] = np.where(fe.experiment.str.startswith("SQ"), "central_50km", "PIG")
    fe = fe.loc[fe.population.eq(fe.analysis_population), ["experiment", "configuration", "C_rmse", "C_bias"]]
    checks = overall.merge(fe, on=["experiment", "configuration"], suffixes=("_observation_grid", "_exact_FE"), validate="one_to_one")
    checks["C_rmse_relative_difference"] = (checks.C_rmse_observation_grid - checks.C_rmse_exact_FE).abs() / checks.C_rmse_exact_FE
    checks["C_bias_absolute_difference"] = (checks.C_bias_observation_grid - checks.C_bias_exact_FE).abs()
    checks.to_csv(HERE / "c_authoritative_metric_checks.csv", index=False)
    if len(checks) != 21 or checks.C_rmse_relative_difference.max() > 0.08:
        raise RuntimeError("Observation-grid C diagnostics disagree materially with exact FE-area diagnostics")

    population_area_error = category.groupby(["experiment", "configuration"], observed=True).area_fraction.sum().sub(1).abs().max()
    speed_area_error = speed.groupby(["experiment", "configuration", "speed_class"], observed=True).area_fraction.sum().sub(1).abs().max()
    if population_area_error > 1e-12 or speed_area_error > 1e-12:
        raise RuntimeError("C diagnostic category fractions do not sum to one")

    png, pdf = make_figure(category)
    merged.to_csv(HERE / "point_c_diagnostics.csv.gz", index=False, compression="gzip")

    verification = {
        "status": "passed",
        "stable_row_id_one_to_one_join": True,
        "unchanged_rows": int(len(merged)),
        "cases": int(merged.groupby(["experiment", "configuration"]).ngroups),
        "category_area_fraction_max_abs_error": float(population_area_error),
        "speed_category_area_fraction_max_abs_error": float(speed_area_error),
        "maximum_C_rmse_relative_difference_vs_exact_FE": float(checks.C_rmse_relative_difference.max()),
        "comparison_note": "Observation-grid equal-area means and exact FE cell integration use different discretizations; agreement is checked within 8%.",
        "reference_interpolation_max_abs_difference_vs_canonical": json.loads(export_manifest_path.read_text())["reference_max_abs_difference_vs_canonical"],
        "target_units": "dimensionless inversion control C",
        "median_control_identity": "exact saved vertex-wise median, then icepack.interpolate to the canonical observation mesh",
    }
    (HERE / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")

    output_names = [
        "point_c_diagnostics.csv.gz", "c_overall_metrics.csv", "c_category_metrics.csv", "c_speed_stratified_metrics.csv",
        "c_support_stratified_metrics.csv", "c_associations.csv", "c_velocity_associations.csv",
        "c_speed_overall_metrics.csv", "c_authoritative_metric_checks.csv",
        png.name, pdf.name, "verification.json", "observation_grid_c.csv.gz", "observation_grid_c_export.json",
        "README.md", "LATEX_SUGGESTIONS.md",
    ]
    parent_manifest = json.loads(parent_manifest_path.read_text())
    export_manifest = json.loads(export_manifest_path.read_text())
    manifest = {
        "schema": "jog-training-representation-c-diagnostic-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "descriptive association between actual training representation and reproduction of inversion-reference C",
        "parent_representation_manifest_id": parent_manifest["manifest_id"],
        "observation_grid_c_export": export_manifest,
        "input_sha256": {
            "point_diagnostics": sha256(point_path),
            "parent_manifest": sha256(parent_manifest_path),
            "exact_FE_C_metrics": sha256(fe_path),
        },
        "population": "unchanged rows from completed representation analysis",
        "weights": "equal-area 450 m observation-grid rows",
        "target": "dimensionless inversion control C; not exp(C) or dimensional basal-friction coefficient",
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "output_sha256": {name: sha256(HERE / name) for name in output_names},
        "source_sha256": {
            Path(__file__).name: sha256(Path(__file__)),
            "export_observation_grid_controls.py": sha256(HERE / "export_observation_grid_controls.py"),
        },
    }
    manifest["manifest_id"] = manifest_id(manifest)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "rows": len(merged), "manifest_id": manifest["manifest_id"]}, indent=2))


if __name__ == "__main__":
    main()
