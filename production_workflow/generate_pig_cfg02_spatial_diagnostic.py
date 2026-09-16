"""Generate the PIG CFG02 diagnostic panels and their evidence audit.

The manuscript is not modified. Each panel is exported as a separate vector
PDF so LaTeX can place the subfigure letter and title below while preserving
the panel's native aspect ratio.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import generate_revision_figures_and_tables as base


ROOT = Path(os.environ.get("JOG_REPOSITORY_ROOT", Path(__file__).resolve().parent.parent))
ARTIFACT_ROOT = Path(os.environ.get("JOG_ARTIFACT_ROOT", ROOT))
RUNS = ARTIFACT_ROOT / "production_runs"
PRED = RUNS / "gate3_full_mesh_ensemble_predictions_20260828_a"
EVAL = RUNS / "gate4_forward_evaluation_support_aligned_20260910"
CDIAG = RUNS / "gate4_c_diagnostics_20260829_a"
UNIFORM = RUNS / "gate4_uniform_c_baselines_20260829_a"
SPLITS = RUNS / "gate2_split_manifests_20260820_a"
SUPPORT = RUNS / "gate2_distribution_diagnostics_20260820_c"
DATASET = RUNS / "gate2_canonical_dataset_20260820_c"
DESIGN = ARTIFACT_ROOT / "production_workflow/frozen_design"
OUT = Path(os.environ.get("JOG_OUTPUT_ROOT", ROOT / "output/pdf"))
ANALYSIS = OUT / "analysis"
PANEL_PDFS = {
    "C": OUT / "figure6a_pig_cfg02_control_difference.pdf",
    "absolute": OUT / "figure6b_pig_cfg02_velocity_error.pdf",
    "difference": OUT / "figure6c_pig_cfg02_uniform_comparison.pdf",
}
PREVIEW = OUT / "figure6_pig_cfg02_spatial_diagnostic_preview.png"
DISPLAY_GRID_M = 1800.0
C_LIMIT = 2.10  # matches Figure 5 to rounding
INVERSION_CONTOUR_LEVEL = 100.0
INVERSION_CONTOUR_COLOR = "#29D8E6"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str) -> None:
    observed = sha(path)
    if observed != expected:
        raise RuntimeError(f"Hash mismatch for {path}: {observed} != {expected}")


def unpack(mask: np.ndarray, count: int, bitorder: str) -> np.ndarray:
    return np.unpackbits(mask, bitorder=bitorder)[:count].astype(bool)


def verify_inputs() -> dict:
    prediction_manifest = read_json(PRED / "REG_PIG_CFG02.json")
    prediction_set = read_json(PRED / "prediction_set_manifest.json")
    evaluation = read_json(EVAL / "evaluation_manifest.json")
    cdiag = read_json(CDIAG / "c_diagnostic_manifest.json")
    split_manifest = read_json(SPLITS / "split_bundle_manifest.json")
    support_manifest = read_json(SUPPORT / "diagnostics_manifest.json")
    dataset_manifest = read_json(DATASET / "dataset_manifest.json")
    baseline = read_json(UNIFORM / "baseline_campaign_manifest.json")
    baseline_forward = read_json(UNIFORM / "solves/REG_PIG_UNIFORM_C/forward_manifest.json")
    model_forward = read_json(
        RUNS / "gate4_forward_solve_campaign_20260828_a/solves/REG_PIG_CFG02_MEDIAN/forward_manifest.json"
    )

    if any(item.get("status") != "complete" for item in (prediction_manifest, evaluation, cdiag, support_manifest, dataset_manifest, model_forward)):
        raise RuntimeError("At least one required finalized manifest is incomplete")
    if baseline.get("status") != "controls_complete" or baseline_forward.get("status") != "complete":
        raise RuntimeError("Uniform-C baseline is incomplete")
    if prediction_manifest["member_job_ids"] != [f"REG_PIG_CFG02_M{i:02d}" for i in range(1, 11)]:
        raise RuntimeError("CFG02 PIG ensemble is not the declared ten-member ensemble")

    prediction_npz = PRED / prediction_manifest["npz_path"]
    require_hash(prediction_npz, prediction_manifest["npz_sha256"])
    map_npz = EVAL / "median_map_data/REG_PIG_CFG02_MEDIAN.npz"
    require_hash(map_npz, evaluation["output_sha256"]["median_map_data/REG_PIG_CFG02_MEDIAN.npz"])
    require_hash(EVAL / "control_population_metrics.csv", evaluation["output_sha256"]["control_population_metrics.csv"])
    require_hash(CDIAG / "control_population_c_metrics.csv", cdiag["output_sha256"]["control_population_c_metrics.csv"])
    require_hash(SUPPORT / "point_support_categories.npz", support_manifest["output_sha256"]["point_support_categories.npz"])
    require_hash(SUPPORT / "support_categories.csv", support_manifest["output_sha256"]["support_categories.csv"])
    require_hash(DATASET / "canonical_master_dataset.csv.gz", dataset_manifest["output_sha256"]["canonical_master_dataset.csv.gz"])
    require_hash(SPLITS / "population_masks/REG_PIG.npz", split_manifest["output_sha256"]["population_masks/REG_PIG.npz"])
    require_hash(SPLITS / "member_splits/REG_PIG.npz", split_manifest["output_sha256"]["member_splits/REG_PIG.npz"])
    require_hash(UNIFORM / "solves/REG_PIG_UNIFORM_C/velocity.npy", baseline_forward["velocity_sha256"])
    require_hash(
        RUNS / "gate4_forward_solve_campaign_20260828_a/solves/REG_PIG_CFG02_MEDIAN/velocity.npy",
        model_forward["velocity_sha256"],
    )

    with np.load(SPLITS / "population_masks/REG_PIG.npz", allow_pickle=False) as pop, np.load(
        SPLITS / "member_splits/REG_PIG.npz", allow_pickle=False
    ) as members:
        count = int(pop["row_count"])
        bitorder = str(pop["bitorder"])
        development = unpack(pop["development"], count, bitorder)
        heldout = unpack(pop["heldout"], count, bitorder)
        train = np.vstack([unpack(row, count, bitorder) for row in members["train"]])
        validation = np.vstack([unpack(row, count, bitorder) for row in members["validation"]])
    if np.any(development & heldout) or not np.all(development | heldout):
        raise RuntimeError("PIG development/held-out populations are not a partition")
    if np.any(train[:, heldout]) or np.any(validation[:, heldout]) or np.any(train & validation):
        raise RuntimeError("PIG rows entered training or validation")

    with np.load(prediction_npz, allow_pickle=False) as prediction, np.load(
        UNIFORM / "controls/REG_PIG_UNIFORM_C.npz", allow_pickle=False
    ) as control:
        for key, control_key in (("coordinates", "coordinates"), ("eligible_mask", "eligible_mask"), ("reference_log_C", "reference_C")):
            if not np.array_equal(prediction[key], control[control_key]):
                raise RuntimeError(f"Prediction/uniform control identity mismatch: {key}")
        eligible = prediction["eligible_mask"].astype(bool)
        if not np.allclose(control["control_C"][eligible], float(control["uniform_C"]), rtol=0, atol=0):
            raise RuntimeError("Uniform C was not applied exactly on the replacement mask")
        if not np.array_equal(control["control_C"][~eligible], control["reference_C"][~eligible]):
            raise RuntimeError("Uniform baseline changed C outside the replacement mask")
        median_recomputed = np.median(prediction["member_log_C"], axis=0)
        if not np.array_equal(median_recomputed, prediction["median_log_C"]):
            raise RuntimeError("Stored CFG02 control is not the exact vertex-wise member median")

    return {
        "prediction_manifest_id": prediction_manifest["manifest_id"],
        "prediction_set_manifest_id": prediction_set["manifest_id"],
        "evaluation_manifest_id": evaluation["manifest_id"],
        "c_diagnostic_manifest_id": cdiag["manifest_id"],
        "support_manifest_id": support_manifest["manifest_id"],
        "split_manifest_id": split_manifest["manifest_id"],
        "baseline_manifest_id": baseline["manifest_id"],
        "model_forward_manifest_id": model_forward["manifest_id"],
        "uniform_forward_manifest_id": baseline_forward["manifest_id"],
        "heldout_rows": int(heldout.sum()),
        "development_rows": int(development.sum()),
        "member_count": int(train.shape[0]),
        "uniform_C": float(baseline_forward["uniform_C"]),
        "replacement_mask_and_retained_reference_verified": True,
    }


def aligned_support(row_ids: np.ndarray) -> np.ndarray:
    raw = pd.read_csv(
        DATASET / "canonical_master_dataset.csv.gz",
        usecols=["row_id", "common_eligible"],
        low_memory=False,
    )
    eligible_ids = raw.loc[raw["common_eligible"].astype(bool), "row_id"].astype(str).reset_index(drop=True)
    with np.load(SUPPORT / "point_support_categories.npz", allow_pickle=False) as support:
        index = support["REG_PIG__row_index"].astype(np.int64)
        category = support["REG_PIG__CFG02_best_ice"].astype(np.uint8)
    lookup = pd.Series(category, index=eligible_ids.iloc[index].to_numpy(str))
    aligned = lookup.reindex(pd.Index(row_ids.astype(str))).to_numpy()
    if np.any(pd.isna(aligned)):
        raise RuntimeError("Could not realign PIG support categories by stable row_id")
    return aligned.astype(np.uint8)


def grid_average(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    work = frame.copy()
    work["ix"] = np.floor(work["x_m"] / DISPLAY_GRID_M).astype(np.int64)
    work["iy"] = np.floor(work["y_m"] / DISPLAY_GRID_M).astype(np.int64)
    grouped = work.groupby(["ix", "iy"], sort=True, observed=True)[columns + ["x_m", "y_m"]].mean().reset_index()
    ix = np.arange(grouped["ix"].min(), grouped["ix"].max() + 1)
    iy = np.arange(grouped["iy"].min(), grouped["iy"].max() + 1)
    arrays = {name: np.full((len(iy), len(ix)), np.nan) for name in columns}
    xi = np.searchsorted(ix, grouped["ix"])
    yi = np.searchsorted(iy, grouped["iy"])
    for name in columns:
        arrays[name][yi, xi] = grouped[name].to_numpy(float)
    x_centers = (ix + 0.5) * DISPLAY_GRID_M / 1000.0
    y_centers = (iy + 0.5) * DISPLAY_GRID_M / 1000.0
    x_edges = np.r_[x_centers - DISPLAY_GRID_M / 2000.0, x_centers[-1] + DISPLAY_GRID_M / 2000.0]
    y_edges = np.r_[y_centers - DISPLAY_GRID_M / 2000.0, y_centers[-1] + DISPLAY_GRID_M / 2000.0]
    return x_edges, y_edges, arrays, x_centers, y_centers


def native_grid(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Place the native 450 m evaluation rows on their exact regular grid."""
    x_centers_m = np.sort(frame["x_m"].unique().astype(float))
    y_centers_m = np.sort(frame["y_m"].unique().astype(float))
    if not np.allclose(np.diff(x_centers_m), 450.0) or not np.allclose(np.diff(y_centers_m), 450.0):
        raise RuntimeError("PIG evaluation coordinates are not on the expected native 450 m grid")
    xi = np.searchsorted(x_centers_m, frame["x_m"].to_numpy(float))
    yi = np.searchsorted(y_centers_m, frame["y_m"].to_numpy(float))
    linear = yi * len(x_centers_m) + xi
    if len(np.unique(linear)) != len(frame):
        raise RuntimeError("Duplicate native PIG evaluation coordinates")
    arrays = {name: np.full((len(y_centers_m), len(x_centers_m)), np.nan) for name in columns}
    for name in columns:
        arrays[name][yi, xi] = frame[name].to_numpy(float)
    return (
        base.center_edges(x_centers_m / 1000.0),
        base.center_edges(y_centers_m / 1000.0),
        arrays,
    )


def load_evidence() -> tuple[dict, pd.DataFrame, dict, dict]:
    with np.load(EVAL / "median_map_data/REG_PIG_CFG02_MEDIAN.npz", allow_pickle=False) as data:
        model_error = data["error_magnitude"].astype(float)
        uniform_sq = data["signed_local_squared_error_improvement"].astype(float) + model_error**2
        if np.min(uniform_sq) < -1e-7:
            raise RuntimeError("Cannot reconstruct the uniform-C error magnitude")
        frame = pd.DataFrame({
            "row_id": data["row_id"].astype(str),
            "x_m": data["x"].astype(float),
            "y_m": data["y"].astype(float),
            "observed_vx": data["observed_vx"].astype(float),
            "observed_vy": data["observed_vy"].astype(float),
            "model_error": model_error,
            "uniform_error": np.sqrt(np.maximum(uniform_sq, 0.0)),
        })
    frame["observed_speed"] = np.hypot(frame["observed_vx"], frame["observed_vy"])
    canonical = pd.read_csv(
        DATASET / "canonical_master_dataset.csv.gz",
        usecols=["row_id", "x", "y", "observed_vx", "observed_vy", "inversion_vx", "inversion_vy"],
        low_memory=False,
    )
    if canonical["row_id"].duplicated().any():
        raise RuntimeError("Canonical row IDs are not unique")
    canonical["inversion_error"] = np.hypot(
        canonical["inversion_vx"] - canonical["observed_vx"],
        canonical["inversion_vy"] - canonical["observed_vy"],
    )
    frame = frame.merge(
        canonical[["row_id", "x", "y", "inversion_error"]],
        on="row_id", how="left", validate="one_to_one", indicator=True,
    )
    if not frame["_merge"].eq("both").all() or not np.isfinite(frame["inversion_error"]).all():
        raise RuntimeError("PIG rows do not align completely with inversion-reference residuals")
    if not np.allclose(frame["x_m"], frame["x"], rtol=0, atol=1e-8) or not np.allclose(frame["y_m"], frame["y"], rtol=0, atol=1e-8):
        raise RuntimeError("PIG row-ID join disagrees with saved coordinates")
    frame = frame.drop(columns=["x", "y", "_merge"])
    frame["error_difference"] = frame["model_error"] - frame["uniform_error"]
    frame["support_category"] = aligned_support(frame["row_id"].to_numpy(str))
    frame["both_supported"] = frame["support_category"].eq(3)

    forward = pd.read_csv(EVAL / "control_population_metrics.csv")
    forward = forward.loc[
        forward["experiment"].eq("REG_PIG")
        & forward["population"].eq("PIG")
        & forward["support_stratum"].eq("all")
        & forward["control_kind"].eq("median")
    ].set_index("configuration")
    expected = {"CFG01": 353.67515829489923, "CFG02": 355.66983489100187, "CFG03": 385.15101060572016}
    for config, value in expected.items():
        if not np.isclose(float(forward.loc[config, "vector_rmse_m_per_a"]), value, rtol=0, atol=1e-9):
            raise RuntimeError(f"Unexpected finalized PIG RMSE for {config}")
    if not np.isclose(np.sqrt(np.mean(frame["model_error"] ** 2)), expected["CFG02"], atol=1e-9):
        raise RuntimeError("CFG02 map data do not reproduce the finalized RMSE")
    if not np.isclose(np.sqrt(np.mean(frame["uniform_error"] ** 2)), 386.70821593286325, atol=1e-9):
        raise RuntimeError("Reconstructed uniform map does not reproduce the finalized RMSE")

    cmetrics = pd.read_csv(CDIAG / "control_population_c_metrics.csv")
    cmetrics = cmetrics.loc[
        cmetrics["experiment"].eq("REG_PIG")
        & cmetrics["population"].eq("PIG")
        & cmetrics["control_kind"].eq("median")
    ].set_index("configuration")

    support_summary = pd.read_csv(SUPPORT / "support_categories.csv")
    support_summary = support_summary.loc[
        support_summary["experiment"].eq("REG_PIG")
        & support_summary["population"].eq("PIG")
        & support_summary["configuration"].eq("CFG02_best_ice")
    ].iloc[0].to_dict()
    if not np.isclose(frame["both_supported"].mean(), float(support_summary["both_fraction"]), atol=1e-12):
        raise RuntimeError("Correctly aligned point support does not reproduce aggregate support")

    with np.load(PRED / "REG_PIG_CFG02.npz", allow_pickle=False) as prediction:
        coords = prediction["coordinates"].astype(float)
        eligible = prediction["eligible_mask"].astype(bool)
        c_difference = prediction["median_log_C"].astype(float) - prediction["reference_log_C"].astype(float)

    region = np.load(DESIGN / "five_region_partition_5km.npz", allow_pickle=False)
    ix = np.rint((coords[:, 0] - region["x_grid"][0]) / 5000.0).astype(int)
    iy = np.rint((coords[:, 1] - region["y_grid"][0]) / 5000.0).astype(int)
    valid = (ix >= 0) & (ix < len(region["x_grid"])) & (iy >= 0) & (iy < len(region["y_grid"]))
    use = valid & eligible
    c_sum = np.zeros(region["region_codes"].shape, dtype=float)
    c_count = np.zeros(region["region_codes"].shape, dtype=np.int32)
    np.add.at(c_sum, (iy[use], ix[use]), c_difference[use])
    np.add.at(c_count, (iy[use], ix[use]), 1)
    c_grid = np.divide(c_sum, c_count, out=np.full_like(c_sum, np.nan), where=c_count > 0)
    c_grid[region["region_codes"] != 1] = np.nan

    row_ix = np.rint((frame["x_m"].to_numpy(float) - region["x_grid"][0]) / 5000.0).astype(int)
    row_iy = np.rint((frame["y_m"].to_numpy(float) - region["y_grid"][0]) / 5000.0).astype(int)
    row_valid = (row_ix >= 0) & (row_ix < len(region["x_grid"])) & (row_iy >= 0) & (row_iy < len(region["y_grid"]))
    row_c = np.full(len(frame), np.nan)
    row_c[row_valid] = c_grid[row_iy[row_valid], row_ix[row_valid]]
    # Boundary rows whose nearest 5 km reporting cell has no eligible control
    # vertex receive the value of the nearest actual eligible CG2 control DOF.
    # This is used only for descriptive row-wise association, never to fill the
    # plotted C field.
    missing = ~np.isfinite(row_c)
    source_xy = coords[eligible] / 1000.0
    source_c = c_difference[eligible]
    query_xy = frame.loc[missing, ["x_m", "y_m"]].to_numpy(float) / 1000.0
    nearest_values = np.empty(len(query_xy), dtype=float)
    for start in range(0, len(query_xy), 256):
        query = query_xy[start : start + 256]
        distance = np.sum((query[:, None, :] - source_xy[None, :, :]) ** 2, axis=2)
        nearest_values[start : start + len(query)] = source_c[np.argmin(distance, axis=1)]
    row_c[missing] = nearest_values
    frame["C_difference"] = row_c
    if not np.isfinite(frame["C_difference"]).all():
        raise RuntimeError("PIG C diagnostic association remains incomplete")

    pig_use = use.copy()
    pig_use[valid] &= region["region_codes"][iy[valid], ix[valid]] == 1
    c_xy = coords[pig_use] / 1000.0
    c_values = c_difference[pig_use]
    c_tri = mtri.Triangulation(c_xy[:, 0], c_xy[:, 1])
    triangle_points = c_xy[c_tri.triangles]
    max_edge = np.maximum.reduce([
        np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 1], axis=1),
        np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 2], axis=1),
        np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 0], axis=1),
    ])
    centroid = triangle_points.mean(axis=1) * 1000.0
    centroid_ix = np.rint((centroid[:, 0] - region["x_grid"][0]) / 5000.0).astype(int)
    centroid_iy = np.rint((centroid[:, 1] - region["y_grid"][0]) / 5000.0).astype(int)
    centroid_valid = (
        (centroid_ix >= 0) & (centroid_ix < len(region["x_grid"]))
        & (centroid_iy >= 0) & (centroid_iy < len(region["y_grid"]))
    )
    centroid_in_pig = np.zeros(len(centroid), dtype=bool)
    centroid_in_pig[centroid_valid] = (
        region["region_codes"][centroid_iy[centroid_valid], centroid_ix[centroid_valid]] == 1
    )
    # Native CG2 spacing is approximately 3--5 km. Mask Delaunay bridges
    # across concave boundaries rather than inventing continuity there.
    c_tri.set_mask((max_edge > 8.0) | ~centroid_in_pig)

    cplot = {
        "x_edges": base.center_edges(region["x_grid"] / 1000.0),
        "y_edges": base.center_edges(region["y_grid"] / 1000.0),
        "difference": c_grid,
        "triangulation": c_tri,
        "difference_native": c_values,
    }

    return cplot, frame, forward.to_dict(orient="index"), {
        "cmetrics": cmetrics.to_dict(orient="index"),
        "support": support_summary,
    }


def group_record(name: str, subset: pd.DataFrame, full: pd.DataFrame) -> dict:
    model_sq = subset["model_error"].to_numpy(float) ** 2
    uniform_sq = subset["uniform_error"].to_numpy(float) ** 2
    return {
        "group": name,
        "rows": int(len(subset)),
        "area_fraction": float(len(subset) / len(full)),
        "observed_speed_median_m_per_a": float(subset["observed_speed"].median()),
        "model_rmse_m_per_a": float(np.sqrt(model_sq.mean())),
        "uniform_rmse_m_per_a": float(np.sqrt(uniform_sq.mean())),
        "relative_rmse": float(np.sqrt(model_sq.mean() / uniform_sq.mean())),
        "fraction_local_error_below_uniform": float((subset["error_difference"] < 0).mean()),
        "fraction_of_model_squared_error": float(model_sq.sum() / np.sum(full["model_error"].to_numpy(float) ** 2)),
        "net_squared_error_reduction_fraction": float(
            np.sum(uniform_sq - model_sq)
            / np.sum(full["uniform_error"].to_numpy(float) ** 2 - full["model_error"].to_numpy(float) ** 2)
        ),
        "abs_C_difference_median": float(subset["C_difference"].abs().median()),
    }


def spatial_audit(frame: pd.DataFrame, forward: dict, extra: dict, provenance: dict) -> tuple[pd.DataFrame, dict]:
    _, _, gradient_grids, x_centers, y_centers = display_grids(frame)
    log_speed = np.log1p(gradient_grids["observed_speed"])
    gx = np.full_like(log_speed, np.nan)
    gy = np.full_like(log_speed, np.nan)
    gx[:, 1:-1] = (log_speed[:, 2:] - log_speed[:, :-2]) / (2 * DISPLAY_GRID_M / 1000.0)
    gy[1:-1, :] = (log_speed[2:, :] - log_speed[:-2, :]) / (2 * DISPLAY_GRID_M / 1000.0)
    gradient = np.hypot(gx, gy)
    first_ix = int(round(x_centers[0] * 1000.0 / DISPLAY_GRID_M - 0.5))
    first_iy = int(round(y_centers[0] * 1000.0 / DISPLAY_GRID_M - 0.5))
    row_ix = np.floor(frame["x_m"].to_numpy(float) / DISPLAY_GRID_M).astype(int) - first_ix
    row_iy = np.floor(frame["y_m"].to_numpy(float) / DISPLAY_GRID_M).astype(int) - first_iy
    row_gradient = np.full(len(frame), np.nan)
    inside = (row_ix >= 0) & (row_ix < gradient.shape[1]) & (row_iy >= 0) & (row_iy < gradient.shape[0])
    row_gradient[inside] = gradient[row_iy[inside], row_ix[inside]]
    gradient_valid = np.isfinite(row_gradient)
    gradient_threshold = float(np.quantile(row_gradient[gradient_valid], 0.9))
    high_gradient = gradient_valid & (row_gradient >= gradient_threshold)

    speed_groups = [
        ("slow_lt_100", frame["observed_speed"] < 100),
        ("intermediate_100_to_500", frame["observed_speed"].between(100, 500, inclusive="left")),
        ("fast_500_to_1000", frame["observed_speed"].between(500, 1000, inclusive="left")),
        ("very_fast_ge_1000", frame["observed_speed"] >= 1000),
        ("supported_both", frame["both_supported"]),
        ("not_supported_by_both", ~frame["both_supported"]),
        ("strong_log_speed_gradient_top_decile", high_gradient),
        ("other_valid_log_speed_gradient", gradient_valid & ~high_gradient),
    ]
    groups = pd.DataFrame([group_record(name, frame.loc[mask], frame) for name, mask in speed_groups])

    c_abs = frame["C_difference"].abs()
    valid = c_abs.notna()
    c_q25, c_q75 = c_abs[valid].quantile([0.25, 0.75])
    e_q25, e_q75 = frame.loc[valid, "model_error"].quantile([0.25, 0.75])
    top_speed = frame["observed_speed"] >= frame["observed_speed"].quantile(0.9)
    top_error = frame["model_error"] >= frame["model_error"].quantile(0.9)
    net = np.sum(frame["uniform_error"] ** 2 - frame["model_error"] ** 2)
    payload = {
        "schema": "jog-pig-cfg02-spatial-diagnostic-v1",
        "status": "complete",
        "provenance": provenance,
        "aggregate": {
            "rows": int(len(frame)),
            "uniform_rmse_m_per_a": float(forward["CFG02"]["uniform_vector_rmse_m_per_a"]),
            "CFG01_rmse_m_per_a": float(forward["CFG01"]["vector_rmse_m_per_a"]),
            "CFG02_rmse_m_per_a": float(forward["CFG02"]["vector_rmse_m_per_a"]),
            "CFG03_rmse_m_per_a": float(forward["CFG03"]["vector_rmse_m_per_a"]),
            "inversion_rmse_m_per_a": float(forward["CFG02"]["inversion_vector_rmse_m_per_a"]),
            "CFG02_relative_rmse": float(forward["CFG02"]["vector_rmse_m_per_a"] / forward["CFG02"]["uniform_vector_rmse_m_per_a"]),
            "CFG02_absolute_rmse_reduction_m_per_a": float(forward["CFG02"]["uniform_vector_rmse_m_per_a"] - forward["CFG02"]["vector_rmse_m_per_a"]),
            "fraction_local_error_below_uniform": float((frame["error_difference"] < 0).mean()),
            "both_support_fraction": float(frame["both_supported"].mean()),
            "minimum_marginal_coverage": float(extra["support"]["minimum_marginal_coverage"]),
            "joint_coverage": float(extra["support"]["joint_coverage"]),
            "C_rmse": float(extra["cmetrics"]["CFG02"]["C_rmse"]),
            "C_bias": float(extra["cmetrics"]["CFG02"]["C_bias"]),
        },
        "cross_diagnostics": {
            "fastest_decile_speed_threshold_m_per_a": float(frame["observed_speed"].quantile(0.9)),
            "fastest_decile_fraction_of_model_squared_error": float(np.sum(frame.loc[top_speed, "model_error"] ** 2) / np.sum(frame["model_error"] ** 2)),
            "fastest_decile_net_squared_error_reduction_fraction": float(np.sum(frame.loc[top_speed, "uniform_error"] ** 2 - frame.loc[top_speed, "model_error"] ** 2) / net),
            "top_error_decile_fraction_in_fastest_speed_decile": float(np.mean(top_speed[top_error])),
            "log_speed_gradient_valid_area_fraction": float(np.mean(gradient_valid)),
            "strong_log_speed_gradient_threshold_per_km": gradient_threshold,
            "top_error_decile_fraction_in_strong_gradient_zone": float(np.mean(high_gradient[top_error])),
            "strong_gradient_fraction_of_model_squared_error": float(np.sum(frame.loc[high_gradient, "model_error"] ** 2) / np.sum(frame["model_error"] ** 2)),
            "high_C_difference_low_velocity_error_fraction": float(np.mean((c_abs[valid] >= c_q75) & (frame.loc[valid, "model_error"] <= e_q25))),
            "low_C_difference_high_velocity_error_fraction": float(np.mean((c_abs[valid] <= c_q25) & (frame.loc[valid, "model_error"] >= e_q75))),
            "C_abs_difference_velocity_error_spearman": float(
                c_abs[valid].rank(method="average").corr(
                    frame.loc[valid, "model_error"].rank(method="average")
                )
            ),
        },
        "support_alignment": {
            "method": "stable row_id join from canonical eligible order to finalized support archive",
            "archived_evaluation_map_support_category_used": False,
            "reason": "the embedded category vector contains unassigned 255 values after a row-order change; velocity and aggregate metrics are unaffected",
        },
        "group_table": "pig_cfg02_spatial_groups.csv",
        "limits": {
            "explicit_grounding_line_mask_available": False,
            "explicit_shear_margin_mask_available": False,
            "velocity_display_grid_m": 450.0,
            "velocity_display_interpolation": "bilinear RGBA rendering only; no metric uses interpolated values",
            "C_display_interpolation": "masked native-control triangulation with Gouraud shading",
            "interpretation_policy": "use observed speed and model-domain linework; do not label unmapped structures as grounding zones or shear margins",
        },
    }
    return groups, payload


def display_grids(frame: pd.DataFrame):
    return grid_average(frame, ["observed_speed", "model_error", "error_difference"])


def native_display_grids(frame: pd.DataFrame):
    return native_grid(frame, ["observed_speed", "model_error", "error_difference", "inversion_error"])


def finish_map(ax, x_edges, y_edges, grids, *, contour_color="0.35", add_locator=True,
               show_inversion_reference=False):
    # The raster edge already identifies the PIG evaluation footprint.  Avoid
    # overlaying the whole-sector/model-domain linework here: several clipped
    # segments crossed the catchment and looked like physical boundaries.
    contours = ax.contour(
        (x_edges[:-1] + x_edges[1:]) / 2,
        (y_edges[:-1] + y_edges[1:]) / 2,
        grids["observed_speed"],
        levels=[100, 500, 1000],
        colors=contour_color,
        linewidths=[0.65, 0.85, 1.05],
        linestyles=[":", "--", "-"],
        zorder=9,
    )
    if show_inversion_reference:
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        ax.contour(
            x_centers, y_centers, grids["inversion_error"],
            levels=[INVERSION_CONTOUR_LEVEL], colors="0.08", linewidths=1.35,
            linestyles="solid", zorder=12,
        )
        ax.contour(
            x_centers, y_centers, grids["inversion_error"],
            levels=[INVERSION_CONTOUR_LEVEL], colors=INVERSION_CONTOUR_COLOR,
            linewidths=0.72, linestyles="solid", zorder=13,
        )
        handle = Line2D(
            [0], [0], color=INVERSION_CONTOUR_COLOR, lw=1.1,
            path_effects=[pe.Stroke(linewidth=2.0, foreground="0.08"), pe.Normal()],
        )
        ax.legend(
            [handle], [r"Inversion: 100 m a$^{-1}$"],
            loc="lower left", bbox_to_anchor=(0.018, 0.018), frameon=True,
            fancybox=False, edgecolor="0.45", handlelength=1.8,
            borderpad=0.25, fontsize=8.6,
        )
    ax.set_xlim(x_edges[0] - 5, x_edges[-1] + 5)
    ax.set_ylim(y_edges[0] - 5, y_edges[-1] + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("Polar stereographic x (km)")
    ax.set_ylabel("Polar stereographic y (km)")
    ax.tick_params(direction="out", length=2.5)
    if add_locator:
        base.add_antarctica_locator(ax, base.region_context()["outline"])


def panel_figure(kind: str, cplot: dict, x_edges, y_edges, grids) -> plt.Figure:
    # A horizontal colorbar leaves the map itself substantially wider when the
    # three panels are placed side by side in a two-column figure.
    fig, ax = plt.subplots(figsize=(5.25, 4.55), constrained_layout=True)
    if kind == "C":
        image = ax.tripcolor(
            cplot["triangulation"],
            cplot["difference_native"],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0, vmin=-C_LIMIT, vmax=C_LIMIT),
            shading="gouraud",
            rasterized=True,
            zorder=2,
        )
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", shrink=0.84,
                            pad=0.10, aspect=28, extend="both")
        cbar.set_label(r"$C_{\rm ML}-C_{\rm ref}$")
    elif kind == "absolute":
        image = ax.imshow(
            np.ma.masked_invalid(grids["model_error"]),
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="inferno",
            norm=LogNorm(vmin=1, vmax=2000),
            interpolation="bilinear",
            interpolation_stage="rgba",
            rasterized=True,
            zorder=2,
        )
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", shrink=0.84,
                            pad=0.10, aspect=28, extend="both")
        cbar.set_label(r"Vector velocity error (m a$^{-1}$; log scale)")
        cbar.set_ticks([1, 10, 100, 1000])
        cbar.set_ticklabels(["1", "10", "100", "1000"])
    elif kind == "difference":
        image = ax.imshow(
            np.ma.masked_invalid(grids["error_difference"]),
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="RdBu_r",
            norm=SymLogNorm(linthresh=10, linscale=1, vmin=-2000, vmax=2000, base=10),
            interpolation="bilinear",
            interpolation_stage="rgba",
            rasterized=True,
            zorder=2,
        )
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", shrink=0.84,
                            pad=0.10, aspect=28, extend="both")
        cbar.set_label("ML - uniform-$C$ error " r"(m a$^{-1}$; symlog)")
        cbar.set_ticks([-1000, -100, -10, 0, 10, 100, 1000])
        cbar.set_ticklabels(["-1000", "-100", "-10", "0", "10", "100", "1000"])
    else:
        raise ValueError(kind)
    cbar.ax.tick_params(labelsize=10.5)
    finish_map(
        ax, x_edges, y_edges, grids,
        contour_color="white" if kind == "absolute" else "0.35",
        show_inversion_reference=(kind == "absolute"),
    )
    return fig


def save_figure(cplot: dict, frame: pd.DataFrame) -> None:
    base.style()
    plt.rcParams.update({"axes.labelsize": 12.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
    OUT.mkdir(parents=True, exist_ok=True)
    x_edges, y_edges, grids = native_display_grids(frame)
    kinds = ["C", "absolute", "difference"]
    figures = [panel_figure(kind, cplot, x_edges, y_edges, grids) for kind in kinds]
    for kind, fig in zip(kinds, figures):
        fig.savefig(PANEL_PDFS[kind], bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    preview, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), constrained_layout=True)
    letters = ["(a)", "(b)", "(c)"]
    for ax, kind, letter in zip(axes, kinds, letters):
        if kind == "C":
            image = ax.tripcolor(cplot["triangulation"], cplot["difference_native"], shading="gouraud",
                                 cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-C_LIMIT, vmax=C_LIMIT),
                                 rasterized=True, zorder=2)
            cbar = preview.colorbar(image, ax=ax, orientation="horizontal", shrink=.84,
                                    pad=.10, aspect=28, extend="both")
            cbar.set_label(r"$C_{\rm ML}-C_{\rm ref}$")
        elif kind == "absolute":
            image = ax.imshow(np.ma.masked_invalid(grids["model_error"]), origin="lower",
                              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap="inferno",
                              norm=LogNorm(vmin=1, vmax=2000), interpolation="bilinear",
                              interpolation_stage="rgba", rasterized=True, zorder=2)
            cbar = preview.colorbar(image, ax=ax, orientation="horizontal", shrink=.84,
                                    pad=.10, aspect=28, extend="both")
            cbar.set_label(r"Error (m a$^{-1}$; log)")
        else:
            image = ax.imshow(np.ma.masked_invalid(grids["error_difference"]), origin="lower",
                              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap="RdBu_r",
                              norm=SymLogNorm(linthresh=10, linscale=1, vmin=-2000, vmax=2000, base=10),
                              interpolation="bilinear", interpolation_stage="rgba", rasterized=True, zorder=2)
            cbar = preview.colorbar(image, ax=ax, orientation="horizontal", shrink=.84,
                                    pad=.10, aspect=28, extend="both")
            cbar.set_label(r"ML - uniform error (m a$^{-1}$; symlog)")
        finish_map(
            ax, x_edges, y_edges, grids,
            contour_color="white" if kind == "absolute" else "0.35",
            show_inversion_reference=(kind == "absolute"),
        )
        ax.text(0.5, -0.36, letter, transform=ax.transAxes, ha="center", va="top", fontweight="bold")
        if ax is not axes[0]:
            ax.set_ylabel("")
    preview.savefig(PREVIEW, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(preview)


def main() -> None:
    provenance = verify_inputs()
    cplot, frame, forward, extra = load_evidence()
    groups, audit = spatial_audit(frame, forward, extra, provenance)
    save_figure(cplot, frame)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    groups.to_csv(ANALYSIS / "pig_cfg02_spatial_groups.csv", index=False)
    audit["outputs"] = {
        **{path.name: sha(path) for path in PANEL_PDFS.values()},
        PREVIEW.name: sha(PREVIEW),
        "pig_cfg02_spatial_groups.csv": sha(ANALYSIS / "pig_cfg02_spatial_groups.csv"),
    }
    (ANALYSIS / "pig_cfg02_spatial_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete",
        "pdfs": [str(path) for path in PANEL_PDFS.values()],
        "preview": str(PREVIEW),
    }, indent=2))


if __name__ == "__main__":
    main()
