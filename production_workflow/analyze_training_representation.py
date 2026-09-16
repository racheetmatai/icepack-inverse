#!/usr/bin/env python3
"""Training-representation diagnostic for frozen JOG holdouts.

This script reads the accepted dataset, splits, support transform definition,
and forward-evaluation archives. It does not train a model or run Icepack.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer


SEED = 20260909
K = 20
REFERENCE_QUERIES = 10_000
FEATURES = {
    "CFG02": ["s", "h", "mag_s", "mag_h", "surface_air_temp"],
    "CFG04": ["b", "mag_b", "heatflux"],
}
CONFIG_LONG = {
    "CFG02": "CFG02_best_ice",
    "CFG04": "CFG04_best_geophysical",
}
SPEED_BINS = [-np.inf, 100.0, 500.0, 1000.0, np.inf]
SPEED_LABELS = ["<100", "100-500", "500-1000", ">=1000"]
REP_LABELS = ["<=50", "50-95", ">95"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload)
    body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def unpack_member_mask(packed: np.ndarray, row_count: int) -> np.ndarray:
    return np.unpackbits(packed, bitorder="little", count=row_count).astype(bool)


def fit_support_transform(reference_path: Path, config: str):
    archive = np.load(reference_path, allow_pickle=False)
    eligible = archive["eligible"].astype(bool)
    flat = np.flatnonzero(eligible.ravel())
    grid = archive["features"].reshape((-1, archive["features"].shape[-1]))[flat].astype(float)
    names = [str(value) for value in archive["feature_names"]]
    columns = [names.index(name) for name in FEATURES[config]]
    values = grid[:, columns]
    quantile = QuantileTransformer(
        n_quantiles=min(1000, len(values)),
        output_distribution="normal",
        random_state=20260811,
        subsample=len(values),
    )
    normalized = quantile.fit_transform(values)
    pca = PCA(n_components=0.99, whiten=True, svd_solver="full")
    pca.fit(normalized)
    return quantile, pca, int(len(values))


def transform_rows(frame: pd.DataFrame, config: str, quantile, pca) -> np.ndarray:
    values = frame[FEATURES[config]].to_numpy(np.float64)
    return np.asarray(pca.transform(quantile.transform(values)), dtype=np.float64)


def query_d20(tree: cKDTree, queries: np.ndarray) -> np.ndarray:
    distances, _ = tree.query(queries, k=K, workers=-1)
    return np.asarray(distances[:, K - 1], dtype=np.float64)


def reference_d20(tree: cKDTree, queries: np.ndarray, self_local_indices: np.ndarray) -> np.ndarray:
    distances, indices = tree.query(queries, k=K + 1, workers=-1)
    result = np.empty(len(queries), dtype=np.float64)
    for row in range(len(queries)):
        keep = indices[row] != self_local_indices[row]
        if np.count_nonzero(keep) >= K:
            result[row] = distances[row][keep][K - 1]
        else:
            # With more than K tied duplicate rows, the tree may omit the
            # query row. The first K returned rows are then valid neighbors.
            result[row] = distances[row][K - 1]
    return result


def empirical_midrank_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return 100.0 * (left + right) / (2.0 * len(ordered))


def brute_d20(training: np.ndarray, query: np.ndarray, self_index: int | None = None) -> float:
    squared = np.sum((training - query) ** 2, axis=1)
    if self_index is not None:
        squared[self_index] = np.inf
    return float(np.sqrt(np.partition(squared, K - 1)[K - 1]))


def experiment_specs() -> list[tuple[str, list[str]]]:
    specs = [(f"SQ{i:02d}", ["CFG02", "CFG04"]) for i in range(1, 11)]
    specs.append(("REG_PIG", ["CFG02"]))
    return specs


def stable_seed(experiment: str, member: int, purpose: int) -> int:
    order = 11 if experiment == "REG_PIG" else int(experiment[-2:])
    return SEED + purpose * 100_000 + order * 100 + member


def read_inputs(args):
    dataset_manifest = json.loads((args.dataset_dir / "dataset_manifest.json").read_text())
    dataset_path = args.dataset_dir / "canonical_master_dataset.csv.gz"
    if sha256(dataset_path) != dataset_manifest["output_sha256"][dataset_path.name]:
        raise ValueError("Canonical dataset hash mismatch")
    columns = [
        "row_id", "common_eligible", "observed_vx", "observed_vy",
        "square_test_id", "region_code", *sorted(set(sum(FEATURES.values(), []))),
    ]
    frame = pd.read_csv(dataset_path, usecols=columns, low_memory=False)
    frame = frame.loc[frame["common_eligible"].astype(bool)].copy()
    eligible_row_ids_original_order = frame["row_id"].astype(str).to_numpy()
    frame = frame.sort_values("row_id", kind="stable").reset_index(drop=True)
    if len(frame) != dataset_manifest["common_eligible_count"]:
        raise AssertionError("Eligible row count mismatch")
    if frame["row_id"].duplicated().any():
        raise AssertionError("Duplicate stable row IDs")
    split_methods = json.loads((args.split_dir / "methods.json").read_text())
    if split_methods["dataset_manifest_id"] != dataset_manifest["manifest_id"]:
        raise ValueError("Split/dataset identity mismatch")
    return frame, eligible_row_ids_original_order, dataset_path, dataset_manifest, split_methods


def select_query_indices(frame: pd.DataFrame, experiment: str) -> np.ndarray:
    if experiment.startswith("SQ"):
        indices = np.flatnonzero(frame["square_test_id"].eq(experiment).to_numpy())
        return indices
    indices = np.flatnonzero(frame["region_code"].to_numpy(np.int8) == 1)
    return indices


def benchmark(args, frame: pd.DataFrame, transformed: np.ndarray) -> dict:
    experiment, member = "SQ01", 1
    archive = np.load(args.split_dir / "member_splits" / f"{experiment}.npz", allow_pickle=False)
    train_mask = unpack_member_mask(archive["train"][member - 1], int(archive["row_count"]))
    train_global = np.flatnonzero(train_mask)
    training = transformed[train_mask]
    query_global = select_query_indices(frame, experiment)[:1000]
    query = transformed[query_global]
    rng = np.random.default_rng(stable_seed(experiment, member, 1))
    ref_local = np.sort(rng.choice(len(training), size=1000, replace=False))
    started = time.perf_counter()
    tree = cKDTree(training, balanced_tree=True, compact_nodes=True)
    build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    held = query_d20(tree, query)
    ref = reference_d20(tree, training[ref_local], ref_local)
    query_seconds = time.perf_counter() - started
    held_checks = [brute_d20(training, query[i]) for i in range(4)]
    ref_checks = [brute_d20(training, training[ref_local[i]], int(ref_local[i])) for i in range(4)]
    exact = bool(
        np.allclose(held[:4], held_checks, rtol=1e-11, atol=1e-12)
        and np.allclose(ref[:4], ref_checks, rtol=1e-11, atol=1e-12)
    )
    if not exact:
        raise AssertionError("cKDTree/brute-force d20 check failed")
    return {
        "experiment": experiment,
        "configuration": "CFG02",
        "member": member,
        "training_rows": int(len(training)),
        "heldout_queries": int(len(query)),
        "reference_queries": int(len(ref_local)),
        "tree_build_seconds": build_seconds,
        "two_query_batches_seconds": query_seconds,
        "brute_force_checks": 8,
        "exact_check_passed": exact,
        "estimated_tree_builds_full_analysis": 210,
        "estimated_tree_build_minutes": build_seconds * 210 / 60.0,
    }


def calculate_case(args, frame, transformed, experiment, config, query_global, output, sample_writer):
    split_path = args.split_dir / "member_splits" / f"{experiment}.npz"
    split = np.load(split_path, allow_pickle=False)
    if int(split["row_count"]) != len(frame):
        raise AssertionError(f"Split row count mismatch: {experiment}")
    test_ids = frame.loc[query_global, "row_id"].to_numpy(str)
    member_d20, member_percentiles = [], []
    cache_dir = output / "member_cache"
    cache_dir.mkdir(exist_ok=True)
    for member in range(1, 11):
        suffix = f"_Q{len(query_global)}" if experiment == "REG_PIG" else ""
        cache = cache_dir / f"{experiment}_{config}{suffix}_M{member:02d}.npz"
        train_mask = unpack_member_mask(split["train"][member - 1], len(frame))
        if np.any(train_mask[query_global]):
            raise AssertionError(f"Training/test overlap: {experiment} M{member:02d}")
        train_global = np.flatnonzero(train_mask)
        if cache.is_file():
            saved = np.load(cache, allow_pickle=False)
            if not np.array_equal(saved["test_row_id"].astype(str), test_ids):
                raise ValueError(f"Stale query cache: {cache}")
            member_d20.append(saved["heldout_d20"])
            member_percentiles.append(saved["heldout_percentile"])
            ref_global = saved["reference_global_index"].astype(np.int64)
        else:
            training = transformed[train_mask]
            tree = cKDTree(training, balanced_tree=True, compact_nodes=True)
            rng = np.random.default_rng(stable_seed(experiment, member, 1))
            sample_size = min(REFERENCE_QUERIES, len(training))
            ref_local = np.sort(rng.choice(len(training), size=sample_size, replace=False))
            ref_global = train_global[ref_local]
            held_d20 = query_d20(tree, transformed[query_global])
            ref_d20 = reference_d20(tree, training[ref_local], ref_local)
            percentiles = empirical_midrank_percentile(ref_d20, held_d20)
            np.savez_compressed(
                cache,
                test_row_id=test_ids,
                heldout_d20=held_d20,
                heldout_percentile=percentiles,
                reference_global_index=ref_global,
                reference_d20=ref_d20,
            )
            member_d20.append(held_d20)
            member_percentiles.append(percentiles)
        sample_writer.writerows(
            (experiment, config, member, "training_reference", row_id)
            for row_id in frame.loc[ref_global, "row_id"].astype(str)
        )
    return (
        np.median(np.vstack(member_d20), axis=0),
        np.median(np.vstack(member_percentiles), axis=0),
    )


def load_velocity(args, experiment, config, query_ids):
    path = args.forward_dir / "median_map_data" / f"{experiment}_{config}_MEDIAN.npz"
    archive = np.load(path, allow_pickle=False)
    row_ids = pd.Index(archive["row_id"].astype(str))
    lookup = row_ids.get_indexer(query_ids)
    if np.any(lookup < 0):
        raise AssertionError(f"Forward archive row-ID mismatch: {experiment} {config}")
    ml_error = archive["error_magnitude"][lookup].astype(float)
    improvement = archive["signed_local_squared_error_improvement"][lookup].astype(float)
    base_squared = ml_error ** 2 + improvement
    if float(np.min(base_squared)) < -1e-7:
        raise AssertionError("Negative reconstructed uniform squared error")
    uniform_error = np.sqrt(np.maximum(base_squared, 0.0))
    observed_speed = np.hypot(
        archive["observed_vx"][lookup], archive["observed_vy"][lookup]
    )
    return ml_error, uniform_error, observed_speed


def load_support(args, eligible_row_ids_original_order, experiment, config, query_ids):
    """Load frozen support categories by stable row ID.

    The support archive stores integer positions in the canonical eligible-row
    order used by describe_heldout_distributions.py.  The training split and
    forward-evaluation tables are sorted by stable row ID, so positions cannot
    be transferred directly between them.
    """
    with np.load(args.support_dir / "point_support_categories.npz", allow_pickle=False) as archive:
        positions = archive[f"{experiment}__row_index"].astype(np.int64)
        categories = archive[f"{experiment}__{CONFIG_LONG[config]}"].astype(np.uint8)
    support_ids = pd.Index(eligible_row_ids_original_order[positions])
    lookup = support_ids.get_indexer(query_ids)
    if np.any(lookup < 0):
        raise AssertionError(f"Support row-ID mismatch: {experiment} {config}")
    return categories[lookup]


def rep_category(percentile):
    return pd.cut(
        percentile,
        bins=[-np.inf, 50.0, 95.0, np.inf],
        labels=REP_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)


def spearman(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return np.nan
    return float(np.corrcoef(rankdata(x[valid], method="average"), rankdata(y[valid], method="average"))[0, 1])


def summarize_case(points: pd.DataFrame):
    categories, associations, speed_rows = [], [], []
    total = len(points)
    for label in REP_LABELS:
        group = points.loc[points["representation_category"].eq(label)]
        if group.empty:
            continue
        ml_rmse = float(np.sqrt(np.mean(group["ml_error"] ** 2)))
        uniform_rmse = float(np.sqrt(np.mean(group["uniform_error"] ** 2)))
        categories.append({
            "experiment": points["experiment"].iloc[0],
            "configuration": points["configuration"].iloc[0],
            "representation_category": label,
            "query_rows": len(group),
            "area_fraction": len(group) / total,
            "observed_speed_median_m_per_a": float(group["observed_speed"].median()),
            "observed_speed_q25_m_per_a": float(group["observed_speed"].quantile(.25)),
            "observed_speed_q75_m_per_a": float(group["observed_speed"].quantile(.75)),
            "ml_rmse_m_per_a": ml_rmse,
            "uniform_rmse_m_per_a": uniform_rmse,
            "rmse_ratio": None if uniform_rmse <= 1e-12 else ml_rmse / uniform_rmse,
            "area_fraction_ml_local_error_lower": float(np.mean(group["ml_error"] < group["uniform_error"])),
        })
    associations.append({
        "experiment": points["experiment"].iloc[0],
        "configuration": points["configuration"].iloc[0],
        "speed_class": "all",
        "query_rows": total,
        "spearman_representation_vs_ml_error": spearman(points["representation_percentile"].to_numpy(), points["ml_error"].to_numpy()),
        "spearman_representation_vs_error_difference": spearman(points["representation_percentile"].to_numpy(), points["ml_error"].to_numpy() - points["uniform_error"].to_numpy()),
    })
    for label in SPEED_LABELS:
        group = points.loc[points["speed_class"].eq(label)]
        if group.empty:
            continue
        associations.append({
            "experiment": points["experiment"].iloc[0],
            "configuration": points["configuration"].iloc[0],
            "speed_class": label,
            "query_rows": len(group),
            "spearman_representation_vs_ml_error": spearman(group["representation_percentile"].to_numpy(), group["ml_error"].to_numpy()),
            "spearman_representation_vs_error_difference": spearman(group["representation_percentile"].to_numpy(), group["ml_error"].to_numpy() - group["uniform_error"].to_numpy()),
        })
        for rep in REP_LABELS:
            subgroup = group.loc[group["representation_category"].eq(rep)]
            if subgroup.empty:
                continue
            ml_rmse = float(np.sqrt(np.mean(subgroup["ml_error"] ** 2)))
            uniform_rmse = float(np.sqrt(np.mean(subgroup["uniform_error"] ** 2)))
            speed_rows.append({
                "experiment": points["experiment"].iloc[0],
                "configuration": points["configuration"].iloc[0],
                "speed_class": label,
                "representation_category": rep,
                "query_rows": len(subgroup),
                "ml_rmse_m_per_a": ml_rmse,
                "uniform_rmse_m_per_a": uniform_rmse,
                "rmse_ratio": None if uniform_rmse <= 1e-12 else ml_rmse / uniform_rmse,
            })
    return categories, associations, speed_rows


def make_figures(points: pd.DataFrame, categories: pd.DataFrame, output: Path):
    colors = {"CFG02": "#2c7fb8", "CFG04": "#d95f0e"}
    square_names = [f"SQ{i:02d}" for i in range(1, 11)]
    fig, (ax, pig_ax) = plt.subplots(1, 2, figsize=(12.6, 5.5), gridspec_kw={"width_ratios": [4.8, 1]}, constrained_layout=True)
    positions = np.arange(1, 11)
    for offset, config in [(-0.17, "CFG02"), (0.17, "CFG04")]:
        datasets = [points.loc[(points.experiment == e) & (points.configuration == config), "representation_percentile"].to_numpy() for e in square_names]
        bp = ax.boxplot(datasets, positions=positions + offset, widths=.28, patch_artist=True, showfliers=False, whis=(5, 95))
        for box in bp["boxes"]: box.set(facecolor=colors[config], alpha=.62, edgecolor=colors[config])
        for key in ["whiskers", "caps", "medians"]:
            for artist in bp[key]: artist.set(color=colors[config], linewidth=1.1)
        ax.plot([], [], color=colors[config], lw=7, alpha=.62, label=config)
    ax.axhline(95, color="#555555", lw=1, ls="--")
    ax.set_xticks(positions, square_names)
    ax.set_ylabel("Training-representation percentile")
    ax.set_xlabel("Central 50 km square")
    ax.set_ylim(0, 101)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Ten spatial tests")
    pig = points.loc[(points.experiment == "REG_PIG") & (points.configuration == "CFG02"), "representation_percentile"].to_numpy()
    bp = pig_ax.boxplot([pig], positions=[1], widths=.45, patch_artist=True, showfliers=False, whis=(5, 95))
    bp["boxes"][0].set(facecolor=colors["CFG02"], alpha=.62, edgecolor=colors["CFG02"])
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]: artist.set(color=colors["CFG02"], linewidth=1.1)
    pig_ax.axhline(95, color="#555555", lw=1, ls="--")
    pig_ax.set_xticks([1], ["PIG"]); pig_ax.set_ylim(0, 101); pig_ax.set_yticklabels([])
    pig_ax.set_title("Catchment test")
    fig.suptitle("Held-out predictor combinations relative to each member's training rows\nHigher percentiles indicate sparser training representation", fontsize=13)
    fig.savefig(output / "figure_A_training_representation_percentiles.png", dpi=240)
    fig.savefig(output / "figure_A_training_representation_percentiles.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True, constrained_layout=True)
    x = np.arange(3)
    for ax, config in zip(axes, ["CFG02", "CFG04"]):
        selected = categories.loc[categories.configuration.eq(config)]
        endpoints = []
        for experiment in square_names:
            group = selected.loc[selected.experiment.eq(experiment)].set_index("representation_category").reindex(REP_LABELS)
            ax.plot(x, group["rmse_ratio"], color="#a9a9a9", lw=.8, alpha=.8)
            if np.isfinite(group["rmse_ratio"].iloc[-1]):
                endpoints.append((float(group["rmse_ratio"].iloc[-1]), experiment[-2:]))
        # Retain square identities without overlapping endpoint labels.
        ordered = sorted(endpoints)
        label_y = []
        minimum_gap = 0.045
        for value, _ in ordered:
            label_y.append(value if not label_y else max(value, label_y[-1] + minimum_gap))
        if label_y and label_y[-1] > 1.84:
            shift = label_y[-1] - 1.84
            label_y = [value - shift for value in label_y]
        for (value, label), placed in zip(ordered, label_y):
            ax.plot([2.01, 2.10], [value, placed], color="#a9a9a9", lw=.55, clip_on=False)
            ax.text(2.12, placed, label, fontsize=6.5, color="#555555", va="center", clip_on=False)
        square_group = selected.loc[selected.experiment.str.startswith("SQ")]
        medians = square_group.groupby("representation_category")["rmse_ratio"].median().reindex(REP_LABELS)
        ax.plot(x, medians, color=colors[config], lw=2.8, marker="o", label="Median across squares")
        if config == "CFG02":
            pig_group = selected.loc[selected.experiment.eq("REG_PIG")].set_index("representation_category").reindex(REP_LABELS)
            ax.plot(x, pig_group["rmse_ratio"], color="#54278f", lw=2, marker="s", ls="--", label="PIG")
        ax.axhline(1, color="black", lw=1, ls="--")
        ax.set_xticks(x, ["Up to 50th", "50th-95th", "Above 95th"])
        ax.set_xlabel("Training-representation percentile category")
        ax.set_title(config)
        ax.set_xlim(-0.10, 2.22)
        ax.grid(axis="y", color="#dddddd", lw=.5)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("RMSE / uniform-C RMSE")
    fig.suptitle("Forward-velocity performance by training representation", fontsize=13)
    fig.savefig(output / "figure_B_rmse_ratio_by_representation.png", dpi=240)
    fig.savefig(output / "figure_B_rmse_ratio_by_representation.pdf")
    plt.close(fig)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame, eligible_row_ids_original_order, dataset_path, dataset_manifest, split_methods = read_inputs(args)
    transforms, transformed = {}, {}
    for config in FEATURES:
        quantile, pca, cells = fit_support_transform(args.reference_grid, config)
        transforms[config] = {"quantile": quantile, "pca": pca, "reference_cells": cells}
        cache = output / f"transformed_{config}.npy"
        if cache.is_file():
            transformed[config] = np.load(cache, mmap_mode="r")
        else:
            values = transform_rows(frame, config, quantile, pca)
            np.save(cache, values)
            transformed[config] = np.load(cache, mmap_mode="r")
    benchmark_result = benchmark(args, frame, transformed["CFG02"])
    (output / "benchmark.json").write_text(json.dumps(benchmark_result, indent=2) + "\n")
    if args.benchmark_only:
        print(json.dumps(benchmark_result, indent=2)); return

    point_frames = []
    with gzip.open(output / "sampled_row_ids.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        sample_writer = csv.writer(stream)
        sample_writer.writerow(["experiment", "configuration", "member", "sample", "row_id"])
        for experiment, configs in experiment_specs():
            query_global = select_query_indices(frame, experiment)
            query_ids = frame.loc[query_global, "row_id"].to_numpy(str)
            sample_writer.writerows(
                (experiment, "shared", 0, "heldout_query", row_id) for row_id in query_ids
            )
            for config in configs:
                d20, percentile = calculate_case(
                    args, frame, transformed[config], experiment, config, query_global, output, sample_writer
                )
                ml_error, uniform_error, speed = load_velocity(args, experiment, config, query_ids)
                support = load_support(
                    args, eligible_row_ids_original_order, experiment, config, query_ids
                )
                points = pd.DataFrame({
                    "experiment": experiment,
                    "configuration": config,
                    "row_id": query_ids,
                    "d20_median_across_members": d20,
                    "representation_percentile": percentile,
                    "representation_category": rep_category(percentile),
                    "observed_speed": speed,
                    "speed_class": pd.cut(speed, bins=SPEED_BINS, labels=SPEED_LABELS, right=False).astype(str),
                    "ml_error": ml_error,
                    "uniform_error": uniform_error,
                    "ml_minus_uniform_error": ml_error - uniform_error,
                    "support_category": support,
                    "both_support": support == 3,
                })
                point_frames.append(points)
                print(json.dumps({"complete": f"{experiment}_{config}", "queries": len(points)}), flush=True)
    all_points = pd.concat(point_frames, ignore_index=True)
    all_points.to_csv(output / "point_diagnostics.csv.gz", index=False, compression="gzip")

    category_rows, association_rows, speed_rows = [], [], []
    for _, group in all_points.groupby(["experiment", "configuration"], sort=False):
        c, a, s = summarize_case(group)
        category_rows.extend(c); association_rows.extend(a); speed_rows.extend(s)
    categories = pd.DataFrame(category_rows)
    associations = pd.DataFrame(association_rows)
    speed_table = pd.DataFrame(speed_rows)
    categories.to_csv(output / "category_metrics.csv", index=False)
    associations.to_csv(output / "associations.csv", index=False)
    speed_table.to_csv(output / "speed_stratified_metrics.csv", index=False)

    distribution = all_points.groupby(["experiment", "configuration"]).agg(
        query_rows=("row_id", "size"),
        percentile_q05=("representation_percentile", lambda x: x.quantile(.05)),
        percentile_q25=("representation_percentile", lambda x: x.quantile(.25)),
        percentile_median=("representation_percentile", "median"),
        percentile_q75=("representation_percentile", lambda x: x.quantile(.75)),
        percentile_q95=("representation_percentile", lambda x: x.quantile(.95)),
        fraction_above_95=("representation_percentile", lambda x: np.mean(x > 95)),
        fraction_both_support=("both_support", "mean"),
        fraction_zero_d20=("d20_median_across_members", lambda x: np.mean(x == 0)),
    ).reset_index()
    distribution.to_csv(output / "representation_summary.csv", index=False)

    support_summary = all_points.groupby(
        ["experiment", "configuration", "both_support"], sort=False
    ).agg(
        query_rows=("row_id", "size"),
        median_representation_percentile=("representation_percentile", "median"),
        fraction_above_95=("representation_percentile", lambda x: np.mean(x > 95)),
        ml_rmse_m_per_a=("ml_error", lambda x: np.sqrt(np.mean(x ** 2))),
        uniform_rmse_m_per_a=("uniform_error", lambda x: np.sqrt(np.mean(x ** 2))),
    ).reset_index()
    totals = support_summary.groupby(["experiment", "configuration"])["query_rows"].transform("sum")
    support_summary["area_fraction"] = support_summary["query_rows"] / totals
    support_summary["rmse_ratio"] = (
        support_summary["ml_rmse_m_per_a"] / support_summary["uniform_rmse_m_per_a"]
    )
    support_summary.to_csv(output / "support_stratified_metrics.csv", index=False)

    # Square-level descriptive association; ten squares, no p-values.
    overall = all_points.groupby(["experiment", "configuration"]).agg(
        median_representation_percentile=("representation_percentile", "median"),
        fraction_above_95=("representation_percentile", lambda x: np.mean(x > 95)),
        ml_rmse=("ml_error", lambda x: np.sqrt(np.mean(x ** 2))),
        uniform_rmse=("uniform_error", lambda x: np.sqrt(np.mean(x ** 2))),
    ).reset_index()
    overall["rmse_ratio"] = overall["ml_rmse"] / overall["uniform_rmse"]
    square_assoc = []
    for config in FEATURES:
        group = overall.loc[overall.experiment.str.startswith("SQ") & overall.configuration.eq(config)]
        square_assoc.append({
            "configuration": config,
            "squares": len(group),
            "spearman_median_percentile_vs_rmse_ratio": spearman(group.median_representation_percentile.to_numpy(), group.rmse_ratio.to_numpy()),
            "spearman_fraction_above_95_vs_rmse_ratio": spearman(group.fraction_above_95.to_numpy(), group.rmse_ratio.to_numpy()),
        })
    pd.DataFrame(square_assoc).to_csv(output / "square_level_associations.csv", index=False)
    overall.to_csv(output / "case_overall_metrics.csv", index=False)

    make_figures(all_points, categories, output)

    # Verification against authoritative full-population square metrics.
    authoritative = pd.read_csv(args.forward_dir / "median_population_metrics.csv")
    checks = []
    for row in overall.loc[overall.experiment.str.startswith("SQ")].itertuples():
        source = authoritative.loc[
            authoritative.control_id.eq(f"{row.experiment}_{row.configuration}_MEDIAN")
            & authoritative.population.eq("central_50km")
            & authoritative.support_stratum.eq("all")
        ].iloc[0]
        checks.append({
            "experiment": row.experiment,
            "configuration": row.configuration,
            "ml_rmse_absolute_difference": abs(row.ml_rmse - source.vector_rmse_m_per_a),
            "uniform_rmse_absolute_difference": abs(row.uniform_rmse - source.uniform_vector_rmse_m_per_a),
        })
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "authoritative_metric_checks.csv", index=False)
    frozen_support_summary = pd.read_csv(args.support_dir / "support_categories.csv")
    support_checks = []
    for row in distribution.loc[distribution.experiment.str.startswith("SQ")].itertuples():
        source = frozen_support_summary.loc[
            frozen_support_summary.experiment.eq(row.experiment)
            & frozen_support_summary.population.eq("central_50km")
            & frozen_support_summary.configuration.eq(CONFIG_LONG[row.configuration])
        ].iloc[0]
        support_checks.append({
            "experiment": row.experiment,
            "configuration": row.configuration,
            "both_support_absolute_difference": abs(row.fraction_both_support - source.both_fraction),
        })
    support_checks_frame = pd.DataFrame(support_checks)
    support_checks_frame.to_csv(output / "support_alignment_checks.csv", index=False)
    area_sums = categories.groupby(["experiment", "configuration"])["area_fraction"].sum()
    verification = {
        "schema": "jog-training-representation-verification-v1",
        "status": "passed",
        "dataset_rows": len(frame),
        "all_training_test_pairs_disjoint": True,
        "spatial_thinning_or_deduplication": False,
        "self_neighbor_removed_by_row_position": True,
        "brute_force_d20_check_passed": benchmark_result["exact_check_passed"],
        "area_fraction_max_abs_error": float(np.max(np.abs(area_sums.to_numpy() - 1))),
        "square_ml_rmse_max_abs_error": float(checks_frame.ml_rmse_absolute_difference.max()),
        "square_uniform_rmse_max_abs_error": float(checks_frame.uniform_rmse_absolute_difference.max()),
        "support_labels_joined_by_stable_row_id": True,
        "square_both_support_max_abs_error": float(support_checks_frame.both_support_absolute_difference.max()),
        "equal_area_weighting": "row means on common-area projected observation pixels, matching the authoritative forward evaluation",
        "pig_sampling": "none; all eligible held-out PIG rows were used",
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    inputs = {
        "dataset": dataset_path,
        "dataset_manifest": args.dataset_dir / "dataset_manifest.json",
        "split_manifest": args.split_dir / "split_bundle_manifest.json",
        "split_methods": args.split_dir / "methods.json",
        "reference_grid": args.reference_grid,
        "support_methods": args.support_dir / "methods.json",
        "support_labels": args.support_dir / "point_support_categories.npz",
        "support_summary": args.support_dir / "support_categories.csv",
        "forward_manifest": args.forward_dir / "evaluation_manifest.json",
        "forward_metrics": args.forward_dir / "median_population_metrics.csv",
    }
    manifest = {
        "schema": "jog-training-representation-diagnostic-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "descriptive association between actual training representation and held-out forward-velocity error",
        "dataset_manifest_id": dataset_manifest["manifest_id"],
        "split_dataset_manifest_id": split_methods["dataset_manifest_id"],
        "seed": SEED,
        "neighbor_k": K,
        "test_queries": "all central-square rows and all eligible PIG rows",
        "training_reference_queries_per_member": REFERENCE_QUERIES,
        "training_rows": "exact member-specific training rows, without thinning, rebalancing, or deduplication",
        "transform": "QuantileTransformer(normal, up to 1000 quantiles, seed 20260811) and whitened full-SVD PCA retaining >=99% variance, fitted to the frozen eligible 5 km sector grid, matching the support analysis",
        "percentile": "midrank empirical percentile relative to each member's sampled training-reference d20; median across ten members",
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    output_files = [p for p in output.iterdir() if p.is_file() and p.name not in {"manifest.json"}]
    manifest["output_sha256"] = {p.name: sha256(p) for p in sorted(output_files)}
    manifest["manifest_id"] = canonical_id(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest_id": manifest["manifest_id"], "output": str(output), "verification": verification}, indent=2))


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=root / "production_workflow/gate2_results/gate2_canonical_dataset_20260820_c")
    parser.add_argument("--split-dir", type=Path, default=root / "production_workflow/gate2_results/gate2_split_manifests_20260820_a")
    parser.add_argument("--support-dir", type=Path, default=root / "production_workflow/gate2_results/gate2_distribution_diagnostics_20260820_c")
    parser.add_argument("--reference-grid", type=Path, default=root / "production_workflow/frozen_design/amundsen_input_support_grid_5km.npz")
    parser.add_argument("--forward-dir", type=Path, default=root / "production_workflow/gate4_forward_evaluation_20260829_a")
    parser.add_argument("--output", type=Path, default=root / "production_workflow/training_representation_diagnostic_20260909_a")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    for name in ["dataset_dir", "split_dir", "support_dir", "reference_grid", "forward_dir"]:
        if not getattr(args, name).exists():
            raise FileNotFoundError(getattr(args, name))
    run(args)


if __name__ == "__main__":
    main()
