#!/usr/bin/env python3
"""Frozen descriptive input-support and reference-C diagnostics.

This program reads the accepted canonical master dataset.  It never changes
the dataset, feature configurations, masks, or experiment locations.  Joint
support retains the outcome-blind 5 km reference representation and cutoffs
used to freeze the experiments; marginal intervals are calculated from each
experiment's actual common-eligible development population.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer


SEED = 20260811
QUANTILES = np.asarray([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
FEATURE_CONFIGURATIONS = {
    "CFG01_all_ice": ["s", "h", "mag_s", "mag_h", "driving_stress", "surface_air_temp"],
    "CFG02_best_ice": ["s", "h", "mag_s", "mag_h", "surface_air_temp"],
    "CFG03_all_geophysical": ["b", "mag_b", "heatflux", "gravity_disturbance", "mag_anomaly"],
    "CFG04_best_geophysical": ["b", "mag_b", "heatflux"],
    "CFG05_best_combined": ["s", "h", "mag_s", "mag_h", "surface_air_temp", "b", "mag_b", "heatflux"],
    "CFG06_best_combined_direction": ["s", "h", "mag_s", "mag_h", "surface_air_temp", "b", "mag_b", "heatflux", "cos_theta_bs"],
}
REGION_NAMES = {
    1: "PIG", 2: "Thwaites", 3: "Dotson",
    4: "PIG-Thwaites inter-catchment",
    5: "Thwaites-Dotson inter-catchment",
}
CATEGORY_NAMES = np.asarray(["neither", "marginal_only", "joint_only", "both"])


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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile_record(values: np.ndarray) -> dict:
    q = np.quantile(values, QUANTILES)
    return {
        "count": int(values.size), "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        **{f"q{int(100*p):02d}": float(v) for p, v in zip(QUANTILES, q)},
    }


def fit_reference(values: np.ndarray) -> tuple[QuantileTransformer, PCA, np.ndarray]:
    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(values)), output_distribution="normal",
        random_state=SEED, subsample=len(values),
    )
    normalized = transformer.fit_transform(values)
    pca = PCA(n_components=0.99, whiten=True, svd_solver="full")
    transformed = pca.fit_transform(normalized)
    return transformer, pca, transformed


def scenario_specs(frame: pd.DataFrame) -> list[dict]:
    specs = []
    for number in range(1, 11):
        sid = f"SQ{number:02d}"
        footprint = frame["square_footprint_id"].eq(sid).to_numpy()
        central = frame["square_test_id"].eq(sid).to_numpy()
        specs.append({
            "id": sid, "kind": "square", "train": ~footprint,
            "eval": {"central_50km": central,
                     "exclusion_annulus": footprint & ~central,
                     "full_130km": footprint},
            "configs": list(FEATURE_CONFIGURATIONS),
        })
    code = frame["region_code"].to_numpy(dtype=np.int8)
    inter = np.isin(code, [4, 5])
    specs.append({
        "id": "REG_INTER", "kind": "regional", "train": np.isin(code, [1, 2, 3]),
        "eval": {"both_corridors": inter,
                 "pig_thwaites_corridor": code == 4,
                 "thwaites_dotson_corridor": code == 5},
        "configs": ["CFG04_best_geophysical", "CFG05_best_combined", "CFG06_best_combined_direction"],
    })
    specs.append({
        "id": "REG_PIG", "kind": "regional", "train": code != 1,
        "eval": {"PIG": code == 1},
        "configs": ["CFG02_best_ice", "CFG01_all_ice", "CFG03_all_geophysical"],
    })
    return specs


def reference_training_mask(spec: dict, grid_coordinates: np.ndarray, grid_regions: np.ndarray) -> np.ndarray:
    if spec["kind"] == "square":
        selected = spec["selected_square"]
        return ~(
            (grid_coordinates[:, 0] >= selected["footprint_xmin_m"])
            & (grid_coordinates[:, 0] < selected["footprint_xmax_m"])
            & (grid_coordinates[:, 1] >= selected["footprint_ymin_m"])
            & (grid_coordinates[:, 1] < selected["footprint_ymax_m"])
        )
    if spec["id"] == "REG_INTER":
        return np.isin(grid_regions, [1, 2, 3])
    return grid_regions != 1


def make_support_heatmap(rows: list[dict], output: Path) -> None:
    primary = [r for r in rows if r["population"] in ("central_50km", "both_corridors", "PIG")]
    labels = [f'{r["experiment"]} {r["configuration"].split("_")[0]}' for r in primary]
    values = np.asarray([[float(r["minimum_marginal_coverage"]), float(r["joint_coverage"]), float(r["both_fraction"])] for r in primary]) * 100
    fig, ax = plt.subplots(figsize=(8.2, max(8, 0.27 * len(primary))), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=100, cmap="viridis")
    ax.set_xticks(range(3), ["Minimum marginal", "Joint", "Both"])
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.axvline(1.5, color="white", lw=0.8)
    for i in range(values.shape[0]):
        for j in range(3):
            ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if values[i,j] < 55 else "black")
    fig.colorbar(image, ax=ax, label="Held-out rows (%)")
    ax.set_title("Input support in primary held-out populations")
    fig.savefig(output / "primary_support_heatmap.png", dpi=220)
    fig.savefig(output / "primary_support_heatmap.svg")
    plt.close(fig)


def make_category_plot(rows: list[dict], output: Path) -> None:
    primary = [r for r in rows if r["population"] in ("central_50km", "both_corridors", "PIG")]
    labels = [f'{r["experiment"]} {r["configuration"].split("_")[0]}' for r in primary]
    keys = ["both_fraction", "marginal_only_fraction", "joint_only_fraction", "neither_fraction"]
    data = np.asarray([[float(r[k]) for k in keys] for r in primary]) * 100
    fig, ax = plt.subplots(figsize=(10, max(8, 0.27 * len(primary))), constrained_layout=True)
    left = np.zeros(len(primary))
    colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    names = ["Both", "Marginal only", "Joint only", "Neither"]
    for column, (name, color) in enumerate(zip(names, colors)):
        ax.barh(np.arange(len(primary)), data[:, column], left=left, color=color, label=name)
        left += data[:, column]
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Held-out rows (%)")
    ax.set_title("Four prespecified point-support categories", pad=12)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.997))
    fig.savefig(output / "primary_support_categories.png", dpi=220)
    fig.savefig(output / "primary_support_categories.svg")
    plt.close(fig)


def make_distribution_heatmap(rows: list[dict], predictors: list[str], output: Path) -> None:
    primary_names = {"central_50km", "both_corridors", "PIG"}
    selected = [r for r in rows if r["population"] in primary_names and r["variable"] in predictors + ["reference_log_C"]]
    experiments = list(dict.fromkeys(r["experiment"] for r in selected))
    variables = predictors + ["reference_log_C"]
    lookup = {(r["experiment"], r["variable"]): float(r["heldout_inside_training_q01_q99_fraction"]) for r in selected}
    values = np.asarray([[lookup[(e, v)] for v in variables] for e in experiments]) * 100
    fig, ax = plt.subplots(figsize=(13, 5.7), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", vmin=70, vmax=100, cmap="magma")
    ax.set_xticks(range(len(variables)), variables, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(experiments)), experiments)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if values[i,j] < 86 else "black")
    fig.colorbar(image, ax=ax, label="Held-out rows within training q01–q99 (%)")
    ax.set_title("Training-versus-held-out univariate distribution coverage\nReference C is diagnostic only")
    fig.savefig(output / "primary_distribution_coverage.png", dpi=220)
    fig.savefig(output / "primary_distribution_coverage.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--reference-grid", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--selected-squares", required=True)
    parser.add_argument("--support-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    dataset_path = dataset_dir / "canonical_master_dataset.csv.gz"
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(dataset_manifest_path.read_text())
    if manifest["status"] != "complete" or sha256(dataset_path) != manifest["output_sha256"][dataset_path.name]:
        raise ValueError("Canonical dataset identity/status check failed")
    predictors = list(manifest["predictors"])
    needed = ["row_id", "x", "y", "reference_log_C", "common_eligible", "region_code",
              "square_test_id", "square_footprint_id"] + predictors
    frame = pd.read_csv(dataset_path, usecols=needed, low_memory=False)
    frame = frame.loc[frame["common_eligible"].astype(bool)].reset_index(drop=True)
    if len(frame) != int(manifest["common_eligible_count"]):
        raise AssertionError("Eligible row count mismatch")
    if not np.isfinite(frame[predictors + ["reference_log_C"]].to_numpy()).all():
        raise AssertionError("Non-finite predictor/target in eligible population")

    reference_path = Path(args.reference_grid).resolve()
    reference = np.load(reference_path, allow_pickle=False)
    eligible_grid = reference["eligible"].astype(bool)
    flat = np.flatnonzero(eligible_grid.ravel())
    grid_features = reference["features"].reshape((-1, reference["features"].shape[-1]))[flat].astype(float)
    grid_names = [str(x) for x in reference["feature_names"]]
    xx, yy = np.meshgrid(reference["x_grid"].astype(float), reference["y_grid"].astype(float))
    grid_coordinates = np.column_stack((xx.ravel()[flat], yy.ravel()[flat]))
    partition = np.load(Path(args.partition).resolve(), allow_pickle=False)
    partition_codes = partition["region_codes"].astype(np.int8)
    grid_regions = (
        partition_codes.ravel()[flat]
        if partition_codes.shape == eligible_grid.shape
        else partition_codes.ravel()
    )
    if len(grid_regions) != len(grid_coordinates):
        raise AssertionError("Reference partition/grid length mismatch")
    selected_rows = {r["square_id"]: r for r in csv.DictReader(Path(args.selected_squares).open(newline="", encoding="utf-8"))}
    support_evidence = json.loads(Path(args.support_evidence).read_text())["support_by_feature_configuration"]
    grid_index = {name: i for i, name in enumerate(grid_names)}

    transforms = {}
    for config, features in FEATURE_CONFIGURATIONS.items():
        old_name = config.replace("CFG0", "").replace("CFG", "")
        if old_name not in support_evidence:
            raise KeyError(f"Missing cutoff evidence for {config}: {old_name}")
        values = grid_features[:, [grid_index[name] for name in features]]
        transformer, pca, transformed = fit_reference(values)
        transforms[config] = {
            "features": features, "transformer": transformer, "pca": pca,
            "grid_transformed": transformed,
            "cutoff": float(support_evidence[old_name]["joint_q95_cutoff"]),
        }

    distribution_rows, marginal_rows, support_rows, population_rows = [], [], [], []
    label_payload = {}
    for spec in scenario_specs(frame):
        if spec["kind"] == "square":
            spec["selected_square"] = {k: float(v) if k.endswith("_m") else v for k, v in selected_rows[spec["id"]].items()}
        train = spec["train"]
        if np.any(train & np.logical_or.reduce(list(spec["eval"].values()))):
            raise AssertionError(f"Training/evaluation overlap: {spec['id']}")
        reference_train = reference_training_mask(spec, grid_coordinates, grid_regions)
        label_payload[spec["id"]] = {
            "row_index": np.flatnonzero(np.logical_or.reduce(list(spec["eval"].values()))).astype(np.int32),
        }
        primary_population = "central_50km" if spec["kind"] == "square" else ("both_corridors" if spec["id"] == "REG_INTER" else "PIG")
        for population, eval_mask in spec["eval"].items():
            population_rows.append({"experiment": spec["id"], "population": population,
                                    "training_rows": int(train.sum()), "heldout_rows": int(eval_mask.sum()),
                                    "primary_population": population == primary_population})
            for variable in predictors + ["reference_log_C"]:
                train_values = frame.loc[train, variable].to_numpy(float)
                eval_values = frame.loc[eval_mask, variable].to_numpy(float)
                tq = quantile_record(train_values); eq = quantile_record(eval_values)
                distribution_rows.extend([
                    {"experiment": spec["id"], "population": population, "sample": "training", "variable": variable, **tq},
                    {"experiment": spec["id"], "population": population, "sample": "heldout", "variable": variable, **eq},
                ])
                distribution_rows[-1]["heldout_inside_training_q01_q99_fraction"] = float(np.mean((eval_values >= tq["q01"]) & (eval_values <= tq["q99"])))
                distribution_rows[-2]["heldout_inside_training_q01_q99_fraction"] = ""

        for config in spec["configs"]:
            info = transforms[config]; features = info["features"]
            train_values = frame.loc[train, features].to_numpy(float)
            lower, upper = np.quantile(train_values, (0.01, 0.99), axis=0)
            neighbors = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean", n_jobs=-1)
            neighbors.fit(info["grid_transformed"][reference_train])
            full_eval = np.logical_or.reduce(list(spec["eval"].values()))
            eval_values = frame.loc[full_eval, features].to_numpy(float)
            transformed = info["pca"].transform(info["transformer"].transform(eval_values))
            distances = neighbors.kneighbors(transformed, return_distance=True)[0][:, 0]
            marginal_matrix = (eval_values >= lower) & (eval_values <= upper)
            marginal_pass = np.all(marginal_matrix, axis=1)
            joint_pass = distances <= info["cutoff"]
            categories = marginal_pass.astype(np.uint8) + 2 * joint_pass.astype(np.uint8)
            label_payload[spec["id"]][config] = categories
            full_indices = np.flatnonzero(full_eval)
            positions = np.full(len(frame), -1, dtype=np.int32)
            positions[full_indices] = np.arange(len(full_indices), dtype=np.int32)
            for population, eval_mask in spec["eval"].items():
                pos = positions[np.flatnonzero(eval_mask)]
                local_categories = categories[pos]
                local_marginal = marginal_matrix[pos]
                local_joint = joint_pass[pos]
                counts = np.bincount(local_categories, minlength=4)
                coverage = np.mean(local_marginal, axis=0)
                for feature, lo, hi, cov in zip(features, lower, upper, coverage):
                    marginal_rows.append({"experiment": spec["id"], "population": population,
                                          "configuration": config, "feature": feature,
                                          "training_q01": float(lo), "training_q99": float(hi),
                                          "heldout_coverage": float(cov)})
                support_rows.append({
                    "experiment": spec["id"], "population": population, "configuration": config,
                    "training_rows": int(train.sum()), "heldout_rows": int(eval_mask.sum()),
                    "reference_training_cells": int(reference_train.sum()),
                    "minimum_marginal_coverage": float(np.min(coverage)),
                    "limiting_feature": features[int(np.argmin(coverage))],
                    "joint_coverage": float(np.mean(local_joint)),
                    "both_fraction": float(counts[3] / counts.sum()),
                    "marginal_only_fraction": float(counts[1] / counts.sum()),
                    "joint_only_fraction": float(counts[2] / counts.sum()),
                    "neither_fraction": float(counts[0] / counts.sum()),
                    "population_passes_marginal_95": bool(np.all(coverage >= 0.95)),
                    "population_passes_joint_95": bool(np.mean(local_joint) >= 0.95),
                    "joint_cutoff": info["cutoff"], "pca_components": int(transformed.shape[1]),
                })

    write_csv(output / "population_summary.csv", population_rows)
    write_csv(output / "distribution_quantiles.csv", distribution_rows)
    write_csv(output / "marginal_support.csv", marginal_rows)
    write_csv(output / "support_categories.csv", support_rows)
    np.savez_compressed(output / "point_support_categories.npz", **{
        f"{experiment}__{key}": value for experiment, payload in label_payload.items() for key, value in payload.items()
    })
    make_support_heatmap(support_rows, output)
    make_category_plot(support_rows, output)
    heldout_distribution_rows = [r for r in distribution_rows if r["sample"] == "heldout"]
    make_distribution_heatmap(heldout_distribution_rows, predictors, output)

    methods = {
        "schema": "jog-heldout-distribution-diagnostics-v1",
        "status": "complete", "selection_reopened": False,
        "outcomes_used_for_support": [],
        "dataset_manifest_id": manifest["manifest_id"],
        "predictors": predictors, "training_target_diagnostic": "reference_log_C",
        "support": {
            "marginal": "each included predictor within actual eligible training-population q01-q99",
            "joint_transform": "frozen eligible 5 km sector grid rank-Gaussian transform and whitened PCA retaining >=99% variance",
            "joint_analogue_pool": "eligible 5 km reference cells belonging to the corresponding non-held-out training geography",
            "joint_cutoffs": {k: v["cutoff"] for k, v in transforms.items()},
            "categories": CATEGORY_NAMES.tolist(),
            "area_fraction": "row fraction equals area fraction because retained MEaSUREs raster cells have common area",
        },
        "reference_C_policy": "descriptive only; never used to define support, masks, locations, or eligibility",
        "experiments": [{"id": s["id"], "kind": s["kind"], "populations": list(s["eval"]), "configurations": s["configs"]} for s in scenario_specs(frame)],
    }
    (output / "methods.json").write_text(json.dumps(methods, indent=2) + "\n")
    outputs = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    result_manifest = {
        "schema": "jog-heldout-distribution-bundle-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest_id": manifest["manifest_id"], "dataset_sha256": sha256(dataset_path),
        "reference_grid_sha256": sha256(reference_path),
        "partition_sha256": sha256(Path(args.partition).resolve()),
        "selected_squares_sha256": sha256(Path(args.selected_squares).resolve()),
        "support_evidence_sha256": sha256(Path(args.support_evidence).resolve()),
        "output_sha256": outputs,
    }
    result_manifest["manifest_id"] = canonical_id(result_manifest)
    (output / "diagnostics_manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n")
    print(json.dumps({"manifest_id": result_manifest["manifest_id"], "eligible_rows": len(frame),
                      "population_rows": len(population_rows), "support_rows": len(support_rows),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
