"""Assess whether ten-member velocity spread diagnoses held-out velocity error.

This is a descriptive reliability diagnostic, not a calibrated uncertainty
analysis.  Each member differs through its reproducibly seeded training and
validation split and network fit.  The ten-member velocity spread therefore
measures only this source of algorithmic variability.

The primary analysis uses the ten independently held-out central squares.
Annuli and complete 130 km footprints are retained as nested diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from evaluate_forward_campaign import (
    build_observation_alignment,
    interpolate_velocity,
    model_registry,
    population_masks,
)
from forward_solve_campaign import build_object
from production_amundsen import manifest_identifier, sha256_file


CONFIG_LABELS = {
    "CFG01": "All ice",
    "CFG02": "Best ice",
    "CFG03": "All geophysical",
    "CFG04": "Best geophysical",
    "CFG05": "Best combined",
    "CFG06": "Combined + alignment",
}
CONFIG_COLORS = {
    "CFG01": "#4477AA",
    "CFG02": "#66CCEE",
    "CFG03": "#228833",
    "CFG04": "#CCBB44",
    "CFG05": "#EE6677",
    "CFG06": "#AA3377",
}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def finite_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x[mask], y[mask]).statistic)
    if method == "pearson":
        return float(pearsonr(x[mask], y[mask]).statistic)
    raise ValueError(method)


def quintile_summary(spread: np.ndarray, error: np.ndarray) -> tuple[list[float], list[int]]:
    # Stable ranks prevent repeated values from producing empty diagnostic bins.
    ranks = pd.Series(spread).rank(method="first").to_numpy()
    bins = np.asarray(pd.qcut(ranks, 5, labels=False), dtype=np.int8)
    means = [float(np.mean(error[bins == index])) for index in range(5)]
    counts = [int(np.sum(bins == index)) for index in range(5)]
    return means, counts


def population_row(ensemble: str, experiment: str, config: str, population: str,
                   spread: np.ndarray, error: np.ndarray) -> dict:
    qmeans, qcounts = quintile_summary(spread, error)
    mean_error = float(np.mean(error))
    return {
        "ensemble_id": ensemble,
        "experiment": experiment,
        "configuration": config,
        "population": population,
        "rows": int(len(error)),
        "velocity_spread_rms_m_per_a": float(np.sqrt(np.mean(spread ** 2))),
        "velocity_spread_mean_m_per_a": float(np.mean(spread)),
        "median_velocity_error_rmse_m_per_a": float(np.sqrt(np.mean(error ** 2))),
        "median_velocity_error_mae_m_per_a": mean_error,
        "pointwise_spearman_rho": finite_correlation(spread, error, "spearman"),
        "pointwise_pearson_r": finite_correlation(spread, error, "pearson"),
        "error_q5_over_q1": None if qmeans[0] == 0 else float(qmeans[4] / qmeans[0]),
        **{f"spread_quintile_{index + 1}_error_mae_m_per_a": qmeans[index] for index in range(5)},
        **{f"spread_quintile_{index + 1}_rows": qcounts[index] for index in range(5)},
        **{f"spread_quintile_{index + 1}_error_relative_to_population_mean":
           float(qmeans[index] / mean_error) for index in range(5)},
    }


def cluster_bootstrap_correlation(frame: pd.DataFrame, iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    squares = sorted(frame["experiment"].unique())
    values = []
    for _ in range(iterations):
        sampled = rng.choice(squares, size=len(squares), replace=True)
        blocks = []
        for occurrence, square in enumerate(sampled):
            block = frame.loc[frame["experiment"].eq(square)].copy()
            block["bootstrap_block"] = occurrence
            blocks.append(block)
        sample = pd.concat(blocks, ignore_index=True)
        values.append(finite_correlation(
            sample["velocity_spread_rms_m_per_a"].to_numpy(float),
            sample["median_velocity_error_rmse_m_per_a"].to_numpy(float),
            "spearman",
        ))
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "iterations": iterations,
        "seed": seed,
        "valid_iterations": int(len(values)),
        "spearman_rho_q025": float(np.quantile(values, 0.025)),
        "spearman_rho_median": float(np.median(values)),
        "spearman_rho_q975": float(np.quantile(values, 0.975)),
    }


def plot_population_scatter(primary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.3), sharex=True, sharey=True)
    for axis, config in zip(axes.flat, CONFIG_LABELS):
        local = primary.loc[primary["configuration"].eq(config)].sort_values("experiment")
        x = local["velocity_spread_rms_m_per_a"].to_numpy(float)
        y = local["median_velocity_error_rmse_m_per_a"].to_numpy(float)
        axis.scatter(x, y, s=48, color=CONFIG_COLORS[config], edgecolor="black", linewidth=.5)
        for _, row in local.iterrows():
            axis.annotate(row["experiment"].replace("SQ", ""),
                          (row["velocity_spread_rms_m_per_a"], row["median_velocity_error_rmse_m_per_a"]),
                          xytext=(4, 3), textcoords="offset points", fontsize=7)
        rho = finite_correlation(x, y, "spearman")
        axis.set_title(f"{CONFIG_LABELS[config]} ({config})\nSpearman ρ = {rho:.2f}", fontsize=10)
        axis.grid(alpha=.2, linewidth=.6)
        axis.set_xscale("log"); axis.set_yscale("log")
    for axis in axes[-1, :]:
        axis.set_xlabel("Ensemble velocity spread RMS (m a$^{-1}$)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Median-model velocity RMSE (m a$^{-1}$)")
    fig.suptitle("Does ten-member velocity spread identify difficult held-out squares?", fontsize=14)
    fig.text(.5, .015, "Central 50 km tests; square numbers label the ten independent spatial replicates",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .95))
    fig.savefig(output / "ensemble_spread_vs_error.png", dpi=300)
    fig.savefig(output / "ensemble_spread_vs_error.svg")
    plt.close(fig)


def plot_pointwise_diagnostics(primary: pd.DataFrame, output: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.1))
    positions = np.arange(1, 7)
    rng = np.random.default_rng(20260830)
    for position, config in zip(positions, CONFIG_LABELS):
        values = primary.loc[primary["configuration"].eq(config), "pointwise_spearman_rho"].to_numpy(float)
        jitter = rng.uniform(-.12, .12, size=len(values))
        left.scatter(position + jitter, values, s=32, color=CONFIG_COLORS[config], alpha=.85,
                     edgecolor="black", linewidth=.35)
        left.plot([position - .18, position + .18], [np.median(values)] * 2, color="black", linewidth=2)
    left.axhline(0, color="black", linewidth=.8)
    left.set_xticks(positions, [key.replace("CFG0", "C") for key in CONFIG_LABELS])
    left.set_ylabel("Pointwise Spearman ρ: spread vs error")
    left.set_xlabel("Feature configuration")
    left.set_title("Within-square association")
    left.grid(axis="y", alpha=.2)

    quintiles = np.arange(1, 6)
    for config in CONFIG_LABELS:
        local = primary.loc[primary["configuration"].eq(config)]
        columns = [f"spread_quintile_{q}_error_relative_to_population_mean" for q in quintiles]
        values = local[columns].to_numpy(float)
        mean = np.mean(values, axis=0)
        sem = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
        right.plot(quintiles, mean, marker="o", linewidth=1.8, color=CONFIG_COLORS[config],
                   label=f"{config}: {CONFIG_LABELS[config]}")
        right.fill_between(quintiles, mean - sem, mean + sem, color=CONFIG_COLORS[config], alpha=.12)
    right.axhline(1, color="black", linewidth=.8)
    right.set_xticks(quintiles)
    right.set_xlabel("Within-square ensemble-spread quintile")
    right.set_ylabel("Mean absolute error / square mean")
    right.set_title("Error enrichment in high-spread locations")
    right.grid(alpha=.2)
    right.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Ten-member spread is a diagnostic, not calibrated predictive uncertainty", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / "pointwise_spread_diagnostics.png", dpi=300)
    fig.savefig(output / "pointwise_spread_diagnostics.svg")
    plt.close(fig)


def run(args) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archives = output / "ensemble_archives"
    archives.mkdir(exist_ok=True)

    object_, _, _ = build_object(args.config.resolve(), args.repo_root.resolve(), args.adoption_record.resolve())
    frame, lookup = build_observation_alignment(object_, args.dataset.resolve())
    observed = frame[["observed_vx", "observed_vy"]].to_numpy(np.float64)

    registry = model_registry(args.campaign_root.resolve())
    square_records = [record for record in registry if record["ensemble_id"].startswith("SQ")]
    groups: dict[str, list[dict]] = {}
    for record in square_records:
        groups.setdefault(record["ensemble_id"], []).append(record)
    if len(groups) != 60:
        raise RuntimeError(f"Expected 60 square ensembles, found {len(groups)}")

    rows = []
    for number, (ensemble, records) in enumerate(sorted(groups.items()), start=1):
        experiment, config = ensemble.split("_")
        members = sorted((item for item in records if item["kind"] == "member"), key=lambda x: x["control_id"])
        medians = [item for item in records if item["kind"] == "median"]
        if len(members) != 10 or len(medians) != 1:
            raise RuntimeError(f"Incomplete ensemble {ensemble}")
        masks = population_masks(frame, experiment)
        full_mask = masks["full_130km"]
        local_observed = observed[full_mask]

        archive_path = archives / f"{ensemble}.npz"
        if archive_path.is_file():
            with np.load(archive_path, allow_pickle=False) as archive:
                spread = archive["velocity_spread"]
                error = archive["median_velocity_error"]
                central_local = archive["central"].astype(bool)
            if len(spread) != int(full_mask.sum()) or len(error) != len(spread):
                raise RuntimeError(f"Invalid resumable archive for {ensemble}")
        else:
            member_values = []
            for member in members:
                values = np.load(member["velocity_path"], allow_pickle=False)
                member_values.append(interpolate_velocity(object_, values, lookup)[full_mask])
            member_values = np.stack(member_values, axis=0)
            median_values = np.load(medians[0]["velocity_path"], allow_pickle=False)
            median_prediction = interpolate_velocity(object_, median_values, lookup)[full_mask]
            spread = np.sqrt(np.var(member_values[:, :, 0], axis=0, ddof=1)
                             + np.var(member_values[:, :, 1], axis=0, ddof=1))
            error = np.linalg.norm(median_prediction - local_observed, axis=1)
            central_local = masks["central_50km"][full_mask]
            atomic_npz(
                archive_path,
                row_id=frame.loc[full_mask, "row_id"].to_numpy(str),
                x=frame.loc[full_mask, "x"].to_numpy(np.float64),
                y=frame.loc[full_mask, "y"].to_numpy(np.float64),
                velocity_spread=spread,
                median_velocity_error=error,
                central=central_local,
            )
        annulus_local = ~central_local
        for population, mask in {
            "central_50km": central_local,
            "exclusion_annulus": annulus_local,
            "full_130km": np.ones(len(error), dtype=bool),
        }.items():
            rows.append(population_row(ensemble, experiment, config, population, spread[mask], error[mask]))
        print(json.dumps({"ensemble": ensemble, "completed": number, "total": len(groups)}), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(output / "ensemble_spread_population_metrics.csv", index=False)
    primary = table.loc[table["population"].eq("central_50km")].copy()
    plot_population_scatter(primary, output)
    plot_pointwise_diagnostics(primary, output)

    overall_rho = finite_correlation(
        primary["velocity_spread_rms_m_per_a"].to_numpy(float),
        primary["median_velocity_error_rmse_m_per_a"].to_numpy(float), "spearman")
    by_config = {}
    for config in CONFIG_LABELS:
        local = primary.loc[primary["configuration"].eq(config)]
        correlations = local["pointwise_spearman_rho"].to_numpy(float)
        by_config[config] = {
            "population_level_spearman_rho": finite_correlation(
                local["velocity_spread_rms_m_per_a"].to_numpy(float),
                local["median_velocity_error_rmse_m_per_a"].to_numpy(float), "spearman"),
            "median_within_square_pointwise_spearman_rho": float(np.median(correlations)),
            "positive_within_square_correlations": int(np.sum(correlations > 0)),
            "mean_error_q5_over_q1": float(np.mean(local["error_q5_over_q1"].to_numpy(float))),
        }
    summary = {
        "schema": "jog-ensemble-spread-diagnostic-summary-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_population": "ten central 50 km square holdouts; six repeated configurations per square",
        "spread_definition": "sqrt(sample variance(vx)+sample variance(vy)) across ten member Icepack velocities",
        "error_definition": "magnitude of median-C-control Icepack velocity minus MEaSUREs observed velocity",
        "interpretation_limit": "algorithmic member variability only; not calibrated predictive uncertainty",
        "population_level_spearman_rho_all_60_repeated_cases": overall_rho,
        "cluster_bootstrap_by_square": cluster_bootstrap_correlation(primary, 10000, 20260830),
        "median_within_case_pointwise_spearman_rho": float(np.median(primary["pointwise_spearman_rho"])),
        "positive_within_case_correlations": int(np.sum(primary["pointwise_spearman_rho"] > 0)),
        "cases": int(len(primary)),
        "by_configuration": by_config,
        "inputs": {
            "forward_evaluation_manifest_id": json.loads(
                (args.evaluation_root.resolve() / "evaluation_manifest.json").read_text(encoding="utf-8")
            )["manifest_id"],
            "dataset_sha256": sha256_file(args.dataset.resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    outputs = {}
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "spread_diagnostic_manifest.json"):
        outputs[path.relative_to(output).as_posix()] = sha256_file(path)
    manifest = {
        **summary,
        "output_sha256": outputs,
    }
    manifest["manifest_id"] = manifest_identifier(manifest)
    atomic_json(output / "spread_diagnostic_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adoption-record", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
