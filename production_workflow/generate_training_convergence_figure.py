#!/usr/bin/env python3
"""Generate an audited convergence summary from the 600 square-model histories."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ROOT
    / "cuda_results"
    / "JOG_PRODUCTION_RESULTS_20260828"
    / "sha256-json-v1-b9ce7f5acd43e094cca85e2c0f7463029c5d73b7a7546bd2c80a2a43b1dc6f24"
    / "runs"
)
OUTPUT = ROOT / "production_workflow" / "final_figures_20260830_a"
PATTERN = re.compile(r"SQ\d{2}_(CFG\d{2})_M\d{2}$")
NORMALIZED_PROGRESS = np.linspace(0.0, 1.0, 241)


def read_run(
    run_dir: Path,
) -> tuple[str, int, int, float, np.ndarray, np.ndarray, float, float]:
    match = PATTERN.fullmatch(run_dir.name)
    if match is None:
        raise ValueError(run_dir.name)
    summary = json.loads((run_dir / "training_summary.json").read_text())
    best = int(summary["best_epoch_one_based"])
    completed = int(summary["epochs_completed"])
    epochs: list[int] = []
    train: list[float] = []
    validation: list[float] = []
    with (run_dir / "history.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            epochs.append(int(row["epoch"]))
            train.append(float(row["data_mse"]))
            validation.append(float(row["val_data_mse"]))
    if len(epochs) != completed or completed < 2:
        raise RuntimeError(f"Incomplete history for {run_dir.name}")
    progress = (np.asarray(epochs, dtype=float) - 1.0) / (completed - 1.0)
    train_array = np.asarray(train)
    validation_array = np.asarray(validation)
    best_progress = (best - 1.0) / (completed - 1.0)
    return (
        match.group(1),
        best,
        completed,
        best_progress,
        np.interp(NORMALIZED_PROGRESS, progress, train_array),
        np.interp(NORMALIZED_PROGRESS, progress, validation_array),
        float(validation_array[0]),
        float(validation_array[best - 1]),
    )


def quantiles(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanquantile(matrix, 0.10, axis=0),
        np.nanmedian(matrix, axis=0),
        np.nanquantile(matrix, 0.90, axis=0),
    )


def main() -> None:
    grouped: dict[
        str,
        list[tuple[int, int, float, np.ndarray, np.ndarray, float, float]],
    ] = {
        f"CFG{i:02d}": [] for i in range(1, 7)
    }
    for run_dir in sorted(RUNS.iterdir()):
        if run_dir.is_dir() and PATTERN.fullmatch(run_dir.name):
            cfg, best, completed, best_progress, train, validation, initial_val, best_val = read_run(run_dir)
            grouped[cfg].append(
                (best, completed, best_progress, train, validation, initial_val, best_val)
            )

    if any(len(runs) != 100 for runs in grouped.values()):
        raise RuntimeError({cfg: len(runs) for cfg, runs in grouped.items()})

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.3), sharex=True, sharey=True)
    train_color = "#2166ac"
    validation_color = "#b2182b"

    manifest: dict[str, object] = {
        "schema": "jog-training-convergence-figure-v1",
        "population": "600 primary-square MLP runs",
        "alignment": "each complete history linearly mapped from first epoch (0) to early-stopping termination (1)",
        "normalized_progress_grid_points": len(NORMALIZED_PROGRESS),
        "configurations": {},
    }

    for ax, (cfg, runs) in zip(axes.flat, grouped.items()):
        train = np.vstack([run[3] for run in runs])
        validation = np.vstack([run[4] for run in runs])
        best_epochs = []
        completed_epochs = []
        best_progresses = []
        initial_validation = []
        best_validation = []
        for best, completed, best_progress, _, _, initial_val, best_val in runs:
            best_epochs.append(best)
            completed_epochs.append(completed)
            best_progresses.append(best_progress)
            initial_validation.append(initial_val)
            best_validation.append(best_val)

        t10, t50, t90 = quantiles(train)
        v10, v50, v90 = quantiles(validation)
        ax.fill_between(NORMALIZED_PROGRESS, t10, t90, color=train_color, alpha=0.16, linewidth=0)
        ax.fill_between(
            NORMALIZED_PROGRESS, v10, v90, color=validation_color, alpha=0.16, linewidth=0
        )
        ax.plot(NORMALIZED_PROGRESS, t50, color=train_color, linewidth=1.8, label="Training")
        ax.plot(
            NORMALIZED_PROGRESS,
            v50,
            color=validation_color,
            linewidth=1.8,
            label="Validation",
        )
        median_best_progress = float(np.median(best_progresses))
        ax.axvline(
            median_best_progress,
            color="0.2",
            linewidth=1.0,
            linestyle="--",
        )
        ax.set_yscale("log")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(5e-4, 8e-1)
        ax.grid(True, which="major", color="0.88", linewidth=0.6)
        q_best = np.quantile(best_epochs, [0.25, 0.5, 0.75])
        q_stop = np.quantile(completed_epochs, [0.25, 0.5, 0.75])
        ax.set_title(
            f"{cfg}: best epoch {q_best[1]:.0f} "
            f"[{q_best[0]:.0f}–{q_best[2]:.0f}]"
        )
        manifest["configurations"][cfg] = {
            "runs": len(runs),
            "best_epoch_quartiles": q_best.tolist(),
            "completed_epoch_quartiles": q_stop.tolist(),
            "median_best_checkpoint_progress": median_best_progress,
            "median_initial_validation_data_mse": float(np.median(initial_validation)),
            "median_best_validation_data_mse": float(np.median(best_validation)),
            "median_validation_decline_factor": float(
                np.median(np.asarray(initial_validation) / np.asarray(best_validation))
            ),
        }

    for ax in axes[:, 0]:
        ax.set_ylabel("Scaled data MSE")
    for ax in axes[-1, :]:
        ax.set_xlabel("Fraction of completed training")
    axes[0, 0].legend(
        loc="upper right",
        frameon=True,
        title="Median; shading = 10th–90th percentile",
        title_fontsize=8,
    )
    fig.suptitle(
        "Training histories for the ten withheld squares "
        "(100 MLPs per configuration)",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUTPUT / f"appendix_training_convergence.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    (OUTPUT / "appendix_training_convergence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
