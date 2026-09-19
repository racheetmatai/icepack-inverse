#!/usr/bin/env python3
"""Appendix figure: can the predictors indicate where transfer succeeds?

Two panels, one per label, each showing the classifier AUC for every
leave-one-square-out test (coloured by configuration) and the PIG CFG02
transfer test (star), for three feature sets. Reads the outputs of
analyze_transfer_predictability.py.

Both panels share one fixed canvas and are saved without tight bounding
boxes, so the two subfigures render at identical size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_revision_figures_and_tables as base  # noqa: E402

FEATURE_ORDER = ["speed_only", "predictors_only", "predictors_and_speed"]
FEATURE_LABELS = ["Observed\nspeed only", "Twelve\npredictors", "Predictors\nand speed"]
PANELS = {"improves": "a", "halves": "b"}
CANVAS = (5.2, 4.4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(label: str, folds: pd.DataFrame, pig: pd.DataFrame, show_legend: bool) -> plt.Figure:
    fig = plt.figure(figsize=CANVAS)
    ax = fig.add_axes([0.15, 0.17, 0.82, 0.78])
    rng = np.random.default_rng(7)
    usable = folds[(folds.label == label) & ~folds.degenerate]
    for i, feature in enumerate(FEATURE_ORDER):
        sub = usable[usable.features == feature]
        for config, colour in base.CONFIG_COLORS.items():
            values = sub[sub.configuration == config].auc.to_numpy()
            x = i + rng.uniform(-0.22, 0.22, size=len(values))
            ax.scatter(x, values, s=16, color=colour, alpha=0.85, linewidths=0, zorder=3)
        median = float(sub.auc.median())
        ax.plot([i - 0.3, i + 0.3], [median, median], color="0.1", lw=2.0, zorder=4)
        pig_auc = pig[(pig.label == label) & (pig.features == feature)].auc
        if len(pig_auc) and np.isfinite(pig_auc.iloc[0]):
            ax.scatter([i + 0.36], [pig_auc.iloc[0]], marker="*", s=130,
                       color="0.1", edgecolor="white", linewidths=0.6, zorder=5)
    ax.axhline(0.5, color="0.45", lw=1.0, ls="--", zorder=1)
    ax.text(2.93, 0.51, "No skill", va="bottom", ha="right", fontsize=9, color="0.35")
    ax.set_xticks(range(len(FEATURE_ORDER)), FEATURE_LABELS)
    ax.set_xlim(-0.55, 2.95)
    ax.set_ylim(0.0, 1.17)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Classifier AUC")
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        handles = [Line2D([], [], marker="o", ls="", color=c, markersize=5, label=k)
                   for k, c in base.CONFIG_COLORS.items()]
        handles += [Line2D([], [], marker="*", ls="", color="0.1", markersize=10, label="PIG"),
                    Line2D([], [], color="0.1", lw=2.0, label="Median")]
        ax.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                  fontsize=8.5, handletextpad=0.3, columnspacing=0.9,
                  bbox_to_anchor=(0.5, 1.02))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="output directory of analyze_transfer_predictability.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base.style()
    folds = pd.read_csv(args.results / "square_folds.csv")
    pig = pd.read_csv(args.results / "pig_transfer.csv")

    outputs = {}
    for label, letter in PANELS.items():
        fig = draw(label, folds, pig, show_legend=(letter == "a"))
        for suffix in ("pdf", "png"):
            path = args.output / f"figure_appendix_transfer_predictability_{letter}.{suffix}"
            fig.savefig(path, dpi=300)
            outputs[path.name] = sha256(path)
        plt.close(fig)
    (args.output / "figure_appendix_transfer_predictability_manifest.json").write_text(
        json.dumps({"schema": "jog-transfer-predictability-figure-v1",
                    "source": {p: sha256(args.results / p)
                               for p in ("square_folds.csv", "pig_transfer.csv")},
                    "outputs": outputs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
