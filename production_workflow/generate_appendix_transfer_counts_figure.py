#!/usr/bin/env python3
"""Appendix figure for "Predicting where transfer succeeds", count version.

Each point is one held-out test: how many of the held-out points the classifier
labels correctly, minus how many would be correct by simply predicting whichever
outcome is more common in that region. Zero means no better than that rule.
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

WORKFLOW = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKFLOW))
import generate_revision_figures_and_tables as base  # noqa: E402

FEATURE_ORDER = ["speed_only", "predictors_only", "predictors_and_speed"]
FEATURE_LABELS = ["Observed\nspeed only", "Twelve\npredictors", "Predictors\nand speed"]
PANELS = {"improves": "a", "halves": "b"}
CANVAS = (5.2, 4.4)
MODEL = "published"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(label: str, folds: pd.DataFrame, pig: pd.DataFrame, show_legend: bool) -> plt.Figure:
    fig = plt.figure(figsize=CANVAS)
    ax = fig.add_axes([0.165, 0.235, 0.805, 0.73])
    rng = np.random.default_rng(7)
    usable = folds[(folds.label == label) & folds.correct.notna() & (folds.model == MODEL)]
    for i, feature in enumerate(FEATURE_ORDER):
        sub = usable[usable.features == feature].copy()
        sub["margin"] = 100.0 * (sub.accuracy - sub.majority_rule_accuracy)
        for config, colour in base.CONFIG_COLORS.items():
            values = sub[sub.configuration == config].margin.to_numpy()
            x = i + rng.uniform(-0.22, 0.22, size=len(values))
            ax.scatter(x, values, s=16, color=colour, alpha=0.85, linewidths=0, zorder=3)
        median = float(sub.margin.median())
        ax.plot([i - 0.3, i + 0.3], [median, median], color="0.1", lw=2.0, zorder=4)
        row = pig[(pig.label == label) & (pig.features == feature) & (pig.model == MODEL)]
        if len(row):
            margin = 100.0 * float(row.accuracy.iloc[0] - row.majority_rule_accuracy.iloc[0])
            ax.scatter([i + 0.36], [margin], marker="*", s=130, color="0.1",
                       edgecolor="white", linewidths=0.6, zorder=5)
    ax.axhline(0.0, color="0.45", lw=1.0, ls="--", zorder=1)
    ax.text(-0.5, 2.0, "Same as guessing the more common outcome",
            va="bottom", ha="left", fontsize=8.5, color="0.35")
    ax.set_xticks(range(len(FEATURE_ORDER)), FEATURE_LABELS)
    ax.set_xlim(-0.55, 2.95)
    ax.set_ylim(-95, 30)
    ax.set_ylabel("Correct labels minus the more common\noutcome (percentage points)")
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        handles = [Line2D([], [], marker="o", ls="", color=c, markersize=5, label=k)
                   for k, c in base.CONFIG_COLORS.items()]
        handles += [Line2D([], [], marker="*", ls="", color="0.1", markersize=10, label="PIG"),
                    Line2D([], [], color="0.1", lw=2.0, label="Median")]
        ax.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                  fontsize=8.5, handletextpad=0.3, columnspacing=0.9,
                  bbox_to_anchor=(0.5, -0.13))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base.style()
    folds = pd.read_csv(args.results / "counts_folds.csv")
    pig = pd.read_csv(args.results / "counts_pig.csv")

    outputs = {}
    for label, letter in PANELS.items():
        fig = draw(label, folds, pig, show_legend=(letter == "a"))
        for suffix in ("pdf", "png"):
            path = args.output / f"figure_appendix_transfer_predictability_{letter}.{suffix}"
            fig.savefig(path, dpi=300)
            outputs[path.name] = sha256(path)
        plt.close(fig)
    (args.output / "figure_appendix_transfer_counts_manifest.json").write_text(
        json.dumps({"schema": "jog-transfer-predictability-counts-figure-v1",
                    "model": MODEL,
                    "source": {p: sha256(args.results / p)
                               for p in ("counts_folds.csv", "counts_pig.csv")},
                    "outputs": outputs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
