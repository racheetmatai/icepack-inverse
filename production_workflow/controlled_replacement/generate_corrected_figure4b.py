#!/usr/bin/env python3
"""Regenerate Figure 4b from corrected controlled-replacement metrics."""

from pathlib import Path
import argparse
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("metrics", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

configs = [f"CFG{i:02d}" for i in range(1, 7)]
squares = [f"SQ{i:02d}" for i in range(1, 11)]
data = pd.read_csv(args.metrics)
data = data.loc[
    data["scenario"].eq("controlled_original_measures")
    & data["experiment"].isin(squares)
]
if len(data) != 60 or data.duplicated(["experiment", "configuration"]).any():
    raise RuntimeError("Corrected square metric population is incomplete")
matrix = data.pivot(index="experiment", columns="configuration", values="relative_rmse").reindex(
    index=squares, columns=configs
)
if matrix.isna().any().any():
    raise RuntimeError("Corrected relative-RMSE matrix is incomplete")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(4.7, 5.0), constrained_layout=True)
norm = TwoSlopeNorm(vcenter=1.0, vmin=0.2, vmax=2.4)
image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", norm=norm)
for i in range(10):
    for j in range(6):
        value = matrix.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.0,
                color="white" if value < 0.42 or value > 1.65 else "black")
        if value >= 1:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#C51B7D", lw=1.25))
ax.set_xticks(range(6), configs, rotation=45, ha="right")
ax.set_yticks(range(10), squares)
ax.set_ylabel("Independent held-out square")
colorbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
colorbar.set_label(r"RMSE$_{ML}$ / RMSE$_{uniform\ C}$")
colorbar.ax.axhline(1.0, color="black", lw=0.7)
args.output.mkdir(parents=True, exist_ok=True)
for suffix in ("pdf", "png"):
    fig.savefig(args.output / f"figure4b_relative_velocity_skill.{suffix}", dpi=300,
                bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

record = {
    "schema": "jog-controlled-replacement-figure4b-v1",
    "status": "complete",
    "input_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
    "improvement_count_by_configuration": {
        config: int((matrix[config] < 1.0).sum()) for config in configs
    },
}
(args.output / "figure4b_corrected_manifest.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(record, indent=2, sort_keys=True))
