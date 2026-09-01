"""Make publication-style sector mosaics for the three best configurations."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "production_workflow/gate4_square_footprint_error_maps_20260829_a"
SQUARES = ROOT / "production_workflow/frozen_design/selected_squares.csv"
SUPPORT = ROOT / "production_workflow/frozen_design/amundsen_input_support_grid_5km.npz"
OUTLINE = ROOT / "tmp/amundsen_v1.geojson"
SUMMARY = ROOT / "production_workflow/gate4_forward_reporting_20260829_b/square_equal_weight_summary.csv"

CONFIGS = {
    "CFG02": "Best ice predictors",
    "CFG01": "All ice predictors",
    "CFG04": "Best geophysical predictors",
}


def edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    step = float(np.median(np.diff(values)))
    return np.r_[values - step / 2, values[-1] + step / 2]


def grid_chunk(x, y, z):
    xx = np.unique(x); yy = np.unique(y)
    grid = np.full((len(yy), len(xx)), np.nan)
    xi = np.searchsorted(xx, x); yi = np.searchsorted(yy, y)
    grid[yi, xi] = z
    return edges(xx / 1000), edges(yy / 1000), grid


def draw(config: str, label: str, rmse: float, output: Path) -> Path:
    archive = np.load(DATA / f"{config}_ten_square_footprint_errors.npz")
    with SQUARES.open(newline="", encoding="utf-8") as stream:
        squares = list(csv.DictReader(stream))
    support = np.load(SUPPORT)
    outline = json.loads(OUTLINE.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8.2, 9.2), constrained_layout=True)
    eligible = np.ma.masked_where(~support["eligible"].astype(bool), support["eligible"].astype(float))
    ax.pcolormesh(edges(support["x_grid"] / 1000), edges(support["y_grid"] / 1000), eligible,
                  cmap="Greys", vmin=0, vmax=2.8, shading="flat", alpha=0.24, zorder=0)

    norm = LogNorm(vmin=1, vmax=2000, clip=False)
    image = None
    for number, row in enumerate(squares, start=1):
        mask = archive["square_number"] == number
        xe, ye, zz = grid_chunk(archive["x"][mask], archive["y"][mask], archive["error_magnitude"][mask])
        image = ax.pcolormesh(xe, ye, zz, cmap="magma", norm=norm, shading="flat", rasterized=True, zorder=2)

        outer_x = np.array([row["footprint_xmin_m"], row["footprint_xmax_m"],
                            row["footprint_xmax_m"], row["footprint_xmin_m"], row["footprint_xmin_m"]], float) / 1000
        outer_y = np.array([row["footprint_ymin_m"], row["footprint_ymin_m"],
                            row["footprint_ymax_m"], row["footprint_ymax_m"], row["footprint_ymin_m"]], float) / 1000
        inner_x = np.array([row["test_xmin_m"], row["test_xmax_m"], row["test_xmax_m"],
                            row["test_xmin_m"], row["test_xmin_m"]], float) / 1000
        inner_y = np.array([row["test_ymin_m"], row["test_ymin_m"], row["test_ymax_m"],
                            row["test_ymax_m"], row["test_ymin_m"]], float) / 1000
        outline_effect = [pe.Stroke(linewidth=2.3, foreground="black"), pe.Normal()]
        ax.plot(outer_x, outer_y, color="white", lw=1.0, ls=(0, (4, 3)), zorder=4,
                path_effects=outline_effect)
        ax.plot(inner_x, inner_y, color="white", lw=1.2, zorder=5, path_effects=outline_effect)
        text = ax.text(float(row["center_x_m"]) / 1000, float(row["center_y_m"]) / 1000,
                       row["square_id"].replace("SQ", ""), ha="center", va="center",
                       color="white", fontsize=7.5, weight="normal", zorder=6)
        text.set_path_effects([pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()])

    for feature in outline["features"]:
        geometry = feature["geometry"]
        lines = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
        for line in lines:
            line = np.asarray(line, dtype=float) / 1000
            ax.plot(line[:, 0], line[:, 1], color="black", lw=0.7, zorder=3)

    cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025, extend="both")
    cb.set_label(r"Vector velocity error magnitude (m a$^{-1}$; log scale)")
    cb.set_ticks([1, 10, 100, 1000, 2000])
    cb.set_ticklabels(["1", "10", "100", "1000", "2000"])
    legend = [
        Line2D([0], [0], color="black", lw=3.0, ls=(0, (4, 3)),
               path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()],
               label="130 km held-out footprint"),
        Line2D([0], [0], color="black", lw=3.0,
               path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()],
               label="Central 50 km test square"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=True, framealpha=0.92, fontsize=8)
    ax.set_title(f"{label} ({config})\nTen held-out spatial tests; median central-square RMSE = {rmse:.1f} m a$^{{-1}}$")
    ax.set_xlabel("EPSG:3031 easting (km)")
    ax.set_ylabel("EPSG:3031 northing (km)")
    ax.set_xlim(-1750, -1050); ax.set_ylim(-800, 35); ax.set_aspect("equal")
    ax.tick_params(direction="out")

    destination = output / f"amundsen-ten-square-errors-{config.lower()}.png"
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY).set_index("configuration")
    for config, label in CONFIGS.items():
        path = draw(config, label, float(summary.loc[config, "vector_RMSE_median_m_per_a"]), args.output)
        print(path)


if __name__ == "__main__":
    main()
