"""Plot diagnostic relative-error and population-normalized P_exp contributions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np


CONFIGS = {"CFG02": "Best ice predictors", "CFG01": "All ice predictors",
           "CFG04": "Best geophysical predictors"}


def edges(values):
    values = np.asarray(values, float); step = float(np.median(np.diff(values)))
    return np.r_[values - step / 2, values[-1] + step / 2]


def grid_chunk(x, y, z):
    xx = np.unique(x); yy = np.unique(y); grid = np.full((len(yy), len(xx)), np.nan)
    grid[np.searchsorted(yy, y), np.searchsorted(xx, x)] = z
    return edges(xx / 1000), edges(yy / 1000), grid


def pexp_contribution(archive, mask):
    base = archive["uniform_squared_error"][mask]
    model = archive["model_squared_error"][mask]
    central = archive["central"][mask].astype(bool)
    values = np.empty_like(base)
    for region in (central, ~central):
        denominator = float(np.mean(base[region]))
        if denominator <= 1e-12:
            values[region] = np.nan
        else:
            values[region] = 100.0 * (base[region] - model[region]) / denominator
    return values


def draw(args, config, label, metric):
    archive = np.load(args.data / f"{config}_ten_square_footprint_errors.npz")
    with args.squares.open(newline="", encoding="utf-8") as stream:
        squares = list(csv.DictReader(stream))
    support = np.load(args.support)
    outline = json.loads(args.outline.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8.2, 9.2), constrained_layout=True)
    eligible = np.ma.masked_where(~support["eligible"].astype(bool), support["eligible"].astype(float))
    ax.pcolormesh(edges(support["x_grid"] / 1000), edges(support["y_grid"] / 1000), eligible,
                  cmap="Greys", vmin=0, vmax=2.8, alpha=0.24, zorder=0)

    if metric == "percent_error":
        norm, cmap = LogNorm(vmin=1, vmax=500), "viridis"
        color_label = "Relative vector velocity error (%; log scale)"
        title_note = "Diagnostic only: relative error is unstable where observed speed is small"
    else:
        norm, cmap = SymLogNorm(linthresh=10, linscale=1, vmin=-1000, vmax=1000, base=10), "RdBu"
        color_label = r"Local contribution to population $P_{exp}$ (%)"
        title_note = r"Mean within each central/annulus population equals its $P_{exp}$"

    image = None
    outline_effect = [pe.Stroke(linewidth=2.3, foreground="black"), pe.Normal()]
    for number, row in enumerate(squares, start=1):
        mask = archive["square_number"] == number
        if metric == "percent_error":
            denominator = archive["observed_speed"][mask]
            values = np.divide(100.0 * archive["error_magnitude"][mask], denominator,
                               out=np.full(mask.sum(), np.inf), where=denominator > 0)
        else:
            values = pexp_contribution(archive, mask)
        xe, ye, zz = grid_chunk(archive["x"][mask], archive["y"][mask], values)
        image = ax.pcolormesh(xe, ye, zz, cmap=cmap, norm=norm, rasterized=True, zorder=2)
        outer_x = np.array([row["footprint_xmin_m"], row["footprint_xmax_m"], row["footprint_xmax_m"],
                            row["footprint_xmin_m"], row["footprint_xmin_m"]], float) / 1000
        outer_y = np.array([row["footprint_ymin_m"], row["footprint_ymin_m"], row["footprint_ymax_m"],
                            row["footprint_ymax_m"], row["footprint_ymin_m"]], float) / 1000
        inner_x = np.array([row["test_xmin_m"], row["test_xmax_m"], row["test_xmax_m"],
                            row["test_xmin_m"], row["test_xmin_m"]], float) / 1000
        inner_y = np.array([row["test_ymin_m"], row["test_ymin_m"], row["test_ymax_m"],
                            row["test_ymax_m"], row["test_ymin_m"]], float) / 1000
        ax.plot(outer_x, outer_y, color="white", lw=1, ls=(0, (4, 3)), zorder=4, path_effects=outline_effect)
        ax.plot(inner_x, inner_y, color="white", lw=1.2, zorder=5, path_effects=outline_effect)
        text = ax.text(float(row["center_x_m"]) / 1000, float(row["center_y_m"]) / 1000,
                       row["square_id"].replace("SQ", ""), ha="center", va="center",
                       color="white", fontsize=7.5, zorder=6)
        text.set_path_effects([pe.Stroke(linewidth=2, foreground="black"), pe.Normal()])

    for feature in outline["features"]:
        geom = feature["geometry"]
        for line in geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]:
            line = np.asarray(line, float) / 1000
            ax.plot(line[:, 0], line[:, 1], color="black", lw=.7, zorder=3)

    cb = fig.colorbar(image, ax=ax, fraction=.046, pad=.025, extend="both")
    cb.set_label(color_label)
    if metric == "percent_error":
        cb.set_ticks([1, 10, 100, 500]); cb.set_ticklabels(["1", "10", "100", "500"])
    else:
        cb.set_ticks([-1000, -100, -10, 0, 10, 100, 1000])
    handles = [
        Line2D([0], [0], color="black", lw=3, ls=(0, (4, 3)),
               path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()],
               label="130 km held-out footprint"),
        Line2D([0], [0], color="black", lw=3,
               path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()],
               label="Central 50 km test square"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=.92, fontsize=8)
    heading = "Relative velocity error" if metric == "percent_error" else r"Spatial $P_{exp}$ contribution"
    ax.set_title(f"{heading}: {label} ({config})\n{title_note}")
    ax.set_xlabel("EPSG:3031 easting (km)"); ax.set_ylabel("EPSG:3031 northing (km)")
    ax.set_xlim(-1750, -1050); ax.set_ylim(-800, 35); ax.set_aspect("equal")
    destination = args.output / f"amundsen-{metric.replace('_','-')}-{config.lower()}.png"
    fig.savefig(destination, dpi=300, facecolor="white"); plt.close(fig)
    print(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--squares", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--outline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    for config, label in CONFIGS.items():
        for metric in ("percent_error", "pexp_contribution"):
            draw(args, config, label, metric)


if __name__ == "__main__":
    main()
