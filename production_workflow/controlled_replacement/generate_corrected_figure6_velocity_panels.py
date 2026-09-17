#!/usr/bin/env python3
"""Regenerate only the affected PIG velocity panels from corrected fields."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.lines import Line2D
import numpy as np


WORKFLOW = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKFLOW))
import generate_revision_figures_and_tables as base


HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("JOG_CONTROLLED_MAP_FIELDS", HERE / "map_fields")) / "REG_PIG_CFG02_controlled_fields.npz"
OUT = Path(os.environ.get("JOG_CONTROLLED_FIGURES", HERE / "figures"))
INVERSION_LEVEL = 100.0
INVERSION_COLOR = "#29D8E6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_fields() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(SOURCE, allow_pickle=True) as data:
        x = data["x"].astype(float)
        y = data["y"].astype(float)
        values = {
            "observed_speed": data["observed_speed"].astype(float),
            "model_error": data["model_error"].astype(float),
            "error_difference": data["model_error"].astype(float) - data["uniform_error"].astype(float),
            "inversion_error": data["inversion_error"].astype(float),
        }
    xc = np.unique(x)
    yc = np.unique(y)
    if not np.allclose(np.diff(xc), 450.0) or not np.allclose(np.diff(yc), 450.0):
        raise RuntimeError("Corrected PIG fields are not on the expected 450 m grid")
    xi = np.searchsorted(xc, x)
    yi = np.searchsorted(yc, y)
    linear = yi * len(xc) + xi
    if len(np.unique(linear)) != len(linear):
        raise RuntimeError("Duplicate PIG coordinates")
    grids = {name: np.full((len(yc), len(xc)), np.nan) for name in values}
    for name, value in values.items():
        grids[name][yi, xi] = value
    return base.center_edges(xc / 1000.0), base.center_edges(yc / 1000.0), grids


def finish(ax, x_edges: np.ndarray, y_edges: np.ndarray, grids: dict[str, np.ndarray], *,
           contour_color: str, inversion: bool) -> None:
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    ax.contour(
        xc, yc, grids["observed_speed"], levels=[100, 500, 1000],
        colors=contour_color, linewidths=[0.65, 0.85, 1.05],
        linestyles=[":", "--", "-"], zorder=9,
    )
    if inversion:
        ax.contour(xc, yc, grids["inversion_error"], levels=[INVERSION_LEVEL],
                   colors="0.08", linewidths=1.35, zorder=12)
        ax.contour(xc, yc, grids["inversion_error"], levels=[INVERSION_LEVEL],
                   colors=INVERSION_COLOR, linewidths=0.72, zorder=13)
        handle = Line2D(
            [0], [0], color=INVERSION_COLOR, lw=1.1,
            path_effects=[pe.Stroke(linewidth=2.0, foreground="0.08"), pe.Normal()],
        )
        ax.legend([handle], [r"Inversion: 100 m a$^{-1}$"], loc="lower left",
                  bbox_to_anchor=(0.018, 0.018), frameon=True, fancybox=False,
                  edgecolor="0.45", handlelength=1.8, borderpad=0.25, fontsize=8.6)
    ax.set_xlim(x_edges[0] - 5, x_edges[-1] + 5)
    ax.set_ylim(y_edges[0] - 5, y_edges[-1] + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("Polar stereographic x (km)")
    ax.set_ylabel("Polar stereographic y (km)")
    ax.tick_params(direction="out", length=2.5)
    base.add_antarctica_locator(ax, base.region_context()["outline"])


def draw(kind: str, x_edges: np.ndarray, y_edges: np.ndarray,
         grids: dict[str, np.ndarray]) -> Path:
    fig, ax = plt.subplots(figsize=(5.25, 4.55), constrained_layout=True)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    if kind == "absolute":
        image = ax.imshow(
            np.ma.masked_invalid(grids["model_error"]), origin="lower", extent=extent,
            cmap="inferno", norm=LogNorm(vmin=1, vmax=2000), interpolation="bilinear",
            interpolation_stage="rgba", rasterized=True, zorder=2,
        )
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", shrink=0.84,
                            pad=0.10, aspect=28, extend="both")
        cbar.set_label(r"Vector velocity error (m a$^{-1}$; log scale)")
        cbar.set_ticks([1, 10, 100, 1000])
        cbar.set_ticklabels(["1", "10", "100", "1000"])
        finish(ax, x_edges, y_edges, grids, contour_color="white", inversion=True)
        name = "figure6b_pig_cfg02_velocity_error"
    elif kind == "difference":
        image = ax.imshow(
            np.ma.masked_invalid(grids["error_difference"]), origin="lower", extent=extent,
            cmap="RdBu_r", norm=SymLogNorm(linthresh=10, linscale=1, vmin=-2000,
                                             vmax=2000, base=10), interpolation="bilinear",
            interpolation_stage="rgba", rasterized=True, zorder=2,
        )
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", shrink=0.84,
                            pad=0.10, aspect=28, extend="both")
        cbar.set_label("ML - uniform-$C$ error " r"(m a$^{-1}$; symlog)")
        cbar.set_ticks([-1000, -100, -10, 0, 10, 100, 1000])
        cbar.set_ticklabels(["-1000", "-100", "-10", "0", "10", "100", "1000"])
        finish(ax, x_edges, y_edges, grids, contour_color="0.35", inversion=False)
        name = "figure6c_pig_cfg02_uniform_comparison"
    else:
        raise ValueError(kind)
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return pdf


def main() -> None:
    base.style()
    plt.rcParams.update({"axes.labelsize": 12.5, "xtick.labelsize": 10.5,
                         "ytick.labelsize": 10.5})
    x_edges, y_edges, grids = grid_fields()
    outputs = [draw(kind, x_edges, y_edges, grids) for kind in ("absolute", "difference")]
    record = {
        "schema": "jog-controlled-replacement-figure6-velocity-panels-v1",
        "status": "complete",
        "source_sha256": sha256(SOURCE),
        "observations": "original paired MEaSUREs raster values",
        "replacement": "PIG holdout only; reference C retained outside",
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    (OUT / "figure6_corrected_velocity_panels_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
