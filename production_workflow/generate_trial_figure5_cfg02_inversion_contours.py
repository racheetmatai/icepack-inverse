"""Create one trial Figure 5 CFG02 error panel with inversion-error contours."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import generate_proposed_figure5 as figure5
import generate_revision_figures_and_tables as base


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "production_workflow/gate2_results/gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz"
OUT = ROOT / "production_workflow/trial_figure5_inversion_contours_20260912"
LEVELS = np.array([100.0])
LINESTYLES = ("solid",)
CONTOUR_COLOR = "#29D8E6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_and_align() -> pd.DataFrame:
    frame = figure5.verified_error_fields("CFG02")
    source = pd.read_csv(
        DATASET,
        usecols=["row_id", "x", "y", "common_eligible", "observed_vx", "observed_vy", "inversion_vx", "inversion_vy"],
        low_memory=False,
    )
    source = source.loc[source["common_eligible"].astype(bool)].copy()
    if source["row_id"].duplicated().any() or source.duplicated(["x", "y"]).any():
        raise RuntimeError("Canonical eligible row IDs or coordinates are not unique")
    source["inversion_error"] = np.hypot(
        source["inversion_vx"] - source["observed_vx"],
        source["inversion_vy"] - source["observed_vy"],
    )
    aligned = frame.merge(
        source[["row_id", "x", "y", "inversion_error"]],
        left_on=["x_m", "y_m"],
        right_on=["x", "y"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(aligned) != len(frame) or not aligned["_merge"].eq("both").all():
        raise RuntimeError("Figure 5 rows do not align completely with canonical eligible rows")
    if not np.isfinite(aligned["inversion_error"]).all():
        raise RuntimeError("Non-finite inversion residual after stable coordinate alignment")
    # Confirm that each archived square number agrees with the frozen footprint ID.
    expected = aligned["square_number"].map(lambda n: f"SQ{int(n):02d}")
    lookup = pd.read_csv(DATASET, usecols=["row_id", "square_footprint_id"], low_memory=False)
    lookup = lookup.loc[lookup["row_id"].isin(aligned["row_id"])]
    check = aligned[["row_id"]].merge(lookup, on="row_id", validate="one_to_one")
    if not np.array_equal(check["square_footprint_id"].to_numpy(str), expected.to_numpy(str)):
        raise RuntimeError("Archived square ownership disagrees with stable row-ID membership")
    return aligned


def display_grid(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["ix"] = np.floor(work["x_m"] / figure5.DISPLAY_GRID_M).astype(np.int64)
    work["iy"] = np.floor(work["y_m"] / figure5.DISPLAY_GRID_M).astype(np.int64)
    return work.groupby(["square_number", "ix", "iy"], sort=True, observed=True).agg(
        x_m=("x_m", "mean"),
        y_m=("y_m", "mean"),
        absolute_error=("absolute_error", "mean"),
        inversion_error=("inversion_error", "mean"),
        native_rows=("row_id", "size"),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    native = load_and_align()
    shown = display_grid(native)
    coords = shown[["x_m", "y_m"]].to_numpy(float) / 1000.0
    owner = shown["square_number"].to_numpy(np.int16)
    tri = figure5.triangulation(coords, owner)

    base.style()
    plt.rcParams.update({
        "axes.labelsize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.3,
    })
    region = base.region_context()
    velocity = base.aggregate_velocity_grid()
    squares = base.square_table()
    fig, ax = plt.subplots(figsize=(5.25, 5.7), constrained_layout=True)
    base.draw_speed_basemap(ax, region, velocity, alpha=0.20)
    image = figure5.draw_field(
        ax,
        tri,
        shown["absolute_error"].to_numpy(float),
        cmap="inferno",
        norm=LogNorm(vmin=1.0, vmax=2000.0),
    )

    # Draw each footprint separately. This prevents contours crossing gaps or
    # connecting distinct withheld footprints. The dark underlay keeps the
    # white contours legible across the full inferno color scale.
    for square_number in sorted(np.unique(owner)):
        mask = owner == square_number
        local_coords = coords[mask]
        local_owner = owner[mask]
        local_values = shown.loc[mask, "inversion_error"].to_numpy(float)
        local_tri = figure5.triangulation(local_coords, local_owner)
        ax.tricontour(local_tri, local_values, levels=LEVELS, colors="0.08", linewidths=1.35,
                      linestyles=LINESTYLES, zorder=12)
        contours = ax.tricontour(local_tri, local_values, levels=LEVELS, colors=CONTOUR_COLOR, linewidths=0.72,
                                 linestyles=LINESTYLES, zorder=13)
        for collection in contours.collections:
            collection.set_path_effects([pe.Normal()])

    figure5.panel_finish(ax, region, squares, "(c)", locator=True)
    figure5.label_primary_squares(ax, squares)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025, extend="both")
    cbar.set_label(r"CFG02 vector velocity error (m a$^{-1}$; log scale)")
    cbar.set_ticks([1, 10, 100, 1000, 2000])
    cbar.set_ticklabels(["1", "10", "100", "1000", "2000"])
    contour_handles = [
        Line2D([0], [0], color=CONTOUR_COLOR, lw=1.1, ls=style,
               path_effects=[pe.Stroke(linewidth=2.0, foreground="0.08"), pe.Normal()])
        for style in LINESTYLES
    ]
    ax.legend(
        contour_handles,
        [r"Inversion reference: 100 m a$^{-1}$"],
        loc="lower left",
        bbox_to_anchor=(0.018, 0.018),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="0.45",
        handlelength=1.8,
        borderpad=0.25,
        fontsize=8.6,
    )

    stem = args.output / "trial_figure5_cfg02_absolute_error_with_inversion_contours"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    q = np.quantile(native["inversion_error"], [0, .25, .5, .75, .9, .95, .99, 1])
    record = {
        "status": "complete",
        "trial_only": True,
        "approved_figure_modified": False,
        "configuration": "CFG02",
        "population": "complete 130 km footprints for SQ01--SQ10",
        "native_rows": int(len(native)),
        "display_cells": int(len(shown)),
        "display_grid_m": figure5.DISPLAY_GRID_M,
        "contour_levels_m_per_a": LEVELS.tolist(),
        "contour_line_styles": dict(zip(map(str, LEVELS.tolist()), LINESTYLES)),
        "inversion_error_native_quantiles_m_per_a": dict(zip(["min", "q25", "q50", "q75", "q90", "q95", "q99", "max"], map(float, q))),
        "error_definition": "sqrt((inversion_vx-observed_vx)^2 + (inversion_vy-observed_vy)^2)",
        "alignment": "complete one-to-one x/y join to unique canonical eligible rows, followed by stable row-ID verification of frozen footprint membership",
        "display_operation": "arithmetic mean of native local error magnitudes within the same 1.8 km display cells used by Figure 5; no smoothing",
        "contour_geometry": "separate triangulation per footprint; triangles longer than 4 km masked",
        "units": "m a^-1",
        "sources": {
            "canonical_dataset": {"path": str(DATASET), "sha256": sha256(DATASET)},
            "cfg02_footprint_errors": {"path": str(figure5.ERROR_ROOT / "CFG02_ten_square_footprint_errors.npz"), "sha256": sha256(figure5.ERROR_ROOT / "CFG02_ten_square_footprint_errors.npz")},
            "approved_figure": {},
        },
        "outputs": {
            stem.with_suffix(".pdf").name: sha256(stem.with_suffix(".pdf")),
            stem.with_suffix(".png").name: sha256(stem.with_suffix(".png")),
        },
    }
    # The approved figure path is resolved here to avoid altering its source.
    approved = ROOT / "output/pdf/figure5_spatial_control_velocity_uniform_comparison.pdf"
    record["sources"]["approved_figure"] = {"path": str(approved), "sha256": sha256(approved)}
    (args.output / "trial_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
