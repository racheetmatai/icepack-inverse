"""Generate the proposed 3 x 2 replacement for manuscript Figure 5.

This is a presentation-only product. It does not modify the manuscript. The
velocity panels use the complete 130 km held-out footprints exported after the
verified Icepack forward evaluations. The two velocity fields are averaged to
a 1.8 km display grid before triangulation; all scientific metrics remain those
computed on the native evaluation population.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import generate_revision_figures_and_tables as base


ROOT = Path(__file__).resolve().parent.parent
ERROR_ROOT = base.FOOTPRINT_ERRORS
OUTPUT_ROOT = ROOT / "output" / "pdf"
CONFIGS = ("CFG02", "CFG04")
DISPLAY_GRID_M = 1800.0
DATASET = ROOT / "production_workflow/gate2_results/gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz"
INVERSION_CONTOUR_LEVEL = 100.0
INVERSION_CONTOUR_COLOR = "#29D8E6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_error_fields(config: str) -> pd.DataFrame:
    manifest_path = ERROR_ROOT / "footprint_error_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Footprint-error export is not complete")

    source = ERROR_ROOT / f"{config}_ten_square_footprint_errors.npz"
    expected = manifest["output_sha256"][source.name]
    observed = sha256(source)
    if observed != expected:
        raise RuntimeError(f"Hash mismatch for {source.name}")

    with np.load(source, allow_pickle=False) as data:
        model_error = np.sqrt(data["model_squared_error"].astype(float))
        uniform_error = np.sqrt(data["uniform_squared_error"].astype(float))
        frame = pd.DataFrame({
            "x_m": data["x"].astype(float),
            "y_m": data["y"].astype(float),
            "square_number": data["square_number"].astype(np.int16),
            "absolute_error": model_error,
            "error_difference": model_error - uniform_error,
        })
    if not np.all(np.isfinite(frame[["x_m", "y_m", "absolute_error", "error_difference"]])):
        raise RuntimeError(f"Non-finite plotting values in {source.name}")
    return frame


def aligned_inversion_error(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the inversion-reference vector residual to archived map rows."""
    source = pd.read_csv(
        DATASET,
        usecols=[
            "row_id", "x", "y", "common_eligible", "observed_vx", "observed_vy",
            "inversion_vx", "inversion_vy", "square_footprint_id",
        ],
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
        source[["row_id", "x", "y", "inversion_error", "square_footprint_id"]],
        left_on=["x_m", "y_m"], right_on=["x", "y"], how="left",
        validate="one_to_one", indicator=True,
    )
    if len(aligned) != len(frame) or not aligned["_merge"].eq("both").all():
        raise RuntimeError("Figure 5 rows do not align completely with canonical eligible rows")
    expected = aligned["square_number"].map(lambda n: f"SQ{int(n):02d}")
    if not np.array_equal(aligned["square_footprint_id"].to_numpy(str), expected.to_numpy(str)):
        raise RuntimeError("Archived square ownership disagrees with stable row-ID membership")
    if not np.isfinite(aligned["inversion_error"]).all():
        raise RuntimeError("Non-finite inversion residual after row alignment")
    return aligned


def display_grid(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Area-bin native 450 m evaluation points to a 1.8 km display grid."""
    work = frame.copy()
    work["ix"] = np.floor(work["x_m"] / DISPLAY_GRID_M).astype(np.int64)
    work["iy"] = np.floor(work["y_m"] / DISPLAY_GRID_M).astype(np.int64)
    grouped = work.groupby(["square_number", "ix", "iy"], sort=True, observed=True).agg(
        x_m=("x_m", "mean"),
        y_m=("y_m", "mean"),
        absolute_error=("absolute_error", "mean"),
        error_difference=("error_difference", "mean"),
    ).reset_index()
    return (
        grouped[["x_m", "y_m"]].to_numpy(float) / 1000.0,
        grouped["square_number"].to_numpy(np.int16),
        grouped["absolute_error"].to_numpy(float),
        grouped["error_difference"].to_numpy(float),
    )


def inversion_display_grid(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["ix"] = np.floor(work["x_m"] / DISPLAY_GRID_M).astype(np.int64)
    work["iy"] = np.floor(work["y_m"] / DISPLAY_GRID_M).astype(np.int64)
    return work.groupby(["square_number", "ix", "iy"], sort=True, observed=True).agg(
        x_m=("x_m", "mean"), y_m=("y_m", "mean"),
        inversion_error=("inversion_error", "mean"),
    ).reset_index()


def draw_inversion_reference_contour(ax, shown: pd.DataFrame) -> None:
    """Draw the 100 m/a inversion residual independently in each footprint."""
    for square_number in sorted(shown["square_number"].unique()):
        local = shown.loc[shown["square_number"].eq(square_number)]
        coords = local[["x_m", "y_m"]].to_numpy(float) / 1000.0
        owner = local["square_number"].to_numpy(np.int16)
        values = local["inversion_error"].to_numpy(float)
        tri = triangulation(coords, owner)
        ax.tricontour(
            tri, values, levels=[INVERSION_CONTOUR_LEVEL], colors="0.08",
            linewidths=1.35, linestyles="solid", zorder=12,
        )
        ax.tricontour(
            tri, values, levels=[INVERSION_CONTOUR_LEVEL], colors=INVERSION_CONTOUR_COLOR,
            linewidths=0.72, linestyles="solid", zorder=13,
        )


def triangulation(coords_km: np.ndarray, owner: np.ndarray) -> mtri.Triangulation:
    tri = mtri.Triangulation(coords_km[:, 0], coords_km[:, 1])
    triangles = tri.triangles
    points = coords_km[triangles]
    longest_edge = np.max(
        np.linalg.norm(points - np.roll(points, 1, axis=1), axis=2), axis=1
    )
    triangle_owner = owner[triangles]
    tri.set_mask(
        (triangle_owner[:, 0] != triangle_owner[:, 1])
        | (triangle_owner[:, 0] != triangle_owner[:, 2])
        | (longest_edge > 4.0)
    )
    return tri


def draw_field(ax, tri, values, *, cmap, norm):
    return ax.tripcolor(
        tri,
        values,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        rasterized=True,
        zorder=2,
    )


def panel_finish(ax, region, squares, panel: str, locator: bool = False) -> None:
    base.draw_figure7_holdout_geometry(ax, squares, show_legend=False)
    base.map_axes(ax, region["outline"])
    ax.text(
        0.018,
        0.975,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="0.08",
        path_effects=[pe.withStroke(linewidth=2.8, foreground="white")],
        zorder=20,
    )
    if locator:
        base.add_antarctica_locator(ax, region["outline"])


def label_primary_squares(ax, squares: pd.DataFrame) -> None:
    """Label squares outside every 130 km footprint using audited callouts."""
    # Hand-tuned label positions avoid all ten held-out footprints. Automatic
    # offsets are not reliable here because several frozen footprints nearly
    # touch. Coordinates are EPSG:3031 kilometres.
    placements = {
        "SQ01": (-1590.0, -25.0, "top"),
        "SQ02": (-1385.0, -310.0, "bottom"),
        "SQ03": (-1475.0, -520.0, "left"),
        "SQ04": (-1120.0, -310.0, "right"),
        "SQ05": (-1635.0, -250.0, "left"),
        "SQ06": (-1545.0, -390.0, "left"),
        "SQ07": (-1140.0, -470.0, "right"),
        "SQ08": (-1120.0, -600.0, "right"),
        "SQ09": (-1435.0, 0.0, "top"),
        "SQ10": (-1160.0, -160.0, "right"),
    }
    for row in squares.itertuples():
        text_x, text_y, side = placements[row.square_id]
        if side == "top":
            anchor_x = row.center_x_m / 1000.0
            anchor_y = row.footprint_ymax_m / 1000.0
        elif side == "bottom":
            anchor_x = row.center_x_m / 1000.0
            anchor_y = row.footprint_ymin_m / 1000.0
        elif side == "left":
            anchor_x = row.footprint_xmin_m / 1000.0
            anchor_y = row.center_y_m / 1000.0
        elif side == "right":
            anchor_x = row.footprint_xmax_m / 1000.0
            anchor_y = row.center_y_m / 1000.0
        else:  # pragma: no cover - frozen placements above are exhaustive.
            raise RuntimeError(f"Unknown label side for {row.square_id}: {side}")

        # The label anchor itself must not fall in any frozen footprint.
        for other in squares.itertuples():
            if (
                other.footprint_xmin_m / 1000.0 <= text_x <= other.footprint_xmax_m / 1000.0
                and other.footprint_ymin_m / 1000.0 <= text_y <= other.footprint_ymax_m / 1000.0
            ):
                raise RuntimeError(
                    f"Label {row.square_id} intrudes into footprint {other.square_id}"
                )
        ax.annotate(
            row.square_id,
            xy=(anchor_x, anchor_y),
            xytext=(text_x, text_y),
            ha="center",
            va="center",
            fontsize=7.8,
            color="0.08",
            bbox={
                "boxstyle": "square,pad=0.12",
                "facecolor": "white",
                "edgecolor": "0.25",
                "linewidth": 0.45,
                "alpha": 0.94,
            },
            arrowprops={"arrowstyle": "-", "color": "0.25", "linewidth": 0.45},
            zorder=19,
        )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.style()
    plt.rcParams.update({
        "axes.titlesize": 13,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
    })

    region = base.region_context()
    velocity_context = base.aggregate_velocity_grid()
    squares = base.square_table()

    c_fields = {cfg: base.load_c_fields(cfg) for cfg in CONFIGS}
    error_frames = {cfg: verified_error_fields(cfg) for cfg in CONFIGS}
    error_fields = {cfg: display_grid(error_frames[cfg]) for cfg in CONFIGS}
    inversion_native = aligned_inversion_error(error_frames["CFG02"])
    inversion_shown = inversion_display_grid(inversion_native)

    all_c_difference = np.concatenate([np.abs(c_fields[cfg][3]) for cfg in CONFIGS])
    c_limit = float(np.nanquantile(all_c_difference, 0.995))

    fig = plt.figure(figsize=(9.4, 12.0))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=(1, 1, 0.055),
        left=0.115,
        right=0.93,
        bottom=0.075,
        top=0.955,
        wspace=0.12,
        hspace=0.16,
    )
    axes = np.empty((3, 2), dtype=object)
    for row in range(3):
        for col in range(2):
            axes[row, col] = fig.add_subplot(grid[row, col])
            base.draw_speed_basemap(axes[row, col], region, velocity_context, alpha=0.20)

    axes[0, 0].set_title("CFG02\nSelected ice predictors", fontweight="bold", pad=7)
    axes[0, 1].set_title("CFG04\nSelected geophysical predictors", fontweight="bold", pad=7)

    c_images = []
    absolute_images = []
    difference_images = []
    letters = (("(a)", "(b)"), ("(c)", "(d)"), ("(e)", "(f)"))

    for col, config in enumerate(CONFIGS):
        coords, _, _, c_difference, owner, _ = c_fields[config]
        c_images.append(base.draw_continuous_c_difference(
            axes[0, col], coords, c_difference, owner, c_limit
        ))

        display_coords, display_owner, absolute_error, error_difference = error_fields[config]
        tri = triangulation(display_coords, display_owner)
        absolute_images.append(draw_field(
            axes[1, col],
            tri,
            absolute_error,
            cmap="inferno",
            norm=LogNorm(vmin=1.0, vmax=2000.0),
        ))
        draw_inversion_reference_contour(axes[1, col], inversion_shown)
        difference_images.append(draw_field(
            axes[2, col],
            tri,
            error_difference,
            cmap="RdBu_r",
            norm=SymLogNorm(linthresh=10.0, linscale=1.0, vmin=-2000.0, vmax=2000.0, base=10),
        ))

        for row in range(3):
            panel_finish(
                axes[row, col],
                region,
                squares,
                letters[row][col],
                locator=(row == 0),
            )
        label_primary_squares(axes[0, col], squares)

    contour_handle = Line2D(
        [0], [0], color=INVERSION_CONTOUR_COLOR, lw=1.1,
        path_effects=[pe.Stroke(linewidth=2.0, foreground="0.08"), pe.Normal()],
    )
    axes[1, 0].legend(
        [contour_handle], [r"Inversion reference: 100 m a$^{-1}$"],
        loc="lower left", bbox_to_anchor=(0.018, 0.018), frameon=True,
        fancybox=False, edgecolor="0.45", handlelength=1.8,
        borderpad=0.25, fontsize=8.6,
    )

    # Avoid repeated coordinate labels while keeping the shared map coordinates clear.
    for row in range(3):
        axes[row, 1].set_ylabel("")
        axes[row, 1].tick_params(labelleft=False)
    for row in range(2):
        for col in range(2):
            axes[row, col].set_xlabel("")
            axes[row, col].tick_params(labelbottom=False)

    cbar0 = fig.colorbar(c_images[0], cax=fig.add_subplot(grid[0, 2]), extend="both")
    cbar0.set_label(r"$C_{\rm ML}-C_{\rm ref}$")
    cbar1 = fig.colorbar(absolute_images[0], cax=fig.add_subplot(grid[1, 2]), extend="both")
    cbar1.set_label(r"Vector velocity error (m a$^{-1}$; log scale)")
    cbar1.set_ticks([1, 10, 100, 1000, 2000])
    cbar1.set_ticklabels(["1", "10", "100", "1000", "2000"])
    cbar2 = fig.colorbar(difference_images[0], cax=fig.add_subplot(grid[2, 2]), extend="both")
    cbar2.set_label(
        "ML minus uniform-$C$ local error\n"
        r"(m a$^{-1}$; symmetric log scale)"
    )
    cbar2.set_ticks([-1000, -100, -10, 0, 10, 100, 1000])
    cbar2.set_ticklabels(["−1000", "−100", "−10", "0", "10", "100", "1000"])

    row_labels = (
        "Control difference",
        "Absolute velocity error",
        "Difference from uniform $C$",
    )
    for y, label in zip((0.813, 0.515, 0.218), row_labels):
        fig.text(0.018, y, label, rotation=90, ha="center", va="center", fontsize=12.5, fontweight="bold")

    geometry_handles = [
        Line2D([0], [0], color="0.15", lw=1.3, label="50 km primary test square"),
        Line2D([0], [0], color="0.15", lw=1.1, ls=(0, (4, 3)), label="130 km held-out footprint"),
    ]
    fig.legend(
        handles=geometry_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.008),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="0.45",
        handlelength=2.7,
        columnspacing=2.0,
    )

    stem = OUTPUT_ROOT / "figure5_spatial_control_velocity_uniform_comparison"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    output_manifest = {
        "schema": "jog-proposed-figure5-v1",
        "status": "complete",
        "configurations": list(CONFIGS),
        "population": "complete_130km_held_out_footprints",
        "velocity_display_grid_m": DISPLAY_GRID_M,
        "row_1": "median_ensemble_C_minus_inversion_reference_C",
        "row_2": "norm_of_median_ensemble_velocity_minus_observed_velocity",
        "row_2_inversion_reference_contour_m_per_a": INVERSION_CONTOUR_LEVEL,
        "inversion_reference_error_definition": "sqrt((inversion_vx-observed_vx)^2 + (inversion_vy-observed_vy)^2)",
        "row_3": "model_local_vector_error_minus_square_specific_uniform_C_local_vector_error",
        "uniform_C_note": "one area-weighted development-population mean C per square; Icepack grounding ramp retained",
        "c_difference_limit_99p5": c_limit,
        "source_manifest_id": json.loads(
            (ERROR_ROOT / "footprint_error_export_manifest.json").read_text(encoding="utf-8")
        )["manifest_id"],
        "source_hashes": {
            f"{cfg}_ten_square_footprint_errors.npz": sha256(
                ERROR_ROOT / f"{cfg}_ten_square_footprint_errors.npz"
            )
            for cfg in CONFIGS
        },
    }
    output_manifest["outputs"] = {
        stem.with_suffix(f".{suffix}").name: sha256(stem.with_suffix(f".{suffix}"))
        for suffix in ("png", "pdf", "svg")
    }
    (OUTPUT_ROOT / "figure5_spatial_control_velocity_uniform_comparison_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
