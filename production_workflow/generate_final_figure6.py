"""Generate the six-configuration held-out velocity-error map figure."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parent.parent
EVALUATION = ROOT / "production_workflow/gate4_forward_evaluation_20260829_a"
FOOTPRINT_DATA = ROOT / "production_workflow/gate4_square_footprint_error_maps_20260830_b"
DESIGN = ROOT / "production_workflow/frozen_design"
SUMMARY = ROOT / "production_workflow/gate4_forward_reporting_20260829_b/square_equal_weight_summary.csv"
OUTLINE = ROOT / "tmp/amundsen_v1.geojson"
OUTPUT = ROOT / "production_workflow/final_figures_20260830_a"
VELOCITY_BASEMAP = OUTPUT / "revision_velocity_grid_5km.npz"
ANTARCTICA_LAND = DESIGN / "ne_110m_land.geojson"

CONFIGS = [f"CFG{i:02d}" for i in range(1, 7)]
LABELS = {
    "CFG01": "All ice predictors",
    "CFG02": "Selected ice predictors",
    "CFG03": "All geophysical predictors",
    "CFG04": "Selected geophysical predictors",
    "CFG05": "Selected combined predictors",
    "CFG06": "Selected combined + alignment",
}
COMPACT_LABELS = {
    "CFG01": "All ice",
    "CFG02": "Selected ice",
    "CFG03": "All geophysical",
    "CFG04": "Selected geophysical",
    "CFG05": "Selected combined",
    "CFG06": "Combined + alignment",
}
VMIN = 1.0
VMAX = 2000.0
XLIM = (-1750.0, -1050.0)
YLIM = (-800.0, 35.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload)
    body.pop("manifest_id", None)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    step = float(np.median(np.diff(values)))
    return np.r_[values - step / 2.0, values[-1] + step / 2.0]


def grid_chunk(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    xx = np.unique(x)
    yy = np.unique(y)
    grid = np.full((len(yy), len(xx)), np.nan)
    grid[np.searchsorted(yy, y), np.searchsorted(xx, x)] = z
    return edges(xx / 1000.0), edges(yy / 1000.0), grid


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def antarctica_rings() -> list[np.ndarray]:
    payload = json.loads(ANTARCTICA_LAND.read_text(encoding="utf-8"))
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    rings: list[np.ndarray] = []
    for feature in payload["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            ring = np.asarray(polygon[0], dtype=float)
            if np.nanmin(ring[:, 1]) > -60:
                continue
            px, py = transformer.transform(ring[:, 0], ring[:, 1])
            rings.append(np.column_stack([px, py]) / 1000.0)
    if not rings:
        raise ValueError("No Antarctic land geometry found")
    return rings


def outline_points(outline: dict) -> np.ndarray:
    pieces = []
    for feature in outline["features"]:
        geometry = feature["geometry"]
        lines = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
        pieces.extend(np.asarray(line, dtype=float) / 1000.0 for line in lines)
    return np.concatenate(pieces)


def add_antarctica_locator(ax, outline: dict) -> None:
    inset = ax.inset_axes([0.79, 0.79, 0.18, 0.18])
    for ring in antarctica_rings():
        inset.add_patch(Polygon(ring, closed=True, facecolor="0.78", edgecolor="0.25", lw=0.35))
    points = outline_points(outline)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    inset.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False,
                              edgecolor="#B2182B", lw=1.0))
    inset.set_xlim(-3000, 3000)
    inset.set_ylim(-3000, 3000)
    inset.set_aspect("equal")
    inset.axis("off")


def draw_speed_basemap(ax, support, velocity, alpha: float = 0.30) -> None:
    speed = np.ma.masked_where(
        ~support["eligible"].astype(bool) | ~np.isfinite(velocity["speed"]),
        velocity["speed"],
    )
    ax.pcolormesh(
        edges(support["x_grid"] / 1000.0),
        edges(support["y_grid"] / 1000.0),
        speed,
        cmap="Greys",
        norm=LogNorm(vmin=1, vmax=max(1000.0, float(np.nanmax(speed)))),
        shading="flat",
        alpha=alpha,
        rasterized=True,
        zorder=0,
    )


def load_inputs():
    evaluation_manifest = json.loads(
        (EVALUATION / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    export_manifest = json.loads(
        (FOOTPRINT_DATA / "footprint_error_export_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    design_manifest = json.loads(
        (DESIGN / "selected_squares_manifest.json").read_text(encoding="utf-8")
    )
    if evaluation_manifest.get("status") != "complete":
        raise ValueError("Forward-evaluation bundle is not complete")
    if (
        export_manifest.get("status") != "complete"
        or export_manifest.get("configs") != CONFIGS
        or canonical_id(export_manifest) != export_manifest.get("manifest_id")
    ):
        raise ValueError("Full-footprint error export is not complete")
    if not design_manifest["validation"]["passed"]:
        raise ValueError("Frozen square design did not pass validation")
    if design_manifest["geometry"] != {
        "test_square_side_km": 50.0,
        "buffer_each_side_km": 40.0,
        "excluded_footprint_side_km": 130.0,
    }:
        raise ValueError("Unexpected frozen square geometry")
    with (DESIGN / "selected_squares.csv").open(newline="", encoding="utf-8") as stream:
        squares = list(csv.DictReader(stream))
    if len(squares) != 10:
        raise ValueError("Expected ten frozen squares")
    summary = pd.read_csv(SUMMARY).set_index("configuration")
    support = np.load(DESIGN / "amundsen_input_support_grid_5km.npz")
    velocity = np.load(VELOCITY_BASEMAP)
    outline = json.loads(OUTLINE.read_text(encoding="utf-8"))

    archives = {}
    all_errors = []
    for config in CONFIGS:
        archives[config] = []
        path = FOOTPRINT_DATA / f"{config}_ten_square_footprint_errors.npz"
        with np.load(path, allow_pickle=False) as archive:
            required = {"x", "y", "error_magnitude", "square_number", "central"}
            if not required.issubset(archive.files):
                raise ValueError(f"Missing full-footprint arrays in {path.name}")
            error = np.asarray(archive["error_magnitude"], dtype=float)
            if not np.isfinite(error).all():
                raise ValueError(f"Non-finite error magnitude in {path.name}")
            all_errors.append(error.copy())
            for number in range(1, 11):
                mask = archive["square_number"] == number
                if not mask.any() or not (~archive["central"][mask].astype(bool)).any():
                    raise ValueError(f"Missing footprint or buffer rows in {path.name}")
                archives[config].append(
                    {
                        "x": archive["x"][mask].copy(),
                        "y": archive["y"][mask].copy(),
                        "error_magnitude": archive["error_magnitude"][mask].copy(),
                    }
                )
    return (
        squares,
        summary,
        support,
        velocity,
        outline,
        archives,
        np.concatenate(all_errors),
        evaluation_manifest,
        export_manifest,
        design_manifest,
    )


def draw_outline(ax, outline: dict) -> None:
    for feature in outline["features"]:
        geometry = feature["geometry"]
        lines = (
            geometry["coordinates"]
            if geometry["type"] == "MultiLineString"
            else [geometry["coordinates"]]
        )
        for line in lines:
            line = np.asarray(line, dtype=float) / 1000.0
            ax.plot(line[:, 0], line[:, 1], color="black", lw=0.55, zorder=3)


def draw_map(
    ax, config, squares, support, velocity, outline, archives, summary, show_legend,
    compact_title=False, show_locator=False, show_title=True,
):
    draw_speed_basemap(ax, support, velocity)
    norm = LogNorm(vmin=VMIN, vmax=VMAX, clip=False)
    image = None
    for square, archive in zip(squares, archives[config]):
        xe, ye, zz = grid_chunk(
            archive["x"], archive["y"], archive["error_magnitude"]
        )
        image = ax.pcolormesh(
            xe,
            ye,
            zz,
            cmap="magma",
            norm=norm,
            shading="flat",
            rasterized=True,
            zorder=2,
        )
        outer_x = np.array(
            [
                square["footprint_xmin_m"],
                square["footprint_xmax_m"],
                square["footprint_xmax_m"],
                square["footprint_xmin_m"],
                square["footprint_xmin_m"],
            ],
            dtype=float,
        ) / 1000.0
        outer_y = np.array(
            [
                square["footprint_ymin_m"],
                square["footprint_ymin_m"],
                square["footprint_ymax_m"],
                square["footprint_ymax_m"],
                square["footprint_ymin_m"],
            ],
            dtype=float,
        ) / 1000.0
        inner_x = np.array(
            [
                square["test_xmin_m"],
                square["test_xmax_m"],
                square["test_xmax_m"],
                square["test_xmin_m"],
                square["test_xmin_m"],
            ],
            dtype=float,
        ) / 1000.0
        inner_y = np.array(
            [
                square["test_ymin_m"],
                square["test_ymin_m"],
                square["test_ymax_m"],
                square["test_ymax_m"],
                square["test_ymin_m"],
            ],
            dtype=float,
        ) / 1000.0
        path_effect = [pe.Stroke(linewidth=1.8, foreground="black"), pe.Normal()]
        ax.plot(
            outer_x,
            outer_y,
            color="white",
            lw=0.75,
            ls=(0, (4, 3)),
            path_effects=path_effect,
            zorder=4,
        )
        ax.plot(
            inner_x,
            inner_y,
            color="white",
            lw=0.9,
            path_effects=path_effect,
            zorder=5,
        )
        ax.annotate(
            square["square_id"],
            xy=(outer_x[3], outer_y[3]),
            xytext=(outer_x[3] + 5.0, outer_y[3] + 10.0),
            ha="left",
            va="bottom",
            color="black",
            fontsize=5.8,
            zorder=7,
            bbox={"boxstyle": "square,pad=0.12", "facecolor": "white",
                  "edgecolor": "0.25", "linewidth": 0.45, "alpha": 0.94},
            arrowprops={"arrowstyle": "-", "color": "0.25", "linewidth": 0.45},
        )
    draw_outline(ax, outline)
    add_antarctica_locator(ax, outline)
    median_rmse = float(summary.loc[config, "vector_RMSE_median_m_per_a"])
    title_label = COMPACT_LABELS[config] if compact_title else LABELS[config]
    rmse_label = "Median RMSE" if compact_title else "Median test-square RMSE"
    if show_title:
        ax.set_title(
            f"{config}: {title_label}\n{rmse_label}: {median_rmse:.1f} m a$^{{-1}}$",
            loc="left",
            fontweight="bold",
        )
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("EPSG:3031 easting (km)")
    ax.set_ylabel("EPSG:3031 northing (km)")
    ax.tick_params(direction="out")
    if show_legend:
        handles = [
            Line2D(
                [0], [0], color="black", lw=2.2, ls=(0, (4, 3)),
                path_effects=[pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()],
                label="130 km held-out footprint",
            ),
            Line2D(
                [0], [0], color="black", lw=2.2,
                path_effects=[pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()],
                label="50 km primary test square",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            ncol=1,
            frameon=True,
            fancybox=False,
            framealpha=0.94,
            edgecolor="0.35",
            facecolor="white",
        )
    return image


def add_colorbar(figure, image, axes):
    colorbar = figure.colorbar(
        image, ax=axes, fraction=0.035, pad=0.025, extend="both"
    )
    colorbar.set_label(r"Vector velocity error magnitude (m a$^{-1}$; log scale)")
    colorbar.set_ticks([1, 3, 10, 30, 100, 300, 1000, 2000])
    colorbar.set_ticklabels(["1", "3", "10", "30", "100", "300", "1000", "2000"])


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (
        squares,
        summary,
        support,
        velocity,
        outline,
        archives,
        all_errors,
        evaluation_manifest,
        export_manifest,
        design_manifest,
    ) = load_inputs()

    panel_outputs = []
    for config in CONFIGS:
        figure, axis = plt.subplots(figsize=(4.35, 5.65), constrained_layout=True)
        image = draw_map(
            axis, config, squares, support, velocity, outline, archives, summary, True,
            show_locator=True, show_title=False,
        )
        add_colorbar(figure, image, axis)
        for suffix in ("png", "svg", "pdf"):
            path = OUTPUT / f"figure6_{config.lower()}_velocity_error.{suffix}"
            figure.savefig(
                path,
                dpi=300 if suffix == "png" else None,
                facecolor="white",
                bbox_inches="tight",
            )
            panel_outputs.append(path)
        plt.close(figure)

    combined, axes = plt.subplots(
        2, 3, figsize=(10.5, 10.0), constrained_layout=True,
        sharex=True, sharey=True,
    )
    image = None
    for index, config in enumerate(CONFIGS):
        image = draw_map(
            axes.flat[index], config, squares, support, velocity, outline, archives,
            summary, False, compact_title=True, show_locator=True, show_title=True,
        )
        row, column = divmod(index, 3)
        if row == 0:
            axes.flat[index].set_xlabel("")
        if column != 0:
            axes.flat[index].set_ylabel("")
    add_colorbar(combined, image, list(axes.flat))
    combined.legend(
        handles=[
            Line2D([0], [0], color="black", lw=2.2, ls=(0, (4, 3)),
                   label="130 km held-out footprint"),
            Line2D([0], [0], color="black", lw=2.2,
                   label="50 km primary test square"),
        ],
        loc="center", bbox_to_anchor=(0.50, 0.505), ncol=2,
        frameon=True, fancybox=False, framealpha=0.96,
        edgecolor="0.35", facecolor="white",
    )
    combined_outputs = []
    for suffix in ("png", "svg", "pdf"):
        path = OUTPUT / f"figure6_velocity_error_mosaics_preview.{suffix}"
        combined.savefig(
            path,
            dpi=300 if suffix == "png" else None,
            facecolor="white",
            bbox_inches="tight",
        )
        combined_outputs.append(path)
    plt.close(combined)

    input_paths = [
        EVALUATION / "evaluation_manifest.json",
        FOOTPRINT_DATA / "footprint_error_export_manifest.json",
        FOOTPRINT_DATA / "verification_manifest.json",
        DESIGN / "selected_squares.csv",
        DESIGN / "selected_squares_manifest.json",
        DESIGN / "amundsen_input_support_grid_5km.npz",
        VELOCITY_BASEMAP,
        ANTARCTICA_LAND,
        SUMMARY,
        OUTLINE,
    ] + [FOOTPRINT_DATA / f"{config}_ten_square_footprint_errors.npz"
         for config in CONFIGS]
    outputs = panel_outputs + combined_outputs
    percentiles = np.percentile(all_errors, [1, 5, 50, 95, 99]).tolist()
    manifest = {
        "schema": "jog-final-figure6-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_manifest_id": evaluation_manifest["manifest_id"],
        "footprint_error_export_manifest_id": export_manifest["manifest_id"],
        "square_design_manifest_sha256": sha256_file(
            DESIGN / "selected_squares_manifest.json"
        ),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in input_paths
        },
        "checks": {
            "configuration_count": len(CONFIGS),
            "square_count": len(squares),
            "map_archive_count": len(CONFIGS),
            "square_configuration_map_count": len(CONFIGS) * len(squares),
            "map_row_count": int(all_errors.size),
            "all_errors_finite": bool(np.isfinite(all_errors).all()),
            "test_square_side_km": design_manifest["geometry"]["test_square_side_km"],
            "buffer_each_side_km": design_manifest["geometry"]["buffer_each_side_km"],
            "footprint_side_km": design_manifest["geometry"]["excluded_footprint_side_km"],
            "common_color_scale_m_per_a": [VMIN, VMAX],
            "error_percentiles_1_5_50_95_99_m_per_a": percentiles,
            "error_minimum_m_per_a": float(all_errors.min()),
            "error_maximum_m_per_a": float(all_errors.max()),
            "panel_letters_embedded": False,
            "regional_basemap": "observed surface speed on the frozen 5 km reporting grid",
            "antarctica_locator": "one locator per complete figure",
        },
        "square_equal_weight_median_rmse_m_per_a": {
            config: float(summary.loc[config, "vector_RMSE_median_m_per_a"])
            for config in CONFIGS
        },
        "output_sha256": {path.name: sha256_file(path) for path in outputs},
    }
    manifest["manifest_id"] = canonical_id(manifest)
    (OUTPUT / "figure6_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest_id": manifest["manifest_id"],
                "panel_count": len(CONFIGS),
                "preview": str(combined_outputs[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
