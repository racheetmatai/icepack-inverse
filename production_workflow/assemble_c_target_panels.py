"""Assemble final held-out C mosaics and inversion-target distributions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd


CONFIG_LABELS = {
    "CFG01": "All ice",
    "CFG02": "Best ice",
    "CFG03": "All geophysical",
    "CFG04": "Best geophysical",
    "CFG05": "Best combined",
    "CFG06": "Combined + alignment",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_id(payload: dict) -> str:
    unsigned = dict(payload); unsigned.pop("manifest_id", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(canonical).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def unpack_mask(path: Path, name: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        count = int(archive["row_count"])
        bitorder = str(archive["bitorder"])
        return np.unpackbits(archive[name], count=count, bitorder=bitorder).astype(bool)


def outline_lines(path: Path) -> list[np.ndarray]:
    geojson = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for feature in geojson["features"]:
        geometry = feature["geometry"]
        lines = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
        result.extend(np.asarray(line, dtype=float) / 1000 for line in lines)
    return result


def box_coordinates(row: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax = float(row[f"{prefix}_xmin_m"]), float(row[f"{prefix}_xmax_m"])
    ymin, ymax = float(row[f"{prefix}_ymin_m"]), float(row[f"{prefix}_ymax_m"])
    return (np.asarray([xmin, xmax, xmax, xmin, xmin]) / 1000,
            np.asarray([ymin, ymin, ymax, ymax, ymin]) / 1000)


def load_square_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def collect_c_data(predictions: Path, squares: list[dict]) -> tuple[dict, float, float, float]:
    data = {}
    field_values = []
    difference_values = []
    reference_template = None
    coordinate_template = None
    eligible_template = None
    for config in CONFIG_LABELS:
        chunks = []
        for number, row in enumerate(squares, start=1):
            path = predictions / f"SQ{number:02d}_{config}.npz"
            with np.load(path, allow_pickle=False) as archive:
                coordinates = archive["coordinates"].astype(float)
                eligible = archive["eligible_mask"].astype(bool)
                reference = archive["reference_log_C"].astype(float)
                predicted = archive["median_log_C"].astype(float)
            if reference_template is None:
                reference_template = reference; coordinate_template = coordinates; eligible_template = eligible
            elif (not np.array_equal(coordinates, coordinate_template)
                  or not np.array_equal(eligible, eligible_template)
                  or not np.array_equal(reference, reference_template)):
                raise RuntimeError(f"Full-mesh identity mismatch: {path}")
            mask = (eligible
                    & (coordinates[:, 0] >= float(row["footprint_xmin_m"]))
                    & (coordinates[:, 0] <= float(row["footprint_xmax_m"]))
                    & (coordinates[:, 1] >= float(row["footprint_ymin_m"]))
                    & (coordinates[:, 1] <= float(row["footprint_ymax_m"])))
            chunks.append({"number": number, "x": coordinates[mask, 0] / 1000,
                           "y": coordinates[mask, 1] / 1000,
                           "reference": reference[mask], "predicted": predicted[mask],
                           "difference": predicted[mask] - reference[mask]})
            field_values.extend((reference[mask], predicted[mask]))
            difference_values.append(predicted[mask] - reference[mask])
        data[config] = chunks
    all_fields = np.concatenate(field_values)
    all_differences = np.concatenate(difference_values)
    vmin, vmax = np.quantile(all_fields, [0.005, 0.995])
    dmax = float(np.quantile(np.abs(all_differences), 0.995))
    return data, float(vmin), float(vmax), dmax


def decorate_map(axis, outlines: list[np.ndarray], squares: list[dict], labels: bool) -> None:
    for line in outlines:
        axis.plot(line[:, 0], line[:, 1], color="black", linewidth=.55, zorder=4)
    effect = [pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()]
    for row in squares:
        outer_x, outer_y = box_coordinates(row, "footprint")
        inner_x, inner_y = box_coordinates(row, "test")
        axis.plot(outer_x, outer_y, color="black", linewidth=.85, linestyle=(0, (4, 3)),
                  path_effects=effect, zorder=5)
        axis.plot(inner_x, inner_y, color="black", linewidth=1.0, path_effects=effect, zorder=5)
        if labels:
            text = axis.text(float(row["center_x_m"]) / 1000, float(row["center_y_m"]) / 1000,
                             row["square_id"].replace("SQ", ""), ha="center", va="center",
                             fontsize=7, color="white", zorder=6)
            text.set_path_effects([pe.Stroke(linewidth=1.8, foreground="black"), pe.Normal()])
    axis.set_xlim(-1750, -1050); axis.set_ylim(-800, 35); axis.set_aspect("equal")
    axis.set_xlabel("EPSG:3031 easting (km)")


def plot_c_mosaics(data: dict, vmin: float, vmax: float, dmax: float, squares: list[dict],
                   outlines: list[np.ndarray], output: Path) -> list[Path]:
    paths = []
    for config, chunks in data.items():
        fig, axes = plt.subplots(1, 3, figsize=(15.8, 7.1), constrained_layout=True, sharex=True, sharey=True)
        handles = [None, None, None]
        for chunk in chunks:
            for index, (field, cmap, norm) in enumerate([
                ("reference", "viridis", None), ("predicted", "viridis", None),
                ("difference", "coolwarm", TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)),
            ]):
                kwargs = {"vmin": vmin, "vmax": vmax} if norm is None else {"norm": norm}
                handles[index] = axes[index].scatter(chunk["x"], chunk["y"], c=chunk[field], s=2.2,
                                                     marker="s", linewidths=0, cmap=cmap,
                                                     rasterized=True, zorder=2, **kwargs)
        titles = ["Inversion-reference C", "Median predicted C", "Predicted − inversion C"]
        for index, axis in enumerate(axes):
            decorate_map(axis, outlines, squares, labels=index == 1)
            axis.set_title(titles[index])
        axes[0].set_ylabel("EPSG:3031 northing (km)")
        cbar = fig.colorbar(handles[0], ax=axes[:2], fraction=.025, pad=.015, extend="both")
        cbar.set_label("C (Icepack logarithmic friction control)")
        dbar = fig.colorbar(handles[2], ax=axes[2], fraction=.05, pad=.02, extend="both")
        dbar.set_label("ΔC")
        legend = [
            Line2D([0], [0], color="black", lw=2, ls=(0, (4, 3)), label="130 km held-out footprint"),
            Line2D([0], [0], color="black", lw=2, label="Central 50 km test square"),
        ]
        axes[0].legend(handles=legend, loc="lower left", fontsize=7.5, framealpha=.9)
        fig.suptitle(f"{CONFIG_LABELS[config]} ({config}): held-out friction-control fields\n"
                     "Only ML-eligible grounded degrees of freedom are shown", fontsize=14)
        path = output / f"heldout_c_mosaic_{config.lower()}.png"
        fig.savefig(path, dpi=300, facecolor="white")
        fig.savefig(path.with_suffix(".svg"))
        plt.close(fig); paths.append(path)
    return paths


def density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=edges, density=True)
    return histogram


def density_in_display_window(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram density normalized by the complete population size."""
    counts, _ = np.histogram(values, bins=edges)
    return counts / (len(values) * np.diff(edges))


def distribution_metrics(training: np.ndarray, heldout: np.ndarray) -> dict:
    q01, q99 = np.quantile(training, [0.01, 0.99])
    grid = np.linspace(0.005, 0.995, 199)
    distance = float(np.mean(np.abs(np.quantile(training, grid) - np.quantile(heldout, grid))))
    iqr = float(np.subtract(*np.quantile(training, [.75, .25])))
    return {
        "training_rows": int(len(training)), "heldout_rows": int(len(heldout)),
        "heldout_inside_training_q01_q99_fraction": float(np.mean((heldout >= q01) & (heldout <= q99))),
        "quantile_distance_normalized_by_training_iqr": None if iqr == 0 else distance / iqr,
        "training_mean": float(np.mean(training)), "heldout_mean": float(np.mean(heldout)),
        "training_q01": float(q01), "training_q99": float(q99),
        "heldout_q01": float(np.quantile(heldout, .01)), "heldout_q99": float(np.quantile(heldout, .99)),
    }


def plot_target_distributions(dataset: Path, splits: Path, output: Path) -> tuple[Path, Path, Path]:
    frame = pd.read_csv(dataset, usecols=["row_id", "common_eligible", "reference_log_C"], low_memory=False)
    frame = frame.loc[frame["common_eligible"].astype(bool)].sort_values("row_id", kind="stable").reset_index(drop=True)
    target = frame["reference_log_C"].to_numpy(float)
    if not np.isfinite(target).all():
        raise RuntimeError("Non-finite inversion targets in common eligible population")
    regional_edges = np.linspace(*np.quantile(target, [.001, .999]), 121)
    regional_centers = (regional_edges[:-1] + regional_edges[1:]) / 2
    records = []

    fig, axes = plt.subplots(2, 5, figsize=(15.4, 6.9), sharex=False, sharey=False)
    for index, axis in enumerate(axes.flat, start=1):
        experiment = f"SQ{index:02d}"
        mask_path = splits / "population_masks" / f"{experiment}.npz"
        development = unpack_mask(mask_path, "development")
        central = unpack_mask(mask_path, "central_50km")
        annulus = unpack_mask(mask_path, "exclusion_annulus")
        if not (len(development) == len(central) == len(target)):
            raise RuntimeError(f"Split alignment failure: {experiment}")
        training_values = target[development]; central_values = target[central]; annulus_values = target[annulus]
        metrics = distribution_metrics(training_values, central_values)
        records.append({"experiment": experiment, "population": "central_50km", **metrics})
        records.append({"experiment": experiment, "population": "exclusion_annulus",
                        **distribution_metrics(training_values, annulus_values)})
        populations = (training_values, annulus_values, central_values)
        display_min = min(float(np.quantile(values, .05)) for values in populations)
        display_max = max(float(np.quantile(values, .95)) for values in populations)
        if not display_max > display_min:
            raise RuntimeError(f"Degenerate target-distribution display range: {experiment}")
        square_edges = np.linspace(display_min, display_max, 121)
        square_centers = (square_edges[:-1] + square_edges[1:]) / 2
        axis.plot(square_centers, density_in_display_window(training_values, square_edges),
                  color="0.35", linewidth=1.4, label="Training and validation")
        axis.plot(square_centers, density_in_display_window(annulus_values, square_edges),
                  color="#D55E00", linewidth=1.2,
                  linestyle="--", label="40 km buffer")
        axis.plot(square_centers, density_in_display_window(central_values, square_edges),
                  color="#0072B2", linewidth=1.7,
                  label="Central 50 km square")
        axis.set_xlim(display_min, display_max)
        axis.set_ylim(bottom=0)
        axis.set_title(experiment, fontsize=10)
        axis.grid(alpha=.16)
    for axis in axes[-1, :]: axis.set_xlabel("Inversion-reference C")
    for axis in axes[:, 0]: axis.set_ylabel("Density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Inversion-reference C distributions for the ten withheld squares\n"
                 "C from the sector-wide reference inversion",
                 fontsize=14, y=.99)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .89))
    fig.tight_layout(rect=(0, 0, 1, .82))
    square_path = output / "square_target_distributions.png"
    fig.savefig(square_path, dpi=300, facecolor="white"); fig.savefig(square_path.with_suffix(".svg"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharex=True, sharey=True)
    for axis, experiment, heldout_name, label in [
        (axes[0], "REG_INTER", "heldout", "Both inter-catchment corridors"),
        (axes[1], "REG_PIG", "heldout", "Pine Island Glacier"),
    ]:
        mask_path = splits / "population_masks" / f"{experiment}.npz"
        development = unpack_mask(mask_path, "development")
        heldout = unpack_mask(mask_path, heldout_name)
        training_values = target[development]; heldout_values = target[heldout]
        metrics = distribution_metrics(training_values, heldout_values)
        records.append({"experiment": experiment, "population": heldout_name, **metrics})
        axis.plot(regional_centers, density(training_values, regional_edges), color="0.35", linewidth=1.5, label="Training and validation")
        axis.plot(regional_centers, density(heldout_values, regional_edges), color="#0072B2", linewidth=1.8, label="Held-out region")
        axis.set_title(f"{label}\nwithin training and validation 1–99%: "
                       f"{100 * metrics['heldout_inside_training_q01_q99_fraction']:.1f}%")
        axis.set_xlabel("Inversion-reference C"); axis.grid(alpha=.16)
    axes[0].set_ylabel("Density")
    fig.suptitle("Inversion-reference C distributions for regional experiments", fontsize=14, y=.99)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .90))
    fig.tight_layout(rect=(0, 0, 1, .82))
    region_path = output / "regional_target_distributions.png"
    fig.savefig(region_path, dpi=300, facecolor="white"); fig.savefig(region_path.with_suffix(".svg"))
    plt.close(fig)

    table_path = output / "target_distribution_summary.csv"
    pd.DataFrame(records).to_csv(table_path, index=False)
    return square_path, region_path, table_path


def run(args) -> dict:
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    squares = load_square_rows(args.squares.resolve())
    if len(squares) != 10:
        raise RuntimeError("Expected ten frozen squares")
    data, vmin, vmax, dmax = collect_c_data(args.predictions.resolve(), squares)
    c_paths = plot_c_mosaics(data, vmin, vmax, dmax, squares,
                             outline_lines(args.outline.resolve()), output)
    square_target, regional_target, target_table = plot_target_distributions(
        args.dataset.resolve(), args.splits.resolve(), output)
    outputs = {}
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "c_target_panel_manifest.json"):
        outputs[path.relative_to(output).as_posix()] = sha256_file(path)
    manifest = {
        "schema": "jog-c-target-panel-bundle-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configurations": list(CONFIG_LABELS), "square_count": len(squares),
        "c_map_scope": "ML-eligible grounded degrees of freedom inside each independently held-out 130 km footprint",
        "target_role": "descriptive inversion target only; observed-velocity agreement remains primary because C may be non-unique",
        "common_c_color_limits": {"minimum": vmin, "maximum": vmax, "difference_absolute_maximum": dmax},
        "inputs": {
            "prediction_set_manifest_id": json.loads(
                (args.predictions.resolve() / "prediction_set_manifest.json").read_text(encoding="utf-8"))["manifest_id"],
            "split_bundle_manifest_id": json.loads(
                (args.splits.resolve() / "split_bundle_manifest.json").read_text(encoding="utf-8"))["manifest_id"],
            "dataset_sha256": sha256_file(args.dataset.resolve()),
            "selected_squares_sha256": sha256_file(args.squares.resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
        "output_sha256": outputs,
    }
    manifest["manifest_id"] = manifest_id(manifest)
    atomic_json(output / "c_target_panel_manifest.json", manifest)
    print(json.dumps({"manifest_id": manifest["manifest_id"], "c_figures": len(c_paths),
                      "target_figures": [square_target.name, regional_target.name],
                      "target_table": target_table.name}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--squares", required=True, type=Path)
    parser.add_argument("--outline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
