#!/usr/bin/env python3
"""Appendix figure: the modeled domain and the region used for metrics.

Every 450 m MEaSUREs pixel centre inside the model mesh is coloured: blue if
it is an eligible dataset row (used for training and every reported metric),
orange otherwise (phi <= 0.1, or no valid velocity).  Percentages are fractions
of the pixels inside the mesh.  Reads the model mesh and the saved dataset
only; no inversion or simulation is run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.tri as mtri
import meshio
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_revision_figures_and_tables as base  # noqa: E402

OUT = ROOT / "manuscript/figures/appendix"
PIXEL_M = 450.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mesh_path() -> Path:
    # The same lookup as base.quadratic_mesh_triangulation.
    path = base.ARTIFACT_ROOT / "amundsen.msh"
    return path if path.is_file() else ROOT / "amundsen.msh"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="directory for the figure files (default: manuscript/figures/appendix)")
    out = parser.parse_args().output
    base.style()

    mesh = meshio.read(mesh_path())
    triangles = np.concatenate([cells.data for cells in mesh.cells if cells.type == "triangle"])
    vertices = np.asarray(mesh.points[:, :2], dtype=float)
    finder = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles).get_trifinder()

    rows = pd.read_csv(base.DATASET / "canonical_master_dataset.csv.gz",
                       usecols=["x", "y", "common_eligible"])
    eligible_rows = rows[rows["common_eligible"].astype(bool)]

    # Pixel-centre lattice of the MEaSUREs grid, anchored on the dataset rows and
    # covering the mesh and every dataset row with a one-pixel margin.
    x_anchor, y_anchor = float(rows["x"].iloc[0]), float(rows["y"].iloc[0])
    if not (np.allclose(np.remainder(rows["x"] - x_anchor, PIXEL_M), 0.0)
            and np.allclose(np.remainder(rows["y"] - y_anchor, PIXEL_M), 0.0)):
        raise RuntimeError("Dataset rows are not on one 450 m pixel lattice")

    def lattice(anchor: float, low: float, high: float) -> np.ndarray:
        first = anchor + np.floor((low - anchor) / PIXEL_M - 1) * PIXEL_M
        last = anchor + np.ceil((high - anchor) / PIXEL_M + 1) * PIXEL_M
        return np.arange(first, last + PIXEL_M / 2, PIXEL_M)

    xs = lattice(x_anchor, min(vertices[:, 0].min(), rows["x"].min()),
                 max(vertices[:, 0].max(), rows["x"].max()))
    ys = lattice(y_anchor, min(vertices[:, 1].min(), rows["y"].min()),
                 max(vertices[:, 1].max(), rows["y"].max()))
    grid_x, grid_y = np.meshgrid(xs, ys)
    in_mesh = finder(grid_x, grid_y) >= 0

    eligible = np.zeros(grid_x.shape, dtype=bool)
    column = np.rint((eligible_rows["x"].to_numpy() - xs[0]) / PIXEL_M).astype(int)
    row = np.rint((eligible_rows["y"].to_numpy() - ys[0]) / PIXEL_M).astype(int)
    eligible[row, column] = True

    # Category grid: 0 outside the mesh (transparent), 1 inside the mesh but
    # excluded from metrics, 2 eligible and inside the mesh.
    category = np.zeros(grid_x.shape, dtype=np.int8)
    category[in_mesh] = 1
    category[in_mesh & eligible] = 2
    n_inside = int(in_mesh.sum())
    n_excluded = int((in_mesh & ~eligible).sum())
    n_eligible = int((in_mesh & eligible).sum())

    display = np.ma.masked_where(category == 0, category)
    cmap = ListedColormap(["#f4a582", "#4477AA"])  # excluded, eligible
    norm = BoundaryNorm([0.5, 1.5, 2.5], 2)
    extent = [(xs[0] - PIXEL_M / 2) / 1000.0, (xs[-1] + PIXEL_M / 2) / 1000.0,
              (ys[0] - PIXEL_M / 2) / 1000.0, (ys[-1] + PIXEL_M / 2) / 1000.0]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 6.2), constrained_layout=True)
    ax.imshow(display, origin="lower", extent=extent, cmap=cmap, norm=norm,
              interpolation="nearest")
    base.map_axes(ax, vertices)
    base.add_antarctica_locator(ax, vertices)
    ax.plot([], [], color="#4477AA", lw=6,
            label=f"Eligible: used for training and metrics ({n_eligible/n_inside:.1%} of modeled domain)")
    ax.plot([], [], color="#f4a582", lw=6,
            label="Modeled but excluded from metrics ($\\phi\\leq0.1$ or no valid\n"
                  f"velocity observation; {n_excluded/n_inside:.1%} of modeled domain)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1,
              frameon=True, framealpha=0.95, borderaxespad=0.0, fontsize=11)

    out.mkdir(parents=True, exist_ok=True)
    stem = "figure_appendix_eligible_region_map"
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=240)
        paths.append(path)
    plt.close(fig)

    record = {
        "schema": "jog-appendix-eligibility-map-v2",
        "status": "complete",
        "sources": {"mesh": str(mesh_path().name),
                    "dataset": "canonical_master_dataset.csv.gz (x, y, common_eligible)"},
        "pixels_450m": {"inside_modeled_domain": n_inside, "excluded_from_metrics": n_excluded,
                        "eligible": n_eligible},
        "outputs": {path.name: sha256(path) for path in paths},
    }
    (out / f"{stem}_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
