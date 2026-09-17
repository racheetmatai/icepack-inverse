#!/usr/bin/env python3
"""Regenerate inversion speed and residual panels with original MEaSUREs values."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


WORKFLOW = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKFLOW))
import generate_revision_figures_and_tables as base


HERE = Path(__file__).resolve().parent
DATASET = Path(os.environ.get(
    "JOG_CANONICAL_DATASET",
    WORKFLOW / "gate2_results/gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz",
))
OBSERVATIONS = Path(os.environ.get(
    "JOG_CORRECTED_OBSERVATIONS", HERE / "observation_audit/corrected_observations.csv.gz"
))
OUT = Path(os.environ.get("JOG_CONTROLLED_FIGURES", HERE / "figures"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    grid = base.grid_context()
    xg, yg = grid["x_grid"], grid["y_grid"]
    sums = {name: np.zeros((len(yg), len(xg)), dtype=float) for name in ("speed", "error")}
    count = np.zeros((len(yg), len(xg)), dtype=np.int64)
    canonical = pd.read_csv(
        DATASET,
        usecols=["row_id", "x", "y", "common_eligible", "inversion_vx", "inversion_vy"],
        low_memory=False,
    )
    observations = pd.read_csv(OBSERVATIONS)
    if canonical["row_id"].duplicated().any() or observations["row_id"].duplicated().any():
        raise RuntimeError("Observation or canonical row IDs are not unique")
    frame = canonical.merge(observations, on="row_id", how="inner", validate="one_to_one")
    frame = frame.loc[frame["common_eligible"].astype(bool)].copy()
    if len(frame) != 1_530_992:
        raise RuntimeError(f"Unexpected eligible row count: {len(frame)}")
    ix = np.rint((frame["x"].to_numpy() - xg[0]) / 5000.0).astype(int)
    iy = np.rint((frame["y"].to_numpy() - yg[0]) / 5000.0).astype(int)
    valid = (ix >= 0) & (ix < len(xg)) & (iy >= 0) & (iy < len(yg))
    speed = np.hypot(frame["observed_vx_raw"], frame["observed_vy_raw"]).to_numpy(float)
    error = np.hypot(
        frame["inversion_vx"] - frame["observed_vx_raw"],
        frame["inversion_vy"] - frame["observed_vy_raw"],
    ).to_numpy(float)
    valid &= np.isfinite(speed) & np.isfinite(error)
    np.add.at(sums["speed"], (iy[valid], ix[valid]), speed[valid])
    np.add.at(sums["error"], (iy[valid], ix[valid]), error[valid])
    np.add.at(count, (iy[valid], ix[valid]), 1)
    result = {"x_grid": xg, "y_grid": yg, "count": count}
    for name, values in sums.items():
        result[name] = np.divide(values, count, out=np.full_like(values, np.nan), where=count > 0)
    return grid, result


def draw(kind: str, region: dict[str, np.ndarray], values: dict[str, np.ndarray]) -> Path:
    x, y = region["x_grid"] / 1000.0, region["y_grid"] / 1000.0
    extent = [x[0] - 2.5, x[-1] + 2.5, y[0] - 2.5, y[-1] + 2.5]
    array = np.ma.masked_where(~region["eligible"] | ~np.isfinite(values[kind]), values[kind])
    fig, ax = plt.subplots(figsize=(4.6, 4.45), constrained_layout=True)
    if kind == "speed":
        image = ax.imshow(
            array, origin="lower", extent=extent, cmap="viridis",
            norm=LogNorm(vmin=max(1, np.nanpercentile(array.compressed(), 1)),
                         vmax=np.nanpercentile(array.compressed(), 99.5)),
        )
        stem = "figure3a_observed_speed"
        label = r"m a$^{-1}$"
    else:
        image = ax.imshow(
            array, origin="lower", extent=extent, cmap="inferno",
            norm=LogNorm(vmin=max(0.5, np.nanpercentile(array.compressed(), 0.1)),
                         vmax=np.nanmax(array.compressed())),
        )
        stem = "figure3c_inversion_velocity_residual"
        label = r"Vector error (m a$^{-1}$; log scale)"
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.02)
    colorbar.set_label(label)
    base.map_axes(ax, region["outline"])
    base.add_antarctica_locator(ax, region["outline"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def main() -> None:
    base.style()
    region, values = aggregate()
    outputs = [draw(kind, region, values) for kind in ("speed", "error")]
    record = {
        "schema": "jog-corrected-inversion-panels-v1",
        "status": "complete",
        "observations": "original paired MEaSUREs raster values at verified pixel centres",
        "aggregation": "mean within 5 km display cells",
        "outputs": {path.name: sha256(path) for path in outputs},
        "inputs": {"dataset": sha256(DATASET), "corrected_observations": sha256(OBSERVATIONS)},
    }
    (OUT / "figure3_corrected_panels_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
