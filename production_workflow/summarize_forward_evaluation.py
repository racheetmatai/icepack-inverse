"""Create frozen aggregate tables and a complete median-field spatial atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


CONFIGS = [f"CFG{i:02d}" for i in range(1, 7)]
CONFIG_LABELS = {
    "CFG01": "All ice", "CFG02": "Selected ice", "CFG03": "All geophysical",
    "CFG04": "Selected geophysical", "CFG05": "Selected combined",
    "CFG06": "Selected combined + alignment",
}
CONTRASTS = [("CFG02", "CFG01"), ("CFG04", "CFG03"),
             ("CFG05", "CFG02"), ("CFG05", "CFG04"), ("CFG06", "CFG05")]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def primary_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["support_stratum"].eq("all")
        & ((frame["experiment"].str.startswith("SQ") & frame["population"].eq("central_50km"))
           | (frame["experiment"].eq("REG_INTER") & frame["population"].eq("both_corridors"))
           | (frame["experiment"].eq("REG_PIG") & frame["population"].eq("PIG")))
    )


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def tables(root: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    medians = pd.read_csv(root / "median_population_metrics.csv")
    members = pd.read_csv(root / "ensemble_member_summary.csv")
    primary = medians.loc[primary_mask(medians)].copy()
    primary.to_csv(output / "primary_median_summary.csv", index=False)
    squares = primary.loc[primary["experiment"].str.startswith("SQ")]
    aggregate = squares.groupby("configuration").agg(
        independent_squares=("experiment", "nunique"),
        vector_RMSE_median_m_per_a=("vector_rmse_m_per_a", "median"),
        vector_RMSE_q25_m_per_a=("vector_rmse_m_per_a", lambda x: x.quantile(.25)),
        vector_RMSE_q75_m_per_a=("vector_rmse_m_per_a", lambda x: x.quantile(.75)),
        P_exp_median_percent=("P_exp_percent", "median"),
        squares_better_than_uniform=("P_exp_percent", lambda x: int((x > 0).sum())),
    ).reset_index()
    aggregate.to_csv(output / "square_equal_weight_summary.csv", index=False)
    pivot = squares.pivot(index="experiment", columns="configuration", values="vector_rmse_m_per_a")
    contrast_rows = []
    for first, second in CONTRASTS:
        difference = pivot[first] - pivot[second]
        wins = int((difference < 0).sum()); losses = int((difference > 0).sum())
        contrast_rows.append({
            "contrast": f"{first}_minus_{second}", "first": first, "second": second,
            "interpretation": "negative favors first configuration",
            "median_paired_RMSE_difference_m_per_a": float(difference.median()),
            "mean_paired_RMSE_difference_m_per_a": float(difference.mean()),
            "first_better_squares": wins, "second_better_squares": losses,
            "ties": int((difference == 0).sum()), "exact_two_sided_sign_p": exact_sign_p(wins, losses),
        })
    order = np.argsort([row["exact_two_sided_sign_p"] for row in contrast_rows])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, contrast_rows[index]["exact_two_sided_sign_p"] * (len(order) - rank)))
        contrast_rows[index]["Holm_adjusted_p"] = running
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(output / "paired_square_contrasts.csv", index=False)
    primary.loc[~primary["experiment"].str.startswith("SQ")].to_csv(
        output / "regional_primary_summary.csv", index=False
    )
    return primary, members, contrasts


def ensemble_band(members: pd.DataFrame, experiment: str, config: str, population: str):
    row = members.loc[
        members["experiment"].eq(experiment) & members["configuration"].eq(config)
        & members["population"].eq(population) & members["support_stratum"].eq("all")
    ]
    if len(row) != 1:
        raise RuntimeError(f"Missing ensemble summary: {experiment}/{config}/{population}")
    row = row.iloc[0]
    return (float(row["vector_rmse_m_per_a__q05"]), float(row["vector_rmse_m_per_a__q95"]))


def square_figure(primary: pd.DataFrame, members: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(15.5, 6.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, .55, 6))
    for number, axis in enumerate(axes.ravel(), start=1):
        experiment = f"SQ{number:02d}"; subset = primary.loc[primary["experiment"].eq(experiment)]
        for index, config in enumerate(CONFIGS):
            row = subset.loc[subset["configuration"].eq(config)].iloc[0]
            low, high = ensemble_band(members, experiment, config, "central_50km")
            axis.vlines(index, low, high, color=colors[index], lw=2, alpha=.75)
            axis.plot(index, row["vector_rmse_m_per_a"], "o", color=colors[index], ms=5)
        uniform = float(subset["uniform_vector_rmse_m_per_a"].iloc[0])
        inversion = float(subset["inversion_vector_rmse_m_per_a"].iloc[0])
        axis.axhline(uniform, color="0.25", lw=1.1, ls="--", label="Uniform C" if number == 1 else None)
        axis.axhline(inversion, color="0.55", lw=1.1, ls=":", label="Inversion" if number == 1 else None)
        axis.set_yscale("log"); axis.set_title(experiment); axis.grid(axis="y", alpha=.2)
        axis.set_xticks(range(6), [str(i) for i in range(1, 7)])
        if number in (1, 6): axis.set_ylabel("Vector RMSE (m a$^{-1}$)")
        if number >= 6: axis.set_xlabel("Feature configuration")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Central-square velocity error: median field and 5–95% member range", fontsize=13)
    figure.savefig(output / "central_square_rmse.png", dpi=240)
    figure.savefig(output / "central_square_rmse.svg")
    plt.close(figure)


def regional_figure(primary: pd.DataFrame, members: pd.DataFrame, output: Path) -> None:
    specs = [("REG_INTER", "both_corridors", ["CFG04", "CFG05", "CFG06"]),
             ("REG_PIG", "PIG", ["CFG02", "CFG01", "CFG03"])]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    for axis, (experiment, population, configs) in zip(axes, specs):
        subset = primary.loc[primary["experiment"].eq(experiment)]
        for index, config in enumerate(configs):
            row = subset.loc[subset["configuration"].eq(config)].iloc[0]
            low, high = ensemble_band(members, experiment, config, population)
            axis.vlines(index, low, high, color=plt.cm.tab10(index), lw=3, alpha=.75)
            axis.plot(index, row["vector_rmse_m_per_a"], "o", color=plt.cm.tab10(index), ms=6)
        axis.axhline(float(subset["uniform_vector_rmse_m_per_a"].iloc[0]), color="0.25", ls="--", lw=1.2)
        axis.axhline(float(subset["inversion_vector_rmse_m_per_a"].iloc[0]), color="0.55", ls=":", lw=1.2)
        axis.set_xticks(range(3), configs); axis.set_ylabel("Vector RMSE (m a$^{-1}$)")
        axis.set_title("Inter-catchment corridors" if experiment == "REG_INTER" else "PIG holdout")
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Regional transfer stress tests: median field and 5–95% member range")
    figure.savefig(output / "regional_rmse.png", dpi=240)
    figure.savefig(output / "regional_rmse.svg")
    plt.close(figure)


def grid(values: np.ndarray, x: np.ndarray, y: np.ndarray):
    ux = np.unique(x); uy = np.unique(y)
    result = np.full((len(uy), len(ux)), np.nan, dtype=np.float64)
    result[np.searchsorted(uy, y), np.searchsorted(ux, x)] = values
    return result, [ux.min(), ux.max(), uy.min(), uy.max()]


def map_atlas(root: Path, output: Path) -> None:
    destination = output / "median_spatial_atlas"; destination.mkdir(exist_ok=True)
    for path in sorted((root / "median_map_data").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            x = archive["x"]; y = archive["y"]
            pred_speed = np.hypot(archive["predicted_vx"], archive["predicted_vy"])
            obs_speed = np.hypot(archive["observed_vx"], archive["observed_vy"])
            signed_speed = pred_speed - obs_speed
            error = archive["error_magnitude"]
            improvement = archive["signed_local_squared_error_improvement"]
            support = archive["support_category"].astype(float)
        arrays = [pred_speed, obs_speed, signed_speed, error, improvement, support]
        gridded = [grid(values, x, y)[0] for values in arrays]; extent = grid(pred_speed, x, y)[1]
        speed_max = float(np.nanquantile(np.concatenate([pred_speed, obs_speed]), .99))
        bias_limit = float(np.nanquantile(np.abs(signed_speed), .99)) or 1.0
        error_max = float(np.nanquantile(error, .99)) or 1.0
        improvement_limit = float(np.nanquantile(np.abs(improvement), .99)) or 1.0
        settings = [
            ("Predicted speed", "viridis", 0, speed_max), ("Observed speed", "viridis", 0, speed_max),
            ("Signed speed bias", "RdBu_r", -bias_limit, bias_limit), ("Vector error", "magma", 0, error_max),
            ("Local squared-error improvement", "RdBu", -improvement_limit, improvement_limit),
            ("Input support category", None, 0, 3),
        ]
        figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
        for axis, data, (title, cmap, vmin, vmax) in zip(axes.ravel(), gridded, settings):
            if title == "Input support category":
                categorical_cmap = ListedColormap(["#7f3b08", "#fdb863", "#5ab4ac", "#01665e"])
                image = axis.imshow(data, origin="lower", extent=extent, cmap=categorical_cmap,
                                    norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], categorical_cmap.N),
                                    interpolation="nearest", aspect="equal")
            else:
                image = axis.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                                    interpolation="nearest", aspect="equal")
            axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
            label = "category" if "category" in title else ("(m a$^{-1}$)$^2$" if "squared" in title else "m a$^{-1}$")
            colorbar = figure.colorbar(image, ax=axis, shrink=.78, label=label)
            if title == "Input support category":
                colorbar.set_ticks([0, 1, 2, 3], labels=["neither", "marginal only", "joint only", "both"])
        figure.suptitle(path.stem.replace("_MEDIAN", "").replace("_", " ") + " — primary held-out population", fontsize=13)
        figure.savefig(destination / f"{path.stem}.png", dpi=200)
        plt.close(figure)


def run(args) -> dict:
    root = args.evaluation_root.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((root / "evaluation_manifest.json").read_text())
    if evaluation.get("status") != "complete" or canonical_id(evaluation) != evaluation.get("manifest_id"):
        raise ValueError("Evaluation bundle is not complete")
    primary, members, contrasts = tables(root, output)
    square_figure(primary, members, output); regional_figure(primary, members, output)
    map_atlas(root, output)
    outputs = {path.relative_to(output).as_posix(): sha256_file(path)
               for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "reporting_manifest.json")}
    manifest = {
        "schema": "jog-forward-evaluation-reporting-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_manifest_id": evaluation["manifest_id"], "spatial_atlas_count": 66,
        "paired_contrasts": len(contrasts), "output_sha256": outputs,
    }
    manifest["manifest_id"] = canonical_id(manifest)
    (output / "reporting_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "manifest_id": manifest["manifest_id"],
                      "files": len(outputs)}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
