"""Generate the frozen composite replacement for legacy Figure 5."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORTING = ROOT / "production_workflow/gate4_forward_reporting_20260829_b"
EVALUATION = ROOT / "production_workflow/gate4_forward_evaluation_20260829_a"
OUTPUT = ROOT / "production_workflow/final_figures_20260830_a"

CONFIGS = [f"CFG{i:02d}" for i in range(1, 7)]
LABELS = {
    "CFG01": "CFG01",
    "CFG02": "CFG02",
    "CFG03": "CFG03",
    "CFG04": "CFG04",
    "CFG05": "CFG05",
    "CFG06": "CFG06",
}
COLORS = {
    "CFG01": "#4477AA",
    "CFG02": "#66CCEE",
    "CFG03": "#228833",
    "CFG04": "#CCBB44",
    "CFG05": "#EE6677",
    "CFG06": "#AA3377",
}


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    reporting_manifest = json.loads(
        (REPORTING / "reporting_manifest.json").read_text(encoding="utf-8")
    )
    evaluation_manifest = json.loads(
        (EVALUATION / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    if reporting_manifest.get("status") != "complete":
        raise ValueError("Reporting bundle is not complete")
    if evaluation_manifest.get("status") != "complete":
        raise ValueError("Evaluation bundle is not complete")
    if reporting_manifest["evaluation_manifest_id"] != evaluation_manifest["manifest_id"]:
        raise ValueError("Reporting/evaluation manifest identity mismatch")

    primary = pd.read_csv(REPORTING / "primary_median_summary.csv")
    controls = pd.read_csv(EVALUATION / "control_population_metrics.csv")
    squares = primary.loc[
        primary["experiment"].str.fullmatch(r"SQ\d{2}")
        & primary["population"].eq("central_50km")
        & primary["support_stratum"].eq("all")
    ].copy()
    pig = controls.loc[
        controls["experiment"].eq("REG_PIG")
        & controls["population"].eq("PIG")
        & controls["support_stratum"].eq("all")
        & controls["configuration"].isin(["CFG01", "CFG02", "CFG03"])
        & controls["control_kind"].isin(["member", "median"])
    ].copy()

    if len(squares) != 60 or squares["experiment"].nunique() != 10:
        raise ValueError("Expected 60 square/configuration primary rows")
    if set(squares["configuration"]) != set(CONFIGS):
        raise ValueError("Square figure is missing a frozen configuration")
    if len(pig) != 33:
        raise ValueError("Expected 30 PIG members and three median controls")
    counts = pig.groupby(["configuration", "control_kind"]).size().to_dict()
    for config in ["CFG01", "CFG02", "CFG03"]:
        if counts.get((config, "member")) != 10 or counts.get((config, "median")) != 1:
            raise ValueError(f"Unexpected PIG control count for {config}")
    return squares, pig, reporting_manifest, evaluation_manifest


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_panel_a(ax_a: plt.Axes, squares: pd.DataFrame, show_title: bool = True) -> None:

    pivot = squares.pivot(
        index="experiment", columns="configuration", values="vector_rmse_m_per_a"
    ).reindex(columns=CONFIGS)
    x = np.arange(len(CONFIGS), dtype=float)
    minimum = pivot.min(axis=0).to_numpy(float)
    maximum = pivot.max(axis=0).to_numpy(float)
    ax_a.fill_between(
        x, minimum, maximum, color="0.72", alpha=0.22, linewidth=0,
        zorder=0,
    )
    for _, row in pivot.iterrows():
        ax_a.plot(x, row.to_numpy(float), color="0.72", lw=0.75, alpha=0.65, zorder=1)
        for index, config in enumerate(CONFIGS):
            ax_a.scatter(
                index,
                float(row[config]),
                s=18,
                facecolor="white",
                edgecolor=COLORS[config],
                linewidth=0.85,
                alpha=0.9,
                zorder=2,
            )
    summary = squares.groupby("configuration")["vector_rmse_m_per_a"].median().reindex(CONFIGS)
    for index, config in enumerate(CONFIGS):
        median = float(summary.loc[config])
        ax_a.plot(
            [index - 0.18, index + 0.18], [median, median],
            color=COLORS[config], lw=3.2, solid_capstyle="butt", zorder=4
        )
    ax_a.set_yscale("log")
    # Reserve a genuinely empty band above the observations for the in-panel legend.
    ax_a.set_ylim(2.5, 900)
    ax_a.set_xticks(x, [LABELS[config] for config in CONFIGS])
    ax_a.set_ylabel(r"Vector RMSE (m a$^{-1}$; log scale)")
    if show_title:
        ax_a.set_title("Ten separate square holdouts", loc="left", fontweight="bold")
    ax_a.grid(axis="y", which="major", color="0.88", lw=0.65)
    ax_a.tick_params(axis="x", length=0, pad=6)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.legend(
        handles=[
            Patch(facecolor="0.72", edgecolor="none", alpha=0.22,
                  label="Shaded area: minimum to maximum across 10 squares"),
            Line2D([0], [0], color="0.72", lw=0.9, marker="o", markersize=4,
                   markerfacecolor="white", markeredgecolor="0.5",
                   label="Circle and gray line: one held-out square"),
            Line2D([0], [0], color="0.25", lw=3.2,
                   label="Short colored line: median across 10 squares"),
        ],
        loc="upper left", ncol=1,
        frameon=True, fancybox=False, framealpha=0.96,
        edgecolor="0.55", facecolor="white", borderpad=0.45,
    )


def draw_panel_b(ax_b: plt.Axes, pig: pd.DataFrame, show_title: bool = True) -> None:
    pig_x = np.arange(3, dtype=float)
    member_offsets = np.linspace(-0.12, 0.12, 10)
    for index, config in enumerate(["CFG01", "CFG02", "CFG03"]):
        subset = pig.loc[pig["configuration"].eq(config)]
        member_values = subset.loc[
            subset["control_kind"].eq("member"), "vector_rmse_m_per_a"
        ].to_numpy(float)
        median = float(subset.loc[
            subset["control_kind"].eq("median"), "vector_rmse_m_per_a"
        ].iloc[0])
        ax_b.scatter(
            index + member_offsets, member_values, s=20, facecolor="white",
            edgecolor=COLORS[config], linewidth=0.85, zorder=2
        )
        ax_b.scatter(
            index, median, marker="D", s=52, color=COLORS[config],
            edgecolor="black", linewidth=0.55, zorder=3
        )
    uniform = float(pig["uniform_vector_rmse_m_per_a"].iloc[0])
    inversion = float(pig["inversion_vector_rmse_m_per_a"].iloc[0])
    ax_b.axhline(uniform, color="0.25", lw=1.2, ls="--", label="Uniform C")
    ax_b.axhline(inversion, color="0.5", lw=1.2, ls=":", label="Inversion reference")
    ax_b.set_ylim(0, 425)
    ax_b.set_xticks(pig_x, [LABELS[config] for config in ["CFG01", "CFG02", "CFG03"]])
    ax_b.set_ylabel(r"Vector RMSE (m a$^{-1}$)")
    if show_title:
        ax_b.set_title("PIG catchment holdout", loc="left", fontweight="bold")
    ax_b.grid(axis="y", color="0.88", lw=0.65)
    ax_b.tick_params(axis="x", length=0, pad=6)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(
        handles=[
            Line2D([0], [0], color="0.25", lw=1.2, ls="--", label="Uniform C"),
            Line2D([0], [0], color="0.5", lw=1.2, ls=":", label="Inversion reference"),
            Line2D([0], [0], color="0.5", lw=0, marker="o", markersize=4,
                   markerfacecolor="white", markeredgecolor="0.5",
                   label="Circle: one trained member\n(10 per configuration)"),
            Line2D([0], [0], color="0.5", lw=0, marker="D", markersize=6,
                   markerfacecolor="0.55", markeredgecolor="black",
                   label="Diamond: forward run using\nthe median C field"),
        ],
        loc="center", bbox_to_anchor=(0.52, 0.47),
        frameon=True, fancybox=False, framealpha=1.0,
        edgecolor="0.35", facecolor="white", borderpad=0.55,
    )


def draw(squares: pd.DataFrame, pig: pd.DataFrame) -> plt.Figure:
    configure_style()
    figure = plt.figure(figsize=(8.2, 5.25), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=[2.35, 1.0])
    draw_panel_a(figure.add_subplot(grid[0, 0]), squares, show_title=True)
    draw_panel_b(figure.add_subplot(grid[0, 1]), pig, show_title=True)
    return figure


def draw_standalone_panel_a(squares: pd.DataFrame) -> plt.Figure:
    configure_style()
    figure, axis = plt.subplots(figsize=(5.3, 5.25), constrained_layout=True)
    draw_panel_a(axis, squares, show_title=False)
    return figure


def draw_standalone_panel_b(pig: pd.DataFrame) -> plt.Figure:
    configure_style()
    figure, axis = plt.subplots(figsize=(3.9, 5.25), constrained_layout=True)
    draw_panel_b(axis, pig, show_title=False)
    return figure


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    squares, pig, reporting_manifest, evaluation_manifest = load_inputs()
    figure = draw(squares, pig)
    panel_a = draw_standalone_panel_a(squares)
    panel_b = draw_standalone_panel_b(pig)
    stems = {
        "png": OUTPUT / "figure5_revised_performance.png",
        "svg": OUTPUT / "figure5_revised_performance.svg",
        "pdf": OUTPUT / "figure5_revised_performance.pdf",
    }
    panel_stems = {
        "panel_a_png": OUTPUT / "figure5a_square_holdouts.png",
        "panel_a_svg": OUTPUT / "figure5a_square_holdouts.svg",
        "panel_a_pdf": OUTPUT / "figure5a_square_holdouts.pdf",
        "panel_b_png": OUTPUT / "figure5b_pig_holdout.png",
        "panel_b_svg": OUTPUT / "figure5b_pig_holdout.svg",
        "panel_b_pdf": OUTPUT / "figure5b_pig_holdout.pdf",
    }
    figure.savefig(stems["png"], dpi=300, facecolor="white", bbox_inches="tight")
    figure.savefig(stems["svg"], facecolor="white", bbox_inches="tight")
    figure.savefig(stems["pdf"], facecolor="white", bbox_inches="tight")
    plt.close(figure)
    panel_a.savefig(panel_stems["panel_a_png"], dpi=300, facecolor="white", bbox_inches="tight")
    panel_a.savefig(panel_stems["panel_a_svg"], facecolor="white", bbox_inches="tight")
    panel_a.savefig(panel_stems["panel_a_pdf"], facecolor="white", bbox_inches="tight")
    plt.close(panel_a)
    panel_b.savefig(panel_stems["panel_b_png"], dpi=300, facecolor="white", bbox_inches="tight")
    panel_b.savefig(panel_stems["panel_b_svg"], facecolor="white", bbox_inches="tight")
    panel_b.savefig(panel_stems["panel_b_pdf"], facecolor="white", bbox_inches="tight")
    plt.close(panel_b)

    summary = squares.groupby("configuration")["vector_rmse_m_per_a"].median()
    manifest = {
        "schema": "jog-final-figure5-v3",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_manifest_id": reporting_manifest["manifest_id"],
        "evaluation_manifest_id": evaluation_manifest["manifest_id"],
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": {
            "primary_median_summary.csv": sha256_file(REPORTING / "primary_median_summary.csv"),
            "control_population_metrics.csv": sha256_file(EVALUATION / "control_population_metrics.csv"),
        },
        "checks": {
            "independent_square_count": int(squares["experiment"].nunique()),
            "square_configuration_count": int(squares["configuration"].nunique()),
            "square_case_count": int(len(squares)),
            "pig_configuration_count": int(pig["configuration"].nunique()),
            "pig_member_control_count": int(pig["control_kind"].eq("member").sum()),
            "pig_median_control_count": int(pig["control_kind"].eq("median").sum()),
            "pig_is_secondary_single_stress_test": True,
        },
        "square_median_rmse_m_per_a": {
            config: float(summary.loc[config]) for config in CONFIGS
        },
        "output_sha256": {
            path.name: sha256_file(path)
            for path in [*stems.values(), *panel_stems.values()]
        },
    }
    manifest["manifest_id"] = canonical_id(manifest)
    (OUTPUT / "figure5_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "manifest_id": manifest["manifest_id"],
                      "output": str(stems["png"])}, indent=2))


if __name__ == "__main__":
    main()
