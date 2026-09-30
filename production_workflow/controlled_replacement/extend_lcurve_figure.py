#!/usr/bin/env python3
"""Draw the L-curve for the sector-wide inversion from the saved runs.

Regular points are the five r_C values of the refined selection window and the
earlier r_C = 0.005 run, which converged (gradient norm below 1e-3) with the
same scientific code but was stopped by a fixed iteration count rather than
the block-wise stopping rule.  r_C = 0.05 did not meet the convergence
criteria and is drawn as an unconverged point; it is not used for curvature.
The maximum-curvature choice is recomputed from the regular points and must
equal the selected r_C.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcurve_selection import _normalized_log_coordinates, menger_curvature  # noqa: E402

SELECTED_REG_C = 0.01414213562
GRADIENT_THRESHOLD = 1e-3
ACCEPTED_CONFIG = "db5cae4d88c687c6bec6d8967119722732305b4dc24463153167510d429853e7"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def science_sources(manifest: dict) -> dict:
    """Hashes of the scientific source files (src/), keyed by relative path."""
    out = {}
    for path, digest in manifest["source_sha256"].items():
        relative = path.split("/icepack-inverse/", 1)[-1]
        if relative.startswith("src/"):
            out[relative] = digest
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-table", type=Path, required=True)
    parser.add_argument("--extra-converged-manifest", type=Path, required=True,
                        help="point_manifest.json of the r_C = 0.005 run")
    parser.add_argument("--unconverged-manifest", type=Path, required=True,
                        help="point_manifest.json of the r_C = 0.05 run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    accepted = pd.read_csv(args.accepted_table)
    required = {"reg_c", "status", "misfit", "unweighted_roughness"}
    if not required.issubset(accepted.columns) or not accepted["status"].eq("valid").all():
        raise RuntimeError("Accepted L-curve table is invalid")

    unconverged = read_json(args.unconverged_manifest)
    if not (
        unconverged.get("schema") == "jog-production-lcurve-point-v2"
        and unconverged.get("status") == "invalid"
        and unconverged.get("native_termination") == "iteration_limit"
        and unconverged.get("solver_log_crosscheck_passed") is True
        and unconverged.get("metrics", {}).get("all_finite") is True
        and unconverged.get("metrics", {}).get("rol_objective_matches_reassembled") is True
        and unconverged.get("metrics", {}).get("weighted_penalty_identity") is True
        and unconverged.get("config_sha256") == ACCEPTED_CONFIG
    ):
        raise RuntimeError("Unconverged r_C = 0.05 run lacks comparable verified values")

    extra = read_json(args.extra_converged_manifest)
    final_gradient = float(extra["attempts"][-1]["gradient_norm"])
    if not (
        extra.get("schema") == "jog-production-lcurve-point-v1"
        and abs(float(extra.get("reg_c")) - 0.005) < 1e-12
        and extra.get("solver_log_crosscheck_passed") is True
        and extra.get("metrics", {}).get("all_finite") is True
        and extra.get("metrics", {}).get("rol_objective_matches_reassembled") is True
        and extra.get("metrics", {}).get("weighted_penalty_identity") is True
        and final_gradient < GRADIENT_THRESHOLD
    ):
        raise RuntimeError("The r_C = 0.005 run does not meet the convergence checks")
    if science_sources(extra) != science_sources(unconverged) or not science_sources(extra):
        raise RuntimeError("The r_C = 0.005 run used different scientific source code")

    points = pd.concat([
        accepted[["reg_c", "misfit", "unweighted_roughness"]],
        pd.DataFrame([{"reg_c": float(extra["reg_c"]),
                       "misfit": float(extra["metrics"]["misfit"]),
                       "unweighted_roughness": float(extra["metrics"]["unweighted_roughness"])}]),
    ], ignore_index=True).sort_values("reg_c").reset_index(drop=True)

    coordinates = _normalized_log_coordinates(points.to_dict("records"))
    points["curvature"] = [
        menger_curvature(coordinates[i - 1], coordinates[i], coordinates[i + 1])
        if 0 < i < len(points) - 1 else np.nan
        for i in range(len(points))
    ]
    selected = points.loc[points["curvature"].idxmax()]
    if abs(float(selected["reg_c"]) - SELECTED_REG_C) > 1e-9:
        raise RuntimeError(f"Maximum curvature moved to r_C = {selected['reg_c']}")
    ranked = points["curvature"].dropna().sort_values(ascending=False)
    runner_up_gap = 1.0 - float(ranked.iloc[1]) / float(ranked.iloc[0])

    other = unconverged["metrics"]
    figure, axis = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    axis.plot(points["misfit"], points["unweighted_roughness"],
              color="#3b6ea8", linewidth=1.5, zorder=1)
    axis.plot([points["misfit"].iloc[-1], other["misfit"]],
              [points["unweighted_roughness"].iloc[-1], other["unweighted_roughness"]],
              color="0.55", linewidth=1.2, linestyle="--", zorder=1)
    axis.scatter(points["misfit"], points["unweighted_roughness"],
                 color="#3b6ea8", s=34, zorder=2, label=r"candidate $r_C$")
    axis.scatter([other["misfit"]], [other["unweighted_roughness"]],
                 facecolors="white", edgecolors="0.35", linewidths=1.2, s=40, zorder=2,
                 label=r"unconverged ($r_C=0.05$)")
    axis.scatter([selected["misfit"]], [selected["unweighted_roughness"]],
                 marker="*", color="#c43b3b", edgecolor="black", linewidth=0.6,
                 s=190, zorder=4, label=rf"selected $r_C={selected['reg_c']:.6f}$")
    labels = list(zip(points["reg_c"], points["misfit"], points["unweighted_roughness"]))
    labels.append((float(unconverged["reg_c"]), other["misfit"], other["unweighted_roughness"]))
    rightmost = max(misfit for _, misfit, _ in labels)
    for reg_c, misfit, roughness in labels:
        # The rightmost label goes to the left of its point so it stays inside the axes.
        at_edge = misfit == rightmost
        axis.annotate(f"{float(reg_c):.5g}", (misfit, roughness),
                      xytext=(-6, 6) if at_edge else (4, 4), textcoords="offset points",
                      ha="right" if at_edge else "left", fontsize=10)
    axis.set_xscale("log"); axis.set_yscale("log")
    # Label ticks at 1, 2, 3 and 5 times each power of ten; the default
    # labels only powers of ten, which leaves one label on the y axis.
    for log_axis in (axis.xaxis, axis.yaxis):
        log_axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        log_axis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        log_axis.set_minor_formatter(ticker.NullFormatter())
    axis.tick_params(labelsize=11)
    # Both are dimensionless terms of the objective (misfit scaled by U = 1 m/a).
    axis.set_xlabel("Velocity misfit (dimensionless)", fontsize=12)
    axis.set_ylabel("Unweighted roughness (dimensionless)", fontsize=12)
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(frameon=False, fontsize=11)
    png = args.output / "lcurve_appendix_extended.png"
    pdf = args.output / "lcurve_appendix_extended.pdf"
    figure.savefig(png, dpi=240); figure.savefig(pdf)
    plt.close(figure)

    record = {
        # End points have no curvature; write null rather than NaN.
        "points": points.astype(object).where(points.notna(), None).to_dict("records"),
        "selected_reg_c": float(selected["reg_c"]),
        "second_highest_curvature_below_selected": runner_up_gap,
        "extra_converged_point": {
            "reg_c": float(extra["reg_c"]), "final_gradient_norm": final_gradient,
            "gradient_threshold": GRADIENT_THRESHOLD, "source": str(args.extra_converged_manifest),
        },
        "unconverged_point_plotted_not_used": {
            "reg_c": float(unconverged["reg_c"]), "misfit": other["misfit"],
            "unweighted_roughness": other["unweighted_roughness"],
            "native_termination": unconverged["native_termination"],
            "source": str(args.unconverged_manifest),
        },
        "excluded_saved_points": [
            {"reg_c": 0.1, "reason": "failed run has no verified finite objective values"},
        ],
    }
    (args.output / "lcurve_extension_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
