#!/usr/bin/env python3
"""Extend the accepted L-curve artwork using verified saved evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-table", type=Path, required=True)
    parser.add_argument("--unconverged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    accepted = pd.read_csv(args.accepted_table)
    required = {"reg_c", "status", "misfit", "unweighted_roughness"}
    if not required.issubset(accepted.columns) or not accepted["status"].eq("valid").all():
        raise RuntimeError("Accepted L-curve table is invalid")
    candidate = read_json(args.unconverged_manifest)
    if not (
        candidate.get("schema") == "jog-production-lcurve-point-v2"
        and candidate.get("status") == "invalid"
        and candidate.get("native_termination") == "iteration_limit"
        and candidate.get("solver_log_crosscheck_passed") is True
        and candidate.get("state_reusable") is True
        and candidate.get("metrics", {}).get("all_finite") is True
        and candidate.get("metrics", {}).get("rol_objective_matches_reassembled") is True
        and candidate.get("metrics", {}).get("weighted_penalty_identity") is True
    ):
        raise RuntimeError("Saved unconverged candidate lacks comparable verified objective values")
    accepted_config = "db5cae4d88c687c6bec6d8967119722732305b4dc24463153167510d429853e7"
    if candidate.get("config_sha256") != accepted_config:
        raise RuntimeError("Unconverged candidate uses a different production configuration")

    figure, axis = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    axis.plot(accepted["misfit"], accepted["unweighted_roughness"],
              color="#3b6ea8", linewidth=1.5, zorder=1)
    axis.scatter(accepted["misfit"], accepted["unweighted_roughness"],
                 color="#3b6ea8", s=34, zorder=2, label="valid converged point")
    selected = accepted.loc[(accepted["reg_c"] - 0.01414213562).abs().idxmin()]
    axis.scatter([selected["misfit"]], [selected["unweighted_roughness"]],
                 marker="*", color="#c43b3b", edgecolor="black", linewidth=0.6,
                 s=190, zorder=4, label=r"selected $r_C=0.014142$")
    metrics = candidate["metrics"]
    axis.scatter([metrics["misfit"]], [metrics["unweighted_roughness"]],
                 facecolors="none", edgecolors="#3b6ea8", linewidths=1.4,
                 s=46, zorder=3)
    for _, point in accepted.iterrows():
        axis.annotate(f"{float(point['reg_c']):.5g}",
                      (point["misfit"], point["unweighted_roughness"]),
                      xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.annotate(f"{float(candidate['reg_c']):.5g}",
                  (metrics["misfit"], metrics["unweighted_roughness"]),
                  xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xlabel("Observation-mean velocity misfit")
    axis.set_ylabel("Unweighted roughness")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(frameon=False)
    png = args.output / "lcurve_appendix_extended.png"
    pdf = args.output / "lcurve_appendix_extended.pdf"
    figure.savefig(png, dpi=240); figure.savefig(pdf)
    plt.close(figure)

    record = {
        "accepted_points": accepted[["reg_c", "misfit", "unweighted_roughness",
                                      "native_termination"]].to_dict("records"),
        "added_unconverged_point": {
            "reg_c": candidate["reg_c"], "misfit": metrics["misfit"],
            "unweighted_roughness": metrics["unweighted_roughness"],
            "native_termination": candidate["native_termination"],
            "status": candidate["status"], "source": str(args.unconverged_manifest),
        },
        "excluded_saved_points": [
            {"reg_c": 0.005, "reason": "older v1 protocol and different production-config hash"},
            {"reg_c": 0.1, "reason": "failed point has no verified finite objective metrics"},
        ],
        "selection_unchanged": 0.01414213562,
    }
    (args.output / "lcurve_extension_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
