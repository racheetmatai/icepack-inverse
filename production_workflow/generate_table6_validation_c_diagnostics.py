#!/usr/bin/env python3
"""Named generator for manuscript Table 6 (validation C RMSE / R_C^2).

Table 6 ("Validation agreement with the inversion-reference control") had no
dedicated generator script; its values were only reproducible by manually
reading each member's saved validation_predictions.csv.gz. This script
aggregates those saved per-member files (no retraining, no new predictions)
into exactly Table 6's numbers, closing that traceability gap.

Requires the CUDA training campaign's per-run validation predictions, e.g.
from cuda_results/JOG_PRODUCTION_RESULTS_20260828/<audit-id>/runs/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SQUARES = [f"SQ{i:02d}" for i in range(1, 11)]
CONFIGS = [f"CFG{i:02d}" for i in range(1, 7)]


def member_rmse_r2(path: Path) -> tuple[float, float]:
    df = pd.read_csv(path, usecols=["reference_log_C", "predicted_log_C"])
    ref = df["reference_log_C"].to_numpy(float)
    pred = df["predicted_log_C"].to_numpy(float)
    resid = pred - ref
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return rmse, r2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True,
                         help="Directory containing one subdirectory per run, e.g. SQ01_CFG01_M01/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for square in SQUARES:
        for configuration in CONFIGS:
            rmses, r2s = [], []
            for member in range(1, 11):
                run_dir = args.runs_root / f"{square}_{configuration}_M{member:02d}"
                path = run_dir / "validation_predictions.csv.gz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                rmse, r2 = member_rmse_r2(path)
                rmses.append(rmse)
                r2s.append(r2)
            rows.append({
                "square": square,
                "configuration": configuration,
                "member_rmse": rmses,
                "member_r2": r2s,
                "median_rmse": float(np.median(rmses)),
                "median_r2": float(np.median(r2s)),
            })

    table = pd.DataFrame(rows)
    table[["square", "configuration", "median_rmse", "median_r2"]].to_csv(
        args.output / "table6_validation_c_diagnostics.csv", index=False)

    # A compact square x configuration cell matching the manuscript layout.
    display = table.pivot(index="square", columns="configuration",
                           values=["median_rmse", "median_r2"])
    cells = pd.DataFrame(index=SQUARES, columns=CONFIGS, dtype=object)
    for square in SQUARES:
        for configuration in CONFIGS:
            row = table[(table.square == square) & (table.configuration == configuration)].iloc[0]
            cells.loc[square, configuration] = f"{row.median_rmse:.3f}/{row.median_r2:.3f}"
    cells.to_csv(args.output / "table6_display_cells.csv")

    manifest = {
        "schema": "jog-table6-validation-c-diagnostics-v1",
        "status": "complete",
        "source": str(args.runs_root),
        "n_runs_used": len(SQUARES) * len(CONFIGS) * 10,
        "manuscript_check_sq01_cfg01": {
            "expected": "0.077/0.984",
            "computed": f"{table[(table.square=='SQ01') & (table.configuration=='CFG01')].iloc[0].median_rmse:.3f}"
                        f"/{table[(table.square=='SQ01') & (table.configuration=='CFG01')].iloc[0].median_r2:.3f}",
        },
    }
    (args.output / "table6_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
