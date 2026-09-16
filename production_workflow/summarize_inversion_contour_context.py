"""Summarize the populations marked by the 100 m/a reference contour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import generate_proposed_figure5 as figure5
import generate_pig_cfg02_spatial_diagnostic as pig_figure


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output/analysis/inversion_reference_contour_context_20260913.json"


def main() -> None:
    inversion = figure5.aligned_inversion_error(figure5.verified_error_fields("CFG02"))
    high = inversion["inversion_error"].to_numpy(float) >= 100.0
    square = {
        "rows": int(len(inversion)),
        "fraction_with_inversion_error_at_least_100": float(high.mean()),
        "inversion_rmse_m_per_a": float(np.sqrt(np.mean(inversion["inversion_error"] ** 2))),
        "configurations": {},
    }
    for config in ("CFG02", "CFG04"):
        model_error = figure5.verified_error_fields(config)["absolute_error"].to_numpy(float)
        square["configurations"][config] = {
            "fraction_of_model_squared_error_where_inversion_error_at_least_100":
                float(np.sum(model_error[high] ** 2) / np.sum(model_error ** 2)),
            "fraction_of_model_error_at_least_100_rows_where_inversion_error_at_least_100":
                float(np.mean(high[model_error >= 100.0])),
        }

    _, pig, _, _ = pig_figure.load_evidence()
    pig_high = pig["inversion_error"].to_numpy(float) >= 100.0
    model_error = pig["model_error"].to_numpy(float)
    payload = {
        "threshold_m_per_a": 100.0,
        "square_footprints": square,
        "PIG_CFG02": {
            "rows": int(len(pig)),
            "fraction_with_inversion_error_at_least_100": float(pig_high.mean()),
            "inversion_rmse_m_per_a": float(np.sqrt(np.mean(pig["inversion_error"] ** 2))),
            "fraction_of_model_squared_error_where_inversion_error_at_least_100":
                float(np.sum(model_error[pig_high] ** 2) / np.sum(model_error ** 2)),
            "fraction_of_model_error_at_least_100_rows_where_inversion_error_at_least_100":
                float(np.mean(pig_high[model_error >= 100.0])),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
