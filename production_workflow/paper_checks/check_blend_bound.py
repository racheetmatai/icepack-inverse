"""Check the appendix bound on the edge transition of the replaced region.

The manuscript states that the control used in the forward simulations equals
the median MLP prediction except in mesh cells that cross the edge of the
replaced region, and that this changes the C RMSE against the
inversion-reference C by at most 0.004. For every square and configuration
(complete 130 km footprint and central 50 km square) and for PIG CFG02, this
script compares the C RMSE of the median prediction with that of the control
actually used in the simulation. It reads archived fields only; nothing is
retrained or re-simulated.

Run in the Icepack/Firedrake environment:

    python production_workflow/paper_checks/check_blend_bound.py \
        --artifact-dir /path/to/unpacked_artifacts --output blend_bound.csv

Expected result: 0 affected rows in every central square, no change for PIG at
four decimals, and a largest footprint change of 0.0038 (SQ01 CFG04).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "production_workflow"))
sys.path.insert(0, str(PACKAGE / "production_workflow/controlled_replacement"))
from evaluate_forward_campaign import build_object, build_observation_alignment  # noqa: E402
from evaluate_controlled_campaign import interpolate_control, population_mask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="Unpacked Zenodo artifacts (contains production_runs/ and production_workflow/)")
    parser.add_argument("--output", type=Path, default=Path("blend_bound.csv"))
    args = parser.parse_args()
    artifacts = args.artifact_dir.resolve()
    runs = artifacts / "production_runs"
    controls = artifacts / "production_workflow/controlled_replacement_20260917_a/controls"

    object_, adoption, _ = build_object(PACKAGE / "production_workflow/amundsen_production_config.json",
                                        artifacts, runs / "gate1_definitive_inversion_adoption_20260819_a.json")
    frame, lookup = build_observation_alignment(
        object_, runs / "gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz")
    reference = interpolate_control(object_, np.asarray(adoption["C"].dat.data_ro, dtype=np.float64), lookup)

    cases = [("REG_PIG", "CFG02")] + [(f"SQ{s:02d}", f"CFG{c:02d}") for s in range(1, 11) for c in range(1, 7)]
    rows = []
    for experiment, config in cases:
        median = np.load(runs / f"gate3_full_mesh_ensemble_predictions_20260828_a/{experiment}_{config}.npz")["median_log_C"]
        applied = np.load(controls / f"CR_{experiment}_{config}_MEDIAN.npz")["control_C"]
        c_median = interpolate_control(object_, median.astype(np.float64), lookup)
        c_applied = interpolate_control(object_, applied.astype(np.float64), lookup)
        if experiment == "REG_PIG":
            populations = {"PIG": population_mask(frame, experiment)}
        else:
            populations = {"full_130km": frame["square_footprint_id"].eq(experiment).to_numpy(),
                           "central_50km": frame["square_test_id"].eq(experiment).to_numpy()}
        for name, mask in populations.items():
            rmse_median = np.sqrt(np.mean((c_median - reference)[mask] ** 2))
            rmse_applied = np.sqrt(np.mean((c_applied - reference)[mask] ** 2))
            affected = int((np.abs(c_median - c_applied)[mask] > 1e-9).sum())
            rows.append(dict(experiment=experiment, configuration=config, population=name, rows=int(mask.sum()),
                             affected_rows=affected, c_rmse_median_prediction=rmse_median,
                             c_rmse_applied_control=rmse_applied, abs_change=abs(rmse_applied - rmse_median)))

    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)
    for name, group in table.groupby("population"):
        worst = group.loc[group["abs_change"].idxmax()]
        print(f"{name}: cases {len(group)}, largest |change| {worst.abs_change:.4f} "
              f"({worst.experiment} {worst.configuration}), affected rows {group.affected_rows.sum()}")


if __name__ == "__main__":
    main()
