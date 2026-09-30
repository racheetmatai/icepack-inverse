"""Break down C error and velocity error by observed speed in the SQ05 and SQ06 footprints.

Supports the Results statement on SQ06: CFG02 has the smaller C error in flow
of 100-1000 m/a, which carries 91% of the squared velocity error for CFG04,
while CFG04 has the smaller C error in slower ice, which carries 4%. For the
complete SQ05 and SQ06 footprints and for CFG02 and CFG04, it reports by speed
class the area share, the share of the gap in squared C error, the share of
the CFG04 squared velocity error, and the share of the gap in squared velocity
error. It reads archived fields only; nothing is re-simulated.

Run in the Icepack/Firedrake environment:

    python production_workflow/paper_checks/check_speed_classes.py \
        --artifact-dir /path/to/unpacked_artifacts
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "production_workflow"))
sys.path.insert(0, str(PACKAGE / "production_workflow/controlled_replacement"))
from evaluate_forward_campaign import build_object, build_observation_alignment, interpolate_velocity  # noqa: E402
from evaluate_controlled_campaign import interpolate_control, velocity_path  # noqa: E402

BINS = [0, 100, 500, 1000, np.inf]
NAMES = ["<100", "100-500", "500-1000", ">=1000"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="Unpacked Zenodo artifacts (contains production_runs/ and production_workflow/)")
    args = parser.parse_args()
    artifacts = args.artifact_dir.resolve()
    runs = artifacts / "production_runs"
    controlled = artifacts / "production_workflow/controlled_replacement_20260917_a"

    object_, adoption, _ = build_object(PACKAGE / "production_workflow/amundsen_production_config.json",
                                        artifacts, runs / "gate1_definitive_inversion_adoption_20260819_a.json")
    frame, lookup = build_observation_alignment(
        object_, runs / "gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz")
    reference = interpolate_control(object_, np.asarray(adoption["C"].dat.data_ro, dtype=np.float64), lookup)
    observed = (pd.read_csv(controlled / "observation_audit/corrected_observations.csv.gz")
                .set_index("row_id").reindex(frame["row_id"].astype(str))[["observed_vx_raw", "observed_vy_raw"]]
                .to_numpy())
    speed = np.linalg.norm(observed, axis=1)

    for square in ["SQ05", "SQ06"]:
        mask = frame["square_footprint_id"].eq(square).to_numpy()
        errors = {}
        for config in ["CFG02", "CFG04"]:
            control = np.load(controlled / "controls" / f"CR_{square}_{config}_MEDIAN.npz")["control_C"]
            c = interpolate_control(object_, control.astype(np.float64), lookup)
            v = interpolate_velocity(object_, np.load(velocity_path(controlled, f"CR_{square}_{config}_MEDIAN")), lookup)
            errors[config] = dict(c2=((c - reference) ** 2)[mask], v2=np.sum((v - observed) ** 2, axis=1)[mask])
        s = speed[mask]
        c_gap = errors["CFG04"]["c2"] - errors["CFG02"]["c2"]   # > 0 where CFG02 has the smaller squared C error
        v_gap = errors["CFG04"]["v2"] - errors["CFG02"]["v2"]   # > 0 where CFG02 has the smaller squared velocity error
        print(f"\n{square} footprint: rows {mask.sum()}; C RMSE CFG02 {np.sqrt(errors['CFG02']['c2'].mean()):.3f}, "
              f"CFG04 {np.sqrt(errors['CFG04']['c2'].mean()):.3f}; velocity RMSE "
              f"CFG02 {np.sqrt(errors['CFG02']['v2'].mean()):.1f}, CFG04 {np.sqrt(errors['CFG04']['v2'].mean()):.1f}")
        print(f"{'speed':>9} {'area %':>7} {'C-error gap %':>14} {'CFG04 velocity sq. error %':>27} "
              f"{'velocity-error gap %':>21}")
        for lo, hi, name in zip(BINS[:-1], BINS[1:], NAMES):
            k = (s >= lo) & (s < hi)
            print(f"{name:>9} {100 * k.mean():7.1f} {100 * c_gap[k].sum() / c_gap.sum():14.1f} "
                  f"{100 * errors['CFG04']['v2'][k].sum() / errors['CFG04']['v2'].sum():27.1f} "
                  f"{100 * v_gap[k].sum() / v_gap.sum():21.1f}")


if __name__ == "__main__":
    main()
