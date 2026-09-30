"""Median validation C RMSE and R_C^2 across the ten MLPs for the two regional tests.

Supports the appendix values for the inter-catchment corridors (0.317/0.764,
0.021/0.999 and 0.020/0.999 for CFG04-CFG06) and gives the same quantities for
PIG (CFG01-CFG03). Same computation as generate_table6_validation_c_diagnostics.py:
each member is scored on its own validation rows, then the median of the ten
values is taken.

    python production_workflow/paper_checks/regional_validation_c.py \
        --training-runs /path/to/unpacked_02_cuda_training_runs

The argument is the unpacked archive 02 (the folder that contains
sha256-*/runs/).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TESTS = (("REG_PIG", ("CFG01", "CFG02", "CFG03")), ("REG_INTER", ("CFG04", "CFG05", "CFG06")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training-runs", type=Path, required=True)
    args = parser.parse_args()
    runs = next(args.training_runs.resolve().glob("sha256*/runs"))
    for experiment, configs in TESTS:
        for config in configs:
            rmse, r2 = [], []
            for member in range(1, 11):
                frame = pd.read_csv(runs / f"{experiment}_{config}_M{member:02d}" / "validation_predictions.csv.gz",
                                    usecols=["reference_log_C", "predicted_log_C"])
                reference = frame["reference_log_C"].to_numpy(float)
                residual = frame["predicted_log_C"].to_numpy(float) - reference
                rmse.append(np.sqrt(np.mean(residual ** 2)))
                r2.append(1 - np.sum(residual ** 2) / np.sum((reference - reference.mean()) ** 2))
            print(f"{experiment} {config}: median validation C RMSE {np.median(rmse):.3f}, "
                  f"R_C^2 {np.median(r2):.3f}")


if __name__ == "__main__":
    main()
