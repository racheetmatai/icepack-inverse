"""Pure integrity checks for controlled-replacement evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


MANUSCRIPT_SCENARIO = "controlled_original_measures"


def relative_rmse(numerator: float, denominator: float, tolerance: float = 1.0e-12) -> float:
    """Return an RMSE ratio, or NaN for a negligible reference RMSE."""
    return float("nan") if abs(denominator) <= tolerance else float(numerator / denominator)


def align_original_observations(frame: pd.DataFrame, observations: pd.DataFrame) -> np.ndarray:
    """Align original raster components to evaluation locations by stable row ID."""
    if frame["row_id"].astype(str).duplicated().any() or observations["row_id"].astype(str).duplicated().any():
        raise ValueError("Row IDs must be unique")
    required = {"row_id", "observed_vx_raw", "observed_vy_raw"}
    if not required.issubset(observations.columns):
        raise ValueError("Original MEaSUREs component columns are missing")
    indexed = observations.assign(row_id=observations["row_id"].astype(str)).set_index("row_id")
    aligned = indexed.reindex(frame["row_id"].astype(str))
    values = aligned[["observed_vx_raw", "observed_vy_raw"]].to_numpy(float)
    if values.shape != (len(frame), 2) or not np.isfinite(values).all():
        raise ValueError("Original MEaSUREs components do not completely align")
    return values


def verify_control_pair(
    reference: np.ndarray,
    ml_control: np.ndarray,
    uniform_control: np.ndarray,
    ml_mask: np.ndarray,
    uniform_mask: np.ndarray,
) -> None:
    """Require identical masks and exact reference preservation outside them."""
    arrays = [reference, ml_control, uniform_control, ml_mask, uniform_mask]
    if len({np.asarray(item).shape for item in arrays}) != 1:
        raise ValueError("Control and mask shapes differ")
    if not np.array_equal(ml_mask, uniform_mask):
        raise ValueError("ML and uniform replacement masks differ")
    outside = ~np.asarray(ml_mask, dtype=bool)
    if not np.array_equal(ml_control[outside], reference[outside]):
        raise ValueError("ML control changed outside the replacement mask")
    if not np.array_equal(uniform_control[outside], reference[outside]):
        raise ValueError("Uniform control changed outside the replacement mask")
    if not (np.isfinite(ml_control).all() and np.isfinite(uniform_control).all()):
        raise ValueError("Control values are not finite")


def manuscript_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Select only the controlled/original-observation manuscript scenario."""
    if "scenario" not in table.columns:
        raise ValueError("Scenario column is missing")
    selected = table.loc[table["scenario"].eq(MANUSCRIPT_SCENARIO)].copy()
    if selected.empty:
        raise ValueError("Controlled manuscript scenario is absent")
    return selected
