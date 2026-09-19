#!/usr/bin/env python
"""Rebuild the joint-support distance thresholds from the frozen 5 km grid.

The joint-support criterion used in square selection and in the reported
per-configuration support is a distance cutoff in a transformed predictor
space.  ``describe_heldout_distributions.py`` *consumes* those cutoffs from
``frozen_design/five_region_partition_and_support.json``; this script is the
missing producer, so that the construction can be verified independently.

For one predictor set the construction is:

1.  Take the predictor values on the frozen eligible 5 km sector grid.
2.  Map each predictor's ranks to standard-normal quantiles
    (``QuantileTransformer``), then apply whitened PCA retaining at least
    99% of the variance.  This is the same transform as ``fit_reference`` in
    ``describe_heldout_distributions.py``.
3.  For every eligible grid point, find the smallest distance in that
    transformed space to another eligible grid point that is geographically
    separated by at least 40 km, where separation is measured as the larger
    of the two axis offsets (a 40 km square exclusion box, matching the
    square buffer geometry of the experimental design).
4.  The 95th percentile of those distances is the threshold.

The script recomputes the threshold for the twelve-predictor selection screen
and for each of the six predictor configurations, and checks every value
against the frozen record.  It writes nothing into the frozen design and
exits non-zero if any threshold fails to reproduce.

This script reads inputs only.  It does not use velocity, the inversion
control, or any model output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

SEED = 20260811
SEPARATION_M = 40000.0
REFERENCE_QUANTILE = 95.0
RELATIVE_TOLERANCE = 1e-9

# Same configuration names and predictor lists as describe_heldout_distributions.py,
# keyed by the identifiers used inside the frozen support record.
FEATURE_CONFIGURATIONS = {
    "1_all_ice": ["s", "h", "mag_s", "mag_h", "driving_stress", "surface_air_temp"],
    "2_best_ice": ["s", "h", "mag_s", "mag_h", "surface_air_temp"],
    "3_all_geophysical": ["b", "mag_b", "heatflux", "gravity_disturbance", "mag_anomaly"],
    "4_best_geophysical": ["b", "mag_b", "heatflux"],
    "5_best_combined": ["s", "h", "mag_s", "mag_h", "surface_air_temp", "b", "mag_b", "heatflux"],
    "6_best_combined_direction": [
        "s", "h", "mag_s", "mag_h", "surface_air_temp",
        "b", "mag_b", "heatflux", "cos_theta_bs",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_grid(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (features, coordinates, feature_names) on the eligible grid points."""
    grid = np.load(path, allow_pickle=False)
    eligible = grid["eligible"].astype(bool)
    flat = np.flatnonzero(eligible.ravel())
    features = grid["features"].reshape((-1, grid["features"].shape[-1]))[flat].astype(float)
    names = [str(name) for name in grid["feature_names"]]
    xx, yy = np.meshgrid(grid["x_grid"].astype(float), grid["y_grid"].astype(float))
    coordinates = np.column_stack((xx.ravel()[flat], yy.ravel()[flat]))
    if features.shape[1] != len(names):
        raise AssertionError("Grid feature count does not match feature_names")
    if not np.isfinite(features).all():
        raise AssertionError("Non-finite predictor value on the eligible reference grid")
    return features, coordinates, names


def separated_mask(coordinates: np.ndarray, block: slice) -> np.ndarray:
    """Pairs at least SEPARATION_M apart, measured as max(|dx|, |dy|)."""
    dx = np.abs(coordinates[block, 0, None] - coordinates[None, :, 0])
    dy = np.abs(coordinates[block, 1, None] - coordinates[None, :, 1])
    return np.maximum(dx, dy) >= SEPARATION_M


def transform_predictors(values: np.ndarray) -> np.ndarray:
    """Rank-Gaussian marginals followed by whitened PCA retaining >=99% variance."""
    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(values)), output_distribution="normal",
        random_state=SEED, subsample=len(values),
    )
    pca = PCA(n_components=0.99, whiten=True, svd_solver="full")
    return pca.fit_transform(transformer.fit_transform(values))


def separated_nearest_distances(
    transformed: np.ndarray, coordinates: np.ndarray, chunk: int = 512,
) -> np.ndarray:
    count = len(transformed)
    nearest = np.empty(count, dtype=float)
    for start in range(0, count, chunk):
        block = slice(start, min(count, start + chunk))
        distances = np.sqrt(
            ((transformed[block, None, :] - transformed[None, :, :]) ** 2).sum(axis=-1)
        )
        distances[~separated_mask(coordinates, block)] = np.inf
        nearest[block] = distances.min(axis=1)
    if not np.isfinite(nearest).all():
        raise AssertionError("A grid point has no separated analogue")
    return nearest


def build_threshold(
    features: np.ndarray, coordinates: np.ndarray, index: dict[str, int],
    predictors: list[str],
) -> tuple[int, float]:
    values = features[:, [index[name] for name in predictors]]
    transformed = transform_predictors(values)
    nearest = separated_nearest_distances(transformed, coordinates)
    return transformed.shape[1], float(np.percentile(nearest, REFERENCE_QUANTILE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-grid", required=True,
                        help="frozen_design/amundsen_input_support_grid_5km.npz")
    parser.add_argument("--support-evidence", required=True,
                        help="frozen_design/five_region_partition_and_support.json")
    parser.add_argument("--output", required=True,
                        help="directory for the verification record (must be empty or absent)")
    args = parser.parse_args()

    grid_path = Path(args.reference_grid).resolve()
    evidence_path = Path(args.support_evidence).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    features, coordinates, names = load_reference_grid(grid_path)
    index = {name: position for position, name in enumerate(names)}
    evidence = json.loads(evidence_path.read_text())

    targets = [("selection_screen", names,
                evidence["support_method"]["pca_components"],
                evidence["support_method"]["joint_cutoff"])]
    for key, predictors in FEATURE_CONFIGURATIONS.items():
        frozen = evidence["support_by_feature_configuration"][key]
        targets.append((key, predictors, frozen["pca_components"], frozen["joint_q95_cutoff"]))

    records, failures = [], []
    for key, predictors, frozen_components, frozen_cutoff in targets:
        components, cutoff = build_threshold(features, coordinates, index, predictors)
        relative_difference = abs(cutoff - frozen_cutoff) / frozen_cutoff
        reproduced = components == frozen_components and relative_difference <= RELATIVE_TOLERANCE
        if not reproduced:
            failures.append(key)
        records.append({
            "key": key,
            "predictors": predictors,
            "predictor_count": len(predictors),
            "pca_components": components,
            "frozen_pca_components": frozen_components,
            "joint_q95_cutoff": cutoff,
            "frozen_joint_q95_cutoff": frozen_cutoff,
            "relative_difference": relative_difference,
            "reproduced": reproduced,
        })
        print(f"{key:28s} {len(predictors):2d} predictors  {components:d} components  "
              f"cutoff {cutoff:.16f}  frozen {frozen_cutoff:.16f}  "
              f"rel.diff {relative_difference:.2e}  {'OK' if reproduced else 'MISMATCH'}")

    report = {
        "status": "complete" if not failures else "mismatch",
        "purpose": "Rebuild and verify the frozen joint-support distance thresholds",
        "method": {
            "grid": "frozen eligible 5 km sector grid",
            "marginal_transform": "rank-to-normal-quantile mapping per predictor",
            "joint_transform": "whitened PCA retaining >=99% variance",
            "separation_m": SEPARATION_M,
            "separation_metric": "max(|dx|, |dy|) between grid points",
            "reference_quantile": REFERENCE_QUANTILE / 100.0,
            "neighbors": 1,
            "seed": SEED,
        },
        "inputs": {
            "reference_grid": str(grid_path.name),
            "reference_grid_sha256": sha256(grid_path),
            "support_evidence": str(evidence_path.name),
            "support_evidence_sha256": sha256(evidence_path),
        },
        "eligible_grid_points": int(len(coordinates)),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "thresholds": records,
    }
    (output / "joint_support_threshold_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if failures:
        raise SystemExit(f"Thresholds did not reproduce: {', '.join(failures)}")
    print(f"\nAll {len(records)} thresholds reproduce the frozen record.")


if __name__ == "__main__":
    main()
