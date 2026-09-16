"""Deterministic data-handling helpers for revised Icepack/ML workflows."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd


ROW_ID_VERSION = "xyh1"
MANIFEST_ID_VERSION = "sha256-json-v1"


def parse_strict_bool(value, *, name="value"):
    """Parse booleans without Python's truthy-string behavior.

    Exact strings ``true`` and ``false`` (case-insensitive, surrounding
    whitespace ignored) are accepted for legacy notebook compatibility.
    Integers and arbitrary strings are rejected.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(
        f"{name} must be a bool or the exact string 'true'/'false'; "
        f"received {value!r}."
    )


def stable_xy_row_ids(x, y):
    """Return order-independent versioned IDs for float64 projected points."""
    x_values = np.asarray(x, dtype="float64")
    y_values = np.asarray(y, dtype="float64")
    if x_values.shape != y_values.shape:
        raise ValueError("x and y coordinate arrays must have matching shapes.")
    if not (np.isfinite(x_values).all() and np.isfinite(y_values).all()):
        raise ValueError("Stable row IDs require finite projected coordinates.")
    pack = struct.Struct(">dd").pack
    return np.asarray(
        [
            f"{ROW_ID_VERSION}-"
            + hashlib.sha256(pack(float(x_value), float(y_value))).hexdigest()[:32]
            for x_value, y_value in zip(x_values, y_values)
        ],
        dtype=object,
    )


def ensure_row_ids(frame):
    """Return a copy with a verified or newly generated ``row_id`` column."""
    if "x" not in frame or "y" not in frame:
        raise ValueError("Stable row IDs require projected x and y columns.")
    result = frame.copy()
    expected = stable_xy_row_ids(result["x"], result["y"])
    if "row_id" in result:
        mismatch = result["row_id"].astype(str).to_numpy() != expected
        if np.any(mismatch):
            raise ValueError(
                f"Existing row_id values do not match {ROW_ID_VERSION}; "
                f"{int(np.sum(mismatch))} mismatches found."
            )
    else:
        result.insert(0, "row_id", expected)
    if result["row_id"].duplicated().any():
        raise ValueError("Duplicate projected x/y coordinates produce duplicate row IDs.")
    return result


def filter_training_rows(
    datasets,
    weights,
    required_columns,
    *,
    phi_threshold=0.1,
    use_boug_anomaly_filter=False,
):
    """Apply and account for the revised training-row filters.

    The Bouguer range filter is disabled by default and is retained only as an
    explicitly requested legacy comparison. Missing/non-finite required model
    values are reported and rejected rather than silently imputed or dropped.
    """
    use_boug_anomaly_filter = parse_strict_bool(
        use_boug_anomaly_filter, name="use_boug_anomaly_filter"
    )
    if len(datasets) != len(weights):
        raise ValueError("Each dataset must have one sample weight.")

    prepared = [ensure_row_ids(dataset) for dataset in datasets]
    frame = pd.concat(prepared, ignore_index=True)
    sample_weights = np.concatenate(
        [np.full(len(dataset), weight, dtype=float) for dataset, weight in zip(prepared, weights)]
    )
    audit = []

    def record(stage, before, after, detail):
        audit.append(
            {
                "stage": stage,
                "rows_before": int(before),
                "rows_after": int(after),
                "rows_removed": int(before - after),
                "detail": detail,
            }
        )

    before = len(frame)
    phi_mask = np.isfinite(frame["phi"].to_numpy()) & (
        frame["phi"].to_numpy() > phi_threshold
    )
    frame = frame.loc[phi_mask].reset_index(drop=True)
    sample_weights = sample_weights[phi_mask]
    record("phi", before, len(frame), f"finite phi > {phi_threshold}")

    if use_boug_anomaly_filter:
        before = len(frame)
        bouguer = frame["boug_anomaly"].to_numpy()
        bouguer_mask = np.isfinite(bouguer) & (bouguer > -200) & (bouguer < 70)
        frame = frame.loc[bouguer_mask].reset_index(drop=True)
        sample_weights = sample_weights[bouguer_mask]
        record(
            "legacy_bouguer",
            before,
            len(frame),
            "explicit legacy comparison: -200 < boug_anomaly < 70",
        )
    else:
        record("legacy_bouguer", len(frame), len(frame), "disabled (revised path)")

    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required training columns: {missing_columns}")
    required = frame[list(required_columns)].to_numpy(dtype=float)
    finite_mask = np.isfinite(required).all(axis=1)
    invalid_count = int(np.sum(~finite_mask))
    record(
        "required_finite",
        len(frame),
        int(np.sum(finite_mask)),
        f"finite values in {list(required_columns)}",
    )
    if invalid_count:
        raise ValueError(
            f"{invalid_count} rows have non-finite required values. Review the "
            "attrition audit; no scientific imputation or silent row drop was applied."
        )

    if len(frame) != len(sample_weights):
        raise ValueError("Filtered rows and sample weights are misaligned.")
    return frame, sample_weights, audit


def deterministic_manifest_id(manifest):
    """Hash canonical JSON content, excluding any existing manifest ID."""
    content = dict(manifest)
    content.pop("manifest_id", None)
    canonical = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"{MANIFEST_ID_VERSION}-{hashlib.sha256(canonical).hexdigest()}"


def file_sha256(path, chunk_size=1024 * 1024):
    """Return a SHA-256 digest without loading a large data file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
