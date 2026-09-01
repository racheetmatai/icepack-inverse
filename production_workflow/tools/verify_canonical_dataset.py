"""Independently verify a canonical revised Amundsen dataset export."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pandas as pd
import shapely


PREDICTORS = [
    "s", "h", "mag_s", "mag_h", "driving_stress", "surface_air_temp",
    "b", "mag_b", "heatflux", "gravity_disturbance", "mag_anomaly",
    "cos_theta_bs",
]
REGIONS = {
    1: "PIG",
    2: "Thwaites",
    3: "Dotson",
    4: "PIG-Thwaites inter-catchment",
    5: "Thwaites-Dotson inter-catchment",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_identifier(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_id", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def stable_ids(x: np.ndarray, y: np.ndarray) -> list[str]:
    pack = struct.Struct(">dd").pack
    return [
        "xyh1-" + hashlib.sha256(pack(float(xv), float(yv))).hexdigest()[:32]
        for xv, yv in zip(x, y)
    ]


def square_assignments(coordinates: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test = np.full(len(coordinates), "", dtype="U4")
    footprint = np.full(len(coordinates), "", dtype="U4")
    x, y = coordinates[:, 0], coordinates[:, 1]
    for row in rows:
        central = (
            (x >= float(row["test_xmin_m"])) & (x < float(row["test_xmax_m"]))
            & (y >= float(row["test_ymin_m"])) & (y < float(row["test_ymax_m"]))
        )
        full = (
            (x >= float(row["footprint_xmin_m"])) & (x < float(row["footprint_xmax_m"]))
            & (y >= float(row["footprint_ymin_m"])) & (y < float(row["footprint_ymax_m"]))
        )
        test[central] = row["square_id"]
        footprint[full] = row["square_id"]
    zone = np.full(len(coordinates), "outside", dtype="U7")
    zone[footprint != ""] = "buffer"
    zone[test != ""] = "test"
    return test, footprint, zone


class RegionClassifier:
    def __init__(self, mesh_paths: list[Path]):
        import firedrake

        self.meshes = []
        self.boundaries = []
        for mesh_path in mesh_paths:
            mesh = firedrake.Mesh(str(mesh_path))
            vertices = np.asarray(mesh.coordinates.dat.data_ro[:, :2], dtype="float64")
            cells = np.asarray(mesh.coordinates.cell_node_map().values, dtype=np.int64)
            self.meshes.append(mesh)
            edges = np.concatenate(
                [cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]], axis=0
            )
            edges.sort(axis=1)
            unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
            self.boundaries.append(
                shapely.MultiLineString(vertices[unique_edges[counts == 1]].tolist())
            )

    def assign(self, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        membership = np.column_stack([
            mesh.locate_cells_ref_coords_and_dists(coordinates)[0] >= 0
            for mesh in self.meshes
        ])
        raw_count = membership.sum(axis=1).astype(np.int8)
        overlap = np.flatnonzero(raw_count > 1)
        if len(overlap):
            points = shapely.points(coordinates[overlap, 0], coordinates[overlap, 1])
            clearances = np.full((len(overlap), 3), -np.inf)
            for column, boundary in enumerate(self.boundaries):
                candidate = membership[overlap, column]
                if np.any(candidate):
                    clearances[candidate, column] = shapely.distance(
                        boundary, points[candidate]
                    )
            selected = np.argmax(clearances, axis=1)
            membership[overlap, :] = False
            membership[overlap, selected] = True
        codes = np.zeros(len(coordinates), dtype=np.int8)
        for column, code in enumerate((1, 2, 3)):
            codes[membership[:, column]] = code
        outside = membership.sum(axis=1) == 0
        codes[outside & (coordinates[:, 1] >= -400_000.0)] = 4
        codes[outside & (coordinates[:, 1] < -400_000.0)] = 5
        return codes, raw_count


def verify_export(export_dir: Path) -> dict:
    export_dir = export_dir.resolve()
    manifest_path = export_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "jog-canonical-master-dataset-v1":
        raise RuntimeError("Unexpected dataset-manifest schema.")
    if manifest.get("manifest_id") != canonical_identifier(manifest):
        raise RuntimeError("Dataset manifest identifier is invalid.")
    if manifest.get("status") != "complete":
        raise RuntimeError("Dataset export is not complete.")
    expected_hashes = manifest.get("output_sha256") or {}
    actual_files = {
        str(path.relative_to(export_dir))
        for path in export_dir.rglob("*") if path.is_file() and path != manifest_path
    }
    if actual_files != set(expected_hashes):
        raise RuntimeError("Dataset file inventory differs from the manifest.")
    for relative, expected in expected_hashes.items():
        if sha256_file(export_dir / relative) != expected:
            raise RuntimeError(f"Dataset output hash mismatch: {relative}")

    schema = json.loads((export_dir / "schema.json").read_text(encoding="utf-8"))
    if not (
        schema.get("predictors_in_order") == PREDICTORS
        and schema.get("training_target") == "reference_log_C"
        and not any("velocity" in name for name in PREDICTORS)
    ):
        raise RuntimeError("Dataset predictor or target schema changed.")
    expected_columns = [row["name"] for row in schema["columns"]]

    partition = np.load(
        export_dir / "provenance" / "five_region_partition_5km.npz",
        allow_pickle=False,
    )
    xx, yy = np.meshgrid(partition["x_grid"], partition["y_grid"])
    grid_mask = partition["eligible"].astype(bool).ravel()
    grid_coordinates = np.column_stack((xx.ravel()[grid_mask], yy.ravel()[grid_mask]))
    grid_codes = partition["region_codes"].ravel()[grid_mask].astype(np.int8)
    classifier = RegionClassifier([
        export_dir / "provenance" / "pig_region.msh",
        export_dir / "provenance" / "thwaites_region.msh",
        export_dir / "provenance" / "dotson_region.msh",
    ])
    reproduced_grid_codes, _ = classifier.assign(grid_coordinates)
    if not np.array_equal(reproduced_grid_codes, grid_codes):
        raise RuntimeError("Direct mesh classifier does not reproduce frozen grid labels.")
    with (export_dir / "provenance" / "selected_squares.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        square_rows = list(csv.DictReader(stream))

    row_count = 0
    eligible_count = 0
    error_valid_count = 0
    errbed_finite_count = 0
    row_ids: set[str] = set()
    eligible_ids: set[str] = set()
    region_counts = {name: 0 for name in REGIONS.values()}
    dataset_path = export_dir / "canonical_master_dataset.csv.gz"
    for chunk in pd.read_csv(dataset_path, chunksize=100_000, keep_default_na=False):
        if list(chunk.columns) != expected_columns:
            raise RuntimeError("Canonical dataset columns or order changed.")
        count = len(chunk)
        row_count += count
        coordinates = chunk[["x", "y"]].to_numpy(dtype="float64")
        calculated_ids = stable_ids(coordinates[:, 0], coordinates[:, 1])
        ids = chunk["row_id"].astype(str).tolist()
        if ids != calculated_ids or any(value in row_ids for value in ids):
            raise RuntimeError("Stable row ID mismatch or duplicate.")
        row_ids.update(ids)

        predictor_values = chunk[PREDICTORS].to_numpy(dtype="float64")
        predictors_finite = np.isfinite(predictor_values).all(axis=1)
        h = chunk["h"].to_numpy(dtype="float64")
        s = chunk["s"].to_numpy(dtype="float64")
        ratio = np.divide(
            1024.0 * np.maximum(0.0, h - s), 917.0 * h,
            out=np.ones_like(h), where=h > 0.0,
        )
        grounded = (np.maximum(1.0 - ratio, 0.0) > 0.1) & (h > 0.0)
        common = predictors_finite & grounded
        recorded_common = chunk["common_eligible"].astype(bool).to_numpy()
        if not (
            np.array_equal(recorded_common, common)
            and np.array_equal(chunk["predictors_finite"].astype(bool), predictors_finite)
            and np.array_equal(chunk["grounded_phi_gt_0_1"].astype(bool), grounded)
            and np.isfinite(chunk[["observed_vx", "observed_vy"]].to_numpy(dtype="float64")).all()
            and np.isfinite(chunk.loc[common, ["reference_log_C", *PREDICTORS]].to_numpy(dtype="float64")).all()
        ):
            raise RuntimeError("Eligibility, observed velocity, target, or predictor check failed.")
        eligible_count += int(common.sum())
        eligible_ids.update(np.asarray(ids, dtype=object)[common].tolist())
        error_valid_count += int(chunk["velocity_error_valid"].astype(bool).sum())
        errbed_finite_count += int(np.isfinite(pd.to_numeric(chunk["bedmachine_errbed"], errors="coerce")).sum())

        codes, raw_membership_count = classifier.assign(coordinates)
        if not (
            np.array_equal(chunk["region_code"].to_numpy(dtype=np.int8), codes)
            and np.array_equal(
                chunk["region_overlap_resolved"].astype(bool).to_numpy(),
                raw_membership_count > 1,
            )
            and chunk["region_name"].astype(str).tolist() == [REGIONS[int(code)] for code in codes]
        ):
            raise RuntimeError("Frozen five-region row assignment changed.")
        for code, name in REGIONS.items():
            region_counts[name] += int(np.sum(common & (codes == code)))

        test, footprint, zone = square_assignments(coordinates, square_rows)
        if not (
            chunk["square_test_id"].astype(str).tolist() == test.tolist()
            and chunk["square_footprint_id"].astype(str).tolist() == footprint.tolist()
            and chunk["square_zone"].astype(str).tolist() == zone.tolist()
        ):
            raise RuntimeError("Frozen nested-square row assignment changed.")

    if row_count != int(manifest["row_count"]) or len(row_ids) != row_count:
        raise RuntimeError("Dataset row count or row-ID uniqueness failed.")
    if eligible_count != int(manifest["common_eligible_count"]):
        raise RuntimeError("Common eligible row count differs from the manifest.")

    with gzip.open(export_dir / "common_eligible_row_ids.txt.gz", "rt", encoding="utf-8") as stream:
        saved_eligible_ids = [line.rstrip("\n") for line in stream]
    if (
        saved_eligible_ids != sorted(saved_eligible_ids)
        or len(saved_eligible_ids) != eligible_count
        or set(saved_eligible_ids) != eligible_ids
    ):
        raise RuntimeError("Saved common-eligible row IDs do not match the dataset.")

    diagnostics = json.loads((export_dir / "diagnostics.json").read_text(encoding="utf-8"))
    if not (
        diagnostics["selected_observations"] == row_count
        and diagnostics["common_eligible"] == eligible_count
        and diagnostics["velocity_error_pair_valid"] == error_valid_count
        and diagnostics["bedmachine_errbed_finite"] == errbed_finite_count
        and diagnostics["eligible_region_counts"] == region_counts
    ):
        raise RuntimeError("Dataset diagnostic counts are not reproducible.")

    return {
        "status": "verified",
        "manifest_id": manifest["manifest_id"],
        "row_count": row_count,
        "common_eligible_count": eligible_count,
        "column_count": len(expected_columns),
        "output_file_count": len(expected_hashes),
        "training_target": "reference_log_C",
        "predictor_count": len(PREDICTORS),
        "velocity_predictor_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_export(args.export_directory), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
