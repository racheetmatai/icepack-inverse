"""Export the canonical revised Amundsen dataset from an adopted inversion."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sys

# When executed by file path, Python places production_workflow/ before the
# repository root. Put the live repository first so ``src`` cannot resolve to
# the intentionally incomplete four-file provenance snapshot in this folder.
REPO_ROOT_HINT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT_HINT) in sys.path:
    sys.path.remove(str(REPO_ROOT_HINT))
sys.path.insert(0, str(REPO_ROOT_HINT))

import numpy as np
import pandas as pd
import shapely

from load_definitive_inversion import load_adopted_definitive_state
from production_amundsen import (
    Preflight, load_config, resolve_design_path, resolve_repo_path,
)


SCHEMA = "jog-canonical-master-dataset-v1"
EXPECTED_COMMON_ELIGIBLE = 1_530_992
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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mesh_membership_and_boundary(mesh_path: Path, coordinates: np.ndarray):
    import firedrake

    mesh = firedrake.Mesh(str(mesh_path))
    vertices = np.asarray(mesh.coordinates.dat.data_ro[:, :2], dtype="float64")
    cells = np.asarray(mesh.coordinates.cell_node_map().values, dtype=np.int64)
    membership = mesh.locate_cells_ref_coords_and_dists(coordinates)[0] >= 0
    edges = np.concatenate(
        [cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_segments = vertices[unique_edges[counts == 1]]
    boundary = shapely.MultiLineString(boundary_segments.tolist())
    return membership, boundary


def assign_regions(
    coordinates: np.ndarray, mesh_paths: dict[str, Path]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen regional-mesh containment and overlap rule directly."""
    order = ("PIG", "Thwaites", "Dotson")
    membership = np.zeros((len(coordinates), 3), dtype=bool)
    boundaries = []
    for column, name in enumerate(order):
        membership[:, column], boundary = _mesh_membership_and_boundary(
            mesh_paths[name], coordinates
        )
        boundaries.append(boundary)
    raw_count = membership.sum(axis=1).astype(np.int8)
    overlap = np.flatnonzero(raw_count > 1)
    if len(overlap):
        points = shapely.points(coordinates[overlap, 0], coordinates[overlap, 1])
        clearances = np.full((len(overlap), 3), -np.inf, dtype="float64")
        for column, boundary in enumerate(boundaries):
            candidate = membership[overlap, column]
            if np.any(candidate):
                clearances[candidate, column] = shapely.distance(
                    boundary, points[candidate]
                )
        selected = np.argmax(clearances, axis=1)
        membership[overlap, :] = False
        membership[overlap, selected] = True
    codes = np.zeros(len(coordinates), dtype=np.int8)
    codes[membership[:, 0]] = 1
    codes[membership[:, 1]] = 2
    codes[membership[:, 2]] = 3
    outside = membership.sum(axis=1) == 0
    codes[outside & (coordinates[:, 1] >= -400_000.0)] = 4
    codes[outside & (coordinates[:, 1] < -400_000.0)] = 5
    if np.any(codes == 0):
        raise RuntimeError("Direct five-region assignment left rows unlabeled.")
    return codes, raw_count


def assign_squares(
    coordinates: np.ndarray, squares_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test = np.full(len(coordinates), "", dtype="U4")
    footprint = np.full(len(coordinates), "", dtype="U4")
    with squares_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10:
        raise RuntimeError("Frozen square table must contain exactly ten squares.")
    x, y = coordinates[:, 0], coordinates[:, 1]
    for row in rows:
        square_id = row["square_id"]
        central = (
            (x >= float(row["test_xmin_m"]))
            & (x < float(row["test_xmax_m"]))
            & (y >= float(row["test_ymin_m"]))
            & (y < float(row["test_ymax_m"]))
        )
        full = (
            (x >= float(row["footprint_xmin_m"]))
            & (x < float(row["footprint_xmax_m"]))
            & (y >= float(row["footprint_ymin_m"]))
            & (y < float(row["footprint_ymax_m"]))
        )
        if np.any((test != "") & central) or np.any((footprint != "") & full):
            raise RuntimeError("Frozen square test or footprint masks overlap.")
        test[central] = square_id
        footprint[full] = square_id
    zone = np.full(len(coordinates), "outside", dtype="U7")
    zone[footprint != ""] = "buffer"
    zone[test != ""] = "test"
    return test, footprint, zone


def field_summary(frame: pd.DataFrame, eligible: np.ndarray) -> pd.DataFrame:
    rows = []
    for population, mask in (("all_selected", np.ones(len(frame), dtype=bool)),
                             ("common_eligible", eligible)):
        for name in [
            "reference_log_C", "reference_friction_C", *PREDICTORS,
            "observed_vx", "observed_vy", "observed_speed",
            "inversion_vx", "inversion_vy", "inversion_speed",
            "err_x", "err_y", "bedmachine_errbed",
        ]:
            values = frame.loc[mask, name].to_numpy(dtype="float64")
            finite = values[np.isfinite(values)]
            quantiles = np.quantile(finite, [0.01, 0.5, 0.99]) if len(finite) else [np.nan] * 3
            rows.append({
                "population": population,
                "field": name,
                "count": int(len(values)),
                "finite_count": int(len(finite)),
                "minimum": float(np.min(finite)) if len(finite) else None,
                "q01": float(quantiles[0]) if len(finite) else None,
                "median": float(quantiles[1]) if len(finite) else None,
                "q99": float(quantiles[2]) if len(finite) else None,
                "maximum": float(np.max(finite)) if len(finite) else None,
            })
    return pd.DataFrame(rows)


def build_frame(object_, velocity, config: dict) -> tuple[pd.DataFrame, dict]:
    raw = object_.get_dataframe(velocity)
    expected = int(config["expected_counts"]["selected_observations"])
    if len(raw) != expected:
        raise RuntimeError(f"Unexpected selected-observation count: {len(raw)} != {expected}")
    renamed = raw.rename(columns={
        "C": "reference_log_C",
        "C_total": "reference_friction_C",
        "x_velocity": "inversion_vx",
        "y_velocity": "inversion_vy",
        "x_velocity_initial": "observed_vx",
        "y_velocity_initial": "observed_vy",
    })
    renamed["observed_speed"] = np.hypot(renamed["observed_vx"], renamed["observed_vy"])
    renamed["inversion_speed"] = np.hypot(renamed["inversion_vx"], renamed["inversion_vy"])
    predictors_finite = np.isfinite(renamed[PREDICTORS].to_numpy(dtype="float64")).all(axis=1)
    h = renamed["h"].to_numpy(dtype="float64")
    s = renamed["s"].to_numpy(dtype="float64")
    water_ratio = np.divide(
        1024.0 * np.maximum(0.0, h - s),
        917.0 * h,
        out=np.ones_like(h),
        where=h > 0.0,
    )
    phi = np.maximum(1.0 - water_ratio, 0.0)
    grounded = (phi > 0.1) & (h > 0.0)
    common_eligible = predictors_finite & grounded
    if int(common_eligible.sum()) != EXPECTED_COMMON_ELIGIBLE:
        raise RuntimeError(f"Unexpected common eligible count: {int(common_eligible.sum())}")
    coordinates = renamed[["x", "y"]].to_numpy(dtype="float64")
    partition_audit_path = resolve_design_path(
        config["frozen_design"]["five_region_support"]["path"]
    )
    partition_audit = json.loads(partition_audit_path.read_text(encoding="utf-8"))
    mesh_paths = {
        name: Path(partition_audit["meshes"][name]["path"])
        for name in ("PIG", "Thwaites", "Dotson")
    }
    squares_path = resolve_design_path(
        config["frozen_design"]["selected_squares"]["path"]
    )
    region_codes, raw_region_membership_count = assign_regions(coordinates, mesh_paths)
    # Reproduce all frozen 5 km labels before applying the same rule to rows.
    frozen_partition_path = resolve_design_path(
        config["frozen_design"]["five_region_partition"]["path"]
    )
    frozen_partition = np.load(frozen_partition_path, allow_pickle=False)
    xx, yy = np.meshgrid(frozen_partition["x_grid"], frozen_partition["y_grid"])
    frozen_mask = frozen_partition["eligible"].astype(bool)
    frozen_coordinates = np.column_stack((xx[frozen_mask], yy[frozen_mask]))
    reproduced_codes, _ = assign_regions(frozen_coordinates, mesh_paths)
    if not np.array_equal(reproduced_codes, frozen_partition["region_codes"][frozen_mask]):
        raise RuntimeError("Direct mesh assignment does not reproduce the frozen 5 km partition.")
    test_ids, footprint_ids, square_zones = assign_squares(coordinates, squares_path)

    frame = pd.DataFrame({
        "row_id": renamed["row_id"].astype(str),
        "x": renamed["x"],
        "y": renamed["y"],
        "reference_log_C": renamed["reference_log_C"],
        "reference_friction_C": renamed["reference_friction_C"],
        **{name: renamed[name] for name in PREDICTORS},
        "observed_vx": renamed["observed_vx"],
        "observed_vy": renamed["observed_vy"],
        "observed_speed": renamed["observed_speed"],
        "inversion_vx": renamed["inversion_vx"],
        "inversion_vy": renamed["inversion_vy"],
        "inversion_speed": renamed["inversion_speed"],
        "velocity_source": renamed["velocity_source"],
        "err_x": renamed["err_x"],
        "err_y": renamed["err_y"],
        "velocity_error_valid": renamed["velocity_error_valid"].astype(bool),
        "bed_class": renamed["bed_class"],
        "bedmachine_source": renamed["bedmachine_source"],
        "bedmachine_errbed": renamed["bedmachine_errbed"],
        "grounded_phi_gt_0_1": grounded,
        "predictors_finite": predictors_finite,
        "common_eligible": common_eligible,
        "region_code": region_codes,
        "region_name": [REGIONS[int(code)] for code in region_codes],
        "region_overlap_resolved": raw_region_membership_count > 1,
        "square_test_id": test_ids,
        "square_footprint_id": footprint_ids,
        "square_zone": square_zones,
    })
    if frame["row_id"].duplicated().any() or not frame["row_id"].str.startswith("xyh1-").all():
        raise RuntimeError("Canonical row IDs are missing or duplicated.")
    if not np.isfinite(frame.loc[common_eligible, ["reference_log_C", *PREDICTORS]].to_numpy()).all():
        raise RuntimeError("Training target or predictor is nonfinite on common eligible rows.")
    if not np.isfinite(frame[["observed_vx", "observed_vy"]].to_numpy()).all():
        raise RuntimeError("Selected observations contain invalid paired velocity.")
    diagnostics = {
        "selected_observations": int(len(frame)),
        "predictors_finite": int(predictors_finite.sum()),
        "grounded_phi_gt_0_1": int(grounded.sum()),
        "common_eligible": int(common_eligible.sum()),
        "velocity_error_pair_valid": int(frame["velocity_error_valid"].sum()),
        "bedmachine_errbed_finite": int(np.isfinite(frame["bedmachine_errbed"]).sum()),
        "direct_region_overlap_rows_all": int(np.sum(raw_region_membership_count > 1)),
        "direct_region_overlap_rows_eligible": int(np.sum(common_eligible & (raw_region_membership_count > 1))),
        "eligible_region_counts": {
            REGIONS[code]: int(np.sum(common_eligible & (region_codes == code)))
            for code in REGIONS
        },
        "eligible_square_test_counts": {
            f"SQ{index:02d}": int(np.sum(common_eligible & (test_ids == f"SQ{index:02d}")))
            for index in range(1, 11)
        },
        "eligible_square_footprint_counts": {
            f"SQ{index:02d}": int(np.sum(common_eligible & (footprint_ids == f"SQ{index:02d}")))
            for index in range(1, 11)
        },
    }
    return frame, diagnostics


def export_dataset(args) -> Path:
    repo_root = args.repo_root.resolve()
    config_path = args.config.resolve()
    adoption_record = args.adoption_record.resolve()
    output_root = args.output_root.resolve()
    final_dir = output_root / args.run_id
    staging = output_root / f".{args.run_id}.incomplete"
    if final_dir.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite export {final_dir} or {staging}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    started = datetime.now(timezone.utc)
    try:
        config = load_config(config_path)
        config["_repo_root"] = str(repo_root)
        preflight = Preflight(config, repo_root, staging / "construction_audit")
        object_ = preflight.build_invert(reg_c=float(args.reg_c))
        adopted = load_adopted_definitive_state(
            adoption_record=adoption_record, object_=object_
        )
        if float(adopted["reg_c"]) != float(args.reg_c):
            raise RuntimeError("Adopted state uses a different regularization value.")
        frame, diagnostics = build_frame(object_, adopted["velocity"], config)

        dataset_path = staging / "canonical_master_dataset.csv.gz"
        frame.to_csv(
            dataset_path,
            index=False,
            float_format="%.17g",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        eligible_ids = frame.loc[frame["common_eligible"], "row_id"].sort_values()
        with (staging / "common_eligible_row_ids.txt.gz").open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_stream,
                compresslevel=6, mtime=0,
            ) as stream:
                stream.write(("\n".join(eligible_ids) + "\n").encode("utf-8"))

        attrition = pd.DataFrame([
            {"stage": "velocity_window_pixels", "retained": int(config["expected_counts"]["velocity_window_width"] * config["expected_counts"]["velocity_window_height"]), "excluded_from_previous": 0, "role": "source population"},
            {"stage": "finite_paired_velocity_and_source_gt_0", "retained": int(config["expected_counts"]["valid_velocity_source_pixels"]), "excluded_from_previous": int(config["expected_counts"]["velocity_window_width"] * config["expected_counts"]["velocity_window_height"] - config["expected_counts"]["valid_velocity_source_pixels"]), "role": "observation validity"},
            {"stage": "inside_computational_mesh", "retained": diagnostics["selected_observations"], "excluded_from_previous": int(config["expected_counts"]["valid_velocity_source_pixels"] - diagnostics["selected_observations"]), "role": "master dataset rows"},
            {"stage": "all_twelve_predictors_finite", "retained": diagnostics["predictors_finite"], "excluded_from_previous": int(diagnostics["selected_observations"] - diagnostics["predictors_finite"]), "role": "common predictor population"},
            {"stage": "grounded_phi_gt_0_1", "retained": diagnostics["common_eligible"], "excluded_from_previous": int(diagnostics["predictors_finite"] - diagnostics["common_eligible"]), "role": "training/evaluation eligible population"},
        ])
        attrition.to_csv(staging / "attrition.csv", index=False)
        field_summary(frame, frame["common_eligible"].to_numpy()).to_csv(
            staging / "field_summary.csv", index=False
        )
        schema = {
            "schema": "jog-canonical-master-dataset-columns-v1",
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "columns": [{"name": name, "dtype": str(frame[name].dtype)} for name in frame.columns],
            "predictors_in_order": PREDICTORS,
            "training_target": "reference_log_C",
            "physical_friction_diagnostic": "reference_friction_C",
            "velocity_policy": "Observed and inversion velocities are evaluation-only and excluded from every predictor configuration.",
            "auxiliary_policy": "Bed class, BedMachine source/errbed, MEaSUREs source/errors, masks, coordinates, and region/square labels are never predictors.",
            "eligibility": "common_eligible = grounded_phi_gt_0_1 AND finite values for the union of all twelve predictors; no feature-specific row removal.",
            "region_assignment": "Direct containment in the frozen PIG, Thwaites, and Dotson triangular meshes. Overlaps use greatest distance to the candidate mesh exterior boundary; points outside all three use the frozen y=-400000 m corridor separator. The method exactly reproduces the frozen eligible 5 km labels.",
            "square_policy": "Central 50 km test square and full 130 km footprint are assigned from frozen half-open coordinate bounds; buffer is footprint minus test.",
        }
        write_json(staging / "schema.json", schema)
        write_json(staging / "diagnostics.json", diagnostics)

        provenance = staging / "provenance"
        provenance.mkdir()
        partition_audit_path = resolve_design_path(
            config["frozen_design"]["five_region_support"]["path"]
        )
        partition_audit = json.loads(
            partition_audit_path.read_text(encoding="utf-8")
        )
        mesh_paths = {
            name: Path(partition_audit["meshes"][name]["path"])
            for name in ("PIG", "Thwaites", "Dotson")
        }
        copied = {
            "adoption_record.json": adoption_record,
            "amundsen_production_config.json": config_path,
            "selected_squares.csv": resolve_design_path(config["frozen_design"]["selected_squares"]["path"]),
            "five_region_partition_5km.npz": resolve_design_path(config["frozen_design"]["five_region_partition"]["path"]),
            "five_region_partition_and_support.json": partition_audit_path,
        }
        for name, mesh_path in mesh_paths.items():
            copied[f"{name.lower()}_region.msh"] = mesh_path
        for destination, source in copied.items():
            shutil.copy2(source, provenance / destination)
        source_paths = [
            Path(__file__).resolve(),
            Path(__file__).resolve().parent / "load_definitive_inversion.py",
            Path(__file__).resolve().parent / "tools" / "verify_definitive_inversion_adoption.py",
            Path(__file__).resolve().parent / "tools" / "verify_lcurve_selection_bundle.py",
            Path(__file__).resolve().parent / "production_amundsen.py",
            repo_root / "src" / "invert_c_theta.py",
            repo_root / "src" / "data_preprocessing.py",
            repo_root / "src" / "feature_units.py",
            repo_root / "src" / "revised_raster_inputs.py",
        ]
        source_snapshot = provenance / "source_snapshot"
        source_snapshot.mkdir()
        for source in source_paths:
            shutil.copy2(source, source_snapshot / source.name)

        output_hashes = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        adoption = json.loads(adoption_record.read_text(encoding="utf-8"))
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "run_id": args.run_id,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "reg_c": float(args.reg_c),
            "adoption_manifest_id": adoption["manifest_id"],
            "selected_point_manifest_id": adoption["point"]["manifest_id"],
            "config_sha256": sha256_file(config_path),
            "row_count": int(len(frame)),
            "common_eligible_count": int(frame["common_eligible"].sum()),
            "predictors": PREDICTORS,
            "training_target": "reference_log_C",
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "executable": sys.executable,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "diagnostics": diagnostics,
            "output_sha256": output_hashes,
        }
        manifest["manifest_id"] = canonical_identifier(manifest)
        write_json(staging / "dataset_manifest.json", manifest)
        os.replace(staging, final_dir)
        return final_dir
    except BaseException:
        # Preserve the incomplete directory for diagnosis; never publish it as complete.
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adoption-record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reg-c", type=float, default=0.01414213562)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    output = export_dataset(parse_args(argv))
    print(json.dumps({"status": "complete", "output_directory": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
