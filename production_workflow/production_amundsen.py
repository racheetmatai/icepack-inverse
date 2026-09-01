#!/usr/bin/env python3
"""Configuration-driven production entry point for the JOG Amundsen revision.

The first implemented mode is a non-optimizing preflight. It verifies the
frozen data, mesh, geometry, predictor definitions, and experiment artifacts
and writes a self-identifying manifest. It never runs an inversion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import numpy as np


WORKFLOW_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = WORKFLOW_ROOT / "amundsen_production_config.json"
DEFAULT_REPO_ROOT = WORKFLOW_ROOT.parent

EXPECTED_TOP_LEVEL_KEYS = {
    "schema",
    "domain",
    "physics",
    "inversion",
    "direction_feature",
    "predictors",
    "excluded_predictors",
    "auxiliary_fields",
    "inputs",
    "frozen_design",
    "expected_counts",
}

FROZEN_PREDICTORS = [
    "s",
    "h",
    "mag_s",
    "mag_h",
    "driving_stress",
    "surface_air_temp",
    "b",
    "mag_b",
    "heatflux",
    "gravity_disturbance",
    "mag_anomaly",
    "cos_theta_bs",
]
FROZEN_DIRICHLET_IDS = [1, 3, 5, 6, 7, 8, 9, 10, 11]
FROZEN_STRESS_IDS = [2, 4]
FROZEN_LCURVE_BASE = [0.01, 0.02, 0.05, 0.1, 0.2]
FROZEN_LCURVE_PROTOCOL = {
    "lcurve_block_iterations": 50,
    "lcurve_min_blocks": 4,
    "lcurve_max_blocks": 6,
    "lcurve_stable_transitions": 2,
    "lcurve_relative_misfit_tolerance": 5e-3,
    "lcurve_relative_roughness_tolerance": 5e-3,
    "lcurve_gradient_safety_tolerance": 1e-3,
    "lcurve_low_extension_reg_c": 0.005,
    "lcurve_high_extension_reg_c": 0.5,
    "lcurve_curvature_ambiguity_ratio": 1.25,
    "lcurve_max_refinement_rounds": 1,
    "lcurve_max_confirmation_rounds": 2,
    "lcurve_max_formal_points": 8,
}


def canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def manifest_identifier(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_id", None)
    return "sha256-json-v1-" + hashlib.sha256(canonical_json(unsigned)).hexdigest()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(config) - EXPECTED_TOP_LEVEL_KEYS
    missing = EXPECTED_TOP_LEVEL_KEYS - set(config)
    if unknown or missing:
        raise ValueError(
            f"Invalid top-level config keys; unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}"
        )
    validate_frozen_config(config)
    return config


def validate_frozen_config(config: dict) -> None:
    if config["schema"] != "jog-amundsen-production-config-v2":
        raise ValueError("Unsupported production-config schema.")
    domain = config["domain"]
    if domain["read_mesh"] is not True:
        raise ValueError("Production must read the frozen mesh; regeneration is forbidden.")
    if domain["dirichlet_ids"] != FROZEN_DIRICHLET_IDS:
        raise ValueError("Dirichlet IDs do not match the frozen Amundsen boundary policy.")
    if domain["stress_ids"] != FROZEN_STRESS_IDS:
        raise ValueError("Stress IDs do not match the two frozen calving fronts.")
    if sorted(domain["all_boundary_ids"]) != list(range(1, 12)):
        raise ValueError("The frozen Amundsen mesh must have boundary tags 1 through 11.")
    if sorted(domain["dirichlet_ids"] + domain["stress_ids"]) != list(range(1, 12)):
        raise ValueError("Dirichlet and stress tags must partition all boundary tags.")
    if config["predictors"] != FROZEN_PREDICTORS:
        raise ValueError("Predictor order or identity differs from the frozen design.")
    if "boug_anomaly" in config["predictors"]:
        raise ValueError("Bouguer anomaly cannot be a revised predictor.")
    if config["inversion"]["loss"] != "nosigma":
        raise ValueError("The revised inversion must use the unweighted nosigma loss.")
    if config["inversion"]["regularization"] != "gradient":
        raise ValueError("The revised inversion must use gradient regularization.")
    inversion = config["inversion"]
    frozen_solver = {
        "regularization_length_m": 7500.0,
        "gradient_tolerance": 1e-4,
        "step_tolerance": 5e-3,
        **FROZEN_LCURVE_PROTOCOL,
    }
    observed_solver = {key: inversion.get(key) for key in frozen_solver}
    if observed_solver != frozen_solver:
        raise ValueError("The L-curve solver controls differ from the frozen protocol.")
    if inversion.get("lcurve_base_reg_c") != FROZEN_LCURVE_BASE:
        raise ValueError("The formal L-curve base grid differs from the frozen protocol.")
    legacy_keys = {
        "max_iterations",
        "continuation_iterations",
        "lcurve_initial_reg_c",
        "lcurve_low_extension",
        "lcurve_high_extension",
    }
    present_legacy = sorted(legacy_keys.intersection(inversion))
    if present_legacy:
        raise ValueError(
            "legacy L-curve keys are forbidden under schema v2: "
            + ", ".join(present_legacy)
        )
    if config["physics"]["initial_log_friction_c"] != 0.0:
        raise ValueError("Every L-curve inversion must start from C=0.")
    if config["physics"]["initial_log_fluidity_theta"] != 0.0:
        raise ValueError("The revised C-only inversion must keep theta=0.")
    direction = config["direction_feature"]
    if direction["surface_slope_floor"] != 1.5e-4:
        raise ValueError("Unexpected surface-slope floor.")
    if direction["bed_slope_floor"] != 5.4e-3:
        raise ValueError("Unexpected bed-slope floor.")
    if direction["clip"] != [-1.0, 1.0]:
        raise ValueError("The alignment feature must be clipped to [-1, 1].")


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def resolve_design_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKFLOW_ROOT / path


def parse_msh_boundary_tags(path: Path) -> list[int]:
    tags = set()
    in_elements = False
    remaining = None
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped == "$Elements":
                in_elements = True
                remaining = None
                continue
            if not in_elements:
                continue
            if remaining is None:
                remaining = int(stripped)
                continue
            if stripped == "$EndElements" or remaining == 0:
                break
            values = [int(value) for value in stripped.split()]
            element_type = values[1]
            number_of_tags = values[2]
            if element_type == 1 and number_of_tags:
                tags.add(values[3])
            remaining -= 1
    return sorted(tags)


def safe_version(distribution: str, module_name: str | None = None) -> str:
    try:
        return importlib.metadata.version(distribution)
    except Exception:
        if module_name:
            try:
                module = importlib.import_module(module_name)
                return str(getattr(module, "__version__", "unknown"))
            except Exception as error:
                return f"unavailable: {error}"
        return "unavailable"


def run_capture(command: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout.strip()
    except Exception as error:
        return f"unavailable: {error}"


class Preflight:
    def __init__(self, config: dict, repo_root: Path, output_dir: Path):
        self.config = config
        self.repo_root = repo_root.resolve()
        self.output_dir = output_dir.resolve()
        self.checks: list[dict] = []
        self.input_metadata: dict = {}
        self.field_summaries: list[dict] = []
        self.counts: dict = {}
        self.warnings: list[str] = []
        self.log_lines: list[str] = []
        self.started = datetime.now(timezone.utc)

    def log(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        self.log_lines.append(line)
        print(line, flush=True)

    def check(
        self,
        name: str,
        passed: bool,
        *,
        expected=None,
        observed=None,
        severity: str = "hard",
        detail: str = "",
    ) -> bool:
        record = {
            "name": name,
            "severity": severity,
            "status": "pass" if bool(passed) else "fail",
            "expected": expected,
            "observed": observed,
            "detail": detail,
        }
        self.checks.append(record)
        self.log(f"{record['status'].upper()} [{severity}] {name}")
        return bool(passed)

    @property
    def hard_failures(self) -> list[dict]:
        return [
            item
            for item in self.checks
            if item["severity"] == "hard" and item["status"] != "pass"
        ]

    def verify_hashes(self) -> None:
        self.log("Hashing immutable scientific inputs and frozen design artifacts")
        for name, specification in self.config["inputs"].items():
            path = resolve_repo_path(self.repo_root, specification["path"])
            exists = self.check(
                f"input_exists:{name}", path.is_file(), expected=True,
                observed=path.is_file(),
            )
            if not exists:
                continue
            observed_hash = sha256_file(path)
            self.check(
                f"input_sha256:{name}",
                observed_hash == specification["sha256"],
                expected=specification["sha256"],
                observed=observed_hash,
            )
            self.input_metadata[name] = {
                **specification,
                "resolved_path": str(path),
                "size_bytes": path.stat().st_size,
                "observed_sha256": observed_hash,
            }

        for name, specification in self.config["frozen_design"].items():
            path = resolve_design_path(specification["path"])
            exists = self.check(
                f"design_exists:{name}", path.is_file(), expected=True,
                observed=path.is_file(),
            )
            if not exists:
                continue
            observed_hash = sha256_file(path)
            self.check(
                f"design_sha256:{name}",
                observed_hash == specification["sha256"],
                expected=specification["sha256"],
                observed=observed_hash,
            )

    def verify_firedrake_runtime(self) -> None:
        """Fail early when Docker was entered without activating Firedrake."""
        compiler = shutil.which("mpicc")
        self.check(
            "firedrake_mpi_compiler_available",
            compiler is not None,
            expected="mpicc on PATH",
            observed=compiler,
            detail=(
                "Run production_workflow/run_production.sh, which activates "
                "/home/firedrake/firedrake before invoking Python."
            ),
        )

    def verify_mesh_and_design(self) -> None:
        mesh_path = resolve_repo_path(
            self.repo_root, self.config["domain"]["mesh_path"]
        )
        tags = parse_msh_boundary_tags(mesh_path)
        self.check(
            "mesh_boundary_tags",
            tags == self.config["domain"]["all_boundary_ids"],
            expected=self.config["domain"]["all_boundary_ids"],
            observed=tags,
        )
        self.check(
            "boundary_partition",
            sorted(
                self.config["domain"]["dirichlet_ids"]
                + self.config["domain"]["stress_ids"]
            ) == tags,
            expected=tags,
            observed=sorted(
                self.config["domain"]["dirichlet_ids"]
                + self.config["domain"]["stress_ids"]
            ),
        )

        squares_path = resolve_design_path(
            self.config["frozen_design"]["selected_squares"]["path"]
        )
        with squares_path.open("r", newline="", encoding="utf-8") as stream:
            squares = list(csv.DictReader(stream))
        ids = [row["square_id"] for row in squares]
        self.check(
            "selected_square_count",
            len(squares) == self.config["expected_counts"]["selected_square_count"],
            expected=self.config["expected_counts"]["selected_square_count"],
            observed=len(squares),
        )
        self.check(
            "selected_square_ids_unique", len(ids) == len(set(ids)),
            expected=len(ids), observed=len(set(ids)),
        )

        nested_path = resolve_design_path(
            self.config["frozen_design"]["nested_square_masks"]["path"]
        )
        with np.load(nested_path, allow_pickle=False) as nested:
            central = nested["central"].astype(bool)
            annulus = nested["exclusion_annulus"].astype(bool)
            footprint = nested["full_footprint"].astype(bool)
            nested_ids = nested["square_ids"].astype(str).tolist()
        self.check(
            "nested_square_id_order", nested_ids == ids,
            expected=ids, observed=nested_ids,
        )
        self.check(
            "nested_central_annulus_disjoint",
            not np.any(central & annulus), expected=True,
            observed=bool(not np.any(central & annulus)),
        )
        self.check(
            "nested_union_equals_footprint",
            np.array_equal(central | annulus, footprint), expected=True,
            observed=bool(np.array_equal(central | annulus, footprint)),
        )

        partition_path = resolve_design_path(
            self.config["frozen_design"]["five_region_partition"]["path"]
        )
        with np.load(partition_path, allow_pickle=False) as partition:
            eligible = partition["eligible"].astype(bool)
            region_codes = partition["region_codes"]
        labels = region_codes[eligible]
        expected_regions = list(
            range(1, self.config["expected_counts"]["partition_region_count"] + 1)
        )
        self.check(
            "five_region_exhaustive",
            bool(np.all(np.isin(labels, expected_regions))),
            expected=expected_regions,
            observed=np.unique(labels).astype(int).tolist(),
        )
        self.check(
            "five_region_eligible_count",
            int(eligible.sum()) == self.config["expected_counts"]["support_grid_eligible"],
            expected=self.config["expected_counts"]["support_grid_eligible"],
            observed=int(eligible.sum()),
        )

        support_path = resolve_design_path(
            self.config["frozen_design"]["support_grid"]["path"]
        )
        with np.load(support_path, allow_pickle=False) as support:
            observed_counts = {
                "support_grid_in_mesh": int(support["inside"].sum()),
                "support_grid_grounded": int(support["grounded"].sum()),
                "support_grid_velocity_available": int(
                    support["velocity_available"].sum()
                ),
                "support_grid_eligible": int(support["eligible"].sum()),
            }
            feature_names = support["feature_names"].astype(str).tolist()
        for key, observed in observed_counts.items():
            self.check(
                key, observed == self.config["expected_counts"][key],
                expected=self.config["expected_counts"][key], observed=observed,
            )
        self.check(
            "support_grid_feature_order",
            feature_names == self.config["predictors"],
            expected=self.config["predictors"], observed=feature_names,
        )

    def verify_raster_metadata(self) -> None:
        sys.path.insert(0, str(self.repo_root))
        from src.data_preprocessing import read_raster_file
        from src.revised_raster_inputs import REVISED_GEOPHYSICS_INPUTS

        self.check(
            "revised_importer_slot_count", len(REVISED_GEOPHYSICS_INPUTS) == 8,
            expected=8, observed=len(REVISED_GEOPHYSICS_INPUTS),
        )
        self.check(
            "bouguer_omitted", REVISED_GEOPHYSICS_INPUTS[1] is None,
            expected=None, observed=REVISED_GEOPHYSICS_INPUTS[1],
        )
        self.check(
            "snow_omitted", REVISED_GEOPHYSICS_INPUTS[5] is None,
            expected=None, observed=REVISED_GEOPHYSICS_INPUTS[5],
        )

        for index, specification in enumerate(REVISED_GEOPHYSICS_INPUTS):
            if specification is None:
                continue
            descriptor = dict(specification)
            descriptor["path"] = str(
                resolve_repo_path(self.repo_root, descriptor["path"])
            )
            raster = read_raster_file(descriptor)
            try:
                attrs = dict(getattr(raster, "attrs", {}))
                coordinates = {}
                if hasattr(raster, "coords"):
                    for axis in ("x", "y"):
                        values = np.asarray(raster.coords[axis].values)
                        coordinates[axis] = {
                            "count": int(values.size),
                            "first": float(values[0]),
                            "last": float(values[-1]),
                        }
                self.input_metadata[f"geophysics_slot_{index}"] = {
                    "descriptor": descriptor,
                    "shape": list(getattr(raster, "shape", ())),
                    "attrs": {key: str(value) for key, value in attrs.items()},
                    "coordinates": coordinates,
                }
                self.check(
                    f"raster_loaded:{index}", True, expected=True, observed=True
                )
            finally:
                close = getattr(raster, "close", None)
                if close is not None:
                    close()

        bed_class = REVISED_GEOPHYSICS_INPUTS[7]
        self.check(
            "bed_class_explicit_missing_crs_exception",
            bed_class.get("assumed_crs") == 3031
            and bed_class.get("expected_crs") == 3031,
            expected={"assumed_crs": 3031, "expected_crs": 3031},
            observed={
                "assumed_crs": bed_class.get("assumed_crs"),
                "expected_crs": bed_class.get("expected_crs"),
            },
        )
        self.warnings.append(
            "The hash-locked bed-class auxiliary has no embedded CRS and is "
            "explicitly interpreted as EPSG:3031."
        )
        self.warnings.append(
            "ADMAP2S does not encode units; nT is recorded from product documentation."
        )

    @staticmethod
    def summarize_field(name: str, values: np.ndarray, population: str) -> dict:
        values = np.asarray(values, dtype="float64")
        finite = values[np.isfinite(values)]
        if not finite.size:
            return {
                "population": population,
                "field": name,
                "count": int(values.size),
                "finite": 0,
                "minimum": None,
                "q01": None,
                "median": None,
                "q99": None,
                "maximum": None,
            }
        quantiles = np.quantile(finite, [0.01, 0.5, 0.99])
        return {
            "population": population,
            "field": name,
            "count": int(values.size),
            "finite": int(finite.size),
            "minimum": float(finite.min()),
            "q01": float(quantiles[0]),
            "median": float(quantiles[1]),
            "q99": float(quantiles[2]),
            "maximum": float(finite.max()),
        }

    def build_invert(self, *, reg_c: float = 1.0):
        from src.invert_c_theta import Invert
        from src.revised_raster_inputs import REVISED_GEOPHYSICS_INPUTS

        domain = self.config["domain"]
        physics = self.config["physics"]
        mesh_path = resolve_repo_path(self.repo_root, domain["mesh_path"])
        outline_path = resolve_repo_path(self.repo_root, domain["outline_path"])
        object_ = Invert(
            outline=str(outline_path),
            mesh_name=str(mesh_path.with_suffix("")),
            read_mesh=True,
            degree=domain["degree"],
            δ=domain["delta_m"],
            m=physics["weertman_exponent_m"],
            ramp_power=physics["ramp_power"],
            reg_constant_c=float(reg_c),
            drichlet_ids=domain["dirichlet_ids"],
            side_ids=domain["side_ids"],
            opts=None,
        )
        object_.import_velocity_data(
            constant_val=physics["constant_friction_scale_c0"]
        )
        specifications = []
        for specification in REVISED_GEOPHYSICS_INPUTS:
            if specification is None:
                specifications.append(None)
            else:
                resolved = dict(specification)
                resolved["path"] = str(
                    resolve_repo_path(self.repo_root, resolved["path"])
                )
                specifications.append(resolved)
        object_.import_geophysics_data(specifications)
        return object_

    def verify_full_fields(self) -> None:
        self.log("Building the frozen Amundsen input object (no optimization)")
        import firedrake
        import icepack
        from icepack.constants import ice_density, water_density
        from src.feature_units import (
            driving_stress_mpa,
            stabilized_gradient_alignment,
        )
        from src.data_preprocessing import stable_xy_row_ids

        object_ = self.build_invert()
        try:
            expected = self.config["expected_counts"]
            self.check(
                "cg2_scalar_dofs", object_.Q.dim() == expected["cg2_scalar_dofs"],
                expected=expected["cg2_scalar_dofs"], observed=object_.Q.dim(),
            )
            summary = object_.velocity_mask_summary
            velocity_checks = {
                "velocity_window_width": int(object_.window.width),
                "velocity_window_height": int(object_.window.height),
                "valid_velocity_source_pixels": int(
                    summary["valid_velocity_and_source"]
                ),
                "selected_observations": int(summary["selected_observations"]),
                "selected_with_valid_error_pair": int(
                    summary["selected_with_valid_error_pair"]
                ),
            }
            for key, observed in velocity_checks.items():
                self.check(
                    key, observed == expected[key], expected=expected[key],
                    observed=observed,
                )
            self.check(
                "velocity_mask_policy",
                summary["policy"]
                == "finite_unmasked_vx_vy_and_source_gt_zero_when_available",
                expected="finite_unmasked_vx_vy_and_source_gt_zero_when_available",
                observed=summary["policy"],
            )
            self.check(
                "error_pairs_are_diagnostic",
                summary["selected_with_valid_error_pair"]
                < summary["selected_observations"],
                expected=True,
                observed=(
                    summary["selected_with_valid_error_pair"]
                    < summary["selected_observations"]
                ),
            )
            self.check(
                "velocity_full_grid_uses_cell_centers",
                getattr(object_, "velocity_full_grid_coordinate_mode", None)
                == "cell_center",
                expected="cell_center",
                observed=getattr(
                    object_, "velocity_full_grid_coordinate_mode", None
                ),
            )
            initial_velocity_finite = bool(
                np.isfinite(object_.u_initial.dat.data_ro).all()
            )
            self.check(
                "solver_initial_velocity_finite_after_documented_fill",
                initial_velocity_finite,
                expected=True,
                observed=initial_velocity_finite,
            )
            raw_velocity_valid = (
                np.isfinite(object_.u_exp_x.dat.data_ro)
                & np.isfinite(object_.u_exp_y.dat.data_ro)
            )
            raw_velocity_missing = int((~raw_velocity_valid).sum())
            self.check(
                "full_mesh_raw_velocity_missing_dofs",
                raw_velocity_missing
                == expected["full_mesh_raw_velocity_missing_dofs"],
                expected=expected["full_mesh_raw_velocity_missing_dofs"],
                observed=raw_velocity_missing,
                detail=(
                    "Raw missing values remain missing diagnostics and are not "
                    "inversion observations."
                ),
            )
            fill_summary = object_.velocity_initial_fill_summary
            self.check(
                "initial_velocity_nearest_filled_dofs",
                fill_summary["filled_dofs"]
                == expected["initial_velocity_nearest_filled_dofs"],
                expected=expected["initial_velocity_nearest_filled_dofs"],
                observed=fill_summary,
            )
            self.check(
                "dirichlet_velocity_nearest_filled_dofs",
                fill_summary["dirichlet_filled_dofs"]
                == expected["dirichlet_velocity_nearest_filled_dofs"],
                expected=expected["dirichlet_velocity_nearest_filled_dofs"],
                observed=fill_summary["dirichlet_filled_dofs"],
            )
            self.check(
                "initial_velocity_fill_does_not_add_observations",
                fill_summary["observations_affected"] == 0,
                expected=0,
                observed=fill_summary["observations_affected"],
            )
            self.counts["velocity_initial_fill"] = fill_summary
            if raw_velocity_missing:
                self.warnings.append(
                    f"Centered linear MEaSUREs sampling is missing at "
                    f"{raw_velocity_missing} of {object_.Q.dim()} full-mesh "
                    "DOFs. These remain excluded observations; the solver-only "
                    "initial/boundary field uses nearest valid centered pixels."
                )
            self.check(
                "bouguer_not_imported", object_.boug_anomaly is None,
                expected=None, observed=str(object_.boug_anomaly),
            )
            self.check(
                "snow_not_imported", object_.snow_accumulation is None,
                expected=None, observed=str(object_.snow_accumulation),
            )

            grad_h = firedrake.interpolate(firedrake.grad(object_.h), object_.V)
            grad_s = firedrake.interpolate(firedrake.grad(object_.s), object_.V)
            grad_b = firedrake.interpolate(firedrake.grad(object_.b), object_.V)
            grad_h_values = np.asarray(grad_h.dat.data_ro, dtype="float64")
            grad_s_values = np.asarray(grad_s.dat.data_ro, dtype="float64")
            grad_b_values = np.asarray(grad_b.dat.data_ro, dtype="float64")
            mag_h = np.linalg.norm(grad_h_values, axis=1)
            mag_s = np.linalg.norm(grad_s_values, axis=1)
            mag_b = np.linalg.norm(grad_b_values, axis=1)
            cos_theta_bs = stabilized_gradient_alignment(
                grad_b_values[:, 0], grad_b_values[:, 1],
                grad_s_values[:, 0], grad_s_values[:, 1],
            )
            full_features = {
                "s": np.asarray(object_.s.dat.data_ro, dtype="float64"),
                "h": np.asarray(object_.h.dat.data_ro, dtype="float64"),
                "mag_s": mag_s,
                "mag_h": mag_h,
                "driving_stress": np.asarray(
                    driving_stress_mpa(object_.h.dat.data_ro, mag_s),
                    dtype="float64",
                ),
                "surface_air_temp": np.asarray(
                    object_.surface_air_temp.dat.data_ro, dtype="float64"
                ),
                "b": np.asarray(object_.b.dat.data_ro, dtype="float64"),
                "mag_b": mag_b,
                "heatflux": np.asarray(object_.heatflux.dat.data_ro, dtype="float64"),
                "gravity_disturbance": np.asarray(
                    object_.gravity_disturbance.dat.data_ro, dtype="float64"
                ),
                "mag_anomaly": np.asarray(
                    object_.mag_anomaly.dat.data_ro, dtype="float64"
                ),
                "cos_theta_bs": cos_theta_bs,
            }
            h_values = full_features["h"]
            s_values = full_features["s"]
            water_ratio = np.divide(
                float(water_density) * np.maximum(0.0, h_values - s_values),
                float(ice_density) * h_values,
                out=np.ones_like(h_values),
                where=h_values > 0.0,
            )
            phi = np.maximum(1.0 - water_ratio, 0.0)
            grounded = (phi > 0.1) & (h_values > 0.0)
            self.check(
                "grounded_cg2_dofs",
                int(grounded.sum()) == expected["grounded_cg2_dofs"],
                expected=expected["grounded_cg2_dofs"],
                observed=int(grounded.sum()),
            )
            feature_matrix = np.column_stack(
                [full_features[name] for name in self.config["predictors"]]
            )
            self.check(
                "all_twelve_full_mesh_predictors_finite_on_grounded",
                bool(np.isfinite(feature_matrix[grounded]).all()),
                expected=True,
                observed=bool(np.isfinite(feature_matrix[grounded]).all()),
            )
            self.check(
                "cos_theta_bs_finite_and_bounded",
                bool(
                    np.isfinite(cos_theta_bs).all()
                    and np.all(cos_theta_bs >= -1.0)
                    and np.all(cos_theta_bs <= 1.0)
                ),
                expected=True,
                observed={
                    "finite": bool(np.isfinite(cos_theta_bs).all()),
                    "minimum": float(cos_theta_bs.min()),
                    "maximum": float(cos_theta_bs.max()),
                },
            )
            for name, values in full_features.items():
                self.field_summaries.append(
                    self.summarize_field(name, values[grounded], "grounded_cg2")
                )

            self.verify_support_oracle(
                object_, grad_h, grad_s, grad_b, driving_stress_mpa,
                stabilized_gradient_alignment,
            )
            self.verify_sparse_population(
                object_, grad_h, grad_s, grad_b, stable_xy_row_ids,
                driving_stress_mpa, stabilized_gradient_alignment,
                float(ice_density), float(water_density),
            )
        finally:
            del object_

    def verify_support_oracle(
        self, object_, grad_h, grad_s, grad_b, driving_stress_mpa,
        stabilized_gradient_alignment,
    ) -> None:
        import firedrake

        path = resolve_design_path(
            self.config["frozen_design"]["support_grid"]["path"]
        )
        with np.load(path, allow_pickle=False) as support:
            x_grid = support["x_grid"]
            y_grid = support["y_grid"]
            inside = support["inside"].astype(bool)
            oracle_features = support["features"]
            feature_names = support["feature_names"].astype(str).tolist()
        xx, yy = np.meshgrid(x_grid, y_grid)
        all_points = np.column_stack((xx.ravel(), yy.ravel()))
        requested = all_points[inside.ravel()]
        point_mesh = firedrake.VertexOnlyMesh(
            object_.mesh, requested, missing_points_behaviour="error"
        )
        scalar_space = firedrake.FunctionSpace(point_mesh, "DG", 0)
        vector_space = firedrake.VectorFunctionSpace(point_mesh, "DG", 0)
        returned = np.asarray(point_mesh.coordinates.dat.data_ro, dtype="float64")
        lookup = {
            (float(point[0]), float(point[1])): index
            for index, point in enumerate(all_points)
        }
        flat_indices = np.asarray(
            [lookup[(float(point[0]), float(point[1]))] for point in returned],
            dtype=int,
        )

        def scalar(field):
            return np.asarray(
                firedrake.interpolate(field, scalar_space).dat.data_ro,
                dtype="float64",
            )

        def vector(field):
            return np.asarray(
                firedrake.interpolate(field, vector_space).dat.data_ro,
                dtype="float64",
            )

        h = scalar(object_.h)
        s = scalar(object_.s)
        b = scalar(object_.b)
        gh = vector(grad_h)
        gs = vector(grad_s)
        gb = vector(grad_b)
        mag_h = np.linalg.norm(gh, axis=1)
        mag_s = np.linalg.norm(gs, axis=1)
        mag_b = np.linalg.norm(gb, axis=1)
        observed = {
            "s": s,
            "h": h,
            "mag_s": mag_s,
            "mag_h": mag_h,
            "driving_stress": driving_stress_mpa(h, mag_s),
            "surface_air_temp": scalar(object_.surface_air_temp),
            "b": b,
            "mag_b": mag_b,
            "heatflux": scalar(object_.heatflux),
            "gravity_disturbance": scalar(object_.gravity_disturbance),
            "mag_anomaly": scalar(object_.mag_anomaly),
            "cos_theta_bs": stabilized_gradient_alignment(
                gb[:, 0], gb[:, 1], gs[:, 0], gs[:, 1]
            ),
        }
        oracle_flat = oracle_features.reshape((-1, len(feature_names)))[flat_indices]
        observed_matrix = np.column_stack([observed[name] for name in feature_names])
        difference = np.abs(observed_matrix - oracle_flat)
        finite_pairs = np.isfinite(observed_matrix) & np.isfinite(oracle_flat)
        same_finiteness = np.array_equal(
            np.isfinite(observed_matrix), np.isfinite(oracle_flat)
        )
        maxima = {}
        passed = same_finiteness
        for index, name in enumerate(feature_names):
            mask = finite_pairs[:, index]
            maximum = float(difference[mask, index].max()) if mask.any() else 0.0
            maxima[name] = maximum
            passed &= bool(
                np.allclose(
                    observed_matrix[mask, index],
                    oracle_flat[mask, index],
                    rtol=1e-10,
                    atol=1e-10,
                )
            )
        self.check(
            "support_grid_elementwise_regression",
            passed,
            expected="all twelve fields rtol=1e-10, atol=1e-10",
            observed={
                "same_finiteness": same_finiteness,
                "maximum_absolute_difference": maxima,
            },
        )

    def verify_sparse_population(
        self, object_, grad_h, grad_s, grad_b, stable_xy_row_ids,
        driving_stress_mpa, stabilized_gradient_alignment,
        ice_density, water_density,
    ) -> None:
        import firedrake

        scalar_space = object_.Δ
        vector_space = firedrake.VectorFunctionSpace(
            scalar_space.mesh(), "DG", 0
        )

        def scalar(field):
            return np.asarray(
                firedrake.interpolate(field, scalar_space).dat.data_ro,
                dtype="float64",
            )

        def vector(field):
            return np.asarray(
                firedrake.interpolate(field, vector_space).dat.data_ro,
                dtype="float64",
            )

        h = scalar(object_.h)
        s = scalar(object_.s)
        b = scalar(object_.b)
        gh = vector(grad_h)
        gs = vector(grad_s)
        gb = vector(grad_b)
        mag_h = np.linalg.norm(gh, axis=1)
        mag_s = np.linalg.norm(gs, axis=1)
        mag_b = np.linalg.norm(gb, axis=1)
        features = np.column_stack(
            [
                s,
                h,
                mag_s,
                mag_h,
                driving_stress_mpa(h, mag_s),
                scalar(object_.surface_air_temp),
                b,
                mag_b,
                scalar(object_.heatflux),
                scalar(object_.gravity_disturbance),
                scalar(object_.mag_anomaly),
                stabilized_gradient_alignment(
                    gb[:, 0], gb[:, 1], gs[:, 0], gs[:, 1]
                ),
            ]
        )
        finite = np.isfinite(features).all(axis=1)
        water_ratio = np.divide(
            water_density * np.maximum(0.0, h - s),
            ice_density * h,
            out=np.ones_like(h),
            where=h > 0.0,
        )
        phi = np.maximum(1.0 - water_ratio, 0.0)
        grounded = (phi > 0.1) & (h > 0.0)
        eligible = finite & grounded
        self.check(
            "common_sparse_population_all_twelve_finite",
            bool(np.isfinite(features[eligible]).all()),
            expected=True,
            observed=bool(np.isfinite(features[eligible]).all()),
        )
        self.counts.update(
            {
                "selected_observations": int(len(h)),
                "sparse_grounded_phi_gt_0_1": int(grounded.sum()),
                "sparse_all_twelve_finite": int(finite.sum()),
                "sparse_common_eligible": int(eligible.sum()),
            }
        )
        coordinates = np.asarray(
            scalar_space.mesh().coordinates.dat.data_ro, dtype="float64"
        )[:, :2]
        unique_coordinates = np.unique(coordinates, axis=0).shape[0]
        self.check(
            "sparse_coordinates_unique",
            unique_coordinates == len(coordinates), expected=len(coordinates),
            observed=unique_coordinates,
        )
        row_ids = stable_xy_row_ids(coordinates[:, 0], coordinates[:, 1])
        self.check(
            "stable_row_id_format",
            bool(np.all(np.char.startswith(row_ids.astype(str), "xyh1-"))),
            expected="xyh1-*", observed=str(row_ids[0]),
        )
        self.check(
            "stable_row_id_count", len(row_ids) == len(coordinates),
            expected=len(coordinates), observed=len(row_ids),
        )

        squares_path = resolve_design_path(
            self.config["frozen_design"]["selected_squares"]["path"]
        )
        with squares_path.open("r", newline="", encoding="utf-8") as stream:
            squares = list(csv.DictReader(stream))
        square_counts = {}
        for row in squares:
            mask = (
                (coordinates[:, 0] >= float(row["test_xmin_m"]))
                & (coordinates[:, 0] < float(row["test_xmax_m"]))
                & (coordinates[:, 1] >= float(row["test_ymin_m"]))
                & (coordinates[:, 1] < float(row["test_ymax_m"]))
            )
            square_counts[row["square_id"]] = int(mask.sum())
        expected_counts = self.config["expected_counts"][
            "central_square_observations"
        ]
        self.check(
            "central_square_observation_counts",
            square_counts == expected_counts,
            expected=expected_counts,
            observed=square_counts,
        )

        sample_indices = np.linspace(
            0, len(coordinates) - 1, min(10000, len(coordinates)), dtype=int
        )
        source, errbed = object_.sample_bedmachine_auxiliaries(
            coordinates[sample_indices, 0], coordinates[sample_indices, 1]
        )
        self.check(
            "bedmachine_source_native_categories",
            bool(np.all(np.isin(source[np.isfinite(source)], [1, 2, 3, 4, 5, 6, 7, 10]))),
            expected=[1, 2, 3, 4, 5, 6, 7, 10],
            observed=np.unique(source[np.isfinite(source)]).astype(int).tolist(),
        )
        errbed_finite = np.isfinite(errbed)
        errbed_finite_fraction = float(errbed_finite.mean())
        self.counts["bedmachine_auxiliary_sample_size"] = int(len(errbed))
        self.counts["bedmachine_errbed_finite_sample"] = int(
            errbed_finite.sum()
        )
        self.check(
            "bedmachine_errbed_auxiliary_has_finite_values",
            bool(errbed_finite.any()),
            expected=True,
            observed={"finite_fraction": errbed_finite_fraction},
            detail="Missing errbed is permitted and cannot filter eligibility.",
        )
        if not errbed_finite.all():
            self.warnings.append(
                "BedMachine errbed contains missing auxiliary samples "
                f"({errbed_finite_fraction:.6f} finite in deterministic audit); "
                "these rows remain eligible."
            )
        for index, name in enumerate(self.config["predictors"]):
            self.field_summaries.append(
                self.summarize_field(name, features[eligible, index], "sparse_eligible")
            )

    def environment(self) -> dict:
        return {
            "utc_started": self.started.isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "versions": {
                "firedrake": safe_version("firedrake", "firedrake"),
                "icepack": safe_version("icepack", "icepack"),
                "numpy": safe_version("numpy", "numpy"),
                "xarray": safe_version("xarray", "xarray"),
                "rasterio": safe_version("rasterio", "rasterio"),
                "pyproj": safe_version("pyproj", "pyproj"),
                "petsc4py": safe_version("petsc4py", "petsc4py"),
                "pyadjoint": safe_version("pyadjoint", "pyadjoint"),
            },
            "git_head": run_capture(["git", "rev-parse", "HEAD"], self.repo_root),
            "git_status": run_capture(["git", "status", "--short"], self.repo_root),
            "docker_image_hint": os.environ.get("JOG_DOCKER_IMAGE", "unrecorded"),
            "docker_image_id": os.environ.get(
                "JOG_DOCKER_IMAGE_ID", "unrecorded"
            ),
        }

    def source_identity(self) -> dict:
        candidates = [
            self.repo_root / "src" / "data_preprocessing.py",
            self.repo_root / "src" / "create_mesh.py",
            self.repo_root / "src" / "helper_functions.py",
            self.repo_root / "src" / "invert_c_theta.py",
            self.repo_root / "src" / "revised_raster_inputs.py",
            self.repo_root / "src" / "feature_units.py",
            Path(__file__).resolve(),
            DEFAULT_CONFIG.resolve(),
            WORKFLOW_ROOT / "lcurve_runtime.py",
            WORKFLOW_ROOT / "lcurve_selection.py",
            WORKFLOW_ROOT / "lcurve_orchestrator.py",
            WORKFLOW_ROOT / "load_definitive_inversion.py",
            WORKFLOW_ROOT / "run_production.sh",
        ]
        return {
            str(path): sha256_file(path)
            for path in candidates
            if path.is_file()
        }

    def write_outputs(self, config_path: Path, exception_text: str | None = None) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / "logs").mkdir()
        snapshot = self.output_dir / "source_snapshot"
        snapshot.mkdir()

        resolved_config = json.loads(json.dumps(self.config))
        resolved_config["resolved_repo_root"] = str(self.repo_root)
        resolved_config["config_sha256"] = sha256_file(config_path)
        (self.output_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "input_metadata.json").write_text(
            json.dumps(self.input_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        environment = self.environment()
        (self.output_dir / "environment.txt").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (self.output_dir / "checks.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["name", "severity", "status", "expected", "observed", "detail"],
            )
            writer.writeheader()
            for item in self.checks:
                writer.writerow(
                    {
                        **item,
                        "expected": json.dumps(item["expected"], sort_keys=True),
                        "observed": json.dumps(item["observed"], sort_keys=True),
                    }
                )
        with (self.output_dir / "field_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            fieldnames = [
                "population", "field", "count", "finite", "minimum", "q01",
                "median", "q99", "maximum",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.field_summaries)

        source_files = [
            self.repo_root / "src" / "data_preprocessing.py",
            self.repo_root / "src" / "create_mesh.py",
            self.repo_root / "src" / "helper_functions.py",
            self.repo_root / "src" / "invert_c_theta.py",
            self.repo_root / "src" / "revised_raster_inputs.py",
            self.repo_root / "src" / "feature_units.py",
            Path(__file__).resolve(),
            WORKFLOW_ROOT / "lcurve_runtime.py",
            WORKFLOW_ROOT / "lcurve_selection.py",
            WORKFLOW_ROOT / "lcurve_orchestrator.py",
            WORKFLOW_ROOT / "load_definitive_inversion.py",
            WORKFLOW_ROOT / "run_production.sh",
            config_path.resolve(),
        ]
        for path in source_files:
            if path.is_file():
                shutil.copy2(path, snapshot / path.name)
        (snapshot / "repository_diff.patch").write_text(
            run_capture(["git", "diff", "--", "src"], self.repo_root) + "\n",
            encoding="utf-8",
        )
        if exception_text:
            self.log_lines.append(exception_text)
        (self.output_dir / "logs" / "preflight.log").write_text(
            "\n".join(self.log_lines) + "\n", encoding="utf-8"
        )

        status = "pass" if not self.hard_failures and exception_text is None else "fail"
        manifest = {
            "schema": "jog-production-preflight-manifest-v1",
            "kind": "whole_sector_input_preflight",
            "status": status,
            "run_id": self.output_dir.name,
            "started_utc": self.started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "working_directory": os.getcwd(),
            "repo_root": str(self.repo_root),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "environment": environment,
            "source_sha256": self.source_identity(),
            "counts": self.counts,
            "warnings": self.warnings,
            "checks": self.checks,
            "hard_failure_count": len(self.hard_failures),
            "exception": exception_text,
        }
        output_hashes = {}
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "preflight_manifest.json":
                output_hashes[str(path.relative_to(self.output_dir))] = sha256_file(path)
        manifest["output_sha256"] = output_hashes
        manifest["manifest_id"] = manifest_identifier(manifest)
        (self.output_dir / "preflight_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    def run(self, *, level: str, config_path: Path) -> dict:
        exception_text = None
        try:
            self.verify_hashes()
            self.verify_mesh_and_design()
            self.verify_raster_metadata()
            if self.hard_failures:
                raise RuntimeError(
                    "Static preflight failed; full field construction is blocked."
                )
            if level == "full":
                self.verify_firedrake_runtime()
                if self.hard_failures:
                    raise RuntimeError(
                        "Firedrake runtime preflight failed; full field "
                        "construction is blocked."
                    )
                self.verify_full_fields()
        except Exception:
            exception_text = traceback.format_exc()
            self.check(
                "preflight_completed_without_exception",
                False,
                expected=True,
                observed=False,
                detail=exception_text.splitlines()[-1],
            )
        else:
            self.check(
                "preflight_completed_without_exception",
                True,
                expected=True,
                observed=True,
            )
        return self.write_outputs(config_path, exception_text=exception_text)


def make_output_dir(output_root: Path, config_path: Path, run_id: str | None) -> Path:
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"preflight_{timestamp}_{sha256_file(config_path)[:10]}"
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("run_id may contain only letters, numbers, hyphens, and underscores.")
    output = output_root / run_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory {output}")
    return output


def explicit_output_dir(output_root: Path, run_id: str | None) -> Path:
    if run_id is None:
        raise ValueError("This command requires an explicit immutable --run-id.")
    if not run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in run_id
    ):
        raise ValueError("run_id may contain only letters, numbers, hyphens, and underscores.")
    return output_root / run_id


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "describe",
            "forward-smoke",
            "lcurve-point",
            "lcurve-confirmation",
            "lcurve",
        ],
        help=(
            "Preflight validates inputs without optimization; describe prints "
            "the config; forward-smoke runs C=theta=0 without optimization; "
            "lcurve-point runs one independently initialized inversion; "
            "lcurve-confirmation runs one verified same-reg_C block; "
            "lcurve executes the resumable frozen study protocol."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--output-root", type=Path,
        default=DEFAULT_REPO_ROOT / "production_runs",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--reg-c", type=float,
        help="Required positive regularization scale for an L-curve child.",
    )
    parser.add_argument(
        "--parent-point-manifest",
        type=Path,
        help="Hash-verified parent point for lcurve-confirmation.",
    )
    parser.add_argument(
        "--confirmation-round",
        type=int,
        help="One-based confirmation round for lcurve-confirmation.",
    )
    parser.add_argument(
        "--level", choices=["static", "full"], default="full",
        help="Full additionally constructs and audits the FE input fields.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing lcurve run under its unchanged frozen contract.",
    )
    parser.add_argument(
        "--forward-smoke-dir",
        type=Path,
        help=(
            "Required passing, source-matched whole-sector forward-smoke "
            "directory for a production lcurve launch."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.command == "describe":
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    repo_root = args.repo_root.resolve()
    if args.resume and args.command != "lcurve":
        raise ValueError("--resume is supported only by the lcurve command.")
    if args.forward_smoke_dir is not None and args.command != "lcurve":
        raise ValueError("--forward-smoke-dir is supported only by lcurve.")
    if args.parent_point_manifest is not None and args.command != "lcurve-confirmation":
        raise ValueError(
            "--parent-point-manifest is supported only by lcurve-confirmation."
        )
    if args.confirmation_round is not None and args.command != "lcurve-confirmation":
        raise ValueError(
            "--confirmation-round is supported only by lcurve-confirmation."
        )
    if args.command == "forward-smoke":
        output_dir = explicit_output_dir(args.output_root.resolve(), args.run_id)
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir}")
        from lcurve_runtime import run_forward_smoke

        manifest = run_forward_smoke(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            output_dir=output_dir,
            preflight_class=Preflight,
        )
        return 0 if manifest["status"] == "pass" else 1
    if args.command == "lcurve-point":
        if args.reg_c is None:
            raise ValueError("lcurve-point requires --reg-c.")
        if args.run_id is None:
            raise ValueError("lcurve-point requires an explicit immutable --run-id.")
        output_dir = make_output_dir(
            args.output_root.resolve(), config_path, args.run_id
        )
        from lcurve_runtime import run_lcurve_point

        manifest = run_lcurve_point(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            output_dir=output_dir,
            reg_c=args.reg_c,
            preflight_class=Preflight,
        )
        return 0 if manifest["status"] in {"valid", "invalid"} else 1
    if args.command == "lcurve-confirmation":
        if args.reg_c is None:
            raise ValueError("lcurve-confirmation requires --reg-c.")
        if args.run_id is None:
            raise ValueError(
                "lcurve-confirmation requires an explicit immutable --run-id."
            )
        if args.parent_point_manifest is None or args.confirmation_round is None:
            raise ValueError(
                "lcurve-confirmation requires --parent-point-manifest and "
                "--confirmation-round."
            )
        output_dir = make_output_dir(
            args.output_root.resolve(), config_path, args.run_id
        )
        from lcurve_runtime import run_lcurve_confirmation

        manifest = run_lcurve_confirmation(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            output_dir=output_dir,
            reg_c=args.reg_c,
            parent_point_manifest_path=args.parent_point_manifest.resolve(),
            confirmation_round=args.confirmation_round,
            preflight_class=Preflight,
        )
        return 0 if manifest["status"] in {"valid", "invalid"} else 1
    if args.command == "lcurve":
        if args.forward_smoke_dir is None and not args.resume:
            raise ValueError("lcurve requires --forward-smoke-dir.")
        output_dir = explicit_output_dir(args.output_root.resolve(), args.run_id)
        from lcurve_orchestrator import run_lcurve_study

        manifest = run_lcurve_study(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            output_dir=output_dir,
            preflight_class=Preflight,
            resume=args.resume,
            forward_smoke_dir=(
                args.forward_smoke_dir.resolve()
                if args.forward_smoke_dir is not None
                else None
            ),
        )
        return 0 if manifest["status"].startswith("complete") else 1
    output_dir = make_output_dir(args.output_root.resolve(), config_path, args.run_id)
    preflight = Preflight(config, repo_root, output_dir)
    manifest = preflight.run(level=args.level, config_path=config_path)
    print(json.dumps({
        "status": manifest["status"],
        "manifest_id": manifest["manifest_id"],
        "output_dir": str(output_dir),
        "hard_failure_count": manifest["hard_failure_count"],
    }, indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
