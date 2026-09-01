"""Production execution primitives for one revised Amundsen L-curve point."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

import numpy as np


ROL_STATUS_PATTERN = re.compile(
    r"Optimization Terminated with Status:\s*(?P<status>.+?)\s*$"
)
ROL_ITERATION_PATTERN = re.compile(
    r"^\s*(?P<iteration>\d+)\s+[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+"
)


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


def _stable_environment(environment: dict) -> dict:
    """Remove per-run metadata while retaining the frozen runtime identity."""
    keys = [
        "docker_image_hint",
        "docker_image_id",
        "executable",
        "git_head",
        "git_status",
        "hostname",
        "platform",
        "python",
        "versions",
    ]
    return {key: environment.get(key) for key in keys}


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def classify_algorithm_state(
    state, *, gradient_tolerance: float, step_tolerance: float
) -> dict:
    """Classify the installed ROL default status test from exposed state fields."""
    values = {
        "objective": float(state.value),
        "gradient_norm": float(state.gnorm),
        "step_norm": float(state.snorm),
        "constraint_norm": float(state.cnorm),
    }
    finite_required = all(
        math.isfinite(values[name])
        for name in ("objective", "gradient_norm", "step_norm")
    )
    if not finite_required:
        termination = "nonfinite_state"
    elif values["gradient_norm"] <= gradient_tolerance:
        termination = "converged_gradient"
    elif values["step_norm"] <= step_tolerance:
        termination = "converged_step"
    else:
        # The installed ROL StatusTest has only gradient, step, and iteration
        # exits. If solve() returned normally and neither tolerance passed, the
        # configured iteration limit is the terminating condition.
        termination = "iteration_limit"
    return {**values, "termination": termination}


def parse_rol_attempts(log_text: str) -> list[dict]:
    """Extract iteration counts and native status text between attempt markers."""
    attempts = []
    for match in re.finditer(r"JOG_ROL_ATTEMPT_BEGIN\s+(\d+)", log_text):
        attempt = int(match.group(1))
        end_marker = f"JOG_ROL_ATTEMPT_END {attempt}"
        end = log_text.find(end_marker, match.end())
        section = log_text[match.end() : end if end >= 0 else len(log_text)]
        iterations = [
            int(item.group("iteration"))
            for item in map(ROL_ITERATION_PATTERN.match, section.splitlines())
            if item is not None
        ]
        statuses = [
            item.group("status").strip()
            for item in map(ROL_STATUS_PATTERN.search, section.splitlines())
            if item is not None
        ]
        attempts.append(
            {
                "attempt": attempt,
                "rol_last_iteration": max(iterations) if iterations else None,
                "rol_status_text": statuses[-1] if statuses else None,
                "end_marker_present": end >= 0,
            }
        )
    return attempts


def native_status_matches(termination: str, native_status: str | None) -> bool:
    """Cross-check exposed ROL state against the native termination message."""
    if not native_status:
        return False
    normalized = native_status.strip().lower()
    if termination == "converged_gradient":
        return "converged" in normalized
    if termination == "converged_step":
        return "converged" in normalized or "step tolerance" in normalized
    if termination == "iteration_limit":
        return "iteration limit" in normalized
    return False


def _flush_c_streams() -> None:
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


@contextmanager
def process_log(path: Path):
    """Redirect Python and native stdout/stderr to one deterministic run log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    _flush_c_streams()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with path.open("wb", buffering=0) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            _flush_c_streams()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_state(
    object_, velocity, output_dir: Path, *, mesh_sha256: str | None = None
) -> dict:
    import firedrake

    object_.C.rename("log_friction_C")
    theta_field = getattr(object_, "\u03b8")
    theta_field.rename("log_fluidity_theta")
    velocity.rename("velocity")
    coordinate_field = firedrake.interpolate(object_.mesh.coordinates, object_.V)
    arrays = {
        "coordinates": np.asarray(coordinate_field.dat.data_ro[:, :2]),
        "C": np.asarray(object_.C.dat.data_ro),
        "theta": np.asarray(theta_field.dat.data_ro),
        "velocity": np.asarray(velocity.dat.data_ro),
    }
    array_files = {}
    for name, values in arrays.items():
        path = output_dir / f"state_{name}.npy"
        np.save(path, values, allow_pickle=False)
        array_files[name] = path.name
    state_schema = {
        "schema": "jog-frozen-mesh-dof-state-v1",
        "mesh_is_external_hash_locked_input": True,
        "mesh_sha256": mesh_sha256,
        "scalar_space": {"family": "CG", "degree": int(object_.degree)},
        "vector_space": {
            "family": "CG",
            "degree": int(object_.degree),
            "dimension": 2,
        },
        "assignment_policy": (
            "Load the exact hash-locked mesh and recorded Firedrake environment; "
            "verify state_coordinates.npy, then assign arrays to matching Q/V DOFs."
        ),
        "arrays": {
            name: {
                "path": array_files[name],
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
            for name, values in arrays.items()
        },
    }
    schema_path = output_dir / "state_schema.json"
    schema_path.write_text(
        json.dumps(state_schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "state_schema": schema_path.name,
        "array_files": array_files,
        "C_minimum": float(np.min(object_.C.dat.data_ro)),
        "C_maximum": float(np.max(object_.C.dat.data_ro)),
        "C_finite": bool(np.isfinite(object_.C.dat.data_ro).all()),
        "theta_minimum": float(np.min(theta_field.dat.data_ro)),
        "theta_maximum": float(np.max(theta_field.dat.data_ro)),
        "theta_finite": bool(np.isfinite(theta_field.dat.data_ro).all()),
        "velocity_finite": bool(np.isfinite(velocity.dat.data_ro).all()),
    }


def _objective_metrics(object_, reg_c: float) -> dict:
    import firedrake

    velocity = object_.simulation_C(object_.C)
    misfit = float(firedrake.assemble(object_.loss_functional_nosigma(velocity)))
    length = firedrake.Constant(7.5e3)
    raw_gradient_form = (
        firedrake.inner(firedrake.grad(object_.C), firedrake.grad(object_.C))
        * firedrake.dx(object_.mesh)
    )
    raw_gradient_integral = float(firedrake.assemble(raw_gradient_form))
    unweighted_form = (
        0.5
        / object_.area
        * length**2
        * firedrake.inner(firedrake.grad(object_.C), firedrake.grad(object_.C))
        * firedrake.dx(object_.mesh)
    )
    unweighted_roughness = float(firedrake.assemble(unweighted_form))
    weighted_penalty = float(
        firedrake.assemble(object_.regularization_C_grad(object_.C))
    )
    objective = misfit + weighted_penalty
    expected_weighted = unweighted_roughness / reg_c**2
    return {
        "misfit": misfit,
        "raw_gradient_integral": raw_gradient_integral,
        "unweighted_roughness": unweighted_roughness,
        "weighted_penalty": weighted_penalty,
        "objective": objective,
        "expected_weighted_penalty": expected_weighted,
        "weighted_penalty_identity": bool(
            np.isclose(weighted_penalty, expected_weighted, rtol=1e-10, atol=1e-12)
        ),
        "all_finite": bool(
            np.isfinite(
                [
                    misfit,
                    raw_gradient_integral,
                    unweighted_roughness,
                    weighted_penalty,
                    objective,
                ]
            ).all()
        ),
        "velocity": velocity,
    }


def relative_change(previous: float, current: float) -> float:
    """Return the full-precision relative change from one block endpoint."""
    previous = float(previous)
    current = float(current)
    if not (math.isfinite(previous) and math.isfinite(current)):
        return math.inf
    denominator = max(abs(previous), np.finfo("float64").tiny)
    return abs(current - previous) / denominator


def assess_practical_stability(
    blocks: list[dict],
    *,
    min_blocks: int,
    stable_transitions: int,
    relative_misfit_tolerance: float,
    relative_roughness_tolerance: float,
    gradient_safety_tolerance: float,
) -> dict:
    """Pure evaluator for the common practical E/R acceptance rule."""
    if min_blocks < 1 or stable_transitions < 1:
        raise ValueError("Block and stable-transition counts must be positive.")
    tolerances = (
        relative_misfit_tolerance,
        relative_roughness_tolerance,
        gradient_safety_tolerance,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("Stability tolerances must be finite and nonnegative.")

    def value(block, name, default=None):
        if name in block:
            return block[name]
        return (block.get("metrics") or {}).get(name, default)

    def within(delta, tolerance):
        return bool(
            math.isfinite(delta)
            and (
                delta <= tolerance
                or math.isclose(delta, tolerance, rel_tol=1e-12, abs_tol=1e-15)
            )
        )

    transitions = []
    for previous, current in zip(blocks[:-1], blocks[1:]):
        delta_e = relative_change(
            value(previous, "misfit", math.nan),
            value(current, "misfit", math.nan),
        )
        delta_r = relative_change(
            value(previous, "unweighted_roughness", math.nan),
            value(current, "unweighted_roughness", math.nan),
        )
        if math.isclose(
            delta_e,
            relative_misfit_tolerance,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            delta_e = float(relative_misfit_tolerance)
        if math.isclose(
            delta_r,
            relative_roughness_tolerance,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            delta_r = float(relative_roughness_tolerance)
        e_passed = within(delta_e, relative_misfit_tolerance)
        r_passed = within(delta_r, relative_roughness_tolerance)
        transitions.append(
            {
                "from_block": int(previous["block"]),
                "to_block": int(current["block"]),
                "relative_misfit_change": delta_e,
                "relative_roughness_change": delta_r,
                "misfit_passed": e_passed,
                "roughness_passed": r_passed,
                "joint_passed": bool(e_passed and r_passed),
            }
        )
    consecutive = 0
    for transition in reversed(transitions):
        if not transition["joint_passed"]:
            break
        consecutive += 1

    terminal_gradient = (
        float(value(blocks[-1], "gradient_norm", math.inf))
        if blocks
        else math.inf
    )
    gradient_passed = bool(
        math.isfinite(terminal_gradient)
        and (
            terminal_gradient <= gradient_safety_tolerance
            or math.isclose(
                terminal_gradient,
                gradient_safety_tolerance,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )
    )
    finite_and_consistent = bool(
        blocks
        and all(
            value(block, "all_finite", False) is True
            and (
                block.get("objective_identities_passed") is True
                or (
                    value(block, "weighted_penalty_identity", False) is True
                    and value(
                        block, "rol_objective_matches_reassembled", False
                    )
                    is True
                )
            )
            for block in blocks
        )
    )
    accepted = bool(
        len(blocks) >= min_blocks
        and consecutive >= stable_transitions
        and gradient_passed
        and finite_and_consistent
    )
    return {
        "accepted": accepted,
        "acceptance_basis": "practical_er_stability" if accepted else None,
        "evaluated_block_count": len(blocks),
        "minimum_blocks": int(min_blocks),
        "required_stable_transitions": int(stable_transitions),
        "consecutive_stable_transitions": int(consecutive),
        "relative_misfit_tolerance": float(relative_misfit_tolerance),
        "relative_roughness_tolerance": float(relative_roughness_tolerance),
        "gradient_safety_tolerance": float(gradient_safety_tolerance),
        "terminal_gradient_norm": terminal_gradient,
        "gradient_safety_passed": gradient_passed,
        "finite_and_objective_consistent": finite_and_consistent,
        "relative_change_definition": (
            "abs(current-previous)/max(abs(previous),float64_tiny)"
        ),
        "transitions": transitions,
    }


def evaluate_er_stability(
    blocks: list[dict],
    *,
    minimum_blocks: int,
    stable_transitions_required: int,
    relative_misfit_tolerance: float,
    relative_roughness_tolerance: float,
    gradient_safety_tolerance: float,
) -> dict:
    """Evaluate the preregistered joint E/R block-stability gate."""
    assessment = assess_practical_stability(
        blocks,
        min_blocks=minimum_blocks,
        stable_transitions=stable_transitions_required,
        relative_misfit_tolerance=relative_misfit_tolerance,
        relative_roughness_tolerance=relative_roughness_tolerance,
        gradient_safety_tolerance=gradient_safety_tolerance,
    )
    return {
        "passed": assessment["accepted"],
        "acceptance_basis": assessment["acceptance_basis"],
        "relative_change_definition": assessment["relative_change_definition"],
        "minimum_blocks": assessment["minimum_blocks"],
        "observed_blocks": assessment["evaluated_block_count"],
        "stable_transitions_required": assessment[
            "required_stable_transitions"
        ],
        "consecutive_stable_transitions": assessment[
            "consecutive_stable_transitions"
        ],
        "relative_misfit_tolerance": assessment[
            "relative_misfit_tolerance"
        ],
        "relative_roughness_tolerance": assessment[
            "relative_roughness_tolerance"
        ],
        "gradient_safety_tolerance": assessment[
            "gradient_safety_tolerance"
        ],
        "terminal_gradient_norm": assessment["terminal_gradient_norm"],
        "gradient_safe": assessment["gradient_safety_passed"],
        "finite_and_objective_consistent": assessment[
            "finite_and_objective_consistent"
        ],
        "transition_evidence": assessment["transitions"],
    }


def run_lcurve_point(
    *,
    config: dict,
    config_path: Path,
    repo_root: Path,
    output_dir: Path,
    reg_c: float,
    preflight_class,
    parent_point_manifest_path: Path | None = None,
    confirmation_round: int | None = None,
) -> dict:
    """Run an independent L-curve point or one verified confirmation block."""
    if not math.isfinite(reg_c) or reg_c <= 0.0:
        raise ValueError("reg_C must be a finite positive number.")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    is_confirmation = parent_point_manifest_path is not None
    if is_confirmation != (confirmation_round is not None):
        raise ValueError(
            "Confirmation requires both parent_point_manifest_path and "
            "confirmation_round."
        )
    if confirmation_round is not None and confirmation_round < 1:
        raise ValueError("confirmation_round must be positive.")
    if confirmation_round is not None and confirmation_round > int(
        config["inversion"].get("lcurve_max_confirmation_rounds", 2)
    ):
        raise ValueError("confirmation_round exceeds the frozen protocol ceiling.")
    parent_manifest = None
    parent_manifest_sha256 = None
    parent_metrics = None
    parent_fields = None
    parent_cumulative_blocks = 0
    parent_input_c = None
    if is_confirmation:
        parent_point_manifest_path = parent_point_manifest_path.resolve()
        parent_manifest = json.loads(
            parent_point_manifest_path.read_text(encoding="utf-8")
        )
        if parent_manifest.get("schema") != "jog-production-lcurve-point-v2":
            raise RuntimeError("Unexpected confirmation-parent point schema.")
        if canonical_identifier(parent_manifest) != parent_manifest.get(
            "manifest_id"
        ):
            raise RuntimeError("Confirmation-parent manifest identifier is invalid.")
        if float(parent_manifest.get("reg_c")) != float(reg_c):
            raise RuntimeError("Confirmation cannot change reg_C.")
        if parent_manifest.get("status") not in {"valid", "invalid"}:
            raise RuntimeError("Confirmation parent lacks a usable saved state.")
        if (
            parent_manifest.get("status") == "invalid"
            and parent_manifest.get("state_reusable") is not True
        ):
            raise RuntimeError(
                "Invalid confirmation parent lacks explicit reusable-state evidence."
            )
        parent_kind = parent_manifest.get("run_kind")
        if confirmation_round == 1 and parent_kind != "independent":
            raise RuntimeError("Confirmation round 1 must start from an independent point.")
        if confirmation_round > 1:
            if parent_kind == "confirmation" and int(
                parent_manifest.get("confirmation_round", 0)
            ) != confirmation_round - 1:
                raise RuntimeError(
                    "A confirmation parent must come from the preceding round."
                )
            if parent_kind not in {"independent", "confirmation"}:
                raise RuntimeError("Confirmation parent has an unknown run kind.")
        parent_manifest_sha256 = sha256_file(parent_point_manifest_path)
        parent_metrics = parent_manifest.get("metrics") or {}
        parent_fields = parent_manifest.get("fields") or {}
        parent_cumulative_blocks = int(
            parent_manifest.get(
                "cumulative_block_count",
                len(parent_manifest.get("blocks") or []),
            )
        )
        parent_blocks = parent_manifest.get("blocks") or []
        if parent_cumulative_blocks < len(parent_blocks) or not parent_blocks:
            raise RuntimeError("Confirmation parent has an invalid cumulative block count.")
        if int(parent_blocks[-1].get("block", -1)) != parent_cumulative_blocks:
            raise RuntimeError(
                "Confirmation parent endpoint block disagrees with its ancestry."
            )
        if parent_kind == "independent" and parent_cumulative_blocks != len(
            parent_blocks
        ):
            raise RuntimeError(
                "Independent confirmation parent has inconsistent block ancestry."
            )
        if parent_kind == "confirmation":
            parent_reference = parent_manifest.get("parent_point_manifest") or {}
            prior_count = int(parent_reference.get("cumulative_block_count", -1))
            if (
                len(parent_blocks) != 1
                or prior_count < 1
                or prior_count + len(parent_blocks) != parent_cumulative_blocks
            ):
                raise RuntimeError(
                    "Confirmation parent has inconsistent chained block ancestry."
                )
        required_parent_metrics = [
            "misfit",
            "unweighted_roughness",
            "weighted_penalty",
            "objective",
        ]
        if not all(
            name in parent_metrics
            and math.isfinite(float(parent_metrics[name]))
            for name in required_parent_metrics
        ):
            raise RuntimeError("Confirmation parent lacks finite endpoint metrics.")
        if not (
            parent_metrics.get("all_finite") is True
            and parent_metrics.get("weighted_penalty_identity") is True
            and parent_metrics.get("rol_objective_matches_reassembled") is True
            and parent_fields.get("C_finite") is True
            and parent_fields.get("theta_finite") is True
            and parent_fields.get("velocity_finite") is True
            and float(parent_fields.get("theta_minimum", math.nan)) == 0.0
            and float(parent_fields.get("theta_maximum", math.nan)) == 0.0
            and parent_manifest.get("solver_log_crosscheck_passed") is True
        ):
            raise RuntimeError(
                "Confirmation parent fails the reusable endpoint-state gate."
            )
    started = datetime.now(timezone.utc)
    active_path = output_dir / "active_process.json"
    active_path.write_text(
        json.dumps(
            {
                "schema": "jog-active-lcurve-point-v2",
                "pid": os.getpid(),
                "started_utc": started.isoformat(),
                "point_directory": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = output_dir / "logs" / "point.log"
    blocks = []
    stability = None
    metrics = None
    fields = None
    input_manifest = None
    exception_text = None
    point_status = "failed"
    native_termination = "failed"
    acceptance_basis = None
    state_reusable = False
    preflight = preflight_class(
        config, repo_root.resolve(), output_dir / "input_preflight"
    )

    with process_log(log_path):
        try:
            input_manifest = preflight.run(level="static", config_path=config_path)
            if input_manifest["status"] != "pass":
                raise RuntimeError("Point-level immutable input preflight failed.")
            if is_confirmation:
                if parent_manifest.get("config_sha256") != sha256_file(config_path):
                    raise RuntimeError("Confirmation parent used a different config.")
                if parent_manifest.get("source_sha256") != preflight.source_identity():
                    raise RuntimeError(
                        "Confirmation parent used different scientific sources."
                    )
                if _stable_environment(
                    parent_manifest.get("environment") or {}
                ) != _stable_environment(preflight.environment()):
                    raise RuntimeError(
                        "Confirmation parent used a different runtime environment."
                    )
            object_ = preflight.build_invert(reg_c=reg_c)
            try:
                inversion = config["inversion"]
                expected = config["expected_counts"]
                if object_.Q.dim() != expected["cg2_scalar_dofs"]:
                    raise RuntimeError("Unexpected scalar finite-element population.")
                if object_.N != expected["selected_observations"]:
                    raise RuntimeError("Unexpected centered observation population.")
                initial_C = np.asarray(object_.C.dat.data_ro, dtype="float64")
                theta_field = getattr(object_, "\u03b8")
                if not np.array_equal(
                    theta_field.dat.data_ro,
                    np.full_like(
                        theta_field.dat.data_ro,
                        config["physics"]["initial_log_fluidity_theta"],
                    ),
                ):
                    raise RuntimeError("L-curve theta must remain fixed at zero.")

                initial_parent_control = None
                if is_confirmation:
                    arrays = parent_fields.get("array_files") or {}
                    required_arrays = {"C", "coordinates", "theta"}
                    if not required_arrays.issubset(arrays):
                        raise RuntimeError(
                            "Confirmation parent lacks required state arrays."
                        )
                    parent_root = parent_point_manifest_path.parent
                    schema_relative = parent_fields.get("state_schema")
                    if not schema_relative:
                        raise RuntimeError(
                            "Confirmation parent lacks its state schema reference."
                        )
                    schema_path = (parent_root / schema_relative).resolve()
                    try:
                        schema_path.relative_to(parent_root.resolve())
                    except ValueError as error:
                        raise RuntimeError(
                            "Confirmation-parent state schema escapes its directory."
                        ) from error
                    expected_schema_hash = (
                        parent_manifest.get("output_sha256") or {}
                    ).get(schema_relative)
                    if (
                        not schema_path.is_file()
                        or not expected_schema_hash
                        or sha256_file(schema_path) != expected_schema_hash
                    ):
                        raise RuntimeError(
                            "Confirmation-parent state schema failed hashing."
                        )
                    state_schema = json.loads(schema_path.read_text(encoding="utf-8"))
                    if state_schema.get("schema") != "jog-frozen-mesh-dof-state-v1":
                        raise RuntimeError(
                            "Confirmation parent has an unexpected state schema."
                        )
                    if state_schema.get("mesh_sha256") != config["inputs"]["mesh"][
                        "sha256"
                    ]:
                        raise RuntimeError(
                            "Confirmation-parent state references a different mesh."
                        )
                    declared_arrays = state_schema.get("arrays") or {}
                    loaded = {}
                    for name in sorted(required_arrays):
                        relative = arrays[name]
                        declaration = declared_arrays.get(name) or {}
                        if declaration.get("path") != relative:
                            raise RuntimeError(
                                f"Confirmation-parent schema disagrees for {name}."
                            )
                        state_path = (parent_root / relative).resolve()
                        try:
                            state_path.relative_to(parent_root.resolve())
                        except ValueError as error:
                            raise RuntimeError(
                                "Confirmation-parent state escapes its directory."
                            ) from error
                        expected_hash = (parent_manifest.get("output_sha256") or {}).get(
                            relative
                        )
                        if (
                            not state_path.is_file()
                            or not expected_hash
                            or sha256_file(state_path) != expected_hash
                        ):
                            raise RuntimeError(
                                f"Confirmation-parent state failed hashing: {name}"
                            )
                        loaded[name] = np.load(state_path, allow_pickle=False)
                        if list(loaded[name].shape) != declaration.get("shape"):
                            raise RuntimeError(
                                f"Confirmation-parent array shape disagrees for {name}."
                            )
                        if str(loaded[name].dtype) != declaration.get("dtype"):
                            raise RuntimeError(
                                f"Confirmation-parent array dtype disagrees for {name}."
                            )
                    if not all(np.isfinite(values).all() for values in loaded.values()):
                        raise RuntimeError(
                            "Confirmation-parent state arrays contain non-finite values."
                        )
                    parent_input_c = {
                        "path": arrays["C"],
                        "sha256": (parent_manifest.get("output_sha256") or {})[
                            arrays["C"]
                        ],
                    }
                    import firedrake

                    coordinate_field = firedrake.interpolate(
                        object_.mesh.coordinates, object_.V
                    )
                    observed_coordinates = np.asarray(
                        coordinate_field.dat.data_ro[:, :2]
                    )
                    if not np.array_equal(
                        observed_coordinates, loaded["coordinates"]
                    ):
                        raise RuntimeError(
                            "Confirmation-parent coordinates do not match the mesh."
                        )
                    if loaded["C"].shape != initial_C.shape:
                        raise RuntimeError(
                            "Confirmation-parent C shape does not match the space."
                        )
                    if not np.array_equal(
                        loaded["theta"], np.asarray(theta_field.dat.data_ro)
                    ):
                        raise RuntimeError(
                            "Confirmation-parent theta differs from frozen zero."
                        )
                    object_.C.dat.data[:] = loaded["C"]
                    initial_parent_control = object_.C.copy(deepcopy=True)
                elif not np.array_equal(
                    initial_C,
                    np.full_like(
                        initial_C,
                        config["physics"]["initial_log_friction_c"],
                    ),
                ):
                    raise RuntimeError(
                        "First L-curve attempt did not start from C=0."
                    )

                import firedrake.adjoint

                block_iterations = int(inversion["lcurve_block_iterations"])
                minimum_blocks = (
                    1 if is_confirmation else int(inversion["lcurve_min_blocks"])
                )
                maximum_blocks = (
                    1 if is_confirmation else int(inversion["lcurve_max_blocks"])
                )
                if not 0 < minimum_blocks <= maximum_blocks:
                    raise ValueError("Invalid frozen L-curve block limits.")
                if block_iterations <= 0:
                    raise ValueError("L-curve block iterations must be positive.")

                continuation_control = initial_parent_control
                velocity = None
                for local_block_number in range(1, maximum_blocks + 1):
                    block_number = parent_cumulative_blocks + local_block_number
                    firedrake.adjoint.get_working_tape().clear_tape()
                    print(
                        f"JOG_ROL_ATTEMPT_BEGIN {local_block_number}", flush=True
                    )
                    estimator = object_.invert_C(
                        gradient_tolerance=inversion["gradient_tolerance"],
                        step_tolerance=inversion["step_tolerance"],
                        max_iterations=block_iterations,
                        loss_fcn_type="nosigma",
                        regularization_grad_fcn=True,
                        initial_control=continuation_control,
                        return_estimator=True,
                        verbose=True,
                    )
                    _flush_c_streams()
                    print(
                        f"JOG_ROL_ATTEMPT_END {local_block_number}", flush=True
                    )
                    state = classify_algorithm_state(
                        estimator._solver.getAlgorithmState(),
                        gradient_tolerance=inversion["gradient_tolerance"],
                        step_tolerance=inversion["step_tolerance"],
                    )
                    metric_payload = _objective_metrics(object_, reg_c)
                    velocity = metric_payload.pop("velocity")
                    block_metrics = metric_payload
                    block_metrics["rol_objective"] = state["objective"]
                    block_metrics["rol_objective_matches_reassembled"] = bool(
                        np.isclose(
                            block_metrics["objective"],
                            block_metrics["rol_objective"],
                            rtol=1e-8,
                            atol=1e-8,
                        )
                    )
                    state_values = {
                        key: value
                        for key, value in state.items()
                        if key != "termination"
                    }
                    block = {
                        "block": block_number,
                        "attempt": local_block_number,
                        "iteration_limit": block_iterations,
                        "start": (
                            "verified_parent_same_reg_c_control"
                            if is_confirmation
                            else (
                                "independent_C_zero"
                                if local_block_number == 1
                                else "same_reg_c_saved_control"
                            )
                        ),
                        **state_values,
                        "native_termination": state["termination"],
                        "metrics": block_metrics,
                        "relative_misfit_change": None,
                        "relative_roughness_change": None,
                    }
                    if is_confirmation:
                        block["parent_point_manifest_id"] = parent_manifest[
                            "manifest_id"
                        ]
                        block["parent_C_path"] = parent_input_c["path"]
                        block["parent_C_sha256"] = parent_input_c["sha256"]
                    elif blocks:
                        block["parent_C_path"] = blocks[-1]["C_path"]
                        block["parent_C_sha256"] = blocks[-1]["C_sha256"]
                    if blocks:
                        previous_metrics = blocks[-1]["metrics"]
                        block["relative_misfit_change"] = relative_change(
                            previous_metrics["misfit"], block_metrics["misfit"]
                        )
                        block["relative_roughness_change"] = relative_change(
                            previous_metrics["unweighted_roughness"],
                            block_metrics["unweighted_roughness"],
                        )
                    elif is_confirmation:
                        block["relative_misfit_change"] = relative_change(
                            parent_metrics["misfit"], block_metrics["misfit"]
                        )
                        block["relative_roughness_change"] = relative_change(
                            parent_metrics["unweighted_roughness"],
                            block_metrics["unweighted_roughness"],
                        )
                    block_state_path = output_dir / f"block_{block_number:02d}_C.npy"
                    np.save(
                        block_state_path,
                        np.asarray(object_.C.dat.data_ro),
                        allow_pickle=False,
                    )
                    block["C_path"] = block_state_path.name
                    block["C_sha256"] = sha256_file(block_state_path)
                    blocks.append(block)
                    (output_dir / f"block_{block_number:02d}_endpoint.json").write_text(
                        json.dumps(block, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    if is_confirmation:
                        delta_e = float(block["relative_misfit_change"])
                        delta_r = float(block["relative_roughness_change"])
                        tolerance_e = float(
                            inversion["lcurve_relative_misfit_tolerance"]
                        )
                        tolerance_r = float(
                            inversion["lcurve_relative_roughness_tolerance"]
                        )
                        gradient_tolerance = float(
                            inversion["lcurve_gradient_safety_tolerance"]
                        )
                        finite_and_consistent = bool(
                            block_metrics["all_finite"]
                            and block_metrics["weighted_penalty_identity"]
                            and block_metrics[
                                "rol_objective_matches_reassembled"
                            ]
                        )
                        gradient_safe = bool(
                            math.isfinite(float(block["gradient_norm"]))
                            and float(block["gradient_norm"])
                            <= gradient_tolerance
                        )
                        passed = bool(
                            math.isfinite(delta_e)
                            and math.isfinite(delta_r)
                            and (
                                delta_e <= tolerance_e
                                or math.isclose(
                                    delta_e,
                                    tolerance_e,
                                    rel_tol=1e-12,
                                    abs_tol=1e-15,
                                )
                            )
                            and (
                                delta_r <= tolerance_r
                                or math.isclose(
                                    delta_r,
                                    tolerance_r,
                                    rel_tol=1e-12,
                                    abs_tol=1e-15,
                                )
                            )
                            and gradient_safe
                            and finite_and_consistent
                        )
                        stability = {
                            "passed": passed,
                            "acceptance_basis": (
                                "practical_er_stability" if passed else None
                            ),
                            "confirmation": True,
                            "relative_misfit_change": delta_e,
                            "relative_roughness_change": delta_r,
                            "relative_misfit_tolerance": tolerance_e,
                            "relative_roughness_tolerance": tolerance_r,
                            "gradient_safety_tolerance": gradient_tolerance,
                            "terminal_gradient_norm": float(
                                block["gradient_norm"]
                            ),
                            "gradient_safe": gradient_safe,
                            "finite_and_objective_consistent": (
                                finite_and_consistent
                            ),
                        }
                    else:
                        stability = evaluate_er_stability(
                            blocks,
                            minimum_blocks=minimum_blocks,
                            stable_transitions_required=int(
                                inversion["lcurve_stable_transitions"]
                            ),
                            relative_misfit_tolerance=float(
                                inversion["lcurve_relative_misfit_tolerance"]
                            ),
                            relative_roughness_tolerance=float(
                                inversion["lcurve_relative_roughness_tolerance"]
                            ),
                            gradient_safety_tolerance=float(
                                inversion["lcurve_gradient_safety_tolerance"]
                            ),
                        )
                    if stability["passed"]:
                        break
                    continuation_control = object_.C.copy(deepcopy=True)

                native_termination = blocks[-1]["native_termination"]
                metrics = blocks[-1]["metrics"]
                acceptance_basis = (
                    stability.get("acceptance_basis")
                    if stability and stability.get("passed")
                    else None
                )
                fields = _save_state(
                    object_,
                    velocity,
                    output_dir,
                    mesh_sha256=config["inputs"]["mesh"]["sha256"],
                )
                scientifically_valid = (
                    stability is not None
                    and stability["passed"]
                    and metrics["all_finite"]
                    and metrics["misfit"] > 0.0
                    and metrics["unweighted_roughness"] > 0.0
                    and metrics["weighted_penalty_identity"]
                    and metrics["rol_objective_matches_reassembled"]
                    and fields["C_finite"]
                    and fields["theta_finite"]
                    and fields["theta_minimum"] == 0.0
                    and fields["theta_maximum"] == 0.0
                    and fields["velocity_finite"]
                )
                point_status = "valid" if scientifically_valid else "invalid"
            finally:
                del object_
        except Exception:
            exception_text = traceback.format_exc()
            print(exception_text, flush=True)
            point_status = "failed"

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed_attempts = parse_rol_attempts(log_text)
    by_attempt = {row["attempt"]: row for row in parsed_attempts}
    for block in blocks:
        block.update(by_attempt.get(block["attempt"], {}))
        block["native_status_matches_state"] = native_status_matches(
            block["native_termination"], block.get("rol_status_text")
        )
        (output_dir / f"block_{int(block['block']):02d}.json").write_text(
            json.dumps(block, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    solver_log_valid = bool(
        blocks
        and len(parsed_attempts) == len(blocks)
        and all(
            block.get("end_marker_present")
            and block.get("rol_last_iteration") is not None
            and block["native_status_matches_state"]
            for block in blocks
        )
    )
    if blocks:
        native_termination = blocks[-1]["native_termination"]
    if point_status != "failed":
        state_reusable = bool(
            blocks
            and metrics
            and metrics["all_finite"]
            and metrics["weighted_penalty_identity"]
            and metrics["rol_objective_matches_reassembled"]
            and fields
            and fields["C_finite"]
            and fields["theta_finite"]
            and fields["theta_minimum"] == 0.0
            and fields["theta_maximum"] == 0.0
            and fields["velocity_finite"]
            and solver_log_valid
        )
        scientifically_valid = bool(
            stability
            and stability["passed"]
            and state_reusable
            and metrics["misfit"] > 0.0
            and metrics["unweighted_roughness"] > 0.0
        )
        point_status = "valid" if scientifically_valid else "invalid"
        if not scientifically_valid:
            acceptance_basis = None
    if point_status != "valid":
        acceptance_basis = None
    block_rows = []
    for block in blocks:
        block_metrics = block.get("metrics") or {}
        block_rows.append(
            {
                "block": block.get("block"),
                "iteration_limit": block.get("iteration_limit"),
                "start": block.get("start"),
                "native_termination": block.get("native_termination"),
                "rol_last_iteration": block.get("rol_last_iteration"),
                "rol_status_text": block.get("rol_status_text"),
                "gradient_norm": block.get("gradient_norm"),
                "step_norm": block.get("step_norm"),
                "misfit": block_metrics.get("misfit"),
                "unweighted_roughness": block_metrics.get(
                    "unweighted_roughness"
                ),
                "weighted_penalty": block_metrics.get("weighted_penalty"),
                "objective": block_metrics.get("objective"),
                "relative_misfit_change": block.get(
                    "relative_misfit_change"
                ),
                "relative_roughness_change": block.get(
                    "relative_roughness_change"
                ),
                "C_path": block.get("C_path"),
                "C_sha256": block.get("C_sha256"),
                "native_status_matches_state": block.get(
                    "native_status_matches_state"
                ),
            }
        )
    _write_csv(output_dir / "blocks.csv", block_rows)
    output_hashes = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {
            "point_manifest.json",
            "active_process.json",
        }:
            output_hashes[str(path.relative_to(output_dir))] = sha256_file(path)
    parent_reference = None
    if is_confirmation:
        parent_reference = {
            "path": str(parent_point_manifest_path),
            "path_relative_to_child": os.path.relpath(
                parent_point_manifest_path, output_dir
            ),
            "manifest_id": parent_manifest["manifest_id"],
            "manifest_sha256": parent_manifest_sha256,
            "cumulative_block_count": parent_cumulative_blocks,
        }
    manifest = {
        "schema": "jog-production-lcurve-point-v2",
        "run_kind": "confirmation" if is_confirmation else "independent",
        "status": point_status,
        "native_termination": native_termination,
        "acceptance_basis": acceptance_basis,
        "state_reusable": state_reusable,
        "reg_c": float(reg_c),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "repo_root": str(repo_root.resolve()),
        "input_preflight_manifest_id": (
            input_manifest.get("manifest_id") if input_manifest else None
        ),
        "blocks": blocks,
        "cumulative_block_count": parent_cumulative_blocks + len(blocks),
        "stability": stability,
        "parent_point_manifest": parent_reference,
        "confirmation_round": confirmation_round,
        "confirmation": stability if is_confirmation else None,
        "solver_log_crosscheck_passed": solver_log_valid,
        "block_policy": (
            "One verified same-reg_C parent-C confirmation block; ROL "
            "trust-region internals are not serialized."
            if is_confirmation
            else (
                "Independent exact C=0 start followed only by same-reg_C "
                "saved-C warm restarts in frozen 50-iteration blocks; ROL "
                "trust-region internals are not serialized between blocks."
            )
        ),
        "metrics": metrics,
        "fields": fields,
        "exception": exception_text,
        "source_sha256": preflight.source_identity(),
        "environment": preflight.environment(),
        "output_sha256": output_hashes,
    }
    manifest["manifest_id"] = canonical_identifier(manifest)
    _atomic_json(output_dir / "point_manifest.json", manifest)
    active_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "status": point_status,
                "native_termination": native_termination,
                "acceptance_basis": acceptance_basis,
                "reg_c": reg_c,
                "manifest_id": manifest["manifest_id"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return manifest


def run_lcurve_confirmation(
    *,
    config: dict,
    config_path: Path,
    repo_root: Path,
    output_dir: Path,
    reg_c: float,
    parent_point_manifest_path: Path,
    confirmation_round: int,
    preflight_class,
) -> dict:
    """Run one immutable, hash-verified same-reg_C confirmation block."""
    return run_lcurve_point(
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        output_dir=output_dir,
        reg_c=reg_c,
        preflight_class=preflight_class,
        parent_point_manifest_path=parent_point_manifest_path,
        confirmation_round=confirmation_round,
    )


def run_forward_smoke(
    *,
    config: dict,
    config_path: Path,
    repo_root: Path,
    output_dir: Path,
    preflight_class,
) -> dict:
    """Run the frozen whole-sector forward model at C=theta=0 without optimization."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    log_path = output_dir / "logs" / "forward_smoke.log"
    started = datetime.now(timezone.utc)
    input_manifest = None
    metrics = None
    fields = None
    checks = {}
    exception_text = None
    status = "failed"
    preflight = preflight_class(
        config, repo_root.resolve(), output_dir / "input_preflight"
    )

    with process_log(log_path):
        object_ = None
        try:
            input_manifest = preflight.run(level="static", config_path=config_path)
            if input_manifest["status"] != "pass":
                raise RuntimeError("Forward-smoke immutable input preflight failed.")
            object_ = preflight.build_invert(reg_c=1.0)
            expected = config["expected_counts"]
            initial_value = config["physics"]["initial_log_friction_c"]
            theta_value = config["physics"]["initial_log_fluidity_theta"]
            theta_field = getattr(object_, "\u03b8")
            checks = {
                "scalar_dof_count": object_.Q.dim() == expected["cg2_scalar_dofs"],
                "observation_count": object_.N == expected["selected_observations"],
                "C_exactly_frozen_initial_value": bool(
                    np.array_equal(
                        object_.C.dat.data_ro,
                        np.full_like(object_.C.dat.data_ro, initial_value),
                    )
                ),
                "theta_exactly_frozen_initial_value": bool(
                    np.array_equal(
                        theta_field.dat.data_ro,
                        np.full_like(theta_field.dat.data_ro, theta_value),
                    )
                ),
            }
            if not all(checks.values()):
                raise RuntimeError("Forward-smoke population or initial-state check failed.")

            import firedrake.adjoint

            firedrake.adjoint.get_working_tape().clear_tape()
            metric_payload = _objective_metrics(object_, 1.0)
            velocity = metric_payload.pop("velocity")
            metrics = metric_payload
            checks.update(
                {
                    "finite_objective_metrics": metrics["all_finite"],
                    "positive_observation_misfit": metrics["misfit"] > 0.0,
                    "zero_unweighted_roughness": bool(
                        np.isclose(
                            metrics["unweighted_roughness"], 0.0, rtol=0.0, atol=1e-12
                        )
                    ),
                    "zero_weighted_penalty": bool(
                        np.isclose(
                            metrics["weighted_penalty"], 0.0, rtol=0.0, atol=1e-12
                        )
                    ),
                    "objective_equals_misfit": bool(
                        np.isclose(
                            metrics["objective"], metrics["misfit"], rtol=1e-12, atol=1e-12
                        )
                    ),
                    "weighted_penalty_identity": metrics[
                        "weighted_penalty_identity"
                    ],
                }
            )
            fields = _save_state(
                object_,
                velocity,
                output_dir,
                mesh_sha256=config["inputs"]["mesh"]["sha256"],
            )
            checks.update(
                {
                    "C_finite": fields["C_finite"],
                    "velocity_finite": fields["velocity_finite"],
                }
            )
            status = "pass" if all(checks.values()) else "fail"
        except Exception:
            exception_text = traceback.format_exc()
            print(exception_text, flush=True)
            status = "failed"
        finally:
            if object_ is not None:
                del object_

    (output_dir / "checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_hashes = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "forward_smoke_manifest.json":
            output_hashes[str(path.relative_to(output_dir))] = sha256_file(path)
    manifest = {
        "schema": "jog-production-forward-smoke-v1",
        "status": status,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "repo_root": str(repo_root.resolve()),
        "input_preflight_manifest_id": (
            input_manifest.get("manifest_id") if input_manifest else None
        ),
        "checks": checks,
        "metrics": metrics,
        "fields": fields,
        "exception": exception_text,
        "source_sha256": preflight.source_identity(),
        "environment": preflight.environment(),
        "output_sha256": output_hashes,
    }
    manifest["manifest_id"] = canonical_identifier(manifest)
    (output_dir / "forward_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "manifest_id": manifest["manifest_id"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return manifest
