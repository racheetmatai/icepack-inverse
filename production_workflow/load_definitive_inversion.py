"""Verified loader for the selected whole-sector L-curve inversion state."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from lcurve_orchestrator import (
    _read_json,
    validate_point_manifest,
    verify_completed_study,
)
from lcurve_runtime import canonical_identifier, sha256_file


def _verified_array(point_dir: Path, relative: str, expected: dict) -> np.ndarray:
    path = point_dir / relative
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != expected["shape"]:
        raise RuntimeError(f"State-array shape mismatch for {path}")
    if str(values.dtype) != expected["dtype"]:
        raise RuntimeError(f"State-array dtype mismatch for {path}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"State array is nonfinite: {path}")
    return values


def load_definitive_state(
    *, study_dir: Path, object_, contract: dict | None = None
) -> dict:
    """Verify and assign selected C/theta/velocity to a matching Invert object."""
    study_dir = study_dir.resolve()
    if contract is None:
        contract = _read_json(study_dir / "run_contract.json")
    if canonical_identifier(contract) != contract.get("manifest_id"):
        raise RuntimeError("L-curve run contract identifier is invalid.")
    for source, expected_hash in contract.get("source_sha256", {}).items():
        source_path = Path(source)
        if not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise RuntimeError(
                f"Current scientific source does not match the run contract: {source}"
            )
    verify_completed_study(study_dir, contract)
    definitive_path = study_dir / "definitive_inversion.json"
    definitive = _read_json(definitive_path)
    if canonical_identifier(definitive) != definitive.get("manifest_id"):
        raise RuntimeError("Definitive inversion reference identifier is invalid.")
    point_path = study_dir / definitive["point_manifest_path"]
    if sha256_file(point_path) != definitive.get("point_manifest_sha256"):
        raise RuntimeError("Definitive point manifest hash does not match.")
    point = _read_json(point_path)
    if point.get("manifest_id") != definitive.get("point_manifest_id"):
        raise RuntimeError("Definitive reference points to a different point.")
    validate_point_manifest(
        point,
        point_dir=point_path.parent,
        reg_c=float(definitive["reg_c"]),
        contract=contract,
    )
    schema_path = point_path.parent / point["fields"]["state_schema"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("schema") != "jog-frozen-mesh-dof-state-v1":
        raise RuntimeError("Unexpected definitive-state schema.")
    if schema.get("mesh_sha256") != contract["expected_input_sha256"]["mesh"]:
        raise RuntimeError("Definitive state belongs to a different frozen mesh.")
    if int(schema["scalar_space"]["degree"]) != int(object_.degree):
        raise RuntimeError("Definitive state uses a different element degree.")
    arrays = {
        name: _verified_array(point_path.parent, spec["path"], spec)
        for name, spec in schema["arrays"].items()
    }
    import firedrake

    coordinate_field = firedrake.interpolate(object_.mesh.coordinates, object_.V)
    current_coordinates = np.asarray(coordinate_field.dat.data_ro[:, :2])
    if not np.array_equal(arrays["coordinates"], current_coordinates):
        raise RuntimeError("Definitive state DOF coordinates do not match this mesh.")
    theta_field = getattr(object_, "\u03b8")
    if arrays["C"].shape != object_.C.dat.data.shape:
        raise RuntimeError("Definitive C shape does not match the scalar space.")
    if arrays["theta"].shape != theta_field.dat.data.shape:
        raise RuntimeError("Definitive theta shape does not match the scalar space.")
    velocity = firedrake.Function(object_.V, name="velocity")
    if arrays["velocity"].shape != velocity.dat.data.shape:
        raise RuntimeError("Definitive velocity shape does not match the vector space.")
    object_.C.dat.data[:] = arrays["C"]
    theta_field.dat.data[:] = arrays["theta"]
    velocity.dat.data[:] = arrays["velocity"]
    return {
        "reg_c": float(definitive["reg_c"]),
        "point_manifest_id": point["manifest_id"],
        "C": object_.C,
        "theta": theta_field,
        "velocity": velocity,
    }


def load_adopted_definitive_state(*, adoption_record: Path, object_) -> dict:
    """Verify and load an adopted confirmed L-curve endpoint without rerunning it."""
    adoption_record = adoption_record.resolve()
    tools_dir = Path(__file__).resolve().parent / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from verify_definitive_inversion_adoption import verify_adoption

    verified = verify_adoption(adoption_record)
    record = _read_json(adoption_record)
    bundle_dir = (
        adoption_record.parent / record["selection_bundle"]["relative_path"]
    ).resolve()
    point_path = (bundle_dir / record["point"]["manifest_relative_path"]).resolve()
    point = _read_json(point_path)
    schema = _read_json(point_path.parent / point["fields"]["state_schema"])
    arrays = {
        name: _verified_array(point_path.parent, spec["path"], spec)
        for name, spec in schema["arrays"].items()
    }

    import firedrake

    coordinate_field = firedrake.interpolate(object_.mesh.coordinates, object_.V)
    current_coordinates = np.asarray(coordinate_field.dat.data_ro[:, :2])
    if not np.array_equal(arrays["coordinates"], current_coordinates):
        raise RuntimeError("Adopted definitive-state coordinates do not match this mesh.")
    theta_field = getattr(object_, "θ")
    velocity = firedrake.Function(object_.V, name="velocity")
    if arrays["C"].shape != object_.C.dat.data.shape:
        raise RuntimeError("Adopted C shape does not match the scalar space.")
    if arrays["theta"].shape != theta_field.dat.data.shape:
        raise RuntimeError("Adopted theta shape does not match the scalar space.")
    if arrays["velocity"].shape != velocity.dat.data.shape:
        raise RuntimeError("Adopted velocity shape does not match the vector space.")
    object_.C.dat.data[:] = arrays["C"]
    theta_field.dat.data[:] = arrays["theta"]
    velocity.dat.data[:] = arrays["velocity"]
    return {
        "reg_c": verified["reg_c"],
        "adoption_manifest_id": verified["adoption_manifest_id"],
        "point_manifest_id": verified["selected_point_manifest_id"],
        "C": object_.C,
        "theta": theta_field,
        "velocity": velocity,
    }
