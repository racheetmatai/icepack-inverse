"""Verify adoption of an existing confirmed L-curve state as definitive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from verify_lcurve_selection_bundle import verify_bundle
except ModuleNotFoundError:  # Imported as production_workflow.tools.*
    from .verify_lcurve_selection_bundle import verify_bundle


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_id", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def _contained(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    path.relative_to(base.resolve())
    return path


def verify_adoption(record_path: Path) -> dict:
    record_path = record_path.resolve()
    record = _read_json(record_path)
    if record.get("schema") != "jog-definitive-inversion-adoption-v1":
        raise RuntimeError("Unexpected definitive-adoption schema.")
    if record.get("manifest_id") != _identifier(record):
        raise RuntimeError("Definitive-adoption identifier is invalid.")
    if not (
        record.get("status") == "complete"
        and record.get("decision") == "adopt_existing_confirmed_state"
        and record.get("new_inversion_run_required") is False
    ):
        raise RuntimeError("Definitive-adoption decision is not complete.")

    bundle = _contained(
        record_path.parent, record["selection_bundle"]["relative_path"]
    )
    verified_bundle = verify_bundle(bundle)
    bundle_manifest = bundle / "selection_bundle.json"
    if not (
        verified_bundle["bundle_manifest_id"]
        == record["selection_bundle"]["manifest_id"]
        and _sha256(bundle_manifest)
        == record["selection_bundle"]["manifest_file_sha256"]
        and float(verified_bundle["selected_reg_c"]) == float(record["reg_c"])
    ):
        raise RuntimeError("Selection bundle does not match the adoption record.")

    point_path = _contained(bundle, record["point"]["manifest_relative_path"])
    point = _read_json(point_path)
    if not (
        _sha256(point_path) == record["point"]["manifest_sha256"]
        and point.get("manifest_id") == record["point"]["manifest_id"]
        and point.get("manifest_id") == record["point"]["selection_effective_id"]
        and point.get("status") == "valid"
        and point.get("state_reusable") is True
        and point.get("run_kind") == "confirmation"
        and float(point.get("reg_c")) == float(record["reg_c"])
        and int(point.get("cumulative_block_count")) == 6
        and point.get("solver_log_crosscheck_passed") is True
    ):
        raise RuntimeError("Selected point is not a reusable confirmed inversion.")

    stability = point.get("stability") or {}
    metrics = point.get("metrics") or {}
    fields = point.get("fields") or {}
    if not (
        stability.get("passed") is True
        and stability.get("finite_and_objective_consistent") is True
        and stability.get("gradient_safe") is True
        and float(stability["terminal_gradient_norm"]) <= 1.0e-3
        and metrics.get("all_finite") is True
        and metrics.get("rol_objective_matches_reassembled") is True
        and metrics.get("weighted_penalty_identity") is True
        and fields.get("C_finite") is True
        and fields.get("theta_finite") is True
        and fields.get("velocity_finite") is True
    ):
        raise RuntimeError("Selected inversion fails the definitive-state gates.")

    preflight_path = point_path.parent / "input_preflight" / "preflight_manifest.json"
    preflight = _read_json(preflight_path)
    if not (
        _sha256(preflight_path) == record["preflight"]["manifest_sha256"]
        and preflight.get("manifest_id") == record["preflight"]["manifest_id"]
        and preflight.get("status") == "pass"
        and int(preflight.get("hard_failure_count")) == 0
        and preflight.get("config_sha256") == record["config_sha256"]
        and point.get("input_preflight_manifest_id") == preflight.get("manifest_id")
    ):
        raise RuntimeError("Selected inversion preflight is not definitive-ready.")

    schema_path = point_path.parent / fields["state_schema"]
    schema = _read_json(schema_path)
    if not (
        _sha256(schema_path) == record["state"]["schema_sha256"]
        and schema.get("schema") == "jog-frozen-mesh-dof-state-v1"
        and schema.get("mesh_sha256") == record["state"]["mesh_sha256"]
    ):
        raise RuntimeError("Definitive state schema or mesh does not match.")

    for name, expected in record["state"]["arrays"].items():
        spec = schema["arrays"][name]
        array_path = point_path.parent / spec["path"]
        values = np.load(array_path, allow_pickle=False)
        if not (
            _sha256(array_path) == expected["sha256"]
            and point["output_sha256"][spec["path"]] == expected["sha256"]
            and list(values.shape) == expected["shape"] == spec["shape"]
            and str(values.dtype) == expected["dtype"] == spec["dtype"]
            and np.isfinite(values).all()
        ):
            raise RuntimeError(f"Definitive state array failed verification: {name}")

    return {
        "status": "verified",
        "adoption_manifest_id": record["manifest_id"],
        "selected_point_manifest_id": point["manifest_id"],
        "reg_c": float(record["reg_c"]),
        "cumulative_block_count": int(point["cumulative_block_count"]),
        "terminal_gradient_norm": float(stability["terminal_gradient_norm"]),
        "state_dof_count": int(schema["arrays"]["C"]["shape"][0]),
        "new_inversion_run_required": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adoption_record", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_adoption(args.adoption_record), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
