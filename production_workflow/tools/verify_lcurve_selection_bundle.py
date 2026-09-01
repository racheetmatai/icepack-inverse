#!/usr/bin/env python3
"""Independently verify a portable L-curve selection-only evidence bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from lcurve_orchestrator import (  # noqa: E402
    _monotonic_diagnostics,
    _stable_environment,
    validate_point_manifest,
)
from lcurve_runtime import canonical_identifier, sha256_file  # noqa: E402
from lcurve_selection import (  # noqa: E402
    corner_neighborhood,
    curvature_ambiguity,
    select_corner,
)


BUNDLE_SCHEMA = "jog-lcurve-selection-bundle-v1"
DECISION_SCHEMA = "jog-lcurve-selection-decision-v1"
SELECTED_REGULARIZATION_SCHEMA = "jog-selected-regularization-v1"
CONTRACT_SCHEMA = "jog-production-lcurve-contract-v2"
SAFE_ENVIRONMENT_DIFFERENCES = {
    "docker_image_hint",
    "docker_image_id",
    "git_status",
}
STABLE_ENVIRONMENT_KEYS = {
    "docker_image_hint",
    "docker_image_id",
    "executable",
    "git_head",
    "git_status",
    "hostname",
    "platform",
    "python",
    "versions",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(root: Path, relative_text: str, *, label: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{label} is not a portable relative path: {relative_text}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} escapes the bundle: {relative_text}") from error
    return path


def _verify_identifier(path: Path, payload: dict) -> None:
    if canonical_identifier(payload) != payload.get("manifest_id"):
        raise RuntimeError(f"Invalid canonical manifest identifier: {path}")


def _verify_complete_inventory(root: Path, declared: dict) -> int:
    expected = set(declared)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "selection_bundle.json"
    }
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            f"Bundle file inventory differs (missing={missing}, unexpected={unexpected})."
        )
    for relative, expected_hash in declared.items():
        path = _inside(root, relative, label="declared bundle output")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Missing or hash-mismatched bundle output: {path}")
    return len(observed)


def _selector_hash(source_hashes: dict) -> str:
    matches = [
        digest
        for source, digest in source_hashes.items()
        if Path(source).name == "lcurve_selection.py"
    ]
    if len(matches) != 1:
        raise RuntimeError("Source identity does not freeze one selector file.")
    return matches[0]


def _point_for_selection(manifest: dict) -> dict:
    metrics = manifest.get("metrics") or {}
    return {
        "reg_c": float(manifest["reg_c"]),
        "status": manifest.get("status"),
        "termination": manifest.get("native_termination"),
        "native_termination": manifest.get("native_termination"),
        "acceptance_basis": manifest.get("acceptance_basis"),
        "misfit": metrics.get("misfit"),
        "unweighted_roughness": metrics.get("unweighted_roughness"),
        "weighted_penalty": metrics.get("weighted_penalty"),
        "objective": metrics.get("objective"),
    }


def _contract_for_environment(contract: dict, stable_environment: dict) -> dict:
    return {
        "config_sha256": contract["config_sha256"],
        "source_sha256": contract["source_sha256"],
        "stable_environment": stable_environment,
        "expected_input_sha256": contract["expected_input_sha256"],
        "expected_design_sha256": contract["expected_design_sha256"],
        "protocol": contract["protocol"],
    }


def _verify_environment_policy(bundle: dict, manifests: dict[str, dict]) -> None:
    policy = bundle.get("environment_policy") or {}
    allowed = policy.get("allowed_differences") or {}
    if not isinstance(allowed, dict) or set(allowed) - SAFE_ENVIRONMENT_DIFFERENCES:
        raise RuntimeError("Bundle waives unsupported environment fields.")
    if any(not isinstance(value, str) or not value.strip() for value in allowed.values()):
        raise RuntimeError("An environment waiver lacks its rationale.")
    groups = policy.get("groups") or {}
    if not groups:
        raise RuntimeError("Bundle lacks composite environment groups.")

    observed_members = set()
    for group_id, record in groups.items():
        stable = record.get("stable_environment") or {}
        if canonical_identifier(stable) != group_id:
            raise RuntimeError("Environment group identifier is invalid.")
        members = record.get("member_manifest_ids") or []
        if len(members) != len(set(members)):
            raise RuntimeError("Environment group repeats a manifest.")
        for manifest_id in members:
            manifest = manifests.get(manifest_id)
            if manifest is None or _stable_environment(
                manifest.get("environment", {})
            ) != stable:
                raise RuntimeError("Environment group membership is inconsistent.")
            if manifest_id in observed_members:
                raise RuntimeError("A manifest belongs to multiple environment groups.")
            observed_members.add(manifest_id)
    if observed_members != set(manifests):
        raise RuntimeError("Environment groups do not cover every accepted manifest.")

    values = {
        key: [record["stable_environment"].get(key) for record in groups.values()]
        for key in STABLE_ENVIRONMENT_KEYS
    }
    observed_differences = sorted(
        key
        for key, candidates in values.items()
        if any(value != candidates[0] for value in candidates[1:])
    )
    if observed_differences != policy.get("observed_differing_keys"):
        raise RuntimeError("Recorded environment differences are incomplete.")
    if set(observed_differences) - set(allowed):
        raise RuntimeError("A non-waived environment field differs.")
    required_equal = sorted(STABLE_ENVIRONMENT_KEYS - set(allowed))
    if required_equal != policy.get("required_equal_keys"):
        raise RuntimeError("Required-equal environment fields are inconsistent.")

    known_ids = sorted(
        {
            value
            for value in values["docker_image_id"]
            if value not in (None, "", "unrecorded")
        }
    )
    known_hints = sorted(
        {
            value
            for value in values["docker_image_hint"]
            if value not in (None, "", "unrecorded")
        }
    )
    if len(known_ids) > 1 or len(known_hints) > 1:
        raise RuntimeError("Composite evidence contains conflicting known images.")
    if known_ids != policy.get("known_docker_image_ids") or known_hints != policy.get(
        "known_docker_image_hints"
    ):
        raise RuntimeError("Known Docker image inventory is inconsistent.")


def _verify_reference(root: Path, reference: dict) -> tuple[Path, dict]:
    path = _inside(root, reference["manifest_path"], label="point manifest")
    if not path.is_file() or sha256_file(path) != reference.get("manifest_sha256"):
        raise RuntimeError("Point manifest reference failed hashing.")
    manifest = _read_json(path)
    _verify_identifier(path, manifest)
    for field in ("manifest_id", "run_kind", "status"):
        if manifest.get(field) != reference.get(field):
            raise RuntimeError(f"Point reference disagrees for {field}.")
    if float(manifest.get("reg_c")) != float(reference.get("reg_c")):
        raise RuntimeError("Point reference disagrees for reg_C.")
    return path, manifest


def verify_bundle(root: Path) -> dict:
    """Verify hashes, point lineages, composite provenance, and selection."""
    root = root.resolve()
    bundle_path = root / "selection_bundle.json"
    bundle = _read_json(bundle_path)
    _verify_identifier(bundle_path, bundle)
    if bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("status") != "complete":
        raise RuntimeError("Unexpected or incomplete selection bundle.")
    if not (
        bundle.get("artifact_kind") == "selection_decision_only"
        and bundle.get("selection_decision_only") is True
        and bundle.get("definitive_inversion_created") is False
    ):
        raise RuntimeError("Bundle improperly claims a definitive inversion.")
    if (root / "definitive_inversion.json").exists():
        raise RuntimeError("A selection-only bundle contains a definitive inversion.")
    output_count = _verify_complete_inventory(root, bundle.get("output_sha256") or {})

    plan = _read_json(root / "assembly_plan.json")
    deviation = bundle.get("manual_protocol_deviation") or {}
    if deviation != plan.get("manual_protocol_deviation") or deviation.get(
        "acknowledged"
    ) is not True:
        raise RuntimeError("Manual refined-window deviation is not frozen correctly.")
    if not bundle.get("exclusions"):
        raise RuntimeError("Selection bundle lacks explicit exclusions.")
    decision_record_path = root / "LCURVE_REFINED_WINDOW_FREEZE_RECORD.md"
    if not decision_record_path.is_file():
        raise RuntimeError("Selection bundle lacks its author decision record.")
    formal_state = _read_json(root / "formal_run_state.json")
    planned_formal_state = plan.get("formal_study_state") or {}
    if (
        formal_state.get("status") != planned_formal_state.get("recorded_status")
        or formal_state.get("phase") != planned_formal_state.get("recorded_phase")
        or (root / "definitive_inversion.json").exists()
    ):
        raise RuntimeError("Formal incomplete-study evidence is inconsistent.")

    for exclusion in bundle.get("exclusions") or []:
        if exclusion.get("disposition") != "excluded_from_selection":
            raise RuntimeError("An exclusion lacks the frozen disposition.")
        artifact_text = exclusion.get("artifact_bundle_path")
        if artifact_text:
            artifact = _inside(root, artifact_text, label="excluded evidence")
            if (
                not artifact.is_file()
                or sha256_file(artifact) != exclusion.get("artifact_sha256")
            ):
                raise RuntimeError("Excluded evidence failed hashing.")

    contract_path = root / "source_contract.json"
    contract = _read_json(contract_path)
    _verify_identifier(contract_path, contract)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise RuntimeError("Unexpected source-contract schema.")
    pinned_contract = plan.get("formal_contract") or {}
    if (
        pinned_contract.get("manifest_id") != contract.get("manifest_id")
        or pinned_contract.get("manifest_sha256") != sha256_file(contract_path)
        or pinned_contract.get("config_sha256") != contract.get("config_sha256")
    ):
        raise RuntimeError("Source contract differs from the assembly recipe.")
    provenance = bundle.get("source_provenance") or {}
    if (
        provenance.get("source_contract_manifest_id") != contract["manifest_id"]
        or provenance.get("source_contract_sha256") != sha256_file(contract_path)
        or provenance.get("config_sha256") != contract["config_sha256"]
        or provenance.get("source_sha256") != contract["source_sha256"]
        or provenance.get("expected_input_sha256")
        != contract["expected_input_sha256"]
        or provenance.get("expected_design_sha256")
        != contract["expected_design_sha256"]
    ):
        raise RuntimeError("Bundle scientific provenance differs from its contract.")
    selector_sha256 = _selector_hash(contract["source_sha256"])
    if (
        provenance.get("selector_source_sha256") != selector_sha256
        or sha256_file(WORKFLOW_ROOT / "lcurve_selection.py") != selector_sha256
    ):
        raise RuntimeError("Verifier is not using the frozen selector source.")

    curve_references = bundle.get("curve_points") or []
    if len(curve_references) != 5 or bundle.get("curve_point_count") != 5:
        raise RuntimeError("Selection bundle does not contain five curve lineages.")
    classifications = [record.get("classification") for record in curve_references]
    if classifications.count("formal_endpoint") != 2 or classifications.count(
        "confirmed_refinement"
    ) != 3:
        raise RuntimeError("Curve lineage classifications are inconsistent.")

    loaded = []
    unique_manifests: dict[str, dict] = {}
    for record in curve_references:
        independent_path, independent = _verify_reference(root, record["independent"])
        effective_path, effective = _verify_reference(root, record["effective"])
        if float(record["reg_c"]) != float(independent["reg_c"]) or float(
            record["reg_c"]
        ) != float(effective["reg_c"]):
            raise RuntimeError("A curve lineage changes reg_C.")
        if independent.get("run_kind") != "independent":
            raise RuntimeError("A curve lineage does not start independently.")
        if record["classification"] == "formal_endpoint":
            if independent["manifest_id"] != effective["manifest_id"]:
                raise RuntimeError("A formal endpoint was replaced by another run.")
        else:
            parent = effective.get("parent_point_manifest") or {}
            if (
                effective.get("run_kind") != "confirmation"
                or int(effective.get("confirmation_round", 0)) != 1
                or parent.get("manifest_id") != independent["manifest_id"]
                or (effective.get("confirmation") or {}).get("passed") is not True
            ):
                raise RuntimeError("A refinement lacks its passing round-one lineage.")
        reg_key = format(float(record["reg_c"]), ".12g")
        pinned_independent = (plan.get("accepted_independent_by_reg_c") or {}).get(
            reg_key
        )
        pinned_effective = (plan.get("effective_by_reg_c") or {}).get(reg_key)
        if (
            not pinned_independent
            or not pinned_effective
            or independent.get("manifest_id")
            != pinned_independent.get("manifest_id")
            or sha256_file(independent_path)
            != pinned_independent.get("manifest_sha256")
            or effective.get("manifest_id") != pinned_effective.get("manifest_id")
            or sha256_file(effective_path) != pinned_effective.get("manifest_sha256")
        ):
            raise RuntimeError(f"Pinned lineage mismatch at reg_C={reg_key}.")
        loaded.append((record, independent_path, independent, effective_path, effective))
        unique_manifests[independent["manifest_id"]] = independent
        unique_manifests[effective["manifest_id"]] = effective

    _verify_environment_policy(bundle, unique_manifests)
    groups = bundle["environment_policy"]["groups"]
    for record, independent_path, independent, effective_path, effective in loaded:
        for reference, path, manifest in (
            (record["independent"], independent_path, independent),
            (record["effective"], effective_path, effective),
        ):
            group = groups.get(reference.get("environment_group_id"))
            if group is None:
                raise RuntimeError("Point reference names an absent environment group.")
            validate_point_manifest(
                manifest,
                point_dir=path.parent,
                reg_c=float(manifest["reg_c"]),
                contract=_contract_for_environment(
                    contract, group["stable_environment"]
                ),
                expected_run_kind=manifest["run_kind"],
                expected_confirmation_round=(
                    int(manifest["confirmation_round"])
                    if manifest["run_kind"] == "confirmation"
                    else None
                ),
                expected_parent_manifest_id=(
                    (manifest.get("parent_point_manifest") or {}).get("manifest_id")
                    if manifest["run_kind"] == "confirmation"
                    else None
                ),
                study_root=root / "evidence",
            )

    before_points = [_point_for_selection(row[2]) for row in loaded]
    after_points = [_point_for_selection(row[4]) for row in loaded]
    before_selection = select_corner(before_points)
    after_selection = select_corner(after_points)
    before_neighborhood = corner_neighborhood(before_points)
    after_neighborhood = corner_neighborhood(after_points)
    ratio = float(contract["protocol"]["curvature_ambiguity_ratio"])
    before_ambiguity = curvature_ambiguity(before_selection, ratio)
    after_ambiguity = curvature_ambiguity(after_selection, ratio)

    decision_path = root / "selection_final.json"
    decision = _read_json(decision_path)
    _verify_identifier(decision_path, decision)
    if decision.get("schema") != DECISION_SCHEMA:
        raise RuntimeError("Unexpected selection-decision schema.")
    if decision.get("selector_source_sha256") != selector_sha256:
        raise RuntimeError("Selection decision records a different selector.")
    if decision.get("manifest_id") != bundle.get("selection_final_manifest_id"):
        raise RuntimeError("Bundle references a different selection decision.")
    expected_pre = {
        "selection": before_selection,
        "ambiguity": before_ambiguity,
        "neighborhood": before_neighborhood,
    }
    expected_post = {
        "selection": after_selection,
        "ambiguity": after_ambiguity,
        "neighborhood": after_neighborhood,
        "monotonic_diagnostics": _monotonic_diagnostics(after_points),
    }
    if decision.get("preconfirmation") != expected_pre or decision.get(
        "postconfirmation"
    ) != expected_post:
        raise RuntimeError("Stored selection evidence is not reproducible.")
    selected = float(after_selection["selected"]["reg_c"])
    if not (
        float(before_selection["selected"]["reg_c"]) == selected
        and before_neighborhood == after_neighborhood
        and selected == float(decision["selected_reg_c"])
        and selected == float(bundle["selected_reg_c"])
    ):
        raise RuntimeError("The selected corner or confirmation triple changed.")
    confirmation = decision.get("confirmation_resolution") or {}
    if not (
        confirmation.get("all_passed") is True
        and confirmation.get("candidate_unchanged") is True
        and confirmation.get("triple_unchanged") is True
        and sorted(float(value) for value in confirmation["confirmation_reg_c"])
        == sorted(float(value) for value in after_neighborhood["confirmation_reg_c"])
    ):
        raise RuntimeError("Selection confirmation is unresolved.")

    confirmation_references = bundle.get("confirmation_manifests") or []
    expected_confirmation_ids = [
        row[4]["manifest_id"]
        for row in loaded
        if row[0]["classification"] == "confirmed_refinement"
    ]
    if (
        bundle.get("confirmation_manifest_count") != 3
        or [row.get("manifest_id") for row in confirmation_references]
        != expected_confirmation_ids
        or confirmation.get("confirmation_manifest_ids")
        != expected_confirmation_ids
    ):
        raise RuntimeError("Confirmation manifest inventory is inconsistent.")

    selected_path = root / "selected_regularization.json"
    selected_artifact = _read_json(selected_path)
    _verify_identifier(selected_path, selected_artifact)
    if not (
        selected_artifact.get("schema") == SELECTED_REGULARIZATION_SCHEMA
        and selected_artifact.get("artifact_kind") == "selection_decision_only"
        and selected_artifact.get("status")
        == "frozen_for_fresh_definitive_inversion"
        and float(selected_artifact.get("reg_c")) == selected
        and selected_artifact.get("selection_final_manifest_id")
        == decision.get("manifest_id")
        and selected_artifact.get("definitive_inversion_created") is False
        and selected_artifact.get("manifest_id")
        == bundle.get("selected_regularization_manifest_id")
    ):
        raise RuntimeError("Selected-regularization artifact is inconsistent.")

    rows = list(csv.DictReader((root / "lcurve_points.csv").open(encoding="utf-8")))
    if len(rows) != 5 or [float(row["reg_c"]) for row in rows] != sorted(
        float(record["reg_c"]) for record in curve_references
    ):
        raise RuntimeError("L-curve point table is inconsistent.")

    return {
        "status": "verified",
        "bundle_manifest_id": bundle["manifest_id"],
        "selection_final_manifest_id": decision["manifest_id"],
        "selected_regularization_manifest_id": selected_artifact["manifest_id"],
        "selected_reg_c": selected,
        "curve_point_count": 5,
        "confirmation_manifest_count": 3,
        "environment_group_count": len(groups),
        "verified_file_count": output_count,
        "artifact_kind": "selection_decision_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle_directory), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
