#!/usr/bin/env python3
"""Assemble a portable, provenance-closed L-curve selection decision bundle.

This tool intentionally does not create a completed production L-curve study or
a definitive-inversion reference.  It packages an explicitly acknowledged
manual refinement window, verifies every accepted endpoint with the production
point validator, reruns the frozen selector, and records only the regularization
selection decision.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from lcurve_orchestrator import (  # noqa: E402
    _monotonic_diagnostics,
    _plot_lcurve,
    _stable_environment,
    _write_curvature,
    validate_point_manifest,
)
from lcurve_runtime import canonical_identifier, sha256_file  # noqa: E402
from lcurve_selection import (  # noqa: E402
    corner_neighborhood,
    curvature_ambiguity,
    select_corner,
)


PLAN_SCHEMA = "jog-lcurve-selection-assembly-plan-v1"
BUNDLE_SCHEMA = "jog-lcurve-selection-bundle-v1"
DECISION_SCHEMA = "jog-lcurve-selection-decision-v1"
SELECTED_REGULARIZATION_SCHEMA = "jog-selected-regularization-v1"
CONTRACT_SCHEMA = "jog-production-lcurve-contract-v2"

STABLE_ENVIRONMENT_KEYS = (
    "docker_image_hint",
    "docker_image_id",
    "executable",
    "git_head",
    "git_status",
    "hostname",
    "platform",
    "python",
    "versions",
)

# These differences change provenance metadata but not the frozen scientific
# sources or numerical runtime.  Known conflicting image IDs are still rejected.
SAFE_ENVIRONMENT_DIFFERENCES = {
    "docker_image_hint",
    "docker_image_id",
    "git_status",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _relative_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} must be a non-escaping relative path: {value}")
    return path


def _within(root: Path, relative: Path, *, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its declared root: {relative}") from error
    return path


def _source_relative(value: str, *, source_root: Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(source_root.resolve())
        except ValueError as error:
            raise ValueError(f"{label} is outside the source root: {value}") from error
    return _relative_path(value, label=label)


def _selector_hash(source_hashes: dict) -> str:
    matches = [
        digest
        for source, digest in source_hashes.items()
        if Path(source).name == "lcurve_selection.py"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "The frozen contract must identify exactly one lcurve_selection.py."
        )
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


def _environment_inventory(manifests: list[dict], policy: dict) -> dict:
    allowed = policy.get("allowed_differences")
    if not isinstance(allowed, dict) or not allowed:
        raise ValueError(
            "environment_policy.allowed_differences must map fields to rationales."
        )
    allowed_keys = set(allowed)
    unsupported = allowed_keys - SAFE_ENVIRONMENT_DIFFERENCES
    if unsupported:
        raise ValueError(
            "Environment differences cannot waive scientific runtime fields: "
            + ", ".join(sorted(unsupported))
        )
    if any(not isinstance(reason, str) or not reason.strip() for reason in allowed.values()):
        raise ValueError("Every allowed environment difference needs a rationale.")

    groups: dict[str, dict] = {}
    for manifest in manifests:
        stable = _stable_environment(manifest.get("environment", {}))
        group_id = canonical_identifier(stable)
        record = groups.setdefault(
            group_id,
            {"stable_environment": stable, "member_manifest_ids": []},
        )
        record["member_manifest_ids"].append(manifest["manifest_id"])

    values_by_key = {
        key: [record["stable_environment"].get(key) for record in groups.values()]
        for key in STABLE_ENVIRONMENT_KEYS
    }
    observed = sorted(
        key
        for key, values in values_by_key.items()
        if any(value != values[0] for value in values[1:])
    )
    unexpected = set(observed) - allowed_keys
    if unexpected:
        raise RuntimeError(
            "Accepted points differ in non-waived runtime fields: "
            + ", ".join(sorted(unexpected))
        )

    known_image_ids = sorted(
        {
            value
            for value in values_by_key["docker_image_id"]
            if value not in (None, "", "unrecorded")
        }
    )
    if len(known_image_ids) > 1:
        raise RuntimeError("Accepted points record conflicting known Docker image IDs.")
    known_image_hints = sorted(
        {
            value
            for value in values_by_key["docker_image_hint"]
            if value not in (None, "", "unrecorded")
        }
    )
    if len(known_image_hints) > 1:
        raise RuntimeError("Accepted points record conflicting known Docker image hints.")

    return {
        "allowed_differences": dict(sorted(allowed.items())),
        "observed_differing_keys": observed,
        "required_equal_keys": sorted(set(STABLE_ENVIRONMENT_KEYS) - allowed_keys),
        "known_docker_image_ids": known_image_ids,
        "known_docker_image_hints": known_image_hints,
        "groups": dict(sorted(groups.items())),
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


def _copy_file_verified(source: Path, destination: Path, expected: str) -> None:
    if not source.is_file():
        raise RuntimeError(f"Required evidence file is missing: {source}")
    observed = sha256_file(source)
    if observed != expected:
        raise RuntimeError(f"Evidence hash mismatch for {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected:
            raise RuntimeError(f"Conflicting evidence destination: {destination}")
        return
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected:
        raise RuntimeError(f"Copied evidence failed hashing: {destination}")


def _copy_point_evidence(
    *,
    source_root: Path,
    bundle_root: Path,
    manifest_relative: Path,
) -> Path:
    source_manifest_path = _within(
        source_root, manifest_relative, label="point manifest"
    )
    manifest = _read_json(source_manifest_path)
    destination_manifest = bundle_root / "evidence" / manifest_relative
    _copy_file_verified(
        source_manifest_path,
        destination_manifest,
        sha256_file(source_manifest_path),
    )

    source_point_dir = source_manifest_path.parent.resolve()
    destination_point_dir = destination_manifest.parent
    for relative_text, digest in sorted((manifest.get("output_sha256") or {}).items()):
        relative = _relative_path(relative_text, label="declared point output")
        source = _within(source_point_dir, relative, label="declared point output")
        destination = destination_point_dir / relative
        _copy_file_verified(source, destination, digest)
    return destination_manifest


def _write_points_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "classification",
        "reg_c",
        "status",
        "run_kind",
        "confirmation_round",
        "native_termination",
        "acceptance_basis",
        "block_count",
        "misfit",
        "unweighted_roughness",
        "weighted_penalty",
        "objective",
        "independent_manifest_id",
        "independent_manifest_path",
        "effective_manifest_id",
        "effective_manifest_path",
        "environment_group_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _validate_manual_record(plan: dict) -> tuple[dict, list[dict]]:
    deviation = plan.get("manual_protocol_deviation")
    if not isinstance(deviation, dict) or deviation.get("acknowledged") is not True:
        raise ValueError("The manual refined-window deviation must be acknowledged.")
    required = ["kind", "description", "rationale", "scope"]
    if any(not isinstance(deviation.get(key), str) or not deviation[key].strip() for key in required):
        raise ValueError(
            "manual_protocol_deviation requires kind, description, rationale, and scope."
        )
    exclusions = plan.get("exclusions")
    if exclusions is None:
        inventory = plan.get("exclusion_inventory")
        if not isinstance(inventory, dict):
            raise ValueError("At least one explicit exclusion record is required.")
        exclusions = []
        for category in ("manifested", "unmanifested", "duplicate_roots", "unexecuted"):
            for original in inventory.get(category) or []:
                record = dict(original)
                record["category"] = category
                record.setdefault("disposition", "excluded_from_selection")
                if category == "manifested" and "path" in record:
                    record["artifact_path"] = record["path"]
                exclusions.append(record)
    if not isinstance(exclusions, list) or not exclusions:
        raise ValueError("At least one explicit exclusion record is required.")
    for record in exclusions:
        if not isinstance(record, dict) or not str(record.get("reason", "")).strip():
            raise ValueError("Every exclusion must be a mapping with a reason.")
        if not str(record.get("disposition", "")).strip():
            raise ValueError("Every exclusion must state its disposition.")
    return dict(deviation), [dict(record) for record in exclusions]


def _copy_exclusions(
    exclusions: list[dict], *, source_root: Path, bundle_root: Path
) -> list[dict]:
    output = []
    for index, original in enumerate(exclusions, start=1):
        record = dict(original)
        artifact_text = record.pop("artifact_path", None)
        if artifact_text is not None:
            relative = _source_relative(
                artifact_text, source_root=source_root, label="excluded artifact"
            )
            source = _within(source_root, relative, label="excluded artifact")
            destination_relative = Path("excluded_evidence") / relative
            destination = bundle_root / destination_relative
            digest = sha256_file(source)
            _copy_file_verified(source, destination, digest)
            record["artifact_bundle_path"] = destination_relative.as_posix()
            record["artifact_sha256"] = digest
            if source.suffix.lower() == ".json":
                try:
                    payload = _read_json(source)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    record["artifact_schema"] = payload.get("schema")
                    record["artifact_manifest_id"] = payload.get("manifest_id")
        record["exclusion_index"] = index
        output.append(record)
    return output


def _manifest_reference(
    *, manifest: dict, manifest_path: Path, bundle_root: Path, environment_group_id: str
) -> dict:
    return {
        "reg_c": float(manifest["reg_c"]),
        "run_kind": manifest["run_kind"],
        "status": manifest["status"],
        "manifest_id": manifest["manifest_id"],
        "manifest_path": manifest_path.relative_to(bundle_root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "environment_group_id": environment_group_id,
    }


def assemble_bundle(
    *,
    source_root: Path,
    plan_path: Path,
    output_dir: Path,
    create_plot: bool = True,
) -> dict:
    """Create one immutable selection-only bundle from an explicit plan."""
    source_root = source_root.resolve()
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable bundle: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    plan = _read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unexpected assembly-plan schema: {plan.get('schema')}")
    deviation, exclusions = _validate_manual_record(plan)

    contract_relative = _relative_path(
        plan["protocol_contract_path"], label="protocol contract"
    )
    contract_path = _within(source_root, contract_relative, label="protocol contract")
    contract = _read_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise RuntimeError("Unexpected frozen L-curve contract schema.")
    if canonical_identifier(contract) != contract.get("manifest_id"):
        raise RuntimeError("Frozen L-curve contract identifier is invalid.")
    expected_contract = plan.get("formal_contract") or {}
    if (
        expected_contract.get("manifest_id") != contract.get("manifest_id")
        or expected_contract.get("manifest_sha256") != sha256_file(contract_path)
        or expected_contract.get("config_sha256") != contract.get("config_sha256")
    ):
        raise RuntimeError("Frozen contract differs from the recipe's pinned identity.")
    selector_sha256 = _selector_hash(contract.get("source_sha256") or {})
    current_selector = WORKFLOW_ROOT / "lcurve_selection.py"
    if sha256_file(current_selector) != selector_sha256:
        raise RuntimeError(
            "Current selector source differs from the source frozen by the points."
        )

    specifications = plan.get("curve_points")
    if not isinstance(specifications, list) or len(specifications) != 5:
        raise ValueError("The refined selection bundle requires exactly five curve points.")
    classifications = [record.get("classification") for record in specifications]
    if classifications.count("formal_endpoint") != 2 or classifications.count(
        "confirmed_refinement"
    ) != 3:
        raise ValueError(
            "curve_points must contain two formal endpoints and three confirmed refinements."
        )

    loaded = []
    unique_manifests: dict[str, dict] = {}
    for specification in specifications:
        independent_relative = _relative_path(
            specification["independent_manifest_path"],
            label="independent point manifest",
        )
        effective_relative = _relative_path(
            specification["effective_manifest_path"],
            label="effective point manifest",
        )
        independent_path = _within(
            source_root, independent_relative, label="independent point manifest"
        )
        effective_path = _within(
            source_root, effective_relative, label="effective point manifest"
        )
        independent = _read_json(independent_path)
        effective = _read_json(effective_path)
        expected_reg_c = float(specification["expected_reg_c"])
        if float(independent.get("reg_c")) != expected_reg_c or float(
            effective.get("reg_c")
        ) != expected_reg_c:
            raise RuntimeError("A planned curve point has the wrong reg_C.")
        if independent.get("run_kind") != "independent":
            raise RuntimeError("Every curve lineage must start from an independent point.")
        if specification["classification"] == "formal_endpoint":
            if independent_path != effective_path:
                raise RuntimeError("A formal endpoint must use its independent manifest.")
        else:
            if effective.get("run_kind") != "confirmation":
                raise RuntimeError("A confirmed refinement must use a confirmation endpoint.")
            parent = effective.get("parent_point_manifest") or {}
            if parent.get("manifest_id") != independent.get("manifest_id"):
                raise RuntimeError("A confirmation does not reference its planned parent.")
            if int(effective.get("confirmation_round", 0)) != 1:
                raise RuntimeError("A refined point must use its round-one confirmation.")
        reg_key = format(expected_reg_c, ".12g")
        pinned_independent = (plan.get("accepted_independent_by_reg_c") or {}).get(
            reg_key
        )
        pinned_effective = (plan.get("effective_by_reg_c") or {}).get(reg_key)
        if not pinned_independent or not pinned_effective:
            raise RuntimeError(f"Recipe does not pin both manifests for reg_C={reg_key}.")
        if (
            independent.get("manifest_id") != pinned_independent.get("manifest_id")
            or sha256_file(independent_path)
            != pinned_independent.get("manifest_sha256")
            or effective.get("manifest_id") != pinned_effective.get("manifest_id")
            or sha256_file(effective_path) != pinned_effective.get("manifest_sha256")
        ):
            raise RuntimeError(f"Pinned point identity mismatch at reg_C={reg_key}.")
        loaded.append(
            {
                "specification": dict(specification),
                "independent_relative": independent_relative,
                "effective_relative": effective_relative,
                "independent": independent,
                "effective": effective,
            }
        )
        unique_manifests[independent["manifest_id"]] = independent
        unique_manifests[effective["manifest_id"]] = effective

    environment_inventory = _environment_inventory(
        list(unique_manifests.values()), plan.get("environment_policy") or {}
    )
    environment_by_manifest = {}
    for group_id, group in environment_inventory["groups"].items():
        for manifest_id in group["member_manifest_ids"]:
            environment_by_manifest[manifest_id] = group_id

    for record in loaded:
        for key in ("independent", "effective"):
            manifest = record[key]
            group_id = environment_by_manifest[manifest["manifest_id"]]
            point_path = (
                _within(
                    source_root,
                    record[f"{key}_relative"],
                    label=f"{key} point manifest",
                )
                if key == "independent"
                else _within(
                    source_root,
                    record["effective_relative"],
                    label="effective point manifest",
                )
            )
            validate_point_manifest(
                manifest,
                point_dir=point_path.parent,
                reg_c=float(manifest["reg_c"]),
                contract=_contract_for_environment(
                    contract,
                    environment_inventory["groups"][group_id]["stable_environment"],
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
                study_root=source_root,
            )

    before_points = [_point_for_selection(row["independent"]) for row in loaded]
    after_points = [_point_for_selection(row["effective"]) for row in loaded]
    before_selection = select_corner(before_points)
    after_selection = select_corner(after_points)
    before_neighborhood = corner_neighborhood(before_points)
    after_neighborhood = corner_neighborhood(after_points)
    expected_selected = float(plan["expected_selected_reg_c"])
    if float(before_selection["selected"]["reg_c"]) != expected_selected or float(
        after_selection["selected"]["reg_c"]
    ) != expected_selected:
        raise RuntimeError("The frozen selector does not reproduce the planned decision.")
    if before_neighborhood != after_neighborhood:
        raise RuntimeError("The selected confirmation triple changed after confirmation.")
    confirmed_values = sorted(
        float(row["specification"]["expected_reg_c"])
        for row in loaded
        if row["specification"]["classification"] == "confirmed_refinement"
    )
    if confirmed_values != sorted(after_neighborhood["confirmation_reg_c"]):
        raise RuntimeError("The three confirmation manifests are not the selected triple.")
    if any(
        (row["effective"].get("confirmation") or {}).get("passed") is not True
        for row in loaded
        if row["specification"]["classification"] == "confirmed_refinement"
    ):
        raise RuntimeError("The selected triple lacks passing confirmation evidence.")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        _write_json(temporary / "assembly_plan.json", plan)
        _copy_file_verified(
            contract_path,
            temporary / "source_contract.json",
            sha256_file(contract_path),
        )
        decision_record = WORKFLOW_ROOT / "LCURVE_REFINED_WINDOW_FREEZE_RECORD.md"
        _copy_file_verified(
            decision_record,
            temporary / "LCURVE_REFINED_WINDOW_FREEZE_RECORD.md",
            sha256_file(decision_record),
        )
        formal_state_text = (plan.get("formal_study_state") or {}).get(
            "run_state_path"
        )
        if not formal_state_text:
            raise ValueError("Recipe must identify the stale formal run state.")
        formal_state_relative = _source_relative(
            formal_state_text, source_root=source_root, label="formal run state"
        )
        formal_state_path = _within(
            source_root, formal_state_relative, label="formal run state"
        )
        _copy_file_verified(
            formal_state_path,
            temporary / "formal_run_state.json",
            sha256_file(formal_state_path),
        )

        copied: dict[tuple[str, str], Path] = {}
        for row in loaded:
            for key in ("independent", "effective"):
                relative = row[f"{key}_relative"]
                copied[(row[key]["manifest_id"], key)] = _copy_point_evidence(
                    source_root=source_root,
                    bundle_root=temporary,
                    manifest_relative=relative,
                )

        enriched_exclusions = _copy_exclusions(
            exclusions, source_root=source_root, bundle_root=temporary
        )

        point_records = []
        point_references = []
        confirmation_references = []
        for row in sorted(
            loaded, key=lambda item: float(item["specification"]["expected_reg_c"])
        ):
            independent_path = temporary / "evidence" / row["independent_relative"]
            effective_path = temporary / "evidence" / row["effective_relative"]
            independent_group = environment_by_manifest[row["independent"]["manifest_id"]]
            effective_group = environment_by_manifest[row["effective"]["manifest_id"]]
            independent_reference = _manifest_reference(
                manifest=row["independent"],
                manifest_path=independent_path,
                bundle_root=temporary,
                environment_group_id=independent_group,
            )
            effective_reference = _manifest_reference(
                manifest=row["effective"],
                manifest_path=effective_path,
                bundle_root=temporary,
                environment_group_id=effective_group,
            )
            point_references.append(
                {
                    "classification": row["specification"]["classification"],
                    "reg_c": float(row["effective"]["reg_c"]),
                    "independent": independent_reference,
                    "effective": effective_reference,
                }
            )
            if row["effective"]["run_kind"] == "confirmation":
                confirmation_references.append(effective_reference)
            metrics = row["effective"]["metrics"]
            point_records.append(
                {
                    "classification": row["specification"]["classification"],
                    "reg_c": float(row["effective"]["reg_c"]),
                    "status": row["effective"]["status"],
                    "run_kind": row["effective"]["run_kind"],
                    "confirmation_round": row["effective"].get("confirmation_round"),
                    "native_termination": row["effective"]["native_termination"],
                    "acceptance_basis": row["effective"]["acceptance_basis"],
                    "block_count": row["effective"]["cumulative_block_count"],
                    "misfit": metrics["misfit"],
                    "unweighted_roughness": metrics["unweighted_roughness"],
                    "weighted_penalty": metrics["weighted_penalty"],
                    "objective": metrics["objective"],
                    "independent_manifest_id": row["independent"]["manifest_id"],
                    "independent_manifest_path": independent_path.relative_to(
                        temporary
                    ).as_posix(),
                    "effective_manifest_id": row["effective"]["manifest_id"],
                    "effective_manifest_path": effective_path.relative_to(
                        temporary
                    ).as_posix(),
                    "environment_group_id": effective_group,
                }
            )

        _write_points_csv(temporary / "lcurve_points.csv", point_records)
        _write_curvature(temporary / "curvature_preconfirmation.csv", before_selection)
        _write_curvature(temporary / "curvature_final.csv", after_selection)
        before_ambiguity = curvature_ambiguity(
            before_selection, float(contract["protocol"]["curvature_ambiguity_ratio"])
        )
        after_ambiguity = curvature_ambiguity(
            after_selection, float(contract["protocol"]["curvature_ambiguity_ratio"])
        )
        decision = {
            "schema": DECISION_SCHEMA,
            "artifact_kind": "selection_decision_only",
            "selector_source_sha256": selector_sha256,
            "selection_rule": contract["protocol"]["selection"],
            "curvature_ambiguity_ratio": contract["protocol"][
                "curvature_ambiguity_ratio"
            ],
            "preconfirmation": {
                "selection": before_selection,
                "ambiguity": before_ambiguity,
                "neighborhood": before_neighborhood,
            },
            "postconfirmation": {
                "selection": after_selection,
                "ambiguity": after_ambiguity,
                "neighborhood": after_neighborhood,
                "monotonic_diagnostics": _monotonic_diagnostics(after_points),
            },
            "confirmation_resolution": {
                "round": 1,
                "confirmation_reg_c": after_neighborhood["confirmation_reg_c"],
                "confirmation_manifest_ids": [
                    reference["manifest_id"] for reference in confirmation_references
                ],
                "all_passed": True,
                "candidate_unchanged": True,
                "triple_unchanged": True,
            },
            "selected_reg_c": expected_selected,
            "decision": (
                "Freeze reg_C for the next fresh whole-sector inversion; this "
                "artifact is not itself a definitive inversion."
            ),
        }
        decision["manifest_id"] = canonical_identifier(decision)
        _write_json(temporary / "selection_final.json", decision)
        selected_effective = next(
            row for row in point_records if float(row["reg_c"]) == expected_selected
        )
        selected_regularization = {
            "schema": SELECTED_REGULARIZATION_SCHEMA,
            "artifact_kind": "selection_decision_only",
            "status": "frozen_for_fresh_definitive_inversion",
            "reg_c": expected_selected,
            "effective_point_manifest_id": selected_effective[
                "effective_manifest_id"
            ],
            "selection_final_manifest_id": decision["manifest_id"],
            "definitive_inversion_created": False,
            "next_step": "fresh_whole_sector_inversion_from_exact_C_zero",
        }
        selected_regularization["manifest_id"] = canonical_identifier(
            selected_regularization
        )
        _write_json(
            temporary / "selected_regularization.json", selected_regularization
        )
        if create_plot:
            _plot_lcurve(
                after_points, expected_selected, temporary / "lcurve_appendix.png"
            )

        parent_hashes = {}
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "selection_bundle.json":
                parent_hashes[path.relative_to(temporary).as_posix()] = sha256_file(path)

        bundle = {
            "schema": BUNDLE_SCHEMA,
            "status": "complete",
            "artifact_kind": "selection_decision_only",
            "selection_decision_only": True,
            "definitive_inversion_created": False,
            "created_utc": _utc_now(),
            "source_provenance": {
                "source_root_at_assembly": str(source_root),
                "assembly_plan_path": str(plan_path),
                "source_contract_manifest_id": contract["manifest_id"],
                "source_contract_sha256": sha256_file(contract_path),
                "config_sha256": contract["config_sha256"],
                "source_sha256": contract["source_sha256"],
                "expected_input_sha256": contract["expected_input_sha256"],
                "expected_design_sha256": contract["expected_design_sha256"],
                "selector_source_sha256": selector_sha256,
            },
            "environment_policy": environment_inventory,
            "environment_reconciliation": plan.get("environment_compatibility"),
            "manual_protocol_deviation": deviation,
            "exclusions": enriched_exclusions,
            "curve_point_count": len(point_references),
            "curve_points": point_references,
            "confirmation_manifest_count": len(confirmation_references),
            "confirmation_manifests": confirmation_references,
            "selected_reg_c": expected_selected,
            "selection_final_manifest_id": decision["manifest_id"],
            "selected_regularization_manifest_id": selected_regularization[
                "manifest_id"
            ],
            "plot_created": create_plot,
            "output_sha256": parent_hashes,
        }
        bundle["manifest_id"] = canonical_identifier(bundle)
        _write_json(temporary / "selection_bundle.json", bundle)
        os.replace(temporary, output_dir)
        return bundle
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    bundle = assemble_bundle(
        source_root=args.source_root,
        plan_path=args.plan,
        output_dir=args.output_dir,
        create_plot=not args.no_plot,
    )
    print(
        json.dumps(
            {
                "status": "assembled",
                "bundle_manifest_id": bundle["manifest_id"],
                "selected_reg_c": bundle["selected_reg_c"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
