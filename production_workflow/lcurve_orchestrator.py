"""Resumable, provenance-locked controller for the revised Amundsen L-curve."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from lcurve_selection import (
    corner_neighborhood,
    curvature_ambiguity,
    extension_side,
    geometric_refinements,
    select_corner,
    valid_points,
)
from lcurve_runtime import canonical_identifier, sha256_file


STUDY_SCHEMA = "jog-production-lcurve-study-v2"
STATE_SCHEMA = "jog-production-lcurve-state-v2"
CONTRACT_SCHEMA = "jog-production-lcurve-contract-v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reg_c_slug(reg_c: float) -> str:
    """Return a readable collision-resistant directory label for a float."""
    if not math.isfinite(reg_c) or reg_c <= 0.0:
        raise ValueError("reg_C must be finite and positive.")
    exact = format(float(reg_c), ".17g")
    readable = format(float(reg_c), ".12g").replace("-", "m").replace(".", "p")
    digest = hashlib.sha256(exact.encode("ascii")).hexdigest()[:8]
    return f"regc_{readable}_{digest}"


def _stable_environment(environment: dict) -> dict:
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


def _snapshot_sources(source_hashes: dict, destination: Path) -> None:
    destination.mkdir()
    used = set()
    for source, digest in sorted(source_hashes.items()):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot snapshot L-curve source {path}")
        name = path.name
        if name in used:
            name = f"{path.stem}_{digest[:10]}{path.suffix}"
        used.add(name)
        shutil.copy2(path, destination / name)


def _verify_output_hashes(root: Path, declared: dict) -> None:
    resolved_root = root.resolve()
    for relative, expected in declared.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(f"Point output escapes its run directory: {relative}") from error
        if not path.is_file():
            raise RuntimeError(f"Declared point output is missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"Point output hash mismatch for {path}: {observed} != {expected}"
            )


def _point_process_is_active(point_dir: Path, run_id: str) -> bool:
    """Return whether a point's recorded Linux process is still that run."""
    marker = point_dir / "active_process.json"
    if not marker.is_file():
        return False
    try:
        record = _read_json(marker)
        pid = int(record["pid"])
        command_path = Path(f"/proc/{pid}/cmdline")
        if not command_path.is_file():
            return False
        command = command_path.read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        )
        return (
            "production_amundsen.py" in command
            and (
                "lcurve-point" in command
                or "lcurve-confirmation" in command
            )
            and run_id in command
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _orchestrator_process_is_active(state: dict, output_dir: Path) -> bool:
    try:
        pid = int(state["orchestrator_pid"])
        if pid == os.getpid():
            return False
        command_path = Path(f"/proc/{pid}/cmdline")
        if not command_path.is_file():
            return False
        command = command_path.read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        )
        return (
            "production_amundsen.py" in command
            and " lcurve " in f" {command} "
            and output_dir.name in command
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def validate_point_manifest(
    manifest: dict,
    *,
    point_dir: Path,
    reg_c: float,
    contract: dict,
    expected_run_kind: str | None = None,
    expected_confirmation_round: int | None = None,
    expected_parent_manifest_id: str | None = None,
    study_root: Path | None = None,
) -> None:
    if manifest.get("schema") != "jog-production-lcurve-point-v2":
        raise RuntimeError("Unexpected L-curve point manifest schema.")
    if canonical_identifier(manifest) != manifest.get("manifest_id"):
        raise RuntimeError("L-curve point manifest identifier is invalid.")
    if float(manifest.get("reg_c")) != float(reg_c):
        raise RuntimeError("L-curve point reg_C does not match its request.")
    if manifest.get("config_sha256") != contract["config_sha256"]:
        raise RuntimeError("L-curve point used a different production config.")
    if manifest.get("source_sha256") != contract["source_sha256"]:
        raise RuntimeError("L-curve point used different scientific source files.")
    if _stable_environment(manifest.get("environment", {})) != contract[
        "stable_environment"
    ]:
        raise RuntimeError("L-curve point used a different frozen runtime environment.")
    preflight_path = point_dir / "input_preflight" / "preflight_manifest.json"
    if not preflight_path.is_file():
        raise RuntimeError("L-curve point is missing its immutable-input preflight.")
    preflight = _read_json(preflight_path)
    if canonical_identifier(preflight) != preflight.get("manifest_id"):
        raise RuntimeError("Point-level input preflight manifest identifier is invalid.")
    if preflight.get("manifest_id") != manifest.get("input_preflight_manifest_id"):
        raise RuntimeError("Point manifest references a different input preflight.")
    if preflight.get("status") != "pass" or preflight.get("hard_failure_count") != 0:
        raise RuntimeError("Point-level immutable-input preflight did not pass.")
    if preflight.get("config_sha256") != contract["config_sha256"]:
        raise RuntimeError("Point-level preflight used a different config.")
    if preflight.get("source_sha256") != contract["source_sha256"]:
        raise RuntimeError("Point-level preflight used different scientific sources.")
    checks = {check["name"]: check for check in preflight.get("checks", [])}
    expected_checks = {
        **{
            f"input_sha256:{name}": digest
            for name, digest in contract["expected_input_sha256"].items()
        },
        **{
            f"design_sha256:{name}": digest
            for name, digest in contract["expected_design_sha256"].items()
        },
    }
    for name, digest in expected_checks.items():
        record = checks.get(name)
        if (
            not record
            or record.get("status") != "pass"
            or record.get("observed") != digest
        ):
            raise RuntimeError(f"Point preflight lacks passing hash evidence: {name}")
    _verify_output_hashes(point_dir, manifest.get("output_sha256", {}))
    status = manifest.get("status")
    if status not in {"valid", "invalid"}:
        raise RuntimeError("L-curve child did not produce a reusable endpoint state.")
    run_kind = manifest.get("run_kind")
    if run_kind not in {"independent", "confirmation"}:
        raise RuntimeError("An L-curve point has an unknown run kind.")
    if expected_run_kind is not None and run_kind != expected_run_kind:
        raise RuntimeError("L-curve point run kind does not match its request.")

    blocks = manifest.get("blocks") or []
    cumulative_blocks = int(manifest.get("cumulative_block_count", len(blocks)))
    if not blocks or cumulative_blocks < len(blocks):
        raise RuntimeError("L-curve point has an invalid block history.")
    parent_manifest = None
    if run_kind == "independent":
        if cumulative_blocks != len(blocks):
            raise RuntimeError("Independent point has inconsistent block ancestry.")
        if status == "valid" and len(blocks) < int(
            contract["protocol"]["minimum_blocks"]
        ):
            raise RuntimeError("A valid point lacks the minimum block history.")
        if manifest.get("parent_point_manifest") not in (None, {}):
            raise RuntimeError("Independent point unexpectedly declares a parent.")
    else:
        round_number = int(manifest.get("confirmation_round", 0))
        if round_number < 1 or len(blocks) != 1:
            raise RuntimeError("A confirmation lacks its one-block round identity.")
        if (
            expected_confirmation_round is not None
            and round_number != int(expected_confirmation_round)
        ):
            raise RuntimeError("Confirmation round does not match its request.")
        parent = manifest.get("parent_point_manifest") or {}
        relative_parent = parent.get("path_relative_to_child")
        portable_parent = (
            (point_dir / relative_parent).resolve() if relative_parent else None
        )
        declared_parent = Path(parent.get("path", ""))
        if not declared_parent.is_absolute():
            declared_parent = (point_dir / declared_parent).resolve()
        parent_path = (
            portable_parent
            if portable_parent is not None and portable_parent.is_file()
            else declared_parent
        )
        if study_root is not None:
            try:
                parent_path.resolve().relative_to(study_root.resolve())
            except ValueError as error:
                raise RuntimeError(
                    "Confirmation parent escapes the immutable study directory."
                ) from error
        if (
            not parent_path.is_file()
            or sha256_file(parent_path) != parent.get("manifest_sha256")
        ):
            raise RuntimeError("A confirmation parent manifest failed hashing.")
        parent_manifest = _read_json(parent_path)
        if canonical_identifier(parent_manifest) != parent_manifest.get("manifest_id"):
            raise RuntimeError("Confirmation parent identifier is invalid.")
        if parent_manifest.get("manifest_id") != parent.get("manifest_id"):
            raise RuntimeError("A confirmation references a different parent ID.")
        if (
            expected_parent_manifest_id is not None
            and parent_manifest.get("manifest_id") != expected_parent_manifest_id
        ):
            raise RuntimeError("Confirmation parent differs from the planned lineage.")
        if float(parent_manifest.get("reg_c")) != float(reg_c):
            raise RuntimeError("Confirmation parent has a different reg_C.")
        if parent_manifest.get("config_sha256") != contract["config_sha256"]:
            raise RuntimeError("Confirmation parent used a different config.")
        if parent_manifest.get("source_sha256") != contract["source_sha256"]:
            raise RuntimeError("Confirmation parent used different sources.")
        if _stable_environment(parent_manifest.get("environment", {})) != contract[
            "stable_environment"
        ]:
            raise RuntimeError("Confirmation parent used a different environment.")
        parent_blocks = parent_manifest.get("blocks") or []
        parent_count = int(
            parent_manifest.get("cumulative_block_count", len(parent_blocks))
        )
        if cumulative_blocks != parent_count + 1:
            raise RuntimeError("Confirmation cumulative block count breaks lineage.")
        if int(blocks[0].get("block", -1)) != cumulative_blocks:
            raise RuntimeError("Confirmation block number breaks lineage.")

    metrics = manifest.get("metrics") or {}
    fields = manifest.get("fields") or {}
    required_metrics = [
        "misfit",
        "raw_gradient_integral",
        "unweighted_roughness",
        "weighted_penalty",
        "objective",
    ]
    if not all(
        key in metrics and math.isfinite(float(metrics[key]))
        for key in required_metrics
    ):
        raise RuntimeError("L-curve point lacks finite objective metrics.")
    if not (
        float(metrics["misfit"]) > 0.0
        and float(metrics["unweighted_roughness"]) > 0.0
        and metrics.get("weighted_penalty_identity") is True
        and metrics.get("rol_objective_matches_reassembled") is True
    ):
        raise RuntimeError("L-curve point fails objective-identity checks.")
    if not (
        fields.get("C_finite") is True
        and fields.get("theta_finite") is True
        and float(fields.get("theta_minimum")) == 0.0
        and float(fields.get("theta_maximum")) == 0.0
        and fields.get("velocity_finite") is True
    ):
        raise RuntimeError("L-curve point fails saved-field checks.")
    if not manifest.get("solver_log_crosscheck_passed"):
        raise RuntimeError("L-curve point lacks its native ROL status cross-check.")
    native_termination = manifest.get("native_termination")
    if native_termination not in {
        "converged_gradient",
        "converged_step",
        "iteration_limit",
    }:
        raise RuntimeError("L-curve point has an unrecognized native termination.")

    acceptance_basis = manifest.get("acceptance_basis")
    if status == "valid":
        allowed_acceptance = {
            "rol_gradient": "converged_gradient",
            "rol_step": "converged_step",
            "practical_er_stability": None,
        }
        if acceptance_basis not in allowed_acceptance:
            raise RuntimeError("A valid point lacks an approved acceptance basis.")
        required_native = allowed_acceptance[acceptance_basis]
        if required_native is not None and native_termination != required_native:
            raise RuntimeError(
                "A ROL-accepted point disagrees with its native termination."
            )
        evidence = (
            manifest.get("confirmation")
            if run_kind == "confirmation"
            else manifest.get("stability")
        ) or {}
        if evidence.get("passed") is not True:
            raise RuntimeError("A valid point lacks passing stability evidence.")
    else:
        if run_kind == "confirmation":
            evidence = manifest.get("confirmation") or {}
            required_deltas = [
                evidence.get("relative_misfit_change"),
                evidence.get("relative_roughness_change"),
            ]
            if evidence.get("passed") is not False or not all(
                value is not None and math.isfinite(float(value))
                for value in required_deltas
            ):
                raise RuntimeError(
                    "Invalid confirmation is not a retryable stability failure."
                )
        if acceptance_basis is not None:
            raise RuntimeError("Invalid point cannot declare an acceptance basis.")


def point_summary(manifest: dict, point_dir: Path, study_dir: Path) -> dict:
    metrics = manifest.get("metrics") or {}
    return {
        "reg_c": float(manifest["reg_c"]),
        "status": manifest.get("status"),
        "termination": manifest.get("native_termination"),
        "native_termination": manifest.get("native_termination"),
        "acceptance_basis": manifest.get("acceptance_basis"),
        "block_count": int(
            manifest.get("cumulative_block_count", len(manifest.get("blocks") or []))
        ),
        "stability_passed": (manifest.get("stability") or {}).get("passed"),
        "run_kind": manifest.get("run_kind"),
        "confirmation_round": manifest.get("confirmation_round"),
        "misfit": metrics.get("misfit"),
        "unweighted_roughness": metrics.get("unweighted_roughness"),
        "weighted_penalty": metrics.get("weighted_penalty"),
        "objective": metrics.get("objective"),
        "point_manifest_id": manifest.get("manifest_id"),
        "point_manifest_path": str(
            (point_dir / "point_manifest.json").relative_to(study_dir)
        ),
    }


def verify_completed_study(output_dir: Path, contract: dict) -> dict:
    """Verify the complete parent/point/definitive chain before reuse."""
    output_dir = output_dir.resolve()
    study_path = output_dir / "study_manifest.json"
    if not study_path.is_file():
        raise RuntimeError("Completed state lacks study_manifest.json.")
    study = _read_json(study_path)
    if study.get("schema") != STUDY_SCHEMA:
        raise RuntimeError("Unexpected completed L-curve study schema.")
    if canonical_identifier(study) != study.get("manifest_id"):
        raise RuntimeError("Completed L-curve study manifest is invalid.")
    if study.get("contract_id") != contract.get("manifest_id"):
        raise RuntimeError("Completed L-curve study references a different contract.")
    if study.get("status") != "complete":
        raise RuntimeError("Completed L-curve study has a non-complete status.")
    _verify_output_hashes(output_dir, study.get("output_sha256", {}))

    table_path = output_dir / "lcurve_points.csv"
    with table_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != study.get("point_count") or len(rows) != study.get(
        "formal_point_count"
    ):
        raise RuntimeError("Completed study point count does not match its table.")
    for row in rows:
        reg_c = float(row["reg_c"])
        point_path = (output_dir / row["point_manifest_path"]).resolve()
        try:
            point_path.relative_to(output_dir)
        except ValueError as error:
            raise RuntimeError("Effective point path escapes the study.") from error
        point = _read_json(point_path)
        validate_point_manifest(
            point,
            point_dir=point_path.parent,
            reg_c=reg_c,
            contract=contract,
            study_root=output_dir,
        )
        expected_id = study.get("point_manifest_ids", {}).get(
            format(reg_c, ".17g")
        )
        if point.get("manifest_id") != row.get("point_manifest_id"):
            raise RuntimeError(f"Point table ID mismatch at reg_C={reg_c}.")
        if point.get("manifest_id") != expected_id:
            raise RuntimeError(f"Study point ID mismatch at reg_C={reg_c}.")

    effective_ids = study.get("effective_point_manifest_ids") or {}
    if effective_ids != study.get("point_manifest_ids"):
        raise RuntimeError("Effective point ID inventories disagree.")
    formal_references = study.get("formal_point_manifests") or []
    if len(formal_references) != int(study.get("formal_manifest_count", -1)) or len(
        formal_references
    ) != int(study.get("formal_point_count", -1)):
        raise RuntimeError("Formal point manifest inventory is inconsistent.")
    known_lineage_ids = set()
    latest_lineage_by_reg = {}
    for reference in formal_references:
        path = (output_dir / reference["path"]).resolve()
        try:
            path.relative_to(output_dir)
        except ValueError as error:
            raise RuntimeError("Formal point manifest escapes the study.") from error
        if not path.is_file() or sha256_file(path) != reference.get(
            "manifest_sha256"
        ):
            raise RuntimeError("Formal point manifest failed its study hash.")
        point = _read_json(path)
        validate_point_manifest(
            point,
            point_dir=path.parent,
            reg_c=float(reference["reg_c"]),
            contract=contract,
            expected_run_kind="independent",
            study_root=output_dir,
        )
        if (
            point.get("manifest_id") != reference.get("manifest_id")
            or point.get("status") != reference.get("status")
            or point.get("status") != "valid"
        ):
            raise RuntimeError("Formal point study inventory is inconsistent.")
        if point["manifest_id"] in known_lineage_ids:
            raise RuntimeError("Formal point inventory contains a duplicate manifest.")
        known_lineage_ids.add(point["manifest_id"])
        latest_lineage_by_reg[float(reference["reg_c"])] = point["manifest_id"]
    confirmation_references = study.get("confirmation_manifests") or []
    if len(confirmation_references) != int(
        study.get("confirmation_manifest_count", -1)
    ):
        raise RuntimeError("Confirmation manifest count is inconsistent.")
    confirmation_ids = []
    seen_confirmation_ids = set()
    for reference in confirmation_references:
        path = (output_dir / reference["path"]).resolve()
        try:
            path.relative_to(output_dir)
        except ValueError as error:
            raise RuntimeError("Confirmation manifest escapes the study.") from error
        if not path.is_file() or sha256_file(path) != reference.get(
            "manifest_sha256"
        ):
            raise RuntimeError("Confirmation manifest failed its study hash.")
        point = _read_json(path)
        validate_point_manifest(
            point,
            point_dir=path.parent,
            reg_c=float(reference["reg_c"]),
            contract=contract,
            expected_run_kind="confirmation",
            expected_confirmation_round=int(reference["round"]),
            expected_parent_manifest_id=reference.get("parent_manifest_id"),
            study_root=output_dir,
        )
        if (
            point.get("manifest_id") != reference.get("manifest_id")
            or point.get("status") != reference.get("status")
        ):
            raise RuntimeError("Confirmation study inventory is inconsistent.")
        if reference.get("parent_manifest_id") not in known_lineage_ids:
            raise RuntimeError("Confirmation parent is absent from prior lineage.")
        reg_c = float(reference["reg_c"])
        if latest_lineage_by_reg.get(reg_c) != reference.get("parent_manifest_id"):
            raise RuntimeError("Confirmation did not continue the latest same-reg lineage.")
        if point["manifest_id"] in seen_confirmation_ids:
            raise RuntimeError("Confirmation inventory contains a duplicate manifest.")
        seen_confirmation_ids.add(point["manifest_id"])
        known_lineage_ids.add(point["manifest_id"])
        latest_lineage_by_reg[reg_c] = point["manifest_id"]
        confirmation_ids.append(point["manifest_id"])
    if confirmation_ids != study.get("confirmation_manifest_ids"):
        raise RuntimeError("Confirmation manifest ID inventory is inconsistent.")
    round_records = study.get("confirmation_rounds") or []
    maximum_rounds = int(contract["protocol"]["maximum_confirmation_rounds"])
    if not 1 <= len(round_records) <= maximum_rounds:
        raise RuntimeError("Confirmation round inventory has an invalid length.")
    references_by_round = {}
    for reference in confirmation_references:
        references_by_round.setdefault(int(reference["round"]), []).append(reference)
    for expected_round, record in enumerate(round_records, start=1):
        if int(record.get("round", 0)) != expected_round:
            raise RuntimeError("Confirmation rounds are not sequential.")
        references = references_by_round.get(expected_round, [])
        requested = [float(value) for value in record.get("confirmation_reg_c", [])]
        if (
            len(requested) != 3
            or len(set(requested)) != 3
            or [float(item["reg_c"]) for item in references] != requested
        ):
            raise RuntimeError("Confirmation round triple disagrees with its inventory.")
        expected_ids = {
            format(float(item["reg_c"]), ".17g"): item["manifest_id"]
            for item in references
        }
        expected_statuses = {
            format(float(item["reg_c"]), ".17g"): item["status"]
            for item in references
        }
        expected_parents = {
            format(float(item["reg_c"]), ".17g"): item["parent_manifest_id"]
            for item in references
        }
        if (
            record.get("confirmation_manifest_ids") != expected_ids
            or record.get("statuses") != expected_statuses
            or record.get("parent_manifest_ids") != expected_parents
        ):
            raise RuntimeError("Confirmation round evidence disagrees with manifests.")
        passed = all(status == "valid" for status in expected_statuses.values())
        if record.get("passed") is not passed:
            raise RuntimeError("Confirmation round pass flag is inconsistent.")
        if expected_round < len(round_records):
            next_requested = [
                float(value)
                for value in round_records[expected_round].get(
                    "confirmation_reg_c", []
                )
            ]
            expected_next = (
                record.get("confirmation_reg_c_after") if passed else requested
            )
            if expected_next is None or next_requested != [
                float(value) for value in expected_next
            ]:
                raise RuntimeError("Confirmation retry triple breaks the protocol.")
    final_round = round_records[-1]
    if not (
        final_round.get("passed") is True
        and final_round.get("triple_stable") is True
        and float(final_round.get("candidate_reg_c_after"))
        == float(study.get("selected_reg_c"))
    ):
        raise RuntimeError("Final confirmation round did not resolve the selected triple.")

    definitive_path = output_dir / "definitive_inversion.json"
    definitive = _read_json(definitive_path)
    if canonical_identifier(definitive) != definitive.get("manifest_id"):
        raise RuntimeError("Definitive inversion reference is invalid.")
    if definitive.get("manifest_id") != study.get(
        "definitive_inversion_manifest_id"
    ):
        raise RuntimeError("Study references a different definitive inversion.")
    selected_path = (output_dir / definitive["point_manifest_path"]).resolve()
    try:
        selected_path.relative_to(output_dir)
    except ValueError as error:
        raise RuntimeError("Definitive point path escapes the study.") from error
    selected = _read_json(selected_path)
    if selected.get("manifest_id") != definitive.get("point_manifest_id"):
        raise RuntimeError("Definitive reference points to a different point ID.")
    if sha256_file(selected_path) != definitive.get("point_manifest_sha256"):
        raise RuntimeError("Definitive point manifest hash does not match.")
    if float(selected["reg_c"]) != float(definitive["reg_c"]):
        raise RuntimeError("Definitive reg_C does not match its selected point.")
    if selected.get("run_kind") != "confirmation" or selected.get("status") != "valid":
        raise RuntimeError("Definitive inversion is not a valid confirmed endpoint.")
    selected_key = format(float(definitive["reg_c"]), ".17g")
    if effective_ids.get(selected_key) != selected.get("manifest_id"):
        raise RuntimeError("Definitive inversion is not the effective selected row.")
    return study


def _default_point_launcher(
    *,
    entrypoint: Path,
    workflow_root: Path,
    config_path: Path,
    repo_root: Path,
    point_root: Path,
    run_id: str,
    reg_c: float,
    log_path: Path,
) -> int:
    command = [
        sys.executable,
        str(entrypoint),
        "lcurve-point",
        "--config",
        str(config_path),
        "--repo-root",
        str(repo_root),
        "--output-root",
        str(point_root),
        "--run-id",
        run_id,
        "--reg-c",
        repr(float(reg_c)),
    ]
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{_utc_now()} LAUNCH {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=workflow_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        stream.write(f"{_utc_now()} RETURN reg_C={reg_c} code={result.returncode}\n")
    return int(result.returncode)


def _default_confirmation_launcher(
    *,
    entrypoint: Path,
    workflow_root: Path,
    config_path: Path,
    repo_root: Path,
    point_root: Path,
    run_id: str,
    reg_c: float,
    parent_point_manifest_path: Path,
    confirmation_round: int,
    log_path: Path,
) -> int:
    command = [
        sys.executable,
        str(entrypoint),
        "lcurve-confirmation",
        "--config",
        str(config_path),
        "--repo-root",
        str(repo_root),
        "--output-root",
        str(point_root),
        "--run-id",
        run_id,
        "--reg-c",
        repr(float(reg_c)),
        "--parent-point-manifest",
        str(parent_point_manifest_path),
        "--confirmation-round",
        str(int(confirmation_round)),
    ]
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{_utc_now()} LAUNCH {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=workflow_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        stream.write(
            f"{_utc_now()} RETURN confirmation_round={confirmation_round} "
            f"reg_C={reg_c} code={result.returncode}\n"
        )
    return int(result.returncode)


def _point_values(config: dict, key: str) -> list[float]:
    values = [float(value) for value in config["inversion"][key]]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{key} contains a non-positive or non-finite value.")
    if len(values) != len(set(values)):
        raise ValueError(f"{key} contains duplicate values.")
    return values


def _build_contract(
    *,
    config: dict,
    config_path: Path,
    repo_root: Path,
    preflight_class,
    forward_smoke_dir: Path | None,
    require_forward_smoke: bool,
) -> dict:
    probe = preflight_class(config, repo_root, Path("/unused/lcurve_contract"))
    environment = probe.environment()
    source_identity = probe.source_identity()
    stable_environment = _stable_environment(environment)
    forward_smoke = None
    if require_forward_smoke:
        if forward_smoke_dir is None:
            raise ValueError(
                "Production L-curve launch requires --forward-smoke-dir."
            )
        smoke_root = forward_smoke_dir.resolve()
        smoke_path = smoke_root / "forward_smoke_manifest.json"
        smoke = _read_json(smoke_path)
        if smoke.get("schema") != "jog-production-forward-smoke-v1":
            raise RuntimeError("Unexpected forward-smoke manifest schema.")
        if canonical_identifier(smoke) != smoke.get("manifest_id"):
            raise RuntimeError("Forward-smoke manifest identifier is invalid.")
        if smoke.get("status") != "pass" or not all(
            smoke.get("checks", {}).values()
        ):
            raise RuntimeError("The prerequisite forward smoke did not pass.")
        if smoke.get("config_sha256") != sha256_file(config_path):
            raise RuntimeError("Forward smoke used a different production config.")
        if smoke.get("source_sha256") != source_identity:
            raise RuntimeError("Forward smoke used different scientific sources.")
        if _stable_environment(smoke.get("environment", {})) != stable_environment:
            raise RuntimeError("Forward smoke used a different frozen environment.")
        _verify_output_hashes(smoke_root, smoke.get("output_sha256", {}))
        preflight_path = smoke_root / "input_preflight" / "preflight_manifest.json"
        smoke_preflight = _read_json(preflight_path)
        if canonical_identifier(smoke_preflight) != smoke_preflight.get(
            "manifest_id"
        ):
            raise RuntimeError("Forward-smoke input-preflight identifier is invalid.")
        if smoke_preflight.get("manifest_id") != smoke.get(
            "input_preflight_manifest_id"
        ):
            raise RuntimeError("Forward smoke references a different preflight.")
        if smoke_preflight.get("status") != "pass" or smoke_preflight.get(
            "hard_failure_count"
        ) != 0:
            raise RuntimeError("Forward-smoke input preflight did not pass.")
        forward_smoke = {
            "path": str(smoke_root),
            "manifest_id": smoke["manifest_id"],
            "manifest_sha256": sha256_file(smoke_path),
            "input_preflight_manifest_id": smoke["input_preflight_manifest_id"],
        }
    contract = {
        "schema": CONTRACT_SCHEMA,
        "created_utc": _utc_now(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "repo_root": str(repo_root.resolve()),
        "source_sha256": source_identity,
        "stable_environment": stable_environment,
        "forward_smoke": forward_smoke,
        "expected_input_sha256": {
            name: specification["sha256"]
            for name, specification in config["inputs"].items()
        },
        "expected_design_sha256": {
            name: specification["sha256"]
            for name, specification in config["frozen_design"].items()
        },
        "protocol": {
            "base_reg_c": _point_values(config, "lcurve_base_reg_c"),
            "low_extension_reg_c": config["inversion"][
                "lcurve_low_extension_reg_c"
            ],
            "high_extension_reg_c": config["inversion"][
                "lcurve_high_extension_reg_c"
            ],
            "independent_initial_C": config["physics"]["initial_log_friction_c"],
            "gradient_tolerance": config["inversion"]["gradient_tolerance"],
            "step_tolerance": config["inversion"]["step_tolerance"],
            "block_iterations": config["inversion"]["lcurve_block_iterations"],
            "minimum_blocks": config["inversion"]["lcurve_min_blocks"],
            "maximum_blocks": config["inversion"]["lcurve_max_blocks"],
            "stable_transitions_required": config["inversion"][
                "lcurve_stable_transitions"
            ],
            "relative_misfit_tolerance": config["inversion"][
                "lcurve_relative_misfit_tolerance"
            ],
            "relative_roughness_tolerance": config["inversion"][
                "lcurve_relative_roughness_tolerance"
            ],
            "gradient_safety_tolerance": config["inversion"][
                "lcurve_gradient_safety_tolerance"
            ],
            "curvature_ambiguity_ratio": config["inversion"][
                "lcurve_curvature_ambiguity_ratio"
            ],
            "maximum_refinement_rounds": config["inversion"][
                "lcurve_max_refinement_rounds"
            ],
            "maximum_confirmation_rounds": config["inversion"][
                "lcurve_max_confirmation_rounds"
            ],
            "maximum_formal_points": config["inversion"][
                "lcurve_max_formal_points"
            ],
            "selection": (
                "maximum Menger curvature after independent [0,1] normalization "
                "of log10(misfit) and log10(unweighted roughness); exact ties "
                "favor smaller reg_C"
            ),
            "adaptive_policy": (
                "one outward endpoint extension only for a boundary-adjacent "
                "corner; one two-sided geometric-midpoint round only when the "
                "largest curvature is less than 1.25 times the second largest"
            ),
            "confirmation": (
                "one same-reg_C 50-iteration block for the selected corner and "
                "its immediate neighbors; repeat once only if the corner changes "
                "or the confirmation deltas fail the frozen stability tolerance"
            ),
        },
    }
    contract["manifest_id"] = canonical_identifier(contract)
    return contract


def _validate_resume_contract(
    contract: dict,
    *,
    config: dict,
    config_path: Path,
    repo_root: Path,
    preflight_class,
    forward_smoke_dir: Path | None,
    require_forward_smoke: bool,
) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise RuntimeError("Unexpected L-curve contract schema.")
    if canonical_identifier(contract) != contract.get("manifest_id"):
        raise RuntimeError("The saved L-curve contract identifier is invalid.")
    current = _build_contract(
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        preflight_class=preflight_class,
        forward_smoke_dir=(
            forward_smoke_dir
            if forward_smoke_dir is not None
            else (
                Path(contract["forward_smoke"]["path"])
                if contract.get("forward_smoke")
                else None
            )
        ),
        require_forward_smoke=require_forward_smoke,
    )
    keys = [
        "config_sha256",
        "repo_root",
        "source_sha256",
        "stable_environment",
        "forward_smoke",
        "expected_input_sha256",
        "expected_design_sha256",
        "protocol",
    ]
    changed = [key for key in keys if current[key] != contract.get(key)]
    if changed:
        raise RuntimeError(
            "Refusing to resume under a changed production contract: "
            + ", ".join(changed)
        )


def _stage_rows(points: list[dict], phases: dict[float, str]) -> list[dict]:
    return [
        {"phase": phases.get(float(point["reg_c"])), **point}
        for point in sorted(points, key=lambda item: float(item["reg_c"]))
    ]


def _write_curvature(path: Path, result: dict) -> None:
    _write_csv(
        path,
        result["curvature_table"],
        [
            "reg_c",
            "misfit",
            "unweighted_roughness",
            "normalized_log_misfit",
            "normalized_log_roughness",
            "curvature",
        ],
    )


def _selection_evidence(result: dict) -> dict:
    table = result["curvature_table"]
    log_misfit = [math.log10(float(row["misfit"])) for row in table]
    log_roughness = [
        math.log10(float(row["unweighted_roughness"])) for row in table
    ]
    return {
        **result,
        "normalization_bounds": {
            "log10_misfit": [min(log_misfit), max(log_misfit)],
            "log10_unweighted_roughness": [
                min(log_roughness),
                max(log_roughness),
            ],
        },
    }


def _monotonic_diagnostics(points: list[dict]) -> dict:
    eligible = valid_points(points)
    misfit_violations = []
    roughness_violations = []
    for lower, upper in zip(eligible[:-1], eligible[1:]):
        pair = [float(lower["reg_c"]), float(upper["reg_c"])]
        if float(upper["misfit"]) > float(lower["misfit"]):
            misfit_violations.append(pair)
        if float(upper["unweighted_roughness"]) < float(
            lower["unweighted_roughness"]
        ):
            roughness_violations.append(pair)
    return {
        "policy": "diagnostic_only_no_smoothing_or_point_removal",
        "expected_with_increasing_reg_c": {
            "misfit": "nonincreasing",
            "unweighted_roughness": "nondecreasing",
        },
        "misfit_violation_pairs": misfit_violations,
        "roughness_violation_pairs": roughness_violations,
    }


def _plot_lcurve(points: list[dict], selected_reg_c: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible = valid_points(points)
    figure, axis = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    axis.plot(
        [point["misfit"] for point in eligible],
        [point["unweighted_roughness"] for point in eligible],
        color="#3b6ea8",
        linewidth=1.5,
        zorder=1,
    )
    axis.scatter(
        [point["misfit"] for point in eligible],
        [point["unweighted_roughness"] for point in eligible],
        color="#3b6ea8",
        s=34,
        zorder=2,
        label="valid converged point",
    )
    selected = next(
        point for point in eligible if float(point["reg_c"]) == selected_reg_c
    )
    axis.scatter(
        [selected["misfit"]],
        [selected["unweighted_roughness"]],
        marker="*",
        color="#c43b3b",
        edgecolor="black",
        linewidth=0.6,
        s=190,
        zorder=3,
        label=f"selected reg_C={selected_reg_c:.6g}",
    )
    for point in eligible:
        axis.annotate(
            f"{float(point['reg_c']):.5g}",
            (point["misfit"], point["unweighted_roughness"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Observation-mean velocity misfit")
    axis.set_ylabel("Unweighted roughness")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=240)
    plt.close(figure)


def run_lcurve_study(
    *,
    config: dict,
    config_path: Path,
    repo_root: Path,
    output_dir: Path,
    preflight_class,
    resume: bool = False,
    point_launcher=None,
    confirmation_launcher=None,
    create_plot: bool = True,
    forward_smoke_dir: Path | None = None,
    require_forward_smoke: bool = True,
) -> dict:
    """Execute or resume the frozen sequential L-curve state machine."""
    output_dir = output_dir.resolve()
    workflow_root = Path(__file__).resolve().parent
    entrypoint = workflow_root / "production_amundsen.py"
    contract_path = output_dir / "run_contract.json"
    state_path = output_dir / "run_state.json"
    log_path = output_dir / "logs" / "orchestrator.log"
    point_root = output_dir / "points"
    launcher = point_launcher or _default_point_launcher
    confirm_launcher = confirmation_launcher or _default_confirmation_launcher

    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume missing study {output_dir}")
        contract = _read_json(contract_path)
        _validate_resume_contract(
            contract,
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            preflight_class=preflight_class,
            forward_smoke_dir=forward_smoke_dir,
            require_forward_smoke=require_forward_smoke,
        )
        state = _read_json(state_path)
        if state.get("schema") != STATE_SCHEMA:
            raise RuntimeError("Unexpected L-curve state schema.")
        if state.get("status") == "complete":
            if (output_dir / "study_manifest.json").is_file():
                return verify_completed_study(output_dir, contract)
            state["status"] = "running"
            state["phase"] = "recovering_finalization"
        if _orchestrator_process_is_active(state, output_dir):
            raise RuntimeError(
                "This L-curve study already has an active orchestrator process."
            )
        state["orchestrator_pid"] = os.getpid()
        state["updated_utc"] = _utc_now()
        _atomic_json(state_path, state)
    else:
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir}")
        contract = _build_contract(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            preflight_class=preflight_class,
            forward_smoke_dir=forward_smoke_dir,
            require_forward_smoke=require_forward_smoke,
        )
        output_dir.mkdir(parents=True)
        (output_dir / "logs").mkdir()
        point_root.mkdir()
        _atomic_json(contract_path, contract)
        shutil.copy2(config_path, output_dir / "resolved_config.json")
        _snapshot_sources(contract["source_sha256"], output_dir / "source_snapshot")
        state = {
            "schema": STATE_SCHEMA,
            "status": "running",
            "phase": "initial",
            "started_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "contract_id": contract["manifest_id"],
            "orchestrator_pid": os.getpid(),
            "requests": [],
            "confirmation_plans": [],
        }
        _atomic_json(state_path, state)

    state.setdefault("confirmation_plans", [])

    def save_state() -> None:
        state["updated_utc"] = _utc_now()
        _atomic_json(state_path, state)

    def ensure_point(reg_c: float, phase: str) -> tuple[dict, Path]:
        matching = [
            request
            for request in state["requests"]
            if request.get("run_kind", "independent") == "independent"
            and float(request["reg_c"]) == float(reg_c)
        ]
        for request in reversed(matching):
            point_dir = point_root / request["run_id"]
            manifest_path = point_dir / "point_manifest.json"
            if manifest_path.is_file():
                manifest = _read_json(manifest_path)
                validate_point_manifest(
                    manifest,
                    point_dir=point_dir,
                    reg_c=reg_c,
                    contract=contract,
                    expected_run_kind="independent",
                    study_root=output_dir,
                )
                request["status"] = "complete"
                request["point_status"] = manifest["status"]
                request["point_manifest_id"] = manifest["manifest_id"]
                save_state()
                return manifest, point_dir
            if request.get("status") == "running":
                if _point_process_is_active(point_dir, request["run_id"]):
                    raise RuntimeError(
                        f"L-curve point {request['run_id']} is still active; "
                        "wait for its manifest before resuming."
                    )
                request["status"] = "interrupted_without_manifest"
                request["finished_utc"] = _utc_now()
                save_state()

        attempt = len(matching) + 1
        run_id = f"{reg_c_slug(reg_c)}_attempt{attempt:02d}"
        request = {
            "run_kind": "independent",
            "reg_c": float(reg_c),
            "phase": phase,
            "attempt": attempt,
            "run_id": run_id,
            "status": "running",
            "started_utc": _utc_now(),
        }
        state["requests"].append(request)
        save_state()
        return_code = launcher(
            entrypoint=entrypoint,
            workflow_root=workflow_root,
            config_path=config_path,
            repo_root=repo_root,
            point_root=point_root,
            run_id=run_id,
            reg_c=reg_c,
            log_path=log_path,
        )
        request["return_code"] = int(return_code)
        request["finished_utc"] = _utc_now()
        point_dir = point_root / run_id
        manifest_path = point_dir / "point_manifest.json"
        if not manifest_path.is_file():
            request["status"] = "interrupted_without_manifest"
            save_state()
            raise RuntimeError(
                f"reg_C={reg_c} returned {return_code} without a point manifest; "
                "resume will preserve this attempt and use a new directory."
            )
        manifest = _read_json(manifest_path)
        validate_point_manifest(
            manifest,
            point_dir=point_dir,
            reg_c=reg_c,
            contract=contract,
            expected_run_kind="independent",
            study_root=output_dir,
        )
        request["status"] = "complete"
        request["point_status"] = manifest["status"]
        request["point_manifest_id"] = manifest["manifest_id"]
        save_state()
        return manifest, point_dir

    def ensure_confirmation(
        reg_c: float,
        *,
        confirmation_round: int,
        parent_manifest_path: Path,
    ) -> tuple[dict, Path]:
        parent_manifest_path = parent_manifest_path.resolve()
        parent_manifest = _read_json(parent_manifest_path)
        parent_manifest_id = parent_manifest.get("manifest_id")
        if canonical_identifier(parent_manifest) != parent_manifest_id:
            raise RuntimeError("Planned confirmation parent identifier is invalid.")
        parent_manifest_sha256 = sha256_file(parent_manifest_path)
        matching = [
            request
            for request in state["requests"]
            if request.get("run_kind") == "confirmation"
            and int(request.get("confirmation_round", 0))
            == int(confirmation_round)
            and float(request["reg_c"]) == float(reg_c)
            and request.get("parent_point_manifest_id") == parent_manifest_id
        ]
        for request in reversed(matching):
            point_dir = point_root / request["run_id"]
            manifest_path = point_dir / "point_manifest.json"
            if manifest_path.is_file():
                manifest = _read_json(manifest_path)
                validate_point_manifest(
                    manifest,
                    point_dir=point_dir,
                    reg_c=reg_c,
                    contract=contract,
                    expected_run_kind="confirmation",
                    expected_confirmation_round=confirmation_round,
                    expected_parent_manifest_id=parent_manifest_id,
                    study_root=output_dir,
                )
                request["status"] = "complete"
                request["point_status"] = manifest["status"]
                request["point_manifest_id"] = manifest["manifest_id"]
                save_state()
                return manifest, point_dir
            if request.get("status") == "running":
                if _point_process_is_active(point_dir, request["run_id"]):
                    raise RuntimeError(
                        f"L-curve confirmation {request['run_id']} is still active; "
                        "wait for its manifest before resuming."
                    )
                request["status"] = "interrupted_without_manifest"
                request["finished_utc"] = _utc_now()
                save_state()

        attempt = len(matching) + 1
        run_id = (
            f"confirm_r{int(confirmation_round):02d}_{reg_c_slug(reg_c)}_"
            f"attempt{attempt:02d}"
        )
        request = {
            "run_kind": "confirmation",
            "confirmation_round": int(confirmation_round),
            "reg_c": float(reg_c),
            "phase": f"confirmation_round_{int(confirmation_round):02d}",
            "parent_point_manifest_id": parent_manifest_id,
            "parent_point_manifest_sha256": parent_manifest_sha256,
            "parent_point_manifest_path": str(parent_manifest_path),
            "attempt": attempt,
            "run_id": run_id,
            "status": "running",
            "started_utc": _utc_now(),
        }
        state["requests"].append(request)
        save_state()
        return_code = confirm_launcher(
            entrypoint=entrypoint,
            workflow_root=workflow_root,
            config_path=config_path,
            repo_root=repo_root,
            point_root=point_root,
            run_id=run_id,
            reg_c=reg_c,
            parent_point_manifest_path=parent_manifest_path,
            confirmation_round=confirmation_round,
            log_path=log_path,
        )
        request["return_code"] = int(return_code)
        request["finished_utc"] = _utc_now()
        point_dir = point_root / run_id
        manifest_path = point_dir / "point_manifest.json"
        if not manifest_path.is_file():
            request["status"] = "interrupted_without_manifest"
            save_state()
            raise RuntimeError(
                f"confirmation round {confirmation_round}, reg_C={reg_c} returned "
                f"{return_code} without a point manifest; resume will preserve "
                "this attempt and use a new directory."
            )
        manifest = _read_json(manifest_path)
        validate_point_manifest(
            manifest,
            point_dir=point_dir,
            reg_c=reg_c,
            contract=contract,
            expected_run_kind="confirmation",
            expected_confirmation_round=confirmation_round,
            expected_parent_manifest_id=parent_manifest_id,
            study_root=output_dir,
        )
        request["status"] = "complete"
        request["point_status"] = manifest["status"]
        request["point_manifest_id"] = manifest["manifest_id"]
        save_state()
        return manifest, point_dir

    summaries: dict[float, dict] = {}
    phases: dict[float, str] = {}

    def run_stage(values: list[float], phase: str) -> None:
        state["phase"] = phase
        save_state()
        for value in values:
            manifest, point_dir = ensure_point(value, phase)
            summaries[float(value)] = point_summary(manifest, point_dir, output_dir)
            phases[float(value)] = phase

    inversion = config["inversion"]
    base_values = _point_values(config, "lcurve_base_reg_c")
    run_stage(base_values, "base")
    base_points = [summaries[value] for value in base_values]
    if len(valid_points(base_points)) != len(base_values):
        raise RuntimeError(
            "Every formal base-grid point must satisfy the common v2 "
            "convergence/stability gate before adaptive selection."
        )
    base_selection = select_corner(base_points)
    side = extension_side(base_points)
    base_ambiguity = curvature_ambiguity(
        base_selection,
        ratio_threshold=float(inversion["lcurve_curvature_ambiguity_ratio"]),
    )
    _write_curvature(output_dir / "curvature_base.csv", base_selection)
    _atomic_json(
        output_dir / "selection_base.json",
        {
            "extension_side": side,
            "ambiguity": base_ambiguity,
            **_selection_evidence(base_selection),
        },
    )

    extension_values = []
    if side == "low":
        extension_values = [float(inversion["lcurve_low_extension_reg_c"])]
    elif side == "high":
        extension_values = [float(inversion["lcurve_high_extension_reg_c"])]
    if extension_values:
        run_stage(extension_values, f"{side}_boundary_extension")
        invalid_extensions = [
            value
            for value in extension_values
            if summaries[value]["status"] != "valid"
            or not valid_points([summaries[value]])
        ]
        if invalid_extensions:
            raise RuntimeError(
                "The single permitted boundary extension is invalid: "
                f"{invalid_extensions}"
            )

    post_extension_points = list(summaries.values())
    post_extension_selection = select_corner(post_extension_points)
    post_extension_ambiguity = curvature_ambiguity(
        post_extension_selection,
        ratio_threshold=float(inversion["lcurve_curvature_ambiguity_ratio"]),
    )
    _write_curvature(
        output_dir / "curvature_post_extension.csv", post_extension_selection
    )
    refinement_values = []
    refinement = None
    if (
        post_extension_ambiguity["is_ambiguous"]
        and int(inversion["lcurve_max_refinement_rounds"]) > 0
    ):
        refinement = geometric_refinements(post_extension_points)
        refinement_values = [
            float(value) for value in refinement["refinement_reg_c"]
        ]
        if len(summaries) + len(refinement_values) > int(
            inversion["lcurve_max_formal_points"]
        ):
            raise RuntimeError("The frozen adaptive L-curve point budget was exceeded.")
    _atomic_json(
        output_dir / "selection_post_extension.json",
        {
            "extension_side": side,
            "extension_reg_c": extension_values,
            "ambiguity": post_extension_ambiguity,
            "refinement": refinement,
            "selection_evidence": _selection_evidence(post_extension_selection),
        },
    )

    if refinement_values:
        run_stage(refinement_values, "ambiguity_midpoint_refinement")
        invalid_refinements = [
            value
            for value in refinement_values
            if summaries[value]["status"] != "valid"
            or not valid_points([summaries[value]])
        ]
        if invalid_refinements:
            raise RuntimeError(
                "The one permitted local refinement is incomplete; invalid "
                f"midpoint reg_C values: {invalid_refinements}"
            )
    formal_point_count = len(summaries)
    if formal_point_count > int(inversion["lcurve_max_formal_points"]):
        raise RuntimeError("The frozen adaptive L-curve point budget was exceeded.")

    formal_references = []
    for value, summary in sorted(summaries.items()):
        path = (output_dir / summary["point_manifest_path"]).resolve()
        formal_references.append(
            {
                "reg_c": float(value),
                "phase": phases[float(value)],
                "status": summary["status"],
                "path": str(path.relative_to(output_dir)),
                "manifest_id": summary["point_manifest_id"],
                "manifest_sha256": sha256_file(path),
            }
        )

    lineage_tips = {
        value: (output_dir / summary["point_manifest_path"]).resolve()
        for value, summary in summaries.items()
    }
    confirmation_rounds = []
    confirmation_references = []
    requested_neighborhood = corner_neighborhood(list(summaries.values()))
    requested_triple = [
        float(value) for value in requested_neighborhood["confirmation_reg_c"]
    ]
    resolved_confirmation = False
    maximum_confirmation_rounds = int(
        inversion["lcurve_max_confirmation_rounds"]
    )

    for round_number in range(1, maximum_confirmation_rounds + 1):
        state["phase"] = f"confirmation_round_{round_number:02d}"
        parent_records = []
        for value in requested_triple:
            parent_path = lineage_tips[value]
            parent_manifest = _read_json(parent_path)
            parent_records.append(
                {
                    "reg_c": value,
                    "path": str(parent_path.relative_to(output_dir)),
                    "manifest_id": parent_manifest["manifest_id"],
                    "manifest_sha256": sha256_file(parent_path),
                }
            )
        plan = {
            "round": round_number,
            "confirmation_reg_c": requested_triple,
            "parent_manifests": parent_records,
        }
        prior_plans = [
            item
            for item in state["confirmation_plans"]
            if int(item["round"]) == round_number
        ]
        if prior_plans:
            if len(prior_plans) != 1 or prior_plans[0] != plan:
                raise RuntimeError(
                    "Saved confirmation plan disagrees with deterministic resume."
                )
        else:
            state["confirmation_plans"].append(plan)
        save_state()

        selection_before = select_corner(list(summaries.values()))
        candidate_before = float(selection_before["selected"]["reg_c"])
        round_manifests = {}
        round_summaries = {}
        for value in requested_triple:
            parent_path = lineage_tips[value]
            child, child_dir = ensure_confirmation(
                value,
                confirmation_round=round_number,
                parent_manifest_path=parent_path,
            )
            child_path = child_dir / "point_manifest.json"
            lineage_tips[value] = child_path.resolve()
            round_manifests[value] = child
            round_summaries[value] = point_summary(child, child_dir, output_dir)
            reference = {
                "round": round_number,
                "reg_c": value,
                "status": child["status"],
                "path": str(child_path.relative_to(output_dir)),
                "manifest_id": child["manifest_id"],
                "manifest_sha256": sha256_file(child_path),
                "parent_manifest_id": (
                    child.get("parent_point_manifest") or {}
                ).get("manifest_id"),
            }
            confirmation_references.append(reference)

        round_passed = all(
            manifest["status"] == "valid"
            and bool(valid_points([round_summaries[value]]))
            for value, manifest in round_manifests.items()
        )
        candidate_after = None
        triple_after = None
        if round_passed:
            for value in requested_triple:
                summaries[value] = round_summaries[value]
            neighborhood_after = corner_neighborhood(list(summaries.values()))
            candidate_after = float(neighborhood_after["candidate_reg_c"])
            triple_after = [
                float(value)
                for value in neighborhood_after["confirmation_reg_c"]
            ]

        round_record = {
            "round": round_number,
            "candidate_reg_c_before": candidate_before,
            "confirmation_reg_c": requested_triple,
            "parent_manifest_ids": {
                format(row["reg_c"], ".17g"): row["manifest_id"]
                for row in parent_records
            },
            "confirmation_manifest_ids": {
                format(value, ".17g"): round_manifests[value]["manifest_id"]
                for value in requested_triple
            },
            "statuses": {
                format(value, ".17g"): round_manifests[value]["status"]
                for value in requested_triple
            },
            "passed": round_passed,
            "candidate_reg_c_after": candidate_after,
            "confirmation_reg_c_after": triple_after,
            "triple_stable": bool(
                round_passed and triple_after == requested_triple
            ),
        }
        confirmation_rounds.append(round_record)
        _atomic_json(
            output_dir / f"confirmation_round_{round_number:02d}.json",
            round_record,
        )
        state["confirmation_rounds"] = confirmation_rounds
        save_state()

        if round_passed and triple_after == requested_triple:
            resolved_confirmation = True
            break
        if round_number == maximum_confirmation_rounds:
            raise RuntimeError(
                "confirmation stability remains unresolved after the frozen "
                "two-round limit; no automatic L-curve corner is allowed."
            )
        if round_passed:
            requested_triple = triple_after
        # A retryable-invalid round advances every same-reg_C lineage by one
        # block, but does not update the curve; repeat the same triple.

    if not resolved_confirmation:
        raise RuntimeError("confirmation stability did not resolve the L-curve corner.")

    all_points = list(summaries.values())
    final_selection = select_corner(all_points)
    selected_reg_c = float(final_selection["selected"]["reg_c"])
    selected_point = summaries[selected_reg_c]
    eligible_values = [float(point["reg_c"]) for point in valid_points(all_points)]
    selected_index = eligible_values.index(selected_reg_c)
    if not 0 < selected_index < len(eligible_values) - 1:
        raise RuntimeError("Final L-curve corner lacks valid neighbors on both sides.")
    if (
        selected_point["status"] != "valid"
        or selected_point.get("run_kind") != "confirmation"
    ):
        raise RuntimeError(
            "The final L-curve corner is not a valid confirmed production point."
        )

    _write_curvature(output_dir / "curvature_final.csv", final_selection)
    point_rows = _stage_rows(all_points, phases)
    _write_csv(
        output_dir / "lcurve_points.csv",
        point_rows,
        [
            "phase",
            "reg_c",
            "status",
            "run_kind",
            "confirmation_round",
            "termination",
            "native_termination",
            "acceptance_basis",
            "block_count",
            "stability_passed",
            "misfit",
            "unweighted_roughness",
            "weighted_penalty",
            "objective",
            "point_manifest_id",
            "point_manifest_path",
        ],
    )
    _atomic_json(
        output_dir / "selection_final.json",
        {
            "selected": final_selection["selected"],
            "valid_neighbor_reg_c": [
                eligible_values[selected_index - 1],
                eligible_values[selected_index + 1],
            ],
            "curvature_table": final_selection["curvature_table"],
            "normalization_bounds": _selection_evidence(final_selection)[
                "normalization_bounds"
            ],
            "monotonic_diagnostics": _monotonic_diagnostics(all_points),
            "confirmation_rounds": confirmation_rounds,
        },
    )
    if create_plot:
        _plot_lcurve(all_points, selected_reg_c, output_dir / "lcurve_appendix.png")

    selected_manifest_path = output_dir / selected_point["point_manifest_path"]
    definitive = {
        "schema": "jog-definitive-inversion-reference-v1",
        "policy": (
            "The selected, same-reg_C-confirmed L-curve endpoint is the "
            "definitive whole-sector reference inversion; no redundant rerun "
            "is required."
        ),
        "reg_c": selected_reg_c,
        "point_manifest_id": selected_point["point_manifest_id"],
        "point_manifest_path": selected_point["point_manifest_path"],
        "point_manifest_sha256": sha256_file(selected_manifest_path),
    }
    definitive["manifest_id"] = canonical_identifier(definitive)
    _atomic_json(output_dir / "definitive_inversion.json", definitive)

    state["status"] = "complete"
    state["phase"] = "complete"
    state["selected_reg_c"] = selected_reg_c
    state["finished_utc"] = _utc_now()
    save_state()

    parent_hashes = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or point_root in path.parents:
            continue
        if path.name == "study_manifest.json":
            continue
        parent_hashes[str(path.relative_to(output_dir))] = sha256_file(path)
    invalid_points = [point for point in point_rows if point["status"] != "valid"]
    if invalid_points:
        raise RuntimeError("Resolved effective L-curve unexpectedly contains invalid rows.")
    effective_manifest_ids = {
        format(float(point["reg_c"]), ".17g"): point["point_manifest_id"]
        for point in point_rows
    }
    manifest = {
        "schema": STUDY_SCHEMA,
        "status": "complete",
        "started_utc": state["started_utc"],
        "finished_utc": state["finished_utc"],
        "contract_id": contract["manifest_id"],
        "config_sha256": contract["config_sha256"],
        "source_sha256": contract["source_sha256"],
        "stable_environment": contract["stable_environment"],
        "base_reg_c": base_values,
        "extension_side": side,
        "extension_reg_c": extension_values,
        "refinement_reg_c": refinement_values,
        "formal_point_count": formal_point_count,
        "formal_manifest_count": len(formal_references),
        "formal_point_manifests": formal_references,
        "point_count": len(point_rows),
        "valid_point_count": len(point_rows),
        "invalid_points": [],
        "confirmation_rounds": confirmation_rounds,
        "confirmation_manifest_count": len(confirmation_references),
        "confirmation_manifests": confirmation_references,
        "confirmation_manifest_ids": [
            reference["manifest_id"] for reference in confirmation_references
        ],
        "selected_reg_c": selected_reg_c,
        "definitive_inversion_manifest_id": definitive["manifest_id"],
        "point_manifest_ids": effective_manifest_ids,
        "effective_point_manifest_ids": effective_manifest_ids,
        "output_sha256": parent_hashes,
    }
    manifest["manifest_id"] = canonical_identifier(manifest)
    _atomic_json(output_dir / "study_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_reg_c": selected_reg_c,
                "manifest_id": manifest["manifest_id"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return manifest
