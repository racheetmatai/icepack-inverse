#!/usr/bin/env python3
"""Independently verify a completed revised L-curve study evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def canonical_identifier(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_id", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_identifier(path: Path, payload: dict) -> None:
    observed = canonical_identifier(payload)
    if observed != payload.get("manifest_id"):
        raise RuntimeError(f"Invalid manifest identifier in {path}")


def verify_hashes(root: Path, hashes: dict) -> int:
    count = 0
    resolved_root = root.resolve()
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(f"Hash target escapes {root}: {relative}") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Missing or hash-mismatched output: {path}")
        count += 1
    return count


def verify_study(root: Path) -> dict:
    root = root.resolve()
    study_path = root / "study_manifest.json"
    study = read_json(study_path)
    verify_identifier(study_path, study)
    if study.get("schema") != "jog-production-lcurve-study-v2":
        raise RuntimeError("Unexpected L-curve study schema.")
    if study.get("status") != "complete":
        raise RuntimeError("L-curve study is not complete.")
    parent_hash_count = verify_hashes(root, study.get("output_sha256", {}))

    contract_path = root / "run_contract.json"
    contract = read_json(contract_path)
    verify_identifier(contract_path, contract)
    if contract.get("schema") != "jog-production-lcurve-contract-v2":
        raise RuntimeError("Unexpected L-curve contract schema.")
    if contract["manifest_id"] != study.get("contract_id"):
        raise RuntimeError("Study references a different run contract.")

    rows = list(
        csv.DictReader((root / "lcurve_points.csv").open(encoding="utf-8"))
    )
    if len(rows) != study.get("point_count") or len(rows) != study.get(
        "formal_point_count"
    ):
        raise RuntimeError("Effective L-curve table has the wrong row count.")
    point_hash_count = 0
    for row in rows:
        point_path = (root / row["point_manifest_path"]).resolve()
        try:
            point_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("Effective point path escapes the study.") from error
        point = read_json(point_path)
        verify_identifier(point_path, point)
        if point.get("schema") != "jog-production-lcurve-point-v2":
            raise RuntimeError(f"Unexpected point schema at {point_path}")
        if point["manifest_id"] != row["point_manifest_id"]:
            raise RuntimeError(f"Point table ID mismatch at reg_C={row['reg_c']}")
        expected_id = study["point_manifest_ids"].get(
            format(float(row["reg_c"]), ".17g")
        )
        if point["manifest_id"] != expected_id:
            raise RuntimeError(f"Study point ID mismatch at reg_C={row['reg_c']}")
        point_hash_count += verify_hashes(
            point_path.parent, point.get("output_sha256", {})
        )

    effective_ids = study.get("effective_point_manifest_ids") or {}
    if effective_ids != study.get("point_manifest_ids"):
        raise RuntimeError("Effective point ID inventories disagree.")
    formal_references = study.get("formal_point_manifests") or []
    if len(formal_references) != study.get("formal_manifest_count") or len(
        formal_references
    ) != study.get("formal_point_count"):
        raise RuntimeError("Formal point inventory is inconsistent.")
    known_lineage_ids = set()
    latest_lineage_by_reg = {}
    for reference in formal_references:
        point_path = (root / reference["path"]).resolve()
        try:
            point_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("Formal point path escapes the study.") from error
        if sha256_file(point_path) != reference.get("manifest_sha256"):
            raise RuntimeError("Formal point manifest hash mismatch.")
        point = read_json(point_path)
        verify_identifier(point_path, point)
        if (
            point.get("schema") != "jog-production-lcurve-point-v2"
            or point.get("run_kind") != "independent"
            or point.get("manifest_id") != reference.get("manifest_id")
            or point.get("status") != reference.get("status")
            or point.get("status") != "valid"
            or float(point.get("reg_c")) != float(reference["reg_c"])
        ):
            raise RuntimeError("Formal point inventory is inconsistent.")
        verify_hashes(point_path.parent, point.get("output_sha256", {}))
        if point["manifest_id"] in known_lineage_ids:
            raise RuntimeError("Formal point inventory contains a duplicate.")
        known_lineage_ids.add(point["manifest_id"])
        latest_lineage_by_reg[float(reference["reg_c"])] = point["manifest_id"]
    confirmation_references = study.get("confirmation_manifests") or []
    if len(confirmation_references) != study.get("confirmation_manifest_count"):
        raise RuntimeError("Confirmation manifest count is inconsistent.")
    confirmation_ids = []
    confirmation_output_hash_count = 0
    for reference in confirmation_references:
        point_path = (root / reference["path"]).resolve()
        try:
            point_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("Confirmation path escapes the study.") from error
        if sha256_file(point_path) != reference.get("manifest_sha256"):
            raise RuntimeError("Confirmation manifest hash mismatch.")
        point = read_json(point_path)
        verify_identifier(point_path, point)
        if (
            point.get("schema") != "jog-production-lcurve-point-v2"
            or point.get("run_kind") != "confirmation"
            or point.get("manifest_id") != reference.get("manifest_id")
            or point.get("status") != reference.get("status")
            or int(point.get("confirmation_round", 0)) != int(reference["round"])
            or float(point.get("reg_c")) != float(reference["reg_c"])
        ):
            raise RuntimeError("Confirmation inventory is inconsistent.")
        parent = point.get("parent_point_manifest") or {}
        portable = parent.get("path_relative_to_child")
        parent_path = (
            (point_path.parent / portable).resolve()
            if portable
            else Path(parent.get("path", "")).resolve()
        )
        try:
            parent_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("Confirmation parent escapes the study.") from error
        if (
            not parent_path.is_file()
            or sha256_file(parent_path) != parent.get("manifest_sha256")
        ):
            raise RuntimeError("Confirmation parent hash mismatch.")
        parent_manifest = read_json(parent_path)
        verify_identifier(parent_path, parent_manifest)
        if parent_manifest.get("manifest_id") != parent.get("manifest_id"):
            raise RuntimeError("Confirmation parent ID mismatch.")
        if parent_manifest.get("manifest_id") not in known_lineage_ids:
            raise RuntimeError("Confirmation parent is absent from prior lineage.")
        reg_c = float(reference["reg_c"])
        if latest_lineage_by_reg.get(reg_c) != parent_manifest.get("manifest_id"):
            raise RuntimeError("Confirmation does not continue the latest lineage.")
        confirmation_output_hash_count += verify_hashes(
            point_path.parent, point.get("output_sha256", {})
        )
        confirmation_ids.append(point["manifest_id"])
        known_lineage_ids.add(point["manifest_id"])
        latest_lineage_by_reg[reg_c] = point["manifest_id"]
    if confirmation_ids != study.get("confirmation_manifest_ids"):
        raise RuntimeError("Confirmation ID inventory is inconsistent.")
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
            raise RuntimeError("Confirmation triple inventory is inconsistent.")
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
            raise RuntimeError("Confirmation round evidence is inconsistent.")
        passed = all(status == "valid" for status in expected_statuses.values())
        if record.get("passed") != passed:
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
                raise RuntimeError("Confirmation retry triple is inconsistent.")
    final_round = round_records[-1]
    if not (
        final_round.get("passed") is True
        and final_round.get("triple_stable") is True
        and float(final_round.get("candidate_reg_c_after"))
        == float(study.get("selected_reg_c"))
    ):
        raise RuntimeError("Final confirmation round is unresolved.")

    definitive_path = root / "definitive_inversion.json"
    definitive = read_json(definitive_path)
    verify_identifier(definitive_path, definitive)
    selected_path = (root / definitive["point_manifest_path"]).resolve()
    try:
        selected_path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("Definitive point path escapes the study.") from error
    selected = read_json(selected_path)
    if selected["manifest_id"] != definitive["point_manifest_id"]:
        raise RuntimeError("Definitive reference points to a different point ID.")
    if sha256_file(selected_path) != definitive["point_manifest_sha256"]:
        raise RuntimeError("Definitive point manifest hash does not match.")
    if definitive["manifest_id"] != study["definitive_inversion_manifest_id"]:
        raise RuntimeError("Study references a different definitive inversion.")
    if selected.get("run_kind") != "confirmation" or selected.get("status") != "valid":
        raise RuntimeError("Definitive inversion is not a valid confirmation.")
    selected_key = format(float(definitive["reg_c"]), ".17g")
    if effective_ids.get(selected_key) != selected.get("manifest_id"):
        raise RuntimeError("Definitive inversion is not the effective selected row.")

    return {
        "study_manifest_id": study["manifest_id"],
        "contract_id": contract["manifest_id"],
        "point_count": len(rows),
        "parent_hash_count": parent_hash_count,
        "point_output_hash_count": point_hash_count,
        "confirmation_manifest_count": len(confirmation_references),
        "confirmation_output_hash_count": confirmation_output_hash_count,
        "selected_reg_c": study["selected_reg_c"],
        "status": "verified",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_study(args.study_directory), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
