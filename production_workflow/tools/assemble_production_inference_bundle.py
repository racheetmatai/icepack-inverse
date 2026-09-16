"""Assemble the verified production checkpoints needed for full-mesh inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = (
    "best_model.keras",
    "scaler_parameters.json",
    "resolved_spec.json",
    "data_identity.json",
    "run_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload)
    body.pop("manifest_id", None)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def assemble(campaign_root: Path, registry_path: Path, output: Path) -> dict:
    campaign_root = campaign_root.resolve()
    registry_path = registry_path.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    completion_path = campaign_root / "production_completion_audit.json"
    completion = read_json(completion_path)
    if not completion.get("passed") or completion.get("counts", {}).get("verified_completed_runs") != 660:
        raise ValueError("Campaign completion audit is not the accepted 660-run audit")
    with registry_path.open(newline="", encoding="utf-8") as stream:
        registry = list(csv.DictReader(stream))
    if len(registry) != 660 or len({row["job_id"] for row in registry}) != 660:
        raise ValueError("Frozen job registry must contain exactly 660 unique jobs")

    audit_by_job = {item["job_id"]: item for item in completion["runs"]}
    if set(audit_by_job) != {row["job_id"] for row in registry}:
        raise ValueError("Completion audit and frozen job registry disagree")

    jobs = []
    methods: set[str] = set()
    for index, row in enumerate(sorted(registry, key=lambda item: item["job_id"]), start=1):
        job_id = row["job_id"]
        source_dir = campaign_root / "runs" / job_id
        run_manifest = read_json(source_dir / "run_manifest.json")
        if run_manifest.get("status") != "complete" or run_manifest.get("job_id") != job_id:
            raise ValueError(f"Invalid run manifest for {job_id}")
        if canonical_id(run_manifest) != run_manifest.get("manifest_id"):
            raise ValueError(f"Canonical run-manifest ID mismatch for {job_id}")
        audit = audit_by_job[job_id]
        if audit.get("run_manifest_id") != run_manifest["manifest_id"]:
            raise ValueError(f"Completion-audit manifest mismatch for {job_id}")
        spec = read_json(source_dir / "resolved_spec.json")
        identity = read_json(source_dir / "data_identity.json")
        expected_job = identity.get("job", {})
        for name in ("experiment", "configuration", "member"):
            if str(expected_job.get(name)) != str(row[name]):
                raise ValueError(f"Registry/data identity disagreement for {job_id}: {name}")
        if float(spec.get("lambda_L2", -1)) != 0.0 or run_manifest.get("lambda_L2") != 0.0:
            raise ValueError(f"Production lambda_L2 is not frozen at zero for {job_id}")

        files: dict[str, str] = {}
        for name in REQUIRED_FILES:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            observed = sha256_file(source)
            if name == "run_manifest.json":
                expected = sha256_file(source)
            else:
                expected = run_manifest["output_sha256"].get(name)
            if observed != expected:
                raise ValueError(f"Declared SHA256 mismatch for {job_id}/{name}")
            destination = output / "runs" / job_id / name
            methods.add(link_or_copy(source, destination))
            files[f"runs/{job_id}/{name}"] = observed
        jobs.append({
            "job_id": job_id,
            "experiment": row["experiment"],
            "configuration": row["configuration"],
            "member": int(row["member"]),
            "features": spec["features"],
            "lambda_L2": 0.0,
            "run_manifest_id": run_manifest["manifest_id"],
            "files": files,
        })
        if index % 50 == 0 or index == 660:
            print(f"Assembled and verified {index}/660 inference checkpoints", flush=True)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for job in jobs:
        grouped.setdefault((job["experiment"], job["configuration"]), []).append(job)
    if len(grouped) != 66:
        raise ValueError(f"Expected 66 ensembles; found {len(grouped)}")
    ensembles = []
    for (experiment, configuration), members in sorted(grouped.items()):
        members.sort(key=lambda item: item["member"])
        if [item["member"] for item in members] != list(range(1, 11)):
            raise ValueError(f"Ensemble membership is incomplete for {experiment}/{configuration}")
        feature_sets = {tuple(item["features"]) for item in members}
        if len(feature_sets) != 1:
            raise ValueError(f"Feature specification differs within {experiment}/{configuration}")
        ensembles.append({
            "ensemble_id": f"{experiment}_{configuration}",
            "experiment": experiment,
            "configuration": configuration,
            "features": list(next(iter(feature_sets))),
            "member_job_ids": [item["job_id"] for item in members],
        })

    index_path = output / "ensemble_registry.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "ensemble_id", "experiment", "configuration", "member_count",
            "member_job_ids",
        ))
        writer.writeheader()
        for item in ensembles:
            writer.writerow({
                "ensemble_id": item["ensemble_id"],
                "experiment": item["experiment"],
                "configuration": item["configuration"],
                "member_count": 10,
                "member_job_ids": ";".join(item["member_job_ids"]),
            })

    manifest = {
        "schema": "jog-production-inference-bundle-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "selection-free full-mesh inference for 660 frozen production MLPs and 66 ten-member medians",
        "source_campaign": {
            "campaign_id": completion["campaign_id"],
            "completion_audit_manifest_id": completion["manifest_id"],
            "completion_audit_sha256": sha256_file(completion_path),
            "production_package_manifest_id": completion["production_package_manifest_id"],
        },
        "frozen_rules": {
            "target": "reference_log_C",
            "lambda_L2": 0.0,
            "members_per_ensemble": 10,
            "combination": "vertex-wise median of ten member log_C controls",
            "replacement": "eligible grounded CG2 DOFs only; definitive inversion log_C retained elsewhere",
            "scaling": "JSON RobustScaler parameters: (x-center)/scale and inverse target transform",
            "checkpoint_loading": "rebuild frozen named architecture and load weights from best_model.keras",
            "prediction_clipping": "none",
        },
        "counts": {"jobs": len(jobs), "ensembles": len(ensembles)},
        "assembly_methods": sorted(methods),
        "ensemble_registry": {
            "path": index_path.name,
            "sha256": sha256_file(index_path),
        },
        "jobs": jobs,
        "ensembles": ensembles,
    }
    manifest["manifest_id"] = canonical_id(manifest)
    (output / "inference_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete", "jobs": len(jobs), "ensembles": len(ensembles),
        "manifest_id": manifest["manifest_id"], "output": str(output),
    }, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--job-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assemble(args.campaign_root, args.job_registry, args.output)


if __name__ == "__main__":
    main()
