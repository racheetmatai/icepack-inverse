"""Finalize a scientifically complete canonical export after provenance-only failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sys


REQUIRED_DATA = {
    "canonical_master_dataset.csv.gz",
    "common_eligible_row_ids.txt.gz",
    "attrition.csv",
    "field_summary.csv",
    "schema.json",
    "diagnostics.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_id", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adoption-record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    workflow = repo_root / "production_workflow"
    staging = args.output_root.resolve() / f".{args.run_id}.incomplete"
    final = args.output_root.resolve() / args.run_id
    if not staging.is_dir() or final.exists():
        raise RuntimeError("Expected one incomplete staging directory and no final directory.")
    present = {path.name for path in staging.iterdir() if path.is_file()}
    if present != REQUIRED_DATA:
        raise RuntimeError(f"Unexpected staging files: {sorted(present)}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    diagnostics = json.loads((staging / "diagnostics.json").read_text(encoding="utf-8"))
    schema = json.loads((staging / "schema.json").read_text(encoding="utf-8"))
    if not (
        diagnostics["selected_observations"] == 1_622_598
        and diagnostics["common_eligible"] == 1_530_992
        and schema["training_target"] == "reference_log_C"
        and len(schema["predictors_in_order"]) == 12
    ):
        raise RuntimeError("Staged scientific outputs do not match the frozen counts/schema.")

    provenance = staging / "provenance"
    if any(provenance.iterdir()):
        raise RuntimeError("Provenance directory is unexpectedly nonempty.")
    partition_audit_path = workflow / config["frozen_design"]["five_region_support"]["path"]
    partition_audit = json.loads(partition_audit_path.read_text(encoding="utf-8"))
    copied = {
        "adoption_record.json": args.adoption_record.resolve(),
        "amundsen_production_config.json": args.config.resolve(),
        "selected_squares.csv": workflow / config["frozen_design"]["selected_squares"]["path"],
        "five_region_partition_5km.npz": workflow / config["frozen_design"]["five_region_partition"]["path"],
        "five_region_partition_and_support.json": partition_audit_path,
        "pig_region.msh": Path(partition_audit["meshes"]["PIG"]["path"]),
        "thwaites_region.msh": Path(partition_audit["meshes"]["Thwaites"]["path"]),
        "dotson_region.msh": Path(partition_audit["meshes"]["Dotson"]["path"]),
    }
    for destination, source in copied.items():
        shutil.copy2(source, provenance / destination)
    source_snapshot = provenance / "source_snapshot"
    source_snapshot.mkdir()
    sources = [
        workflow / "export_canonical_dataset.py",
        Path(__file__).resolve(),
        workflow / "load_definitive_inversion.py",
        workflow / "tools" / "verify_definitive_inversion_adoption.py",
        workflow / "tools" / "verify_lcurve_selection_bundle.py",
        workflow / "production_amundsen.py",
        repo_root / "src" / "invert_c_theta.py",
        repo_root / "src" / "data_preprocessing.py",
        repo_root / "src" / "feature_units.py",
        repo_root / "src" / "revised_raster_inputs.py",
    ]
    for source in sources:
        shutil.copy2(source, source_snapshot / source.name)

    output_hashes = {
        str(path.relative_to(staging)): sha256_file(path)
        for path in sorted(staging.rglob("*")) if path.is_file()
    }
    adoption = json.loads(args.adoption_record.read_text(encoding="utf-8"))
    started = datetime.fromtimestamp(
        (staging / "canonical_master_dataset.csv.gz").stat().st_mtime,
        tz=timezone.utc,
    ).isoformat()
    manifest = {
        "schema": "jog-canonical-master-dataset-v1",
        "status": "complete",
        "run_id": args.run_id,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "reg_c": float(adoption["reg_c"]),
        "adoption_manifest_id": adoption["manifest_id"],
        "selected_point_manifest_id": adoption["point"]["manifest_id"],
        "config_sha256": sha256_file(args.config),
        "row_count": int(diagnostics["selected_observations"]),
        "common_eligible_count": int(diagnostics["common_eligible"]),
        "predictors": schema["predictors_in_order"],
        "training_target": schema["training_target"],
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "recovery": {
            "recovered_from_provenance_only_failure": True,
            "failure": "NameError after all scientific tables were written and before provenance copying or manifest creation",
            "scientific_outputs_recomputed": False,
            "finalizer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "diagnostics": diagnostics,
        "output_sha256": output_hashes,
    }
    manifest["manifest_id"] = identifier(manifest)
    (staging / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, final)
    print(json.dumps({"status": "complete", "manifest_id": manifest["manifest_id"], "output_directory": str(final)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
