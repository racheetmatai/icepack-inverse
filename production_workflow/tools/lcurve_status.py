#!/usr/bin/env python3
"""Print a compact, read-only status summary for a revised L-curve study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("study_directory", type=Path)
args = parser.parse_args()
root = args.study_directory.resolve()
state_path = root / "run_state.json"
if not state_path.is_file():
    raise SystemExit(f"No run_state.json yet under {root}")
state = read_json(state_path)
requests = state.get("requests", [])
summary = {
    "study_directory": str(root),
    "status": state.get("status"),
    "phase": state.get("phase"),
    "started_utc": state.get("started_utc"),
    "updated_utc": state.get("updated_utc"),
    "request_count": len(requests),
    "independent_request_count": sum(
        request.get("run_kind", "independent") == "independent"
        for request in requests
    ),
    "confirmation_request_count": sum(
        request.get("run_kind") == "confirmation" for request in requests
    ),
    "complete_request_count": sum(
        request.get("status") == "complete" for request in requests
    ),
    "active_requests": [
        {
            key: request.get(key)
            for key in (
                "run_kind",
                "confirmation_round",
                "reg_c",
                "phase",
                "attempt",
                "run_id",
                "started_utc",
            )
        }
        for request in requests
        if request.get("status") == "running"
    ],
    "point_results": [],
}
for request in requests:
    manifest_path = root / "points" / request["run_id"] / "point_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        summary["point_results"].append(
            {
                "reg_c": manifest.get("reg_c"),
                "run_kind": manifest.get("run_kind"),
                "confirmation_round": manifest.get("confirmation_round"),
                "status": manifest.get("status"),
                "native_termination": manifest.get("native_termination"),
                "acceptance_basis": manifest.get("acceptance_basis"),
                "block_count": manifest.get(
                    "cumulative_block_count", len(manifest.get("blocks", []))
                ),
                "stability_passed": (
                    manifest.get("stability") or {}
                ).get("passed"),
                "misfit": (manifest.get("metrics") or {}).get("misfit"),
                "unweighted_roughness": (
                    manifest.get("metrics") or {}
                ).get("unweighted_roughness"),
                "run_id": request["run_id"],
            }
        )
study_path = root / "study_manifest.json"
if study_path.is_file():
    study = read_json(study_path)
    summary["selected_reg_c"] = study.get("selected_reg_c")
    summary["study_manifest_id"] = study.get("manifest_id")
    summary["formal_point_count"] = study.get("formal_point_count")
    summary["confirmation_manifest_count"] = study.get(
        "confirmation_manifest_count"
    )
print(json.dumps(summary, indent=2, sort_keys=True))
