"""Independent audit for the completed forward-evaluation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path) -> dict:
    root = root.resolve(); manifest = read_json(root / "evaluation_manifest.json")
    checks = {
        "schema": manifest.get("schema") == "jog-forward-evaluation-bundle-v1",
        "status": manifest.get("status") == "complete",
        "manifest_id": canonical_id(manifest) == manifest.get("manifest_id"),
        "declared_control_count": manifest.get("evaluated_controls") == 726,
        "declared_median_maps": manifest.get("median_map_archives") == 66,
    }
    actual = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(p for p in root.rglob("*") if p.is_file()
                           and p.name not in {"evaluation_manifest.json", "verification_manifest.json", "host_verification_manifest.json"})
    }
    checks["exact_output_inventory_and_hashes"] = actual == manifest.get("output_sha256")
    result_paths = sorted((root / "control_metrics").glob("*.json"))
    map_paths = sorted((root / "median_map_data").glob("*.npz"))
    checks["actual_counts"] = len(result_paths) == 726 and len(map_paths) == 66
    failures, ensemble_kinds = [], {}
    for path in result_paths:
        item = read_json(path); problems = []
        if (item.get("schema") != "jog-forward-control-evaluation-v1"
                or canonical_id(item) != item.get("manifest_id")
                or item.get("control_id") != path.stem):
            problems.append("identity")
        if item.get("control_kind") not in {"member", "median"} or not item.get("metrics"):
            problems.append("kind_or_metrics")
        ensemble_kinds.setdefault(item.get("ensemble_id"), []).append(item.get("control_kind"))
        for row in item.get("metrics", []):
            required = ["rows", "vector_rmse_m_per_a", "vector_mae_m_per_a",
                        "uniform_vector_rmse_m_per_a", "inversion_vector_rmse_m_per_a"]
            if row.get("rows", 0) <= 0 or not all(np.isfinite(float(row[name])) for name in required):
                problems.append("nonfinite_metric"); break
            if row.get("P_exp_defined") != (row.get("P_exp_percent") is not None):
                problems.append("P_exp_definition"); break
        if problems:
            failures.append({"control_id": path.stem, "problems": sorted(set(problems))})
    checks["ensemble_structure"] = (
        len(ensemble_kinds) == 66
        and all(kinds.count("member") == 10 and kinds.count("median") == 1
                for kinds in ensemble_kinds.values())
    )
    for path in map_paths:
        with np.load(path, allow_pickle=False) as archive:
            sizes = {len(archive[name]) for name in archive.files}
            numeric = [archive[name] for name in archive.files if name != "row_id"]
            if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0 or not all(np.isfinite(x).all() for x in numeric):
                failures.append({"control_id": path.stem, "problems": ["map_archive"]})
    metrics_path = root / "control_population_metrics.csv"
    median_path = root / "median_population_metrics.csv"
    summary_path = root / "ensemble_member_summary.csv"
    try:
        all_metrics = pd.read_csv(metrics_path)
        median_metrics = pd.read_csv(median_path)
        summaries = pd.read_csv(summary_path)
        checks["tables"] = bool(
            set(all_metrics["control_id"]) == {p.stem for p in result_paths}
            and set(median_metrics["control_kind"]) == {"median"}
            and len(set(summaries["ensemble_id"])) == 66
            and summaries["members"].eq(10).all()
        )
    except Exception:
        checks["tables"] = False
    checks["all_control_and_map_artifacts"] = not failures
    result = {
        "schema": "jog-forward-evaluation-verification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_manifest_id": manifest.get("manifest_id"),
        "checks": checks, "failures": failures,
        "counts": {"control_metrics": len(result_paths), "median_maps": len(map_paths),
                   "ensembles": len(ensemble_kinds)},
        "passed": bool(all(checks.values()) and not failures),
    }
    result["manifest_id"] = canonical_id(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); result = verify(args.evaluation_root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
