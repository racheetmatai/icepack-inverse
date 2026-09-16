"""Independent structural and numerical verifier for the spread diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_id(payload: dict) -> str:
    unsigned = dict(payload); unsigned.pop("manifest_id", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(canonical).hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True))


def spearman(x, y) -> float:
    return float(pd.Series(np.asarray(x, dtype=float)).rank(method="average").corr(
        pd.Series(np.asarray(y, dtype=float)).rank(method="average")))


def run(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "spread_diagnostic_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "schema_status": manifest.get("schema") == "jog-ensemble-spread-diagnostic-summary-v1"
                         and manifest.get("status") == "complete",
        "manifest_id": manifest_id(manifest) == manifest.get("manifest_id"),
        "declared_hashes": True,
        "inventory": True,
        "numerical_recomputation": True,
        "summary_recomputation": True,
    }
    declared = manifest.get("output_sha256", {})
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.name not in {"spread_diagnostic_manifest.json", "verification_manifest.json"}
    }
    checks["inventory"] = actual_files == set(declared)
    for relative, expected in declared.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            checks["declared_hashes"] = False

    table = pd.read_csv(root / "ensemble_spread_population_metrics.csv")
    archives = sorted((root / "ensemble_archives").glob("*.npz"))
    checks["inventory"] &= len(archives) == 60 and len(table) == 180
    for path in archives:
        ensemble = path.stem
        with np.load(path, allow_pickle=False) as archive:
            spread = archive["velocity_spread"].astype(float)
            error = archive["median_velocity_error"].astype(float)
            central = archive["central"].astype(bool)
            row_id = archive["row_id"]
        if (len(spread) != len(error) or len(spread) != len(central) or len(spread) != len(row_id)
                or not np.isfinite(spread).all() or not np.isfinite(error).all()
                or np.any(spread < 0) or len(np.unique(row_id)) != len(row_id)):
            checks["numerical_recomputation"] = False
            continue
        masks = {"central_50km": central, "exclusion_annulus": ~central,
                 "full_130km": np.ones(len(central), dtype=bool)}
        for population, mask in masks.items():
            rows = table.loc[table["ensemble_id"].eq(ensemble) & table["population"].eq(population)]
            if len(rows) != 1 or not np.any(mask):
                checks["numerical_recomputation"] = False
                continue
            row = rows.iloc[0]
            expected = {
                "rows": int(mask.sum()),
                "velocity_spread_rms_m_per_a": float(np.sqrt(np.mean(spread[mask] ** 2))),
                "velocity_spread_mean_m_per_a": float(np.mean(spread[mask])),
                "median_velocity_error_rmse_m_per_a": float(np.sqrt(np.mean(error[mask] ** 2))),
                "median_velocity_error_mae_m_per_a": float(np.mean(error[mask])),
                "pointwise_spearman_rho": spearman(spread[mask], error[mask]),
            }
            for column, value in expected.items():
                if column == "rows":
                    valid = int(row[column]) == value
                else:
                    valid = close(float(row[column]), value)
                checks["numerical_recomputation"] &= valid

    primary = table.loc[table["population"].eq("central_50km")]
    rho = spearman(primary["velocity_spread_rms_m_per_a"],
                   primary["median_velocity_error_rmse_m_per_a"])
    checks["summary_recomputation"] &= close(
        rho, float(manifest["population_level_spearman_rho_all_60_repeated_cases"]))
    checks["summary_recomputation"] &= (
        int((primary["pointwise_spearman_rho"] > 0).sum())
        == int(manifest["positive_within_case_correlations"]))
    checks["summary_recomputation"] &= close(
        float(primary["pointwise_spearman_rho"].median()),
        float(manifest["median_within_case_pointwise_spearman_rho"]))

    result = {
        "schema": "jog-ensemble-spread-diagnostic-verification-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "ensembles": len(archives),
        "population_rows": len(table),
        "parent_manifest_id": manifest.get("manifest_id"),
    }
    result["manifest_id"] = manifest_id(result)
    (root / "verification_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
