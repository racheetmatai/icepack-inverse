from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_id(value: dict) -> str:
    unsigned = dict(value); unsigned.pop("manifest_id", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


manifest = json.loads((ROOT / "manifest.json").read_text())
table = pd.read_csv(ROOT / "rmse_comparison.csv")
expected = {(f"SQ{i:02d}", f"CFG{j:02d}") for i in range(1, 11) for j in range(1, 7)}
expected.add(("REG_PIG", "CFG02"))
actual = set(zip(table.experiment, table.configuration))

authoritative_matches = []
for row in table.itertuples(index=False):
    record = json.loads((WORKSPACE / "production_workflow/gate4_forward_evaluation_support_aligned_20260910/control_metrics" /
                         f"{row.control_id}.json").read_text())
    population = "central_50km" if row.experiment.startswith("SQ") else "PIG"
    metric = [x for x in record["metrics"] if x["population"] == population and x["support_stratum"] == "all"]
    authoritative_matches.append(len(metric) == 1 and
        np.isclose(row.grid_ml_rmse_m_per_a, metric[0]["vector_rmse_m_per_a"], rtol=1e-13) and
        np.isclose(row.grid_uniform_rmse_m_per_a, metric[0]["uniform_vector_rmse_m_per_a"], rtol=1e-13) and
        np.isclose(row.grid_inversion_rmse_m_per_a, metric[0]["inversion_vector_rmse_m_per_a"], rtol=1e-13))

checks = {
    "manifest_valid": manifest_id(manifest) == manifest["manifest_id"],
    "csv_hash_valid": digest(ROOT / "rmse_comparison.csv") == manifest["outputs"]["rmse_comparison.csv"],
    "exact_case_population": actual == expected and len(table) == 61,
    "all_grid_metrics_match_authoritative_bundle": all(authoritative_matches),
    "finite_metrics": bool(np.isfinite(table.select_dtypes(include=[np.number])).all().all()),
    "positive_rows_cells_area": bool((table[["observation_rows", "fem_cells", "fem_area_m2"]] > 0).all().all()),
    "relative_rmse_formula": bool(np.allclose(table.grid_relative_rmse,
        table.grid_ml_rmse_m_per_a / table.grid_uniform_rmse_m_per_a, rtol=1e-13)) and bool(np.allclose(
        table.fem_relative_rmse, table.fem_ml_rmse_m_per_a / table.fem_uniform_rmse_m_per_a, rtol=1e-13)),
    "no_improvement_classification_changes": bool((table.grid_improves_uniform == table.fem_improves_uniform).all()),
    "quadrature_stable": float(table.quadrature4_vs6_ml_abs_difference.max()) < 1e-9,
    "synthetic_checks": manifest["synthetic_zero_rmse"] == 0.0 and abs(manifest["synthetic_3_4_vector_rmse"]-5) < 1e-10,
}
result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
          "cases": len(table), "manifest_id_checked": manifest["manifest_id"]}
(ROOT / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2))
if result["status"] != "passed":
    raise SystemExit(1)
