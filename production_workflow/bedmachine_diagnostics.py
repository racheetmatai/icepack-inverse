"""Prespecified BedMachine-v2 provenance diagnostics for median predictions.

Uses native source categories and continuous errbed only.  It does not create
post-hoc uncertainty bins or alter any evaluation population.
"""

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
    unsigned = dict(payload); unsigned.pop("manifest_id", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    return pearson, spearman


def run(args) -> dict:
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(
        args.dataset,
        usecols=["row_id", "common_eligible", "bedmachine_source", "bedmachine_errbed"],
        low_memory=False,
    )
    dataset = dataset.loc[dataset.common_eligible.astype(bool)].copy()
    dataset["row_id"] = dataset.row_id.astype(str)
    if dataset.row_id.duplicated().any():
        raise RuntimeError("Canonical eligible row IDs are not unique")
    lookup = dataset.set_index("row_id")

    source_rows, continuous_rows, distribution_rows = [], [], []
    archives = sorted(args.evaluation_maps.glob("*_MEDIAN.npz"))
    if len(archives) != 66:
        raise RuntimeError(f"Expected 66 median map archives, found {len(archives)}")

    for path in archives:
        control_id = path.stem
        ensemble_id = control_id.removesuffix("_MEDIAN")
        experiment, configuration = ensemble_id.rsplit("_CFG", 1)
        configuration = "CFG" + configuration
        with np.load(path, allow_pickle=False) as archive:
            row_ids = archive["row_id"].astype(str)
            predicted = np.column_stack((archive["predicted_vx"], archive["predicted_vy"]))
            observed = np.column_stack((archive["observed_vx"], archive["observed_vy"]))
        auxiliary = lookup.reindex(row_ids)
        if auxiliary.index.hasnans or auxiliary.bedmachine_source.isna().all():
            raise RuntimeError(f"Failed row-ID alignment for {path.name}")
        residual = predicted - observed
        vector_error = np.linalg.norm(residual, axis=1)
        squared_error = np.sum(residual * residual, axis=1)
        errbed = auxiliary.bedmachine_errbed.to_numpy(dtype=float)

        for source, index in auxiliary.groupby("bedmachine_source", dropna=False).groups.items():
            ii = auxiliary.index.get_indexer(index)
            values = vector_error[ii]
            source_rows.append({
                "ensemble_id": ensemble_id, "experiment": experiment,
                "configuration": configuration, "bedmachine_source": str(source),
                "rows": int(len(ii)), "vector_rmse_m_per_a": float(np.sqrt(np.mean(squared_error[ii]))),
                "vector_mae_m_per_a": float(np.mean(values)),
                "median_vector_error_m_per_a": float(np.median(values)),
            })

        pearson, spearman = correlation(errbed, vector_error)
        finite = np.isfinite(errbed)
        continuous_rows.append({
            "ensemble_id": ensemble_id, "experiment": experiment,
            "configuration": configuration, "rows": int(len(errbed)),
            "finite_errbed_rows": int(finite.sum()), "missing_errbed_rows": int((~finite).sum()),
            "pearson_errbed_vs_vector_error": pearson,
            "spearman_errbed_vs_vector_error": spearman,
        })

        finite_values = errbed[finite]
        distribution_rows.append({
            "ensemble_id": ensemble_id, "experiment": experiment,
            "configuration": configuration, "rows": int(len(errbed)),
            "finite_errbed_rows": int(len(finite_values)),
            "errbed_mean_m": float(np.mean(finite_values)) if len(finite_values) else None,
            "errbed_q01_m": float(np.quantile(finite_values, .01)) if len(finite_values) else None,
            "errbed_q25_m": float(np.quantile(finite_values, .25)) if len(finite_values) else None,
            "errbed_median_m": float(np.median(finite_values)) if len(finite_values) else None,
            "errbed_q75_m": float(np.quantile(finite_values, .75)) if len(finite_values) else None,
            "errbed_q99_m": float(np.quantile(finite_values, .99)) if len(finite_values) else None,
        })

    outputs = {
        "native_source_metrics.csv": pd.DataFrame(source_rows),
        "continuous_errbed_correlations.csv": pd.DataFrame(continuous_rows),
        "errbed_distributions.csv": pd.DataFrame(distribution_rows),
    }
    hashes = {}
    for name, frame in outputs.items():
        path = output / name; frame.to_csv(path, index=False); hashes[name] = sha256_file(path)
    manifest = {
        "schema": "jog-bedmachine-auxiliary-diagnostics-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "66 median-model primary evaluation populations",
        "policy": "native source categories and continuous errbed; no post-hoc bins; no row deletion",
        "dataset_sha256": sha256_file(args.dataset), "median_archives": len(archives),
        "output_sha256": hashes, "source_sha256": sha256_file(Path(__file__)),
    }
    manifest["manifest_id"] = canonical_id(manifest)
    (output / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--evaluation-maps", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
