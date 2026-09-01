"""Evaluate all model-driven and uniform-C Icepack velocities on MEaSUREs rows.

Velocity functions are evaluated with Icepack/Firedrake interpolation on the
frozen observation mesh.  No nearest-mesh-node lookup is used.  Equal-area
projected observation pixels make row means the corresponding area means.
Per-control JSON files make the long 726-control pass safely resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forward_solve_campaign import build_object
from production_amundsen import manifest_identifier, sha256_file


CATEGORY_NAMES = {0: "neither", 1: "marginal_only", 2: "joint_only", 3: "both"}
CONFIG_NAMES = {
    "CFG01": "CFG01_all_ice", "CFG02": "CFG02_best_ice",
    "CFG03": "CFG03_all_geophysical", "CFG04": "CFG04_best_geophysical",
    "CFG05": "CFG05_best_combined", "CFG06": "CFG06_best_combined_direction",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def experiment_from_ensemble(ensemble_id: str) -> str:
    if "_CFG" not in ensemble_id:
        raise ValueError(ensemble_id)
    return ensemble_id.rsplit("_CFG", 1)[0]


def config_from_ensemble(ensemble_id: str) -> str:
    return "CFG" + ensemble_id.rsplit("_CFG", 1)[1][:2]


def population_masks(frame: pd.DataFrame, experiment: str) -> dict[str, np.ndarray]:
    if experiment.startswith("SQ"):
        central = frame["square_test_id"].eq(experiment).to_numpy()
        full = frame["square_footprint_id"].eq(experiment).to_numpy()
        return {"central_50km": central, "exclusion_annulus": full & ~central, "full_130km": full}
    region = frame["region_code"].to_numpy(np.int8)
    if experiment == "REG_INTER":
        return {"both_corridors": np.isin(region, [4, 5]),
                "pig_thwaites_corridor": region == 4,
                "thwaites_dotson_corridor": region == 5}
    if experiment == "REG_PIG":
        return {"PIG": region == 1}
    raise ValueError(experiment)


def metrics(prediction: np.ndarray, observed: np.ndarray, baseline: np.ndarray,
            inversion: np.ndarray, mask: np.ndarray) -> dict:
    count = int(mask.sum())
    if count == 0:
        raise ValueError("Empty evaluation population")
    pred = prediction[mask]; obs = observed[mask]; base = baseline[mask]; inv = inversion[mask]
    residual = pred - obs; base_residual = base - obs; inv_residual = inv - obs
    squared = np.sum(residual * residual, axis=1)
    base_squared = np.sum(base_residual * base_residual, axis=1)
    inv_squared = np.sum(inv_residual * inv_residual, axis=1)
    mse = float(np.mean(squared)); base_mse = float(np.mean(base_squared))
    obs_rms = float(np.sqrt(np.mean(np.sum(obs * obs, axis=1))))
    base_rmse = float(np.sqrt(base_mse))
    threshold = 1e-6 * max(obs_rms, 1.0)
    p_exp = None if base_rmse <= threshold else float(100.0 * (1.0 - mse / base_mse))
    pred_speed = np.linalg.norm(pred, axis=1); obs_speed = np.linalg.norm(obs, axis=1)
    return {
        "rows": count, "vector_rmse_m_per_a": float(np.sqrt(mse)),
        "vector_mae_m_per_a": float(np.mean(np.sqrt(squared))),
        "vx_bias_m_per_a": float(np.mean(residual[:, 0])),
        "vy_bias_m_per_a": float(np.mean(residual[:, 1])),
        "speed_bias_m_per_a": float(np.mean(pred_speed - obs_speed)),
        "speed_rmse_m_per_a": float(np.sqrt(np.mean((pred_speed - obs_speed) ** 2))),
        "observed_vector_rms_m_per_a": obs_rms,
        "uniform_vector_rmse_m_per_a": base_rmse,
        "inversion_vector_rmse_m_per_a": float(np.sqrt(np.mean(inv_squared))),
        "P_exp_percent": p_exp,
        "P_exp_defined": p_exp is not None,
        "P_exp_denominator_tolerance_m_per_a": threshold,
    }


def support_categories(path: Path, experiment: str, config: str, row_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        indices = archive[f"{experiment}__row_index"]
        values = archive[f"{experiment}__{CONFIG_NAMES[config]}"]
    result = np.full(row_count, 255, dtype=np.uint8)
    result[indices] = values
    return result


def interpolate_velocity(object_, values: np.ndarray, eligible_lookup: np.ndarray) -> np.ndarray:
    import firedrake
    import icepack
    function = firedrake.Function(object_.V, name="evaluated_velocity")
    function.dat.data[:] = values
    ux, uy = firedrake.split(function)
    x = np.asarray(icepack.interpolate(ux, object_.Δ).dat.data_ro, dtype=np.float64)
    y = np.asarray(icepack.interpolate(uy, object_.Δ).dat.data_ro, dtype=np.float64)
    prediction = np.column_stack((x[eligible_lookup], y[eligible_lookup]))
    if prediction.shape != (len(eligible_lookup), 2) or not np.isfinite(prediction).all():
        raise RuntimeError("Observation-mesh velocity interpolation failed")
    return prediction


def build_observation_alignment(object_, dataset_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    from src.data_preprocessing import stable_xy_row_ids
    columns = ["row_id", "x", "y", "common_eligible", "observed_vx", "observed_vy",
               "region_code", "square_test_id", "square_footprint_id",
               "bedmachine_source", "bedmachine_errbed"]
    raw = pd.read_csv(dataset_path, usecols=columns, low_memory=False)
    frame = raw.loc[raw["common_eligible"].astype(bool)].copy()
    frame = frame.sort_values("row_id", kind="stable").reset_index(drop=True)
    delta_coordinates = np.asarray(object_.Δ.mesh().coordinates.dat.data_ro[:, :2], dtype=np.float64)
    delta_ids = np.asarray(stable_xy_row_ids(delta_coordinates[:, 0], delta_coordinates[:, 1]), dtype=str)
    delta_index = pd.Index(delta_ids)
    if delta_index.has_duplicates or frame["row_id"].duplicated().any():
        raise RuntimeError("Observation-mesh row IDs are not unique")
    lookup = delta_index.get_indexer(frame["row_id"].astype(str))
    if np.any(lookup < 0):
        raise RuntimeError("Canonical eligible rows do not align with the observation mesh")
    return frame, lookup.astype(np.int64)


def model_registry(campaign_root: Path) -> list[dict]:
    records = []
    for path in sorted((campaign_root / "solves").glob("*/forward_manifest.json")):
        manifest = read_json(path)
        if manifest.get("status") != "complete" or manifest_identifier(manifest) != manifest.get("manifest_id"):
            raise ValueError(f"Invalid model-driven solve: {path}")
        velocity_path = path.parent / manifest["velocity_path"]
        if sha256_file(velocity_path) != manifest["velocity_sha256"]:
            raise ValueError(f"Velocity hash mismatch: {velocity_path}")
        records.append({"control_id": manifest["control_id"], "ensemble_id": manifest["ensemble_id"],
                        "kind": manifest["control_kind"], "velocity_path": velocity_path,
                        "forward_manifest_id": manifest["manifest_id"]})
    if len(records) != 726:
        raise RuntimeError(f"Model campaign has {len(records)} controls, not 726")
    return records


def baseline_registry(root: Path) -> dict[str, dict]:
    campaign = read_json(root / "baseline_campaign_manifest.json")
    if campaign.get("count") != 12 or manifest_identifier(campaign) != campaign.get("manifest_id"):
        raise ValueError("Invalid uniform-baseline campaign")
    records = {}
    for declared in campaign["controls"]:
        path = root / "solves" / declared["control_id"] / "forward_manifest.json"
        manifest = read_json(path); velocity = path.parent / manifest["velocity_path"]
        if (manifest.get("status") != "complete" or manifest_identifier(manifest) != manifest.get("manifest_id")
                or sha256_file(velocity) != manifest.get("velocity_sha256")):
            raise ValueError(f"Invalid uniform-baseline solve: {path}")
        records[declared["experiment"]] = {"velocity_path": velocity, "uniform_C": declared["uniform_C"],
                                                       "forward_manifest_id": manifest["manifest_id"]}
    return records


def control_result(args, record, prediction, observed, inversion, baseline, frame, support) -> dict:
    experiment = experiment_from_ensemble(record["ensemble_id"])
    config = config_from_ensemble(record["ensemble_id"])
    rows = []
    masks = population_masks(frame, experiment)
    for population, mask in masks.items():
        item = metrics(prediction, observed, baseline, inversion, mask)
        rows.append({"population": population, "support_stratum": "all", **item})
        for value, name in CATEGORY_NAMES.items():
            stratum = mask & (support == value)
            if np.any(stratum):
                rows.append({"population": population, "support_stratum": name,
                             **metrics(prediction, observed, baseline, inversion, stratum)})
    result = {
        "schema": "jog-forward-control-evaluation-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "control_id": record["control_id"], "ensemble_id": record["ensemble_id"],
        "control_kind": record["kind"], "experiment": experiment, "configuration": config,
        "forward_manifest_id": record["forward_manifest_id"], "metrics": rows,
        "interpolation": "Icepack/Firedrake finite-element interpolation to the frozen observation mesh",
        "observational_reference": "MEaSUREs observed vx and vy; inversion velocity is secondary only",
    }
    result["manifest_id"] = manifest_identifier(result)
    return result


def write_tables(output: Path, result_paths: list[Path]) -> None:
    rows = []
    for path in result_paths:
        result = read_json(path)
        for metric_row in result["metrics"]:
            rows.append({key: value for key, value in {
                "control_id": result["control_id"], "ensemble_id": result["ensemble_id"],
                "control_kind": result["control_kind"], "experiment": result["experiment"],
                "configuration": result["configuration"], **metric_row}.items()})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "control_population_metrics.csv", index=False)
    numeric = ["vector_rmse_m_per_a", "vector_mae_m_per_a", "vx_bias_m_per_a", "vy_bias_m_per_a",
               "speed_bias_m_per_a", "speed_rmse_m_per_a", "P_exp_percent"]
    members = frame.loc[frame["control_kind"].eq("member")]
    grouped = members.groupby(["ensemble_id", "experiment", "configuration", "population", "support_stratum"], dropna=False)
    records = []
    for keys, group in grouped:
        row = dict(zip(["ensemble_id", "experiment", "configuration", "population", "support_stratum"], keys))
        row["members"] = len(group)
        for column in numeric:
            values = group[column].dropna().to_numpy(float)
            row[f"{column}__mean"] = float(np.mean(values)) if len(values) else None
            row[f"{column}__sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            row[f"{column}__q05"] = float(np.quantile(values, .05)) if len(values) else None
            row[f"{column}__q95"] = float(np.quantile(values, .95)) if len(values) else None
        records.append(row)
    pd.DataFrame(records).to_csv(output / "ensemble_member_summary.csv", index=False)
    frame.loc[frame["control_kind"].eq("median")].to_csv(output / "median_population_metrics.csv", index=False)


def run(args) -> dict:
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    controls_dir = output / "control_metrics"; controls_dir.mkdir(exist_ok=True)
    maps_dir = output / "median_map_data"; maps_dir.mkdir(exist_ok=True)
    object_, adoption, _ = build_object(args.config.resolve(), args.repo_root.resolve(), args.adoption_record.resolve())
    frame, lookup = build_observation_alignment(object_, args.dataset.resolve())
    observed = frame[["observed_vx", "observed_vy"]].to_numpy(np.float64)
    inversion = interpolate_velocity(object_, np.asarray(adoption["velocity"].dat.data_ro), lookup)
    baselines = baseline_registry(args.baseline_root.resolve())
    baseline_predictions = {
        experiment: interpolate_velocity(object_, np.load(item["velocity_path"], allow_pickle=False), lookup)
        for experiment, item in baselines.items()
    }
    support_path = args.support_bundle.resolve() / "point_support_categories.npz"
    registry = model_registry(args.campaign_root.resolve())
    result_paths = []
    for number, record in enumerate(registry, start=1):
        destination = controls_dir / f"{record['control_id']}.json"
        if destination.is_file():
            existing = read_json(destination)
            if (existing.get("forward_manifest_id") == record["forward_manifest_id"]
                    and manifest_identifier(existing) == existing.get("manifest_id")):
                result_paths.append(destination); continue
        experiment = experiment_from_ensemble(record["ensemble_id"])
        config = config_from_ensemble(record["ensemble_id"])
        support = support_categories(support_path, experiment, config, len(frame))
        prediction = interpolate_velocity(object_, np.load(record["velocity_path"], allow_pickle=False), lookup)
        result = control_result(args, record, prediction, observed, inversion,
                                baseline_predictions[experiment], frame, support)
        atomic_json(destination, result); result_paths.append(destination)
        if record["kind"] == "median":
            primary_name = "central_50km" if experiment.startswith("SQ") else ("both_corridors" if experiment == "REG_INTER" else "PIG")
            mask = population_masks(frame, experiment)[primary_name]
            local = (np.sum((baseline_predictions[experiment][mask] - observed[mask]) ** 2, axis=1)
                     - np.sum((prediction[mask] - observed[mask]) ** 2, axis=1))
            atomic_npz(maps_dir / f"{record['control_id']}.npz",
                       row_id=frame.loc[mask, "row_id"].to_numpy(str),
                       x=frame.loc[mask, "x"].to_numpy(np.float64),
                       y=frame.loc[mask, "y"].to_numpy(np.float64),
                       predicted_vx=prediction[mask, 0], predicted_vy=prediction[mask, 1],
                       observed_vx=observed[mask, 0], observed_vy=observed[mask, 1],
                       error_magnitude=np.linalg.norm(prediction[mask] - observed[mask], axis=1),
                       signed_local_squared_error_improvement=local,
                       support_category=support[mask])
        if number % 25 == 0 or number == len(registry):
            print(json.dumps({"evaluated": number, "total": len(registry)}), flush=True)
    write_tables(output, result_paths)
    outputs = {}
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "evaluation_manifest.json"):
        outputs[path.relative_to(output).as_posix()] = sha256_file(path)
    manifest = {
        "schema": "jog-forward-evaluation-bundle-v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "evaluated_controls": len(result_paths),
        "median_map_archives": len(list(maps_dir.glob("*.npz"))),
        "dataset_sha256": sha256_file(args.dataset.resolve()),
        "support_manifest_id": read_json(args.support_bundle.resolve() / "diagnostics_manifest.json")["manifest_id"],
        "adoption_manifest_id": adoption["adoption_manifest_id"], "output_sha256": outputs,
        "P_exp_definition": "100*(1-model vector MSE/uniform-C vector MSE), denominator once per population",
        "independent_spatial_replicates": "ten squares; members and nested regions are uncertainty/diagnostic levels",
    }
    manifest["manifest_id"] = manifest_identifier(manifest)
    atomic_json(output / "evaluation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adoption-record", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--support-bundle", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
