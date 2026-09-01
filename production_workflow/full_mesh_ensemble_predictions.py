"""Build full-mesh member and median log-C controls from frozen MLP checkpoints.

This stage performs inference only.  It does not run Icepack forward solves.
Predictions replace the adopted definitive inversion control only on the frozen
eligible grounded CG2 mask; every other degree of freedom is retained exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from load_definitive_inversion import load_adopted_definitive_state
from production_amundsen import Preflight, load_config, manifest_identifier, sha256_file


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_inference_bundle(root: Path) -> dict:
    manifest_path = root / "inference_bundle_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "jog-production-inference-bundle-v1":
        raise ValueError("Unexpected inference-bundle schema")
    if manifest.get("status") != "complete" or manifest_identifier(manifest) != manifest.get("manifest_id"):
        raise ValueError("Inference-bundle manifest is incomplete or has an invalid ID")
    if manifest.get("counts") != {"jobs": 660, "ensembles": 66}:
        raise ValueError("Inference bundle does not contain the frozen 660 jobs / 66 ensembles")
    for job in manifest["jobs"]:
        for relative, expected in job["files"].items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Inference-bundle file hash mismatch: {relative}")
    return manifest


def construct_hybrid_controls(
    reference_log_c: np.ndarray,
    eligible: np.ndarray,
    eligible_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ten hybrid member controls and their vertex-wise median."""
    reference = np.asarray(reference_log_c, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    predictions = np.asarray(eligible_predictions, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[0] != 10:
        raise ValueError("Every ensemble must contain exactly ten member predictions")
    if predictions.shape[1] != int(mask.sum()) or reference.shape != mask.shape:
        raise ValueError("Prediction, mask, and reference shapes disagree")
    if not np.isfinite(reference).all() or not np.isfinite(predictions).all():
        raise ValueError("Reference or predicted controls contain nonfinite values")
    members = np.repeat(reference[None, :], 10, axis=0)
    members[:, mask] = predictions
    median = np.median(members, axis=0)
    if not np.array_equal(members[:, ~mask], np.repeat(reference[None, ~mask], 10, axis=0)):
        raise RuntimeError("Member replacement changed control values outside the eligible mask")
    if not np.array_equal(median[~mask], reference[~mask]):
        raise RuntimeError("Median replacement changed control values outside the eligible mask")
    return members, median


def build_model(tf, input_count: int):
    inputs = tf.keras.Input(shape=(int(input_count),), name="predictors")
    value = inputs
    for index in range(10):
        value = tf.keras.layers.Dense(200, name=f"hidden_dense_{index + 1:02d}")(value)
        value = tf.keras.layers.BatchNormalization(name=f"hidden_bn_{index + 1:02d}")(value)
        value = tf.keras.layers.Activation("silu", name=f"hidden_silu_{index + 1:02d}")(value)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="scaled_log_C")(value)
    return tf.keras.Model(inputs, outputs, name="jog_log_C_mlp")


def predict_member(tf, run_dir: Path, features: dict[str, np.ndarray], eligible: np.ndarray) -> np.ndarray:
    spec = read_json(run_dir / "resolved_spec.json")
    scalers = read_json(run_dir / "scaler_parameters.json")
    names = spec["features"]
    matrix = np.column_stack([features[name][eligible] for name in names])
    center = np.asarray(scalers["input"]["center"], dtype=np.float64)
    scale = np.asarray(scalers["input"]["scale"], dtype=np.float64)
    if matrix.shape[1] != len(center) or center.shape != scale.shape or np.any(scale == 0.0):
        raise ValueError(f"Invalid scaler dimensions in {run_dir}")
    scaled_input = ((matrix - center) / scale).astype(np.float32)
    model = build_model(tf, scaled_input.shape[1])
    model.load_weights(run_dir / "best_model.keras")
    scaled_prediction = model(scaled_input, training=False).numpy().reshape(-1).astype(np.float64)
    target_center = float(scalers["target"]["center"][0])
    target_scale = float(scalers["target"]["scale"][0])
    prediction = scaled_prediction * target_scale + target_center
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite full-mesh prediction from {run_dir.name}")
    tf.keras.backend.clear_session()
    return prediction


def full_mesh_context(config_path: Path, repo_root: Path, adoption_record: Path):
    repo_root = repo_root.resolve()
    repo_text = str(repo_root)
    if repo_text in sys.path:
        sys.path.remove(repo_text)
    sys.path.insert(0, repo_text)
    import firedrake
    from icepack.constants import ice_density, water_density
    from src.feature_units import driving_stress_mpa, stabilized_gradient_alignment

    config = load_config(config_path)
    preflight = Preflight(config, repo_root, Path("/tmp/jog_full_mesh_context_unused"))
    object_ = preflight.build_invert(reg_c=0.01414213562)
    adoption = load_adopted_definitive_state(adoption_record=adoption_record, object_=object_)
    grad_h = firedrake.interpolate(firedrake.grad(object_.h), object_.V)
    grad_s = firedrake.interpolate(firedrake.grad(object_.s), object_.V)
    grad_b = firedrake.interpolate(firedrake.grad(object_.b), object_.V)
    gh = np.asarray(grad_h.dat.data_ro, dtype=np.float64)
    gs = np.asarray(grad_s.dat.data_ro, dtype=np.float64)
    gb = np.asarray(grad_b.dat.data_ro, dtype=np.float64)
    h = np.asarray(object_.h.dat.data_ro, dtype=np.float64)
    s = np.asarray(object_.s.dat.data_ro, dtype=np.float64)
    features = {
        "s": s,
        "h": h,
        "mag_s": np.linalg.norm(gs, axis=1),
        "mag_h": np.linalg.norm(gh, axis=1),
        "surface_air_temp": np.asarray(object_.surface_air_temp.dat.data_ro, dtype=np.float64),
        "b": np.asarray(object_.b.dat.data_ro, dtype=np.float64),
        "mag_b": np.linalg.norm(gb, axis=1),
        "heatflux": np.asarray(object_.heatflux.dat.data_ro, dtype=np.float64),
        "gravity_disturbance": np.asarray(object_.gravity_disturbance.dat.data_ro, dtype=np.float64),
        "mag_anomaly": np.asarray(object_.mag_anomaly.dat.data_ro, dtype=np.float64),
        "cos_theta_bs": stabilized_gradient_alignment(gb[:, 0], gb[:, 1], gs[:, 0], gs[:, 1]),
    }
    features["driving_stress"] = np.asarray(
        driving_stress_mpa(h, features["mag_s"]), dtype=np.float64
    )
    water_ratio = np.divide(
        float(water_density) * np.maximum(0.0, h - s),
        float(ice_density) * h,
        out=np.ones_like(h), where=h > 0.0,
    )
    grounded = (np.maximum(1.0 - water_ratio, 0.0) > 0.1) & (h > 0.0)
    union = np.column_stack([features[name] for name in config["predictors"]])
    eligible = grounded & np.isfinite(union).all(axis=1)
    expected = config["expected_counts"]
    if object_.Q.dim() != expected["cg2_scalar_dofs"] or int(eligible.sum()) != expected["grounded_cg2_dofs"]:
        raise RuntimeError("Full-mesh DOF or eligible-grounded count differs from the frozen preflight")
    coordinates = np.asarray(
        firedrake.interpolate(object_.mesh.coordinates, object_.V).dat.data_ro[:, :2],
        dtype=np.float64,
    ).copy()
    reference_log_c = np.asarray(object_.C.dat.data_ro, dtype=np.float64).copy()
    return object_, features, eligible, coordinates, reference_log_c, adoption


def run(args) -> dict:
    bundle_root = args.inference_bundle.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = validate_inference_bundle(bundle_root)
    object_, features, eligible, coordinates, reference_log_c, adoption = full_mesh_context(
        args.config.resolve(), args.repo_root.resolve(), args.adoption_record.resolve()
    )
    del object_

    import tensorflow as tf

    jobs = {item["job_id"]: item for item in bundle["jobs"]}
    if args.member_smoke_job_id:
        job_id = args.member_smoke_job_id
        if job_id not in jobs:
            raise ValueError(f"Unknown smoke job ID: {job_id}")
        prediction = predict_member(tf, bundle_root / "runs" / job_id, features, eligible)
        hybrid = reference_log_c.copy()
        hybrid[eligible] = prediction
        npz_path = output / f"{job_id}_full_mesh_smoke.npz"
        temporary = npz_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, coordinates=coordinates, eligible_mask=eligible,
                reference_log_C=reference_log_c, predicted_log_C=hybrid,
            )
        os.replace(temporary, npz_path)
        smoke = {
            "schema": "jog-full-mesh-member-prediction-smoke-v1",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "features": jobs[job_id]["features"],
            "run_manifest_id": jobs[job_id]["run_manifest_id"],
            "inference_bundle_manifest_id": bundle["manifest_id"],
            "adoption_manifest_id": adoption["adoption_manifest_id"],
            "definitive_point_manifest_id": adoption["point_manifest_id"],
            "reg_C": float(adoption["reg_c"]),
            "counts": {
                "cg2_dofs": int(len(eligible)),
                "eligible_grounded_dofs": int(eligible.sum()),
                "retained_definitive_dofs": int((~eligible).sum()),
            },
            "all_predictions_finite": bool(np.isfinite(hybrid).all()),
            "outside_mask_exact": bool(np.array_equal(hybrid[~eligible], reference_log_c[~eligible])),
            "predicted_log_C_minimum": float(hybrid.min()),
            "predicted_log_C_maximum": float(hybrid.max()),
            "npz_path": npz_path.name,
            "npz_sha256": sha256_file(npz_path),
            "environment": {
                "hostname": socket.gethostname(), "platform": platform.platform(),
                "python": sys.version, "tensorflow": tf.__version__,
            },
        }
        smoke["manifest_id"] = manifest_identifier(smoke)
        atomic_json(output / f"{job_id}_full_mesh_smoke.json", smoke)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return smoke

    selected_ensembles = bundle["ensembles"]
    if args.ensemble_id:
        selected_ensembles = [item for item in selected_ensembles if item["ensemble_id"] == args.ensemble_id]
        if len(selected_ensembles) != 1:
            raise ValueError(f"Unknown ensemble ID: {args.ensemble_id}")
    if args.max_ensembles is not None:
        selected_ensembles = selected_ensembles[: args.max_ensembles]

    outputs = []
    for sequence, ensemble in enumerate(selected_ensembles, start=1):
        ensemble_id = ensemble["ensemble_id"]
        npz_path = output / f"{ensemble_id}.npz"
        item_manifest_path = output / f"{ensemble_id}.json"
        if item_manifest_path.is_file() and npz_path.is_file():
            previous = read_json(item_manifest_path)
            if previous.get("status") == "complete" and sha256_file(npz_path) == previous.get("npz_sha256"):
                outputs.append(previous)
                print(f"Verified existing {ensemble_id} ({sequence}/{len(selected_ensembles)})", flush=True)
                continue
        predictions = []
        for job_id in ensemble["member_job_ids"]:
            prediction = predict_member(tf, bundle_root / "runs" / job_id, features, eligible)
            predictions.append(prediction)
            print(f"Predicted {job_id}", flush=True)
        eligible_predictions = np.vstack(predictions)
        members, median = construct_hybrid_controls(reference_log_c, eligible, eligible_predictions)
        temporary = npz_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                coordinates=coordinates,
                eligible_mask=eligible,
                reference_log_C=reference_log_c,
                member_log_C=members,
                median_log_C=median,
                member_job_ids=np.asarray(ensemble["member_job_ids"], dtype="U32"),
            )
        os.replace(temporary, npz_path)
        item = {
            "schema": "jog-full-mesh-ensemble-prediction-v1",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "ensemble_id": ensemble_id,
            "experiment": ensemble["experiment"],
            "configuration": ensemble["configuration"],
            "features": ensemble["features"],
            "member_job_ids": ensemble["member_job_ids"],
            "member_run_manifest_ids": [jobs[name]["run_manifest_id"] for name in ensemble["member_job_ids"]],
            "inference_bundle_manifest_id": bundle["manifest_id"],
            "adoption_manifest_id": adoption["adoption_manifest_id"],
            "definitive_point_manifest_id": adoption["point_manifest_id"],
            "reg_C": float(adoption["reg_c"]),
            "counts": {
                "cg2_dofs": int(len(eligible)),
                "eligible_grounded_dofs": int(eligible.sum()),
                "retained_definitive_dofs": int((~eligible).sum()),
                "members": 10,
            },
            "control_summary": {
                "reference_minimum": float(reference_log_c.min()),
                "reference_maximum": float(reference_log_c.max()),
                "member_minimum": float(members.min()),
                "member_maximum": float(members.max()),
                "median_minimum": float(median.min()),
                "median_maximum": float(median.max()),
            },
            "outside_mask_exact": bool(
                np.array_equal(members[:, ~eligible], np.repeat(reference_log_c[None, ~eligible], 10, axis=0))
                and np.array_equal(median[~eligible], reference_log_c[~eligible])
            ),
            "npz_path": npz_path.name,
            "npz_sha256": sha256_file(npz_path),
        }
        item["manifest_id"] = manifest_identifier(item)
        atomic_json(item_manifest_path, item)
        outputs.append(item)
        print(f"Completed ensemble {ensemble_id} ({sequence}/{len(selected_ensembles)})", flush=True)

    root_manifest = {
        "schema": "jog-full-mesh-ensemble-prediction-set-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "full" if len(outputs) == 66 else "bounded",
        "environment": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version, "tensorflow": tf.__version__,
        },
        "inference_bundle_manifest_id": bundle["manifest_id"],
        "adoption_manifest_id": adoption["adoption_manifest_id"],
        "reg_C": float(adoption["reg_c"]),
        "counts": {
            "ensembles": len(outputs), "member_controls": len(outputs) * 10,
            "median_controls": len(outputs), "cg2_dofs": len(eligible),
            "eligible_grounded_dofs": int(eligible.sum()),
        },
        "ensemble_manifests": [
            {"path": f"{item['ensemble_id']}.json", "manifest_id": item["manifest_id"],
             "sha256": sha256_file(output / f"{item['ensemble_id']}.json")}
            for item in outputs
        ],
    }
    root_manifest["manifest_id"] = manifest_identifier(root_manifest)
    atomic_json(output / "prediction_set_manifest.json", root_manifest)
    print(json.dumps({
        "status": "complete", "scope": root_manifest["scope"],
        "ensembles": len(outputs), "manifest_id": root_manifest["manifest_id"],
    }, indent=2))
    return root_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-bundle", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adoption-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--member-smoke-job-id")
    parser.add_argument("--ensemble-id")
    parser.add_argument("--max-ensembles", type=int)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
