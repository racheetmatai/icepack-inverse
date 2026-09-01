"""Resumable Icepack forward solves for frozen member and median log-C controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import time
import traceback
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


def verify_prediction_set(root: Path) -> dict:
    manifest = read_json(root / "prediction_set_manifest.json")
    if (
        manifest.get("schema") != "jog-full-mesh-ensemble-prediction-set-v1"
        or manifest.get("status") != "complete"
        or manifest.get("scope") != "full"
        or manifest_identifier(manifest) != manifest.get("manifest_id")
        or manifest.get("counts", {}).get("member_controls") != 660
        or manifest.get("counts", {}).get("median_controls") != 66
    ):
        raise ValueError("Full-mesh prediction set is not the verified complete population")
    verification = read_json(root / "verification_manifest.json")
    if (
        verification.get("schema") != "jog-full-mesh-ensemble-prediction-verification-v1"
        or verification.get("passed") is not True
        or verification.get("prediction_set_manifest_id") != manifest["manifest_id"]
        or manifest_identifier(verification) != verification.get("manifest_id")
    ):
        raise ValueError("Independent full-mesh prediction verification is missing or invalid")
    return manifest


def build_object(config_path: Path, repo_root: Path, adoption_record: Path):
    repo_root = repo_root.resolve()
    repo_text = str(repo_root)
    if repo_text in sys.path:
        sys.path.remove(repo_text)
    sys.path.insert(0, repo_text)
    config = load_config(config_path)
    preflight = Preflight(config, repo_root, Path("/tmp/jog_forward_context_unused"))
    object_ = preflight.build_invert(reg_c=0.01414213562)
    object_._jog_snes_max_it = 50
    adoption = load_adopted_definitive_state(adoption_record=adoption_record, object_=object_)
    return object_, adoption, config


def configure_solver(object_, snes_max_it: int) -> None:
    """Keep the frozen solver algorithm/tolerances and change only its ceiling."""
    if int(getattr(object_, "_jog_snes_max_it", 50)) == int(snes_max_it):
        return
    object_.opts = {
        "dirichlet_ids": object_.drichlet_ids,
        "side_wall_ids": object_.side_ids,
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "snes_type": "newtontr",
            "snes_max_it": int(snes_max_it),
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }
    object_.create_model_weertman()
    object_._jog_snes_max_it = int(snes_max_it)


def control_registry(prediction_root: Path, prediction_set: dict) -> list[dict]:
    registry = []
    for declared in prediction_set["ensemble_manifests"]:
        item = read_json(prediction_root / declared["path"])
        for index, job_id in enumerate(item["member_job_ids"]):
            registry.append({
                "control_id": job_id,
                "ensemble_id": item["ensemble_id"],
                "kind": "member",
                "member_index": index,
                "prediction_manifest": declared["path"],
                "prediction_npz": item["npz_path"],
                "prediction_npz_sha256": item["npz_sha256"],
            })
        registry.append({
            "control_id": f"{item['ensemble_id']}_MEDIAN",
            "ensemble_id": item["ensemble_id"],
            "kind": "median",
            "member_index": None,
            "prediction_manifest": declared["path"],
            "prediction_npz": item["npz_path"],
            "prediction_npz_sha256": item["npz_sha256"],
        })
    if len(registry) != 726 or len({item["control_id"] for item in registry}) != 726:
        raise RuntimeError("Forward-control registry is not exactly 660 members plus 66 medians")
    return registry


def load_control(prediction_root: Path, item: dict) -> tuple[np.ndarray, np.ndarray]:
    npz_path = prediction_root / item["prediction_npz"]
    if sha256_file(npz_path) != item["prediction_npz_sha256"]:
        raise ValueError(f"Prediction NPZ hash mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        coordinates = archive["coordinates"].copy()
        if item["kind"] == "member":
            control = archive["member_log_C"][item["member_index"]].copy()
        else:
            control = archive["median_log_C"].copy()
    if control.shape != (35797,) or coordinates.shape != (35797, 2) or not np.isfinite(control).all():
        raise ValueError(f"Invalid control array for {item['control_id']}")
    return coordinates, control


def verified_existing(output: Path, item: dict, prediction_set_id: str) -> bool:
    manifest_path = output / "solves" / item["control_id"] / "forward_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    velocity_path = manifest_path.parent / (manifest.get("velocity_path") or "")
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("control_id") == item["control_id"]
        and manifest.get("prediction_set_manifest_id") == prediction_set_id
        and manifest_identifier(manifest) == manifest.get("manifest_id")
        and velocity_path.is_file()
        and sha256_file(velocity_path) == manifest.get("velocity_sha256")
    )


def solve_one(
    object_, adoption: dict, prediction_root: Path, output: Path, item: dict,
    prediction_set_id: str, primary_max_it: int, retry_max_it: int,
):
    import firedrake
    import firedrake.adjoint

    destination = output / "solves" / item["control_id"]
    destination.mkdir(parents=True, exist_ok=True)
    previous_path = destination / "forward_manifest.json"
    prior_failure = None
    if previous_path.is_file():
        previous = read_json(previous_path)
        if previous.get("status") == "failed":
            previous_sha = sha256_file(previous_path)
            archive = (
                output / "failed_attempts" / item["control_id"]
                / previous["manifest_id"] / "forward_manifest.json"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.is_file() and sha256_file(archive) != previous_sha:
                raise RuntimeError(f"Prior-failure archive collision for {item['control_id']}")
            if not archive.is_file():
                shutil.copy2(previous_path, archive)
            prior_failure = {
                "manifest_id": previous["manifest_id"],
                "manifest_sha256": previous_sha,
                "archive_path": str(archive.relative_to(output)),
                "exception": previous.get("exception"),
            }
    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    coordinates, values = load_control(prediction_root, item)
    current_coordinates = np.asarray(
        firedrake.interpolate(object_.mesh.coordinates, object_.V).dat.data_ro[:, :2], dtype=np.float64
    )
    if not np.array_equal(coordinates, current_coordinates):
        raise RuntimeError(f"Control coordinates do not match the solve mesh: {item['control_id']}")
    control = firedrake.Function(object_.Q, name="log_friction_C")
    control.dat.data[:] = values
    if not np.array_equal(control.dat.data_ro, values):
        raise RuntimeError("Assigned log-C control differs from the prediction artifact")
    status = "failed"
    exception = None
    velocity_values = None
    attempts = []
    limits = (
        [int(retry_max_it)]
        if prior_failure and "DIVERGED_MAX_IT" in str(prior_failure.get("exception"))
        else [int(primary_max_it), int(retry_max_it)]
    )
    for snes_max_it in dict.fromkeys(limits):
        configure_solver(object_, snes_max_it)
        firedrake.adjoint.get_working_tape().clear_tape()
        attempt_started = time.perf_counter()
        attempt_exception = None
        try:
            velocity = object_.simulation_C(control)
            velocity_values = np.asarray(velocity.dat.data_ro, dtype=np.float64).copy()
            if velocity_values.shape != (35797, 2) or not np.isfinite(velocity_values).all():
                raise RuntimeError("Forward velocity is nonfinite or has the wrong shape")
            status = "complete"
        except Exception:
            attempt_exception = traceback.format_exc()
            exception = attempt_exception
            print(attempt_exception, flush=True)
        attempts.append({
            "snes_max_it": int(snes_max_it),
            "status": status if status == "complete" else "failed",
            "elapsed_seconds": float(time.perf_counter() - attempt_started),
            "exception": attempt_exception,
        })
        if status == "complete":
            exception = None
            break
        if "DIVERGED_MAX_IT" not in str(attempt_exception):
            break

    manifest = {
        "schema": "jog-icepack-forward-control-v1",
        "status": status,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "control_id": item["control_id"],
        "ensemble_id": item["ensemble_id"],
        "control_kind": item["kind"],
        "prediction_set_manifest_id": prediction_set_id,
        "prediction_npz": item["prediction_npz"],
        "prediction_npz_sha256": item["prediction_npz_sha256"],
        "control_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "adoption_manifest_id": adoption["adoption_manifest_id"],
        "definitive_point_manifest_id": adoption["point_manifest_id"],
        "reg_C": float(adoption["reg_c"]),
        "solver_policy": {
            "initial_velocity": "frozen object_.u_initial for every independent solve",
            "diagnostic_solver_type": "petsc",
            "snes_type": "newtontr",
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "primary_snes_max_it": int(primary_max_it),
            "retry_snes_max_it": int(retry_max_it),
            "retry_trigger": "DIVERGED_MAX_IT only",
            "retry_semantics": "identical solver algorithm and tolerances; only the nonlinear-iteration ceiling changes",
            "fallback": "no alternate algorithm and no silent substitution",
        },
        "prior_failure": prior_failure,
        "attempts": attempts,
        "exception": exception,
        "environment": {
            "hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
        },
        "velocity_path": None,
        "velocity_sha256": None,
        "velocity_summary": None,
    }
    if status == "complete":
        velocity_path = destination / "velocity.npy"
        temporary = destination / "velocity.npy.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, velocity_values, allow_pickle=False)
        os.replace(temporary, velocity_path)
        speed = np.linalg.norm(velocity_values, axis=1)
        manifest.update({
            "velocity_path": velocity_path.name,
            "velocity_sha256": sha256_file(velocity_path),
            "velocity_summary": {
                "finite": True, "minimum_speed": float(speed.min()),
                "median_speed": float(np.median(speed)), "maximum_speed": float(speed.max()),
            },
        })
    manifest["manifest_id"] = manifest_identifier(manifest)
    atomic_json(destination / "forward_manifest.json", manifest)
    print(json.dumps({
        "control_id": item["control_id"], "status": status,
        "elapsed_seconds": manifest["elapsed_seconds"], "manifest_id": manifest["manifest_id"],
    }), flush=True)
    return manifest


def run(args) -> dict:
    prediction_root = args.prediction_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_set = verify_prediction_set(prediction_root)
    registry = control_registry(prediction_root, prediction_set)
    if args.control_id:
        registry = [item for item in registry if item["control_id"] == args.control_id]
        if len(registry) != 1:
            raise ValueError(f"Unknown control ID: {args.control_id}")
    elif args.shard_count is not None:
        if args.shard_count < 1 or args.shard_index < 0 or args.shard_index >= args.shard_count:
            raise ValueError("Invalid shard index/count")
        registry = [item for index, item in enumerate(registry) if index % args.shard_count == args.shard_index]

    pending = [
        item for item in registry
        if not verified_existing(output, item, prediction_set["manifest_id"])
    ]
    print(json.dumps({
        "registered": len(registry), "already_verified": len(registry) - len(pending),
        "pending": len(pending), "control_id": args.control_id,
    }), flush=True)
    if not pending:
        return {"status": "complete", "completed": len(registry), "failed": 0}

    object_, adoption, config = build_object(
        args.config.resolve(), args.repo_root.resolve(), args.adoption_record.resolve()
    )
    completed = 0
    failed = 0
    for item in pending:
        manifest = solve_one(
            object_, adoption, prediction_root, output, item,
            prediction_set["manifest_id"], args.primary_max_it, args.retry_max_it,
        )
        completed += manifest["status"] == "complete"
        failed += manifest["status"] != "complete"
        if failed and args.stop_on_failure:
            break
    summary = {
        "schema": "jog-icepack-forward-campaign-worker-v1",
        "status": "complete" if failed == 0 else "incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_set_manifest_id": prediction_set["manifest_id"],
        "registered_to_worker": len(registry),
        "already_verified": len(registry) - len(pending),
        "newly_completed": int(completed),
        "newly_failed": int(failed),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "control_id": args.control_id,
        "primary_snes_max_it": args.primary_max_it,
        "retry_snes_max_it": args.retry_max_it,
        "source_sha256": sha256_file(Path(__file__)),
    }
    summary["manifest_id"] = manifest_identifier(summary)
    suffix = args.control_id or f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
    atomic_json(output / f"worker_{suffix}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adoption-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-id")
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--primary-max-it", type=int, default=50)
    parser.add_argument("--retry-max-it", type=int, default=100)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
