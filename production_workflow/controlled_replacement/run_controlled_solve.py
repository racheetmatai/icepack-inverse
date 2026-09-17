#!/usr/bin/env python3
"""Run and verify one controlled-replacement Icepack forward solve."""

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

from forward_solve_campaign import build_object, configure_solver
from production_amundsen import manifest_identifier, sha256_file


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def find_job(registry: dict, job_id: str) -> dict:
    matches = [item for item in registry["jobs"] if item["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicated job ID: {job_id}")
    return matches[0]


def archive_failure(output: Path, destination: Path, job_id: str) -> dict | None:
    path = destination / "forward_manifest.json"
    if not path.is_file():
        return None
    previous = read_json(path)
    if previous.get("status") != "failed":
        return None
    prior_hash = sha256_file(path)
    identity = previous.get("manifest_id") or prior_hash
    target = output / "failed_attempts" / job_id / identity / "forward_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256_file(target) != prior_hash:
        raise RuntimeError(f"Prior-failure archive collision for {job_id}")
    if not target.is_file():
        shutil.copy2(path, target)
    return {"manifest_id": identity, "sha256": prior_hash,
            "path": str(target.relative_to(output)), "exception": previous.get("exception")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-max-it", type=int, default=50)
    parser.add_argument("--retry-max-it", type=int, default=100)
    args = parser.parse_args()

    registry = read_json(args.registry)
    if registry.get("schema") != "jog-controlled-replacement-registry-v1" or registry.get("status") != "complete":
        raise ValueError("Invalid controlled-replacement registry")
    item = find_job(registry, args.job_id)
    control_path = args.registry.parent / item["control_path"]
    if sha256_file(control_path) != item["control_sha256"]:
        raise ValueError("Controlled input hash mismatch")

    output = args.output.resolve()
    destination = output / "solves" / args.job_id
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / "active.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Job already has an active lock: {args.job_id}") from exc
    os.write(descriptor, f"{socket.gethostname()} {os.getpid()}\n".encode())
    os.close(descriptor)

    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    status = "failed"
    exception = None
    velocity_values = None
    attempts: list[dict] = []
    prior_failure = None
    try:
        prior_failure = archive_failure(output, destination, args.job_id)
        object_, adoption, _ = build_object(args.config.resolve(), args.repo_root.resolve(), args.adoption.resolve())
        import firedrake
        import firedrake.adjoint

        with np.load(control_path, allow_pickle=False) as archive:
            coordinates = archive["coordinates"].astype(np.float64)
            reference = archive["reference_C"].astype(np.float64)
            replacement = archive["replacement_mask"].astype(bool)
            values = archive["control_C"].astype(np.float64)
        current_coordinates = np.asarray(
            firedrake.interpolate(object_.mesh.coordinates, object_.V).dat.data_ro[:, :2], dtype=np.float64
        )
        adopted_reference = np.asarray(adoption["C"].dat.data_ro, dtype=np.float64)
        if not np.array_equal(coordinates, current_coordinates):
            raise RuntimeError("Control coordinates do not match the solve mesh")
        if not np.array_equal(reference, adopted_reference):
            raise RuntimeError("Controlled input reference differs from the adopted inversion")
        if not np.array_equal(values[~replacement], reference[~replacement]):
            raise RuntimeError("Controlled input changed C outside the replacement mask")
        if not (np.isfinite(values).all() and replacement.any()):
            raise RuntimeError("Controlled input is nonfinite or has an empty replacement mask")

        control = firedrake.Function(object_.Q, name="log_friction_C")
        control.dat.data[:] = values
        if not np.array_equal(control.dat.data_ro, values):
            raise RuntimeError("Assigned control differs from the verified input")

        limits = [args.primary_max_it, args.retry_max_it]
        if prior_failure and "DIVERGED_MAX_IT" in str(prior_failure.get("exception")):
            limits = [args.retry_max_it]
        for limit in dict.fromkeys(int(value) for value in limits):
            configure_solver(object_, limit)
            firedrake.adjoint.get_working_tape().clear_tape()
            attempt_start = time.perf_counter()
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
                "snes_max_it": limit,
                "status": status if status == "complete" else "failed",
                "elapsed_seconds": float(time.perf_counter() - attempt_start),
                "exception": attempt_exception,
            })
            if status == "complete":
                exception = None
                break
            if "DIVERGED_MAX_IT" not in str(attempt_exception):
                break

        manifest = {
            "schema": "jog-controlled-replacement-forward-v1",
            "status": status,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": float(time.perf_counter() - start_time),
            "job_id": args.job_id,
            "experiment": item["experiment"],
            "configuration": item["configuration"],
            "control_kind": item["control_kind"],
            "control_path": item["control_path"],
            "control_file_sha256": item["control_sha256"],
            "control_values_sha256": array_hash(values),
            "replacement_mask_sha256": array_hash(replacement),
            "replacement_dofs": int(replacement.sum()),
            "outside_reference_exact": bool(np.array_equal(values[~replacement], reference[~replacement])),
            "adoption_manifest_id": adoption["adoption_manifest_id"],
            "definitive_point_manifest_id": adoption["point_manifest_id"],
            "reg_C": float(adoption["reg_c"]),
            "solver_policy": {
                "initial_velocity": "unchanged object_.u_initial for each independent solve",
                "diagnostic_solver_type": "petsc", "snes_type": "newtontr",
                "ksp_type": "gmres", "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "primary_snes_max_it": args.primary_max_it,
                "retry_snes_max_it": args.retry_max_it,
                "retry_trigger": "DIVERGED_MAX_IT only",
            },
            "prior_failure": prior_failure,
            "attempts": attempts,
            "exception": exception,
            "environment": {"hostname": socket.gethostname(), "platform": platform.platform(),
                            "python": sys.version, "pid": os.getpid()},
            "velocity_path": None, "velocity_sha256": None, "velocity_summary": None,
        }
        if status == "complete":
            velocity_path = destination / "velocity.npy"
            temporary = velocity_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, velocity_values, allow_pickle=False)
            os.replace(temporary, velocity_path)
            speed = np.linalg.norm(velocity_values, axis=1)
            manifest.update({
                "velocity_path": velocity_path.name,
                "velocity_sha256": sha256_file(velocity_path),
                "velocity_summary": {"finite": True, "minimum_speed": float(speed.min()),
                                     "median_speed": float(np.median(speed)),
                                     "maximum_speed": float(speed.max())},
            })
        manifest["manifest_id"] = manifest_identifier(manifest)
        atomic_json(destination / "forward_manifest.json", manifest)
        print(json.dumps({"job_id": args.job_id, "status": status,
                          "elapsed_seconds": manifest["elapsed_seconds"],
                          "manifest_id": manifest["manifest_id"]}, sort_keys=True), flush=True)
        if status != "complete":
            raise SystemExit(2)
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
