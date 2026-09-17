#!/usr/bin/env python3
"""Run four inter-catchment controlled-replacement jobs with one context.

The container exposes one CPU, so concurrent nonlinear solves cannot increase
throughput.  Reusing one verified model context also matches the accepted
campaign implementation and avoids rebuilding the same mesh for every job.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from forward_solve_campaign import build_object, configure_solver
from production_amundsen import manifest_identifier, sha256_file
from run_controlled_solve import atomic_json, archive_failure, array_hash, read_json


def verified(output: Path, item: dict) -> bool:
    path = output / "solves" / item["job_id"] / "forward_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = read_json(path)
        velocity = path.parent / (manifest.get("velocity_path") or "")
        return bool(
            manifest.get("schema") == "jog-controlled-replacement-forward-v1"
            and manifest.get("status") == "complete"
            and manifest.get("job_id") == item["job_id"]
            and manifest.get("control_file_sha256") == item["control_sha256"]
            and manifest_identifier(manifest) == manifest.get("manifest_id")
            and velocity.is_file() and sha256_file(velocity) == manifest.get("velocity_sha256")
        )
    except Exception:
        return False


def solve_job(object_, adoption: dict, registry_path: Path, output: Path, item: dict,
              primary_max_it: int, retry_max_it: int) -> dict:
    import firedrake
    import firedrake.adjoint

    job_id = item["job_id"]
    control_path = registry_path.parent / item["control_path"]
    if sha256_file(control_path) != item["control_sha256"]:
        raise ValueError(f"Controlled input hash mismatch: {job_id}")
    destination = output / "solves" / job_id
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / "active.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, f"{socket.gethostname()} {os.getpid()}\n".encode()); os.close(descriptor)
    started = datetime.now(timezone.utc); start_time = time.perf_counter()
    prior_failure = None; status = "failed"; exception = None; velocity_values = None
    attempts: list[dict] = []
    try:
        prior_failure = archive_failure(output, destination, job_id)
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
            raise RuntimeError("Controlled input is nonfinite or has an empty mask")
        control = firedrake.Function(object_.Q, name="log_friction_C")
        control.dat.data[:] = values
        if not np.array_equal(control.dat.data_ro, values):
            raise RuntimeError("Assigned control differs from verified input")

        limits = [retry_max_it] if prior_failure and "DIVERGED_MAX_IT" in str(
            prior_failure.get("exception")) else [primary_max_it, retry_max_it]
        for limit in dict.fromkeys(int(value) for value in limits):
            configure_solver(object_, limit)
            firedrake.adjoint.get_working_tape().clear_tape()
            attempt_start = time.perf_counter(); attempt_exception = None
            try:
                velocity = object_.simulation_C(control)
                velocity_values = np.asarray(velocity.dat.data_ro, dtype=np.float64).copy()
                if velocity_values.shape != (35797, 2) or not np.isfinite(velocity_values).all():
                    raise RuntimeError("Forward velocity is nonfinite or has the wrong shape")
                status = "complete"
            except Exception:
                attempt_exception = traceback.format_exc(); exception = attempt_exception
                print(attempt_exception, flush=True)
            attempts.append({"snes_max_it": limit,
                             "status": status if status == "complete" else "failed",
                             "elapsed_seconds": float(time.perf_counter()-attempt_start),
                             "exception": attempt_exception})
            if status == "complete":
                exception = None; break
            if "DIVERGED_MAX_IT" not in str(attempt_exception):
                break

        manifest = {
            "schema": "jog-controlled-replacement-forward-v1", "status": status,
            "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": float(time.perf_counter()-start_time), "job_id": job_id,
            "experiment": item["experiment"], "configuration": item["configuration"],
            "control_kind": item["control_kind"], "control_path": item["control_path"],
            "control_file_sha256": item["control_sha256"],
            "control_values_sha256": array_hash(values),
            "replacement_mask_sha256": array_hash(replacement),
            "replacement_dofs": int(replacement.sum()),
            "outside_reference_exact": bool(np.array_equal(values[~replacement], reference[~replacement])),
            "adoption_manifest_id": adoption["adoption_manifest_id"],
            "definitive_point_manifest_id": adoption["point_manifest_id"], "reg_C": float(adoption["reg_c"]),
            "solver_policy": {"initial_velocity": "unchanged object_.u_initial for every independent solve",
                              "diagnostic_solver_type": "petsc", "snes_type": "newtontr",
                              "ksp_type": "gmres", "pc_type": "lu",
                              "pc_factor_mat_solver_type": "mumps",
                              "primary_snes_max_it": primary_max_it,
                              "retry_snes_max_it": retry_max_it,
                              "retry_trigger": "DIVERGED_MAX_IT only"},
            "prior_failure": prior_failure, "attempts": attempts, "exception": exception,
            "environment": {"hostname": socket.gethostname(), "platform": platform.platform(),
                            "python": sys.version, "pid": os.getpid()},
            "velocity_path": None, "velocity_sha256": None, "velocity_summary": None,
        }
        if status == "complete":
            velocity_path = destination / "velocity.npy"; temporary = velocity_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, velocity_values, allow_pickle=False)
            os.replace(temporary, velocity_path)
            speed = np.linalg.norm(velocity_values, axis=1)
            manifest.update({"velocity_path": velocity_path.name,
                             "velocity_sha256": sha256_file(velocity_path),
                             "velocity_summary": {"finite": True, "minimum_speed": float(speed.min()),
                                                  "median_speed": float(np.median(speed)),
                                                  "maximum_speed": float(speed.max())}})
        manifest["manifest_id"] = manifest_identifier(manifest)
        atomic_json(destination / "forward_manifest.json", manifest)
        print(json.dumps({"job_id": job_id, "status": status,
                          "elapsed_seconds": manifest["elapsed_seconds"],
                          "manifest_id": manifest["manifest_id"]}, sort_keys=True), flush=True)
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-max-it", type=int, default=50)
    parser.add_argument("--retry-max-it", type=int, default=100)
    args = parser.parse_args()
    registry = read_json(args.registry)
    expected = 4
    if registry.get("schema") != "jog-controlled-replacement-registry-v1" or len(registry["jobs"]) != expected:
        raise ValueError("Invalid four-job controlled inter-catchment registry")
    args.output.mkdir(parents=True, exist_ok=True); (args.output/"logs").mkdir(exist_ok=True)
    lock = args.output / "batch_launcher.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, f"{os.getpid()}\n".encode()); os.close(descriptor)
    failed: list[str] = []
    try:
        pending = [item for item in registry["jobs"] if not verified(args.output, item)]
        print(json.dumps({"registered": expected, "already_verified": expected-len(pending),
                          "pending": len(pending), "workers": 1,
                          "reason": "container exposes one CPU; model context reused"}), flush=True)
        if pending:
            object_, adoption, _ = build_object(args.config.resolve(), args.repo_root.resolve(), args.adoption.resolve())
            for index, item in enumerate(pending, start=1):
                log_path = args.output / "logs" / f"{item['job_id']}.log"
                with log_path.open("a", encoding="utf-8") as log, \
                        contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    print(f"START {datetime.now(timezone.utc).isoformat()} {index}/{len(pending)}", flush=True)
                    try:
                        manifest = solve_job(object_, adoption, args.registry, args.output, item,
                                             args.primary_max_it, args.retry_max_it)
                        if manifest["status"] != "complete": failed.append(item["job_id"])
                    except Exception:
                        failed.append(item["job_id"]); print(traceback.format_exc(), flush=True)
                    print(f"END {datetime.now(timezone.utc).isoformat()}", flush=True)
                print(json.dumps({"finished": item["job_id"], "index": index,
                                  "pending": len(pending)-index, "failed": len(failed)}), flush=True)
        complete = sum(verified(args.output, item) for item in registry["jobs"])
        summary = {"schema": "jog-controlled-replacement-launcher-v1",
                   "status": "complete" if complete == expected else "incomplete",
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "registered": expected, "verified_complete": complete,
                   "failed_job_ids": failed, "workers": 1,
                   "concurrency_decision": "one CPU exposed; one reused model context avoids contention",
                   "registry_sha256": sha256_file(args.registry),
                   "worker_sha256": sha256_file(Path(__file__))}
        summary["manifest_id"] = manifest_identifier(summary)
        atomic_json(args.output / "campaign_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        raise SystemExit(0 if summary["status"] == "complete" else 2)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
