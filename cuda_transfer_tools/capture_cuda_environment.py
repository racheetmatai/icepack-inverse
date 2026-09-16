#!/usr/bin/env python3
"""Capture the CUDA host software/GPU environment before any L2 fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as error:
        return {"command": command, "error": repr(error)}


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    packages = {}
    for name in ("numpy", "pandas", "sklearn", "joblib", "tensorflow"):
        try:
            module = __import__(name); packages[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            packages[name] = {"import_error": repr(error)}
    tensorflow = {}
    try:
        import tensorflow as tf
        tensorflow = {
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "visible_gpus": [device.name for device in tf.config.get_visible_devices("GPU")],
            "build_info": tf.sysconfig.get_build_info(),
            "cuda_built": bool(tf.test.is_built_with_cuda()),
        }
    except Exception as error:
        tensorflow = {"error": repr(error)}
    record = {
        "schema": "jog-cuda-environment-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "executable": sys.executable, "environment_name": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": packages, "tensorflow": tensorflow,
        "nvidia_smi": command_output(["nvidia-smi"]),
        "nvidia_smi_query": command_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
    }
    record["passed"] = bool(tensorflow.get("physical_gpus")) and bool(tensorflow.get("cuda_built"))
    record["manifest_id"] = canonical_id(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": record["passed"], "manifest_id": record["manifest_id"]}, indent=2))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
