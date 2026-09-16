"""Build the small source-only exact-checkpoint CUDA smoke package."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAME = "JOG_CUDA_SMOKE_HOTFIX_20260820_A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload)
    body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def main() -> None:
    destination = ROOT / "cuda_transfer" / NAME
    archive = ROOT / "cuda_transfer" / f"{NAME}.zip"
    if destination.exists() or archive.exists():
        raise FileExistsError("Refusing to replace an existing hotfix package")
    package = destination / "icepack-mlp"
    shutil.copytree(ROOT / "icepack-mlp" / "production_training", package / "production_training",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (package / "tests").mkdir(parents=True)
    shutil.copy2(ROOT / "icepack-mlp" / "tests" / "test_exact_checkpoint_reload.py",
                 package / "tests" / "test_exact_checkpoint_reload.py")
    (package / "tests" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(ROOT / "cuda_transfer_tools" / "CUDA_SMOKE_HOTFIX_README.md", destination / "README.md")
    shutil.copy2(ROOT / "cuda_transfer_tools" / "verify_cuda_smoke_hotfix.py", destination / "verify_cuda_smoke_hotfix.py")
    files = {
        path.relative_to(destination).as_posix(): sha256(path)
        for path in sorted(destination.rglob("*")) if path.is_file()
    }
    manifest = {
        "schema": "jog-cuda-smoke-hotfix-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_bundle_manifest_id": "sha256-json-v1-303df5600ae81d7a21dd66b333fa76929669ce0af880bbe4a22bc8d9aed4a625",
        "selected_lambda_L2": 0.0,
        "purpose": "exact-checkpoint correction and bounded production-split CUDA smoke",
        "files": files,
    }
    manifest["manifest_id"] = canonical_id(manifest)
    (destination / "hotfix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                bundle.write(path, f"{NAME}/{path.relative_to(destination).as_posix()}")
    (archive.with_suffix(archive.suffix + ".sha256")).write_text(
        f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(destination)
    print(archive)


if __name__ == "__main__":
    main()
