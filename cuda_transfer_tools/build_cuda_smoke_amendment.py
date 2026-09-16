"""Build the source-only smoke acceptance amendment."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAME = "JOG_CUDA_SMOKE_AMENDMENT_20260820_B"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    destination = ROOT / "cuda_transfer" / NAME
    archive = ROOT / "cuda_transfer" / f"{NAME}.zip"
    if destination.exists() or archive.exists():
        raise FileExistsError("Refusing to replace an existing amendment")
    destination.mkdir()
    shutil.copy2(ROOT / "icepack-mlp" / "production_training" / "finalize_cuda_smoke.py",
                 destination / "finalize_cuda_smoke.py")
    shutil.copy2(ROOT / "cuda_transfer_tools" / "CUDA_SMOKE_AMENDMENT_README.md",
                 destination / "README.md")
    files = {path.name: sha256(path) for path in sorted(destination.iterdir()) if path.is_file()}
    manifest = {
        "schema": "jog-cuda-smoke-amendment-v1",
        "parent_hotfix_manifest_id": "sha256-json-v1-2d8dbbac56d4e21dbd2296f770f3a15570aa0341a3e2304f10e4bc750fc67bf2",
        "purpose": "finalize completed smoke using persisted-to-persisted prediction comparison",
        "files": files,
    }
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    manifest["manifest_id"] = "sha256-json-v1-" + hashlib.sha256(body).hexdigest()
    (destination / "amendment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(destination.iterdir()):
            if path.is_file():
                bundle.write(path, f"{NAME}/{path.name}")
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "sha256": sha256(archive),
                      "manifest_id": manifest["manifest_id"]}, indent=2))


if __name__ == "__main__":
    main()
