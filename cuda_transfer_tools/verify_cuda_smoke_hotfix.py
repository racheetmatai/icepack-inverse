"""Standard-library verifier for the JOG CUDA smoke hotfix package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


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
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent).resolve()
    manifest = json.loads((root / "hotfix_manifest.json").read_text(encoding="utf-8"))
    actual = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "hotfix_manifest.json"
    }
    checks = {
        "schema": manifest.get("schema") == "jog-cuda-smoke-hotfix-v1",
        "manifest_id": canonical_id(manifest) == manifest.get("manifest_id"),
        "file_inventory_and_hashes": actual == manifest.get("files"),
        "baseline_bundle_id": manifest.get("baseline_bundle_manifest_id")
        == "sha256-json-v1-303df5600ae81d7a21dd66b333fa76929669ce0af880bbe4a22bc8d9aed4a625",
    }
    result = {"checks": checks, "passed": all(checks.values()), "manifest_id": manifest.get("manifest_id")}
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
