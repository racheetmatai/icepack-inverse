#!/usr/bin/env python3
"""Verify a transferred CUDA bundle using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: dict) -> str:
    body = dict(payload); body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("bundle")
    root = Path(parser.parse_args().bundle).resolve()
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest.get("output_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(relative)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "bundle_manifest.json"}
    declared = set(manifest.get("output_sha256", {}))
    result = {
        "schema": manifest.get("schema") == "jog-cuda-transfer-bundle-v1",
        "complete": manifest.get("status") == "complete",
        "manifest_id": canonical_id(manifest) == manifest.get("manifest_id"),
        "declared_hashes": not failures,
        "file_inventory": actual == declared,
        "failures": failures[:20], "missing_or_extra": sorted(actual ^ declared)[:20],
    }
    result["passed"] = all(value for key, value in result.items() if key not in {"failures", "missing_or_extra", "passed"})
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
