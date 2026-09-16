"""Small canonical hashing and manifest helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest_id(payload: dict) -> str:
    body = dict(payload)
    body.pop("manifest_id", None)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def verify_declared_outputs(root: Path, manifest: dict) -> None:
    failures = [
        relative for relative, expected in manifest["output_sha256"].items()
        if not (root / relative).is_file()
        or file_sha256(root / relative) != expected
    ]
    if failures:
        raise ValueError(f"Declared output hash failures: {failures[:5]}")
