#!/usr/bin/env python3
"""Verify the archived artifacts used by the published analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("paper", "full", "all"), default="all")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/artifacts.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0
    selected = [
        item
        for item in manifest["archives"]
        if args.profile == "all" or args.profile in item.get("profiles", [])
    ]
    for item in selected:
        path = args.artifact_dir / item["filename"]
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        observed_size = path.stat().st_size
        if observed_size != item["bytes"]:
            failures.append(
                f"size mismatch: {path.name}: {observed_size} != {item['bytes']}"
            )
            continue
        observed_hash = sha256(path)
        if observed_hash != item["sha256"]:
            failures.append(
                f"hash mismatch: {path.name}: {observed_hash} != {item['sha256']}"
            )
            continue
        checked += 1

    result = {
        "schema": "jog-reproduction-artifact-verification-v1",
        "passed": not failures and checked == len(selected),
        "checked": checked,
        "declared": len(selected),
        "profile": args.profile,
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
