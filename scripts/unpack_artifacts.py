#!/usr/bin/env python3
"""Extract verified Zenodo archives into the expected workflow layout."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
    archive.extractall(destination, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("paper", "full", "all"), default="paper")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/artifacts.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output or (args.artifact_dir / "unpacked")
    output.mkdir(parents=True, exist_ok=True)
    selected = [
        item
        for item in manifest["archives"]
        if args.profile == "all" or args.profile in item.get("profiles", [])
    ]
    for item in selected:
        path = args.artifact_dir / item["filename"]
        with tarfile.open(path, "r:gz") as archive:
            safe_extract(archive, output)
    print(output)


if __name__ == "__main__":
    main()
