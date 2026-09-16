#!/usr/bin/env python3
"""Check that every frozen paper figure is present and non-empty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    expected = sorted(
        path.relative_to(args.reference)
        for path in args.reference.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".png"}
    )
    missing = [str(path) for path in expected if not (args.candidate / path).is_file()]
    empty = [
        str(path)
        for path in expected
        if (args.candidate / path).is_file()
        and (args.candidate / path).stat().st_size == 0
    ]
    result = {
        "schema": "jog-paper-output-verification-v1",
        "passed": not missing and not empty,
        "expected": len(expected),
        "missing": missing,
        "empty": empty,
        "note": "File presence check only; this does not verify numerical or visual agreement.",
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
