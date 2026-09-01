#!/usr/bin/env python3
"""Verify a production forward-smoke manifest and every declared output."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("run_directory", type=Path)
args = parser.parse_args()
root = args.run_directory.resolve()
manifest_path = root / "forward_smoke_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
unsigned = dict(manifest)
observed_identifier = unsigned.pop("manifest_id")
payload = json.dumps(
    unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode("utf-8")
expected_identifier = "sha256-json-v1-" + hashlib.sha256(payload).hexdigest()
bad_outputs = {}
for relative, expected_hash in manifest["output_sha256"].items():
    path = root / relative
    observed_hash = sha256_file(path) if path.is_file() else None
    if observed_hash != expected_hash:
        bad_outputs[relative] = {
            "expected": expected_hash,
            "observed": observed_hash,
        }
result = {
    "status": manifest["status"],
    "manifest_id": observed_identifier,
    "manifest_id_valid": observed_identifier == expected_identifier,
    "declared_output_count": len(manifest["output_sha256"]),
    "bad_output_count": len(bad_outputs),
    "check_count": len(manifest["checks"]),
    "all_checks_pass": bool(manifest["checks"]) and all(manifest["checks"].values()),
    "bad_outputs": bad_outputs,
}
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(
    0
    if result["status"] == "pass"
    and result["manifest_id_valid"]
    and not bad_outputs
    and result["all_checks_pass"]
    else 1
)
