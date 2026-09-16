"""Verify full-footprint error exports against accepted central map archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CONFIGS = tuple(f"CFG{i:02d}" for i in range(1, 7))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_id", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def sorted_xyz(x, y, z):
    order = np.lexsort((y, x))
    return x[order], y[order], z[order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--central-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.export_root / "footprint_error_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "schema": manifest.get("schema") == "jog-square-footprint-error-export-v1",
        "status": manifest.get("status") == "complete",
        "manifest_id": manifest.get("manifest_id") == identifier(manifest),
        "six_configs": manifest.get("configs") == list(CONFIGS),
        "ten_squares": manifest.get("square_count") == 10,
        "sixty_controls": manifest.get("verified_median_controls") == 60,
        "output_hashes": True,
        "arrays": True,
        "central_exactly_reproduces_accepted_maps": True,
        "buffers_nonempty": True,
    }
    details = {}
    for name, expected in manifest.get("output_sha256", {}).items():
        if sha256_file(args.export_root / name) != expected:
            checks["output_hashes"] = False

    for config in CONFIGS:
        path = args.export_root / f"{config}_ten_square_footprint_errors.npz"
        with np.load(path, allow_pickle=False) as full:
            required = {"x", "y", "error_magnitude", "central", "square_number"}
            if not required.issubset(full.files):
                checks["arrays"] = False
                continue
            lengths = {len(full[name]) for name in required}
            if len(lengths) != 1 or not np.isfinite(full["error_magnitude"]).all():
                checks["arrays"] = False
            config_details = {}
            for number in range(1, 11):
                full_mask = full["square_number"] == number
                central_mask = full_mask & full["central"].astype(bool)
                buffer_mask = full_mask & ~full["central"].astype(bool)
                if int(buffer_mask.sum()) <= 0:
                    checks["buffers_nonempty"] = False
                accepted_path = (
                    args.central_root / f"SQ{number:02d}_{config}_MEDIAN.npz"
                )
                with np.load(accepted_path, allow_pickle=False) as accepted:
                    fx, fy, fz = sorted_xyz(
                        full["x"][central_mask],
                        full["y"][central_mask],
                        full["error_magnitude"][central_mask],
                    )
                    ax, ay, az = sorted_xyz(
                        accepted["x"], accepted["y"], accepted["error_magnitude"]
                    )
                    exact = (
                        np.array_equal(fx, ax)
                        and np.array_equal(fy, ay)
                        and np.array_equal(fz, az)
                    )
                    if not exact:
                        checks["central_exactly_reproduces_accepted_maps"] = False
                config_details[f"SQ{number:02d}"] = {
                    "full_rows": int(full_mask.sum()),
                    "central_rows": int(central_mask.sum()),
                    "buffer_rows": int(buffer_mask.sum()),
                    "central_exact": bool(exact),
                }
            details[config] = config_details

    passed = all(checks.values())
    result = {
        "schema": "jog-square-footprint-error-export-verification-v1",
        "status": "complete" if passed else "failed",
        "passed": passed,
        "export_manifest_id": manifest.get("manifest_id"),
        "checks": checks,
        "details": details,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    result["manifest_id"] = identifier(result)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": passed, "manifest_id": result["manifest_id"]}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
