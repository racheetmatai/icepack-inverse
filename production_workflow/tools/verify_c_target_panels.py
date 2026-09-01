"""Independent integrity verifier for the Gate-4 C/target panel bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest_id(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_id", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256-json-v1-" + hashlib.sha256(encoded).hexdigest()


def run(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root / "c_target_panel_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("output_sha256", {})
    actual = {
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if path.is_file()
        and path.name not in {"c_target_panel_manifest.json", "verification_manifest.json"}
    }

    checks = {
        "schema_status": (
            manifest.get("schema") == "jog-c-target-panel-bundle-v1"
            and manifest.get("status") == "complete"
        ),
        "manifest_id": canonical_manifest_id(manifest) == manifest.get("manifest_id"),
        "inventory": actual == set(declared),
        "declared_hashes": True,
        "configuration_set": manifest.get("configurations")
        == [f"CFG{index:02d}" for index in range(1, 7)],
        "square_count": manifest.get("square_count") == 10,
        "target_table": True,
    }
    for relative, expected in declared.items():
        path = root / relative
        checks["declared_hashes"] &= path.is_file() and sha256_file(path) == expected

    table = pd.read_csv(root / "target_distribution_summary.csv")
    expected_pairs = {
        *[(f"SQ{index:02d}", population) for index in range(1, 11)
          for population in ("central_50km", "exclusion_annulus")],
        ("REG_INTER", "heldout"),
        ("REG_PIG", "heldout"),
    }
    actual_pairs = set(zip(table["experiment"], table["population"]))
    numeric = table.select_dtypes(include="number")
    checks["target_table"] &= len(table) == 22 and actual_pairs == expected_pairs
    checks["target_table"] &= numeric.notna().all().all()
    checks["target_table"] &= (table[["training_rows", "heldout_rows"]] > 0).all().all()
    support = table["heldout_inside_training_q01_q99_fraction"]
    checks["target_table"] &= support.between(0, 1, inclusive="both").all()

    for config in manifest["configurations"]:
        stem = f"heldout_c_mosaic_{config.lower()}"
        checks["inventory"] &= {f"{stem}.png", f"{stem}.svg"}.issubset(actual)
    checks["inventory"] &= {
        "square_target_distributions.png", "square_target_distributions.svg",
        "regional_target_distributions.png", "regional_target_distributions.svg",
    }.issubset(actual)

    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "schema": "jog-c-target-panel-bundle-verification-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "parent_manifest_id": manifest.get("manifest_id"),
        "declared_outputs": len(declared),
        "target_rows": len(table),
    }
    result["manifest_id"] = canonical_manifest_id(result)
    (root / "verification_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
