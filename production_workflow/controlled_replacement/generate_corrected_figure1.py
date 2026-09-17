#!/usr/bin/env python3
"""Regenerate Figure 1 with original paired MEaSUREs observations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")


WORKFLOW = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKFLOW))
import generate_revision_figures_and_tables as base

from generate_corrected_inversion_panels import aggregate


HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("JOG_CONTROLLED_FIGURES", HERE / "figures"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    base.style()
    _, values = aggregate()
    OUT.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    base.aggregate_velocity_grid = lambda: values
    paths = base.figure1()
    selected = [
        path for path in paths
        if path.suffix.lower() in {".pdf", ".png"}
        and path.stem in {
            "figure1a_observed_speed_and_regions",
            "figure1b_holdout_geometry",
        }
    ]
    record = {
        "schema": "jog-corrected-figure1-v1",
        "status": "complete",
        "observations": "original paired MEaSUREs raster values at verified pixel centres",
        "aggregation": "mean observed speed within 5 km display cells",
        "outputs": {path.name: sha256(path) for path in selected},
    }
    (OUT / "figure1_corrected_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
