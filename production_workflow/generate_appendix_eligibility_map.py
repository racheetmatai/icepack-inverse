#!/usr/bin/env python3
"""Appendix figure: modeled domain vs. the region used for metrics.

Addresses reviewer comment R-DET-25a (review_text.txt L266): "Metrics are
only reported when phi > 0.1, but that region is not shown anywhere...
There should be a figure explicitly showing the region." The prior response
added text (the population paragraph in Appendix section
"Eligible rows, predictor construction, and support") but, on review, never
added the requested figure: every existing map already restricts display to
the eligible region, so the excluded fringe is never shown for contrast.

Uses only the frozen 5 km support grid
(frozen_design/amundsen_input_support_grid_5km.npz), which is the same grid
every other figure already masks by; no new computation, inversion, or
simulation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_revision_figures_and_tables as base  # noqa: E402

# base.DESIGN honours JOG_ARTIFACT_ROOT in the packaged workflow, so the
# frozen grid is found in an unpacked artifact directory as well as here.
DESIGN = base.DESIGN
OUT = ROOT / "manuscript/figures/appendix"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def support_grid() -> dict[str, np.ndarray]:
    with np.load(DESIGN / "amundsen_input_support_grid_5km.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="directory for the figure files (default: manuscript/figures/appendix)")
    out = parser.parse_args().output
    base.style()
    s = support_grid()
    r = base.region_context()
    inside = s["inside"]
    grounded = s["grounded"]
    velocity_available = s["velocity_available"]
    eligible = s["eligible"]
    if not np.array_equal(eligible, inside & grounded & velocity_available):
        raise RuntimeError("eligible mask no longer equals inside & grounded & velocity_available")

    x, y = s["x_grid"] / 1000.0, s["y_grid"] / 1000.0
    extent = [x[0] - 2.5, x[-1] + 2.5, y[0] - 2.5, y[-1] + 2.5]

    # Category grid: 0 outside modeled domain (transparent), 1 modeled but
    # excluded from metrics (not grounded, or grounded without a paired
    # velocity observation), 2 eligible (used for training and metrics).
    category = np.zeros(inside.shape, dtype=np.int8)
    category[inside & ~eligible] = 1
    category[eligible] = 2
    n_inside = int(inside.sum())
    n_excluded = int((inside & ~eligible).sum())
    n_eligible = int(eligible.sum())

    display = np.ma.masked_where(category == 0, category)
    cmap = ListedColormap(["#f4a582", "#4477AA"])  # excluded, eligible
    norm = BoundaryNorm([0.5, 1.5, 2.5], 2)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 6.2), constrained_layout=True)
    ax.imshow(display, origin="lower", extent=extent, cmap=cmap, norm=norm)
    base.map_axes(ax, r["outline"])
    base.add_antarctica_locator(ax, r["outline"])
    ax.plot([], [], color="#4477AA", lw=6,
            label=f"Eligible: used for training and metrics ({n_eligible/n_inside:.1%} of modeled domain)")
    ax.plot([], [], color="#f4a582", lw=6,
            label=f"Modeled but excluded from metrics ($\\phi\\leq0.1$ or no paired\nvelocity observation; {n_excluded/n_inside:.1%} of modeled domain)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1,
              frameon=True, framealpha=0.95, borderaxespad=0.0, fontsize=9.0)

    out.mkdir(parents=True, exist_ok=True)
    stem = "figure_appendix_eligible_region_map"
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        paths.append(path)
    plt.close(fig)

    record = {
        "schema": "jog-appendix-eligibility-map-v1",
        "status": "complete",
        "addresses": "R-DET-25a (review_text.txt L266): figure explicitly showing the modeled region vs. the region used for metrics",
        "source": "production_workflow/frozen_design/amundsen_input_support_grid_5km.npz (frozen; no new computation)",
        "grid_cells_5km": {"inside_modeled_domain": n_inside, "excluded_from_metrics": n_excluded, "eligible": n_eligible},
        "outputs": {path.name: sha256(path) for path in paths},
    }
    (out / f"{stem}_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
