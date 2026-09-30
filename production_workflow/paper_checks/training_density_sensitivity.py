"""Sensitivity of the training-density analysis to the neighbour count and to a 40 km separation.

Read-only. Reuses the saved transformed predictors and per-row C errors of
production_workflow/training_representation_diagnostic_20260909_a and the same
member training splits, held-out rows, reference samples (same seeds) and
percentile definition as production_workflow/analyze_training_representation.py.
Only the neighbour search changes:

  k20        baseline (must reproduce the published percentiles)
  k100, k500 larger neighbour counts, no separation
  k20_sep40  k = 20, neighbours must lie >= 40 km away (max(|dx|, |dy|)),
  k100_sep40 k = 100, same separation
             applied to both the held-out rows and the reference training rows.

Supports the appendix statement that the training-density result does not
depend on the neighbour count or on a 40 km separation. Run in the
Icepack/Firedrake environment (about 7 minutes on 16 processes):

    python production_workflow/paper_checks/training_density_sensitivity.py \
        --artifact-dir /path/to/unpacked_artifacts --workdir /tmp/density_sens --workers 16

--workdir is scratch space for compact arrays and the output tables.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

SEED = 20260909
REFERENCE_QUERIES = 10_000
SEPARATION_M = 40_000.0
DIAGNOSTIC = "production_workflow/training_representation_diagnostic_20260909_a"
DATASET = "production_runs/gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz"
SPLITS = "production_runs/gate2_split_manifests_20260820_a/member_splits"
VARIANTS = {"k20": (20, False), "k100": (100, False), "k500": (500, False),
            "k20_sep40": (20, True), "k100_sep40": (100, True)}
CASES = [(f"SQ{i:02d}", c) for i in range(1, 11) for c in ("CFG02", "CFG04")] + [("REG_PIG", "CFG02")]

_STATE: dict = {}


def stable_seed(experiment: str, member: int, purpose: int) -> int:
    order = 11 if experiment == "REG_PIG" else int(experiment[-2:])
    return SEED + purpose * 100_000 + order * 100 + member


def midrank_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return 100.0 * (left + right) / (2.0 * len(ordered))


def load_frame(artifacts: Path) -> pd.DataFrame:
    frame = pd.read_csv(artifacts / DATASET, usecols=["row_id", "x", "y", "common_eligible", "square_test_id", "region_code"],
                        low_memory=False)
    frame = frame.loc[frame["common_eligible"].astype(bool)].sort_values("row_id", kind="stable")
    return frame.reset_index(drop=True)


def query_indices(frame: pd.DataFrame, experiment: str) -> np.ndarray:
    if experiment.startswith("SQ"):
        return np.flatnonzero(frame["square_test_id"].eq(experiment).to_numpy())
    return np.flatnonzero(frame["region_code"].to_numpy(np.int8) == 1)


def kth_plain(tree, queries, k, self_local=None):
    """k-th neighbour distance; with self_local, the query row itself is skipped."""
    if self_local is None:
        d, _ = tree.query(queries, k=k)
        return d[:, k - 1] if k > 1 else d
    d, i = tree.query(queries, k=k + 1)
    out = np.empty(len(queries))
    for r in range(len(queries)):
        keep = i[r] != self_local[r]
        out[r] = d[r][keep][k - 1] if np.count_nonzero(keep) >= k else d[r][k - 1]
    return out


def kth_separated(tree, train_xy, queries, query_xy, k):
    """Exact k-th neighbour distance among training rows at least SEPARATION_M away.

    Asks for progressively more candidates in increasing distance order; the k-th
    geographically valid candidate is exact once at least k valid ones are found.
    """
    out = np.full(len(queries), np.nan)
    todo = np.arange(len(queries))
    candidates = k + 64
    n_train = len(train_xy)
    while len(todo):
        candidates = min(candidates, n_train)
        batch = max(1, int(4e6 // candidates))
        remaining = []
        for start in range(0, len(todo), batch):
            rows = todo[start:start + batch]
            d, i = tree.query(queries[rows], k=candidates)
            d = np.atleast_2d(d); i = np.atleast_2d(i)
            far = np.maximum(np.abs(train_xy[i, 0] - query_xy[rows, None, 0]),
                             np.abs(train_xy[i, 1] - query_xy[rows, None, 1])) >= SEPARATION_M
            count = far.sum(axis=1)
            done = count >= k
            if np.any(done):
                position = np.argmax(np.cumsum(far[done], axis=1) >= k, axis=1)
                out[rows[done]] = d[done][np.arange(done.sum()), position]
            remaining.append(rows[~done])
        todo = np.concatenate(remaining) if remaining else np.array([], dtype=int)
        if len(todo) and candidates >= n_train:
            raise RuntimeError("Fewer than k training rows lie beyond the separation distance")
        candidates *= 8
    return out


def init_worker(workdir: str, artifacts: str) -> None:
    # Workers read compact arrays written once by the main process, not the CSV.
    _STATE["workdir"] = Path(workdir)
    _STATE["splits"] = Path(artifacts) / SPLITS
    _STATE["xy"] = np.load(Path(workdir) / "xy.npy", mmap_mode="r")
    _STATE["features"] = {c: np.load(Path(artifacts) / DIAGNOSTIC / f"transformed_{c}.npy", mmap_mode="r")
                          for c in ("CFG02", "CFG04")}


def member_task(args):
    experiment, config, member = args
    xy = np.asarray(_STATE["xy"])
    features = np.asarray(_STATE["features"][config])
    split = np.load(_STATE["splits"] / f"{experiment}.npz", allow_pickle=False)
    train_mask = np.unpackbits(split["train"][member - 1], bitorder="little", count=len(xy)).astype(bool)
    queries = np.load(_STATE["workdir"] / f"queries_{experiment}.npy")
    assert not train_mask[queries].any()
    training = features[train_mask]
    train_xy = xy[train_mask]
    tree = cKDTree(training, balanced_tree=True, compact_nodes=True)
    rng = np.random.default_rng(stable_seed(experiment, member, 1))
    ref_local = np.sort(rng.choice(len(training), size=min(REFERENCE_QUERIES, len(training)), replace=False))
    result, timing = {}, {}
    for name, (k, separated) in VARIANTS.items():
        t0 = time.time()
        if separated:
            held = kth_separated(tree, train_xy, features[queries], xy[queries], k)
            ref = kth_separated(tree, train_xy, training[ref_local], train_xy[ref_local], k)
        else:
            held = kth_plain(tree, features[queries], k)
            ref = kth_plain(tree, training[ref_local], k, self_local=ref_local)
        result[name] = midrank_percentile(ref, held)
        timing[name] = round(time.time() - t0, 1)
    return experiment, config, member, result, timing


def summarize(points: pd.DataFrame, pct: np.ndarray) -> dict:
    category = np.where(pct <= 50, "<=50", np.where(pct <= 95, "50-95", ">95"))
    out = {}
    for label in ("<=50", "50-95", ">95"):
        sel = category == label
        out[f"C_rmse_{label}"] = float(np.sqrt(np.mean(points["C_squared_difference"].to_numpy()[sel]))) if sel.any() else np.nan
        out[f"rows_{label}"] = int(sel.sum())
    out["spearman_p_vs_abs_C_error"] = float(spearmanr(pct, points["C_absolute_difference"].to_numpy()).statistic)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="Unpacked Zenodo artifacts (contains production_runs/ and production_workflow/)")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--cases", default="all", help="'all' or e.g. SQ01:CFG02")
    parser.add_argument("--members", default="1-10")
    args = parser.parse_args()
    cases = CASES if args.cases == "all" else [tuple(c.split(":")) for c in args.cases.split(",")]
    lo, hi = (int(v) for v in args.members.split("-"))
    tasks = [(e, c, m) for e, c in cases for m in range(lo, hi + 1)]
    artifacts = args.artifact_dir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(artifacts / DIAGNOSTIC / "c_diagnostic/point_c_diagnostics.csv.gz")
    frame = load_frame(artifacts)
    np.save(args.workdir / "xy.npy", frame[["x", "y"]].to_numpy(np.float64))
    for experiment in {e for e, _ in cases}:
        np.save(args.workdir / f"queries_{experiment}.npy", query_indices(frame, experiment))
    per_member: dict = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(args.workdir), str(artifacts))) as pool:
        for experiment, config, member, result, timing in pool.map(member_task, tasks):
            per_member.setdefault((experiment, config), {})[member] = result
            print(json.dumps({"case": f"{experiment}_{config}_M{member:02d}", "seconds": timing}), flush=True)
    rows = []
    for (experiment, config), members in per_member.items():
        case_points = points[(points.experiment == experiment) & (points.configuration == config)]
        order = frame.loc[query_indices(frame, experiment), "row_id"].to_numpy(str)
        case_points = case_points.set_index("row_id").loc[order].reset_index()
        for name in VARIANTS:
            pct = np.median(np.vstack([members[m][name] for m in sorted(members)]), axis=0)
            row = {"experiment": experiment, "configuration": config, "variant": name, "members": len(members)}
            row.update(summarize(case_points, pct))
            if name == "k20":
                published = case_points["representation_percentile"].to_numpy()
                row["max_abs_diff_vs_published_percentile"] = float(np.max(np.abs(pct - published)))
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.workdir / "density_sensitivity_cases.csv", index=False)
    squares = table[table.experiment.str.startswith("SQ")]
    medians = squares.groupby(["configuration", "variant"])[
        ["C_rmse_<=50", "C_rmse_50-95", "C_rmse_>95", "spearman_p_vs_abs_C_error"]].median()
    medians.to_csv(args.workdir / "density_sensitivity_square_medians.csv")
    print(medians.round(3).to_string())
    print(table[table.experiment == "REG_PIG"].round(3).to_string())
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
