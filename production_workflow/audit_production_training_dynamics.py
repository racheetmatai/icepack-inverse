"""Blinded audit of production-training dynamics.

This script reads only run manifests, resolved specifications, training
summaries, history.csv, and learning_rate_history.csv. It deliberately never
opens held-out test data or validation_predictions.csv.gz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MAX_EPOCHS = 1500
EARLY_STOPPING_PATIENCE = 45
LR_PATIENCE = 15
MIN_DELTA = 1.0e-4
TAIL_WINDOW = EARLY_STOPPING_PATIENCE


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def audit_run(run_dir: Path) -> dict:
    manifest = read_json(run_dir / "run_manifest.json")
    summary = read_json(run_dir / "training_summary.json")
    spec = read_json(run_dir / "resolved_spec.json")
    history = pd.read_csv(run_dir / "history.csv")
    lr_history = pd.read_csv(run_dir / "learning_rate_history.csv")

    # Hard guards against silently changing the frozen protocol.
    policy = spec["policy"]
    assert int(policy["max_epochs"]) == MAX_EPOCHS
    assert int(policy["early_stopping"]["patience"]) == EARLY_STOPPING_PATIENCE
    assert int(policy["reduce_lr"]["patience"]) == LR_PATIENCE
    assert float(policy["early_stopping"]["min_delta"]) == MIN_DELTA
    assert float(policy["reduce_lr"]["min_delta"]) == MIN_DELTA
    assert manifest["status"] == "complete"
    assert manifest["mode"] == "model-training"
    assert bool(summary["all_history_finite"])

    epochs = int(summary["epochs_completed"])
    best_epoch = int(summary["best_epoch_one_based"])
    assert len(history) == epochs == len(lr_history)
    assert history["epoch"].tolist() == list(range(1, epochs + 1))
    assert lr_history["epoch"].tolist() == list(range(1, epochs + 1))

    val = history["val_data_mse"].to_numpy(dtype=float)
    lr = lr_history["learning_rate"].to_numpy(dtype=float)
    assert np.isfinite(val).all() and np.isfinite(lr).all()
    assert best_epoch == int(np.argmin(val)) + 1
    assert np.isclose(float(summary["best_val_data_mse_scaled"]), val.min())

    tail = val[-min(TAIL_WINDOW, epochs):]
    tail_slope = linear_slope(tail)
    fitted_tail_change = tail_slope * max(len(tail) - 1, 0)
    meaningful_tail_decrease = bool(fitted_tail_change < -MIN_DELTA)
    best_within_patience = bool(best_epoch > MAX_EPOCHS - EARLY_STOPPING_PATIENCE)
    reached_cap = summary["termination"] == "maximum_epochs"
    potential_cap_constraint = bool(
        reached_cap and (best_within_patience or meaningful_tail_decrease)
    )

    lr_changes = int(np.count_nonzero(~np.isclose(lr[1:], lr[:-1], rtol=1e-7, atol=1e-12)))
    job = manifest["data_identity"]["job"]

    return {
        "job_id": manifest["job_id"],
        "experiment": job["experiment"],
        "configuration": job["configuration"],
        "member": int(job["member"]),
        "termination": summary["termination"],
        "epochs_completed": epochs,
        "best_epoch_one_based": best_epoch,
        "epochs_after_exact_best": epochs - best_epoch,
        "cap_headroom_after_best": MAX_EPOCHS - best_epoch,
        "best_epoch_fraction": best_epoch / MAX_EPOCHS,
        "best_val_data_mse_scaled": float(val.min()),
        "final_val_data_mse_scaled": float(val[-1]),
        "final_minus_best_val_mse": float(val[-1] - val.min()),
        "final_over_best_val_mse": float(val[-1] / val.min()),
        "initial_learning_rate": float(lr[0]),
        "terminal_learning_rate": float(lr[-1]),
        "learning_rate_reductions": lr_changes,
        "tail_window_epochs": len(tail),
        "tail_linear_slope_per_epoch": tail_slope,
        "tail_fitted_change": fitted_tail_change,
        "meaningful_tail_decrease": meaningful_tail_decrease,
        "best_within_final_patience_window": best_within_patience,
        "potential_cap_constraint": potential_cap_constraint,
        "run_manifest_id": manifest["manifest_id"],
    }


def grouped_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(column, sort=True):
        rows.append({
            column: value,
            "runs": len(group),
            "early_stopping_runs": int((group["termination"] == "early_stopping").sum()),
            "maximum_epoch_runs": int((group["termination"] == "maximum_epochs").sum()),
            "potential_cap_constraint_runs": int(group["potential_cap_constraint"].sum()),
            "epochs_median": float(group["epochs_completed"].median()),
            "epochs_q05": float(group["epochs_completed"].quantile(0.05)),
            "epochs_q95": float(group["epochs_completed"].quantile(0.95)),
            "best_epoch_median": float(group["best_epoch_one_based"].median()),
            "terminal_lr_median": float(group["terminal_learning_rate"].median()),
            "lr_reductions_median": float(group["learning_rate_reductions"].median()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs_root = args.campaign_root.resolve() / "runs"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    if len(run_dirs) != 660:
        raise ValueError(f"Expected 660 run directories; found {len(run_dirs)}")

    records = []
    for index, run_dir in enumerate(run_dirs, start=1):
        records.append(audit_run(run_dir))
        if index % 50 == 0 or index == len(run_dirs):
            print(f"Audited {index}/{len(run_dirs)} runs", flush=True)

    frame = pd.DataFrame(records).sort_values("job_id").reset_index(drop=True)
    by_configuration = grouped_summary(frame, "configuration")
    by_experiment = grouped_summary(frame, "experiment")

    counts = frame["termination"].value_counts().to_dict()
    flagged = frame.loc[frame["potential_cap_constraint"]]
    decision = {
        "schema": "jog-production-training-dynamics-audit-v1",
        "status": "complete",
        "blinded_to_held_out_test_outcomes": True,
        "inputs_opened": [
            "run_manifest.json", "resolved_spec.json", "training_summary.json",
            "history.csv", "learning_rate_history.csv",
        ],
        "inputs_explicitly_not_opened": ["validation_predictions.csv.gz", "held-out test data"],
        "frozen_rules": {
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "learning_rate_patience": LR_PATIENCE,
            "min_delta": MIN_DELTA,
            "potential_cap_constraint": (
                "termination=maximum_epochs and either exact best is within final "
                "45 epochs or fitted validation-MSE decrease over final 45 epochs exceeds 1e-4"
            ),
        },
        "counts": {
            "runs": len(frame),
            "early_stopping": int(counts.get("early_stopping", 0)),
            "maximum_epochs": int(counts.get("maximum_epochs", 0)),
            "potential_cap_constraint": int(len(flagged)),
            "best_within_final_patience_window": int(frame["best_within_final_patience_window"].sum()),
            "meaningful_tail_decrease": int(frame["meaningful_tail_decrease"].sum()),
        },
        "epoch_distribution": {
            "minimum": int(frame["epochs_completed"].min()),
            "q05": float(frame["epochs_completed"].quantile(0.05)),
            "median": float(frame["epochs_completed"].median()),
            "q95": float(frame["epochs_completed"].quantile(0.95)),
            "maximum": int(frame["epochs_completed"].max()),
        },
        "flagged_job_ids": flagged["job_id"].tolist(),
        "decision": (
            "no_cap_problem_detected_keep_frozen_checkpoints"
            if len(flagged) == 0
            else "potential_cap_constraint_requires_author_review_before_forward_evaluation"
        ),
    }

    frame.to_csv(output / "run_dynamics.csv", index=False)
    by_configuration.to_csv(output / "by_configuration.csv", index=False)
    by_experiment.to_csv(output / "by_experiment.csv", index=False)
    (output / "audit_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    inventory = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_sha256.json":
            inventory[path.name] = file_sha256(path)
    (output / "output_sha256.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
