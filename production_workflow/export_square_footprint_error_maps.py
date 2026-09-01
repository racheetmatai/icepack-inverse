"""Export observation-grid errors over all ten complete square footprints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_forward_campaign import (
    baseline_registry, build_observation_alignment, build_object,
    interpolate_velocity,
)
from production_amundsen import manifest_identifier, sha256_file


CONFIGS = tuple(f"CFG{i:02d}" for i in range(1, 7))


def median_registry(campaign_root: Path, configs: tuple[str, ...]) -> dict[str, dict]:
    records = {}
    for square_number in range(1, 11):
        square_id = f"SQ{square_number:02d}"
        for config in configs:
            control_id = f"{square_id}_{config}_MEDIAN"
            path = campaign_root / "solves" / control_id / "forward_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            velocity_path = path.parent / manifest["velocity_path"]
            if (
                manifest.get("status") != "complete"
                or manifest.get("control_id") != control_id
                or manifest.get("control_kind") != "median"
                or manifest_identifier(manifest) != manifest.get("manifest_id")
                or sha256_file(velocity_path) != manifest.get("velocity_sha256")
            ):
                raise ValueError(f"Invalid median forward solve: {path}")
            records[control_id] = {
                "velocity_path": velocity_path,
                "forward_manifest_id": manifest["manifest_id"],
            }
    expected = len(configs) * 10
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} verified median controls")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adoption-record", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--squares", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    object_, _, _ = build_object(args.config, args.repo_root, args.adoption_record)
    frame, lookup = build_observation_alignment(object_, args.dataset)
    configs = tuple(args.configs)
    registry = median_registry(args.campaign_root, configs)
    with args.squares.open(newline="", encoding="utf-8") as stream:
        squares = list(csv.DictReader(stream))
    observed = frame[["observed_vx", "observed_vy"]].to_numpy(dtype=float)
    observed_speed = np.linalg.norm(observed, axis=1)
    x = frame.x.to_numpy(dtype=float); y = frame.y.to_numpy(dtype=float)
    baselines = baseline_registry(args.baseline_root)
    square_context = {}
    for sequence, square in enumerate(squares, start=1):
        square_id = square["square_id"]
        full = ((x >= float(square["footprint_xmin_m"])) & (x < float(square["footprint_xmax_m"])) &
                (y >= float(square["footprint_ymin_m"])) & (y < float(square["footprint_ymax_m"])))
        central = ((x >= float(square["test_xmin_m"])) & (x < float(square["test_xmax_m"])) &
                   (y >= float(square["test_ymin_m"])) & (y < float(square["test_ymax_m"])))
        baseline_velocity = np.load(baselines[square_id]["velocity_path"], allow_pickle=False)
        baseline_prediction = interpolate_velocity(object_, baseline_velocity, lookup)
        baseline_squared_error = np.sum((baseline_prediction - observed) ** 2, axis=1)
        square_context[square_id] = {
            "sequence": sequence, "full": full, "central": central,
            "baseline_squared_error": baseline_squared_error[full],
        }
        print(f"Exported {square_id} uniform baseline ({sequence}/10)", flush=True)

    outputs = {}
    for config in configs:
        chunks = []
        for sequence, square in enumerate(squares, start=1):
            square_id = square["square_id"]
            control_id = f"{square_id}_{config}_MEDIAN"
            record = registry[control_id]
            velocity = np.load(record["velocity_path"], allow_pickle=False)
            prediction = interpolate_velocity(object_, velocity, lookup)
            context = square_context[square_id]
            full = context["full"]; central = context["central"]
            model_squared_error = np.sum((prediction - observed) ** 2, axis=1)
            error = np.sqrt(model_squared_error)
            chunks.append((x[full], y[full], error[full], observed_speed[full],
                           model_squared_error[full], context["baseline_squared_error"], central[full],
                           np.full(int(full.sum()), sequence, dtype=np.int8)))
            print(f"Exported {control_id} ({sequence}/10)", flush=True)
        output_path = args.output / f"{config}_ten_square_footprint_errors.npz"
        np.savez_compressed(
            output_path,
            x=np.concatenate([c[0] for c in chunks]),
            y=np.concatenate([c[1] for c in chunks]),
            error_magnitude=np.concatenate([c[2] for c in chunks]),
            observed_speed=np.concatenate([c[3] for c in chunks]),
            model_squared_error=np.concatenate([c[4] for c in chunks]),
            uniform_squared_error=np.concatenate([c[5] for c in chunks]),
            central=np.concatenate([c[6] for c in chunks]),
            square_number=np.concatenate([c[7] for c in chunks]),
        )
        outputs[output_path.name] = sha256_file(output_path)

    manifest = {
        "schema": "jog-square-footprint-error-export-v1",
        "status": "complete",
        "configs": list(configs),
        "square_count": len(squares),
        "verified_median_controls": len(registry),
        "population": "complete_130km_footprint_with_nested_50km_primary_square",
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "squares_sha256": sha256_file(args.squares),
        "output_sha256": outputs,
    }
    manifest["manifest_id"] = manifest_identifier(manifest)
    (args.output / "footprint_error_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "manifest_id": manifest["manifest_id"],
                      "outputs": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
