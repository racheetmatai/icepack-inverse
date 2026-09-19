#!/usr/bin/env python3
"""Export per-observation-row fields for ALL six configurations.

Same computation as the archived export_corrected_map_fields.py, which
hardcodes CONFIGS = ("CFG02", "CFG04"); this variant takes --configs and a
separate --output so the archived map_fields/ and the frozen script are left
untouched. Re-exporting CFG02/CFG04 here reproduces the archived arrays,
which is used as a correctness check.

No solve is performed: saved velocity fields are interpolated onto the
observation rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_controlled_campaign import (
    build_object,
    build_observation_alignment,
    interpolate_control,
    interpolate_velocity,
    read_json,
    velocity_path,
)
from production_amundsen import sha256_file

SQUARES = [f"SQ{i:02d}" for i in range(1, 11)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corrected-observations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    object_, adoption, _ = build_object(args.config, args.repo_root, args.adoption)
    frame, lookup = build_observation_alignment(object_, args.dataset)
    corrected = pd.read_csv(args.corrected_observations)
    if corrected["row_id"].duplicated().any():
        raise RuntimeError("Corrected observation row IDs are not unique")
    aligned = corrected.set_index("row_id").reindex(frame["row_id"].astype(str))
    if aligned.isna().any().any():
        raise RuntimeError("Corrected observations do not cover the evaluation rows")
    observed = aligned[["observed_vx_raw", "observed_vy_raw"]].to_numpy(np.float64)
    inversion = interpolate_velocity(object_, np.asarray(adoption["velocity"].dat.data_ro), lookup)
    reference_c = interpolate_control(object_, np.asarray(adoption["C"].dat.data_ro), lookup)
    inversion_error = np.linalg.norm(inversion - observed, axis=1)
    observed_speed = np.linalg.norm(observed, axis=1)

    output_hashes: dict[str, str] = {}
    for configuration in args.configs:
        pieces = []
        for square_number, experiment in enumerate(SQUARES, start=1):
            footprint = frame["square_footprint_id"].eq(experiment).to_numpy()
            primary = frame["square_test_id"].eq(experiment).to_numpy()
            if not np.all(primary <= footprint):
                raise RuntimeError(f"Primary square not contained in footprint: {experiment}")
            model = interpolate_velocity(
                object_, np.load(velocity_path(args.new_root, f"CR_{experiment}_{configuration}_MEDIAN"),
                                 allow_pickle=False), lookup)
            uniform = interpolate_velocity(
                object_, np.load(velocity_path(args.new_root, f"CR_{experiment}_UNIFORM"),
                                 allow_pickle=False), lookup)
            with np.load(args.predictions / f"{experiment}_{configuration}.npz", allow_pickle=False) as archive:
                predicted_c = interpolate_control(object_, archive["median_log_C"].astype(np.float64), lookup)
            index = np.flatnonzero(footprint)
            pieces.append({
                "row_id": frame.loc[footprint, "row_id"].astype(str).to_numpy(),
                "x": frame.loc[footprint, "x"].to_numpy(np.float64),
                "y": frame.loc[footprint, "y"].to_numpy(np.float64),
                "square_number": np.full(len(index), square_number, dtype=np.int16),
                "central_square": primary[index],
                "observed_speed": observed_speed[index],
                "model_error": np.linalg.norm(model[index] - observed[index], axis=1),
                "uniform_error": np.linalg.norm(uniform[index] - observed[index], axis=1),
                "inversion_error": inversion_error[index],
                "C_difference": predicted_c[index] - reference_c[index],
            })
        names = pieces[0].keys()
        combined = {name: np.concatenate([piece[name] for piece in pieces]) for name in names}
        path = args.output / f"{configuration}_ten_square_controlled_fields.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **combined)
        output_hashes[path.name] = sha256_file(path)
        print(f"wrote {path.name} rows={len(combined['row_id'])}", flush=True)

    (args.output / "map_field_export_all_configs_manifest.json").write_text(
        json.dumps({"schema": "jog-map-fields-all-configs-v1", "status": "complete",
                    "configs": list(args.configs), "output_sha256": output_hashes},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output_hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
