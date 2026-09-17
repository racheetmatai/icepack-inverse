#!/usr/bin/env python3
"""Export corrected row-level fields for affected square and PIG figures."""

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
    population_mask,
    read_json,
    velocity_path,
)
from production_amundsen import sha256_file


SQUARES = [f"SQ{i:02d}" for i in range(1, 11)]
CONFIGS = ("CFG02", "CFG04")


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


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
    for configuration in CONFIGS:
        pieces = []
        for square_number, experiment in enumerate(SQUARES, start=1):
            footprint = frame["square_footprint_id"].eq(experiment).to_numpy()
            primary = frame["square_test_id"].eq(experiment).to_numpy()
            if not np.all(primary <= footprint):
                raise RuntimeError(f"Primary square is not contained in footprint: {experiment}")
            model = interpolate_velocity(
                object_, np.load(velocity_path(args.new_root, f"CR_{experiment}_{configuration}_MEDIAN"),
                                 allow_pickle=False), lookup
            )
            uniform = interpolate_velocity(
                object_, np.load(velocity_path(args.new_root, f"CR_{experiment}_UNIFORM"),
                                 allow_pickle=False), lookup
            )
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
        save_npz(path, **combined)
        output_hashes[path.name] = sha256_file(path)

    experiment = "REG_PIG"
    heldout = population_mask(frame, experiment)
    model = interpolate_velocity(
        object_, np.load(velocity_path(args.new_root, "CR_REG_PIG_CFG02_MEDIAN"), allow_pickle=False), lookup
    )
    uniform = interpolate_velocity(
        object_, np.load(velocity_path(args.new_root, "CR_REG_PIG_UNIFORM"), allow_pickle=False), lookup
    )
    with np.load(args.predictions / "REG_PIG_CFG02.npz", allow_pickle=False) as archive:
        predicted_c = interpolate_control(object_, archive["median_log_C"].astype(np.float64), lookup)
    index = np.flatnonzero(heldout)
    pig_path = args.output / "REG_PIG_CFG02_controlled_fields.npz"
    save_npz(
        pig_path,
        row_id=frame.loc[heldout, "row_id"].astype(str).to_numpy(),
        x=frame.loc[heldout, "x"].to_numpy(np.float64),
        y=frame.loc[heldout, "y"].to_numpy(np.float64),
        observed_speed=observed_speed[index],
        model_error=np.linalg.norm(model[index] - observed[index], axis=1),
        uniform_error=np.linalg.norm(uniform[index] - observed[index], axis=1),
        inversion_error=inversion_error[index],
        C_difference=predicted_c[index] - reference_c[index],
    )
    output_hashes[pig_path.name] = sha256_file(pig_path)

    record = {
        "schema": "jog-controlled-replacement-map-fields-v1",
        "status": "complete",
        "sampling": "all retained 450 m observation rows",
        "observational_reference": "original paired MEaSUREs raster components at verified pixel centres",
        "replacement": "controlled footprint/catchment only; reference C retained outside",
        "output_sha256": output_hashes,
        "inputs": {
            "dataset_sha256": sha256_file(args.dataset),
            "corrected_observations_sha256": sha256_file(args.corrected_observations),
            "adoption_sha256": sha256_file(args.adoption),
        },
    }
    (args.output / "map_field_export_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
