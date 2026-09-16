#!/usr/bin/env python3
"""Regenerate manuscript figures from the archived analysis artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


INTRODUCTION = [
    "figure1a_observed_speed_and_regions.pdf",
    "figure1b_holdout_geometry.pdf",
    "figure3a_observed_speed.pdf",
    "figure3b_inversion_reference_c.pdf",
    "figure3c_inversion_velocity_residual.pdf",
]
RESULTS = [
    "figure3_square_velocity_rmse.pdf",
    "figure4a_input_support.pdf",
    "figure4b_relative_velocity_skill.pdf",
    "figure5_spatial_control_velocity_uniform_comparison.pdf",
    "figure6a_pig_cfg02_control_difference.pdf",
    "figure6b_pig_cfg02_velocity_error.pdf",
    "figure6c_pig_cfg02_uniform_comparison.pdf",
]
APPENDIX = [
    "appendix_training_convergence.pdf",
    "lcurve_appendix.png",
    "method_overview.pdf",
    "regional_target_distributions.png",
    "square_target_distributions.png",
]


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def copy_required(name: str, sources: list[Path], destination: Path) -> None:
    for source in sources:
        candidate = source / name
        if candidate.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            return
    raise FileNotFoundError(f"Could not locate generated figure: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reproduced_paper"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    artifacts = args.artifact_dir.resolve()
    output = args.output.resolve()
    work = output / "_work"
    figures = output / "figures"
    if not (artifacts / "production_runs").is_dir():
        raise SystemExit("The unpacked artifact directory must contain production_runs/.")

    env = os.environ.copy()
    env["JOG_REPOSITORY_ROOT"] = str(repo)
    env["JOG_ARTIFACT_ROOT"] = str(artifacts)
    env["JOG_OUTPUT_ROOT"] = str(work)

    if not args.verify_only:
        work.mkdir(parents=True, exist_ok=True)
        python = sys.executable
        workflow = repo / "production_workflow"

        run([python, str(workflow / "generate_revision_figures_and_tables.py"), "--paper-only"], env)
        run([python, str(workflow / "generate_proposed_figure5.py")], env)
        run([python, str(workflow / "generate_pig_cfg02_spatial_diagnostic.py")], env)
        run([python, str(workflow / "generate_training_convergence_figure.py")], env)

        corrected = (
            artifacts
            / "production_runs/gate4_forward_evaluation_support_aligned_20260910"
            / "control_population_metrics.csv"
        )
        run(
            [python, str(workflow / "generate_figure3_square_rmse.py"), str(corrected), str(work)],
            env,
        )

        target_out = work / "target_distributions"
        run(
            [
                python,
                str(workflow / "assemble_c_target_panels.py"),
                "--predictions",
                str(artifacts / "production_runs/gate3_full_mesh_ensemble_predictions_20260828_a"),
                "--dataset",
                str(artifacts / "production_runs/gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz"),
                "--splits",
                str(artifacts / "production_runs/gate2_split_manifests_20260820_a"),
                "--squares",
                str(artifacts / "production_workflow/frozen_design/selected_squares.csv"),
                "--outline",
                str(artifacts / "data/geojson/amundsen_v1.geojson"),
                "--output",
                str(target_out),
            ],
            env,
        )

        method_path = work / "method_overview.pdf"
        method_env = env.copy()
        method_env["JOG_METHOD_OVERVIEW_OUTPUT"] = str(method_path)
        run([python, str(workflow / "generate_method_overview.py")], method_env)

        lcurve = (
            artifacts
            / "production_runs/gate1_lcurve_selection_bundle_20260819_a/lcurve_appendix.png"
        )
        shutil.copy2(lcurve, work / "lcurve_appendix.png")

        sources = [work, target_out]
        for name in INTRODUCTION:
            copy_required(name, sources, figures / "introduction_methods" / name)
        for name in RESULTS:
            copy_required(name, sources, figures / "results" / name)
        for name in APPENDIX:
            copy_required(name, sources, figures / "appendix" / name)

    run(
        [
            sys.executable,
            str(repo / "scripts/verify_paper_outputs.py"),
            "--reference",
            str(repo / "manuscript/figures"),
            "--candidate",
            str(figures),
        ],
        env,
    )

    import pandas as pd
    verified_tables = []
    archived = artifacts / 'production_workflow/final_figures_20260830_a'
    for name in ('table1_predictors_and_configurations.csv',
                 'table2_primary_performance.csv',
                 'table3_high_support_low_skill_examples.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(work / name), pd.read_csv(archived / name))
        verified_tables.append(name)
    report = {'passed': True, 'tables_matching_archived_values': verified_tables,
              'figure_presence_verified': True,
              'rendering': 'Run compare_figure_rendering.py for visual comparison.'}
    (output / 'reproduction_verification.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == "__main__":
    main()
