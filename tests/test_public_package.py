from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicPackageTests(unittest.TestCase):
    def test_artifact_manifest_schema(self) -> None:
        payload = json.loads((ROOT / "configs/artifacts.json").read_text())
        self.assertEqual(payload["schema"], "jog-reproduction-artifacts-v1")
        self.assertIn("archives", payload)

    def test_no_private_provenance_note(self) -> None:
        self.assertFalse((ROOT / "PRIVATE_PROVENANCE_NOTES.md").exists())

    def test_required_entry_points_exist(self) -> None:
        required = [
            "scripts/verify_artifacts.py",
            "scripts/unpack_artifacts.py",
            "scripts/reproduce_paper.py",
            "production_workflow/run_production.sh",
            "production_workflow/controlled_replacement/build_controlled_controls.py",
            "production_workflow/controlled_replacement/build_intercatchment_controls.py",
            "production_workflow/controlled_replacement/evaluate_controlled_campaign.py",
            "production_workflow/controlled_replacement/export_map_fields_all_configs.py",
            "production_workflow/analyze_transfer_predictability.py",
            "production_workflow/analyze_transfer_predictability_counts.py",
            "production_workflow/check_transfer_classifier_thresholds.py",
            "production_workflow/check_transfer_classifier_fit.py",
            "production_workflow/generate_appendix_transfer_predictability_figure.py",
            "production_workflow/generate_appendix_transfer_counts_figure.py",
            "production_workflow/generate_appendix_eligibility_map.py",
            "icepack-mlp/production_training/train.py",
        ]
        for name in required:
            self.assertTrue((ROOT / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
