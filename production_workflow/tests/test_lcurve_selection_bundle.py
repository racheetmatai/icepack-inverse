from pathlib import Path
import sys
import tempfile
import unittest


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT))
sys.path.insert(0, str(WORKFLOW_ROOT / "tools"))

from lcurve_runtime import canonical_identifier
from assemble_lcurve_selection_bundle import (
    _environment_inventory,
    _source_relative,
    _validate_manual_record,
)


def manifest(environment):
    payload = {"environment": environment}
    payload["manifest_id"] = canonical_identifier(payload)
    return payload


class LCurveSelectionBundleTests(unittest.TestCase):
    def setUp(self):
        self.base_environment = {
            "docker_image_hint": "icepack-image-correct",
            "docker_image_id": "sha256:image",
            "executable": "/venv/bin/python",
            "git_head": "abc",
            "git_status": "clean",
            "hostname": "container",
            "platform": "linux",
            "python": "3.10",
            "versions": {"numpy": "1.26"},
        }

    def test_composite_environment_allows_only_recording_differences(self):
        second = dict(self.base_environment)
        second.update(
            docker_image_hint="unrecorded",
            docker_image_id="unrecorded",
            git_status="extra non-scientific files",
        )
        inventory = _environment_inventory(
            [manifest(self.base_environment), manifest(second)],
            {
                "allowed_differences": {
                    "docker_image_hint": "missing metadata",
                    "docker_image_id": "missing metadata",
                    "git_status": "frozen source hashes are checked separately",
                }
            },
        )
        self.assertEqual(
            inventory["observed_differing_keys"],
            ["docker_image_hint", "docker_image_id", "git_status"],
        )
        self.assertEqual(inventory["known_docker_image_ids"], ["sha256:image"])

    def test_composite_environment_rejects_runtime_difference(self):
        second = dict(self.base_environment)
        second["python"] = "3.11"
        with self.assertRaisesRegex(RuntimeError, "non-waived runtime fields"):
            _environment_inventory(
                [manifest(self.base_environment), manifest(second)],
                {"allowed_differences": {"git_status": "not scientific source"}},
            )

    def test_nested_exclusion_inventory_is_normalized(self):
        deviation = {
            "acknowledged": True,
            "kind": "manual window",
            "description": "five points",
            "rationale": "author decision",
            "scope": "selection only",
        }
        _, exclusions = _validate_manual_record(
            {
                "manual_protocol_deviation": deviation,
                "exclusion_inventory": {
                    "manifested": [
                        {"path": "/runs/failed.json", "reason": "failed"}
                    ],
                    "unexecuted": [{"reg_c": 0.2, "reason": "not run"}],
                },
            }
        )
        self.assertEqual(len(exclusions), 2)
        self.assertEqual(exclusions[0]["artifact_path"], "/runs/failed.json")
        self.assertTrue(
            all(row["disposition"] == "excluded_from_selection" for row in exclusions)
        )

    def test_absolute_evidence_path_must_be_inside_source_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inside = root / "point.json"
            self.assertEqual(
                _source_relative(str(inside), source_root=root, label="point"),
                Path("point.json"),
            )
            with self.assertRaisesRegex(ValueError, "outside the source root"):
                _source_relative(
                    str(root.parent / "outside.json"),
                    source_root=root,
                    label="point",
                )


if __name__ == "__main__":
    unittest.main()
