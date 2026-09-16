from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from production_training.data import load_prepared_job
from production_training.integrity import canonical_manifest_id, file_sha256
from production_training.spec import FEATURE_CONFIGURATIONS, resolved_run_spec
from production_training.select_l2 import select_candidate


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class PortableTrainingTests(unittest.TestCase):
    def test_frozen_l2_selection_requires_meaningful_improvement(self):
        records = [{"lambda_L2": value, "best_val_data_mse_scaled": mse} for value, mse in (
            (0.0, 0.10000), (1e-6, 0.09995), (1e-5, 0.10020), (1e-4, 0.10100))]
        selected, _ = select_candidate(records)
        self.assertEqual(selected["lambda_L2"], 0.0)
        records[1]["best_val_data_mse_scaled"] = 0.09980
        selected, _ = select_candidate(records)
        self.assertEqual(selected["lambda_L2"], 1e-6)

    def test_frozen_spec_rejects_invalid_changes(self):
        self.assertEqual(len(FEATURE_CONFIGURATIONS), 6)
        self.assertEqual(resolved_run_spec("CFG06", 1e-5)["policy"]["max_epochs"], 1500)
        with self.assertRaises(ValueError):
            resolved_run_spec("CFG06", -1)
        with self.assertRaises(ValueError):
            resolved_run_spec("CFG06", 0, max_epochs=1501)

    def test_exact_membership_and_train_only_scalers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); dataset = root / "dataset"; splits = root / "splits"
            dataset.mkdir(); (splits / "member_splits").mkdir(parents=True)
            row_ids = np.asarray([f"row-{i:02d}" for i in range(20)], dtype=object)
            features = FEATURE_CONFIGURATIONS["CFG01"]
            frame = pd.DataFrame({"row_id": row_ids, "common_eligible": True, "reference_log_C": np.arange(20.0)})
            for offset, feature in enumerate(features):
                frame[feature] = np.arange(20.0) + offset
            data_path = dataset / "canonical_master_dataset.csv.gz"
            frame.to_csv(data_path, index=False, compression={"method": "gzip", "mtime": 0})
            dataset_manifest = {
                "schema": "jog-canonical-master-dataset-v1", "status": "complete", "common_eligible_count": 20,
                "output_sha256": {data_path.name: file_sha256(data_path)},
            }
            dataset_manifest["manifest_id"] = canonical_manifest_id(dataset_manifest)
            write_json(dataset / "dataset_manifest.json", dataset_manifest)

            index_path = splits / "sorted_common_eligible_row_ids.txt.gz"
            with index_path.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as binary:
                    binary.write(("\n".join(row_ids) + "\n").encode())
            train = np.zeros(20, bool); train[:14] = True
            validation = np.zeros(20, bool); validation[14:18] = True
            split_path = splits / "member_splits" / "SQ01.npz"
            np.savez_compressed(
                split_path, row_count=np.int64(20), bitorder=np.asarray("little"), members=np.asarray([1]),
                train=np.asarray([np.packbits(train, bitorder="little")]),
                validation=np.asarray([np.packbits(validation, bitorder="little")]),
                split_seeds=np.asarray([11]), model_seeds=np.asarray([21]), shuffle_seeds=np.asarray([31]),
            )
            job = {"job_id": "SQ01_CFG01_M01", "experiment": "SQ01", "configuration": "CFG01", "member": "1",
                   "split_id": "SQ01_M01", "split_seed": "11", "model_seed": "21", "shuffle_seed": "31",
                   "split_file": "member_splits/SQ01.npz"}
            member = {"split_id": "SQ01_M01", "experiment": "SQ01", "member": "1", "offset": "0",
                      "split_seed": "11", "model_seed": "21", "shuffle_seed": "31", "train_rows": "14",
                      "validation_rows": "4", "train_membership_id": "train-id", "validation_membership_id": "val-id",
                      "split_file": "member_splits/SQ01.npz", "split_file_sha256": file_sha256(split_path)}
            write_csv(splits / "job_registry.csv", [job]); write_csv(splits / "member_splits.csv", [member])
            outputs = {
                "job_registry.csv": file_sha256(splits / "job_registry.csv"),
                "member_splits.csv": file_sha256(splits / "member_splits.csv"),
                "member_splits/SQ01.npz": file_sha256(split_path),
                "sorted_common_eligible_row_ids.txt.gz": file_sha256(index_path),
            }
            split_manifest = {
                "schema": "jog-training-split-bundle-v1", "status": "complete",
                "dataset_manifest_id": dataset_manifest["manifest_id"], "dataset_sha256": file_sha256(data_path),
                "eligible_rows": 20, "row_index_sha256": file_sha256(index_path), "output_sha256": outputs,
            }
            split_manifest["manifest_id"] = canonical_manifest_id(split_manifest)
            write_json(splits / "split_bundle_manifest.json", split_manifest)

            prepared = load_prepared_job(dataset, splits, job["job_id"])
            self.assertEqual(prepared.x_train.shape, (14, 6))
            self.assertEqual(prepared.x_validation.shape, (4, 6))
            self.assertEqual(prepared.train_row_ids.tolist(), row_ids[:14].tolist())
            self.assertEqual(prepared.validation_row_ids.tolist(), row_ids[14:18].tolist())
            self.assertAlmostEqual(float(prepared.input_scaler.center_[0]), 6.5)
            self.assertAlmostEqual(float(prepared.target_scaler.center_[0]), 6.5)


if __name__ == "__main__":
    unittest.main()
