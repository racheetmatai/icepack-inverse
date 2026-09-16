from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from production_training.campaign import (
    _atomic_json, _read_json, _training_process_is_active,
    _validate_requested_shards,
)


class ProductionCampaignTests(unittest.TestCase):
    def test_shard_and_worker_validation(self):
        self.assertEqual(_validate_requested_shards([0, 12], 2), [0, 12])
        with self.assertRaises(ValueError):
            _validate_requested_shards([0, 0], 1)
        with self.assertRaises(ValueError):
            _validate_requested_shards([24], 1)
        with self.assertRaises(ValueError):
            _validate_requested_shards([0], 2)

    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state" / "worker.json"
            _atomic_json(path, {"status": "running", "count": 3})
            self.assertEqual(_read_json(path), {"count": 3, "status": "running"})

    def test_process_owned_heartbeat_freshness(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            marker = work / "active_process.json"
            _atomic_json(marker, {
                "status": "running", "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            })
            self.assertTrue(_training_process_is_active(work))
            _atomic_json(marker, {
                "status": "running",
                "heartbeat_utc": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            })
            self.assertFalse(_training_process_is_active(work))


if __name__ == "__main__":
    unittest.main()
