"""Regression checks for differently ordered eligible/support/evaluation rows."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_forward_campaign import support_categories, verify_unchanged_totals


class SupportAlignmentTests(unittest.TestCase):
    def test_shuffled_ids_partition_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point_support_categories.npz"
            ids = np.array(["d", "a", "c", "b"])
            frame = pd.DataFrame({"row_id": ["a", "b", "c", "d"], "region_code": [1]*4})
            pd.DataFrame([dict(experiment="REG_PIG", population="PIG", configuration="CFG02_best_ice",
                               heldout_rows=4, neither_fraction=.25, marginal_only_fraction=.25,
                               joint_only_fraction=.25, both_fraction=.25)]).to_csv(path.parent / "support_categories.csv", index=False)
            np.savez(path, REG_PIG__row_index=np.arange(4), REG_PIG__CFG02_best_ice=np.arange(4,dtype=np.uint8))
            np.testing.assert_array_equal(support_categories(path,"REG_PIG","CFG02",ids,frame),[1,3,2,0])
            with self.assertRaises(ValueError):
                support_categories(path,"REG_PIG","CFG02",np.array(["d","a","c","c"]),frame)
            for indices, labels in [(np.arange(3),np.arange(3)), (np.arange(4),np.array([0,1,2,255])),
                                    (np.array([0,1,2,2]),np.arange(4))]:
                np.savez(path,REG_PIG__row_index=indices,REG_PIG__CFG02_best_ice=labels)
                with self.assertRaises(ValueError):
                    support_categories(path,"REG_PIG","CFG02",ids,frame)

    def test_whole_population_guard(self):
        old = {"metrics":[{"population":"PIG","support_stratum":"all","rows":4,"rmse":5.0}]}
        verify_unchanged_totals(old,old)
        new = {"metrics":[{"population":"PIG","support_stratum":"all","rows":4,"rmse":6.0}]}
        with self.assertRaises(AssertionError):
            verify_unchanged_totals(new,old)


if __name__ == "__main__":
    unittest.main()
