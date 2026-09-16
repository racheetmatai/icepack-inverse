from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from production_training.model import reload_exact_best_model


class _FakeModel:
    pass


class ExactCheckpointReloadTests(unittest.TestCase):
    def test_exact_checkpoint_is_loaded_without_duplicate_export(self):
        loaded = []
        model = _FakeModel()

        def load_model(path):
            loaded.append(Path(path))
            return model

        fake_tf = SimpleNamespace(keras=SimpleNamespace(models=SimpleNamespace(load_model=load_model)))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = reload_exact_best_model(fake_tf, output)

            self.assertIs(result, model)
            self.assertEqual(loaded, [output / "best_model.keras"])


if __name__ == "__main__":
    unittest.main()
