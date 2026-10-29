"""Regression checks for reproducible, lossless historical data transformation."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("import_legacy", Path(__file__).resolve().parents[1] / "scripts/import_legacy.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArtifactTests(unittest.TestCase):
    def test_per_agent_means_are_not_repeated_training_seeds(self):
        series, raw = module.normalize_summary({
            "C:/private/run/logs/agent0/value_loss/agent0/value_loss": [[100., 1, 2.]],
            "C:/private/run/logs/agent1/value_loss/agent1/value_loss": [[100., 1, 4.]],
            "C:/private/run/logs/Reward_Curve/episode_reward": [[100., 1, -7.]],
        })
        self.assertEqual(series["value_loss"], [{"step": 1, "value": 3.}])
        self.assertEqual(series["episode_reward"][0]["value"], -7.)
        self.assertEqual(len(raw), 3)
        self.assertTrue(all("private" not in key for key in raw))

    def test_invalid_measurement_is_rejected(self):
        with self.assertRaises(ValueError):
            module.normalize_summary({"metric": [[1, 1, float("nan")]]})


if __name__ == "__main__":
    unittest.main()
