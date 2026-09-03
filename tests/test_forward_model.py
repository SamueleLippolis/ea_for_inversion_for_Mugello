import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake_forward_model import (
    SourceModelParameters,
    evaluate_forward_model,
    load_intensity_observations,
)


class ForwardModelTest(unittest.TestCase):
    def test_colline_pisane_forward_model_for_all_sites(self):
        model = SourceModelParameters(43.475, 10.55, 349.5, 79.5, 45.0, 9.0,
                                      13.0, 13.0, 0.54, -0.54, 3.69, 2e24)
        observations = load_intensity_observations(
            ROOT / "data/liv1846-n108.txt")
        result = evaluate_forward_model(model, observations)
        self.assertEqual(len(result.predicted_intensities), 108)
        self.assertTrue(math.isfinite(result.residual))
        self.assertEqual(result.residual, 146.0)
        self.assertEqual(result.predicted_intensities[:3], (7, 6, 7))
        self.assertTrue(all(1 <= value <= 11 for value in result.predicted_intensities))
        self.assertTrue(all(value >= 0 for value in result.kinematic_values))


if __name__ == "__main__":
    unittest.main()
