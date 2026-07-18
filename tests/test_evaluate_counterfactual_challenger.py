import unittest

import numpy as np

from scripts.evaluate_counterfactual_challenger import (
    artifact_pair_probabilities,
)


class CounterfactualChallengerEvaluationTests(unittest.TestCase):
    def test_pair_probability_uses_frozen_artifact_scores(self):
        artifact = {
            "feature_names": ["shape"],
            "base_raw_score": 0.0,
            "learning_rate": 0.1,
            "calibration": {"scale": 1.0, "intercept": 0.0},
            "trees": [],
            "linear_correction": {
                "feature_indices": [0],
                "coefficients": [1.0],
            },
        }
        records = [
            {
                "left_features": np.asarray([2.0]),
                "right_features": np.asarray([-2.0]),
            }
        ]
        probability = artifact_pair_probabilities(records, artifact)
        self.assertAlmostEqual(probability[0], 1.0 / (1.0 + np.exp(-4.0)))


if __name__ == "__main__":
    unittest.main()
