import unittest

import numpy as np

from scripts.train_counterfactual_policy import (
    export_policy_correction,
    fit_offset_logistic_correction,
    grouped_split,
    offset_probabilities,
)
from scripts.train_value_model import exported_probabilities


class CounterfactualPolicyTrainingTests(unittest.TestCase):
    def test_offset_correction_learns_without_recalibrating_baseline(self):
        vectors = np.asarray([[1.0], [-1.0], [2.0], [-2.0]])
        labels = np.asarray([1.0, 0.0, 1.0, 0.0])
        offsets = np.zeros(4)
        coefficients = fit_offset_logistic_correction(
            offsets,
            vectors,
            labels,
            np.ones(4),
            l2=0.1,
        )
        probabilities = offset_probabilities(offsets, vectors, coefficients)
        self.assertGreater(coefficients[0], 0)
        self.assertTrue(np.all(probabilities[labels == 1] > 0.5))
        self.assertTrue(np.all(probabilities[labels == 0] < 0.5))

    def test_exported_correction_matches_runtime_probability(self):
        baseline = {
            "format": "ghq-gradient-boosted-value-v1",
            "feature_names": ["base"],
            "base_raw_score": 0.4,
            "learning_rate": 0.1,
            "calibration": {"kind": "platt", "scale": 0.8, "intercept": -0.2},
            "trees": [],
            "metadata": {},
        }
        artifact = export_policy_correction(
            baseline,
            ["base", "shape"],
            np.asarray([1]),
            np.asarray([2.0]),
            np.asarray([0.6]),
            "dataset",
            {},
        )
        vectors = np.asarray([[0.0, 3.0], [0.0, -1.0]])
        expected = 1.0 / (1.0 + np.exp(-(0.8 * 0.4 - 0.2 + 0.3 * vectors[:, 1])))
        np.testing.assert_allclose(
            exported_probabilities(artifact, vectors), expected, atol=1e-12
        )
        self.assertEqual(artifact["calibration"], baseline["calibration"])

    def test_grouped_split_never_leaks_source_games(self):
        records = [
            {"source_game_id": f"game-{game}"}
            for game in range(12)
            for _ in range(2)
        ]
        splits = grouped_split(records, 7)
        game_sets = {
            name: {records[int(index)]["source_game_id"] for index in indices}
            for name, indices in splits.items()
        }
        self.assertTrue(game_sets["train"])
        self.assertTrue(game_sets["validation"])
        self.assertTrue(game_sets["test"])
        self.assertFalse(game_sets["train"] & game_sets["validation"])
        self.assertFalse(game_sets["train"] & game_sets["test"])
        self.assertFalse(game_sets["validation"] & game_sets["test"])


if __name__ == "__main__":
    unittest.main()
