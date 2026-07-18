import unittest

import numpy as np

from scripts.evaluate_counterfactual_challenger import (
    artifact_pair_probabilities,
    terminal_improvement_gate,
    training_source_game_overlap,
    training_root_overlap,
)


class CounterfactualChallengerEvaluationTests(unittest.TestCase):
    def test_terminal_gate_requires_enough_strictly_better_outcome_pairs(self):
        baseline = {"log_loss": 0.7, "accuracy": 0.5}
        improved = {"log_loss": 0.6, "accuracy": 0.6}
        tied = {"log_loss": 0.7, "accuracy": 0.6}
        regressed = {"log_loss": 0.6, "accuracy": 0.4}

        self.assertFalse(terminal_improvement_gate(None, improved, 4, 4))
        self.assertFalse(terminal_improvement_gate(baseline, improved, 3, 4))
        self.assertFalse(terminal_improvement_gate(baseline, tied, 4, 4))
        self.assertFalse(terminal_improvement_gate(baseline, regressed, 4, 4))
        self.assertTrue(terminal_improvement_gate(baseline, improved, 4, 4))

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

    def test_pair_probability_applies_policy_head_without_value_recalibration(self):
        artifact = {
            "feature_names": ["diff_shape"],
            "base_raw_score": 0.0,
            "learning_rate": 0.1,
            "calibration": {"scale": 1.0, "intercept": 0.0},
            "trees": [],
            "policy_correction": {
                "feature_indices": [0],
                "coefficients": [0.5],
            },
        }
        records = [
            {
                "left_features": np.asarray([2.0]),
                "right_features": np.asarray([-2.0]),
            }
        ]
        probability = artifact_pair_probabilities(records, artifact)
        self.assertAlmostEqual(probability[0], 1.0 / (1.0 + np.exp(-2.0)))

    def test_training_root_overlap_requires_provenance_and_detects_leakage(self):
        records = [{"root_id": "fresh"}, {"root_id": "leaked"}]
        self.assertEqual(training_root_overlap(records, {"metadata": {}}), (False, []))
        self.assertEqual(
            training_root_overlap(
                records,
                {
                    "metadata": {
                        "counterfactual_training_root_ids": ["old", "leaked"]
                    }
                },
            ),
            (True, ["leaked"]),
        )

    def test_training_source_game_overlap_requires_provenance_and_detects_leakage(self):
        records = [
            {"source_game_id": "game-a"},
            {"source_game_id": "game-b"},
        ]
        self.assertEqual(
            training_source_game_overlap(records, {"metadata": {}}),
            (False, []),
        )
        self.assertEqual(
            training_source_game_overlap(
                records,
                {
                    "metadata": {
                        "counterfactual_training_source_game_ids": [
                            "game-b",
                            "game-c",
                        ]
                    }
                },
            ),
            (True, ["game-b"]),
        )


if __name__ == "__main__":
    unittest.main()
