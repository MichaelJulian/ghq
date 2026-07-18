import unittest
from types import SimpleNamespace

import numpy as np

from scripts.train_value_correction import (
    exported_correction,
    select_stable_correction_candidate,
)
from scripts.train_value_model import exported_probabilities, sigmoid


class ValueCorrectionExportTests(unittest.TestCase):
    def test_selection_prefers_more_regularization_inside_tiny_loss_tolerance(self):
        candidates = [
            {
                "share": 0.1,
                "score": 0.4000,
                "regularization_c": 1.0,
                "constraints_passed": True,
            },
            {
                "share": 0.1,
                "score": 0.4004,
                "regularization_c": 0.1,
                "constraints_passed": True,
            },
            {
                "share": 0.1,
                "score": 0.4010,
                "regularization_c": 0.01,
                "constraints_passed": True,
            },
        ]
        selected, feasible = select_stable_correction_candidate(candidates)
        self.assertTrue(feasible)
        self.assertEqual(selected["regularization_c"], 0.1)

    def test_standardized_correction_exports_with_probability_parity(self):
        baseline = {
            "format": "ghq-gradient-boosted-value-v1",
            "feature_names": ["base_a", "base_b"],
            "base_raw_score": 0.4,
            "learning_rate": 0.1,
            "calibration": {"kind": "platt", "scale": 0.8, "intercept": -0.2},
            "trees": [],
            "metadata": {"dataset_sha256": "incumbent"},
        }
        scaler = SimpleNamespace(
            mean_=np.asarray([2.0, 4.0]),
            scale_=np.asarray([2.0, 0.5]),
        )
        model = SimpleNamespace(
            coef_=np.asarray([[1.25, 0.4, -0.3]]),
            intercept_=np.asarray([0.15]),
        )
        artifact = exported_correction(
            baseline,
            ["base_a", "base_b", "shape_a", "shape_b"],
            np.asarray([2, 3]),
            scaler,
            model,
            "correction-data",
            {},
        )
        vectors = np.asarray(
            [[0.0, 0.0, 6.0, 3.0], [1.0, -1.0, 0.0, 5.0]]
        )
        baseline_logit = 0.8 * 0.4 - 0.2
        standardized = (vectors[:, 2:] - scaler.mean_) / scaler.scale_
        expected = sigmoid(
            0.15
            + 1.25 * baseline_logit
            + standardized @ np.asarray([0.4, -0.3])
        )
        np.testing.assert_allclose(
            exported_probabilities(artifact, vectors), expected, atol=1e-12
        )

    def test_nested_corrections_are_rejected(self):
        baseline = {
            "feature_names": ["base"],
            "linear_correction": {
                "feature_indices": [0],
                "coefficients": [1.0],
            },
        }
        with self.assertRaisesRegex(ValueError, "nested"):
            exported_correction(
                baseline,
                ["base", "new"],
                np.asarray([1]),
                SimpleNamespace(mean_=np.asarray([0.0]), scale_=np.asarray([1.0])),
                SimpleNamespace(
                    coef_=np.asarray([[1.0, 0.1]]),
                    intercept_=np.asarray([0.0]),
                ),
                "data",
                {},
            )


if __name__ == "__main__":
    unittest.main()
