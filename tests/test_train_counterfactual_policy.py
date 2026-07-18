import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from scripts.train_counterfactual_policy import (
    export_policy_correction,
    fit_offset_logistic_correction,
    grouped_split,
    load_counterfactual_reports,
    offset_probabilities,
)
from scripts.train_value_model import exported_probabilities


class CounterfactualPolicyTrainingTests(unittest.TestCase):
    def test_incomplete_rollout_report_is_rejected(self):
        report = {
            "format": "ghq-counterfactual-rollout-report-v1",
            "featureSchema": ["base", "shape"],
            "expectedBranches": 2,
            "completedBranches": 1,
            "missingBranches": 1,
            "pairs": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "report is incomplete"):
                load_counterfactual_reports([path])

    def test_unverified_pair_is_not_admitted(self):
        branches = [
            {
                "status": "completed",
                "candidateRank": rank,
                "featuresV3": [float(rank), 0.0],
                "rolloutValue": value,
                "valueSource": "terminal",
                "unverifiedFallbackDecisions": unverified,
            }
            for rank, value, unverified in ((1, 1.0, 1), (2, 0.0, 0))
        ]
        report = {
            "format": "ghq-counterfactual-rollout-report-v1",
            "featureSchema": ["base", "shape"],
            "expectedBranches": 2,
            "completedBranches": 2,
            "missingBranches": 0,
            "pairs": [
                {
                    "confident": True,
                    "rootId": "root-1",
                    "sourceGameId": "game-1",
                    "rootPlayer": "RED",
                    "sourceTurnNumber": 12,
                    "branches": branches,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.json"
            path.write_text(json.dumps(report))
            _, records, _ = load_counterfactual_reports([path])
        self.assertEqual(records, [])

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
