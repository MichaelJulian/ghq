import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.train_counterfactual_policy import (
    correction_feature_indices,
    export_policy_correction,
    fit_offset_logistic_correction,
    grouped_folds,
    grouped_split,
    load_counterfactual_reports,
    offset_probabilities,
    safe_cross_validated_linear_candidate,
    stability_random_states,
    rank_correction_features,
)
from scripts.train_value_model import exported_probabilities


class CounterfactualPolicyTrainingTests(unittest.TestCase):
    def test_difference_scope_only_selects_antisymmetric_features(self):
        names = ["own_material", "diff_material", "opp_pressure", "diff_shape"]
        np.testing.assert_array_equal(
            correction_feature_indices(names, 3, "difference"),
            np.asarray([1, 3]),
        )

    def test_empty_difference_scope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no features are eligible"):
            correction_feature_indices(
                ["own_material", "opp_material"], 1, "difference"
            )

    def test_feature_ranking_uses_training_residual_signal(self):
        vectors = np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]
        )
        labels = np.asarray([1.0, 0.0, 1.0, 0.0])
        ranking = rank_correction_features(
            np.zeros(4), vectors, labels, np.ones(4)
        )
        self.assertEqual(ranking.tolist(), [0, 1])

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

    def test_analyzer_ineligible_pair_is_not_admitted(self):
        branches = [
            {
                "status": "completed",
                "candidateRank": rank,
                "featuresV3": [float(rank), 0.0],
                "rolloutValue": value,
                "valueSource": "terminal",
                "unverifiedFallbackDecisions": 0,
            }
            for rank, value in ((1, 1.0), (2, 0.0))
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
                    "trainingEligible": False,
                    "trainingExclusion": "replicate-disagreement",
                    "rootId": "root-1",
                    "sourceGameId": "game-1",
                    "rootPlayer": "RED",
                    "sourceTurnNumber": 12,
                    "branches": branches,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replicate-disagreement.json"
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

    def test_policy_correction_does_not_recalibrate_value_probability(self):
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
            ["base", "diff_shape"],
            np.asarray([1]),
            np.asarray([2.0]),
            np.asarray([0.6]),
            "dataset",
            {},
            correction_target="policy",
            policy_scale=0.25,
        )
        vectors = np.asarray([[0.0, 3.0], [0.0, -1.0]])
        expected = 1.0 / (1.0 + np.exp(-(0.8 * 0.4 - 0.2)))
        np.testing.assert_allclose(
            exported_probabilities(artifact, vectors), expected, atol=1e-12
        )
        self.assertNotIn("linear_correction", artifact)
        self.assertEqual(artifact["policy_correction"]["feature_indices"], [1])
        self.assertEqual(artifact["policy_correction"]["scale"], 0.25)

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

    def test_grouped_cross_validation_assigns_each_source_game_once(self):
        records = [
            {"source_game_id": f"game-{game}"}
            for game in range(15)
            for _ in range(2)
        ]
        folds = grouped_folds(records, 11, fold_count=5)
        fold_games = [
            {records[int(index)]["source_game_id"] for index in fold}
            for fold in folds
        ]
        self.assertTrue(all(fold_games))
        self.assertEqual(sum(len(games) for games in fold_games), 15)
        self.assertEqual(len(set.union(*fold_games)), 15)

    def test_stability_random_states_are_repeatable_and_unique(self):
        self.assertEqual(stability_random_states(42), [42, 23, 101])
        self.assertEqual(stability_random_states(23), [23, 42, 101])
        self.assertEqual(stability_random_states(7), [7, 42, 23, 101])

    def test_safe_cross_validation_rejects_optimizer_failure(self):
        records = [
            {
                "source_game_id": f"game-{index}",
                "left_features": np.asarray([0.0]),
                "right_features": np.asarray([0.0]),
                "label": float(index % 2),
                "rollout_delta": 1.0,
                "terminal_pair": False,
            }
            for index in range(10)
        ]
        baseline = {
            "format": "ghq-gradient-boosted-value-v1",
            "feature_names": ["x"],
            "base_raw_score": 0.0,
            "learning_rate": 0.1,
            "calibration": {"kind": "platt", "scale": 1.0, "intercept": 0.0},
            "trees": [],
        }
        with mock.patch(
            "scripts.train_counterfactual_policy.cross_validated_linear_candidate",
            side_effect=RuntimeError("unstable"),
        ):
            candidate = safe_cross_validated_linear_candidate(
                records,
                baseline,
                np.asarray([0]),
                1,
                0.1,
                [np.arange(10)],
            )
        self.assertFalse(candidate["passed"])
        self.assertEqual(candidate["error"], "unstable")


if __name__ == "__main__":
    unittest.main()
