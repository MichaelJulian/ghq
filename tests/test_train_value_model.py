import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.train_value_model import (
    align_append_only_baseline_schema,
    chronological_split,
    evaluation_unit,
    exported_probabilities,
    exported_raw_scores,
    game_balanced_weights,
    load_dataset,
    requested_self_play_shares,
    select_validation_candidate,
    main as train_main,
    validation_selection_gates,
    validate_self_play_behavior_checkpoint,
    validate_self_play_code_version,
)


class ValueModelWeightTests(unittest.TestCase):
    def test_append_only_schema_can_score_the_exact_incumbent_trees(self):
        baseline = {
            "feature_names": ["a", "b"],
            "trees": [{"feature": [0, -2, -2]}],
            "metadata": {"checkpoint": "incumbent"},
        }
        aligned = align_append_only_baseline_schema(
            baseline, ["a", "b", "formation"]
        )
        self.assertEqual(aligned["feature_names"], ["a", "b", "formation"])
        self.assertEqual(baseline["feature_names"], ["a", "b"])
        self.assertEqual(
            aligned["metadata"]["baseline_schema_alignment"],
            {
                "kind": "append-only",
                "original_feature_count": 2,
                "aligned_feature_count": 3,
            },
        )

    def test_append_only_schema_rejects_reordered_features(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            align_append_only_baseline_schema(
                {"feature_names": ["a", "b"], "trees": []},
                ["b", "a", "formation"],
            )

    def test_constrained_selection_uses_smallest_feasible_self_play_share(self):
        candidates = [
            {"share": 0.12, "score": 0.55, "constraints_passed": True},
            {"share": 0.08, "score": 0.62, "constraints_passed": True},
            {"share": 0.08, "score": 0.60, "constraints_passed": True},
            {"share": 0.04, "score": 0.58, "constraints_passed": False},
        ]
        selected, feasible = select_validation_candidate(
            candidates, baseline_constrained=True
        )
        self.assertTrue(feasible)
        self.assertEqual(selected["share"], 0.08)
        self.assertEqual(selected["score"], 0.60)

    def test_unconstrained_selection_still_minimizes_validation_loss(self):
        candidates = [
            {"share": 0.02, "score": 0.70, "constraints_passed": True},
            {"share": 0.12, "score": 0.55, "constraints_passed": True},
        ]
        selected, feasible = select_validation_candidate(
            candidates, baseline_constrained=False
        )
        self.assertTrue(feasible)
        self.assertEqual(selected["share"], 0.12)

    def test_infeasible_selection_cannot_clear_promotion(self):
        candidates = [
            {"share": 0.02, "score": 0.70, "constraints_passed": False},
            {"share": 0.12, "score": 0.55, "constraints_passed": False},
        ]
        selected, feasible = select_validation_candidate(
            candidates, baseline_constrained=True
        )
        self.assertFalse(feasible)
        self.assertEqual(selected["share"], 0.12)

    def test_exported_raw_scores_reproduce_calibrated_probabilities(self):
        artifact = {
            "base_raw_score": 0.4,
            "learning_rate": 0.1,
            "trees": [],
            "calibration": {"scale": 0.75, "intercept": -0.2},
        }
        vectors = np.asarray([[0.0], [1.0]])
        raw = exported_raw_scores(artifact, vectors)
        expected = 1.0 / (1.0 + np.exp(-(0.75 * raw - 0.2)))
        np.testing.assert_allclose(
            exported_probabilities(artifact, vectors), expected
        )

    def test_validation_selection_gates_preserve_human_and_improve_self_play(self):
        def sources(human, self_play, human_red=None, human_blue=None):
            return {
                "human": {
                    "log_loss": human,
                    "by_perspective": {
                        "RED": {"log_loss": human if human_red is None else human_red},
                        "BLUE": {"log_loss": human if human_blue is None else human_blue},
                    },
                },
                "vercel_self_play": {
                    "log_loss": self_play,
                    "by_perspective": {
                        "RED": {"log_loss": self_play},
                        "BLUE": {"log_loss": self_play},
                    },
                },
            }

        baseline = sources(0.40, 0.80)
        passing = validation_selection_gates(sources(0.409, 0.70), baseline)
        self.assertTrue(all(gate["passed"] for gate in passing))

        human_regression = validation_selection_gates(
            sources(0.42, 0.70), baseline
        )
        self.assertFalse(
            next(
                gate for gate in human_regression if gate["name"] == "human-log-loss"
            )["passed"]
        )
        self_play_regression = validation_selection_gates(
            sources(0.40, 0.81), baseline
        )
        self.assertFalse(
            next(
                gate
                for gate in self_play_regression
                if gate["name"] == "self-play-log-loss"
            )["passed"]
        )

    def test_validation_selection_gates_support_human_only_baselines(self):
        def human(log_loss):
            return {
                "human": {
                    "log_loss": log_loss,
                    "by_perspective": {
                        "RED": {"log_loss": log_loss},
                        "BLUE": {"log_loss": log_loss},
                    },
                }
            }

        gates = validation_selection_gates(human(0.39), human(0.40))
        self.assertEqual(
            [gate["name"] for gate in gates],
            ["human-log-loss", "human-red-log-loss", "human-blue-log-loss"],
        )
        self.assertTrue(all(gate["passed"] for gate in gates))

    def test_share_grid_is_selected_on_validation_and_exported_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            model = root / "model.json"
            report = root / "report.json"
            records = [
                {
                    "type": "schema",
                    "format": "ghq-value-features-v1",
                    "feature_names": ["signal", "progress"],
                    "self_play_code_version": "commit-a",
                    "self_play_behavior_value_model_checkpoint": "checkpoint-a",
                    "self_play_search_backend": "native-python",
                    "self_play_value_model_backend": "native-gbdt",
                    "paired_complete_only": True,
                    "exact_hq_audit_required": True,
                    "paratrooper_policy_audit_required": True,
                    "zero_unverified_fallbacks_required": True,
                    "color_swap_integrity_verified": True,
                    "exact_hq_audit_sha256": "a" * 64,
                    "exact_hq_audit_max_nodes": 2_000_000,
                }
            ]
            for unit in range(30):
                label = unit % 2
                created_at = f"2026-07-{unit + 1:02d}T00:00:00Z"
                records.append(
                    {
                        "type": "sample",
                        "source": "human",
                        "game_id": f"human-{unit:02d}",
                        "created_at": created_at,
                        "outcome_reason": "hq-capture",
                        "turn": 5,
                        "perspective": "RED" if unit % 2 == 0 else "BLUE",
                        "label": label,
                        "features": [label, unit / 30],
                    }
                )
                for member in range(2):
                    records.append(
                        {
                            "type": "sample",
                            "source": "vercel_self_play",
                            "generation_id": "generation",
                            "pair_id": f"pair-{unit:02d}",
                            "code_version": "commit-a",
                            "behavior_value_model_checkpoint": "checkpoint-a",
                            "behavior_search_backend": "native-python",
                            "behavior_value_model_backend": "native-gbdt",
                            "game_id": f"self-{unit:02d}-{member}",
                            "created_at": created_at,
                            "outcome_reason": "hq-capture",
                            "turn": 5,
                            "perspective": "RED" if member == 0 else "BLUE",
                            "label": label if member == 0 else 1 - label,
                            "features": [label if member == 0 else 1 - label, unit / 30],
                        }
                    )
            dataset.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            argv = [
                "train_value_model.py",
                "--dataset",
                str(dataset),
                "--output",
                str(model),
                "--report",
                str(report),
                "--self-play-train-shares",
                "0.02,0.08",
                "--self-play-code-version",
                "commit-a",
                "--self-play-behavior-checkpoint",
                "checkpoint-a",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                train_main()

            evidence = json.loads(report.read_text(encoding="utf-8"))
            artifact = json.loads(model.read_text(encoding="utf-8"))
            self.assertIn(evidence["self_play_train_share"], (0.02, 0.08))
            self.assertEqual(len(evidence["candidate_validation"]), 8)
            self.assertEqual(
                artifact["metadata"]["self_play_train_share"],
                evidence["self_play_train_share"],
            )

    def test_dataset_loader_rejects_self_play_without_audit_attestations(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            records = [
                {
                    "type": "schema",
                    "format": "ghq-value-features-v1",
                    "feature_names": ["signal"],
                },
                {
                    "type": "sample",
                    "source": "vercel_self_play",
                    "game_id": "game-1",
                    "generation_id": "generation",
                    "pair_id": "pair-1",
                    "turn": 1,
                    "perspective": "RED",
                    "label": 1,
                    "features": [1],
                },
            ]
            dataset.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "paired_complete_only"):
                load_dataset(dataset)

    def test_dataset_loader_rejects_incomplete_attested_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            schema = {
                "type": "schema",
                "format": "ghq-value-features-v1",
                "feature_names": ["signal"],
                "code_version": "commit-a",
                "behavior_value_model_checkpoint": "checkpoint-a",
                "self_play_search_backend": "native-python",
                "self_play_value_model_backend": "native-gbdt",
                "paired_complete_only": True,
                "exact_hq_audit_required": True,
                "paratrooper_policy_audit_required": True,
                "zero_unverified_fallbacks_required": True,
                "color_swap_integrity_verified": True,
                "exact_hq_audit_sha256": "a" * 64,
                "exact_hq_audit_max_nodes": 2_000_000,
            }
            sample = {
                "type": "sample",
                "source": "vercel_self_play",
                "game_id": "game-1",
                "generation_id": "generation",
                "pair_id": "pair-1",
                "code_version": "commit-a",
                "behavior_value_model_checkpoint": "checkpoint-a",
                "behavior_search_backend": "native-python",
                "behavior_value_model_backend": "native-gbdt",
                "turn": 1,
                "perspective": "RED",
                "label": 1,
                "features": [1],
            }
            dataset.write_text(
                json.dumps(schema) + "\n" + json.dumps(sample) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete color-swapped"):
                load_dataset(dataset)

    def test_parses_a_validation_selected_self_play_share_grid(self):
        self.assertEqual(
            requested_self_play_shares(None, "0.02, 0.04,0.08"),
            [0.02, 0.04, 0.08],
        )
        self.assertEqual(requested_self_play_shares(None, None), [0.5])
        self.assertEqual(requested_self_play_shares(0.1, None), [0.1])

    def test_rejects_invalid_or_duplicate_self_play_shares(self):
        for grid in ("", "0", "1", "0.04,0.04", "word"):
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    requested_self_play_shares(None, grid)

    def test_self_play_requires_explicit_color_swapped_pair(self):
        with self.assertRaisesRegex(ValueError, "requires pair_id"):
            evaluation_unit(
                {"source": "vercel_self_play", "game_id": "generation-0001"}
            )

    def test_chronological_split_keeps_color_swapped_pairs_together(self):
        rows = []
        for pair in range(1, 31):
            for color_game in range(2):
                game_number = 2 * pair - 1 + color_game
                rows.append(
                    {
                        "source": "vercel_self_play",
                        "game_id": f"generation-{game_number:04d}",
                        "pair_id": f"generation-pair-{pair:04d}",
                        "created_at": f"2026-07-{pair:02d}T00:00:0{color_game}Z",
                    }
                )

        splits = chronological_split(rows)
        self.assertEqual(
            {name: len(indices) for name, indices in splits.items()},
            {"train": 36, "calibration": 6, "validation": 8, "test": 10},
        )
        split_by_index = {
            int(index): name for name, indices in splits.items() for index in indices
        }
        for pair in range(30):
            self.assertEqual(split_by_index[2 * pair], split_by_index[2 * pair + 1])

    def test_requires_one_exact_self_play_search_revision(self):
        rows = [
            {"source": "human", "game_id": "human"},
            {
                "source": "vercel_self_play",
                "game_id": "self-play",
                "code_version": "commit-a",
            },
        ]
        self.assertEqual(
            validate_self_play_code_version(rows, "commit-a"), "commit-a"
        )

    def test_rejects_missing_mixed_or_unexpected_search_revisions(self):
        with self.assertRaisesRegex(ValueError, "exact code_version"):
            validate_self_play_code_version(
                [{"source": "vercel_self_play", "game_id": "missing"}]
            )
        with self.assertRaisesRegex(ValueError, "mix search revisions"):
            validate_self_play_code_version(
                [
                    {
                        "source": "vercel_self_play",
                        "game_id": "a",
                        "code_version": "commit-a",
                    },
                    {
                        "source": "vercel_self_play",
                        "game_id": "b",
                        "code_version": "commit-b",
                    },
                ]
            )
        with self.assertRaisesRegex(ValueError, "code version mismatch"):
            validate_self_play_code_version(
                [
                    {
                        "source": "vercel_self_play",
                        "game_id": "a",
                        "code_version": "commit-a",
                    }
                ],
                "commit-b",
            )

    def test_requires_one_exact_self_play_behavior_checkpoint(self):
        rows = [
            {"source": "human", "game_id": "human"},
            {
                "source": "vercel_self_play",
                "game_id": "self-play",
                "behavior_value_model_checkpoint": "checkpoint-a",
            },
        ]
        self.assertEqual(
            validate_self_play_behavior_checkpoint(rows, "checkpoint-a"),
            "checkpoint-a",
        )

    def test_rejects_missing_mixed_or_unexpected_behavior_checkpoints(self):
        with self.assertRaisesRegex(ValueError, "exact behavior value-model"):
            validate_self_play_behavior_checkpoint(
                [{"source": "vercel_self_play", "game_id": "missing"}]
            )
        with self.assertRaisesRegex(ValueError, "mix behavior checkpoints"):
            validate_self_play_behavior_checkpoint(
                [
                    {
                        "source": "vercel_self_play",
                        "game_id": "a",
                        "behavior_value_model_checkpoint": "checkpoint-a",
                    },
                    {
                        "source": "vercel_self_play",
                        "game_id": "b",
                        "behavior_value_model_checkpoint": "checkpoint-b",
                    },
                ]
            )
        with self.assertRaisesRegex(ValueError, "behavior checkpoint mismatch"):
            validate_self_play_behavior_checkpoint(
                [
                    {
                        "source": "vercel_self_play",
                        "game_id": "a",
                        "behavior_value_model_checkpoint": "checkpoint-a",
                    }
                ],
                "checkpoint-b",
            )

    def test_balances_red_and_blue_winning_games_within_each_source(self):
        rows = []
        for game, red_wins, positions in (
            ("red-1", True, 3),
            ("red-2", True, 1),
            ("red-3", True, 2),
            ("blue-1", False, 4),
        ):
            for offset in range(positions):
                perspective = "RED" if offset % 2 == 0 else "BLUE"
                label = int((perspective == "RED") == red_wins)
                rows.append(
                    {
                        "game_id": game,
                        "generation_id": "vercel-test",
                        "perspective": perspective,
                        "label": label,
                    }
                )
        weights = game_balanced_weights(
            rows,
            np.arange(len(rows)),
            balance_outcomes=True,
        )
        red_weight = sum(
            weight
            for row, weight in zip(rows, weights)
            if (row["perspective"] == "RED") == (row["label"] == 1)
        )
        blue_weight = weights.sum() - red_weight
        self.assertAlmostEqual(red_weight, blue_weight)

    def test_still_gives_each_game_equal_weight_inside_an_outcome(self):
        rows = [
            {
                "game_id": "red-long" if index < 4 else "red-short",
                "generation_id": "vercel-test",
                "perspective": "RED",
                "label": 1,
            }
            for index in range(5)
        ]
        weights = game_balanced_weights(
            rows,
            np.arange(len(rows)),
            balance_outcomes=True,
        )
        self.assertAlmostEqual(weights[:4].sum(), weights[4:].sum())

    def test_can_limit_outcome_balancing_to_self_play(self):
        rows = []
        for source, red_wins in (
            ("human", True),
            ("human", True),
            ("human", False),
            ("vercel-test", True),
            ("vercel-test", True),
            ("vercel-test", False),
        ):
            rows.append(
                {
                    "game_id": f"{source}-{len(rows)}",
                    "source": "human" if source == "human" else "vercel_self_play",
                    "perspective": "RED",
                    "label": int(red_wins),
                }
            )
        weights = game_balanced_weights(
            rows,
            np.arange(len(rows)),
            balance_outcomes=True,
            outcome_balance_sources={"vercel_self_play"},
        )
        human = [weight for row, weight in zip(rows, weights) if row["source"] == "human"]
        self_play = [
            weight
            for row, weight in zip(rows, weights)
            if row["source"] == "vercel_self_play"
        ]
        self.assertAlmostEqual(human[0], human[2])
        self.assertAlmostEqual(self_play[0] + self_play[1], self_play[2])


if __name__ == "__main__":
    unittest.main()
