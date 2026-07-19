import json
import tempfile
import unittest
from pathlib import Path


from scripts import evaluate_search_checkpoints as evaluator


class SearchCheckpointEvaluationTests(unittest.TestCase):
    def test_reads_a_unique_v1_corpus(self):
        payload = {
            "format": "ghq-tactical-search-corpus-v1",
            "positions": [
                {
                    "id": "one",
                    "fen": evaluator.engine.STARTING_FEN,
                    "turnNumber": 1,
                    "personality": "balanced",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(evaluator.read_corpus(path), payload)

    def test_rejects_duplicate_position_ids(self):
        position = {
            "id": "duplicate",
            "fen": evaluator.engine.STARTING_FEN,
            "turnNumber": 1,
            "personality": "balanced",
        }
        payload = {
            "format": "ghq-tactical-search-corpus-v1",
            "positions": [position, dict(position)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate ids"):
                evaluator.read_corpus(path)

    def test_candidate_checks_reject_uncertified_and_worse_turn(self):
        position = {
            "acceptance": {
                "requireFinalSafetyCertified": True,
                "minimumCompletedDepth": 2,
                "maxCandidateForcedLoss": 0,
                "maxCandidateCriticalExposure": 1,
            }
        }
        baseline = {
            "checkpoint": {
                "engineSha256": "engine",
                "valueRuntimeSha256": "runtime",
                "modelSha256": "model",
            },
            "opponentReply": {
                "complete": True,
                "twoTurnExchange": {"netMaterialExchange": 0.0},
            },
            "resultingSafety": {
                "forcedLoss": 0.0,
                "criticalExposure": 1.0,
                "immediateHqLoss": False,
            }
        }
        candidate = {
            "checkpoint": {
                "engineSha256": "engine",
                "valueRuntimeSha256": "runtime",
                "modelSha256": "model",
            },
            "selectedMoves": ["skip"],
            "opponentReply": {
                "complete": True,
                "twoTurnExchange": {"netMaterialExchange": -3.0},
            },
            "search": {
                "completedDepth": 0,
                "finalSafetyCertified": False,
            },
            "resultingSafety": {
                "forcedLoss": 3.0,
                "criticalExposure": 4.0,
                "immediateHqLoss": False,
            },
        }
        checks = evaluator.candidate_checks(position, baseline, candidate)
        self.assertTrue(checks)
        self.assertFalse(all(item["passed"] for item in checks))
        self.assertIn(
            "final-safety-certified",
            [item["name"] for item in checks if not item["passed"]],
        )
        self.assertIn(
            "no-worse-than-baseline-forced-loss",
            [item["name"] for item in checks if not item["passed"]],
        )

    def test_candidate_checks_accept_certified_non_regression(self):
        position = {
            "acceptance": {
                "forbiddenMoves": ["f5h3"],
                "requireFinalSafetyCertified": True,
                "minimumCompletedDepth": 2,
                "maxCandidateForcedLoss": 0,
                "maxCandidateCriticalExposure": 2,
            }
        }
        baseline = {
            "checkpoint": {
                "engineSha256": "engine",
                "valueRuntimeSha256": "runtime",
                "modelSha256": "model",
            },
            "opponentReply": {
                "complete": True,
                "twoTurnExchange": {"netMaterialExchange": 0.0},
            },
            "resultingSafety": {
                "forcedLoss": 0.0,
                "criticalExposure": 2.0,
                "immediateHqLoss": False,
            }
        }
        candidate = {
            "checkpoint": {
                "engineSha256": "engine",
                "valueRuntimeSha256": "runtime",
                "modelSha256": "model",
            },
            "selectedMoves": ["b5a6", "skip"],
            "opponentReply": {
                "complete": True,
                "twoTurnExchange": {"netMaterialExchange": 1.0},
            },
            "search": {
                "completedDepth": 2,
                "finalSafetyCertified": True,
            },
            "resultingSafety": {
                "forcedLoss": 0.0,
                "criticalExposure": 1.0,
                "immediateHqLoss": False,
            },
        }
        checks = evaluator.candidate_checks(position, baseline, candidate)
        self.assertTrue(checks)
        self.assertTrue(all(item["passed"] for item in checks))

    def test_candidate_checks_reject_model_provenance_drift(self):
        position = {"acceptance": {}}
        common = {
            "opponentReply": {
                "complete": True,
                "twoTurnExchange": {"netMaterialExchange": 0.0},
            },
            "resultingSafety": {
                "forcedLoss": 0.0,
                "criticalExposure": 0.0,
                "immediateHqLoss": False,
            },
        }
        baseline = {
            **common,
            "checkpoint": {
                "engineSha256": "engine",
                "valueRuntimeSha256": "runtime",
                "modelSha256": "baseline-model",
            },
        }
        candidate = {
            **common,
            "checkpoint": {
                "engineSha256": "engine",
                "valueRuntimeSha256": "runtime",
                "modelSha256": "candidate-model",
            },
            "selectedMoves": ["skip"],
            "search": {
                "completedDepth": 2,
                "finalSafetyCertified": True,
            },
        }

        checks = evaluator.candidate_checks(position, baseline, candidate)
        failed = [item["name"] for item in checks if not item["passed"]]
        self.assertEqual(failed, ["same-value-model"])

    def test_replays_one_complete_opponent_reply(self):
        before = evaluator.engine.BaseBoard()
        after_selected = before.copy()
        selected = ["rhd1", "rte1", "rfc1"]
        for uci in selected:
            move = next(
                move
                for move in after_selected.generate_legal_moves()
                if move.uci() == uci
            )
            after_selected.push(move)
        isolated = {
            "result": {
                "principal_variation": selected
                + ["rhe8", "rtd8", "rpg8"]
            }
        }

        reply = evaluator.replay_opponent_reply(
            before, after_selected, selected, isolated
        )

        self.assertTrue(reply["complete"])
        self.assertEqual(reply["moves"], ["rhe8", "rtd8", "rpg8"])
        self.assertEqual(
            reply["twoTurnExchange"]["netMaterialExchange"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
