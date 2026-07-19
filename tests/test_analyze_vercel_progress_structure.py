import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_vercel_progress_structure as progress_structure  # noqa: E402


class ProgressStructureAnalysisTests(unittest.TestCase):
    def test_pair_diversity_treats_color_swaps_as_one_unit(self):
        report = progress_structure.summarize_pair_diversity(
            [
                {"gameId": "generation-0001", "currentFen": "a"},
                {"gameId": "generation-0002", "currentFen": "b"},
                {"gameId": "generation-0003", "currentFen": "c"},
                {"gameId": "generation-0004", "currentFen": "c"},
                {"gameId": "generation-0005", "currentFen": "a"},
                {"gameId": "invalid", "currentFen": "d"},
            ]
        )

        self.assertEqual(report["completeColorSwapPairs"], 2)
        self.assertEqual(report["incompleteColorSwapPairs"], 1)
        self.assertEqual(report["unpairedSnapshotGames"], 1)
        self.assertEqual(report["uniquePairTrajectories"], 2)

    def test_summarizes_metrics_and_separates_danger_from_congestion(self):
        safe = {
            metric: 0.0 for metric in progress_structure.METRIC_NAMES
        }
        safe.update(
            {
                "gameId": "game-1",
                "completedTurns": 10,
                "side": "RED",
                "to_move": False,
                "immobile_units": 1.0,
            }
        )
        danger = dict(safe)
        danger.update(
            {
                "gameId": "game-2",
                "side": "BLUE",
                "to_move": True,
                "immobile_units": 0.0,
                "support_penalty": 2.0,
            }
        )

        report = progress_structure.summarize_metric_rows([safe, danger])

        self.assertEqual(report["sidePositions"], 2)
        self.assertEqual(report["metrics"]["support_penalty"]["mean"], 1.0)
        self.assertEqual(report["structuralDebtPositions"], 1)
        self.assertEqual(report["structuralDangerPositions"], 1)
        self.assertEqual(report["tacticalDangerPositions"], 0)
        self.assertEqual(report["constrainedPositions"], 1)

        danger["tactical_risk_value"] = 3.0
        tactical_report = progress_structure.summarize_metric_rows(
            [safe, danger]
        )
        self.assertEqual(tactical_report["tacticalDangerPositions"], 1)
        self.assertEqual(tactical_report["repairRequiredPositions"], 1)
        self.assertEqual(
            tactical_report["immediateTacticalDangerPositions"], 0
        )

        immediate = dict(danger)
        immediate.update({"gameId": "game-3", "to_move": False})
        immediate_report = progress_structure.summarize_metric_rows(
            [safe, danger, immediate]
        )
        self.assertEqual(immediate_report["repairRequiredPositions"], 1)
        self.assertEqual(
            immediate_report["immediateTacticalDangerPositions"], 1
        )
        self.assertEqual(
            immediate_report["immediateForcedCapturePositions"], 1
        )
        self.assertEqual(
            immediate_report["immediateCaptureThreatPositions"], 1
        )

    def test_empty_progress_is_a_valid_report(self):
        report = progress_structure.analyze_summary(
            {
                "generationId": "generation-1",
                "progress": {"snapshots": []},
            }
        )

        self.assertEqual(report["snapshotGames"], 0)
        self.assertEqual(report["uniqueSnapshotPositions"], 0)
        self.assertEqual(report["maxPositionMultiplicity"], 0)
        self.assertEqual(report["positionMultiplicity"], [])
        self.assertEqual(report["pairDiversity"]["completeColorSwapPairs"], 0)
        self.assertEqual(report["snapshotTelemetry"], [])
        self.assertEqual(report["workflowRuns"], {})
        self.assertEqual(report["activeProgressRuntime"], {})
        self.assertEqual(report["sidePositions"], 0)
        self.assertEqual(report["metrics"], {})

    def test_compares_runtime_deltas_and_repair_obligations(self):
        before = {
            "generationId": "generation-1",
            "completedTurns": [20],
            "structuralDebtPositions": 2,
            "immediateForcedCapturePositions": 1,
            "snapshotTelemetry": [
                {
                    "gameId": "game-1",
                    "decisions": 20,
                    "depthAtLeastTwoDecisions": 16,
                    "fallbackDecisions": 1,
                    "unverifiedFallbackDecisions": 0,
                    "timedOutDecisions": 16,
                }
            ],
            "positionMetrics": [
                {
                    "gameId": "game-1",
                    "side": "RED",
                    "to_move": True,
                    "tactical_risk_value": 5.0,
                    "piece_inventory": {"ARTILLERY": 1},
                },
                {
                    "gameId": "game-1",
                    "side": "BLUE",
                    "to_move": False,
                    "tactical_risk_value": 0.0,
                    "piece_inventory": {"ARMORED_ARTILLERY": 1},
                }
            ],
        }
        after = {
            "generationId": "generation-1",
            "completedTurns": [30],
            "structuralDebtPositions": 1,
            "immediateForcedCapturePositions": 0,
            "snapshotTelemetry": [
                {
                    "gameId": "game-1",
                    "decisions": 30,
                    "depthAtLeastTwoDecisions": 25,
                    "fallbackDecisions": 2,
                    "unverifiedFallbackDecisions": 0,
                    "timedOutDecisions": 25,
                }
            ],
            "positionMetrics": [
                {
                    "gameId": "game-1",
                    "side": "RED",
                    "to_move": True,
                    "tactical_risk_value": 0.0,
                    "piece_inventory": {},
                },
                {
                    "gameId": "game-1",
                    "side": "BLUE",
                    "to_move": False,
                    "tactical_risk_value": 0.0,
                    "piece_inventory": {},
                }
            ],
        }

        comparison = progress_structure.compare_checkpoint_reports(
            before, after
        )
        self.assertEqual(comparison["sharedGames"], 1)
        self.assertEqual(comparison["counterDeltas"]["decisions"], 10)
        self.assertEqual(
            comparison["counterDeltas"]["depthAtLeastTwoDecisions"], 9
        )
        self.assertEqual(comparison["counterDeltas"]["fallbackDecisions"], 1)
        self.assertEqual(
            comparison["sameSideRiskFreeAtLaterCheckpoint"], 1
        )
        self.assertEqual(comparison["threatenedInventoryRetained"], 0)
        self.assertEqual(comparison["favorableMaterialExchanges"], 1)
        self.assertEqual(comparison["unfavorableMaterialExchanges"], 0)
        self.assertEqual(
            comparison["repairOutcomes"][0]["ownMaterialLost"], 3.0
        )
        self.assertEqual(
            comparison["repairOutcomes"][0]["opponentMaterialLost"], 5.0
        )
        self.assertEqual(
            comparison["repairOutcomes"][0]["netMaterialExchange"], 2.0
        )
        self.assertEqual(comparison["structuralDebtPositionDelta"], -1)
        self.assertEqual(
            comparison["immediateForcedCapturePositionDelta"], -1
        )
        self.assertEqual(
            comparison["immediateCaptureThreatPositionDelta"], -1
        )

    def test_identifies_the_live_forced_armored_artillery_target(self):
        board = progress_structure.engine.BaseBoard(
            "qf1i2p1/i1ir↓if2/1ir↓t↓1h↓r↓f/8/7I/"
            "I4T↑I1/1FIIH↑R↑IQ/IP2IF1F RR iii r"
        )
        targets = progress_structure.forced_capture_targets(
            board, progress_structure.engine.RED
        )
        self.assertEqual(
            targets,
            [
                {
                    "square": "f3",
                    "pieceType": "ARMORED_ARTILLERY",
                    "value": 5.0,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
