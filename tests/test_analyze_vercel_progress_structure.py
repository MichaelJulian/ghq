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
                "immobile_units": 1.0,
            }
        )
        danger = dict(safe)
        danger.update(
            {
                "gameId": "game-2",
                "side": "BLUE",
                "immobile_units": 0.0,
                "support_penalty": 2.0,
            }
        )

        report = progress_structure.summarize_metric_rows([safe, danger])

        self.assertEqual(report["sidePositions"], 2)
        self.assertEqual(report["metrics"]["support_penalty"]["mean"], 1.0)
        self.assertEqual(report["structuralDangerPositions"], 1)
        self.assertEqual(report["constrainedPositions"], 1)

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
        self.assertEqual(report["sidePositions"], 0)
        self.assertEqual(report["metrics"], {})


if __name__ == "__main__":
    unittest.main()
