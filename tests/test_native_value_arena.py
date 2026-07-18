"""Fast tests for the native value-artifact screening arena."""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "api"))

import _engine as engine  # noqa: E402
from run_native_value_arena import (  # noqa: E402
    GameResult,
    red_value_function,
    summarize,
)


class NativeValueArenaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.incumbent_path = ROOT / "api" / "_model_incumbent.json"
        self.incumbent = json.loads(self.incumbent_path.read_text())

    def arguments(self) -> argparse.Namespace:
        return argparse.Namespace(
            baseline=self.incumbent_path,
            challenger=self.incumbent_path,
            games=2,
            time_ms=100,
            max_depth=2,
            beam_width=6,
            max_turns=20,
            repetition_limit=3,
            seed=1,
            workers=1,
            output=None,
        )

    def game(
        self,
        index: int,
        *,
        seeded: int = 0,
        decisions: int = 10,
        openings: int = 4,
        verified: int = 9,
    ) -> GameResult:
        return GameResult(
            index=index,
            pair=0,
            seed=123,
            personality="balanced",
            challenger_color="RED" if index == 0 else "BLUE",
            winner="RED" if index == 0 else "BLUE",
            termination="by HQ capture",
            turns=decisions,
            challenger_score=1.0,
            decisions=decisions,
            opening_book_decisions=openings,
            verified_decisions=verified,
            fallback_counts={"seeded": seeded} if seeded else {"none": decisions},
            completed_depth_counts={"2": max(0, decisions - openings)},
            move_turns=[["a1a2"] for _ in range(decisions)],
        )

    def test_external_artifact_value_is_zero_sum(self) -> None:
        evaluate = red_value_function(self.incumbent)
        red = evaluate(engine.STARTING_FEN, 1)
        rotated = engine.BaseBoard(engine.STARTING_FEN).mirror().board_fen()
        mirrored_red = evaluate(rotated, 1)
        self.assertAlmostEqual(red + mirrored_red, 1.0, places=12)

    def test_quality_gate_rejects_one_seeded_fallback(self) -> None:
        report = summarize(
            self.arguments(),
            [self.game(0, seeded=1), self.game(1)],
        )
        self.assertFalse(report["qualityGate"]["passed"])
        self.assertIn(
            "seeded-fallback-decision", report["qualityGate"]["reasons"]
        )

    def test_quality_gate_requires_real_searched_decisions(self) -> None:
        report = summarize(
            self.arguments(),
            [
                self.game(0, decisions=4, openings=4, verified=4),
                self.game(1, decisions=4, openings=4, verified=4),
            ],
        )
        self.assertFalse(report["qualityGate"]["passed"])
        self.assertIn("no-searched-decisions", report["qualityGate"]["reasons"])

    def test_policy_divergence_uses_color_normalized_pair_lines(self) -> None:
        left = self.game(0)
        right = self.game(1)
        right.move_turns[6] = ["b1b2"]
        report = summarize(self.arguments(), [left, right])
        self.assertEqual(report["policyDivergence"]["divergedPairs"], 1)
        self.assertEqual(
            report["policyDivergence"]["pairs"][0]["commonPrefixTurns"], 6
        )


if __name__ == "__main__":
    unittest.main()
