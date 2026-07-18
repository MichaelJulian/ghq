"""Contract tests for the Vercel native-Python search core."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ghq_native_search", ROOT / "api" / "native_search.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


STARTING_FEN = (
    "qr↓6/iii5/8/8/8/8/5III/6R↑Q "
    "IIIIIFFFPRRTH iiiiifffprrth r"
)
AVOIDABLE_IMMEDIATE_HQ_LOSSES = (
    (59, "qi1i4/i1i3i1/1i3i2/8/F5f1/1FF1f1r↓1/I2I1f2/1I4Q1 - - r"),
    (76, "3q4/4i1i1/3I4/4F2i/3R↗2II/2I5/1I1I4/I1I1P2Q - - b"),
    (80, "5q2/8/3FI1I1/5I2/2II4/1I6/8/7Q - - b"),
    (90, "7q/8/7F/6F1/4II1I/7I/1I6/I6Q - - b"),
    (85, "1q6/2i5/Fi1i4/2i1ii2/8/6f1/3I2fi/4I1Q1 - - r"),
    (117, "q7/1I6/8/8/2if1i2/4f3/8/4Q3 - - r"),
    (77, "8/q7/1i6/I1i5/6i1/5i2/5f2/4i2Q - - r"),
    (57, "8/q1i5/1i6/2i5/2F3i1/1F1I3f/I1I5/1I1I1fQ1 - - r"),
)


class NativeSearchContractTest(unittest.TestCase):
    def test_returns_a_legal_complete_turn_and_resumable_state(self) -> None:
        result = MODULE.run_native_search(
            {
                "fen": STARTING_FEN,
                "personality": "balanced",
                "timeMs": 100,
                "maxDepth": 1,
                "beamWidth": 2,
                "turnNumber": 5,
                "maxActions": 3,
            }
        )
        self.assertEqual(result["sideToMove"], "RED")
        self.assertEqual(len(result["search"]["best_turn"]["actions"]), 3)
        self.assertEqual(result["search"]["search"]["backend"], "native-python")
        self.assertEqual(result["search"]["search"]["value_model_backend"], "native-gbdt")
        self.assertEqual(result["codeVersion"], MODULE.CODE_VERSION)
        self.assertEqual(
            result["search"]["search"]["code_version"], MODULE.CODE_VERSION
        )
        self.assertNotEqual(result["fen"], result["resultingFen"])
        self.assertGreater(len(result["serializedState"]), 20)

        described = MODULE.describe_native_position(
            {
                "mode": "describe",
                "fen": result["resultingFen"],
                "personality": "balanced",
                "turnNumber": 6,
            }
        )
        self.assertEqual(described["fen"], result["resultingFen"])
        self.assertEqual(described["serializedState"], result["serializedState"])
        self.assertEqual(described["codeVersion"], MODULE.CODE_VERSION)

        resumed = MODULE.run_native_search(
            {
                "serializedState": result["serializedState"],
                "personality": "balanced",
                "timeMs": 100,
                "maxDepth": 1,
                "beamWidth": 2,
                "turnNumber": 6,
                "maxActions": 3,
            }
        )
        self.assertEqual(resumed["fen"], result["resultingFen"])
        self.assertEqual(resumed["sideToMove"], "BLUE")

    def test_rejects_unknown_personality(self) -> None:
        with self.assertRaisesRegex(MODULE.NativeSearchInputError, "Unknown"):
            MODULE.run_native_search({"fen": STARTING_FEN, "personality": "nope"})

    def test_value_backed_native_search_avoids_every_known_immediate_hq_loss(
        self,
    ) -> None:
        for turn_number, fen in AVOIDABLE_IMMEDIATE_HQ_LOSSES:
            with self.subTest(turn_number=turn_number):
                result = MODULE.run_native_search(
                    {
                        "fen": fen,
                        "personality": "balanced",
                        "timeMs": 1_000,
                        "maxDepth": 2,
                        "beamWidth": 6,
                        "turnNumber": turn_number,
                        "maxActions": 3,
                    }
                )
                escaped = MODULE.engine.BaseBoard(result["resultingFen"])
                verifier = MODULE.ghq_ai.Searcher(
                    "balanced",
                    time_ms=60_000,
                    beam_width=6,
                    turn_number=turn_number + 1,
                )
                self.assertFalse(
                    verifier.exact_same_turn_hq_capture(
                        escaped, escaped.turn, [100_000]
                    ),
                    result["search"]["best_turn"]["all_moves"],
                )


if __name__ == "__main__":
    unittest.main()
