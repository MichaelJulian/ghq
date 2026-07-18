"""Exact prediction parity fixtures for the TypeScript and Python runtimes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

import _engine as engine  # noqa: E402
import _value_model as value_model  # noqa: E402


class NativeValueModelTest(unittest.TestCase):
    def assert_prediction(
        self, fen: str, turn: int, expected: float, version: str = "incumbent"
    ) -> None:
        actual = value_model.predict_zero_sum(
            fen, turn, engine.RED, version=version
        )
        self.assertAlmostEqual(actual, expected, places=15)
        blue = value_model.predict_zero_sum(
            fen, turn, engine.BLUE, version=version
        )
        self.assertAlmostEqual(actual + blue, 1.0, places=15)

    def test_matches_typescript_inference_on_opening_and_tactical_positions(self) -> None:
        # Values are emitted by predictZeroSumWinProbability in
        # src/game/value-model/inference.ts. Exact agreement also exercises
        # the native feature extraction, trees, calibration, and zero-sum
        # normalization.
        self.assert_prediction(engine.STARTING_FEN, 1, 0.537824809609784)
        self.assert_prediction(
            "q1r↓1ip2/ir↘1i4/1ii5/3ff3/7r↓/6f1/2FI1I1I/1I2I1FQ I iii r",
            31,
            0.013502156202695882,
        )
        self.assert_prediction(
            "8/q7/1i6/I1i5/6i1/5i2/5f2/4i2Q - - r",
            31,
            0.01173207104201132,
        )

    def test_matches_typescript_challenger_with_append_only_features(self) -> None:
        self.assert_prediction(
            engine.STARTING_FEN, 1, 0.5337958232173801, "challenger"
        )
        self.assert_prediction(
            "q1r↓1ip2/ir↘1i4/1ii5/3ff3/7r↓/6f1/2FI1I1I/1I2I1FQ I iii r",
            31,
            0.022013671574450853,
            "challenger",
        )


if __name__ == "__main__":
    unittest.main()
