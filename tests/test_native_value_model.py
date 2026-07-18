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
            engine.STARTING_FEN, 1, 0.5332769823702843, "challenger"
        )
        self.assert_prediction(
            "q1r↓1ip2/ir↘1i4/1ii5/3ff3/7r↓/6f1/2FI1I1I/1I2I1FQ I iii r",
            31,
            0.016427033102013284,
            "challenger",
        )

    def test_v2_formation_features_match_typescript_fixtures(self) -> None:
        def position(infantry_squares: list[str]) -> engine.BaseBoard:
            board = engine.BaseBoard(None)
            board.clear_board()
            board.set_piece_at(
                engine.parse_square("h1"), engine.Piece(engine.HQ, engine.RED)
            )
            board.set_piece_at(
                engine.parse_square("a8"), engine.Piece(engine.HQ, engine.BLUE)
            )
            for square in infantry_squares:
                board.set_piece_at(
                    engine.parse_square(square),
                    engine.Piece(engine.INFANTRY, engine.RED),
                )
            return board

        vertical = value_model._side_features(
            position(["f3", "f4", "f5"]), engine.RED
        )
        staggered = value_model._side_features(
            position(["f3", "g4", "h3"]), engine.RED
        )
        # These are the same board fixtures and expectations asserted by
        # src/game/value-model/features.test.ts. Together, the two suites
        # protect the append-only native/TypeScript inference boundary.
        expected = {
            "infantry_vertical_adjacent_pairs": (2.0, 0.0),
            "infantry_diagonal_adjacent_pairs": (0.0, 2.0),
            "infantry_same_file_run_excess": (1.0, 0.0),
            "infantry_distinct_files": (1.0, 3.0),
            "infantry_frontier_count": (1.0, 1.0),
        }
        for name, (vertical_value, staggered_value) in expected.items():
            self.assertEqual(vertical[name], vertical_value, name)
            self.assertEqual(staggered[name], staggered_value, name)

        appended_names = {
            "infantry_vertical_adjacent_pairs",
            "infantry_diagonal_adjacent_pairs",
            "infantry_same_file_run_excess",
            "infantry_isolated_count",
            "infantry_distinct_files",
            "infantry_file_span",
            "infantry_rank_span",
            "infantry_frontier_count",
            "material_pair_distance_mean",
            "material_file_span",
            "material_rank_span",
        }
        self.assertTrue(appended_names.issubset(vertical))

    def test_v3_hq_approach_features_match_typescript_fixture(self) -> None:
        board = engine.BaseBoard(None)
        board.clear_board()
        for square, piece_type, color in (
            ("h1", engine.HQ, engine.RED),
            ("a8", engine.HQ, engine.BLUE),
            ("h3", engine.INFANTRY, engine.BLUE),
            ("f1", engine.ARMORED_INFANTRY, engine.BLUE),
            ("e4", engine.AIRBORNE_INFANTRY, engine.BLUE),
            ("g2", engine.INFANTRY, engine.RED),
        ):
            board.set_piece_at(
                engine.parse_square(square), engine.Piece(piece_type, color)
            )
        features = value_model._side_features(board, engine.RED)
        expected = {
            "hq_enemy_infantry_distance_min": 2.0,
            "hq_enemy_armored_infantry_distance_min": 2.0,
            "hq_enemy_airborne_infantry_distance_min": 6.0,
            "hq_enemy_infantry_within_two": 2.0,
            "hq_enemy_infantry_within_three": 2.0,
            "hq_friendly_infantry_within_two": 1.0,
            "hq_friendly_infantry_within_three": 1.0,
            "hq_attack_pressure": 5.0,
            "hq_defense_density": 2.0,
        }
        for name, expected_value in expected.items():
            self.assertEqual(features[name], expected_value, name)


if __name__ == "__main__":
    unittest.main()
