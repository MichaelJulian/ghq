import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_hq_loss  # noqa: E402


class ExactHqLossAuditTests(unittest.TestCase):
    def test_classifies_an_unthreatened_hq_position_as_survivable(self):
        result = audit_hq_loss.audit_hq_loss(
            "q7/8/8/8/8/8/8/7Q - - b", example_limit=2
        )

        self.assertFalse(result["forced_hq_loss"])
        self.assertFalse(result["inconclusive"])
        self.assertFalse(result["exhaustive"])
        self.assertGreater(result["safe_turns"], 0)
        self.assertLessEqual(len(result["safe_examples"]), 2)

    def test_classifies_a_production_forced_mate_as_forced(self):
        result = audit_hq_loss.audit_hq_loss(
            "q7/7i/i7/1i2i3/i7/5i2/i7/6Qf - - r", example_limit=2
        )

        self.assertTrue(result["forced_hq_loss"])
        self.assertEqual(result["safe_turns"], 0)
        self.assertEqual(result["complete_turn_states"], 5)
        self.assertTrue(result["exhaustive"])

    def test_node_limit_is_inconclusive_never_a_false_forced_mate(self):
        result = audit_hq_loss.audit_hq_loss(
            "q7/8/8/8/8/8/8/7Q - - b",
            example_limit=2,
            max_nodes=1,
        )

        self.assertFalse(result["forced_hq_loss"])
        self.assertTrue(result["inconclusive"])
        self.assertFalse(result["exhaustive"])
        self.assertEqual(result["nodes_visited"], 1)

    def test_audit_hq_attack_search_collapses_artillery_orientation_clones(self):
        board = audit_hq_loss.engine.BaseBoard(
            "q7/8/8/8/8/8/8/R6Q R - r"
        )
        moves = list(audit_hq_loss.exact_hq_capture_moves(board))
        relocations = [
            (move.from_square, move.to_square)
            for move in moves
            if move.name == "MoveAndOrient"
        ]

        self.assertEqual(len(relocations), len(set(relocations)))
        self.assertFalse(any(move.name == "Skip" for move in moves))
        self.assertFalse(
            any(
                move.name == "MoveAndOrient"
                and move.from_square == move.to_square
                for move in moves
            )
        )

    def test_finds_remote_artillery_hq_interdiction_within_default_budget(self):
        result = audit_hq_loss.audit_hq_loss(
            "4p3/qi6/i1i5/8/I7/6i1/1H↑I4i/5P1Q - - r",
            example_limit=2,
        )

        self.assertFalse(result["forced_hq_loss"])
        self.assertFalse(result["inconclusive"])
        self.assertGreater(result["safe_turns"], 0)
        self.assertTrue(
            any(
                "b2c1→" in example["moves"]
                for example in result["safe_examples"]
            )
        )


if __name__ == "__main__":
    unittest.main()
