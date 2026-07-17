import unittest

from scripts.analyze_human_quiet_streaks import (
    maximum_quiet_streak,
    strategic_metrics,
)


class HumanQuietStreakTests(unittest.TestCase):
    def test_capture_and_reinforcement_reset_the_streak(self):
        turns = {
            1: [{"type": "Move"}],
            2: [{"type": "Move"}],
            3: [{"type": "Move"}],
            4: [{"type": "Reinforce"}],
            5: [{"type": "Move"}],
            6: [{"type": "Move"}],
            7: [{"type": "Move"}],
            8: [{"type": "Move"}],
        }
        self.assertEqual(maximum_quiet_streak(turns, {3}), 4)

    def test_strategic_metrics_are_color_relative(self):
        board = [[None for _ in range(8)] for _ in range(8)]
        board[0][0] = {"type": "HQ", "player": "BLUE"}
        board[7][7] = {"type": "HQ", "player": "RED"}
        board[5][2] = {"type": "INFANTRY", "player": "RED"}
        board[2][5] = {"type": "INFANTRY", "player": "BLUE"}
        self.assertEqual(strategic_metrics(board, "RED"), (3, 5))
        self.assertEqual(strategic_metrics(board, "BLUE"), (3, 5))


if __name__ == "__main__":
    unittest.main()
