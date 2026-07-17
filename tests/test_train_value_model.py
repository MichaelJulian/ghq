import unittest

import numpy as np

from scripts.train_value_model import game_balanced_weights


class ValueModelWeightTests(unittest.TestCase):
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
