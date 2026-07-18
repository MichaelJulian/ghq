from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_tactical_hq_model", ROOT / "scripts" / "train_tactical_hq_model.py"
)
assert SPEC and SPEC.loader
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


class TacticalHqTrainerTest(unittest.TestCase):
    def test_color_swaps_are_one_evaluation_unit(self) -> None:
        first = {"generationId": "g", "gameId": "g-0001"}
        swap = {"generationId": "g", "gameId": "g-0002"}
        next_pair = {"generationId": "g", "gameId": "g-0003"}
        self.assertEqual(trainer.pair_id(first), trainer.pair_id(swap))
        self.assertNotEqual(trainer.pair_id(first), trainer.pair_id(next_pair))

    def test_feature_selection_is_tactical_and_append_only(self) -> None:
        names = [
            "turn_progress",
            "own_material_total",
            "own_hq_escape_squares",
            "opp_hq_attack_pressure",
            "diff_hq_enemy_infantry_distance_min",
            "own_board_armored_infantry",
        ]
        selected = [names[index] for index in trainer.tactical_feature_indices(names)]
        self.assertIn("turn_progress", selected)
        self.assertIn("own_hq_escape_squares", selected)
        self.assertIn("opp_hq_attack_pressure", selected)
        self.assertIn("diff_hq_enemy_infantry_distance_min", selected)
        self.assertIn("own_board_armored_infantry", selected)
        self.assertNotIn("own_material_total", selected)

    def test_group_split_never_leaks_a_color_pair(self) -> None:
        groups = np.asarray(
            [f"pair-{group:02d}" for group in range(30) for _ in range(4)],
            dtype=object,
        )
        labels = np.asarray([label for _ in range(30) for label in (0, 0, 0, 1)])
        split = trainer.split_indices(labels, groups, random_state=19)
        group_sets = {
            name: set(groups[indices].tolist()) for name, indices in split.items()
        }
        self.assertTrue(group_sets["train"].isdisjoint(group_sets["validation"]))
        self.assertTrue(group_sets["train"].isdisjoint(group_sets["test"]))
        self.assertTrue(group_sets["validation"].isdisjoint(group_sets["test"]))

    def test_high_precision_threshold_requires_two_flags(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 0])
        scores = np.asarray([0.95, 0.90, 0.60, 0.20, 0.10])
        threshold = trainer.high_precision_threshold(labels, scores, 0.95)
        self.assertIsNotNone(threshold)
        flagged = scores >= float(threshold)
        self.assertEqual(int(np.sum(flagged)), 2)
        self.assertEqual(int(np.sum(labels[flagged])), 2)


if __name__ == "__main__":
    unittest.main()
