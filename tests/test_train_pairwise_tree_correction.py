import unittest

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from scripts.train_pairwise_tree_correction import (
    correction_scores,
    fit_pairwise_tree_boosting,
    grouped_folds,
    pair_probabilities,
    rank_pairwise_features,
    remapped_tree,
)


class PairwiseTreeCorrectionTests(unittest.TestCase):
    def test_feature_ranking_uses_pair_difference_residual(self):
        left = np.asarray([[2.0, 0.0], [-2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        right = -left
        labels = np.asarray([1.0, 0.0, 1.0, 0.0])
        ranking = rank_pairwise_features(
            left,
            right,
            np.zeros(4),
            labels,
            np.ones(4),
        )
        self.assertEqual(ranking.tolist(), [0, 1])

    def test_boosting_learns_decomposable_pair_order(self):
        left = np.asarray([[2.0], [-2.0], [1.0], [-1.0]])
        right = -left
        labels = np.asarray([1.0, 0.0, 1.0, 0.0])
        trees = fit_pairwise_tree_boosting(
            left,
            right,
            np.zeros(4),
            labels,
            np.ones(4),
            n_estimators=12,
            max_depth=1,
            min_samples_leaf=1,
            learning_rate=0.2,
            random_state=7,
        )
        probabilities = pair_probabilities(
            trees, left, right, np.zeros(4), 0.2
        )
        self.assertTrue(np.all(probabilities[labels == 1] > 0.5))
        self.assertTrue(np.all(probabilities[labels == 0] < 0.5))
        scores = correction_scores(trees, np.asarray([[2.0], [-2.0]]), 0.2)
        self.assertGreater(scores[0], scores[1])

    def test_exported_tree_features_are_remapped(self):
        tree = DecisionTreeRegressor(max_depth=1, random_state=1)
        tree.fit(np.asarray([[0.0], [1.0]]), np.asarray([0.0, 1.0]))
        rendered = remapped_tree(tree, np.asarray([17]))
        self.assertEqual(rendered["feature"][0], 17)
        self.assertTrue(all(index in (17, -2) for index in rendered["feature"]))

    def test_grouped_folds_do_not_leak_source_games(self):
        records = [
            {"source_game_id": f"game-{game}"}
            for game in range(12)
            for _ in range(2)
        ]
        folds = grouped_folds(records, random_state=3, fold_count=3)
        seen = set()
        for fold in folds:
            games = {records[int(index)]["source_game_id"] for index in fold}
            self.assertFalse(seen & games)
            seen.update(games)
        self.assertEqual(seen, {f"game-{game}" for game in range(12)})

if __name__ == "__main__":
    unittest.main()
