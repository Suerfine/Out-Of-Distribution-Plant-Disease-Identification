"""Regression checks for consistent composition ranking, including exact ties."""
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dpd.evaluate import (calibrate, evaluate, evaluate_split, predictions_table,
                          ranked_indices, seen_confusion_matrix, topk_hits)


def pair_fixture():
    ids = np.arange(333)
    return pd.DataFrame({
        "pair_id": ids, "crop_id": ids % 55, "disease_id": ids % 175,
        "crop": [f"crop_{i % 55}" for i in ids],
        "disease": [f"disease_{i % 175}" for i in ids],
        "pair": [f"pair_{i}" for i in ids],
        "is_unseen": np.isin(ids, [7, 10, 19, 39, 84, 107, 124, 133, 307, 332]),
    })


class RankingTests(unittest.TestCase):
    def test_lower_pair_id_wins_exact_ties_and_masked_candidates_stay_last(self):
        scores = np.array([2.0, 2.0, 3.0, -np.inf])
        np.testing.assert_array_equal(ranked_indices(scores), [2, 0, 1, 3])
        np.testing.assert_array_equal(ranked_indices(scores[None, :], 2), [[2, 0]])
        np.testing.assert_array_equal(ranked_indices(scores, 20), [2, 0, 1, 3])
        with self.assertRaisesRegex(ValueError, "positive"):
            ranked_indices(scores, 0)

    def test_top5_cutoff_ties_have_one_deterministic_winner(self):
        scores = np.zeros((2, 333))
        scores[:, :4] = 10
        scores[:, 4:6] = 9
        labels = np.array([4, 5])
        np.testing.assert_array_equal(topk_hits(scores, labels, 5), [True, False])

    def test_metrics_prediction_csv_and_confusion_matrix_agree_on_ties(self):
        pairs = pair_fixture()
        scores = np.zeros((3, 333))
        scores[2, 6:8] = 4
        labels = np.array([0, 5, 6])
        expected = np.array([0, 0, 6])
        metrics = evaluate_split(scores, labels, pairs)
        predictions = predictions_table(["a", "b", "c"], scores, labels, pairs)
        matrix = seen_confusion_matrix(scores, labels, pairs)
        np.testing.assert_array_equal(predictions.pred1, pairs.pair.to_numpy()[expected])
        self.assertEqual(metrics["top1"], predictions.correct_top1.mean())
        self.assertEqual(metrics["top1"], 2 / 3)
        self.assertEqual(metrics["top5"], 2 / 3)
        self.assertEqual(metrics["crop_top1"], np.mean(pairs.crop_id.to_numpy()[expected] == pairs.crop_id.to_numpy()[labels]))
        self.assertEqual(metrics["disease_top1"], np.mean(pairs.disease_id.to_numpy()[expected] == pairs.disease_id.to_numpy()[labels]))
        self.assertEqual(matrix.shape, (323, 333))
        self.assertEqual(matrix.to_numpy().sum(), 3)
        for label, predicted in zip(labels, expected):
            self.assertEqual(matrix.loc[f"pair_{label}", f"pair_{predicted}"], 1)

    def test_calibration_then_tie_rule_matches_live_app_and_unseen_diagnostic(self):
        from dpd.inference import rank_scores

        pairs = pair_fixture()
        scores = np.full((1, 333), -10.0)
        scores[0, 0], scores[0, 7] = 2, 1
        mask = pairs.is_unseen.to_numpy()
        for gamma, winner in ((1.0, 0), (1.5, 7)):
            with self.subTest(gamma=gamma):
                order, calibrated, _ = rank_scores(scores[0], mask, gamma, 1.0)
                self.assertEqual(order[0], winner)
                np.testing.assert_array_equal(order, ranked_indices(calibrate(scores, mask, gamma))[0])
                self.assertEqual(calibrated[order[0]], calibrated.max())
                predicted = predictions_table(["a"], scores, np.array([0]), pairs, gamma)
                self.assertEqual(predicted.pred1.iloc[0], f"pair_{winner}")
                matrix = seen_confusion_matrix(scores, np.array([0]), pairs, gamma)
                self.assertEqual(matrix.loc["pair_0", f"pair_{winner}"], 1)
        unseen_scores = np.zeros_like(scores)
        result = evaluate(scores, np.array([0]), unseen_scores, np.array([7]), pairs, gamma=1.0)
        self.assertEqual(result["diagnostic_unseen_only_top1"], 1.0)

    def test_saved_score_app_view_matches_prediction_csv_on_ties(self):
        from dpd import ui

        pairs = pair_fixture()
        scores = np.zeros((1, 333))
        scores[0, 6:8] = 4
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            np.savez(folder / "scores.npz", test_seen_scores=scores, test_unseen_scores=scores,
                     test_seen_labels=np.array([6]), test_unseen_labels=np.array([7]))
            for split, label in (("test_seen", 6), ("test_unseen", 7)):
                predictions_table([split + ".jpg"], scores, np.array([label]), pairs).to_csv(
                    folder / f"predictions_{split}.csv", index=False)
            with patch("dpd.ui.load_metadata", return_value=pairs):
                view = inspect.unwrap(ui.test_predictions)(str(folder), "cnn_independent", 0.0)
            np.testing.assert_array_equal(view.pred_id, [6, 6])
            np.testing.assert_array_equal(view.crop_ok & view.disease_ok, [True, False])
            np.testing.assert_array_equal(view.topk.iloc[0], [6, 7, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()
