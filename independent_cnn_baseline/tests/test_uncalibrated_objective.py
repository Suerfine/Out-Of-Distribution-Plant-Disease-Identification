"""Tests for the corrected objective using generated logits, without training."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dpd.evaluate import evaluate
from independent_cnn_baseline.uncalibrated_objective import validation_uncalibrated


class UncalibratedObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.pairs = pd.DataFrame({"pair_id": range(4), "crop_id": [0, 0, 1, 1],
            "disease_id": [0, 1, 0, 1], "pair": ["ax", "ay", "bx", "by"],
            "is_unseen": [False, False, False, True]})
        self.frames = {"validation_seen": pd.DataFrame({"pair_id": [0, 0]}),
                       "validation_unseen": pd.DataFrame({"pair_id": [1, 1]})}
        self.scores = np.array([[5., 0., -10., 100.], [5., 0., -10., 100.],
                                [2., 0., -10., 100.], [2., 0., -10., 100.]])

    def test_calibration_gain_cannot_enter_objective_and_real_unseen_is_hidden(self):
        with patch("independent_cnn_baseline.uncalibrated_objective.compose_pair_scores", return_value=self.scores.copy()), \
             patch("independent_cnn_baseline.engine.select_gamma", side_effect=AssertionError("No gamma search")):
            row = validation_uncalibrated(np.zeros((4, 2)), np.zeros((4, 2)), self.frames, self.pairs)
        self.assertEqual(row["gamma"], 0.)
        self.assertEqual(row["seen_top1"], 1.)
        self.assertEqual(row["unseen_top1"], 0.)
        self.assertEqual(row["harmonic_mean_top1"], 0.)
        masked = self.scores.copy()
        masked[:, 3] = -np.inf
        val_pairs = self.pairs.assign(is_unseen=[False, True, False, True])
        calibrated = evaluate(masked[:2], np.array([0, 0]), masked[2:], np.array([1, 1]), val_pairs, gamma=3.)
        self.assertEqual(calibrated["harmonic_mean_top1"], 1.)

    def test_real_unseen_validation_labels_and_nonfinite_logits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid validation factor logits"):
            validation_uncalibrated(np.full((4, 2), np.nan), np.zeros((4, 2)), self.frames, self.pairs)
        self.frames["validation_unseen"]["pair_id"] = [3, 3]
        with self.assertRaisesRegex(ValueError, "Official unseen"):
            validation_uncalibrated(np.zeros((4, 2)), np.zeros((4, 2)), self.frames, self.pairs)


if __name__ == "__main__":
    unittest.main()
