import unittest

import numpy as np

from model.spr_eeg import purify_eeg_sequences


class SprEegTest(unittest.TestCase):
    def test_mislabeled_feature_outliers_receive_lower_clean_probability(self):
        rng = np.random.default_rng(7)
        sequence_count, sequence_length = 20, 4
        true_labels = np.tile(np.array([[0, 0, 1, 1]]), (sequence_count, 1))
        centers = np.array([[1.0, 0.0], [0.0, 1.0]])
        features = centers[true_labels] + rng.normal(0.0, 0.03, size=(sequence_count, sequence_length, 2))
        pseudo_labels = true_labels.copy()
        corrupt_flat = np.array([0, 4, 8, 12, 18, 22, 26, 30])
        pseudo_labels.reshape(-1)[corrupt_flat] = 1 - pseudo_labels.reshape(-1)[corrupt_flat]
        confidences = np.full((sequence_count, sequence_length), 0.99)

        result = purify_eeg_sequences(
            features,
            pseudo_labels,
            confidences,
            confidence_threshold=0.9,
            min_confident_epochs=2,
            clean_threshold=0.5,
            min_clean_epochs=2,
            sequence_threshold=0.4,
            ensembles=5,
            bmm_iters=10,
            seed=13,
        )
        flat_probabilities = result.epoch_clean_probabilities.reshape(-1)
        clean_flat = np.setdiff1d(np.arange(flat_probabilities.size), corrupt_flat)
        self.assertGreater(flat_probabilities[clean_flat].mean(), flat_probabilities[corrupt_flat].mean())

    def test_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            purify_eeg_sequences(
                np.zeros((2, 3)),
                np.zeros((2, 3)),
                np.zeros((2, 3)),
                confidence_threshold=0.9,
                min_confident_epochs=2,
                clean_threshold=0.5,
                min_clean_epochs=2,
                sequence_threshold=0.5,
                ensembles=2,
                bmm_iters=2,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
