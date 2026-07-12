import unittest

import numpy as np
import torch

from model.spr_eeg import (
    EpochMemoryRecord,
    NTXentLoss,
    PurifiedEpochBuffer,
    purify_eeg_sequences,
    self_centered_clean_probabilities,
    symmetric_label_noise,
)


class SprEegTest(unittest.TestCase):
    def test_nt_xent_is_finite_and_backpropagates(self):
        first = torch.randn(8, 16, requires_grad=True)
        second = first.detach() + 0.05 * torch.randn(8, 16)
        second.requires_grad_(True)
        loss = NTXentLoss(temperature=0.5)(first, second)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(first.grad)
        self.assertGreater(float(first.grad.abs().sum()), 0.0)

    def test_symmetric_noise_always_changes_selected_labels(self):
        labels = np.arange(1000) % 5
        noisy, mask = symmetric_label_noise(labels, 0.4, 5, seed=17)
        self.assertTrue(np.all(noisy[mask] != labels[mask]))
        self.assertTrue(np.all(noisy[~mask] == labels[~mask]))
        self.assertAlmostEqual(float(mask.mean()), 0.4, delta=0.04)

    def test_pure_filter_has_no_confidence_input_and_lowers_outliers(self):
        rng = np.random.default_rng(9)
        features = np.concatenate(
            (
                np.array([[1.0, 0.0]]) + rng.normal(0, 0.02, size=(40, 2)),
                np.array([[0.0, 1.0]]) + rng.normal(0, 0.02, size=(8, 2)),
            )
        )
        observed_labels = np.zeros(48, dtype=np.int64)
        clean = self_centered_clean_probabilities(features, observed_labels, seed=3)
        self.assertGreater(clean[:40].mean(), clean[40:].mean())

    def test_purified_buffer_respects_source_and_total_capacity(self):
        source = [EpochMemoryRecord(f"source-{i // 20}", i % 20, i % 5, 1.0, i % 5) for i in range(80)]
        memory = PurifiedEpochBuffer(capacity=50, source_capacity=30, num_classes=5)
        memory.seed_source(source, seed=4)
        incoming = [EpochMemoryRecord(f"new-{i // 20}", i % 20, i % 5, 0.2 + i / 1000) for i in range(100)]
        memory.update(incoming)
        self.assertEqual(len(memory.source), 30)
        self.assertEqual(len(memory.dynamic), 20)
        self.assertEqual(len(memory), 50)

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
