import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from puridiver_eeg import (  # noqa: E402
    _classification_loss,
    _low_component_probability,
    build_purification_state,
)


class PuriDivEREEGTest(unittest.TestCase):
    def test_loss_gmm_assigns_higher_clean_probability_to_low_loss_mode(self):
        rng = np.random.default_rng(11)
        low_losses = rng.normal(0.08, 0.01, size=80)
        high_losses = rng.normal(0.82, 0.03, size=80)
        losses = np.concatenate((low_losses, high_losses))

        clean_probability = _low_component_probability(losses, seed=17)

        self.assertGreater(clean_probability[:80].mean(), 0.95)
        self.assertLess(clean_probability[80:].mean(), 0.05)

    def test_source_replay_is_protected_from_gmm_rejection(self):
        new_probabilities = torch.tensor(
            [[[0.9, 0.8], [0.1, 0.2]], [[0.2, 0.3], [0.8, 0.7]]],
            dtype=torch.float32,
        )
        replay_probabilities = torch.full((5, 2, 2), 0.5, dtype=torch.float32)
        replay_labels = torch.zeros((5, 2), dtype=torch.long)
        args = SimpleNamespace(
            seed=5,
            train_len=3,
            puridiver_soft_temperature=0.5,
            puridiver_clean_threshold=0.5,
            puridiver_uncertainty_threshold=0.5,
            model_param=SimpleNamespace(NumClasses=2),
        )

        predictions = [
            (new_probabilities, torch.zeros((2, 2), dtype=torch.long)),
            (new_probabilities, torch.zeros((2, 2), dtype=torch.long)),
            (replay_probabilities, replay_labels),
        ]
        with (
            patch("puridiver_eeg._collect_predictions", side_effect=predictions),
            patch(
                "puridiver_eeg._low_component_probability",
                side_effect=lambda values, seed: np.zeros_like(values, dtype=np.float32),
            ),
        ):
            state = build_purification_state(None, None, None, None, args)

        torch.testing.assert_close(state.replay_clean[:3], torch.ones((3, 2)))
        torch.testing.assert_close(state.replay_clean[3:], torch.zeros((2, 2)))
        torch.testing.assert_close(state.replay_low_uncertainty[:3], torch.zeros((3, 2)))
        self.assertEqual(state.summary["source_replay_protected"], 3)

    def test_classification_loss_uses_joint_weighted_sample_normalization(self):
        hard_ce = torch.tensor([1.0, 100.0, 100.0, 100.0])
        soft_ce = torch.tensor([100.0, 4.0, 8.0, 100.0])
        clean_mask = torch.tensor([True, False, False, False])
        relabel_mask = torch.tensor([False, True, True, False])

        loss = _classification_loss(
            hard_ce,
            soft_ce,
            clean_mask,
            relabel_mask,
            relabel_weight=0.5,
        )

        expected = (1.0 + 0.5 * 4.0 + 0.5 * 8.0) / (1.0 + 0.5 + 0.5)
        self.assertAlmostEqual(loss.item(), expected)


if __name__ == "__main__":
    unittest.main()
