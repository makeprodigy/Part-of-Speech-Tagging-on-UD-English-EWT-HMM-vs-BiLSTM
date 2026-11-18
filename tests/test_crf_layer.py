import itertools
import unittest

import torch

from src.crf_layer import LinearChainCRF

class TestLinearChainCRF(unittest.TestCase):
    def setUp(self):
        self.crf = LinearChainCRF(tag_count=3, padding_tag_id=0)
        with torch.no_grad():
            self.crf.start_transitions.copy_(torch.tensor([0.0, 0.2, -0.1]))
            self.crf.end_transitions.copy_(torch.tensor([0.0, -0.2, 0.3]))
            self.crf.transitions.copy_(torch.tensor([
                [0.0, 0.0, 0.0],
                [0.0, 0.4, -0.3],
                [0.0, -0.2, 0.5],
            ]))
        self.emissions = torch.tensor([[
            [0.0, 1.0, 0.2],
            [0.0, 0.1, 1.2],
        ]])

    def test_partition_matches_brute_force(self):
        scores = []
        for tags in itertools.product((1, 2), repeat=2):
            scores.append(
                self.crf._sequence_score(self.emissions[0], torch.tensor(tags), 2)
            )
        expected = torch.logsumexp(torch.stack(scores), dim=0)
        actual = self.crf._log_partition(self.emissions[0], 2)
        self.assertTrue(torch.allclose(actual, expected))

    def test_viterbi_returns_best_sequence(self):
        decoded = self.crf.decode(
            self.emissions,
            torch.tensor([2]),
        )
        expected = max(
            itertools.product((1, 2), repeat=2),
            key=lambda tags: self.crf._sequence_score(
                self.emissions[0],
                torch.tensor(tags),
                2,
            ).item(),
        )
        self.assertEqual(decoded[0, :2].tolist(), list(expected))
        self.assertEqual(decoded.shape, (1, 2))

    def test_loss_is_finite(self):
        loss = self.crf.neg_log_likelihood(
            self.emissions,
            torch.tensor([[1, 2]]),
            torch.tensor([2]),
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
