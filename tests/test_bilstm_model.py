"""Tests for the Phase 2 word-only BiLSTM and training loop."""

import unittest

import torch

from src.bilstm_model import BiLSTMConfig, WordBiLSTMTagger
from src.neural_data import collate_encoded_sentences
from src.preprocessing import build_neural_vocabulary, encode_sentences


def create_batches(sentences, vocabulary, batch_size):
    encoded_sentences = encode_sentences(sentences, vocabulary)
    return [
        collate_encoded_sentences(
            encoded_sentences[start:start + batch_size],
            vocabulary,
        )
        for start in range(0, len(encoded_sentences), batch_size)
    ]


class TestWordBiLSTMTagger(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            [("The", "DET"), ("dog", "NOUN"), ("runs", "VERB")],
            [("A", "DET"), ("cat", "NOUN")],
        ]
        self.vocabulary = build_neural_vocabulary(self.sentences)
        self.batches = create_batches(
            self.sentences,
            self.vocabulary,
            batch_size=2,
        )
        self.config = BiLSTMConfig(
            word_embedding_dim=8,
            hidden_size=6,
            max_epochs=2,
            random_seed=7,
        )

    def test_logits_have_expected_shape(self):
        model = WordBiLSTMTagger(self.vocabulary, self.config)
        logits = model(self.batches[0].word_ids)
        self.assertEqual(
            logits.shape,
            torch.Size([
                2,
                3,
                len(self.vocabulary.id_to_tag),
            ]),
        )

    def test_masked_loss_ignores_padding(self):
        model = WordBiLSTMTagger(self.vocabulary, self.config)
        batch = self.batches[0]
        logits = model(batch.word_ids)
        original_loss = model.compute_masked_loss(
            logits,
            batch.tag_ids,
            batch.token_mask,
        )

        padded_logits = torch.cat(
            [logits, torch.randn(2, 2, logits.shape[-1])],
            dim=1,
        )
        padded_tag_ids = torch.cat(
            [
                batch.tag_ids,
                torch.full(
                    (2, 2),
                    self.vocabulary.padding_tag_id,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
        padded_mask = torch.cat(
            [batch.token_mask, torch.zeros(2, 2, dtype=torch.bool)],
            dim=1,
        )
        padded_loss = model.compute_masked_loss(
            padded_logits,
            padded_tag_ids,
            padded_mask,
        )
        self.assertAlmostEqual(
            original_loss.item(),
            padded_loss.item(),
            places=6,
        )

    def test_softmax_loss_cannot_use_padding_tag(self):
        model = WordBiLSTMTagger(self.vocabulary, self.config)
        batch = self.batches[0]
        logits = torch.zeros(
            batch.tag_ids.shape[0],
            batch.tag_ids.shape[1],
            len(self.vocabulary.id_to_tag),
        )
        logits[..., self.vocabulary.padding_tag_id] = 100.0
        loss = model.compute_masked_loss(
            logits,
            batch.tag_ids,
            batch.token_mask,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertNotEqual(loss.item(), 0.0)

    def test_prediction_requires_fit_and_preserves_lengths(self):
        model = WordBiLSTMTagger(self.vocabulary, self.config)
        with self.assertRaises(RuntimeError):
            model.predict_batch(self.batches[0])

        model._is_fitted = True
        predictions = model.predict_batch(self.batches[0])
        self.assertEqual([len(sequence) for sequence in predictions], [3, 2])
        for sequence in predictions:
            self.assertTrue(
                set(sequence).issubset(set(self.vocabulary.id_to_tag[1:]))
            )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            BiLSTMConfig(hidden_size=0)
        with self.assertRaises(ValueError):
            BiLSTMConfig(dropout=1.0)

    def test_crf_decoding_returns_valid_sequences(self):
        model = WordBiLSTMTagger(
            self.vocabulary,
            BiLSTMConfig(
                word_embedding_dim=8,
                hidden_size=6,
                use_crf=True,
                max_epochs=1,
            ),
        )
        model._is_fitted = True
        predictions = model.predict_batch(self.batches[0])
        self.assertEqual([len(sequence) for sequence in predictions], [3, 2])
        for sequence in predictions:
            self.assertTrue(
                set(sequence).issubset(set(self.vocabulary.id_to_tag[1:]))
            )

    def test_crf_training_produces_finite_loss(self):
        model = WordBiLSTMTagger(
            self.vocabulary,
            BiLSTMConfig(
                word_embedding_dim=8,
                hidden_size=6,
                use_crf=True,
                max_epochs=1,
            ),
        )
        loss = model.compute_masked_loss(
            model(self.batches[0].word_ids),
            self.batches[0].tag_ids,
            self.batches[0].token_mask,
        )
        self.assertTrue(torch.isfinite(loss))


class TestWordBiLSTMTraining(unittest.TestCase):
    def test_one_batch_update_changes_parameters(self):
        sentences = [
            [("alpha", "NOUN"), ("runs", "VERB")],
            [("beta", "NOUN"), ("sleeps", "VERB")],
        ]
        vocabulary = build_neural_vocabulary(sentences)
        batches = create_batches(sentences, vocabulary, batch_size=2)
        model = WordBiLSTMTagger(
            vocabulary,
            BiLSTMConfig(
                word_embedding_dim=8,
                hidden_size=6,
                max_epochs=1,
                random_seed=13,
            ),
        )
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        model.fit(batches, batches)

        self.assertTrue(any(
            not torch.equal(initial_parameters[name], parameter)
            for name, parameter in model.named_parameters()
        ))

    def test_fixed_seed_reproduces_training(self):
        sentences = [
            [("alpha", "NOUN"), ("runs", "VERB")],
            [("beta", "NOUN"), ("sleeps", "VERB")],
        ]
        vocabulary = build_neural_vocabulary(sentences)
        batches = create_batches(sentences, vocabulary, batch_size=2)
        config = BiLSTMConfig(
            word_embedding_dim=8,
            hidden_size=6,
            max_epochs=2,
            random_seed=17,
        )
        torch.manual_seed(config.random_seed)
        first_model = WordBiLSTMTagger(vocabulary, config)
        torch.manual_seed(config.random_seed)
        second_model = WordBiLSTMTagger(vocabulary, config)

        first_model.fit(batches, batches)
        second_model.fit(batches, batches)

        for first_parameter, second_parameter in zip(
            first_model.parameters(),
            second_model.parameters(),
        ):
            self.assertTrue(torch.equal(first_parameter, second_parameter))

    def test_tiny_dataset_can_be_overfit(self):
        sentences = [
            [("alpha", "NOUN"), ("runs", "VERB")],
            [("beta", "NOUN"), ("sleeps", "VERB")],
        ]
        vocabulary = build_neural_vocabulary(sentences)
        batches = create_batches(sentences, vocabulary, batch_size=2)
        model = WordBiLSTMTagger(
            vocabulary,
            BiLSTMConfig(
                word_embedding_dim=10,
                hidden_size=8,
                learning_rate=0.03,
                max_epochs=60,
                random_seed=11,
            ),
        )

        history = model.fit(batches, batches)

        self.assertTrue(history)
        self.assertLess(history[-1].development_loss, history[0].development_loss)
        predictions = model.predict_batch(batches[0])
        self.assertEqual(predictions, [["NOUN", "VERB"], ["NOUN", "VERB"]])


if __name__ == "__main__":
    unittest.main()
