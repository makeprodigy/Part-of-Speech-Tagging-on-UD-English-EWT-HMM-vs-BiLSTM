"""Tests for the Phase 3 character-enhanced BiLSTM."""

import unittest

import torch

from src.bilstm_model import (
    BiLSTMConfig,
    CharacterBiLSTMEncoder,
    CharacterBiLSTMTagger,
    WordBiLSTMTagger,
)
from src.neural_data import collate_encoded_sentences
from src.preprocessing import build_neural_vocabulary, encode_sentences


def create_batch(sentences, vocabulary):
    encoded_sentences = encode_sentences(sentences, vocabulary)
    return collate_encoded_sentences(encoded_sentences, vocabulary)


class TestCharacterBiLSTMEncoder(unittest.TestCase):
    def setUp(self):
        self.training_sentences = [
            [("train", "NOUN"), ("runs", "VERB")],
            [("word", "NOUN")],
        ]
        self.vocabulary = build_neural_vocabulary(self.training_sentences)
        self.config = BiLSTMConfig(
            character_embedding_dim=8,
            character_hidden_size=6,
        )

    def test_representation_shape_for_variable_word_lengths(self):
        batch = create_batch(self.training_sentences, self.vocabulary)
        encoder = CharacterBiLSTMEncoder(self.vocabulary, self.config)
        representations = encoder(
            batch.character_ids,
            batch.character_lengths,
        )
        self.assertEqual(
            representations.shape,
            torch.Size([
                2,
                2,
                2 * self.config.character_hidden_size,
            ]),
        )

    def test_unseen_word_receives_character_representation(self):
        batch = create_batch(
            [[("trained", "NOUN")]],
            self.vocabulary,
        )
        encoder = CharacterBiLSTMEncoder(self.vocabulary, self.config)
        representations = encoder(
            batch.character_ids,
            batch.character_lengths,
        )
        self.assertEqual(representations.shape, torch.Size([1, 1, 12]))
        self.assertTrue(torch.isfinite(representations).all())

    def test_character_padding_does_not_change_representation(self):
        batch = create_batch(
            [[("train", "NOUN")], [("word", "NOUN")]],
            self.vocabulary,
        )
        encoder = CharacterBiLSTMEncoder(self.vocabulary, self.config)
        encoder.eval()

        original_representation = encoder(
            batch.character_ids[:1],
            batch.character_lengths[:1],
        )
        padded_character_ids = torch.cat(
            [
                batch.character_ids[:1],
                torch.full(
                    (1, 1, 3),
                    self.vocabulary.padding_character_id,
                    dtype=torch.long,
                ),
            ],
            dim=-1,
        )
        padded_representation = encoder(
            padded_character_ids,
            batch.character_lengths[:1],
        )
        self.assertTrue(torch.allclose(
            original_representation,
            padded_representation,
        ))


class TestCharacterBiLSTMTagger(unittest.TestCase):
    def test_character_model_matches_word_model_contract(self):
        sentences = [
            [("The", "DET"), ("dog", "NOUN")],
            [("runs", "VERB")],
        ]
        vocabulary = build_neural_vocabulary(sentences)
        batch = create_batch(sentences, vocabulary)
        config = BiLSTMConfig(
            word_embedding_dim=8,
            character_embedding_dim=6,
            character_hidden_size=5,
            hidden_size=7,
        )
        word_model = WordBiLSTMTagger(vocabulary, config)
        character_model = CharacterBiLSTMTagger(vocabulary, config)

        word_logits = word_model.forward_batch(batch)
        character_logits = character_model.forward_batch(batch)

        expected_shape = torch.Size([2, 2, len(vocabulary.id_to_tag)])
        self.assertEqual(word_logits.shape, expected_shape)
        self.assertEqual(character_logits.shape, expected_shape)

        character_model._is_fitted = True
        predictions = character_model.predict_batch(batch)
        self.assertEqual([len(sequence) for sequence in predictions], [2, 1])
        self.assertTrue(all(
            tag != vocabulary.id_to_tag[vocabulary.padding_tag_id]
            for sequence in predictions
            for tag in sequence
        ))

    def test_character_model_can_train_with_shared_training_loop(self):
        sentences = [
            [("alpha", "NOUN"), ("runs", "VERB")],
            [("beta", "NOUN"), ("sleeps", "VERB")],
        ]
        vocabulary = build_neural_vocabulary(sentences)
        batch = create_batch(sentences, vocabulary)
        model = CharacterBiLSTMTagger(
            vocabulary,
            BiLSTMConfig(
                word_embedding_dim=8,
                character_embedding_dim=6,
                character_hidden_size=5,
                hidden_size=7,
                learning_rate=0.03,
                max_epochs=2,
                random_seed=19,
            ),
        )

        history = model.fit([batch], [batch])

        self.assertEqual(len(history), 2)
        self.assertTrue(all(
            metric.development_loss >= 0
            for metric in history
        ))


if __name__ == "__main__":
    unittest.main()
