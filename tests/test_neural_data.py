"""Tests for Phase 2 neural vocabulary, encoding, and batch contracts."""

import unittest

import torch

from src.neural_data import collate_encoded_sentences
from src.preprocessing import (
    PADDING_CHARACTER,
    PADDING_TAG,
    PADDING_WORD,
    UNKNOWN_CHARACTER,
    UNKNOWN_WORD,
    build_neural_vocabulary,
    encode_sentence,
    encode_sentences,
)


class TestNeuralVocabulary(unittest.TestCase):
    def setUp(self):
        self.training_sentences = [
            [("The", "DET"), ("dog", "NOUN")],
            [("runs", "VERB"), (".", "PUNCT")],
        ]
        self.vocabulary = build_neural_vocabulary(self.training_sentences)

    def test_training_only_mappings_and_special_symbols(self):
        self.assertEqual(
            self.vocabulary.original_training_words,
            frozenset({"The", "dog", "runs", "."}),
        )
        self.assertNotIn("cat", self.vocabulary.word_to_id)
        self.assertNotIn("x", self.vocabulary.char_to_id)
        self.assertEqual(self.vocabulary.id_to_word[0], PADDING_WORD)
        self.assertEqual(self.vocabulary.id_to_word[1], UNKNOWN_WORD)
        self.assertEqual(self.vocabulary.id_to_char[0], PADDING_CHARACTER)
        self.assertEqual(self.vocabulary.id_to_char[1], UNKNOWN_CHARACTER)
        self.assertEqual(self.vocabulary.id_to_tag[0], PADDING_TAG)
        self.assertGreater(self.vocabulary.tag_to_id["DET"], 0)

    def test_unknown_words_and_characters_are_encoded(self):
        encoded = encode_sentence(
            [("dogz", "NOUN")],
            self.vocabulary,
        )
        self.assertEqual(encoded.word_ids, (self.vocabulary.unknown_word_id,))
        self.assertEqual(
            encoded.character_ids[0][0],
            self.vocabulary.char_to_id["d"],
        )
        self.assertEqual(
            encoded.character_ids[0][3],
            self.vocabulary.unknown_character_id,
        )

    def test_invalid_training_data_is_rejected(self):
        with self.assertRaises(ValueError):
            build_neural_vocabulary([])
        with self.assertRaises(ValueError):
            build_neural_vocabulary([[("word", "INVALID")]])
        with self.assertRaises(ValueError):
            encode_sentence([], self.vocabulary)


class TestNeuralBatch(unittest.TestCase):
    def setUp(self):
        training_sentences = [
            [("The", "DET"), ("dog", "NOUN")],
            [("runs", "VERB")],
        ]
        vocabulary = build_neural_vocabulary(training_sentences)
        self.vocabulary = vocabulary
        self.encoded_sentences = encode_sentences(
            training_sentences,
            vocabulary,
        )

    def test_collation_shapes_and_masks(self):
        batch = collate_encoded_sentences(
            self.encoded_sentences,
            self.vocabulary,
        )

        self.assertEqual(batch.word_ids.shape, torch.Size([2, 2]))
        self.assertEqual(batch.character_ids.shape[0:2], torch.Size([2, 2]))
        self.assertEqual(batch.tag_ids.shape, torch.Size([2, 2]))
        self.assertEqual(batch.token_mask.dtype, torch.bool)
        self.assertTrue(torch.equal(
            batch.token_mask,
            torch.tensor([[True, True], [True, False]]),
        ))
        self.assertTrue(torch.equal(
            batch.sentence_lengths,
            torch.tensor([2, 1]),
        ))
        self.assertEqual(
            batch.word_ids[1, 1].item(),
            self.vocabulary.padding_word_id,
        )
        self.assertEqual(
            batch.tag_ids[1, 1].item(),
            self.vocabulary.padding_tag_id,
        )

    def test_original_words_and_empty_batch_contract(self):
        batch = collate_encoded_sentences(
            self.encoded_sentences,
            self.vocabulary,
        )
        self.assertEqual(batch.original_words[1], ("runs",))
        with self.assertRaises(ValueError):
            collate_encoded_sentences([], self.vocabulary)


if __name__ == "__main__":
    unittest.main()
