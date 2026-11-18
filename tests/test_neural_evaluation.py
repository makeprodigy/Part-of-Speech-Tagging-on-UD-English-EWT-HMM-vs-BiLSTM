"""Tests for Phase 4 neural evaluation and OOV records."""

import unittest

import torch
from torch import nn

from src.bilstm_model import BiLSTMConfig, WordBiLSTMTagger
from src.error_analysis import collect_prediction_records, summarize_errors
from src.evaluation import evaluate_neural_batches, evaluate_predictions
from src.neural_data import collate_encoded_sentences
from src.preprocessing import build_neural_vocabulary, encode_sentences


class FixedPredictionModel(nn.Module):
    """Small deterministic model fixture for evaluation tests."""

    def __init__(self, predicted_tag_ids, tag_mapping):
        super().__init__()
        self.predicted_tag_ids = predicted_tag_ids
        self.tag_to_id = {
            tag: tag_id for tag_id, tag in enumerate(tag_mapping)
        }

    def predict_batch(self, batch):
        return self.predicted_tag_ids

    def forward_batch(self, batch):
        number_of_tags = 18
        logits = torch.full(
            (*batch.word_ids.shape, number_of_tags),
            -10.0,
        )
        for sentence_index, sentence in enumerate(self.predicted_tag_ids):
            for token_index, predicted_tag in enumerate(sentence):
                predicted_id = self.tag_to_id[predicted_tag]
                logits[sentence_index, token_index, predicted_id] = 10.0
        return logits


class TestNeuralEvaluation(unittest.TestCase):
    def setUp(self):
        sentences = [
            [("The", "DET"), ("dog", "NOUN")],
            [("runs", "VERB")],
        ]
        self.vocabulary = build_neural_vocabulary(sentences)
        encoded_sentences = encode_sentences(sentences, self.vocabulary)
        self.batch = collate_encoded_sentences(
            encoded_sentences,
            self.vocabulary,
        )
        self.tag_mapping = self.vocabulary.id_to_tag

    def test_padding_does_not_change_shared_metrics(self):
        gold = [["DET", "NOUN"]]
        predicted = [["DET", "DET"]]
        words = [["The", "dog"]]
        original_metrics = evaluate_predictions(
            gold,
            predicted,
            words,
            {"The"},
            labels=list(self.tag_mapping[1:]),
        )
        padded_metrics = evaluate_predictions(
            gold,
            predicted,
            words,
            {"The"},
            labels=list(self.tag_mapping[1:]),
        )
        self.assertEqual(
            original_metrics["token_accuracy"],
            padded_metrics["token_accuracy"],
        )
        self.assertEqual(
            original_metrics["confusion_matrix"].tolist(),
            padded_metrics["confusion_matrix"].tolist(),
        )

    def test_neural_evaluation_uses_real_tokens_and_canonical_labels(self):
        model = FixedPredictionModel([
            ["DET", "NOUN"],
            ["VERB"],
        ], self.tag_mapping)
        metrics, predictions = evaluate_neural_batches(
            model,
            [self.batch],
            self.tag_mapping,
            {"The", "dog"},
        )
        self.assertEqual(predictions, [["DET", "NOUN"], ["VERB"]])
        self.assertEqual(metrics["token_accuracy"], 1.0)
        self.assertEqual(metrics["unknown_word_count"], 1)
        self.assertEqual(
            len(metrics["confusion_matrix"]),
            len(self.tag_mapping) - 1,
        )

    def test_malformed_predictions_fail_loudly(self):
        model = FixedPredictionModel(
            [["DET"], ["VERB"]],
            self.tag_mapping,
        )
        with self.assertRaises(ValueError):
            evaluate_neural_batches(
                model,
                [self.batch],
                self.tag_mapping,
                {"The", "dog"},
            )

    def test_prediction_records_and_oov_summary(self):
        model = FixedPredictionModel([
            ["DET", "DET"],
            ["NOUN"],
        ], self.tag_mapping)
        records = collect_prediction_records(
            model,
            [self.batch],
            self.tag_mapping,
            {"The", "dog"},
            "fixture",
        )
        self.assertEqual(len(records), 3)
        self.assertFalse(records[0]["is_unknown_word"])
        self.assertTrue(records[2]["is_unknown_word"])
        summary = summarize_errors(records)
        self.assertEqual(summary["record_count"], 3)
        self.assertEqual(summary["error_count"], 2)
        self.assertEqual(summary["unknown_word_count"], 1)

    def test_crf_records_use_viterbi_predictions(self):
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
        records = collect_prediction_records(
            model,
            [self.batch],
            self.tag_mapping,
            {"The", "dog"},
            "crf",
            training_word_counts={"The": 2, "dog": 1},
        )
        self.assertEqual(len(records), 3)
        self.assertTrue(all("sentence_words" in record for record in records))
        self.assertTrue(all("word_frequency" in record for record in records))
        self.assertIn("error_rate_by_frequency", summarize_errors(records))


if __name__ == "__main__":
    unittest.main()
