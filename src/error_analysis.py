"""Prediction records and OOV summaries for the neural experiments."""

from collections import Counter

import torch

from src.neural_data import NeuralBatch
from src.preprocessing import UNKNOWN_WORD


def collect_prediction_records(
    model,
    batches: list[NeuralBatch],
    tag_mapping: tuple[str, ...],
    training_vocabulary: set[str] | frozenset[str],
    model_name: str,
    training_word_counts: dict[str, int] | None = None,
) -> list[dict]:
    """Keep the prediction details needed for later error analysis."""
    if not batches:
        raise ValueError("Cannot collect records from an empty batch collection")

    clean_training_vocabulary = (
        set(training_vocabulary) - {UNKNOWN_WORD}
    )
    records = []
    model.eval()

    with torch.no_grad():
        for batch in batches:
            logits = model.forward_batch(batch)
            if hasattr(model, "decode_batch"):
                predicted_tag_ids = model.decode_batch(
                    logits,
                    batch.sentence_lengths,
                )
            else:
                if hasattr(model, "mask_padding_tag_logits"):
                    logits = model.mask_padding_tag_logits(logits)
                predicted_tag_ids = logits.argmax(dim=-1)
            confidence_scores = logits.softmax(dim=-1).amax(dim=-1)

            for sentence_index, sentence_words in enumerate(batch.original_words):
                sentence_length = int(
                    batch.sentence_lengths[sentence_index].item()
                )
                for token_index, word in enumerate(sentence_words):
                    if token_index >= sentence_length:
                        break
                    gold_tag_id = int(
                        batch.tag_ids[sentence_index, token_index].item()
                    )
                    predicted_tag_id = int(
                        predicted_tag_ids[sentence_index, token_index].item()
                    )
                    gold_tag = tag_mapping[gold_tag_id]
                    predicted_tag = tag_mapping[predicted_tag_id]
                    word_frequency = (
                        training_word_counts.get(word, 0)
                        if training_word_counts is not None
                        else None
                    )
                    records.append({
                        "model_name": model_name,
                        "sentence_index": sentence_index,
                        "token_index": token_index,
                        "word": word,
                        "is_unknown_word": word not in clean_training_vocabulary,
                        "gold_tag": gold_tag,
                        "predicted_tag": predicted_tag,
                        "is_correct": gold_tag == predicted_tag,
                        "confidence": float(
                            confidence_scores[sentence_index, token_index].item()
                        ),
                        "sentence_length": sentence_length,
                        "sentence_words": list(sentence_words),
                        "word_frequency": word_frequency,
                    })
    return records


def summarize_errors(records: list[dict]) -> dict:
    """Summarize errors, OOV behavior, and frequency bands."""
    error_records = [
        record for record in records if not record["is_correct"]
    ]
    confusion_counts = Counter(
        f"{record['gold_tag']} -> {record['predicted_tag']}"
        for record in error_records
    )
    oov_error_counts = Counter(
        f"{record['gold_tag']} -> {record['predicted_tag']}"
        for record in error_records
        if record["is_unknown_word"]
    )
    frequency_groups = Counter()
    frequency_errors = Counter()
    for record in records:
        frequency = record["word_frequency"]
        if frequency is None:
            continue
        if frequency == 0:
            group = "OOV"
        elif frequency == 1:
            group = "1"
        elif frequency <= 5:
            group = "2-5"
        elif frequency <= 20:
            group = "6-20"
        else:
            group = ">20"
        frequency_groups[group] += 1
        frequency_errors[group] += not record["is_correct"]
    return {
        "record_count": len(records),
        "error_count": len(error_records),
        "unknown_word_count": sum(
            record["is_unknown_word"] for record in records
        ),
        "unknown_word_error_count": sum(
            record["is_unknown_word"] for record in error_records
        ),
        "top_confusions": confusion_counts.most_common(10),
        "top_unknown_word_confusions": oov_error_counts.most_common(10),
        "error_rate_by_frequency": {
            group: {
                "count": frequency_groups[group],
                "errors": frequency_errors[group],
                "error_rate": (
                    frequency_errors[group] / frequency_groups[group]
                ),
            }
            for group in frequency_groups
        },
    }
