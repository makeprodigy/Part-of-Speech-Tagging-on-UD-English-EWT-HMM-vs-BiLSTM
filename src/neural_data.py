from src.preprocessing import encode_sentence, encode_sentences
"""Batch collation utilities for neural sequence taggers."""

from dataclasses import dataclass

import torch

from src.preprocessing import (
    EncodedSentence,
    NeuralVocabulary,
)


@dataclass(frozen=True)
class NeuralBatch:
    """Padded tensors and metadata for one batch of encoded sentences."""

    word_ids: torch.Tensor
    character_ids: torch.Tensor
    tag_ids: torch.Tensor
    token_mask: torch.Tensor
    sentence_lengths: torch.Tensor
    character_lengths: torch.Tensor
    original_words: tuple[tuple[str, ...], ...]


def collate_encoded_sentences(
    sentences: list[EncodedSentence],
    vocabulary: NeuralVocabulary,
) -> NeuralBatch:
    """Pad encoded sentences to the largest sentence and word in the batch."""
    if not sentences:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(sentences)
    max_sentence_length = max(len(sentence.word_ids) for sentence in sentences)
    max_word_length = max(
        (
            len(characters)
            for sentence in sentences
            for characters in sentence.character_ids
        ),
        default=1,
    )

    word_ids = torch.full(
        (batch_size, max_sentence_length),
        vocabulary.padding_word_id,
        dtype=torch.long,
    )
    character_ids = torch.full(
        (batch_size, max_sentence_length, max_word_length),
        vocabulary.padding_character_id,
        dtype=torch.long,
    )
    tag_ids = torch.full(
        (batch_size, max_sentence_length),
        vocabulary.padding_tag_id,
        dtype=torch.long,
    )
    token_mask = torch.zeros(
        (batch_size, max_sentence_length),
        dtype=torch.bool,
    )
    sentence_lengths = torch.zeros(batch_size, dtype=torch.long)
    character_lengths = torch.zeros(
        (batch_size, max_sentence_length),
        dtype=torch.long,
    )

    for sentence_index, sentence in enumerate(sentences):
        sentence_length = len(sentence.word_ids)
        sentence_lengths[sentence_index] = sentence_length
        token_mask[sentence_index, :sentence_length] = True
        word_ids[sentence_index, :sentence_length] = torch.tensor(
            sentence.word_ids,
            dtype=torch.long,
        )
        tag_ids[sentence_index, :sentence_length] = torch.tensor(
            sentence.tag_ids,
            dtype=torch.long,
        )

        for token_index, characters in enumerate(sentence.character_ids):
            character_length = len(characters)
            character_lengths[sentence_index, token_index] = character_length
            character_ids[
                sentence_index,
                token_index,
                :character_length,
            ] = torch.tensor(characters, dtype=torch.long)

    return NeuralBatch(
        word_ids=word_ids,
        character_ids=character_ids,
        tag_ids=tag_ids,
        token_mask=token_mask,
        sentence_lengths=sentence_lengths,
        character_lengths=character_lengths,
        original_words=tuple(sentence.original_words for sentence in sentences),
    )
