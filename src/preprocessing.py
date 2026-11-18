"""Corpus preprocessing and vocabulary utilities for HMM POS tagging."""

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from src.dataset_loader import UPOS_TAGS

START_TAG = "<START>"
END_TAG = "<END>"
UNKNOWN_WORD = "<UNK>"

# Phase 1 uses START_TAG, END_TAG, UNKNOWN_WORD, and HMMVocabulary below.
# Phase 2 adds separate padding and character symbols for neural batches.
PADDING_WORD = "<PAD>"
UNKNOWN_CHARACTER = "<UNK_CHAR>"
PADDING_CHARACTER = "<PAD_CHAR>"
PADDING_TAG = "<PAD_TAG>"


@dataclass(frozen=True)
class HMMVocabulary:
    """Phase 1 HMM vocabulary and empirical training lexicon."""

    words: frozenset[str]
    tags: tuple[str, ...]
    original_words: frozenset[str] = frozenset()


@dataclass(frozen=True)
class NeuralVocabulary:
    """Phase 2 training-only mappings used by neural sequence taggers."""

    word_to_id: Mapping[str, int]
    char_to_id: Mapping[str, int]
    tag_to_id: Mapping[str, int]
    id_to_word: tuple[str, ...]
    id_to_char: tuple[str, ...]
    id_to_tag: tuple[str, ...]
    original_training_words: frozenset[str]

    @property
    def padding_word_id(self) -> int:
        return self.word_to_id[PADDING_WORD]

    @property
    def padding_word(self) -> str:
        return PADDING_WORD

    @property
    def unknown_word_id(self) -> int:
        return self.word_to_id[UNKNOWN_WORD]

    @property
    def unknown_word(self) -> str:
        return UNKNOWN_WORD

    @property
    def padding_character_id(self) -> int:
        return self.char_to_id[PADDING_CHARACTER]

    @property
    def unknown_character_id(self) -> int:
        return self.char_to_id[UNKNOWN_CHARACTER]

    @property
    def padding_tag_id(self) -> int:
        return self.tag_to_id[PADDING_TAG]

    @property
    def padding_tag(self) -> str:
        return PADDING_TAG


@dataclass(frozen=True)
class EncodedSentence:
    """Integer representation of one sentence before batch padding."""

    word_ids: tuple[int, ...]
    character_ids: tuple[tuple[int, ...], ...]
    tag_ids: tuple[int, ...]
    original_words: tuple[str, ...]


# Phase 2: Neural vocabulary construction and sentence encoding.
def _create_mapping(values: Sequence[str]) -> Mapping[str, int]:
    """Create a stable read-only mapping and matching integer vocabulary."""
    return MappingProxyType({value: index for index, value in enumerate(values)})


def build_neural_vocabulary(
    sentences: list[list[tuple[str, str]]],
) -> NeuralVocabulary:
    """Build word, character, and UPOS mappings using training sentences only."""
    if not sentences or any(not sentence for sentence in sentences):
        raise ValueError("Training sentences must contain no empty sentences")

    training_words = {
        word for sentence in sentences for word, _ in sentence
    }
    training_characters = {
        character
        for sentence in sentences
        for word, _ in sentence
        for character in word
    }
    observed_tags = {
        tag for sentence in sentences for _, tag in sentence
    }
    unknown_tags = observed_tags - UPOS_TAGS
    if unknown_tags:
        raise ValueError(f"Training data contains invalid UPOS tags: {sorted(unknown_tags)}")

    words = (
        [PADDING_WORD, UNKNOWN_WORD]
        + sorted(training_words - {PADDING_WORD, UNKNOWN_WORD})
    )
    characters = (
        [PADDING_CHARACTER, UNKNOWN_CHARACTER]
        + sorted(training_characters - {PADDING_CHARACTER, UNKNOWN_CHARACTER})
    )
    tags = [PADDING_TAG] + sorted(UPOS_TAGS)

    return NeuralVocabulary(
        word_to_id=_create_mapping(words),
        char_to_id=_create_mapping(characters),
        tag_to_id=_create_mapping(tags),
        id_to_word=tuple(words),
        id_to_char=tuple(characters),
        id_to_tag=tuple(tags),
        original_training_words=frozenset(
            training_words - {PADDING_WORD, UNKNOWN_WORD}
        ),
    )


def encode_sentence(
    sentence: list[tuple[str, str]],
    vocabulary: NeuralVocabulary,
) -> EncodedSentence:
    """Encode one tagged sentence using a training-fitted neural vocabulary."""
    if not sentence:
        raise ValueError("Cannot encode an empty sentence")

    word_ids = []
    character_ids = []
    tag_ids = []
    original_words = []

    for word, tag in sentence:
        if not word or not tag:
            raise ValueError(f"Invalid token: {(word, tag)}")
        if tag not in vocabulary.tag_to_id or tag == PADDING_TAG:
            raise ValueError(f"Unknown or reserved UPOS tag: {tag}")

        word_ids.append(vocabulary.word_to_id.get(word, vocabulary.unknown_word_id))
        character_ids.append(tuple(
            vocabulary.char_to_id.get(
                character,
                vocabulary.unknown_character_id,
            )
            for character in word
        ))
        tag_ids.append(vocabulary.tag_to_id[tag])
        original_words.append(word)

    return EncodedSentence(
        word_ids=tuple(word_ids),
        character_ids=tuple(character_ids),
        tag_ids=tuple(tag_ids),
        original_words=tuple(original_words),
    )


def encode_sentences(
    sentences: list[list[tuple[str, str]]],
    vocabulary: NeuralVocabulary,
) -> list[EncodedSentence]:
    """Encode a collection of tagged sentences with one fitted vocabulary."""
    return [encode_sentence(sentence, vocabulary) for sentence in sentences]


def extract_original_vocabulary(
    sentences: list[list[tuple[str, str]]],
) -> set[str]:
    """Extract distinct empirical word types from training sentences, excluding <UNK>."""
    return {
        word
        for sentence in sentences
        for word, _ in sentence
        if word != UNKNOWN_WORD
    }


def build_vocabulary(
    sentences: list[list[tuple[str, str]]],
    minimum_word_frequency: int = 1,
) -> HMMVocabulary:
    """Build model vocabulary with frequency pruning and preserve empirical training lexicon."""
    word_counts = Counter()
    tags = set()

    for sentence in sentences:
        for word, tag in sentence:
            word_counts[word] += 1
            tags.add(tag)

    original_words = frozenset(
        word for word in word_counts if word != UNKNOWN_WORD
    )
    known_words = {
        word
        for word, count in word_counts.items()
        if count >= minimum_word_frequency and word != UNKNOWN_WORD
    }
    known_words.add(UNKNOWN_WORD)

    return HMMVocabulary(
        words=frozenset(known_words),
        tags=tuple(sorted(tags)),
        original_words=original_words,
    )


def normalize_word(word: str, vocabulary: HMMVocabulary) -> str:
    """Map unseen word to UNK token."""
    return word if word in vocabulary.words else UNKNOWN_WORD