"""BiLSTM taggers used in the Phase 2 POS experiments."""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from src.crf_layer import LinearChainCRF
from src.neural_data import NeuralBatch
from src.preprocessing import NeuralVocabulary


@dataclass(frozen=True)
class BiLSTMConfig:
    """Settings shared by the word and word-plus-character models."""

    word_embedding_dim: int = 128
    character_embedding_dim: int = 32
    character_hidden_size: int = 32
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 20
    early_stopping_patience: int = 4
    random_seed: int = 42
    use_crf: bool = False

    def __post_init__(self):
        if self.word_embedding_dim < 1:
            raise ValueError("word_embedding_dim must be positive")
        if self.character_embedding_dim < 1:
            raise ValueError("character_embedding_dim must be positive")
        if self.character_hidden_size < 1:
            raise ValueError("character_hidden_size must be positive")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")


class WordBiLSTMTagger(nn.Module):
    """BiLSTM tagger that starts with word embeddings."""

    def __init__(
        self,
        vocabulary: NeuralVocabulary,
        config: BiLSTMConfig | None = None,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.config = config or BiLSTMConfig()
        self._is_fitted = False

        self.word_embeddings = nn.Embedding(
            num_embeddings=len(vocabulary.id_to_word),
            embedding_dim=self.config.word_embedding_dim,
            padding_idx=vocabulary.padding_word_id,
        )
        self.sentence_encoder = nn.LSTM(
            input_size=self.config.word_embedding_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                self.config.dropout
                if self.config.num_layers > 1
                else 0.0
            ),
        )
        self.input_dropout = nn.Dropout(self.config.dropout)
        self.tag_classifier = nn.Linear(
            2 * self.config.hidden_size,
            len(vocabulary.id_to_tag),
        )
        if self.config.use_crf:
            self.crf = LinearChainCRF(
                tag_count=len(vocabulary.id_to_tag),
                padding_tag_id=vocabulary.padding_tag_id,
            )

    def forward(self, word_ids: Tensor) -> Tensor:
        """Return one tag-score vector for each padded token."""
        if word_ids.ndim != 2:
            raise ValueError(
                "word_ids must have shape (batch_size, sequence_length)"
            )

        word_representations = self.word_embeddings(word_ids)
        return self._encode_sentence(word_representations)

    def forward_batch(self, batch: NeuralBatch) -> Tensor:
        """Run the word-only model on a batch."""
        return self.forward(batch.word_ids)

    def _encode_sentence(self, word_representations: Tensor) -> Tensor:
        """Apply dropout, the sentence encoder, and the tag layer."""
        word_representations = self.input_dropout(word_representations)
        contextual_representations, _ = self.sentence_encoder(
            word_representations
        )
        return self.tag_classifier(contextual_representations)

    def compute_masked_loss(
        self,
        logits: Tensor,
        tag_ids: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        """Calculate the loss without counting padded positions."""
        if logits.ndim != 3:
            raise ValueError("logits must have shape (batch, sequence, tags)")
        if tag_ids.shape != logits.shape[:2]:
            raise ValueError("tag_ids must match the first two logits dimensions")
        if token_mask.shape != tag_ids.shape or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be Boolean and match tag_ids")
        if not torch.any(token_mask):
            raise ValueError("token_mask must contain at least one real token")

        if self.config.use_crf:
            sentence_lengths = token_mask.sum(dim=1)
            return self.crf.neg_log_likelihood(
                logits,
                tag_ids,
                sentence_lengths,
            )
        active_logits = self.mask_padding_tag_logits(logits)[token_mask]
        active_tag_ids = tag_ids[token_mask]
        return nn.functional.cross_entropy(active_logits, active_tag_ids)

    def mask_padding_tag_logits(self, logits: Tensor) -> Tensor:
        """Keep the padding symbol out of real-token predictions."""
        masked_logits = logits.clone()
        masked_logits[..., self.vocabulary.padding_tag_id] = float("-inf")
        return masked_logits

    def predict_batch(self, batch: NeuralBatch) -> list[list[str]]:
        """Return one unpadded tag sequence per sentence."""
        if not self._is_fitted:
            raise RuntimeError("The model must be fitted before prediction")

        self.eval()
        with torch.no_grad():
            logits = self.forward_batch(batch)
            predicted_tag_ids = self.decode_batch(logits, batch.sentence_lengths)

        predictions = []
        for sentence_index, sentence_length in enumerate(
            batch.sentence_lengths.tolist()
        ):
            sentence_tag_ids = predicted_tag_ids[
                sentence_index,
                :sentence_length,
            ].tolist()
            predictions.append([
                self.vocabulary.id_to_tag[tag_id]
                for tag_id in sentence_tag_ids
            ])
        return predictions

    def decode_batch(self, logits: Tensor, sentence_lengths: Tensor) -> Tensor:
        """Use argmax for softmax models and Viterbi for CRF models."""
        logits = self.mask_padding_tag_logits(logits)
        if not self.config.use_crf:
            return logits.argmax(dim=-1)

        return self.crf.decode(logits, sentence_lengths)

    def fit(
        self,
        training_batches: Iterable[NeuralBatch],
        development_batches: Iterable[NeuralBatch],
        device: str | torch.device = "cpu",
    ):
        """Train and restore the checkpoint with the best dev F1."""
        from src.training import train_bilstm

        history = train_bilstm(
            model=self,
            training_batches=training_batches,
            development_batches=development_batches,
            device=device,
        )
        self._is_fitted = True
        return history


class CharacterBiLSTMEncoder(nn.Module):
    """Turn the characters of each word into a fixed-size vector."""

    def __init__(
        self,
        vocabulary: NeuralVocabulary,
        config: BiLSTMConfig,
    ):
        super().__init__()
        self.character_embeddings = nn.Embedding(
            num_embeddings=len(vocabulary.id_to_char),
            embedding_dim=config.character_embedding_dim,
            padding_idx=vocabulary.padding_character_id,
        )
        self.character_encoder = nn.LSTM(
            input_size=config.character_embedding_dim,
            hidden_size=config.character_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        character_ids: Tensor,
        character_lengths: Tensor,
    ) -> Tensor:
        """Encode `(batch, sentence, word)` characters as word vectors."""
        if character_ids.ndim != 3:
            raise ValueError(
                "character_ids must have shape (batch, sentence, word)"
            )
        if character_lengths.shape != character_ids.shape[:2]:
            raise ValueError(
                "character_lengths must match batch and sentence dimensions"
            )
        if torch.any(character_lengths < 0):
            raise ValueError("character_lengths cannot be negative")

        batch_size, sentence_length, max_word_length = character_ids.shape
        flattened_character_ids = character_ids.reshape(
            batch_size * sentence_length,
            max_word_length,
        )
        flattened_lengths = character_lengths.reshape(-1).to(torch.long)
        # Phase 3: padded sentence positions have length zero. Pack them as
        # one PAD_CHAR step; token_mask excludes their sentence-level logits.
        packed_lengths = flattened_lengths.clamp_min(1)
        embedded_characters = self.character_embeddings(flattened_character_ids)
        packed_characters = nn.utils.rnn.pack_padded_sequence(
            embedded_characters,
            packed_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden_states, _) = self.character_encoder(packed_characters)
        last_forward_state = hidden_states[-2]
        last_backward_state = hidden_states[-1]
        word_character_vectors = torch.cat(
            [last_forward_state, last_backward_state],
            dim=-1,
        )
        return word_character_vectors.reshape(
            batch_size,
            sentence_length,
            2 * self.character_encoder.hidden_size,
        )


class CharacterBiLSTMTagger(WordBiLSTMTagger):
    """BiLSTM tagger using word and character representations."""

    def __init__(
        self,
        vocabulary: NeuralVocabulary,
        config: BiLSTMConfig | None = None,
    ):
        super().__init__(vocabulary, config)
        self.character_encoder = CharacterBiLSTMEncoder(
            vocabulary,
            self.config,
        )
        self.sentence_encoder = nn.LSTM(
            input_size=(
                self.config.word_embedding_dim
                + 2 * self.config.character_hidden_size
            ),
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                self.config.dropout
                if self.config.num_layers > 1
                else 0.0
            ),
        )

    def forward(self, word_ids: Tensor) -> Tensor:
        """Reject word-only calls because character tensors are required."""
        raise ValueError(
            "CharacterBiLSTMTagger.forward requires a NeuralBatch; "
            "call forward_batch instead"
        )

    def forward_batch(self, batch: NeuralBatch) -> Tensor:
        """Return logits from concatenated word and character vectors."""
        word_representations = self.word_embeddings(batch.word_ids)
        character_representations = self.character_encoder(
            batch.character_ids,
            batch.character_lengths,
        )
        combined_representations = torch.cat(
            [word_representations, character_representations],
            dim=-1,
        )
        return self._encode_sentence(combined_representations)
