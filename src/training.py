"""Training helpers for the Phase 2 neural taggers."""

import copy
import random
from dataclasses import asdict, dataclass
from typing import Iterable

# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import torch
from sklearn.metrics import f1_score
# pyrefly: ignore[missing-import]
from torch import nn
from tqdm.auto import tqdm

# pyrefly: ignore[missing-import]
from src.bilstm_model import WordBiLSTMTagger
from src.neural_data import NeuralBatch


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics collected after one training epoch."""

    epoch: int
    training_loss: float
    development_loss: float
    development_accuracy: float
    development_macro_f1: float


def set_random_seed(seed: int) -> None:
    """Seed the libraries used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: NeuralBatch, device: torch.device) -> NeuralBatch:
    """Move a batch to the selected device."""
    return NeuralBatch(
        word_ids=batch.word_ids.to(device),
        character_ids=batch.character_ids.to(device),
        tag_ids=batch.tag_ids.to(device),
        token_mask=batch.token_mask.to(device),
        sentence_lengths=batch.sentence_lengths.to(device),
        character_lengths=batch.character_lengths.to(device),
        original_words=batch.original_words,
    )


def _evaluate_model(
    model: WordBiLSTMTagger,
    batches: list[NeuralBatch],
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate loss, accuracy, and macro F1 on development data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    gold_tag_ids = []
    predicted_tag_ids = []

    with torch.no_grad():
        for original_batch in batches:
            batch = _move_batch(original_batch, device)
            logits = model.forward_batch(batch)
            loss = model.compute_masked_loss(
                logits,
                batch.tag_ids,
                batch.token_mask,
            )
            active_predictions = model.decode_batch(
                logits,
                batch.sentence_lengths,
            )[batch.token_mask]
            active_gold = batch.tag_ids[batch.token_mask]
            token_count = int(active_gold.numel())
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
            gold_tag_ids.extend(active_gold.cpu().tolist())
            predicted_tag_ids.extend(active_predictions.cpu().tolist())

    if total_tokens == 0:
        raise ValueError("Development batches contain no real tokens")

    return (
        total_loss / total_tokens,
        float(np.mean(
            np.asarray(gold_tag_ids) == np.asarray(predicted_tag_ids)
        )),
        float(f1_score(
            gold_tag_ids,
            predicted_tag_ids,
            average="macro",
            zero_division=0,
        )),
    )


def train_bilstm(
    model: WordBiLSTMTagger,
    training_batches: Iterable[NeuralBatch],
    development_batches: Iterable[NeuralBatch],
    device: str | torch.device = "cpu",
) -> list[EpochMetrics]:
    """Train a model and keep its best development checkpoint."""
    set_random_seed(model.config.random_seed)
    selected_device = torch.device(device)
    model.to(selected_device)
    train_batches = list(training_batches)
    dev_batches = list(development_batches)
    if not train_batches or not dev_batches:
        raise ValueError("Training and development batches must not be empty")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model.config.learning_rate,
    )
    best_macro_f1 = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, model.config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        batch_iterator = tqdm(
            train_batches,
            desc=f"Epoch {epoch}/{model.config.max_epochs}",
            unit="batch",
            leave=False,
        )
        for original_batch in batch_iterator:
            batch = _move_batch(original_batch, selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_batch(batch)
            loss = model.compute_masked_loss(
                logits,
                batch.tag_ids,
                batch.token_mask,
            )
            loss.backward()
            optimizer.step()

            token_count = int(batch.token_mask.sum().item())
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count

        if total_tokens == 0:
            raise ValueError("Training batches contain no real tokens")

        development_loss, development_accuracy, development_macro_f1 = (
            _evaluate_model(model, dev_batches, selected_device)
        )
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            training_loss=total_loss / total_tokens,
            development_loss=development_loss,
            development_accuracy=development_accuracy,
            development_macro_f1=development_macro_f1,
        )
        history.append(epoch_metrics)

        if development_macro_f1 > best_macro_f1:
            best_macro_f1 = development_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            checkpoint_note = " | best checkpoint"
        else:
            epochs_without_improvement += 1
            checkpoint_note = ""
        print(
            f"Epoch {epoch:02d}/{model.config.max_epochs} | "
            f"train loss: {epoch_metrics.training_loss:.4f} | "
            f"dev loss: {epoch_metrics.development_loss:.4f} | "
            f"dev accuracy: {epoch_metrics.development_accuracy:.4f} | "
            f"dev Macro F1: {epoch_metrics.development_macro_f1:.4f}"
            f"{checkpoint_note}",
            flush=True,
        )
        if (
            epochs_without_improvement
            >= model.config.early_stopping_patience
        ):
            print(
                f"Early stopping after {epoch} epochs "
                f"({model.config.early_stopping_patience} "
                "non-improving epochs).",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    model._is_fitted = True
    return history


def history_as_dicts(history: list[EpochMetrics]) -> list[dict]:
    """Convert training history to JSON-friendly dictionaries."""
    return [asdict(metrics) for metrics in history]
