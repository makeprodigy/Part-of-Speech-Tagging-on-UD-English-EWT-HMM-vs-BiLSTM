"""Run the Phase 2 HMM, word-only BiLSTM, and character BiLSTM comparison."""

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from itertools import product
from pathlib import Path

# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import torch
import sklearn
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pyrefly: ignore[missing-import]
from src.bilstm_model import (
    BiLSTMConfig,
    CharacterBiLSTMTagger,
    WordBiLSTMTagger,
)
from src.dataset_loader import extract_words_tags, load_conllu
from src.dataset_registry import DATASET_SPLITS, validate_dataset_files
# pyrefly: ignore[missing-import]
from src.error_analysis import collect_prediction_records, summarize_errors
from src.evaluation import evaluate_neural_batches, evaluate_predictions
from src.hmm_model import HMMPOSTagger
from src.neural_data import NeuralBatch, collate_encoded_sentences
from src.preprocessing import (
    build_neural_vocabulary,
    encode_sentences,
    extract_original_vocabulary,
)
from src.training import history_as_dicts, set_random_seed


def parse_arguments() -> argparse.Namespace:
    """Parse runner options without changing the default full-data protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Limit each split for a smoke run; default uses every sentence.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Maximum development-training epochs for each neural model.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Stop after this many non-improving development epochs.",
    )
    parser.add_argument(
        "--learning-rates",
        type=str,
        default="0.0005,0.001",
        help="Comma-separated learning-rate candidates selected on development.",
    )
    parser.add_argument(
        "--dropouts",
        type=str,
        default="0.2,0.3",
        help="Comma-separated dropout candidates selected on development.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="128,256",
        help="Comma-separated hidden-size candidates selected on development.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of sentences per neural batch.",
    )
    parser.add_argument(
        "--use-crf",
        action="store_true",
        help="Use a linear-chain CRF loss and Viterbi decoding.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/phase2_bilstm_results.json"),
    )
    parser.add_argument(
        "--predictions-json",
        type=Path,
        default=Path("experiments/phase2_bilstm_predictions.json"),
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("reports/phase2_results.csv"),
    )
    return parser.parse_args()


def parse_candidates(raw_value: str, converter, argument_name: str) -> list:
    """Parse a non-empty comma-separated hyperparameter candidate list."""
    try:
        candidates = [converter(value.strip()) for value in raw_value.split(",")]
    except ValueError as error:
        raise ValueError(
            f"{argument_name} must contain comma-separated numeric values"
        ) from error
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError(f"{argument_name} must contain positive values")
    return candidates


def limit_sentences(sentences, maximum: int | None):
    """Apply an optional deterministic prefix limit for smoke experiments."""
    if maximum is None:
        return sentences
    if maximum < 2:
        raise ValueError("--max-sentences must be at least two")
    return sentences[:maximum]


def create_batches(sentences, vocabulary, batch_size: int) -> list[NeuralBatch]:
    """Encode sentences and collate deterministic contiguous batches."""
    encoded_sentences = encode_sentences(sentences, vocabulary)
    return [
        collate_encoded_sentences(
            encoded_sentences[start:start + batch_size],
            vocabulary,
        )
        for start in range(0, len(encoded_sentences), batch_size)
    ]


def json_safe_metrics(metrics: dict) -> dict:
    """Convert NumPy metric values to JSON-compatible values."""
    safe_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            safe_metrics[key] = value.tolist()
        elif isinstance(value, dict):
            safe_metrics[key] = {
                nested_key: float(nested_value)
                for nested_key, nested_value in value.items()
            }
        elif isinstance(value, (np.floating, np.integer)):
            safe_metrics[key] = value.item()
        else:
            safe_metrics[key] = value
    return safe_metrics


def train_and_evaluate_neural_model(
    model,
    model_name: str,
    train_batches: list[NeuralBatch],
    dev_batches: list[NeuralBatch],
    test_batches: list[NeuralBatch],
    vocabulary,
    config: BiLSTMConfig,
    output_prefix: Path | None = None,
    history=None,
    training_seconds: float | None = None,
    training_word_counts: Counter | None = None,
) -> tuple[dict, list[dict]]:
    """Train one neural model and return comparable test metrics and records."""
    if history is None:
        training_start = time.perf_counter()
        history = model.fit(train_batches, dev_batches)
        training_seconds = time.perf_counter() - training_start
    elif training_seconds is None:
        raise ValueError(
            "training_seconds is required when using a fitted model"
        )
    best_epoch = max(
        history,
        key=lambda metrics: metrics.development_macro_f1,
    ).epoch

    inference_start = time.perf_counter()
    metrics, predictions = evaluate_neural_batches(
        model,
        test_batches,
        vocabulary.id_to_tag,
        vocabulary.original_training_words,
    )
    inference_seconds = time.perf_counter() - inference_start
    records = collect_prediction_records(
        model,
        test_batches,
        vocabulary.id_to_tag,
        vocabulary.original_training_words,
        model_name,
        training_word_counts=training_word_counts,
    )
    safe_metrics = json_safe_metrics(metrics)
    safe_metrics.update({
        "model_name": model_name,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "best_development_epoch": best_epoch,
        "configuration": asdict(config),
        "development_history": history_as_dicts(history),
        "error_summary": summarize_errors(records),
    })
    if output_prefix is not None:
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "configuration": asdict(config),
                "word_to_id": dict(vocabulary.word_to_id),
                "char_to_id": dict(vocabulary.char_to_id),
                "tag_to_id": dict(vocabulary.tag_to_id),
                "metrics": safe_metrics,
            },
            output_prefix.with_suffix(".pt"),
        )
    return safe_metrics, records


def train_candidate(
    model,
    train_batches: list[NeuralBatch],
    dev_batches: list[NeuralBatch],
) -> tuple[dict, list]:
    """Fit one candidate and return development selection data."""
    training_start = time.perf_counter()
    history = model.fit(train_batches, dev_batches)
    training_seconds = time.perf_counter() - training_start
    best_epoch_metrics = max(
        history,
        key=lambda metrics: metrics.development_macro_f1,
    )
    return {
        "configuration": asdict(model.config),
        "best_epoch": best_epoch_metrics.epoch,
        "development_accuracy": best_epoch_metrics.development_accuracy,
        "development_macro_f1": best_epoch_metrics.development_macro_f1,
        "training_seconds": training_seconds,
        "epochs_completed": len(history),
    }, history


def evaluate_hmm(
    train_sentences,
    dev_sentences,
    test_sentences,
) -> dict:
    """Tune HMM smoothing on development data and evaluate test data."""
    train_words, _ = extract_words_tags(train_sentences)
    dev_words, dev_tags = extract_words_tags(dev_sentences)
    test_words, test_tags = extract_words_tags(test_sentences)
    training_vocabulary = extract_original_vocabulary(train_sentences)
    best_model = None
    best_dev_macro_f1 = float("-inf")
    best_alpha = None

    for smoothing in (0.01, 0.1, 1.0):
        candidate_model = HMMPOSTagger(
            smoothing=smoothing,
            minimum_word_frequency=1,
        ).fit(train_sentences)
        candidate_metrics = evaluate_predictions(
            dev_tags,
            candidate_model.predict_batch(dev_words),
            dev_words,
            training_vocabulary,
            labels=sorted(candidate_model.vocabulary.tags),
        )
        if candidate_metrics["macro_f1"] > best_dev_macro_f1:
            best_model = candidate_model
            best_dev_macro_f1 = candidate_metrics["macro_f1"]
            best_alpha = smoothing

    test_predictions = best_model.predict_batch(test_words)
    metrics = evaluate_predictions(
        test_tags,
        test_predictions,
        test_words,
        training_vocabulary,
        labels=sorted(best_model.vocabulary.tags),
    )
    return {
        "model_name": "hmm",
        "smoothing": best_alpha,
        "_model": best_model,
        **json_safe_metrics(metrics),
    }


def run_experiment(arguments: argparse.Namespace) -> dict:
    """Run the controlled Phase 2 comparison and save existing-path artifacts."""
    max_epochs = getattr(arguments, "max_epochs", 20)
    early_stopping_patience = getattr(
        arguments,
        "early_stopping_patience",
        4,
    )
    learning_rates_argument = getattr(
        arguments,
        "learning_rates",
        "0.0005,0.001",
    )
    dropouts_argument = getattr(arguments, "dropouts", "0.2,0.3")
    hidden_sizes_argument = getattr(arguments, "hidden_sizes", "128,256")
    validate_dataset_files()
    set_random_seed(42)
    split_sentences = {
        split_name: limit_sentences(
            load_conllu(DATASET_SPLITS[split_name]),
            arguments.max_sentences,
        )
        for split_name in ("train", "dev", "test")
    }
    train_sentences = split_sentences["train"]
    dev_sentences = split_sentences["dev"]
    test_sentences = split_sentences["test"]
    vocabulary = build_neural_vocabulary(train_sentences)
    training_word_counts = Counter(
        word for sentence in train_sentences for word, _ in sentence
    )
    train_batches = create_batches(
        train_sentences,
        vocabulary,
        arguments.batch_size,
    )
    dev_batches = create_batches(
        dev_sentences,
        vocabulary,
        arguments.batch_size,
    )
    test_batches = create_batches(
        test_sentences,
        vocabulary,
        arguments.batch_size,
    )
    learning_rates = parse_candidates(
        learning_rates_argument,
        float,
        "--learning-rates",
    )
    dropouts = parse_candidates(dropouts_argument, float, "--dropouts")
    hidden_sizes = parse_candidates(
        hidden_sizes_argument,
        int,
        "--hidden-sizes",
    )
    if early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be positive")
    candidate_hyperparameters = list(product(
        learning_rates,
        dropouts,
        hidden_sizes,
    ))
    hmm_result = evaluate_hmm(train_sentences, dev_sentences, test_sentences)
    hmm_model_instance = hmm_result.pop("_model", None)
    results = {
        "protocol": {
            "max_sentences": arguments.max_sentences,
            "batch_size": arguments.batch_size,
            "use_crf": arguments.use_crf,
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "optimizer": "Adam",
            "character_encoder_type": "BiLSTM",
            "dropout_placement": (
                "Input word/combined representation and between sentence "
                "BiLSTM layers when num_layers > 1; disabled during evaluation."
            ),
            "candidate_learning_rates": learning_rates,
            "candidate_dropouts": dropouts,
            "candidate_hidden_sizes": hidden_sizes,
            "candidate_count_per_model": len(candidate_hyperparameters),
            "device": "cpu",
            "dataset_revision": "English Web Treebank official repository splits",
            "dataset_files": {
                split_name: str(DATASET_SPLITS[split_name])
                for split_name in ("train", "dev", "test")
            },
            "training_vocabulary_size": len(
                vocabulary.original_training_words
            ),
            "split_sentence_counts": {
                split_name: len(sentences)
                for split_name, sentences in split_sentences.items()
            },
            "split_token_counts": {
                split_name: sum(len(sentence) for sentence in sentences)
                for split_name, sentences in split_sentences.items()
            },
            "target_tag_count": len(vocabulary.id_to_tag) - 1,
            "checkpoint_selection_metric": "development_macro_f1",
            "test_evaluation_policy": (
                "Test data is evaluated only after neural settings are fixed."
            ),
            "model_assumptions": {
                "word_bilstm": (
                    "Trainable word embeddings, bidirectional LSTM, "
                    "token-level classifier, masked cross-entropy."
                ),
                "character_bilstm": (
                    "Trainable word and character embeddings, character "
                    "BiLSTM, sentence BiLSTM, masked cross-entropy."
                ),
                "hmm": (
                    "First-order HMM with add-alpha smoothing and "
                    "log-space Viterbi decoding."
                ),
            },
        },
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "split_token_counts": {
                split_name: sum(len(sentence) for sentence in sentences)
                for split_name, sentences in split_sentences.items()
            },
        },
        "models": [
            hmm_result,
        ],
        "predictions": {},
    }
    prediction_records = {}
    write_artifacts = getattr(arguments, "write_artifacts", True)
    training_progress = tqdm(
        total=len(candidate_hyperparameters) * 2,
        desc="Neural training runs",
        unit="run",
    )
    for model_name, model_type in (
        ("word_bilstm", WordBiLSTMTagger),
        ("character_bilstm", CharacterBiLSTMTagger),
    ):
        selected_model = None
        selected_config = None
        selected_history = None
        selected_training_seconds = None
        best_development_macro_f1 = float("-inf")
        tuning_results = []
        for learning_rate, dropout, hidden_size in candidate_hyperparameters:
            config = BiLSTMConfig(
                batch_size=arguments.batch_size,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                learning_rate=learning_rate,
                dropout=dropout,
                hidden_size=hidden_size,
                random_seed=42,
                use_crf=arguments.use_crf,
            )
            set_random_seed(config.random_seed)
            candidate_model = model_type(vocabulary, config)
            candidate_result, candidate_history = train_candidate(
                candidate_model,
                train_batches,
                dev_batches,
            )
            training_progress.update(1)
            tuning_results.append(candidate_result)
            if (
                candidate_result["development_macro_f1"]
                > best_development_macro_f1
            ):
                selected_model = candidate_model
                selected_config = config
                selected_history = candidate_history
                selected_training_seconds = candidate_result["training_seconds"]
                best_development_macro_f1 = (
                    candidate_result["development_macro_f1"]
                )

        model_metrics, model_records = train_and_evaluate_neural_model(
            selected_model,
            model_name,
            train_batches,
            dev_batches,
            test_batches,
            vocabulary,
            selected_config,
            (
                arguments.output_json.with_name(model_name)
                if write_artifacts
                else None
            ),
            history=selected_history,
            training_seconds=selected_training_seconds,
            training_word_counts=training_word_counts,
        )
        model_metrics["tuning_candidates"] = tuning_results
        results["models"].append(model_metrics)
        prediction_records[model_name] = model_records
        results.setdefault("_trained_models", {})[model_name] = selected_model
        results.setdefault("_best_configs", {})[model_name] = selected_config
    results["_vocabulary"] = vocabulary
    if hmm_model_instance is not None:
        results["_trained_models"]["hmm"] = hmm_model_instance

    training_progress.close()
    if write_artifacts:
        serializable_results = {k: v for k, v in results.items() if not k.startswith("_")}
        arguments.output_json.write_text(
            json.dumps(serializable_results, indent=2),
            encoding="utf-8",
        )
        arguments.predictions_json.write_text(
            json.dumps(prediction_records, indent=2),
            encoding="utf-8",
        )
        with arguments.report_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as report_file:
            report_writer = csv.DictWriter(
                report_file,
                fieldnames=[
                    "model_name",
                    "token_accuracy",
                    "macro_f1",
                    "unknown_word_accuracy",
                    "unknown_word_count",
                    "parameter_count",
                    "training_seconds",
                    "inference_seconds",
                    "best_development_epoch",
                ],
            )
            report_writer.writeheader()
            for model_result in results["models"]:
                report_writer.writerow({
                    field: model_result.get(field)
                    for field in report_writer.fieldnames
                })
    return results


def main() -> None:
    """Run the Phase 2 comparison from the command line."""
    arguments = parse_arguments()
    results = run_experiment(arguments)
    for model_result in results["models"]:
        print(
            f"{model_result['model_name']}: "
            f"accuracy={model_result['token_accuracy']:.4f}, "
            f"macro_f1={model_result['macro_f1']:.4f}, "
            f"unknown_accuracy={model_result['unknown_word_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
