# Copyright 2026 Nuri ML contributors
"""Fit identical fastText classifiers under externally supplied initializations."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

import fasttext
import numpy as np
from fasttext.FastText import _FastText
from pydantic import TypeAdapter

from fasttext_experiment import report_predictions
from nuri_data import LABELS, Example
from prepare_corpus import clean_text

WORD_LIST: Final = TypeAdapter(list[str])


@dataclass(frozen=True, slots=True)
class Condition:
    """One initialization treatment paired with its training seed."""

    name: str
    vectors: Path | None
    seed: int


@dataclass(frozen=True, slots=True)
class Run:
    """One paired seed/condition result."""

    condition: str
    seed: int
    accuracy: float
    macro_f1: float
    top2_accuracy: float
    train_accuracy: float
    seconds: float
    dictionary_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluatedRun:
    """Metrics and diagnostics from the same trained classifier."""

    metrics: Run
    ranked: list[list[str]]
    dictionary: tuple[str, ...]
    report: str


def train_model(training: Path, condition: Condition, epochs: int = 100) -> _FastText:
    """Fix all learning settings; epoch zero inspects native initialization."""
    return fasttext.train_supervised(
        str(training),
        dim=100,
        epoch=epochs,
        lr=0.5,
        wordNgrams=2,
        minn=2,
        maxn=5,
        bucket=200000,
        thread=1,
        seed=condition.seed,
        verbose=0,
        pretrainedVectors="" if condition.vectors is None else str(condition.vectors),
    )


def native_initialization(
    training: Path, seed: int, words: tuple[str, ...]
) -> tuple[tuple[str, ...], float]:
    """Measure native raw word rows, using only the training vocabulary."""
    native = train_model(training, Condition("scratch", None, seed), epochs=0)
    target = float(
        np.median(
            [
                np.linalg.norm(native.get_input_vector(native.get_word_id(word)))
                for word in words
                if word != "</s>"
            ]
        )
    )
    return tuple(WORD_LIST.validate_python(native.get_words(), strict=True)), target


def decode_predictions(labels: Sequence[Sequence[str]]) -> list[list[str]]:
    """Decode fastText numeric labels in their predicted score order."""
    return [[LABELS[int(label.removeprefix("__label__"))] for label in row] for row in labels]


def fit_condition(
    training: Path,
    condition: Condition,
    splits: tuple[tuple[Example, ...], tuple[Example, ...]],
) -> EvaluatedRun:
    """Evaluate train fit and held-out predictions without changing hyperparameters."""
    train, val = splits
    start = perf_counter()
    model = train_model(training, condition)
    dictionary = tuple(WORD_LIST.validate_python(model.get_words(), strict=True))
    train_ranked = decode_predictions(model.predict([clean_text(r.text) for r in train], k=1)[0])
    ranked = decode_predictions(model.predict([clean_text(r.text) for r in val], k=5)[0])
    seconds = perf_counter() - start
    accuracy, macro_f1, top2, report = report_predictions(val, ranked)
    metrics = Run(
        condition.name,
        condition.seed,
        accuracy,
        macro_f1,
        top2,
        sum(row.label == top[0] for row, top in zip(train, train_ranked, strict=True)) / len(train),
        seconds,
        hashlib.sha256("\n".join(dictionary).encode()).hexdigest(),
    )
    return EvaluatedRun(metrics, ranked, dictionary, report)
