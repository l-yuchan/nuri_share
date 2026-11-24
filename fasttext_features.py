# Copyright 2026 Nuri ML contributors
"""Build corpus-trained fastText representations independently of class labels."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import fasttext
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import normalize


@dataclass(frozen=True, slots=True)
class CorpusVectors:
    """Frozen sentence features and the train-vocabulary initialization file."""

    train: NDArray[np.float32]
    val: NDArray[np.float32]
    word_vectors: Path
    pretrain_seconds: float
    vectorize_seconds: float


def train_vectors(corpus: Path, texts: tuple[list[str], list[str]], threads: int) -> CorpusVectors:
    """Train CBOW on external text and export vectors for downstream classifiers."""
    models_dir = Path("artifacts/fasttext")
    models_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    embedding = fasttext.train_unsupervised(
        str(corpus),
        model="cbow",
        dim=100,
        epoch=5,
        minn=2,
        maxn=5,
        minCount=3,
        bucket=200000,
        thread=threads,
        verbose=0,
    )
    embedding.save_model(str(models_dir / "cbow.bin"))
    pretrain_seconds = perf_counter() - start
    vector_path = models_dir / "nuri_train.vec"
    vocabulary = sorted({word for text in texts[0] for word in text.split()} | {"</s>"})
    with vector_path.open("w", encoding="utf-8") as handle:
        _ = handle.write(f"{len(vocabulary)} 100\n")
        for word in vocabulary:
            vector = embedding.get_word_vector(word)
            _ = handle.write(word + " " + " ".join(f"{value:.8g}" for value in vector) + "\n")
    start = perf_counter()
    x_train = normalize(np.stack([embedding.get_sentence_vector(text) for text in texts[0]]))
    x_val = normalize(np.stack([embedding.get_sentence_vector(text) for text in texts[1]]))
    return CorpusVectors(x_train, x_val, vector_path, pretrain_seconds, perf_counter() - start)
