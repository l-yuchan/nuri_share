# Copyright 2026 Nuri ML contributors
"""Transform fixed word vectors without changing their vocabulary or geometry."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from nuri_data import DatasetError

MIN_WORDS: Final = 3


@dataclass(frozen=True, slots=True)
class WordVectors:
    """Ordered vocabulary and its finite, nonzero word vectors."""

    words: tuple[str, ...]
    values: NDArray[np.float64]

    def write(self, path: Path) -> None:
        """Write the standard fastText text-vector format at double precision."""
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            _ = handle.write(f"{len(self.words)} {self.values.shape[1]}\n")
            for word, vector in zip(self.words, self.values, strict=True):
                _ = handle.write(word + " " + " ".join(f"{v:.17g}" for v in vector) + "\n")

    def scaled(self, target_median: float) -> "WordVectors":
        """Apply one positive scalar; preserve directions and relative lengths."""
        ordinary = np.array([word != "</s>" for word in self.words])
        median = float(np.median(np.linalg.norm(self.values[ordinary], axis=1)))
        return WordVectors(self.words, self.values * (target_median / median))

    def shuffled(self, seed: int) -> "WordVectors":
        """Use a random single cycle: move every ordinary word, fixing EOS."""
        ordinary = np.array([i for i, word in enumerate(self.words) if word != "</s>"])
        order = np.random.default_rng(seed).permutation(ordinary)
        donors = np.arange(len(self.words))
        donors[order] = np.roll(order, 1)
        return WordVectors(self.words, self.values[donors].copy())


def read_vectors(path: Path) -> WordVectors:
    """Reject malformed or degenerate external vector files at the boundary."""
    with path.open(encoding="utf-8") as handle:
        count, dimension = map(int, next(handle).split())
        lines = [line.split() for line in handle]
    words = tuple(row[0] for row in lines)
    if len(lines) != count or len(set(words)) != count or count < MIN_WORDS:
        reason = "Vector count mismatch, duplicate vocabulary, or too few words"
        raise DatasetError(reason)
    if dimension < 1 or any(len(row) != dimension + 1 for row in lines):
        reason = "Vector dimension mismatch"
        raise DatasetError(reason)
    values = np.array([row[1:] for row in lines], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0):
        reason = "Vectors must be finite and nonzero"
        raise DatasetError(reason)
    return WordVectors(words, values)
