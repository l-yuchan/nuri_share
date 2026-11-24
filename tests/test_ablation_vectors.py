# Copyright 2026 Nuri ML contributors
from pathlib import Path

import numpy as np
import pytest

from ablation_vectors import WordVectors, read_vectors
from nuri_data import DatasetError


def test_scaling_changes_only_global_magnitude() -> None:
    values = np.array([[3.0, 4.0], [0.0, 10.0], [-15.0, 0.0], [1.0, 2.0]])
    vectors = WordVectors(("a", "b", "c", "</s>"), values)
    transformed = vectors.scaled(0.1)
    np.testing.assert_allclose(transformed.values, values * 0.01)
    np.testing.assert_array_equal(vectors.values, values)


def test_shuffle_moves_every_word_preserves_vectors_and_fixes_eos() -> None:
    values = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [9.0, 0.0]])
    vectors = WordVectors(("a", "b", "c", "</s>"), values)
    transformed = vectors.shuffled(42)
    assert transformed.words == ("a", "b", "c", "</s>")
    assert np.all(transformed.values[:3, 0] != [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.sort(transformed.values[:, 0]), [1.0, 2.0, 3.0, 9.0])
    np.testing.assert_array_equal(transformed.values[3], [9.0, 0.0])


@pytest.mark.parametrize(
    "payload",
    [
        "3 2\na 1 0\nb 0 1\n",
        "3 2\na 1 0\nb 1\nc 1 1\n",
        "3 2\na 1 0\nb nan 1\nc 1 1\n",
        "3 2\na 1 0\nb 0 0\nc 1 1\n",
        "3 2\na 1 0\na 0 1\nc 1 1\n",
    ],
)
def test_vector_boundary_rejects_invalid_inputs(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "input.vec"
    _ = path.write_text(payload, encoding="utf-8")
    with pytest.raises(DatasetError):
        _ = read_vectors(path)


def test_fasttext_vector_format_preserves_word_associations(tmp_path: Path) -> None:
    path = tmp_path / "output.vec"
    vectors = WordVectors(("a", "b", "</s>"), np.array([[3.0, 4.0], [6.0, 8.0], [1.0, 2.0]]))
    vectors.write(path)
    loaded = read_vectors(path)
    assert loaded.words == ("a", "b", "</s>")
    np.testing.assert_array_equal(loaded.values, [[3.0, 4.0], [6.0, 8.0], [1.0, 2.0]])
