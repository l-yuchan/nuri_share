# Copyright 2026 Nuri ML contributors
"""Tests for the linking encoders and the score-matrix contract."""

import numpy as np
import pytest
import torch

from indicator_linking_experiment import TextBank, TokenEncoder
from indicator_task import ranking_metrics


def make_bank() -> TextBank:
    """Build a four-text bank: two rows and two definitions."""
    token_ids = torch.tensor([[1, 2], [2, 3], [1, 3], [3, 4]], dtype=torch.int64)
    mask = torch.ones_like(token_ids, dtype=torch.float32)
    return TextBank(
        token_ids=token_ids,
        mask=mask,
        definition_index=torch.tensor([2, 3], dtype=torch.int64),
        row_index=torch.tensor([0, 1], dtype=torch.int64),
        vocabulary=("<pad>", "가", "나", "다", "라"),
        train_oov_rate=0.0,
    )


def test_trainable_encoder_does_not_mutate_the_caller_table() -> None:
    """Training must not write through the numpy table shared between conditions.

    ``nn.Embedding`` keeps the tensor it is given, so an unaliased clone is the
    only thing that stops one fitted fold from rewriting the next fold's
    initialization.
    """
    vectors = np.ones((5, 4), dtype=np.float32)
    vectors[0] = 0.0
    original = vectors.copy()
    encoder = TokenEncoder(make_bank(), vectors, trainable=True)
    with torch.no_grad():
        encoder.embedding.weight.add_(5.0)
    assert np.array_equal(vectors, original)
    assert encoder.embedding.weight.requires_grad


def test_frozen_encoder_keeps_weights_frozen() -> None:
    """A frozen condition trains only its projection."""
    vectors = np.ones((5, 4), dtype=np.float32)
    vectors[0] = 0.0
    encoder = TokenEncoder(make_bank(), vectors, trainable=False)
    assert not encoder.embedding.weight.requires_grad
    assert encoder.projection.weight.requires_grad


def test_ranking_metrics_rejects_a_one_dimensional_truth() -> None:
    """Truth must be one candidate index per score row."""
    with pytest.raises(ValueError, match="shapes disagree"):
        ranking_metrics(np.zeros((2, 3)), np.zeros((2, 3)))
