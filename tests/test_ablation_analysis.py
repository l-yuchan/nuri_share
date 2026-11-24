# Copyright 2026 Nuri ML contributors
import numpy as np

from analyze_ablation import classification_scores, grouped_bootstrap


def test_macro_f1_keeps_absent_classes_in_the_denominator() -> None:
    counts = np.array([[9.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(classification_scores(counts), [0.9, 6 / 19])


def test_bootstrap_resamples_whole_groups_and_pools_their_rows() -> None:
    counts = np.zeros((1, 1, 2, 2, 2))
    counts[0, 0, 0, 0, 0] = 9
    counts[0, 0, 1, 1, 0] = 1
    sampled = grouped_bootstrap(counts, samples=100, seed=42)
    np.testing.assert_allclose(np.unique(sampled[:, 0, 0]), [0.0, 0.9, 1.0])
