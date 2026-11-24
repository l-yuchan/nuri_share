# Copyright 2026 Nuri ML contributors
import numpy as np

from hierarchical_experiment import (
    LEVELS,
    LevelRun,
    _level_confusion,
    area_metrics,
    average_rank,
    level_metrics,
    spearman,
    top_k,
)


def test_average_rank_assigns_tied_positions() -> None:
    ranks = average_rank(np.array([4.0, 1.0, 4.0, 2.0]))
    np.testing.assert_allclose(ranks, [3.5, 1.0, 3.5, 2.0])


def test_spearman_detects_monotone_and_rejects_constant() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(truth, np.array([10.0, 20.0, 30.0, 45.0])) == 1.0
    assert spearman(truth, truth[::-1].copy()) == -1.0
    assert spearman(truth, np.full(4, 2.5)) == 0.0


def test_top_k_returns_classes_in_probability_order() -> None:
    proba = np.array([[0.1, 0.6, 0.2, 0.05, 0.05], [0.4, 0.3, 0.2, 0.1, 0.0]])
    ranked = top_k(proba, np.array(["a", "b", "c", "d", "e"]))
    assert ranked[0] == ["b", "c"]
    assert ranked[1] == ["a", "b"]


def test_area_metrics_counts_gold_in_top2() -> None:
    truth = ["a", "b", "c"]
    predicted = ["a", "a", "c"]
    ranked = [["a", "b"], ["a", "b"], ["b", "c"]]
    report = area_metrics(truth, predicted, ranked)

    assert report["accuracy"] == 2 / 3
    assert report["top2_accuracy"] == 1.0
    assert 0 < report["macro_f1"] <= 1


def test_level_metrics_reward_ordinal_agreement() -> None:
    truth = np.array([1, 2, 3, 4])
    perfect = level_metrics(truth, truth.copy())
    assert perfect == {"accuracy": 1.0, "mae": 0.0, "qwk": 1.0}

    off_by_one = level_metrics(truth, np.array([2, 1, 4, 3]))
    assert off_by_one["accuracy"] == 0.0
    assert off_by_one["mae"] == 1.0
    assert off_by_one["qwk"] > 0


def test_level_confusion_is_truth_by_prediction() -> None:
    run = LevelRun(
        truth=np.array([1, 1, 2, 3]),
        expected=np.array([1.2, 1.4, 2.1, 3.9]),
        argmax=np.array([1, 2, 2, 4]),
        proba=np.zeros((4, 4)),
        metrics=[],
    )
    confusion = _level_confusion(run)

    assert confusion["1"]["1"] == 1
    assert confusion["1"]["2"] == 1
    assert confusion["2"]["2"] == 1
    assert confusion["3"]["4"] == 1
    assert set(confusion) == {str(level) for level in LEVELS}
