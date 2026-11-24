# Copyright 2026 Nuri ML contributors
import json

import pytest

from fasttext_experiment import report_predictions
from nuri_data import LABELS, Example


def test_ranking_metrics_include_all_labels_and_distinguish_top_two() -> None:
    examples = tuple(
        Example(str(i), "text", label, str(i), "source") for i, label in enumerate(LABELS)
    )
    ranked = [
        [LABELS[0], LABELS[1]],
        [LABELS[0], LABELS[1]],
        [LABELS[0], LABELS[2]],
        [LABELS[0], LABELS[1]],
        [LABELS[0], LABELS[1]],
    ]
    accuracy, macro_f1, top2, report = report_predictions(examples, ranked)
    assert (accuracy, macro_f1, top2) == pytest.approx((0.2, 1 / 15, 0.6))
    assert json.loads(report)["confusion_matrix"] == [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]
