# Copyright 2026 Nuri ML contributors
"""Tests for the indicator bank, the linking metrics and leakage rejection."""

import csv
from pathlib import Path

import numpy as np
import pytest

from indicator_task import (
    DatasetError,
    indicator_groups,
    load_linking_data,
    paired_bootstrap,
    ranking_metrics,
    target_indices,
)
from nuri_data import DEFAULT_DATA_DIR

FIELDS = [
    "id",
    "source",
    "session",
    "group_id",
    "label",
    "content_category",
    "item_no",
    "item_text",
    "context",
    "level",
    "age",
    "text",
    "label_source",
    "use_default",
    "split",
]


def row(
    identifier: str, group: str, definition: str, text: str, level: int = 1
) -> dict[str, object]:
    """Build one KICCE-shaped row with an explicit indicator definition."""
    return {
        "id": identifier,
        "source": "kicce",
        "session": "s01",
        "group_id": group,
        "label": "의사소통",
        "content_category": "장면",
        "item_no": group.split("-")[1],
        "item_text": definition,
        "context": "",
        "level": str(level),
        "age": "3-5세",
        "text": text,
        "label_source": "kicce",
        "use_default": "yes",
        "split": "train",
    }


def write_split(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a CSV in the supplied dataset schema."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_bank_is_indicator_disjoint() -> None:
    """Training and validation indicators must not overlap."""
    data = load_linking_data(DEFAULT_DATA_DIR)
    train = {item.key for item in data.train_indicators}
    val = {item.key for item in data.val_indicators}
    assert len(train) == 48
    assert len(val) == 11
    assert train & val == set()
    assert len(data.all_indicators) == 59
    assert len(data.train_rows) == 572
    assert len(data.val_rows) == 132
    assert {row.level for row in data.train_rows} == {1, 2, 3, 4}


def test_targets_point_at_the_labeled_indicator() -> None:
    """Every target index must resolve to the row's own indicator key."""
    data = load_linking_data(DEFAULT_DATA_DIR)
    groups = indicator_groups(data)
    assert target_indices(data.val_rows, data).tolist() == [
        groups[row.indicator] for row in data.val_rows
    ]


def test_ranking_metrics_matches_hand_computed_ranks() -> None:
    """Recall@k and MRR follow the row-wise rank of the true candidate."""
    scores = np.array(
        [
            [0.9, 0.5, 0.1],
            [0.1, 0.9, 0.5],
            [0.1, 0.2, 0.9],
        ]
    )
    truth = np.array([0, 0, 2])
    metrics = ranking_metrics(scores, truth, ks=(1, 2, 3))
    assert metrics["recall@1"] == pytest.approx(2 / 3)
    assert metrics["recall@2"] == pytest.approx(2 / 3)
    assert metrics["recall@3"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3 + 1) / 3)
    assert metrics["n"] == 3


def test_paired_bootstrap_is_zero_for_identical_models() -> None:
    """A model compared with itself has a zero delta and a zero interval."""
    values = np.array([1.0, 0.5, 0.25, 1.0])
    groups = np.array([0, 0, 1, 2])
    interval = paired_bootstrap(values, values, groups, n_boot=200, seed=7)
    assert interval["delta"] == pytest.approx(0.0)
    assert interval["ci_low"] == pytest.approx(0.0)
    assert interval["ci_high"] == pytest.approx(0.0)


def test_paired_bootstrap_resamples_groups_not_rows() -> None:
    """Whole groups move together, so a one-group difference has a wide interval."""
    treatment = np.array([1.0, 1.0, 0.0, 0.0])
    control = np.array([0.0, 0.0, 0.0, 0.0])
    groups = np.array([0, 0, 1, 1])
    interval = paired_bootstrap(treatment, control, groups, n_boot=500, seed=3)
    assert interval["delta"] == pytest.approx(0.5)
    assert interval["ci_low"] == pytest.approx(0.0)
    assert interval["ci_high"] == pytest.approx(1.0)


def test_shared_definition_across_splits_is_rejected(tmp_path: Path) -> None:
    """The same definition in train and validation is leakage, not a new indicator."""
    write_split(
        tmp_path / "train.csv",
        [row("T1", "KI-1-01", "신체 각 부분의 이름을 안다.", "다리를 가리킨다.", 1)],
    )
    write_split(
        tmp_path / "val.csv",
        [row("V1", "KI-9-09", "신체 각 부분의 이름을 안다.", "팔을 가리킨다.", 1)],
    )
    with pytest.raises(DatasetError, match="definition overlap"):
        load_linking_data(tmp_path)


def test_repeated_text_across_splits_is_rejected(tmp_path: Path) -> None:
    """Identical normalized observations across splits are leakage."""
    write_split(
        tmp_path / "train.csv",
        [row("T1", "KI-1-01", "신체 각 부분의 이름을 안다.", "다리를 가리킨다.", 1)],
    )
    write_split(
        tmp_path / "val.csv",
        [row("V1", "KI-9-09", "노래를 즐겨 부른다.", "다리를  가리킨다.", 1)],
    )
    with pytest.raises(DatasetError, match="text overlap"):
        load_linking_data(tmp_path)


def test_inconsistent_definition_inside_one_indicator_is_rejected(tmp_path: Path) -> None:
    """One indicator cannot carry two different definitions."""
    write_split(
        tmp_path / "train.csv",
        [
            row("T1", "KI-1-01", "신체 각 부분의 이름을 안다.", "다리를 가리킨다.", 1),
            row("T2", "KI-1-01", "다른 정의다.", "팔을 가리킨다.", 2),
        ],
    )
    write_split(
        tmp_path / "val.csv",
        [row("V1", "KI-9-09", "노래를 즐겨 부른다.", "노래를 부른다.", 1)],
    )
    with pytest.raises(DatasetError, match="definitions"):
        load_linking_data(tmp_path)
