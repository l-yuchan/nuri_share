# Copyright 2026 Nuri ML contributors
import csv
from pathlib import Path

import pytest

from experiment import make_vectorizer
from nuri_data import DatasetError, decompose_jamo, normalize_syllables, read_examples, read_splits


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\uac01", "\u1100\u1161\u11a8"),
        ("\ud55c\uae00  ABC\t123!", "\u1112\u1161\u11ab\u1100\u1173\u11af abc 123!"),
        ("\u1100\u1161", "\u1100\u1161"),
        ("", ""),
    ],
)
def test_jamo_decomposition_preserves_non_hangul(text: str, expected: str) -> None:
    assert decompose_jamo(text) == expected


def test_syllable_normalization_composes_decomposed_input() -> None:
    assert normalize_syllables("\u1100\u1161\tA") == "\uac00 a"


def test_validation_terms_do_not_enter_tfidf_vocabulary() -> None:
    vectorizer = make_vectorizer(decompose_jamo, (2, 3))
    _ = vectorizer.fit_transform(["\uac00\ub098"])
    validation = vectorizer.transform(["XYZ"])
    assert vectorizer.inverse_transform(validation)[0].size == 0


def test_default_filter_excludes_unselected_rows(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "text", "label", "group_id", "source", "use_default"))
        writer.writerow(("a", "first", "\uc790\uc5f0\ud0d0\uad6c", "g1", "test", "yes"))
        writer.writerow(("b", "second", "\uc790\uc5f0\ud0d0\uad6c", "g2", "test", "no"))
    assert [row.identifier for row in read_examples(path, default_only=True)] == ["a"]


def test_split_rejects_group_leakage(tmp_path: Path) -> None:
    for name, text in (("train.csv", "first"), ("val.csv", "second")):
        with (tmp_path / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("id", "text", "label", "group_id", "source", "use_default"))
            writer.writerow((name, text, "\uc790\uc5f0\ud0d0\uad6c", "shared", "test", "yes"))
    with pytest.raises(DatasetError, match="group_id overlap"):
        _ = read_splits(tmp_path)
