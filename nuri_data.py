# Copyright 2026 Nuri ML contributors
"""Read the supplied splits and normalize Korean text without extra tokenizers."""

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

LABELS: Final = (
    "\uc0ac\ud68c\uad00\uacc4",
    "\uc2e0\uccb4\uc6b4\ub3d9\u00b7\uac74\uac15",
    "\uc608\uc220\uacbd\ud5d8",
    "\uc758\uc0ac\uc18c\ud1b5",
    "\uc790\uc5f0\ud0d0\uad6c",
)
DEFAULT_DATA_DIR: Final = Path(__file__).parent / "data"


@dataclass(frozen=True, slots=True)
class Example:
    """One labeled behavioral description; metadata never becomes a feature."""

    identifier: str
    text: str
    label: str
    group: str
    source: str


@dataclass(frozen=True, slots=True)
class DatasetError(ValueError):
    """Invalid data at the CSV or train/validation boundary."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the data-boundary failure reason."""
        return self.reason


def normalize_syllables(text: str) -> str:
    """Compose Hangul, lowercase Latin letters and collapse whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def decompose_jamo(text: str) -> str:
    """Use Unicode NFD: retain distinct initial, medial and final jamo."""
    return unicodedata.normalize("NFD", normalize_syllables(text))


def read_examples(path: Path, *, default_only: bool = False) -> tuple[Example, ...]:
    """Read text and labels, optionally applying the provided selection flag."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "text", "label", "group_id", "source", "use_default"}
        if not required.issubset(reader.fieldnames or []):
            reason = f"{path}: missing required columns"
            raise DatasetError(reason)
        examples: list[Example] = []
        for row in reader:
            if default_only and row["use_default"] != "yes":
                continue
            if (
                row["label"] not in LABELS
                or not normalize_syllables(row["text"])
                or not row["group_id"]
            ):
                reason = f"{path}: invalid row {reader.line_num}"
                raise DatasetError(reason)
            examples.append(
                Example(row["id"], row["text"], row["label"], row["group_id"], row["source"])
            )
    if not examples:
        reason = f"{path}: empty dataset"
        raise DatasetError(reason)
    return tuple(examples)


def read_splits(
    data_dir: Path, *, default_only: bool = False
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    """Keep the supplied split and reject group or normalized-text leakage."""
    train = read_examples(data_dir / "train.csv", default_only=default_only)
    val = read_examples(data_dir / "val.csv", default_only=default_only)
    if {row.group for row in train} & {row.group for row in val}:
        reason = "Train/validation group_id overlap"
        raise DatasetError(reason)
    if {normalize_syllables(row.text) for row in train} & {
        normalize_syllables(row.text) for row in val
    }:
        reason = "Train/validation normalized text overlap"
        raise DatasetError(reason)
    if {row.label for row in train} != set(LABELS) or {row.label for row in val} != set(LABELS):
        reason = "Both splits must contain all five labels"
        raise DatasetError(reason)
    return train, val
