# Copyright 2026 Nuri ML contributors
"""Build the observation-to-indicator linking task from the supplied splits.

Every KICCE row carries one indicator definition (``item_text``) and a
developmental level. That supports a single-label linking task: given an
observation, rank the candidate indicator definitions and return the one the
observation is evidence for.

The supplied split is indicator-disjoint (48 training definitions, 11
validation definitions, no overlap), so the validation rows measure linking to
definitions that never appeared in training. Only the definition text carries
over; no validation row or label is used to fit a representation.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from nuri_data import DatasetError, normalize_syllables

LABELS: Final = (
    "사회관계",
    "신체운동·건강",
    "예술경험",
    "의사소통",
    "자연탐구",
)
REQUIRED_COLUMNS: Final = frozenset(
    {"id", "text", "label", "group_id", "source", "item_text", "level"}
)
UNK: Final = "<unk>"
PAD: Final = "<pad>"


@dataclass(frozen=True, slots=True)
class Indicator:
    """One Nuri indicator that observations can be linked to."""

    key: str
    definition: str
    area: str
    split: str


@dataclass(frozen=True, slots=True)
class Observation:
    """One behavioral description and the indicator it was labeled with."""

    identifier: str
    text: str
    definition: str
    indicator: str
    area: str
    level: int | None
    source: str


@dataclass(frozen=True, slots=True)
class LinkingData:
    """The indicator bank restricted to KICCE rows, which carry definitions."""

    train_indicators: tuple[Indicator, ...]
    val_indicators: tuple[Indicator, ...]
    train_rows: tuple[Observation, ...]
    val_rows: tuple[Observation, ...]

    @property
    def all_indicators(self) -> tuple[Indicator, ...]:
        """Candidate set in a fixed order: training indicators, then validation."""
        return (*self.train_indicators, *self.val_indicators)

    @property
    def candidate_keys(self) -> tuple[str, ...]:
        """Candidate indicator keys in the same order as ``all_indicators``."""
        return tuple(item.key for item in self.all_indicators)

    @property
    def rows(self) -> tuple[Observation, ...]:
        """Every observation, training rows first."""
        return (*self.train_rows, *self.val_rows)


def read_observations(path: Path) -> tuple[Observation, ...]:
    """Read KICCE rows that carry both an indicator definition and a level."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
            reason = f"{path}: missing required columns"
            raise DatasetError(reason)
        rows: list[Observation] = []
        for row in reader:
            definition = (row["item_text"] or "").strip()
            if not definition:
                continue
            if row["label"] not in LABELS or not (row["group_id"] or "").strip():
                reason = f"{path}: invalid row {reader.line_num}"
                raise DatasetError(reason)
            level_text = (row["level"] or "").strip()
            rows.append(
                Observation(
                    identifier=row["id"],
                    text=normalize_syllables(row["text"]),
                    definition=definition,
                    indicator=row["group_id"],
                    area=row["label"],
                    level=int(level_text) if level_text else None,
                    source=row["source"],
                )
            )
        if not rows:
            reason = f"{path}: no KICCE rows"
            raise DatasetError(reason)
    return tuple(rows)


def build_bank(rows: tuple[Observation, ...], split: str) -> tuple[Indicator, ...]:
    """Collapse rows into one indicator per ``group_id`` and check consistency."""
    definitions: dict[str, set[str]] = defaultdict(set)
    areas: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        definitions[row.indicator].add(row.definition)
        areas[row.indicator].add(row.area)
    bank: list[Indicator] = []
    for key in sorted(definitions):
        if len(definitions[key]) != 1:
            reason = f"{split}: indicator {key} has {len(definitions[key])} definitions"
            raise DatasetError(reason)
        if len(areas[key]) != 1:
            reason = f"{split}: indicator {key} spans several areas"
            raise DatasetError(reason)
        bank.append(
            Indicator(
                key=key,
                definition=next(iter(definitions[key])),
                area=next(iter(areas[key])),
                split=split,
            )
        )
    return tuple(bank)


def load_linking_data(data_dir: Path) -> LinkingData:
    """Read both splits and reject indicator or text leakage across them."""
    train_rows = read_observations(data_dir / "train.csv")
    val_rows = read_observations(data_dir / "val.csv")
    train_keys = {row.indicator for row in train_rows}
    val_keys = {row.indicator for row in val_rows}
    if train_keys & val_keys:
        reason = "Train/validation indicator overlap"
        raise DatasetError(reason)
    train_defs = {row.definition for row in train_rows}
    val_defs = {row.definition for row in val_rows}
    if train_defs & val_defs:
        reason = "Train/validation definition overlap"
        raise DatasetError(reason)
    if {row.text for row in train_rows} & {row.text for row in val_rows}:
        reason = "Train/validation normalized text overlap"
        raise DatasetError(reason)
    return LinkingData(
        train_indicators=build_bank(train_rows, "train"),
        val_indicators=build_bank(val_rows, "val"),
        train_rows=train_rows,
        val_rows=val_rows,
    )


def indicator_groups(data: LinkingData) -> dict[str, int]:
    """Map indicator key to its candidate index."""
    return {key: index for index, key in enumerate(data.candidate_keys)}


def target_indices(rows: tuple[Observation, ...], data: LinkingData) -> np.ndarray:
    """Candidate index of the labeled indicator for every row."""
    groups = indicator_groups(data)
    return np.array([groups[row.indicator] for row in rows], dtype=np.int64)


def ranking_metrics(
    scores: np.ndarray, truth: np.ndarray, ks: tuple[int, ...] = (1, 3)
) -> dict[str, float]:
    """Recall@k and mean reciprocal rank for one score matrix.

    ``scores`` has one row per observation and one column per candidate; the
    rank of ``truth`` decides every metric. Ties are broken by candidate order,
    exactly as ``np.argsort`` does on the descending scores.
    """
    if truth.ndim != 1 or scores.ndim != truth.ndim + 1 or scores.shape[0] != truth.shape[0]:
        reason = "scores and truth shapes disagree"
        raise ValueError(reason)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.argmax(order == truth[:, None], axis=1)
    metrics = {f"recall@{k}": float(np.mean(ranks < k)) for k in ks}
    metrics["mrr"] = float(np.mean(1.0 / (ranks + 1.0)))
    metrics["n"] = float(len(truth))
    return metrics


def per_row_reciprocal_rank(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-row reciprocal rank, used for paired bootstrap differences."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.argmax(order == truth[:, None], axis=1)
    return 1.0 / (ranks + 1.0)


def paired_bootstrap(
    treatment: np.ndarray,
    control: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile interval for the mean paired difference, resampling groups.

    Whole indicator groups are resampled together; every candidate model is
    scored on the same replicate, so the interval describes the difference
    rather than two separate estimates.
    """
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    index_by_group = {key: np.flatnonzero(groups == key) for key in unique}
    deltas = np.empty(n_boot, dtype=np.float64)
    diff = treatment - control
    for replicate in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([index_by_group[key] for key in picked])
        deltas[replicate] = float(diff[index].mean())
    return {
        "delta": float(diff.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }
