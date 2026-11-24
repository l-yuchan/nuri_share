# Copyright 2026 Nuri ML contributors
"""Evaluate an area + developmental-level system with anti-confound checks.

The supplied KICCE rows carry an area label (5 classes) and a developmental
level (1-4, three vignettes per level within each indicator group). This
experiment asks three questions:

1. Is the level learnable from text alone, on unseen indicator groups?
2. Should the level be conditioned on the predicted area, or run in parallel?
3. Does the score secretly act as a class weight?

The system uses a group-aware calibrated SVM for area probabilities and a
multinomial logistic head for the level. Diagnostics test both directions of
leakage and the weighted-training form the score must avoid.

Run with ``uv run python hierarchical_experiment.py``.
"""

import csv
import hashlib
import json
import platform
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, cast

import numpy as np
import sklearn
import typer
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from experiment import make_vectorizer
from nuri_data import DEFAULT_DATA_DIR, LABELS, Example, normalize_syllables, read_splits

SEED: Final = 42
AREA_C: Final = 10.0
LEVEL_C: Final = 1.0
FOLDS: Final = 5
OUTPUT_DIR: Final = Path("results_hierarchical")
LEVELS: Final = (1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class Split:
    """Area rows (all sources) and the per-row level map for one split."""

    examples: tuple[Example, ...]
    levels: dict[str, int | None]


@dataclass(frozen=True, slots=True)
class LevelData:
    """KICCE-only matrices, rows and level targets used by the level head."""

    x_train: csr_matrix
    x_val: csr_matrix
    train_rows: tuple[Example, ...]
    val_rows: tuple[Example, ...]
    y_train: np.ndarray


@dataclass(frozen=True, slots=True)
class AreaRun:
    """Area predictions for the validation split."""

    truth: list[str]
    predicted: list[str]
    ranked: list[list[str]]
    proba: np.ndarray
    kicce_indices: list[int]
    metrics: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class LevelRun:
    """Level predictions for the KICCE validation rows."""

    truth: np.ndarray
    expected: np.ndarray
    argmax: np.ndarray
    proba: np.ndarray
    metrics: list[dict[str, object]]


def read_levels(data_dir: Path) -> dict[str, int | None]:
    """Read the KICCE developmental level for every row; casebook rows are None."""
    levels: dict[str, int | None] = {}
    for name in ("train.csv", "val.csv"):
        with (data_dir / name).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                levels[row["id"]] = int(row["level"]) if row["level"] else None
    return levels


def kicce_rows(split: Split) -> tuple[tuple[Example, ...], np.ndarray]:
    """Return KICCE examples and their levels; level rows are balanced 1-4."""
    rows = tuple(row for row in split.examples if split.levels[row.identifier] is not None)
    y = np.array([split.levels[row.identifier] for row in rows], dtype=int)
    return rows, y


def average_rank(values: np.ndarray) -> np.ndarray:
    """Rank values, assigning the average rank to ties."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rank correlation without a scipy dependency."""
    a = average_rank(left)
    b = average_rank(right)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def within_group_spearman(
    rows: tuple[Example, ...], truth: np.ndarray, score: np.ndarray
) -> list[float]:
    """Correlate true level with the predicted score inside each indicator group."""
    by_group: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_group.setdefault(row.group, []).append(index)
    return [spearman(truth[indices].astype(float), score[indices]) for indices in by_group.values()]


def top_k(proba: np.ndarray, classes: np.ndarray, k: int = 2) -> list[list[str]]:
    """Return the k highest-probability classes per row."""
    order = np.argsort(-proba, axis=1, kind="stable")
    return [[str(classes[index]) for index in row[:k]] for row in order]


def area_metrics(
    truth: list[str], predicted: list[str], ranked: list[list[str]]
) -> dict[str, float]:
    """Accuracy, Macro-F1 and top-2 for the area head."""
    return {
        "accuracy": float(np.mean([a == b for a, b in zip(truth, predicted, strict=True)])),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "top2_accuracy": float(
            np.mean([gold in row for gold, row in zip(truth, ranked, strict=True)])
        ),
    }


def level_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Exact accuracy, MAE and quadratic weighted kappa for the level head."""
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "mae": float(mean_absolute_error(truth, predicted)),
        "qwk": float(cohen_kappa_score(truth, predicted, weights="quadratic")),
    }


def fitted_area(
    x_train: csr_matrix, labels: list[str], groups: np.ndarray
) -> CalibratedClassifierCV:
    """Fit the calibrated SVM using group-aware calibration folds."""
    splits = list(GroupKFold(n_splits=FOLDS).split(x_train, labels, groups))
    model = CalibratedClassifierCV(
        LinearSVC(C=AREA_C, dual=True, max_iter=10000, random_state=SEED),
        cv=splits,
        method="sigmoid",
    )
    return model.fit(x_train, labels)


def oof_level_proba(
    x_train: csr_matrix, y_train: np.ndarray, groups: np.ndarray, x_val: csr_matrix
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold level probabilities and full-train validation probabilities."""
    classes = np.unique(y_train)
    oof = np.zeros((x_train.shape[0], len(classes)))
    for train_index, held_index in GroupKFold(n_splits=FOLDS).split(x_train, y_train, groups):
        fold = LogisticRegression(C=LEVEL_C, max_iter=5000, random_state=SEED).fit(
            x_train[train_index], y_train[train_index]
        )
        oof[held_index] = fold.predict_proba(x_train[held_index])
    final = LogisticRegression(C=LEVEL_C, max_iter=5000, random_state=SEED).fit(x_train, y_train)
    return oof, final.predict_proba(x_val), final.classes_


def oof_area_proba(
    x_train: csr_matrix, y_train: np.ndarray, groups: np.ndarray, x_val: csr_matrix
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-aware out-of-fold area scores (softmax) and a full-train model's validation scores."""
    classes = np.unique(y_train)
    oof = np.zeros((x_train.shape[0], len(classes)))
    for train_index, held_index in GroupKFold(n_splits=FOLDS).split(x_train, y_train, groups):
        fold = LinearSVC(C=AREA_C, dual=True, max_iter=10000, random_state=SEED).fit(
            x_train[train_index], y_train[train_index]
        )
        oof[held_index] = _softmax(fold.decision_function(x_train[held_index]))
    final = LinearSVC(C=AREA_C, dual=True, max_iter=10000, random_state=SEED).fit(x_train, y_train)
    return oof, _softmax(final.decision_function(x_val)), final.classes_


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax for decision scores."""
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _augment(x: csr_matrix, features: np.ndarray) -> csr_matrix:
    """Append dense features to a sparse text matrix."""
    return cast("csr_matrix", hstack([x, csr_matrix(np.asarray(features, dtype=float))]))


def run_area(
    x_train: csr_matrix,
    x_val: csr_matrix,
    train_examples: tuple[Example, ...],
    val_examples: tuple[Example, ...],
    levels: dict[str, int | None],
) -> AreaRun:
    """Fit the area system and the reference SVM; collect metrics."""
    labels = [row.label for row in train_examples]
    groups = np.array([row.group for row in train_examples])
    system = fitted_area(x_train, labels, groups)
    proba = system.predict_proba(x_val)
    predicted = [str(label) for label in system.classes_[np.argmax(proba, axis=1)]]
    ranked = top_k(proba, system.classes_)

    reference = LinearSVC(C=AREA_C, dual=True, max_iter=10000, random_state=SEED).fit(
        x_train, labels
    )
    scores = reference.decision_function(x_val)
    reference_pred = [str(label) for label in reference.classes_[np.argmax(scores, axis=1)]]
    reference_ranked = [
        [str(reference.classes_[index]) for index in np.argsort(-row, kind="stable")[:2]]
        for row in scores
    ]
    kicce_indices = [
        index for index, row in enumerate(val_examples) if levels[row.identifier] is not None
    ]
    truth = [row.label for row in val_examples]
    kicce_truth = [truth[index] for index in kicce_indices]
    kicce_pred = [predicted[index] for index in kicce_indices]
    kicce_ranked = [ranked[index] for index in kicce_indices]
    metrics: list[dict[str, object]] = [
        _metric_row(
            "area",
            "svm_decision_reference",
            len(truth),
            area_metrics(truth, reference_pred, reference_ranked),
        ),
        _metric_row(
            "area", "calibrated_svm_system", len(truth), area_metrics(truth, predicted, ranked)
        ),
        _metric_row(
            "area_kicce",
            "calibrated_svm_system",
            len(kicce_truth),
            area_metrics(kicce_truth, kicce_pred, kicce_ranked),
        ),
    ]
    return AreaRun(truth, predicted, ranked, proba, kicce_indices, metrics)


def run_level(
    x_train: csr_matrix, x_val: csr_matrix, truth_train: np.ndarray, truth_val: np.ndarray
) -> LevelRun:
    """Fit the multinomial level head and collect its metrics."""
    model = LogisticRegression(C=LEVEL_C, max_iter=5000, random_state=SEED).fit(
        x_train, truth_train
    )
    proba = model.predict_proba(x_val)
    expected = proba @ model.classes_
    argmax = model.classes_[np.argmax(proba, axis=1)]
    metrics = [
        _metric_row("level", "argmax", len(truth_val), level_metrics(truth_val, argmax)),
        _metric_row(
            "level",
            "expected_rounded",
            len(truth_val),
            level_metrics(truth_val, np.rint(expected).astype(int)),
        ),
    ]
    return LevelRun(truth_val, expected, argmax, proba, metrics)


def level_baselines(
    level_train: np.ndarray,
    level_val: np.ndarray,
    train_kicce: tuple[Example, ...],
    val_kicce: tuple[Example, ...],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Global-mean, oracle-area-mean and length-only baselines for the level task."""
    label_train = np.array([row.label for row in train_kicce])
    level_mean_by_area = {
        label: float(level_train[label_train == label].mean())
        for label in LABELS
        if label in label_train
    }
    global_mean = np.full(len(level_val), int(np.rint(level_train.mean())))
    oracle = np.rint(np.array([level_mean_by_area[row.label] for row in val_kicce])).astype(int)
    lengths_train = np.array([[len(row.text)] for row in train_kicce], dtype=float)
    lengths_val = np.array([[len(row.text)] for row in val_kicce], dtype=float)
    length_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(
        lengths_train, level_train
    )
    metrics = [
        _metric_row(
            "level", "global_mean_baseline", len(level_val), level_metrics(level_val, global_mean)
        ),
        _metric_row(
            "level", "oracle_area_mean_baseline", len(level_val), level_metrics(level_val, oracle)
        ),
        _metric_row(
            "level",
            "length_only_baseline",
            len(level_val),
            level_metrics(level_val, length_model.predict(lengths_val)),
        ),
    ]
    return metrics, level_mean_by_area


def confound_diagnostics(area: AreaRun, level: LevelRun, data: LevelData) -> dict[str, object]:
    """Test whether the score is an area proxy or a training weight in disguise."""
    groups = np.array([row.group for row in data.train_rows])
    area_labels = [row.label for row in data.train_rows]
    kicce_area = [area.predicted[index] for index in area.kicce_indices]
    kicce_truth = [area.truth[index] for index in area.kicce_indices]
    from_level = LogisticRegression(max_iter=1000, random_state=SEED).fit(
        data.y_train.reshape(-1, 1), area_labels
    )
    oof_level, val_level_proba, level_classes = oof_level_proba(
        data.x_train, data.y_train, groups, data.x_val
    )
    oof_area, val_area_proba, area_classes = oof_area_proba(
        data.x_train, np.array(area_labels), groups, data.x_val
    )
    level_probs_val = val_level_proba[:, [int(np.where(level_classes == k)[0][0]) for k in LEVELS]]
    kicce_only = LinearSVC(C=AREA_C, dual=True, max_iter=10000, random_state=SEED).fit(
        data.x_train, area_labels
    )
    stacked = LinearSVC(C=AREA_C, dual=True, max_iter=10000, random_state=SEED).fit(
        _augment(data.x_train, oof_level), area_labels
    )
    stacked_expected = LinearSVC(C=AREA_C, dual=True, max_iter=10000, random_state=SEED).fit(
        _augment(data.x_train, (oof_level @ level_classes).reshape(-1, 1)), area_labels
    )
    weighted = LinearSVC(C=AREA_C, dual=True, max_iter=10000, random_state=SEED).fit(
        data.x_train, area_labels, sample_weight=data.y_train.astype(float)
    )
    area_probs_val = val_area_proba[
        :, [int(np.where(area_classes == label)[0][0]) for label in LABELS]
    ]
    with_area = LogisticRegression(C=LEVEL_C, max_iter=5000, random_state=SEED).fit(
        _augment(data.x_train, oof_area), data.y_train
    )
    return {
        "area_from_level_only_accuracy": float(
            from_level.score(level.truth.reshape(-1, 1), [row.label for row in data.val_rows])
        ),
        "area_from_level_only_chance": round(1 / len(LABELS), 6),
        "area_accuracy_kicce": float(
            np.mean([a == b for a, b in zip(kicce_truth, kicce_area, strict=True)])
        ),
        "area_kicce_only_accuracy": float(
            kicce_only.score(data.x_val, [row.label for row in data.val_rows])
        ),
        "area_with_level_probs_accuracy": float(
            stacked.score(
                _augment(data.x_val, level_probs_val), [row.label for row in data.val_rows]
            )
        ),
        "area_with_level_expected_accuracy": float(
            stacked_expected.score(
                _augment(data.x_val, (val_level_proba @ level_classes).reshape(-1, 1)),
                [row.label for row in data.val_rows],
            )
        ),
        "area_weighted_by_level_accuracy": float(
            weighted.score(data.x_val, [row.label for row in data.val_rows])
        ),
        "level_mae_with_area_probs": float(
            mean_absolute_error(
                level.truth,
                with_area.predict_proba(_augment(data.x_val, area_probs_val)) @ with_area.classes_,
            )
        ),
        "level_mae_text_only": float(mean_absolute_error(level.truth, level.expected)),
        "joint_exact_accuracy": float(
            np.mean(
                [
                    predicted == gold_area and int(pred_level) == int(gold_level)
                    for predicted, gold_area, pred_level, gold_level in zip(
                        kicce_area, kicce_truth, level.argmax, level.truth, strict=True
                    )
                ]
            )
        ),
    }


def _metric_row(task: str, setting: str, n: int, values: dict[str, float]) -> dict[str, object]:
    """Build one metrics.csv row."""
    return {"task": task, "setting": setting, "n": n, **{k: round(v, 6) for k, v in values.items()}}


def diagnostics_summary(level: LevelRun, data: LevelData) -> dict[str, object]:
    """Group-level ordinal consistency and distribution facts."""
    rhos = within_group_spearman(data.val_rows, level.truth, level.expected)
    label_train = np.array([row.label for row in data.train_rows])
    return {
        "groups": len(rhos),
        "within_group_spearman_mean": float(np.mean(rhos)),
        "within_group_spearman_min": float(np.min(rhos)),
        "within_group_spearman_max": float(np.max(rhos)),
        "level_distribution_train": dict(Counter(data.y_train.tolist())),
        "level_distribution_val": dict(Counter(level.truth.tolist())),
        "level_mean_by_area": {
            label: float(data.y_train[label_train == label].mean())
            for label in LABELS
            if label in label_train
        },
        "level_confusion": _level_confusion(level),
    }


def _level_confusion(level: LevelRun) -> dict[str, dict[str, int]]:
    """Truth-by-prediction counts for the four levels."""
    counts = {str(truth): dict.fromkeys((str(k) for k in LEVELS), 0) for truth in LEVELS}
    for truth, predicted in zip(level.truth, level.argmax, strict=True):
        counts[str(int(truth))][str(int(predicted))] += 1
    return counts


def main(
    *,
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATA_DIR,
    output_dir: Annotated[Path, typer.Option()] = OUTPUT_DIR,
) -> None:
    """Run the area+level evaluation and write metrics, predictions and diagnostics."""
    train_examples, val_examples = read_splits(data_dir)
    levels = read_levels(data_dir)
    train_kicce, level_train = kicce_rows(Split(train_examples, levels))
    val_kicce, level_val = kicce_rows(Split(val_examples, levels))
    vectorizer = make_vectorizer(normalize_syllables, (2, 5))
    x_area_train = cast(
        "csr_matrix", vectorizer.fit_transform([row.text for row in train_examples])
    )
    x_area_val = cast("csr_matrix", vectorizer.transform([row.text for row in val_examples]))
    x_level_train = cast("csr_matrix", vectorizer.transform([row.text for row in train_kicce]))
    x_level_val = cast("csr_matrix", vectorizer.transform([row.text for row in val_kicce]))

    area = run_area(x_area_train, x_area_val, train_examples, val_examples, levels)
    data = LevelData(x_level_train, x_level_val, train_kicce, val_kicce, level_train)
    level = run_level(data.x_train, data.x_val, data.y_train, level_val)
    baseline_metrics, _ = level_baselines(data.y_train, level_val, data.train_rows, data.val_rows)
    diagnostics = diagnostics_summary(level, data)
    diagnostics.update(confound_diagnostics(area, level, data))
    metrics = [*area.metrics, *level.metrics, *baseline_metrics]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(output_dir / "metrics.csv", metrics)
    _write_predictions(output_dir / "predictions.csv", val_examples, levels, area, level)
    metadata = {
        "seed": SEED,
        "area_c": AREA_C,
        "level_c": LEVEL_C,
        "folds": FOLDS,
        "features": "syllable_2-5_char_tfidf",
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "data_hashes": {
            name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in ("train.csv", "val.csv")
        },
        "counts": {
            "train_rows": len(train_examples),
            "val_rows": len(val_examples),
            "train_kicce": len(train_kicce),
            "val_kicce": len(val_kicce),
        },
        "diagnostics": diagnostics,
    }
    _ = (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(json.dumps(metrics, ensure_ascii=False, indent=2))
    typer.echo(json.dumps(diagnostics, ensure_ascii=False, indent=2))


def _write_metrics(path: Path, metrics: list[dict[str, object]]) -> None:
    """Write one row per setting and metric set."""
    fields = ["task", "setting", "n", "accuracy", "macro_f1", "top2_accuracy", "mae", "qwk"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)


def _write_predictions(
    path: Path,
    examples: tuple[Example, ...],
    levels: dict[str, int | None],
    area: AreaRun,
    level: LevelRun,
) -> None:
    """Write per-row system output; level columns stay empty for casebook rows."""
    level_index = {
        identifier: index
        for index, identifier in enumerate(
            row.identifier for row in examples if levels[row.identifier] is not None
        )
    }
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "id",
                "source",
                "group_id",
                "text",
                "gold_area",
                "area_top1",
                "area_top2",
                "area_confidence",
                "gold_level",
                "level_argmax",
                "level_expected",
                "level_confidence",
            )
        )
        for index, row in enumerate(examples):
            values: list[object] = [
                row.identifier,
                row.source,
                row.group,
                row.text,
                row.label,
                area.predicted[index],
                area.ranked[index][1],
                round(float(np.max(area.proba[index])), 4),
            ]
            position = level_index.get(row.identifier)
            if position is None:
                values.extend(("", "", "", ""))
            else:
                values.extend(
                    (
                        levels[row.identifier],
                        int(level.argmax[position]),
                        round(float(level.expected[position]), 3),
                        round(float(np.max(level.proba[position])), 4),
                    )
                )
            writer.writerow(values)


if __name__ == "__main__":
    typer.run(main)
