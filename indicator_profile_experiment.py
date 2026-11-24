# Copyright 2026 Nuri ML contributors
"""Evaluate the end-to-end child profile: predicted links, levels and rankings.

The linking experiment exports the top-5 predicted indicators for every
validation row. This script joins those predictions with a level head and
simulates children from the 11 held-out indicators:

- a child is assessed on K indicators, one true level 1-4 per indicator;
- the description holds m vignettes per indicator, sampled at that level;
- every vignette is linked with the exported predictions and scored by the
  level head;
- scores are aggregated per predicted indicator and ranked within the child.

Unlike the earlier profile experiment, the assessed indicators are not given to
the model: a missed indicator stays missing and a linked indicator outside the
child's true set counts as a spurious output. That makes the number a full
pipeline result rather than an aggregation ceiling.

Run with ``uv run python indicator_profile_experiment.py``.
"""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import numpy as np
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

from experiment import make_vectorizer
from indicator_task import LinkingData, load_linking_data
from nuri_data import DEFAULT_DATA_DIR, normalize_syllables

SEED: Final = 20261121
TOP_K: Final = 2
CHILDREN: Final = 500
ITEMS_PER_CHILD: Final = (4, 8, 11)
OBSERVATIONS: Final = (1, 2, 3)
LEVELS: Final = (1, 2, 3, 4)
DEFAULT_OUTPUT_DIR: Final = Path("results_indicator_profile")
DEFAULT_PREDICTIONS: Final = Path("results_indicator_linking/val_row_predictions.csv")


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """Candidate set and sampling state for the random-link control."""

    candidates: tuple[str, ...]
    rng: np.random.Generator


@dataclass(frozen=True, slots=True)
class Child:
    """One synthetic child: K indicators, true levels and sampled vignette ids."""

    indicators: tuple[str, ...]
    levels: tuple[int, ...]
    vignettes: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class RowPrediction:
    """Exported linking prediction and level distribution for one row."""

    indicator: str
    level: int
    linked: tuple[str, ...]
    level_probability: tuple[float, ...]


def read_predictions(path: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Read the top-5 linked indicators per validation row."""
    predictions: dict[str, tuple[str, tuple[str, ...]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            predictions[row["id"]] = (
                row["true_indicator"],
                (row["top1"], row["top2"], row["top3"], row["top5"]),
            )
    return predictions


def fit_level_head(
    data: LinkingData,
) -> tuple[LogisticRegression, Callable[[list[str]], np.ndarray]]:
    """Fit the observation-to-level classifier on training rows only."""
    vectorizer = make_vectorizer(normalize_syllables, (2, 5))
    matrix = vectorizer.fit_transform([row.text for row in data.train_rows])
    labels = np.array([row.level for row in data.train_rows], dtype=np.int64)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    model.fit(matrix, labels)

    def predict(texts: list[str]) -> np.ndarray:
        return model.predict_proba(vectorizer.transform(texts))

    return model, predict


def level_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Accuracy, MAE, QWK and one-level accuracy for an ordinal output."""
    return {
        "accuracy": float(np.mean(predicted == truth)),
        "mae": float(np.mean(np.abs(predicted.astype(np.float64) - truth))),
        "qwk": float(cohen_kappa_score(truth, predicted, weights="quadratic")),
        "within1": float(np.mean(np.abs(predicted - truth) <= 1)),
        "n": float(len(truth)),
    }


def build_children(
    pools: dict[str, dict[int, list[str]]],
    *,
    children: int,
    items_per_child: int,
    observations: int,
    seed: int,
) -> list[Child]:
    """Sample children whose items, levels and vignettes are all held out."""
    rng = np.random.default_rng(seed)
    indicators = sorted(pools)
    output: list[Child] = []
    for _ in range(children):
        chosen = [
            indicators[int(index)]
            for index in rng.choice(len(indicators), items_per_child, replace=False)
        ]
        levels = [int(rng.integers(1, len(LEVELS) + 1)) for _ in chosen]
        vignettes: list[tuple[str, ...]] = []
        for indicator, level in zip(chosen, levels, strict=True):
            pool = pools[indicator][level]
            picked = rng.choice(len(pool), min(observations, len(pool)), replace=False)
            vignettes.append(tuple(pool[int(index)] for index in picked))
        output.append(Child(tuple(chosen), tuple(levels), tuple(vignettes)))
    return output


def band(levels: tuple[int, ...], *, strong: bool) -> set[int]:
    """Two-item band: the top two levels for strong, the bottom two for weak."""
    ordered = sorted(levels, reverse=strong)
    if len(ordered) <= TOP_K:
        return set(ordered)
    return set(ordered[:TOP_K])


def linked_candidates(
    prediction: RowPrediction, link_mode: str, context: "ScoringContext"
) -> list[str]:
    """Return the indicators this one vignette contributes to."""
    if link_mode == "oracle":
        return [prediction.indicator]
    if link_mode == "random":
        return [context.candidates[int(context.rng.integers(0, len(context.candidates)))]]
    if link_mode == "predicted1":
        return list(prediction.linked[:1])
    return list(prediction.linked[:3])


def score_child(
    child: Child,
    rows: dict[str, RowPrediction],
    context: "ScoringContext",
    *,
    link_mode: str,
    level_mode: str,
) -> dict[str, float]:
    """Aggregate one child and return the ranking and coverage outcomes."""
    truth = dict(zip(child.indicators, child.levels, strict=True))
    aggregated: dict[str, list[float]] = {}
    hits = 0
    vignettes = 0
    for position, indicator in enumerate(child.indicators):
        true_level = child.levels[position]
        for identifier in child.vignettes[position]:
            prediction = rows[identifier]
            linked = linked_candidates(prediction, link_mode, context)
            if level_mode == "oracle":
                value = float(true_level)
            else:
                value = float(
                    np.dot(prediction.level_probability, np.array(LEVELS, dtype=np.float64))
                )
            vignettes += 1
            hits += int(linked[0] == indicator)
            for key in linked:
                aggregated.setdefault(key, []).append(value)
    scores = {key: float(np.mean(values)) for key, values in aggregated.items()}
    # Ties are broken at random: with oracle levels every score is an exact tie,
    # and a fixed key order would make the weak ranking unreachable.
    jitter = {key: float(context.rng.random()) for key in scores}
    strong_order = sorted(scores, key=lambda key: (-scores[key], jitter[key]))
    weak_order = sorted(scores, key=lambda key: (scores[key], jitter[key]))
    strong = band(child.levels, strong=True)
    weak = band(child.levels, strong=False)
    strong_hits = sum(1 for key in strong_order[:TOP_K] if key in truth and truth[key] in strong)
    weak_hits = sum(1 for key in weak_order[:TOP_K] if key in truth and truth[key] in weak)
    missing = sum(1 for key in child.indicators if key not in aggregated)
    spurious = sum(1 for key in aggregated if key not in truth)
    return {
        "strong_band_hits": strong_hits / TOP_K,
        "weak_band_hits": weak_hits / TOP_K,
        "link_top1": hits / max(vignettes, 1),
        "missing_rate": missing / len(child.indicators),
        "spurious_rate": spurious / max(len(aggregated), 1),
    }


def main(
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_DIR,
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATA_DIR,
    predictions_path: Annotated[Path, typer.Option()] = DEFAULT_PREDICTIONS,
    children: Annotated[int, typer.Option(min=1)] = CHILDREN,
) -> None:
    """Run the end-to-end profile evaluation and write metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_linking_data(data_dir)
    selected = read_predictions(predictions_path)
    level_model, predict_level = fit_level_head(data)
    val_texts = [row.text for row in data.val_rows]
    probabilities = predict_level(val_texts)
    classes = [int(value) for value in level_model.classes_]
    full = np.zeros((len(val_texts), len(LEVELS)), dtype=np.float64)
    for column, value in enumerate(classes):
        full[:, value - 1] = probabilities[:, column]
    rows: dict[str, RowPrediction] = {}
    for position, row in enumerate(data.val_rows):
        true_indicator, linked = selected[row.identifier]
        rows[row.identifier] = RowPrediction(
            indicator=true_indicator,
            level=int(row.level) if row.level is not None else 0,
            linked=linked,
            level_probability=tuple(float(value) for value in full[position]),
        )

    level_report = level_metrics(
        np.array([classes[int(value)] for value in probabilities.argmax(axis=1)]),
        np.array([row.level for row in data.val_rows], dtype=np.int64),
    )

    pools: dict[str, dict[int, list[str]]] = {}
    for row in data.val_rows:
        if row.level is None:
            continue
        pools.setdefault(row.indicator, {}).setdefault(row.level, []).append(row.identifier)
    context = ScoringContext(tuple(data.candidate_keys), np.random.default_rng(SEED))

    strategies = (
        ("oracle_link_oracle_level", "oracle", "oracle"),
        ("oracle_link", "oracle", "predicted"),
        ("predicted_link_oracle_level", "predicted1", "oracle"),
        ("predicted_link_top1", "predicted1", "predicted"),
        ("predicted_link_top3", "predicted3", "predicted"),
        ("random_link", "random", "predicted"),
    )
    records: list[dict[str, object]] = []
    for items in ITEMS_PER_CHILD:
        for observations in OBSERVATIONS:
            sampled = build_children(
                pools,
                children=children,
                items_per_child=items,
                observations=observations,
                seed=SEED + items * 10 + observations,
            )
            for name, link_mode, level_mode in strategies:
                outcomes = [
                    score_child(
                        child,
                        rows,
                        context,
                        link_mode=link_mode,
                        level_mode=level_mode,
                    )
                    for child in sampled
                ]
                records.append(
                    {
                        "items": items,
                        "observations": observations,
                        "strategy": name,
                        "children": len(outcomes),
                        "strong_band_recall": float(
                            np.mean([outcome["strong_band_hits"] for outcome in outcomes])
                        ),
                        "weak_band_recall": float(
                            np.mean([outcome["weak_band_hits"] for outcome in outcomes])
                        ),
                        "link_top1": float(np.mean([outcome["link_top1"] for outcome in outcomes])),
                        "missing_rate": float(
                            np.mean([outcome["missing_rate"] for outcome in outcomes])
                        ),
                        "spurious_rate": float(
                            np.mean([outcome["spurious_rate"] for outcome in outcomes])
                        ),
                    }
                )

    with (output_dir / "profile_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "level_metrics.json").write_text(
        json.dumps(
            {
                "model": "tfidf_syllable_2-5 + multinomial logistic C=1",
                "validation_rows": int(level_report["n"]),
                **level_report,
                "training_indicators": len(data.train_indicators),
                "validation_indicators": len(data.val_indicators),
                "linking_source": str(predictions_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo((output_dir / "level_metrics.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    typer.run(main)
