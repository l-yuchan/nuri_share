# Copyright 2026 Nuri ML contributors
"""Simulate child profiles and evaluate per-item level and within-child rankings.

The final system takes a description of one child and should report, for each Nuri
item covered by the description, a level 1-4 and the strongest/weakest items. This
experiment builds synthetic children from held-out KICCE groups:

- a child is assessed on K indicators, one true level 1-4 per indicator;
- the description contains m vignettes per indicator, sampled from that indicator's
  validation rows at the true level;
- the level head (trained on the training groups) scores each vignette, the scores
  are aggregated per indicator, and the items are ranked within the child.

No child is compared with another child. All metrics are per-item or within-child.

Run with ``uv run python child_profile_experiment.py``.
"""

import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, cast

import numpy as np
import sklearn
import typer
from sklearn.linear_model import LogisticRegression

from experiment import make_vectorizer
from hierarchical_experiment import (
    Split,
    kicce_rows,
    level_metrics,
    read_levels,
    spearman,
)
from nuri_data import DEFAULT_DATA_DIR, Example, normalize_syllables, read_splits

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

SEED: Final = 20261121
LEVEL_C: Final = 1.0
OUTPUT_DIR: Final = Path("results_child_profile")
LEVELS: Final = (1, 2, 3, 4)
TOP_K: Final = 2
CHILDREN: Final = 500
ITEMS_PER_CHILD: Final = (4, 8, 11)
OBSERVATIONS: Final = (1, 2, 3)
METHODS: Final = ("aggregate", "single")
FULL_ITEMS: Final = max(ITEMS_PER_CHILD)
EXAMPLE_OBSERVATIONS: Final = 2


@dataclass(frozen=True, slots=True)
class Profile:
    """One synthetic child: items, true levels and the observed vignette texts."""

    groups: tuple[str, ...]
    truth: tuple[int, ...]
    observations: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ItemScore:
    """One item's aggregated score for one child."""

    group: str
    truth: int
    score: float
    label: int


def group_pools(
    examples: tuple[Example, ...], levels: dict[str, int | None]
) -> dict[str, dict[int, list[str]]]:
    """Index validation vignettes by indicator group and true level."""
    pools: dict[str, dict[int, list[str]]] = {}
    for row in examples:
        level = levels[row.identifier]
        if level is None:
            continue
        pools.setdefault(row.group, {}).setdefault(level, []).append(row.text)
    return pools


def build_profiles(
    pools: dict[str, dict[int, list[str]]],
    *,
    children: int,
    items_per_child: int,
    observations: int,
    seed: int,
) -> list[Profile]:
    """Sample children with independent item levels and fixed-level observations."""
    rng = np.random.default_rng(seed)
    groups = sorted(pools)
    profiles: list[Profile] = []
    for _ in range(children):
        chosen = [
            groups[int(index)] for index in rng.choice(len(groups), items_per_child, replace=False)
        ]
        truth = [int(rng.integers(1, len(LEVELS) + 1)) for _ in chosen]
        observed: dict[str, tuple[str, ...]] = {}
        for group, level in zip(chosen, truth, strict=True):
            pool = pools[group][level]
            picked = rng.choice(len(pool), min(observations, len(pool)), replace=False)
            observed[group] = tuple(pool[int(index)] for index in picked)
        profiles.append(Profile(tuple(chosen), tuple(truth), observed))
    return profiles


def score_profiles(
    model: LogisticRegression,
    x_by_text: dict[str, np.ndarray],
    profiles: list[Profile],
    method: str,
) -> list[list[ItemScore]]:
    """Aggregate per-vignette distributions into one score per indicator.

    ``aggregate`` averages the observation distributions; ``single`` uses only the
    first observation, which isolates the value of accumulating evidence.
    """
    scored: list[list[ItemScore]] = []
    for profile in profiles:
        items: list[ItemScore] = []
        for group, truth in zip(profile.groups, profile.truth, strict=True):
            rows = np.array([x_by_text[text] for text in profile.observations[group]])
            if method == "single":
                rows = rows[:1]
            mean = rows.mean(axis=0)
            score = float(mean @ model.classes_)
            label = int(model.classes_[int(mean.argmax())])
            items.append(ItemScore(group, truth, score, label))
        scored.append(items)
    return scored


def ndcg_at_k(scores: list[float], truth: list[int], *, weak: bool, k: int = TOP_K) -> float:
    """Graded-relevance NDCG over the top k, using 2^rel - 1."""
    relevance = [float((len(LEVELS) + 1 - level) if weak else level) for level in truth]
    order = np.argsort(np.asarray(scores) if weak else -np.asarray(scores), axis=0, kind="stable")
    ideal = np.argsort(-np.asarray(relevance), axis=0, kind="stable")[:k]
    dcg = sum(
        (2 ** relevance[int(index)] - 1) / math.log2(position + 2)
        for position, index in enumerate(order[:k])
    )
    idcg = sum(
        (2 ** relevance[int(index)] - 1) / math.log2(position + 2)
        for position, index in enumerate(ideal)
    )
    return dcg / idcg if idcg else 0.0


def band_recall(scores: list[float], truth: list[int], *, weak: bool, k: int = TOP_K) -> float:
    """Share of the predicted top k whose truth is inside the true top-k band."""
    order = np.argsort(np.asarray(scores) if weak else -np.asarray(scores), kind="stable")
    threshold = sorted(truth, reverse=not weak)[k - 1]
    hits = [
        (truth[int(index)] <= threshold) if weak else (truth[int(index)] >= threshold)
        for index in order[:k]
    ]
    return sum(hits) / k


def evaluate(scored: list[list[ItemScore]]) -> dict[str, float]:
    """Compute ranking and per-item metrics over every child."""
    truth_all: list[int] = []
    label_all: list[int] = []
    for items in scored:
        truth_all.extend(item.truth for item in items)
        label_all.extend(item.label for item in items)
    per_item = level_metrics(np.asarray(truth_all), np.asarray(label_all))
    within = float(np.mean([abs(a - b) <= 1 for a, b in zip(truth_all, label_all, strict=True)]))
    return {
        **_ranking_metrics(scored),
        "item_accuracy": per_item["accuracy"],
        "item_mae": per_item["mae"],
        "item_qwk": per_item["qwk"],
        "item_within1": within,
    }


def _ranking_metrics(scored: list[list[ItemScore]]) -> dict[str, float]:
    """Within-child top-2 strong/weak metrics, independent of per-item labels."""
    strong_ndcg, weak_ndcg = [], []
    strong_band, weak_band = [], []
    rho = []
    for items in scored:
        scores = [item.score for item in items]
        truth = [item.truth for item in items]
        strong_ndcg.append(ndcg_at_k(scores, truth, weak=False))
        weak_ndcg.append(ndcg_at_k(scores, truth, weak=True))
        strong_band.append(band_recall(scores, truth, weak=False))
        weak_band.append(band_recall(scores, truth, weak=True))
        rho.append(spearman(np.asarray(truth, dtype=float), np.asarray(scores)))
    return {
        "ndcg_strong": float(np.mean(strong_ndcg)),
        "ndcg_weak": float(np.mean(weak_ndcg)),
        "band_strong": float(np.mean(strong_band)),
        "band_weak": float(np.mean(weak_band)),
        "spearman": float(np.mean(rho)),
    }


def random_baseline(profiles: list[Profile], seed: int, repeats: int = 5) -> dict[str, float]:
    """Random rankings calibrate the ranking metrics; per-item metrics are undefined."""
    rng = np.random.default_rng(seed)
    metrics = []
    for _ in range(repeats):
        scored = [
            [
                ItemScore(group, truth, float(rng.random()), truth)
                for group, truth in zip(p.groups, p.truth, strict=True)
            ]
            for p in profiles
        ]
        metrics.append(_ranking_metrics(scored))
    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}


def main(
    *,
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATA_DIR,
    output_dir: Annotated[Path, typer.Option()] = OUTPUT_DIR,
    children: Annotated[int, typer.Option()] = CHILDREN,
) -> None:
    """Run the child-profile grid and write metrics, examples and metadata."""
    train_examples, val_examples = read_splits(data_dir)
    levels = read_levels(data_dir)
    train_kicce, level_train = kicce_rows(Split(train_examples, levels))
    val_kicce, _ = kicce_rows(Split(val_examples, levels))
    pools = group_pools(val_kicce, levels)

    vectorizer = make_vectorizer(normalize_syllables, (2, 5))
    x_train = cast("csr_matrix", vectorizer.fit_transform([row.text for row in train_kicce]))
    model = LogisticRegression(C=LEVEL_C, max_iter=5000, random_state=SEED).fit(
        x_train, level_train
    )

    rows: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    for items_per_child in ITEMS_PER_CHILD:
        for observations in OBSERVATIONS:
            profiles = build_profiles(
                pools,
                children=children,
                items_per_child=items_per_child,
                observations=observations,
                seed=SEED + items_per_child * 10 + observations,
            )
            texts = sorted(
                {
                    text
                    for profile in profiles
                    for group in profile.observations
                    for text in profile.observations[group]
                }
            )
            proba = model.predict_proba(vectorizer.transform(texts))
            x_by_text = dict(zip(texts, proba, strict=True))
            for method in METHODS:
                scored = score_profiles(model, x_by_text, profiles, method)
                metrics = evaluate(scored)
                rows.append(
                    {
                        "items_per_child": items_per_child,
                        "observations": observations,
                        "method": method,
                        "children": children,
                        **{key: round(value, 6) for key, value in metrics.items()},
                    }
                )
                if (
                    items_per_child == FULL_ITEMS
                    and observations == EXAMPLE_OBSERVATIONS
                    and method == "aggregate"
                ):
                    examples.append(_example(profiles[0], scored[0]))
            baseline = random_baseline(profiles, seed=SEED + 7)
            rows.append(
                {
                    "items_per_child": items_per_child,
                    "observations": observations,
                    "method": "random",
                    "children": children,
                    **{key: round(value, 6) for key, value in baseline.items()},
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(output_dir / "metrics.csv", rows)
    _ = (output_dir / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "seed": SEED,
        "children_per_cell": children,
        "items_per_child": list(ITEMS_PER_CHILD),
        "observations": list(OBSERVATIONS),
        "methods": list(METHODS),
        "level_model": {"type": "multinomial_logistic", "c": LEVEL_C, "features": "syllable_2-5"},
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "data_hashes": {
            name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in ("train.csv", "val.csv")
        },
        "notes": [
            (
                "Each child covers K held-out indicators; one true level 1-4 per indicator, "
                "independent across children."
            ),
            "All m vignettes of an indicator share the true level (consistent child); m<=3.",
            "No cross-child comparison is used anywhere.",
            (
                "Items are known from the assessment checklist; free-text item linking is "
                "out of scope."
            ),
        ],
    }
    _ = (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))


def _example(profile: Profile, items: list[ItemScore]) -> dict[str, object]:
    """Render one child profile with its ranked items for inspection."""
    order = sorted(items, key=lambda item: -item.score)
    return {
        "items": [
            {
                "item": item.group,
                "true_level": item.truth,
                "predicted_level": item.label,
                "score": round(item.score, 3),
                "observations": list(profile.observations[item.group]),
            }
            for item in order
        ],
        "strongest_2": [item.group for item in order[:2]],
        "weakest_2": [item.group for item in order[-2:]],
    }


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the grid as one row per cell and method."""
    fields = [
        "items_per_child",
        "observations",
        "method",
        "children",
        "ndcg_strong",
        "ndcg_weak",
        "band_strong",
        "band_weak",
        "spearman",
        "item_accuracy",
        "item_mae",
        "item_qwk",
        "item_within1",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    typer.run(main)
