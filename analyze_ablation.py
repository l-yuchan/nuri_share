# Copyright 2026 Nuri ML contributors
"""Summarize paired runs with uncertainty over seeds and validation groups."""

import csv
import json
from pathlib import Path
from typing import Annotated, Final

import numpy as np
import typer
from numpy.typing import NDArray

from ablation_experiment import SEEDS
from nuri_data import DEFAULT_DATA_DIR, LABELS, DatasetError, read_splits

OUTPUT_DIR: Final = Path("results_ablation")
CONTRASTS: Final = (
    ("scale_aligned", "aligned_scaled", "aligned_raw"),
    ("scale_shuffled", "shuffled_scaled", "shuffled_raw"),
    ("alignment_scaled", "aligned_scaled", "shuffled_scaled"),
    ("alignment_raw", "aligned_raw", "shuffled_raw"),
    ("scaled_vs_scratch", "aligned_scaled", "scratch"),
)


def classification_scores(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute accuracy and fixed-class Macro-F1 for stacked confusion matrices."""
    diagonal = np.diagonal(counts, axis1=-2, axis2=-1)
    denominator = counts.sum(axis=-1) + counts.sum(axis=-2)
    f1 = np.divide(
        2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator != 0
    ).mean(axis=-1)
    accuracy = diagonal.sum(axis=-1) / counts.sum(axis=(-2, -1))
    return np.stack((accuracy, f1), axis=-1)


def grouped_bootstrap(
    counts: NDArray[np.float64], samples: int = 5000, seed: int = 20261117
) -> NDArray[np.float64]:
    """Resample paired seeds and whole groups, retaining every row in each group."""
    seeds, _, groups, _, _ = counts.shape
    rng = np.random.default_rng(seed)
    group_weights = rng.multinomial(groups, np.full(groups, 1 / groups), size=samples)
    seed_weights = rng.multinomial(seeds, np.full(seeds, 1 / seeds), size=samples)
    resampled = np.einsum("bg,scgij->bscij", group_weights, counts, optimize=True)
    scores = classification_scores(resampled)
    return np.einsum("bs,bscm->bcm", seed_weights / seeds, scores, optimize=True)


def main(
    *,
    output_dir: Annotated[Path, typer.Option()] = OUTPUT_DIR,
) -> None:
    """Analyze the completed fixed experiment; no configuration is selected here."""
    _, val = read_splits(DEFAULT_DATA_DIR)
    by_id = {row.identifier: row for row in val}
    with (output_dir / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    conditions = sorted({name for _, a, b in CONTRASTS for name in (a, b)})
    seeds = list(SEEDS)
    groups = sorted({row.group for row in val})
    expected = {
        (condition, seed, identifier)
        for condition in conditions
        for seed in seeds
        for identifier in by_id
    }
    counts = np.zeros((len(seeds), len(conditions), len(groups), len(LABELS), len(LABELS)))
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        key = (row["condition"], int(row["seed"]), row["id"])
        actual = by_id[row["id"]]
        if key in seen or row["group"] != actual.group or row["label"] != actual.label:
            reason = "Predictions duplicate rows or disagree with the supplied validation split"
            raise DatasetError(reason)
        seen.add(key)
        counts[
            seeds.index(int(row["seed"])),
            conditions.index(row["condition"]),
            groups.index(actual.group),
            LABELS.index(actual.label),
            LABELS.index(row["top1"]),
        ] += 1
    if seen != expected:
        reason = "Predictions are missing seed/condition/validation rows"
        raise DatasetError(reason)
    point = classification_scores(counts.sum(axis=2))
    bootstrap = grouped_bootstrap(counts)
    with (output_dir / "run_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        runs = list(csv.DictReader(handle))
    summary = []
    for i, condition in enumerate(conditions):
        summary.append(
            {
                "condition": condition,
                "runs": len(seeds),
                "accuracy_mean": float(point[:, i, 0].mean()),
                "accuracy_seed_sd": float(point[:, i, 0].std(ddof=1)),
                "macro_f1_mean": float(point[:, i, 1].mean()),
                "macro_f1_seed_sd": float(point[:, i, 1].std(ddof=1)),
                "train_accuracy_mean": float(
                    np.mean(
                        [
                            float(row["train_accuracy"])
                            for row in runs
                            if row["condition"] == condition
                        ]
                    )
                ),
            }
        )
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    contrasts = []
    for name, treatment, control in CONTRASTS:
        a, b = conditions.index(treatment), conditions.index(control)
        difference = point[:, a] - point[:, b]
        intervals = np.quantile(bootstrap[:, a] - bootstrap[:, b], [0.025, 0.975], axis=0)
        contrasts.append(
            {
                "name": name,
                "treatment": treatment,
                "control": control,
                "accuracy_delta_mean": float(difference[:, 0].mean()),
                "macro_f1_delta_mean": float(difference[:, 1].mean()),
                "accuracy_delta_by_seed": difference[:, 0].tolist(),
                "macro_f1_delta_by_seed": difference[:, 1].tolist(),
                "accuracy_delta_95_interval": intervals[:, 0].tolist(),
                "macro_f1_delta_95_interval": intervals[:, 1].tolist(),
            }
        )
    analysis = {
        "seeds": seeds,
        "conditions": conditions,
        "validation_groups": len(groups),
        "bootstrap": {
            "samples": 5000,
            "seed": 20261117,
            "method": "Paired crossed resampling of training seeds and whole validation groups",
            "estimand": "Mean per-seed pooled-row accuracy and fixed-five-label Macro-F1",
            "note": "Exploratory percentile intervals; no multiplicity adjustment; only 13 groups.",
        },
        "contrasts": contrasts,
    }
    _ = (output_dir / "contrasts.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    typer.echo(json.dumps(contrasts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    typer.run(main)
