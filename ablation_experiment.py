# Copyright 2026 Nuri ML contributors
"""Separate initialization scale from word-vector alignment in fixed fastText runs."""

import csv
import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Final

import numpy as np
import typer

from ablation_model import Condition, Run, fit_condition, native_initialization
from ablation_vectors import read_vectors
from nuri_data import DEFAULT_DATA_DIR, LABELS, DatasetError, read_splits
from prepare_corpus import clean_text

SEEDS: Final[tuple[int, ...]] = (13, 42, 73, 101, 137)
VECTOR_PATH: Final = Path("artifacts/fasttext/nuri_train.vec")
OUTPUT_DIR: Final = Path("results_ablation")


def main(
    *,
    output_dir: Annotated[Path, typer.Option()] = OUTPUT_DIR,
) -> None:
    """Run five conditions at five fixed seeds, without refitting corpus embeddings."""
    train, val = read_splits(DEFAULT_DATA_DIR)
    vectors = read_vectors(VECTOR_PATH)
    train_text = [clean_text(row.text) for row in train]
    expected = {word for text in train_text for word in text.split()} | {"</s>"}
    if set(vectors.words) != expected:
        reason = "Pretrained vector vocabulary must equal Nuri training vocabulary plus EOS"
        raise DatasetError(reason)
    artifacts = Path("artifacts/ablation")
    artifacts.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    training = artifacts / "train.txt"
    _ = training.write_text(
        "".join(
            f"__label__{LABELS.index(row.label)} {text}\n"
            for row, text in zip(train, train_text, strict=True)
        ),
        encoding="utf-8",
    )
    runs: list[Run] = []
    reports = {}
    initialization_reports = []
    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        predictions = csv.writer(handle)
        predictions.writerow(
            ("condition", "seed", "id", "group", "source", "label", "top1", "top2")
        )
        for seed in SEEDS:
            native_dictionary, target = native_initialization(training, seed, vectors.words)
            shuffled = vectors.shuffled(seed)
            treatments = (
                ("aligned_raw", vectors),
                ("aligned_scaled", vectors.scaled(target)),
                ("shuffled_raw", shuffled),
                ("shuffled_scaled", shuffled.scaled(target)),
            )
            conditions: list[tuple[str, Path | None]] = [("scratch", None)]
            for name, table in treatments:
                path = artifacts / f"{name}_seed{seed}.vec"
                table.write(path)
                conditions.append((name, path))
            initialization_reports.append(
                {
                    "seed": seed,
                    "native_median_word_norm": target,
                    "raw_median_word_norm": float(
                        np.median(
                            np.linalg.norm(
                                vectors.values[np.array(vectors.words) != "</s>"], axis=1
                            )
                        )
                    ),
                    "vector_hashes": {
                        name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in conditions
                        if path is not None
                    },
                }
            )
            for name, path in conditions:
                outcome = fit_condition(training, Condition(name, path, seed), (train, val))
                dictionary = outcome.dictionary
                if dictionary != native_dictionary:
                    reason = "Condition changes native word IDs; comparison is confounded"
                    raise DatasetError(reason)
                runs.append(outcome.metrics)
                reports[f"{name}_seed{seed}"] = json.loads(outcome.report)
                for row, top in zip(val, outcome.ranked, strict=True):
                    predictions.writerow(
                        (
                            name,
                            seed,
                            row.identifier,
                            row.group,
                            row.source,
                            row.label,
                            top[0],
                            top[1],
                        )
                    )
                handle.flush()
                typer.echo(
                    f"{name} seed={seed}: accuracy={outcome.metrics.accuracy:.4f}"
                    f" macro_f1={outcome.metrics.macro_f1:.4f}"
                )
    with (output_dir / "run_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(runs[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in runs)
    _ = (output_dir / "reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("fasttext-wheel", "numpy", "scikit-learn")},
        "seeds": SEEDS,
        "train_rows": len(train),
        "val_rows": len(val),
        "val_groups": len({row.group for row in val}),
        "sha256": {
            name: hashlib.sha256((DEFAULT_DATA_DIR / name).read_bytes()).hexdigest()
            for name in ("train.csv", "val.csv")
        },
        "original_vectors_sha256": hashlib.sha256(VECTOR_PATH.read_bytes()).hexdigest(),
        "settings": {
            "dim": 100,
            "epoch": 100,
            "lr": 0.5,
            "wordNgrams": 2,
            "minn": 2,
            "maxn": 5,
            "bucket": 200000,
            "thread": 1,
        },
        "initialization": initialization_reports,
        "scale": "one global positive scalar matching native median non-EOS word-row norm",
        "shuffle": "seeded random single cycle over all non-EOS rows; EOS stays at its own row",
        "fixed": "same pretrained corpus vectors, train/val split, word IDs and hyperparameters",
        "limitation": "Permutation changes word-to-norm and word-to-direction links together.",
        "evaluation": "Post-hoc exploratory validation ablation; not independent test performance.",
    }
    _ = (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"Completed {len(runs)} paired runs; artifacts: {output_dir}")


if __name__ == "__main__":
    typer.run(main)
