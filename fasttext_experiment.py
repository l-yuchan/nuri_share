# Copyright 2026 Nuri ML contributors
"""Compare supervised fastText and corpus-trained sentence vectors."""

import csv
import hashlib
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Annotated

import fasttext
import numpy as np
import typer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

from experiment import write_predictions
from fasttext_features import train_vectors
from nuri_data import DEFAULT_DATA_DIR, LABELS, Example, read_splits
from prepare_corpus import clean_text


def report_predictions(
    examples: tuple[Example, ...], ranked: list[list[str]]
) -> tuple[float, float, float, str]:
    """Evaluate all labels explicitly, with source-specific diagnostic reports."""
    actual = [row.label for row in examples]
    predicted = [row[0] for row in ranked]
    report = {
        "classification_report": classification_report(
            actual, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
        "by_source": {
            source: classification_report(
                [row.label for row in examples if row.source == source],
                [
                    pred
                    for row, pred in zip(examples, predicted, strict=True)
                    if row.source == source
                ],
                labels=LABELS,
                output_dict=True,
                zero_division=0,
            )
            for source in sorted({row.source for row in examples})
        },
    }
    return (
        float(accuracy_score(actual, predicted)),
        float(f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)),
        sum(row.label in top[:2] for row, top in zip(examples, ranked, strict=True))
        / len(examples),
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def main(
    *,
    corpus_dir: Annotated[Path, typer.Option()] = Path("artifacts/corpus"),
    output_dir: Annotated[Path, typer.Option()] = Path("results_fasttext"),
    threads: Annotated[int, typer.Option(min=1)] = 4,
) -> None:
    """Train a CBOW model, six frozen-vector classifiers and four supervised settings."""
    warnings.simplefilter("error", ConvergenceWarning)
    train, val = read_splits(DEFAULT_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("artifacts/fasttext")
    models_dir.mkdir(parents=True, exist_ok=True)
    train_text = [clean_text(row.text) for row in train]
    val_text = [clean_text(row.text) for row in val]
    training_path = models_dir / "train.txt"
    _ = training_path.write_text(
        "\n".join(
            f"__label__{LABELS.index(row.label)} {text}"
            for row, text in zip(train, train_text, strict=True)
        )
        + "\n",
        encoding="utf-8",
    )
    vectors = train_vectors(corpus_dir / "sentences.txt", (train_text, val_text), threads)
    results = []
    reports = {}
    best_key = (-1.0, -1.0, -1.0)
    best_name = ""

    def record(name: str, ranked: list[list[str]], seconds: float) -> None:
        """Accumulate metrics and retain each model's predictions for error analysis."""
        nonlocal best_key, best_name
        accuracy, macro_f1, top2, report = report_predictions(val, ranked)
        results.append(
            {
                "setting": name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "top2_accuracy": top2,
                "fit_and_predict_seconds": seconds,
            }
        )
        reports[name] = json.loads(report)
        write_predictions(output_dir / f"{name}_predictions.csv", val, ranked)
        key = (macro_f1, accuracy, top2)
        if key > best_key:
            best_key, best_name = key, name
            write_predictions(output_dir / "best_predictions.csv", val, ranked)
        typer.echo(f"{name}: accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} top2={top2:.4f}")

    for c in (0.1, 1.0, 10.0):
        for model_name, classifier in (
            ("logistic", LogisticRegression(C=c, max_iter=2000, random_state=42)),
            ("svm", LinearSVC(C=c, dual=True, max_iter=10000, random_state=42)),
        ):
            start = perf_counter()
            _ = classifier.fit(vectors.train, [row.label for row in train])
            scores = classifier.decision_function(vectors.val)
            indices = np.argsort(scores, axis=1, kind="stable")[:, ::-1]
            ranked = [[str(classifier.classes_[i]) for i in row_indices] for row_indices in indices]
            record(f"cbow_{model_name}_C{c:g}", ranked, perf_counter() - start)
    for pretrained in (False, True):
        for epochs in (25, 100):
            start = perf_counter()
            model = fasttext.train_supervised(
                str(training_path),
                dim=100,
                epoch=epochs,
                lr=0.5,
                wordNgrams=2,
                minn=2,
                maxn=5,
                bucket=200000,
                thread=1,
                seed=42,
                verbose=0,
                pretrainedVectors=str(vectors.word_vectors) if pretrained else "",
            )
            # Batch prediction avoids the upstream NumPy 2 single-string copy=False bug.
            predicted_labels, _ = model.predict(val_text, k=5)
            ranked = [
                [LABELS[int(label.removeprefix("__label__"))] for label in labels]
                for labels in predicted_labels
            ]
            name = f"supervised_{'pretrained' if pretrained else 'scratch'}_epoch{epochs}"
            model.save_model(str(models_dir / f"{name}.bin"))
            record(name, ranked, perf_counter() - start)
    results.sort(key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    _ = (output_dir / "reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("fasttext-wheel", "numpy", "scikit-learn")},
        "train_rows": len(train),
        "val_rows": len(val),
        "sha256": {
            name: hashlib.sha256((DEFAULT_DATA_DIR / name).read_bytes()).hexdigest()
            for name in ("train.csv", "val.csv")
        },
        "corpus_manifest": json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8")),
        "pretraining": {
            "algorithm": "cbow",
            "dim": 100,
            "epoch": 5,
            "minCount": 3,
            "minn": 2,
            "maxn": 5,
            "bucket": 200000,
            "threads": threads,
            "seed": 0,
            "seconds": vectors.pretrain_seconds,
        },
        "vectorize_seconds": vectors.vectorize_seconds,
        "supervised": {
            "dim": 100,
            "lr": 0.5,
            "wordNgrams": 2,
            "minn": 2,
            "maxn": 5,
            "bucket": 200000,
            "threads": 1,
            "seed": 42,
            "pretrained_vectors": "CBOW word vectors for Nuri TRAIN vocabulary only",
        },
        "determinism": "Multithreaded CBOW is not bitwise deterministic; use --threads 1",
        "text": "NFC lowercase whitespace tokens; reserved __label__ syntax removed",
        "best_setting": best_name,
        "selection": "validation macro_f1, then accuracy, then top2",
        "evaluation_note": "Exploratory validation comparison, not independent test performance.",
    }
    _ = (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"Best: {best_name}; pretraining={vectors.pretrain_seconds:.2f}s")


if __name__ == "__main__":
    typer.run(main)
