# Copyright 2026 Nuri ML contributors
"""Compare TF-IDF feature settings on the provided validation split.

Run with ``uv run python experiment.py``; dependencies are locked by uv.lock.
"""

import csv
import hashlib
import json
import platform
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Final

import numpy as np
import sklearn
import typer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.svm import LinearSVC

from nuri_data import DEFAULT_DATA_DIR, Example, decompose_jamo, normalize_syllables, read_splits

if TYPE_CHECKING:
    from numpy.typing import NDArray

SEED: Final = 42
DEFAULT_OUTPUT_DIR: Final = Path("results")
FEATURES: Final = (
    ("syllable_2-5", normalize_syllables, (2, 5)),
    ("jamo_2-5", decompose_jamo, (2, 5)),
    ("jamo_3-6", decompose_jamo, (3, 6)),
    ("jamo_3-8", decompose_jamo, (3, 8)),
)


@dataclass(frozen=True, slots=True)
class Result:
    """A single validation result; timing excludes file I/O."""

    features: str
    model: str
    c: float
    vocabulary_size: int
    accuracy: float
    macro_f1: float
    top2_accuracy: float
    vectorize_seconds: float
    fit_seconds: float
    predict_seconds: float


def make_vectorizer(preprocessor: Callable[[str], str], ngrams: tuple[int, int]) -> TfidfVectorizer:
    """Build train-only TF-IDF; spaces and punctuation remain in character n-grams."""
    return TfidfVectorizer(
        analyzer="char",
        preprocessor=preprocessor,
        lowercase=False,
        ngram_range=ngrams,
        sublinear_tf=True,
        min_df=1,
        norm="l2",
        dtype=np.float64,
    )


def write_predictions(
    path: Path,
    examples: tuple[Example, ...],
    ranked_labels: list[list[str]],
) -> None:
    """Export top two labels so validation errors can be inspected."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "source", "group_id", "text", "label", "top1", "top2", "correct"))
        for row, ranked in zip(examples, ranked_labels, strict=True):
            writer.writerow(
                (
                    row.identifier,
                    row.source,
                    row.group,
                    row.text,
                    row.label,
                    ranked[0],
                    ranked[1],
                    row.label == ranked[0],
                )
            )


def main(
    *,
    data_dir: Annotated[
        Path, typer.Option(help="Directory containing train.csv and val.csv")
    ] = DEFAULT_DATA_DIR,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for experiment artifacts")
    ] = DEFAULT_OUTPUT_DIR,
    default_only: Annotated[bool, typer.Option(help="Keep only use_default=yes rows")] = False,
) -> None:
    """Run 24 settings; select by validation Macro-F1, then accuracy and top-2."""
    warnings.simplefilter("error", ConvergenceWarning)
    train, val = read_splits(data_dir, default_only=default_only)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_text = [row.text for row in train]
    train_labels = [row.label for row in train]
    val_text = [row.text for row in val]
    val_labels = [row.label for row in val]
    results: list[Result] = []
    reports = {}
    best_key = (-1.0, -1.0, -1.0)
    best_name = ""
    best_ranked: list[list[str]] = []
    typer.echo(f"Train={len(train)}, val={len(val)}, default_only={default_only}")
    for feature_name, preprocessor, ngrams in FEATURES:
        start = perf_counter()
        vectorizer = make_vectorizer(preprocessor, ngrams)
        x_train = vectorizer.fit_transform(train_text)
        x_val = vectorizer.transform(val_text)
        vectorize_seconds = perf_counter() - start
        for c in (0.1, 1.0, 10.0):
            classifiers = (
                ("logistic_regression", LogisticRegression(C=c, max_iter=2000, random_state=SEED)),
                ("linear_svm", LinearSVC(C=c, dual=True, max_iter=10000, random_state=SEED)),
            )
            for model_name, classifier in classifiers:
                start = perf_counter()
                _ = classifier.fit(x_train, train_labels)
                fit_seconds = perf_counter() - start
                start = perf_counter()
                scores: NDArray[np.float64] = classifier.decision_function(x_val)
                classes: NDArray[np.str_] = classifier.classes_
                # Match sklearn top-k tie handling: a larger class index wins ties.
                indices: NDArray[np.intp] = np.argsort(scores, axis=1, kind="stable")[:, ::-1]
                ranked: list[list[str]] = [
                    [str(label) for label in classes[row_indices]] for row_indices in indices
                ]
                predicted = [row[0] for row in ranked]
                predict_seconds = perf_counter() - start
                result = Result(
                    feature_name,
                    model_name,
                    c,
                    len(vectorizer.get_feature_names_out()),
                    float(accuracy_score(val_labels, predicted)),
                    float(f1_score(val_labels, predicted, average="macro", zero_division=0)),
                    float(
                        top_k_accuracy_score(val_labels, scores, k=2, labels=classifier.classes_)
                    ),
                    vectorize_seconds,
                    fit_seconds,
                    predict_seconds,
                )
                results.append(result)
                name = f"{feature_name}__{model_name}__C{c:g}"
                reports[name] = {
                    "classification_report": classification_report(
                        val_labels,
                        predicted,
                        output_dict=True,
                        zero_division=0,
                    ),
                    "labels": classifier.classes_.tolist(),
                    "confusion_matrix": confusion_matrix(
                        val_labels,
                        predicted,
                        labels=classifier.classes_,
                    ).tolist(),
                    "by_source": {
                        source: classification_report(
                            [row.label for row in val if row.source == source],
                            [
                                pred
                                for row, pred in zip(val, predicted, strict=True)
                                if row.source == source
                            ],
                            labels=classifier.classes_,
                            output_dict=True,
                            zero_division=0,
                        )
                        for source in sorted({row.source for row in val})
                    },
                }
                key = (result.macro_f1, result.accuracy, result.top2_accuracy)
                if key > best_key:
                    best_key, best_name, best_ranked = key, name, ranked
                typer.echo(
                    " ".join(
                        (
                            f"{name}: accuracy={result.accuracy:.4f}",
                            f"macro_f1={result.macro_f1:.4f} top2={result.top2_accuracy:.4f}",
                            f"fit={fit_seconds:.3f}s",
                        )
                    )
                )
    results.sort(key=lambda row: (row.macro_f1, row.accuracy, row.top2_accuracy), reverse=True)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)
    _ = (output_dir / "reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "seed": SEED,
        "default_only": default_only,
        "train_rows": len(train),
        "val_rows": len(val),
        "train_labels": dict(Counter(train_labels)),
        "val_labels": dict(Counter(val_labels)),
        "train_groups": len({row.group for row in train}),
        "val_groups": len({row.group for row in val}),
        "sha256": {
            name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in ("train.csv", "val.csv")
        },
        "best_setting": best_name,
        "selection": "validation macro_f1, then accuracy, then top2_accuracy",
        "evaluation_note": "Exploratory validation comparison, not an independent test score.",
        "jamo": "Unicode NFD; initial and final consonants remain distinct",
        "tfidf": "char, sublinear_tf=True, min_df=1, norm=l2; fitted on train only",
    }
    _ = (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_predictions(output_dir / "best_predictions.csv", val, best_ranked)
    typer.echo(f"Best: {best_name}; artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    typer.run(main)
