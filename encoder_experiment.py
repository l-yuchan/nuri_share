# Copyright 2026 Nuri ML contributors
"""Compare frozen multilingual E5 embeddings on the supplied Nuri split."""

import csv
import hashlib
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Annotated, Final

import numpy as np
import torch
import typer
from sentence_transformers import SentenceTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from experiment import write_predictions
from fasttext_experiment import report_predictions
from nuri_data import DEFAULT_DATA_DIR, normalize_syllables, read_splits

MODEL: Final = "intfloat/multilingual-e5-small"
REVISION: Final = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MAX_TOKENS: Final = 512


def main(
    *,
    output_dir: Annotated[Path, typer.Option()] = Path("results_encoder"),
    batch_size: Annotated[int, typer.Option(min=1)] = 16,
    threads: Annotated[int, typer.Option(min=1)] = 4,
) -> None:
    """Encode descriptions once, then train six linear classifiers on train labels."""
    warnings.simplefilter("error", ConvergenceWarning)
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    train, val = read_splits(DEFAULT_DATA_DIR)
    texts = ["query: " + normalize_syllables(row.text) for row in (*train, *val)]
    output_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    encoder = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    encoder.max_seq_length = MAX_TOKENS
    load_seconds = perf_counter() - start
    tokenized = encoder.tokenizer(texts, truncation=False)
    lengths = [len(row) for row in tokenized["input_ids"]]
    start = perf_counter()
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    encode_seconds = perf_counter() - start
    artifacts = Path("artifacts/encoder")
    artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifacts / "embeddings.npz", embeddings=embeddings)
    x_train, x_val = embeddings[: len(train)], embeddings[len(train) :]
    results = []
    reports = {}
    best_key = (-1.0, -1.0, -1.0)
    best_name = ""
    for c in (0.1, 1.0, 10.0):
        for model_name, classifier in (
            ("logistic", LogisticRegression(C=c, max_iter=2000, random_state=42)),
            ("svm", LinearSVC(C=c, dual=True, max_iter=10000, random_state=42)),
        ):
            start = perf_counter()
            _ = classifier.fit(x_train, [row.label for row in train])
            scores = classifier.decision_function(x_val)
            indices = np.argsort(scores, axis=1, kind="stable")[:, ::-1]
            ranked = [[str(classifier.classes_[i]) for i in row] for row in indices]
            seconds = perf_counter() - start
            accuracy, macro_f1, top2, report = report_predictions(val, ranked)
            name = f"e5_{model_name}_C{c:g}"
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
        "versions": {
            name: version(name)
            for name in ("sentence-transformers", "transformers", "torch", "scikit-learn", "numpy")
        },
        "model": MODEL,
        "revision": REVISION,
        "encoder": "frozen CPU; query: prefix for classification; L2 normalized; no fine-tuning",
        "max_seq_length": MAX_TOKENS,
        "batch_size": batch_size,
        "threads": threads,
        "seed": 42,
        "train_rows": len(train),
        "val_rows": len(val),
        "truncated_train": sum(length > MAX_TOKENS for length in lengths[: len(train)]),
        "truncated_val": sum(length > MAX_TOKENS for length in lengths[len(train) :]),
        "maximum_token_count": max(lengths),
        "load_seconds": load_seconds,
        "encode_seconds": encode_seconds,
        "input_text_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
        "sha256": {
            name: hashlib.sha256((DEFAULT_DATA_DIR / name).read_bytes()).hexdigest()
            for name in ("train.csv", "val.csv")
        },
        "external_corpus": "Not used; pretrained E5 weights are frozen",
        "pretraining_contamination": "Upstream E5 pretraining data overlap was not audited",
        "best_setting": best_name,
        "selection": "validation macro_f1, then accuracy, then top2",
        "evaluation_note": "Exploratory validation comparison, not independent test performance.",
    }
    _ = (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"Best: {best_name}; encoding={encode_seconds:.2f}s")


if __name__ == "__main__":
    typer.run(main)
