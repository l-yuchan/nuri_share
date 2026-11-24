# Copyright 2026 Nuri ML contributors
"""Control for max-similarity: the same scorer with random token vectors.

Max-similarity compares the best-matching token pair, so two texts that share a
word compare that word to itself at cosine 1.0. The control replaces every
vector with a random draw while keeping the vocabulary, the tokenizer and the
scorer identical, so any remaining score is shared surface tokens.

Run with ``uv run python indicator_maxsim_control.py``.
"""

import json
from pathlib import Path
from typing import Annotated, Final

import numpy as np
import typer

from indicator_linking_experiment import build_text_bank, maxsim_scores
from indicator_task import load_linking_data, ranking_metrics, target_indices
from nuri_data import DEFAULT_DATA_DIR

DEFAULT_OUTPUT: Final = Path("results_indicator_linking/maxsim_control.json")


def main(
    output: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT,
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATA_DIR,
    seeds: Annotated[str, typer.Option()] = "13,42,73",
) -> None:
    """Score the random-vector twin of cc_maxsim on both protocols."""
    data = load_linking_data(data_dir)
    bank = build_text_bank(data)
    train_truth = target_indices(data.train_rows, data)
    val_truth = target_indices(data.val_rows, data)
    n_train = len(data.train_rows)
    records: list[dict[str, float]] = []
    for value in seeds.split(","):
        rng = np.random.default_rng(int(value))
        vectors = rng.normal(0.0, 0.1, (len(bank.vocabulary), 300)).astype(np.float32)
        vectors[0] = 0.0
        scores = maxsim_scores(bank, vectors)
        cv = ranking_metrics(scores[:n_train], train_truth)
        val = ranking_metrics(scores[n_train:], val_truth)
        records.append(
            {
                "seed": float(value),
                "cv_recall_at_1": cv["recall@1"],
                "cv_recall_at_3": cv["recall@3"],
                "cv_mrr": cv["mrr"],
                "val_recall_at_1": val["recall@1"],
                "val_recall_at_3": val["recall@3"],
                "val_mrr": val["mrr"],
            }
        )
    summary = {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
        if key != "seed"
    }
    payload = {"condition": "random_maxsim", "seeds": records, "mean": summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(payload["mean"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    typer.run(main)
