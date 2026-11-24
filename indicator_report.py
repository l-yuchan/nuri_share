# Copyright 2026 Nuri ML contributors
"""Render the linking results as markdown tables straight from the artifacts.

Run after ``indicator_linking_experiment.py``:

    uv run python indicator_report.py --results results_indicator_linking

The document quotes this output, so a reported number cannot drift from the
saved metrics.
"""

import csv
import json
from pathlib import Path
from typing import Annotated, Final

import typer

DEFAULT_RESULTS: Final = Path("results_indicator_linking")
CONDITION_ORDER: Final = (
    "rule_char2",
    "tfidf_cosine",
    "tfidf_pair",
    "cc_zeroshot",
    "cc_maxsim",
    "e5_zeroshot",
    "random_frozen",
    "random_train",
    "cc_frozen",
    "cc_train",
    "shuffled_frozen",
    "shuffled_train",
    "e5_proj",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(
    results: Annotated[Path, typer.Option()] = DEFAULT_RESULTS,
) -> None:
    """Write ``report.md`` with the main table and the paired contrasts."""
    summary = read_csv(results / "summary.csv")
    runs = read_csv(results / "runs.csv")
    contrasts = json.loads((results / "contrasts.json").read_text(encoding="utf-8"))
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    lookup = {(row["condition"], row["protocol"]): row for row in summary}
    seeds = sorted({row["seed"] for row in runs})
    lines: list[str] = [
        "# Indicator linking results",
        "",
        (
            f"- validation indicators (unseen): {metadata['val_indicators']}"
            f" of {metadata['total_candidates']} candidates"
        ),
        (
            f"- training indicators: {metadata['train_indicators']}"
            f", overlap: {metadata['indicator_overlap']}"
        ),
        f"- epochs: {metadata['epochs']}, seeds: {metadata['seeds']}",
        f"- vocabulary: {metadata['vocabulary']}, train OOV rate: {metadata['train_oov_rate']:.4f}",
        "",
        "## Held-out-indicator validation (11 unseen definitions, 59 candidates)",
        "",
        "| Condition | Recall@1 | Recall@3 | MRR | Seed SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITION_ORDER:
        row = lookup.get((condition, "val"))
        if row is None:
            continue
        lines.append(
            f"| {condition} | {float(row['recall_at_1_mean']) * 100:.2f}%"
            f" | {float(row['recall_at_3_mean']) * 100:.2f}%"
            f" | {float(row['mrr_mean']):.4f}"
            f" | {float(row['recall_at_1_sd']) * 100:.2f}pp |"
        )
    lines += [
        "",
        "## Group cross-validation (48 held-out indicators inside the training split)",
        "",
        "| Condition | Recall@1 | Recall@3 | MRR |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITION_ORDER:
        row = lookup.get((condition, "cv"))
        if row is None:
            continue
        lines.append(
            f"| {condition} | {float(row['recall_at_1_mean']) * 100:.2f}%"
            f" | {float(row['recall_at_3_mean']) * 100:.2f}%"
            f" | {float(row['mrr_mean']):.4f} |"
        )
    lines += [
        "",
        "## Paired group-bootstrap contrasts (95% percentile interval)",
        "",
        (
            "| Treatment - control | Protocol | Recall@1 delta | 95% interval"
            " | MRR delta | 95% interval |"
        ),
        "|---|---|---:|---|---:|---|",
    ]
    for row in contrasts:
        lines.append(
            f"| {row['treatment']} - {row['control']} | {row['protocol']}"
            f" | {row['recall_at_1_delta'] * 100:+.2f}pp"
            f" | [{row['recall_at_1_ci_low'] * 100:+.2f}, {row['recall_at_1_ci_high'] * 100:+.2f}]"
            f" | {row['mrr_delta']:+.4f}"
            f" | [{row['mrr_ci_low']:+.4f}, {row['mrr_ci_high']:+.4f}] |"
        )
    lines += ["", f"Seeds: {', '.join(str(seed) for seed in seeds)}", ""]
    control = results / "maxsim_control.json"
    if control.exists():
        payload = json.loads(control.read_text(encoding="utf-8"))
        mean = payload["mean"]
        lines += [
            "## Max-similarity control (random token vectors)",
            "",
            "| Protocol | Recall@1 | Recall@3 | MRR |",
            "|---|---:|---:|---:|",
            (
                f"| cv | {mean['cv_recall_at_1'] * 100:.2f}%"
                f" | {mean['cv_recall_at_3'] * 100:.2f}% | {mean['cv_mrr']:.4f} |"
            ),
            (
                f"| val | {mean['val_recall_at_1'] * 100:.2f}%"
                f" | {mean['val_recall_at_3'] * 100:.2f}% | {mean['val_mrr']:.4f} |"
            ),
            "",
            "Same scorer and vocabulary with random vectors; the remaining score is",
            "shared surface tokens comparing to themselves.",
            "",
        ]
    if "scaling" in metadata:
        scaling = metadata["scaling"]
        lines += [
            "## Vector scaling",
            "",
            f"- cc.ko.300 median token norm: {scaling['cc_median_norm']:.4f}",
            f"- random median token norm: {scaling['random_median_norm']:.4f}",
            f"- common factor applied to cc vectors: {scaling['factor']:.4f}",
            f"- shuffle seed: {scaling['shuffle_seed']}",
            "",
        ]
    if "cc_ko_300" in metadata:
        cc = metadata["cc_ko_300"]
        lines += [
            "## Source model",
            "",
            f"- file: {cc['file_bytes']} bytes, sha256 {cc['sha256']}",
            f"- dimension: {int(cc['dimension'])}",
            "",
        ]
    (results / "report.md").write_text("\n".join(lines), encoding="utf-8")
    typer.echo("\n".join(lines))


if __name__ == "__main__":
    typer.run(main)
