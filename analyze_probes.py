# Copyright 2026 Nuri ML contributors
"""Grade the returned LLM and human dataset-quality probes.

The LLM answers are pasted model output (one line per item); the human answers
are the CSV exported by ``probes/human_probe.html``. Both are compared with the
supplied gold labels and with the fitted TF-IDF model recorded in
``probes/manifest.json``.

Run with ``uv run python analyze_probes.py --llm probes/llm/results/gpt``.
"""

import csv
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final, TypedDict

import typer

from nuri_data import LABELS

PROBE_DIR: Final = Path("probes")
DEFAULT_MANIFEST: Final = PROBE_DIR / "manifest.json"
DEFAULT_OUTPUT: Final = PROBE_DIR / "analysis.md"
ID_PATTERN: Final = re.compile(r"^[NK]\d{3}$")
RESULT_SUFFIXES: Final = (".txt", ".md", ".csv", ".tsv")
LOW_SCORE: Final = 2
HIGH_SCORE: Final = 4
MIN_JUDGES: Final = 2
MIN_NAME_PARTS: Final = 2
STRONG_CONSENSUS: Final = 3


class ClassScore(TypedDict):
    """Per-area accuracy."""

    n: int
    accuracy: float


class Summary(TypedDict):
    """Metrics computed for any single judge."""

    n: int
    accuracy: float
    macro_f1: float
    top2_accuracy: float
    by_class: dict[str, ClassScore]
    by_source: dict[str, float]
    group_mean_accuracy: float
    groups: int
    confusion: dict[str, dict[str, int]]


class LlmSuspect(TypedDict):
    """An item whose gold label the LLM scored low while another area scored high."""

    id: str
    text: str
    gold: str
    gold_rating: int
    llm_top1: str
    model_top1: str
    reason: str


class LlmReport(Summary):
    """Full report for one LLM answer file."""

    file: str
    parsed: int
    missing: list[str]
    unknown: list[str]
    gold_rating_mean: float
    gold_rating_distribution: dict[int, int]
    low_gold_rating: int
    ambiguous: int
    suspects: list[LlmSuspect]
    agreement_with_model: float
    kappa_with_model: float
    issues: list[str]


class HumanSuspect(TypedDict):
    """An item the person flagged as mislabeled."""

    id: str
    text: str
    gold: str
    chosen: str
    note: str


class GroupAccuracy(TypedDict):
    """Accuracy within one selection stratum or hint condition."""

    n: int
    accuracy: float


class HumanReport(Summary):
    """Full report for the human probe export."""

    file: str
    total_items: int
    answered: int
    ambiguous: int
    by_stratum: dict[str, GroupAccuracy]
    by_hint_use: dict[str, GroupAccuracy]
    median_seconds: float
    hits: int
    suspects: list[HumanSuspect]


@dataclass(frozen=True, slots=True)
class Judgement:
    """One parsed LLM answer for one item."""

    identifier: str
    ratings: tuple[int, ...]
    top1: str
    reason: str


@dataclass(slots=True)
class Item:
    """Gold label plus the fitted model's answer for one validation row."""

    identifier: str
    gold: str
    source: str
    group: str
    model_top1: str
    text: str = ""


def strip_markup(line: str) -> str:
    """Remove code fences, table pipes and emphasis from one answer line."""
    cleaned = line.strip().replace("**", "").replace("`", "")
    return cleaned.strip("|").strip()


def split_fields(line: str) -> list[str]:
    """Split a pipe, tab or comma row without losing a single-field line."""
    stripped = strip_markup(line)
    if "|" in stripped:
        parts = stripped.split("|")
    elif "\t" in stripped:
        parts = stripped.split("\t")
    elif "," in stripped:
        parts = stripped.split(",")
    else:
        parts = stripped.split(None, 1)
    return [part.strip() for part in parts]


def parse_line(line: str) -> tuple[Judgement, list[str]] | None:
    """Parse one line into scores, a top label and a reason."""
    parts = split_fields(line)
    if not parts or not ID_PATTERN.match(parts[0]):
        return None
    numbers: list[int] = []
    labels: list[str] = []
    texts: list[str] = []
    for part in parts[1:]:
        tokens = part.replace(",", " ").split()
        if tokens and all(token.isdigit() for token in tokens):
            numbers.extend(int(token) for token in tokens)
        elif part in LABELS:
            labels.append(part)
        elif part:
            texts.append(part)
    if len(numbers) != len(LABELS) or any(number < 1 or number > len(LABELS) for number in numbers):
        return None
    highest = LABELS[numbers.index(max(numbers))]
    warnings = []
    if labels and labels[0] != highest:
        warnings.append(f"{parts[0]}: top1 differs from the highest score")
    top1 = labels[0] if labels else highest
    return Judgement(parts[0], tuple(numbers), top1, texts[-1] if texts else ""), warnings


def is_noise(line: str) -> bool:
    """Skip blank lines, Markdown fences and table headers."""
    stripped = strip_markup(line)
    if not stripped or stripped.startswith("```"):
        return True
    if set(stripped) <= set("|-: "):
        return True
    return stripped.lower() == "id" or stripped.lower().startswith("id ")


def parse_llm_output(text: str) -> tuple[dict[str, Judgement], list[str]]:
    """Parse a pasted answer block; report unparseable lines instead of failing."""
    answers: dict[str, Judgement] = {}
    issues: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if is_noise(line):
            continue
        parsed = parse_line(line)
        if parsed is None:
            if strip_markup(line):
                issues.append(f"line {number}: {line.strip()[:80]}")
            continue
        judgement, warnings = parsed
        issues.extend(f"line {number}: {warning}" for warning in warnings)
        if judgement.identifier in answers:
            issues.append(f"line {number}: duplicate {judgement.identifier} ignored")
            continue
        answers[judgement.identifier] = judgement
    return answers, issues


def macro_f1(gold: list[str], predicted: list[str]) -> float:
    """Fixed-class Macro-F1 over the five areas."""
    scores: list[float] = []
    for label in LABELS:
        true_positive = sum(
            actual == label and call == label for actual, call in zip(gold, predicted, strict=True)
        )
        predicted_positive = sum(call == label for call in predicted)
        actual_positive = sum(actual == label for actual in gold)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def cohen_kappa(left: list[str], right: list[str]) -> float:
    """Agreement corrected for chance between two label sequences."""
    if not left:
        return 0.0
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(left_counts[label] * right_counts[label] for label in set(left) | set(right)) / (
        len(left) * len(right)
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def describe(
    gold: list[str], predicted: list[str], groups: list[str], sources: list[str]
) -> Summary:
    """Accuracy, Macro-F1 and breakdowns for one judge."""
    correct = [actual == call for actual, call in zip(gold, predicted, strict=True)]
    by_group: dict[str, list[bool]] = {}
    by_source: dict[str, list[bool]] = {}
    for ok, group, source in zip(correct, groups, sources, strict=True):
        by_group.setdefault(group, []).append(ok)
        by_source.setdefault(source, []).append(ok)
    by_class: dict[str, ClassScore] = {}
    for label in LABELS:
        hits = [ok for ok, actual in zip(correct, gold, strict=True) if actual == label]
        by_class[label] = {"n": len(hits), "accuracy": sum(hits) / len(hits) if hits else 0.0}
    confusion: dict[str, dict[str, int]] = {actual: dict.fromkeys(LABELS, 0) for actual in LABELS}
    for actual, call in zip(gold, predicted, strict=True):
        confusion[actual][call] += 1
    return {
        "n": len(gold),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "macro_f1": macro_f1(gold, predicted),
        "top2_accuracy": 0.0,
        "by_class": by_class,
        "by_source": {key: sum(value) / len(value) for key, value in by_source.items()},
        "group_mean_accuracy": (
            sum(sum(value) / len(value) for value in by_group.values()) / len(by_group)
            if by_group
            else 0.0
        ),
        "groups": len(by_group),
        "confusion": confusion,
    }


def load_manifest(path: Path) -> dict[str, Item]:
    """Read the probe manifest and index validation rows by id."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: dict[str, Item] = {}
    for row in raw["validation"]:
        items[row["id"]] = Item(
            identifier=row["id"],
            gold=row["label"],
            source=row["source"],
            group=row["group"],
            model_top1=row["top1"],
        )
    return items


def fill_texts(items: dict[str, Item], path: Path) -> None:
    """Attach the blinded item text from the probe TSV when it is available."""
    if not path.exists():
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            item = items.get(row["id"])
            if item is not None:
                item.text = row["text"]


def result_files(paths: list[Path]) -> list[Path]:
    """Expand directories into the answer files they contain."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(child for child in path.rglob("*") if child.suffix in RESULT_SUFFIXES)
            )
        elif path.exists():
            files.append(path)
    return files


def merge_result_sets(files: list[Path]) -> dict[str, list[Path]]:
    """Group ``batch_<n>_<run>`` files into one judge per model directory and run."""
    grouped: dict[str, list[Path]] = {}
    for path in files:
        parts = path.stem.split("_")
        run = parts[-1] if len(parts) >= MIN_NAME_PARTS else "all"
        key = f"{path.parent.name}/run{run}"
        grouped.setdefault(key, []).append(path)
    return dict(sorted(grouped.items()))


def read_group(group: list[Path]) -> tuple[dict[str, Judgement], list[str]]:
    """Parse every file in a group, keeping the first answer for duplicate ids."""
    answers: dict[str, Judgement] = {}
    issues: list[str] = []
    for path in group:
        partial, partial_issues = parse_llm_output(path.read_text(encoding="utf-8"))
        for identifier, judgement in partial.items():
            if identifier in answers:
                issues.append(f"{path.name}: duplicate {identifier} ignored")
            else:
                answers[identifier] = judgement
        issues.extend(f"{path.name}: {issue}" for issue in partial_issues)
    return answers, issues


def human_predictions(path: Path) -> dict[str, str]:
    """Read the person's first choice per item, skipping ambiguous answers."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["id"]: row["choice1"]
            for row in csv.DictReader(handle)
            if row.get("id") and row["ambiguous"] != "yes" and row["choice1"]
        }


def render_agreement(judges: dict[str, dict[str, str]]) -> list[str]:
    """Pairwise raw agreement and Cohen kappa on each pair's shared items."""
    names = list(judges)
    lines = [table_row(["판단", *names]), table_row(["---"] * (len(names) + 1))]
    for left_name in names:
        cells = [left_name]
        for right_name in names:
            shared = sorted(set(judges[left_name]) & set(judges[right_name]))
            left = [judges[left_name][identifier] for identifier in shared]
            right = [judges[right_name][identifier] for identifier in shared]
            if left_name == right_name:
                cells.append("—")
            else:
                agree = (
                    sum(a == b for a, b in zip(left, right, strict=True)) / len(shared)
                    if shared
                    else 0.0
                )
                cells.append(f"{pct(agree)} (κ {cohen_kappa(left, right):.2f}, n={len(shared)})")
        lines.append(table_row(cells))
    return lines


def render_all_wrong(judges: dict[str, dict[str, str]], items: dict[str, Item]) -> list[str]:
    """List items where every available LLM run and the person disagree with gold."""
    candidates = [name for name in judges if name != "tf-idf"]
    if len(candidates) < MIN_JUDGES:
        return []
    required = max(3, len(candidates) - 1)
    rows = []
    for identifier in sorted(items):
        gold = items[identifier].gold
        picks = {
            name: judges[name][identifier] for name in candidates if identifier in judges[name]
        }
        if len(picks) < required or not all(pick != gold for pick in picks.values()):
            continue
        summary = " · ".join(f"{name.split('/')[0][:3]}:{pick}" for name, pick in picks.items())
        rows.append(table_row([identifier, items[identifier].text[:60], gold, summary]))
    lines = ["## 전원 불일치 (가능한 모든 LLM 실행 + 사람)", ""]
    if not rows:
        return [*lines, "없습니다.", ""]
    lines.extend(
        (
            table_row(["id", "기록", "골드", "각 판단의 선택"]),
            table_row(["---"] * 4),
            *rows,
            "",
        )
    )
    return lines


def llm_report(path: Path, answers: dict[str, Judgement], items: dict[str, Item]) -> LlmReport:
    """Grade one LLM answer file."""
    missing = sorted(set(items) - set(answers))
    unknown = sorted(set(answers) - set(items))
    present = [items[identifier] for identifier in sorted(answers) if identifier in items]
    gold = [item.gold for item in present]
    predicted = [answers[item.identifier].top1 for item in present]
    summary = describe(
        gold, predicted, [item.group for item in present], [item.source for item in present]
    )
    gold_ratings = [answers[item.identifier].ratings[LABELS.index(item.gold)] for item in present]
    top2_hits = 0
    ambiguous = 0
    suspects: list[LlmSuspect] = []
    for item in present:
        judgement = answers[item.identifier]
        order = sorted(range(len(LABELS)), key=lambda index: -judgement.ratings[index])
        if item.gold in (LABELS[order[0]], LABELS[order[1]]):
            top2_hits += 1
        if judgement.ratings[order[0]] - judgement.ratings[order[1]] <= 1:
            ambiguous += 1
        gold_rating = judgement.ratings[LABELS.index(item.gold)]
        best_other = max(
            rating for index, rating in enumerate(judgement.ratings) if LABELS[index] != item.gold
        )
        if gold_rating <= LOW_SCORE and best_other >= HIGH_SCORE:
            suspects.append(
                {
                    "id": item.identifier,
                    "text": item.text[:70],
                    "gold": item.gold,
                    "gold_rating": gold_rating,
                    "llm_top1": judgement.top1,
                    "model_top1": item.model_top1,
                    "reason": judgement.reason,
                }
            )
    summary["top2_accuracy"] = top2_hits / len(present) if present else 0.0
    return {
        **summary,
        "file": str(path),
        "parsed": len(answers) - len(unknown),
        "missing": missing,
        "unknown": unknown,
        "gold_rating_mean": sum(gold_ratings) / len(gold_ratings) if gold_ratings else 0.0,
        "gold_rating_distribution": dict(sorted(Counter(gold_ratings).items())),
        "low_gold_rating": sum(rating <= LOW_SCORE for rating in gold_ratings),
        "ambiguous": ambiguous,
        "suspects": suspects,
        "agreement_with_model": (
            sum(answers[item.identifier].top1 == item.model_top1 for item in present) / len(present)
            if present
            else 0.0
        ),
        "kappa_with_model": cohen_kappa(predicted, [item.model_top1 for item in present]),
        "issues": [],
    }


def human_report(path: Path) -> HumanReport:
    """Grade the CSV exported by the human probe page."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("id")]
    answered = [row for row in rows if row["ambiguous"] == "yes" or row["choice1"]]
    scored = [row for row in answered if row["ambiguous"] != "yes"]
    gold = [row["gold"] for row in scored]
    predicted = [row["choice1"] for row in scored]
    summary = describe(
        gold,
        predicted,
        [row["group_id"] for row in scored],
        [row["source"] for row in scored],
    )
    by_stratum: dict[str, list[bool]] = {}
    by_hint: dict[str, list[bool]] = {}
    for row in scored:
        ok = row["correct"] == "yes"
        by_stratum.setdefault(row.get("stratum", ""), []).append(ok)
        by_hint.setdefault(row["hint_used"], []).append(ok)
    times = [int(row["elapsed_ms"]) for row in scored if row["elapsed_ms"].isdigit()]
    summary["top2_accuracy"] = (
        sum(
            row["correct"] == "yes" or (bool(row["choice2"]) and row["choice2"] == row["gold"])
            for row in scored
        )
        / len(scored)
        if scored
        else 0.0
    )
    return {
        **summary,
        "file": str(path),
        "total_items": len(rows),
        "answered": len(answered),
        "ambiguous": len(answered) - len(scored),
        "by_stratum": {
            key: {"n": len(value), "accuracy": sum(value) / len(value)}
            for key, value in by_stratum.items()
        },
        "by_hint_use": {
            key: {"n": len(value), "accuracy": sum(value) / len(value)}
            for key, value in by_hint.items()
        },
        "median_seconds": statistics.median(times) / 1000 if times else 0.0,
        "hits": sum(row["correct"] == "yes" for row in scored),
        "suspects": [
            {
                "id": row["id"],
                "text": "",
                "gold": row["gold"],
                "chosen": row["choice1"],
                "note": row["note"],
            }
            for row in rows
            if row.get("suspect_label") == "yes"
        ],
    }


def pct(value: float) -> str:
    """Format a fraction as a one-decimal percentage."""
    return f"{100 * value:.1f}%"


def table_row(cells: list[str]) -> str:
    """Render one Markdown table row."""
    return "| " + " | ".join(cells) + " |"


def render_summary(llm_reports: list[LlmReport], human_result: HumanReport | None) -> list[str]:
    """Headline table across judges plus the shared strong label warnings."""
    lines = [
        "## 요약",
        "",
        table_row(["판단", "응답", "1순위", "Macro-F1", "2순위", "골드 평균"]),
        table_row(["---"] * 6),
    ]
    for report in sorted(llm_reports, key=lambda item: -item["accuracy"]):
        total = report["parsed"] + len(report["missing"])
        lines.append(
            table_row(
                [
                    report["file"].replace("\\", "/"),
                    f"{report['parsed']}/{total}",
                    pct(report["accuracy"]),
                    f"{report['macro_f1']:.3f}",
                    pct(report["top2_accuracy"]),
                    f"{report['gold_rating_mean']:.2f}",
                ]
            )
        )
    if human_result is not None:
        lines.append(
            table_row(
                [
                    "human",
                    f"{human_result['answered']}/{human_result['total_items']}",
                    pct(human_result["accuracy"]),
                    f"{human_result['macro_f1']:.3f}",
                    pct(human_result["top2_accuracy"]),
                    "—",
                ]
            )
        )
    flags: Counter[str] = Counter()
    for report in llm_reports:
        for identifier in {row["id"] for row in report["suspects"]}:
            flags[identifier] += 1
    strong = sorted(identifier for identifier, count in flags.items() if count >= STRONG_CONSENSUS)
    if flags:
        lines.extend(
            (
                "",
                "3개 이상 실행이 동시에 '골드 점수 2 이하 + 다른 영역 4 이상'으로 표시한 항목: "
                f"{len(strong)}개" + (f" ({', '.join(strong)})" if strong else ""),
            )
        )
    lines.extend(
        (
            "",
            "골드 평균은 LLM이 골드 라벨에 준 1~5점 평균입니다. 2순위는 점수 순위 기준입니다.",
            "",
        )
    )
    return lines


def render_llm(report: LlmReport) -> list[str]:
    """Render the Markdown blocks for one LLM answer file."""
    total = report["parsed"] + len(report["missing"])
    lines = [f"### {report['file']}", ""]
    lines.extend(
        (
            table_row(["지표", "값"]),
            table_row(["---", "---"]),
            table_row(["응답 항목", f"{report['parsed']} / {total}"]),
            table_row(["1순위 정확도", pct(report["accuracy"])]),
            table_row(["Macro-F1", f"{report['macro_f1']:.3f}"]),
            table_row(["2순위 정확도 (점수 기준)", pct(report["top2_accuracy"])]),
            table_row(
                [
                    "그룹 평균 정확도",
                    f"{pct(report['group_mean_accuracy'])} ({report['groups']}개 그룹)",
                ]
            ),
            table_row(["골드 라벨 평균 점수", f"{report['gold_rating_mean']:.2f} / 5"]),
            table_row(["골드 라벨 1~2점", str(report["low_gold_rating"])]),
            table_row(["점수 1점 차 이내(모호)", str(report["ambiguous"])]),
            table_row(
                [
                    "TF-IDF 모델과 합의",
                    (
                        f"{pct(report['agreement_with_model'])} "
                        f"(kappa {report['kappa_with_model']:.3f})"
                    ),
                ]
            ),
            "",
        )
    )
    lines.append(
        "영역별 정확도: "
        + " · ".join(
            f"{label} {pct(value['accuracy'])} ({value['n']})"
            for label, value in report["by_class"].items()
        )
    )
    lines.append("")
    lines.append(
        "출처별 정확도: "
        + " · ".join(f"{source} {pct(value)}" for source, value in report["by_source"].items())
    )
    if report["missing"]:
        lines.extend(
            ("", f"누락 항목 {len(report['missing'])}개: {', '.join(report['missing'][:30])}")
        )
    if report["unknown"]:
        lines.extend(
            (
                "",
                f"검증셋에 없는 id {len(report['unknown'])}개: {', '.join(report['unknown'][:30])}",
            )
        )
    if report["suspects"]:
        lines.extend(
            (
                "",
                "라벨 오류 의심 (골드 점수 2 이하, 다른 영역 4 이상):",
                "",
                table_row(["id", "기록", "골드", "골드점수", "LLM 1순위", "모델 1순위"]),
                table_row(["---"] * 6),
            )
        )
        lines.extend(
            table_row(
                [
                    row["id"],
                    row["text"],
                    row["gold"],
                    str(row["gold_rating"]),
                    row["llm_top1"],
                    row["model_top1"],
                ]
            )
            for row in report["suspects"]
        )
    lines.append("")
    return lines


def render_human(report: HumanReport) -> list[str]:
    """Render the Markdown blocks for the human probe export."""
    lines = [
        "## 사람 판단",
        "",
        table_row(["지표", "값"]),
        table_row(["---", "---"]),
        table_row(["응답", f"{report['answered']} / {report['total_items']}"]),
        table_row(["판단 어려움", str(report["ambiguous"])]),
        table_row(["1순위 정확도", f"{report['hits']}건, {pct(report['accuracy'])}"]),
        table_row(["Macro-F1", f"{report['macro_f1']:.3f}"]),
        table_row(["2순위 정확도", pct(report["top2_accuracy"])]),
        table_row(
            [
                "그룹 평균 정확도",
                f"{pct(report['group_mean_accuracy'])} ({report['groups']}개 그룹)",
            ]
        ),
        table_row(["문항당 중앙 시간", f"{report['median_seconds']:.1f}초"]),
        "",
    ]
    lines.append(
        "층별 정확도: "
        + " · ".join(
            f"{key} {pct(value['accuracy'])} ({value['n']})"
            for key, value in report["by_stratum"].items()
        )
    )
    lines.append("")
    lines.append(
        "힌트 사용별 정확도: "
        + " · ".join(
            f"{key} {pct(value['accuracy'])} ({value['n']})"
            for key, value in report["by_hint_use"].items()
        )
    )
    lines.append("")
    lines.append(
        "영역별 정확도: "
        + " · ".join(
            f"{label} {pct(value['accuracy'])} ({value['n']})"
            for label, value in report["by_class"].items()
        )
    )
    lines.append("")
    if report["suspects"]:
        lines.extend(
            (
                "사람이 라벨 오류로 표시한 항목:",
                "",
                table_row(["id", "골드", "내 선택", "메모"]),
                table_row(["---"] * 4),
            )
        )
        lines.extend(
            table_row([row["id"], row["gold"], row["chosen"], row["note"]])
            for row in report["suspects"]
        )
        lines.append("")
    return lines


def render_ensemble(
    answers_by_file: dict[str, dict[str, Judgement]], items: dict[str, Item]
) -> list[str]:
    """Majority-vote the LLM runs and report the combined score."""
    shared = (
        sorted(set.intersection(*(set(answers) for answers in answers_by_file.values())))
        if answers_by_file
        else []
    )
    lines = ["## 앙상블 (제출된 LLM 실행 다수결)", ""]
    if not shared:
        return [*lines, "공통으로 응답한 항목이 없습니다.", ""]
    gold = [items[identifier].gold for identifier in shared]
    votes: list[str] = []
    for identifier in shared:
        counts = Counter(answers[identifier].top1 for answers in answers_by_file.values())
        votes.append(min(counts, key=lambda label: (-counts[label], LABELS.index(label))))
    report = describe(
        gold,
        votes,
        [items[identifier].group for identifier in shared],
        [items[identifier].source for identifier in shared],
    )
    lines.extend(
        (
            table_row(["지표", "값"]),
            table_row(["---", "---"]),
            table_row(["공통 항목", str(len(shared))]),
            table_row(["다수결 정확도", pct(report["accuracy"])]),
            table_row(["Macro-F1", f"{report['macro_f1']:.3f}"]),
            "",
            (
                f"실행 파일 {len(answers_by_file)}개. 골드 라벨 오류 후보는 각 실행의 "
                "'의심' 목록과 교집합을 보세요."
            ),
            "",
        )
    )
    return lines


def main(
    *,
    manifest: Annotated[
        Path, typer.Option(help="Probe manifest from build_probes.py")
    ] = DEFAULT_MANIFEST,
    llm: Annotated[
        list[Path] | None, typer.Option(help="LLM answer file or directory (repeatable)")
    ] = None,
    human: Annotated[Path | None, typer.Option(help="CSV exported by human_probe.html")] = None,
    merge_batches: Annotated[
        bool,
        typer.Option(help="Merge batch_<n>_<run> files into one judge per model directory and run"),
    ] = False,
    output: Annotated[Path, typer.Option(help="Markdown report path")] = DEFAULT_OUTPUT,
) -> None:
    """Grade whichever probe results are supplied and write one report pair."""
    if not manifest.exists():
        typer.echo(f"Manifest not found: {manifest}. Run build_probes.py first.", err=True)
        raise typer.Exit(code=1)
    items = load_manifest(manifest)
    fill_texts(items, manifest.parent / "llm" / "items_all.tsv")

    files = result_files(llm or [])
    sets = merge_result_sets(files) if merge_batches else {str(path): [path] for path in files}
    llm_reports: list[LlmReport] = []
    answers_by_file: dict[str, dict[str, Judgement]] = {}
    head = [
        "# 데이터셋 프로브 채점",
        "",
        f"- 실행: {datetime.now(UTC).isoformat()}",
        f"- 기준: 검증 {len(items)}행, 골드 라벨은 누리과정 5영역",
        f"- 병합: {'모델·실행 단위' if merge_batches else '파일 단위'}",
        "",
    ]
    lines: list[str] = []
    for name, group in sets.items():
        answers, issues = read_group(group)
        report = llm_report(Path(name), answers, items)
        report["issues"] = issues
        answers_by_file[name] = answers
        llm_reports.append(report)
        lines.extend(render_llm(report))
        if issues:
            lines.extend(
                (
                    f"형식 경고 {len(issues)}건 (앞 10건): " + " / ".join(issues[:10]),
                    "",
                )
            )

    human_result: HumanReport | None = None
    if human is not None and human.exists():
        human_result = human_report(human)
        lines.extend(render_human(human_result))

    judges: dict[str, dict[str, str]] = {
        name: {identifier: answer.top1 for identifier, answer in answers.items()}
        for name, answers in answers_by_file.items()
    }
    judges["tf-idf"] = {identifier: item.model_top1 for identifier, item in items.items()}
    if human is not None and human.exists():
        judges["human"] = human_predictions(human)
    if len(judges) > 1:
        lines.extend(("## 판단 간 합의", "", *render_agreement(judges), ""))
    if answers_by_file:
        lines.extend(render_ensemble(answers_by_file, items))
        lines.extend(render_all_wrong(judges, items))

    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        "\n".join([*head, *render_summary(llm_reports, human_result), *lines]) + "\n",
        encoding="utf-8",
    )
    _ = output.with_suffix(".json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "manifest": str(manifest),
                "llm": llm_reports,
                "human": human_result,
                "agreement": judges,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Wrote {output} and {output.with_suffix('.json')}")
    if not files and human is None:
        typer.echo("No results supplied; the report only contains the header.")


if __name__ == "__main__":
    typer.run(main)
