# Copyright 2026 Nuri ML contributors
"""Build two dataset-quality probes for the Nuri validation split.

The probes test whether the supplied labels are recoverable by a strong external
reader (an LLM given only the five-area definitions) and by a human given the
same text. They never expose gold labels to either judge before judging.

Outputs go to ``probes/``: a prompt pack (one Markdown file per batch), a
self-contained HTML page, and a manifest holding the model scores and the
selection rule so the returned answers can be graded later.

Run with ``uv run python build_probes.py``.
"""

import base64
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import numpy as np
import typer
from sklearn.svm import LinearSVC

from experiment import make_vectorizer
from nuri_data import DEFAULT_DATA_DIR, LABELS, Example, normalize_syllables, read_splits

if TYPE_CHECKING:
    from numpy.typing import NDArray

SEED: Final = 42
BEST_C: Final = 10.0
PROBE_DIR: Final = Path("probes")
TEMPLATE_PATH: Final = Path("templates") / "human_probe.html"
LLM_BATCH_SIZE: Final = 40
HUMAN_HARD_TARGET: Final = 40
HUMAN_CONTROL_TARGET: Final = 16
MAX_PER_GROUP: Final = 5
MAX_PER_ITEM: Final = 2
CONTROL_MARGIN_QUANTILE: Final = 0.75

AREA_SUMMARY: Final = (
    (
        "사회관계",
        "나를 알고 존중하기 · 더불어 생활하기 · 사회에 관심 가지기",
        (
            "자기 인식과 조절, 또래와 교사와의 관계, 협력과 갈등 해결, 규칙과 예절, "
            "가족·이웃·지역사회·문화와 다양성에 대한 관심."
        ),
    ),
    (
        "신체운동·건강",
        "신체활동 즐기기 · 건강하게 생활하기 · 안전하게 생활하기",
        (
            "대근육과 소근육 움직임, 균형과 이동, 도구 조작과 신체 인식, 식사·수면·위생, "
            "질병 예방과 바른 자세, 교통·생활 안전과 위험 대처."
        ),
    ),
    (
        "예술경험",
        "아름다움 찾아보기 · 창의적으로 표현하기 · 예술 감상하기",
        (
            "자연과 사물, 소리와 움직임에서 아름다움 느끼기, 미술·음악·신체 표현·극놀이로 "
            "창의적으로 표현하기, 작품과 공연을 감상하고 반응하기."
        ),
    ),
    (
        "의사소통",
        "듣기와 말하기 · 읽기와 쓰기에 관심 가지기 · 책과 이야기 즐기기",
        (
            "주의 깊게 듣기, 상황에 맞게 말하기, 어휘와 대화 주고받기, 글자와 상징 인식, "
            "읽기와 쓰기 시도, 그림책과 이야기를 듣고 이해하고 꾸미기."
        ),
    ),
    (
        "자연탐구",
        "탐구과정 즐기기 · 생활 속에서 탐구하기 · 자연과 더불어 살기",
        (
            "관찰·질문·예측·실험·비교, 수·공간·도형·측정·분류·규칙, 물질·생명·날씨·계절, "
            "자연 현상과 생명체에 대한 관심과 존중, 환경 보호."
        ),
    ),
)

BOUNDARY_RULES: Final = (
    (
        "판단 기준은 '유아가 무엇을 하려 했는가'(행동의 목적)이다. "
        "같은 장면도 목적이 다르면 영역이 달라진다."
    ),
    (
        "관계·감정·규칙·예절이 초점이면 사회관계. "
        "말하고 듣고 읽고 쓰는 언어 행위 자체가 초점이면 의사소통."
    ),
    (
        "몸을 움직이거나 손으로 도구를 조작하는 것이 초점이면 신체운동·건강. "
        "재료와 자연을 관찰해 알아내는 것이 초점이면 자연탐구."
    ),
    (
        "아름다움을 느끼거나 표현하고 감상하는 것이 초점이면 예술경험. "
        "자연물을 보며 심미감을 느낀 것이 초점이면 예술, "
        "차이를 발견하고 관찰한 것이 초점이면 자연탐구."
    ),
    (
        "교사의 지시를 따랐다는 사실만으로는 영역을 정할 수 없다. "
        "유아가 실제로 무엇을 배우고 표현했는지 본다."
    ),
    (
        "말로 감정을 표현한 장면은 의사소통일 수 있고, "
        "관계 회복과 배려가 초점이면 사회관계다. "
        "이 경계가 원자료에서 가장 자주 혼동된다."
    ),
)


@dataclass(frozen=True, slots=True)
class Scored:
    """One validation row with the fitted TF-IDF model's ranking."""

    identifier: str
    group: str
    item_key: str
    label: str
    text: str
    source: str
    context: str
    age: str
    top1: str
    top2: str
    top_score: float
    margin: float
    gold_score: float


@dataclass(frozen=True, slots=True)
class HumanItem:
    """A row selected for the human probe, with why it was selected."""

    scored: Scored
    stratum: str


def flatten(text: str) -> str:
    """Collapse whitespace so one row stays on one line in every export."""
    return " ".join(text.split())


def encode_gold(label: str) -> str:
    """Hide the gold label from casual inspection of the HTML source."""
    return base64.b64encode(label.encode("utf-8")).decode("ascii")


def read_metadata(data_dir: Path, names: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """Read context and age columns that ``Example`` deliberately omits."""
    metadata: dict[str, tuple[str, str]] = {}
    for name in names:
        with (data_dir / name).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                metadata[row["id"]] = (row.get("context", ""), row.get("age", ""))
    return metadata


def score_validation(
    train: tuple[Example, ...],
    val: tuple[Example, ...],
    metadata: dict[str, tuple[str, str]],
) -> list[Scored]:
    """Fit the best TF-IDF setting and keep its ranking and margin per row."""
    vectorizer = make_vectorizer(normalize_syllables, (2, 5))
    x_train = vectorizer.fit_transform([row.text for row in train])
    classifier = LinearSVC(C=BEST_C, dual=True, max_iter=10000, random_state=SEED)
    _ = classifier.fit(x_train, [row.label for row in train])
    scores: NDArray[np.float64] = classifier.decision_function(
        vectorizer.transform([row.text for row in val])
    )
    classes: NDArray[np.str_] = classifier.classes_
    indices: NDArray[np.intp] = np.argsort(scores, axis=1, kind="stable")[:, ::-1]
    scored: list[Scored] = []
    for position, row in enumerate(val):
        ranking = [str(classes[index]) for index in indices[position]]
        context, age = metadata.get(row.identifier, ("", ""))
        top_score = float(scores[position, indices[position, 0]])
        scored.append(
            Scored(
                identifier=row.identifier,
                group=row.group,
                item_key=f"{row.group}|{row.text[:20]}",
                label=row.label,
                text=flatten(row.text),
                source=row.source,
                context=context,
                age=age,
                top1=ranking[0],
                top2=ranking[1],
                top_score=top_score,
                margin=top_score - float(scores[position, indices[position, 1]]),
                gold_score=float(scores[position, list(classes).index(row.label)]),
            )
        )
    return scored


class SelectionBudget:
    """Enforce group and near-duplicate caps while filling the human probe."""

    def __init__(self) -> None:
        """Start with no group, item or row selected."""
        self.groups: Counter[str] = Counter()
        self.items: Counter[str] = Counter()
        self.chosen: set[str] = set()

    def take(self, candidate: Scored, *, per_item: bool = True) -> bool:
        """Add the candidate when both caps allow it."""
        if candidate.identifier in self.chosen:
            return False
        if self.groups[candidate.group] >= MAX_PER_GROUP:
            return False
        if per_item and self.items[candidate.item_key] >= MAX_PER_ITEM:
            return False
        self.chosen.add(candidate.identifier)
        self.groups[candidate.group] += 1
        self.items[candidate.item_key] += 1
        return True


def take_hard(scored: list[Scored], budget: SelectionBudget, chosen: dict[str, str]) -> None:
    """Take the worst model errors first, then the lowest-margin correct rows."""
    wrong = sorted(
        (row for row in scored if row.top1 != row.label),
        key=lambda row: (row.top2 == row.label, row.gold_score - row.top_score, row.identifier),
    )
    correct = [row for row in scored if row.top1 == row.label]
    margins = np.array([row.margin for row in correct])
    low_cut = float(np.quantile(margins, 1 - CONTROL_MARGIN_QUANTILE))
    hard_pool = wrong + sorted(
        (row for row in correct if row.margin <= low_cut),
        key=lambda row: (row.margin, row.identifier),
    )
    for row in hard_pool:
        if len(chosen) >= HUMAN_HARD_TARGET:
            break
        if budget.take(row):
            chosen[row.identifier] = "hard"


def take_control(scored: list[Scored], budget: SelectionBudget, chosen: dict[str, str]) -> None:
    """Take high-margin correct rows, balanced across labels and cap-aware."""
    correct = [row for row in scored if row.top1 == row.label and row.identifier not in chosen]
    margins = np.array([row.margin for row in correct])
    high_cut = float(np.quantile(margins, CONTROL_MARGIN_QUANTILE))
    control_pool = sorted(correct, key=lambda row: (-row.margin, row.identifier))
    preferred = [row for row in control_pool if row.margin >= high_cut]
    fallback = [row for row in control_pool if row.margin < high_cut]
    by_label: dict[str, list[Scored]] = {label: [] for label in LABELS}
    for row in preferred + fallback:
        by_label[row.label].append(row)
    control_count = 0
    while control_count < HUMAN_CONTROL_TARGET:
        progressed = False
        for label in LABELS:
            if control_count >= HUMAN_CONTROL_TARGET:
                break
            while by_label[label]:
                candidate = by_label[label].pop(0)
                if not budget.take(candidate):
                    continue
                chosen[candidate.identifier] = "control"
                control_count += 1
                progressed = True
                break
        if not progressed:
            break


def select_human_items(scored: list[Scored]) -> list[HumanItem]:
    """Pick hard rows plus a high-margin control set, capped per group and item."""
    budget = SelectionBudget()
    chosen: dict[str, str] = {}
    take_hard(scored, budget, chosen)
    take_control(scored, budget, chosen)

    selected = [
        HumanItem(scored=row, stratum=chosen[row.identifier])
        for row in scored
        if row.identifier in chosen
    ]
    order = np.random.default_rng(SEED).permutation(len(selected))
    return [selected[int(index)] for index in order]


def definitions_text() -> str:
    """Render the five-area definitions as plain text for the LLM prompt."""
    lines: list[str] = []
    for index, (name, subareas, description) in enumerate(AREA_SUMMARY, start=1):
        lines.append(f"{index}) {name} — {subareas}")
        lines.append(f"   {description}")
    return "\n".join(lines)


def definitions_html() -> str:
    """Render the same definitions as safe HTML for the human page."""
    parts: list[str] = ['<ul style="margin:0 0 10px;padding-left:18px">']
    for name, subareas, description in AREA_SUMMARY:
        parts.append(f"<li><b>{name}</b> — {subareas}<br>{description}</li>")
    parts.append(
        '</ul><p style="margin:0 0 6px"><b>경계 규칙</b></p><ul style="margin:0;padding-left:18px">'
    )
    parts.extend(f"<li>{rule}</li>" for rule in BOUNDARY_RULES)
    parts.append("</ul>")
    return "".join(parts)


def render_prompt(number: int, total: int, items: list[Scored]) -> str:
    """Render one self-contained batch: task, definitions, output contract, items."""
    area_order = " ".join(LABELS)
    blocks = "\n".join(f"{row.identifier}\t{row.text}" for row in items)
    return f"""# 누리과정 5개 영역 분류 ({number}/{total} 배치)

당신은 만 3~5세 유아교육 현장의 관찰 기록을 누리과정 5개 영역으로 분류하는 평가자입니다.
`[항목]` 블록의 기록을 하나씩 읽고, 다섯 영역 각각에 대해 "이 기록이 그 영역을 보여주는가"를
1~5점으로 평가한 뒤 가장 적합한 영역 하나를 고르세요.

## 지켜야 할 것
- 기록에 없는 정보를 상상하지 말고, 기록된 행동과 그 목적만 근거로 삼으세요.
- 기록이 짧거나 애매해도 다섯 점수를 모두 채우세요. "모르겠음"은 허용되지 않습니다.
- 인터넷 검색이나 외부 자료를 쓰지 말고 아래 정의만 사용하세요.
- 영역은 반드시 하나만 고릅니다.
- 이 배치에 있는 항목만 처리하고, 다음 배치를 예상해 이어 쓰지 마세요.

## 영역 정의
{definitions_text()}

## 경계 규칙
{chr(10).join(f"- {rule}" for rule in BOUNDARY_RULES)}

## 출력 형식
항목마다 정확히 한 줄씩, 아래 형식으로만 출력하세요. 인사말, 설명, 코드블록 표시는 쓰지 마세요.

id | {area_order} | 1순위 | 사유

- 점수는 1(전혀 아님)~5(매우 적합) 정수 다섯 개를 위 순서대로 씁니다.
- 1순위는 점수가 가장 높은 영역입니다.
- 사유는 30자 이내로, `|` 문자 없이 씁니다.
- 형식 예시(아래 `E001`은 실제 항목이 아닙니다):
  `E001 | 4 5 1 2 1 | 신체운동·건강 | 미끄럼틀 오르기, 균형과 다리 힘`

## [항목]
id<TAB>기록
{blocks}
"""


def render_llm_readme(batches: int) -> str:
    """Explain how to run the web batches and where to save the answers."""
    return f"""# LLM 판단 테스트 (배치 프롬프트)

`batch_01.md` ~ `batch_{batches:02d}.md` 각각이 독립된 하나의 프롬프트입니다.
파일 전체를 복사해 **새 대화**에 붙여넣고, 모델이 출력한 답변 전체를 저장하세요.

## 절차
1. 모델별 폴더를 만듭니다: `probes/llm/results/gpt/`, `claude/`, `gemini/` 등.
2. 각 배치를 새 대화에서 실행하고, 출력 전체를 `batch_01.txt` 처럼 저장합니다.
3. 같은 배치를 새 대화에서 한 번 더 실행하면 자기 일관성을 볼 수 있습니다.
4. 두 개 이상 모델에서 실행하면 모델 간 합의도(사람이 도달 가능한 상한의 추정)를 볼 수 있습니다.

권장 모델: gpt, claude, gemini. 웹 UI에서 파일 첨부가 되면 배치 파일을 그대로 올려도 됩니다.

## 주의
- 출력이 중간에 잘리면 "이어서"라고 요청하지 말고, 새 대화에서 같은 배치를 다시 실행하세요.
- 모델이 표를 Markdown으로 감싸거나 일부 항목을 건너뛰어도 됩니다.
  채점 스크립트가 누락과 형식 오류를 보고합니다.
- 정답 라벨은 프롬프트에 포함되어 있지 않습니다. 모델에게 정답을 알려주거나 검색을 시키지 마세요.

## 채점
```sh
uv run python analyze_probes.py --llm probes/llm/results/gpt
```
결과는 `probes/analysis.md`와 `probes/analysis.json`에 기록됩니다.
"""


def build_payload(items: list[HumanItem], session: str) -> dict[str, object]:
    """Serialize the human probe items; gold labels stay base64-encoded."""
    return {
        "session": session,
        "items": [
            {
                "id": item.scored.identifier,
                "t": item.scored.text,
                "c": item.scored.context,
                "a": item.scored.age,
                "s": item.scored.source,
                "gid": item.scored.group,
                "g": encode_gold(item.scored.label),
                "k": item.stratum,
            }
            for item in items
        ],
    }


def write_manifest(
    path: Path,
    scored: list[Scored],
    selected: list[HumanItem],
    data_dir: Path,
    llm_order: list[str],
) -> None:
    """Record the model ranking and selection so grading needs no refit."""
    hashes = {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("train.csv", "val.csv")
    }
    payload = {
        "created_by": "build_probes.py",
        "seed": SEED,
        "model": {"features": "syllable_2-5", "classifier": "linear_svm", "c": BEST_C},
        "data_hashes": hashes,
        "labels": list(LABELS),
        "validation": [
            {
                "id": row.identifier,
                "group": row.group,
                "source": row.source,
                "label": row.label,
                "top1": row.top1,
                "top2": row.top2,
                "margin": round(row.margin, 6),
            }
            for row in scored
        ],
        "human_selection": [
            {"id": item.scored.identifier, "stratum": item.stratum} for item in selected
        ],
        "llm_item_order": llm_order,
    }
    _ = path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_items_tsv(path: Path, scored: list[Scored], order: list[str]) -> None:
    """Write the blind item list in the order the batches use."""
    by_id = {row.identifier: row for row in scored}
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("id", "text"))
        for identifier in order:
            writer.writerow((identifier, by_id[identifier].text))


def main(
    *,
    data_dir: Annotated[
        Path, typer.Option(help="Directory containing train.csv and val.csv")
    ] = DEFAULT_DATA_DIR,
    output_dir: Annotated[Path, typer.Option(help="Directory for probe artifacts")] = PROBE_DIR,
    batch_size: Annotated[int, typer.Option(help="Items per LLM prompt file")] = LLM_BATCH_SIZE,
) -> None:
    """Generate the LLM prompt pack, the human HTML page and the manifest."""
    train, val = read_splits(data_dir)
    metadata = read_metadata(data_dir, ("train.csv", "val.csv"))
    scored = score_validation(train, val, metadata)

    order = [row.identifier for row in scored]
    order = [order[int(index)] for index in np.random.default_rng(SEED).permutation(len(order))]
    by_id = {row.identifier: row for row in scored}
    batches = [
        [by_id[identifier] for identifier in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]

    llm_dir = output_dir / "llm"
    results_dir = llm_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(batches, start=1):
        path = llm_dir / f"batch_{index:02d}.md"
        _ = path.write_text(render_prompt(index, len(batches), batch), encoding="utf-8")
    write_items_tsv(llm_dir / "items_all.tsv", scored, order)
    _ = (llm_dir / "prompt.md").write_text(
        render_prompt(1, len(batches), batches[0]), encoding="utf-8"
    )
    _ = (llm_dir / "README.md").write_text(render_llm_readme(len(batches)), encoding="utf-8")

    selected = select_human_items(scored)
    session = "20261121"
    payload = build_payload(selected, session)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace(
        "__ITEMS_JSON__",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
    )
    html = html.replace("__DEFINITIONS_HTML__", json.dumps(definitions_html(), ensure_ascii=False))
    _ = (output_dir / "human_probe.html").write_text(html, encoding="utf-8")

    write_manifest(output_dir / "manifest.json", scored, selected, data_dir, order)
    strata = Counter(item.stratum for item in selected)
    typer.echo(f"Validation rows: {len(scored)}; LLM batches: {len(batches)}")
    typer.echo(f"Human probe: {len(selected)} items {dict(strata)}")
    typer.echo(f"Wrote {output_dir / 'human_probe.html'} and {llm_dir}")


if __name__ == "__main__":
    typer.run(main)
