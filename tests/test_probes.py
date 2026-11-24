# Copyright 2026 Nuri ML contributors
from pathlib import Path

from analyze_probes import (
    Item,
    Judgement,
    describe,
    human_report,
    llm_report,
    macro_f1,
    parse_llm_output,
)
from build_probes import MAX_PER_GROUP, Scored, select_human_items
from nuri_data import LABELS


def make_scored(
    identifier: str,
    group: str,
    label: str,
    top1: str,
    *,
    margin: float = 1.0,
) -> Scored:
    return Scored(
        identifier=identifier,
        group=group,
        item_key=f"{group}|{identifier}",
        label=label,
        text=f"text {identifier}",
        source="kicce",
        context="놀이",
        age="3세",
        top1=top1,
        top2=top1,
        top_score=2.0,
        margin=margin,
        gold_score=2.0,
    )


def test_parse_accepts_pipe_comma_and_table_forms() -> None:
    payload = """```text
id | 사회관계 신체운동·건강 예술경험 의사소통 자연탐구 | 1순위 | 사유
|---|---|---|---|---|
K013 | 4 5 1 2 1 | 신체운동·건강 | 가위질 시도
K014,2 2 3 2 2,의사소통,친구와 대화
K015\t1 1 1 5 1\t의사소통\t말로 표현
```
이상입니다.
"""
    answers, issues = parse_llm_output(payload)

    assert set(answers) == {"K013", "K014", "K015"}
    assert answers["K013"].top1 == "신체운동·건강"
    assert answers["K013"].ratings == (4, 5, 1, 2, 1)
    assert answers["K015"].reason == "말로 표현"
    assert any("top1 differs" in issue for issue in issues)
    assert any("이상입니다" in issue for issue in issues)


def test_parse_rejects_wrong_score_count_without_dropping_other_rows() -> None:
    payload = "K013 | 4 5 1 2 | 신체운동·건강 | 네 개만\nK014 | 1 1 1 5 1 | 의사소통 | 정상\n"
    answers, issues = parse_llm_output(payload)

    assert set(answers) == {"K014"}
    assert len(issues) == 1


def test_human_selection_is_deterministic_and_respects_group_cap() -> None:
    rows: list[Scored] = []
    groups = [f"G{index:02d}" for index in range(13)]
    for group_index, group in enumerate(groups):
        for row_index in range(12):
            label = LABELS[row_index % len(LABELS)]
            wrong = row_index < 4
            rows.append(
                make_scored(
                    f"X{group_index:02d}{row_index:02d}",
                    group,
                    label,
                    LABELS[(row_index + 1) % len(LABELS)] if wrong else label,
                    margin=float(row_index),
                )
            )

    first = select_human_items(rows)
    second = select_human_items(rows)

    assert [item.scored.identifier for item in first] == [item.scored.identifier for item in second]
    assert len(first) > 0
    for group in {item.scored.group for item in first}:
        assert sum(item.scored.group == group for item in first) <= MAX_PER_GROUP
    assert {item.stratum for item in first} == {"hard", "control"}


def test_describe_and_macro_f1_use_five_fixed_labels() -> None:
    gold = [LABELS[0], LABELS[1], LABELS[1]]
    predicted = [LABELS[0], LABELS[0], LABELS[1]]
    report = describe(gold, predicted, ["g1", "g2", "g2"], ["kicce"] * 3)

    assert report["accuracy"] == 2 / 3
    assert report["groups"] == 2
    assert report["by_class"][LABELS[2]]["n"] == 0
    assert 0 < macro_f1(gold, predicted) < 1
    assert report["confusion"][LABELS[1]][LABELS[0]] == 1


def test_llm_report_flags_items_where_gold_scores_low() -> None:
    items = {
        "K001": Item("K001", LABELS[0], "kicce", "g1", LABELS[0]),
        "K002": Item("K002", LABELS[1], "kicce", "g1", LABELS[1]),
    }
    answers = {
        "K001": Judgement("K001", (5, 1, 1, 1, 1), LABELS[0], "ok"),
        "K002": Judgement("K002", (5, 1, 1, 4, 1), LABELS[0], "wrong"),
    }

    report = llm_report(Path("answers.txt"), answers, items)

    assert report["accuracy"] == 0.5
    assert report["parsed"] == 2
    assert report["missing"] == []
    assert [row["id"] for row in report["suspects"]] == ["K002"]
    assert report["suspects"][0]["gold_rating"] == 1


def test_human_report_counts_ambiguous_and_top2(tmp_path: Path) -> None:
    path = tmp_path / "human.csv"
    header = (
        "id,source,group_id,stratum,choice1,choice2,ambiguous,hint_used,elapsed_ms,"
        "gold,correct,suspect_label,note\n"
    )
    rows = (
        header
        + f"K001,kicce,g1,hard,{LABELS[0]},{LABELS[1]},no,yes,1200,{LABELS[0]},yes,no,\n"
        + f"K002,kicce,g1,hard,{LABELS[0]},{LABELS[1]},no,no,900,{LABELS[1]},no,yes,애매함\n"
        + "K003,kicce,g1,hard,,,yes,no,4000,자연탐구,no,no,\n"
    )
    _ = path.write_text(rows, encoding="utf-8")

    report = human_report(path)

    assert report["answered"] == 3
    assert report["ambiguous"] == 1
    assert report["hits"] == 1
    assert report["top2_accuracy"] == 1.0
    assert report["suspects"][0]["id"] == "K002"
    assert report["by_hint_use"]["yes"]["n"] == 1
