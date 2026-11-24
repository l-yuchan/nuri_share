# 데이터

이 디렉터리의 `train.csv` / `val.csv`가 모든 실험이 읽는 고정 분할이다.
`nuri_data.DEFAULT_DATA_DIR`이 이 폴더를 가리키며, 다른 위치를 쓰려면 각
스크립트의 `--data-dir` 옵션으로 바꾼다.

| 파일 | 행 수 | 설명 |
|---|---:|---|
| `train.csv` | 666 | 학습 |
| `val.csv` | 158 | 검증 |
| `source/2026-11-15_kicce-scale_s01.csv` | — | KICCE 유아관찰척도 원자료 |
| `source/2026-11-08_casebook-pdf_s01.csv` | — | 사례집 원자료 |

- 두 분할은 제공된 그대로이며 다시 섞지 않는다. `group_id`와 정규화 텍스트가
  겹치면 `read_splits`가 실패한다.
- 필수 열: `id, source, session, group_id, label, content_category, item_no,
  item_text, context, level, age, text, label_source, use_default, split, note`.
- KICCE 행에는 지표 정의(`item_text`)와 발달 수준(`level` 1–4)이 있고, 사례집
  행에는 없다.
- 원본 제공 파일 `ACDT_W11_dataset_v2.xlsx`는 재배포 범위를 확인하지 않아
  포함하지 않았다. 위 CSV는 그 파일에서 추출한 것이며, 배포·인용 조건은 원
  자료 제공처를 따른다.

## fastText 사전학습 말뭉치

`prepare_corpus.py`는 `data/` 아래에서 정확히 하나의 `TL1.zip`(AI-Hub 학습
말뭉치)을 찾아 문장을 추출한다. `TL1.zip`은 이 저장소에 포함되어 있지 않다.
추출 결과 `artifacts/corpus/sentences.txt`(147,017문장, 27.9 MB)와
`manifest.json`은 포함되어 있으므로, 말뭉치 원본 없이도 fastText 실험을
재현할 수 있다. 원본부터 다시 만들려면 `TL1.zip`을 `data/`에 두고
`uv run --extra semantic python prepare_corpus.py`를 실행한다.
