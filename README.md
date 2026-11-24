# 누리 관찰 기록 ML 실험 — 인계용 저장소

아동 관찰 기록(한국어 텍스트)을 누리과정 5개 영역과 세부 지표에 연결하고, 지표별
발달 수준과 아동 내 강·약 항목을 산출하는 실험 기록과 코드다. 29조 프로젝트의
`ml/` 실험 폴더를 독립 저장소로 정리한 export 판이며, 통합 시스템을 만드는 팀원이
그대로 실행·검증할 수 있도록 코드, 고정 분할, 결과 CSV, 재현 절차를 함께 담았다.

**상태: 연구·실험 단계.** 모든 수치는 같은 666/158 분할에서 설정 선택까지 수행한
탐색적 검증 점수이며, 독립 테스트 성능이나 서비스 성능이 아니다.

## 5분 요약

```text
관찰 기록 1건
  → (1) 5개 영역 예측            보정 SVM        71.52% (top-2 89.24%)
  → (2) 발달 수준 1–4 예측        다항 로지스틱    56.06% (±1 85.61%, QWK 0.576)
  → (3) 어느 지표의 근거인가      E5 제로샷 검색    Recall@1 24.24% (후보 59개)
  → (4) 아동 내 강·약 top-2      수준 집계        NDCG@2 0.88 / 0.90 (오라클 링크 기준)
```

- **(1) 영역 분류의 천장은 약 69–72%.** 문자 n-gram TF-IDF + 선형 모델이 가장
  좋았고, fastText(62.7%)와 고정 E5 임베딩(52.5%)은 넘지 못했다. 외부 LLM 판단자는
  81–84%이므로 남은 격차는 데이터 오염이 아니라 표현력이다.
- **(2) 수준은 영역과 독립인 별도 축이다.** 영역별 평균 수준이 모두 2.50이고,
  수준을 영역 특징으로 넣으면 이득이 없다. 두 헤드는 병렬로 둔다.
- **(3) 현재 병목은 지표 연결이다.** 학습하지 않은 E5가 유일하게 표면 기준선을
  넘지만 24%다. 학습을 붙이면 미학습 지표 전이가 5%대로 무너진다.
- **(4) 연결이 완벽하면 강·약 순위는 작동한다.** 그러나 예측 링크(24%)를 쓰면
  band@2가 강한 항목 0.81 → 0.16–0.27, 약한 항목 0.81 → 0.28–0.39로 떨어진다.
  다음 작업의 우선순위가 여기 있다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `*.py` | 실험 8종 + 공용 모듈(`nuri_data.py`, `indicator_task.py`) |
| `tests/` | pytest 49건 (데이터 경계, 지표 계산, 선택 규칙, 부트스트랩) |
| `data/` | 고정 분할 `train.csv`(666) / `val.csv`(158) + 원자료 CSV 2종 |
| `results*/` | 실험별 지표·예측·혼동행렬·메타데이터(해시·버전) |
| `artifacts/corpus/` | fastText 사전학습 말뭉치 147,017문장 + manifest |
| `probes/` | LLM·사람 블라인드 판정 프롬프트, 오프라인 HTML, 채점 결과 |
| `scripts/` | `verify.sh`, `run_all.sh`, `run_indicator_experiments.sh` |
| 문서 | `RUNNING.md`, `EXPERIMENTS.md`, `PROJECT_STATUS.md`, 실험별 상세 5종 |

## 빠른 시작

```sh
export UV_PROJECT_ENVIRONMENT=.venv-semantic    # PowerShell: $env:UV_PROJECT_ENVIRONMENT=".venv-semantic"
uv sync --locked --extra semantic
bash scripts/verify.sh                          # lint + 49 tests + TF-IDF 재현
```

`verify.sh`는 마지막에 문서에 적힌 best 행(`syllable_2-5 / linear_svm / C=10`,
accuracy 0.689873)을 그대로 재생성해 출력한다. 개별 실험 명령, 실행 시간,
필요한 외부 자산은 [RUNNING.md](RUNNING.md)에 정리했다.

## 데이터와 외부 자산

- **포함:** 고정 분할 CSV, 원자료 CSV 2종, fastText 말뭉치 텍스트(27.9 MB),
  모든 결과 CSV/JSON, 프로브 프롬프트와 판정 결과.
- **미포함(재생성 또는 외부 필요):** fastText `.bin`·`.vec`(약 480 MB),
  E5 임베딩 캐시, `cc.ko.300.bin`(7.2 GB), `data/TL1.zip`(AI-Hub 원본).
- `cc.ko.300.bin`은 로컬에 없고 `mini` 호스트에서만 사용했다. 배포 URL은 아직
  확인하지 않았으며 **미결 항목**이다. 파일 식별자는 sha256
  `a021ebbd…58acd7`, 7,243,669,409 bytes, 300차원이다.
- 원본 제공 파일 `ACDT_W11_dataset_v2.xlsx`는 재배포 범위가 확인되지 않아 넣지
  않았다. 데이터 이용 조건은 `data/README.md`를 따른다.

## 문서 읽는 순서

1. 이 문서(현재 상태 요약)
2. [PROJECT_STATUS.md](PROJECT_STATUS.md) — 팀 맥락, 결정된 것/미정인 것
3. [EXPERIMENTS.md](EXPERIMENTS.md) — 지표 정의, 실험 3.1–3.8, 종합 결론, 다음 실험
4. [RUNNING.md](RUNNING.md) — 환경, 실행, 외부 자산, 문제 해결
5. 상세: [ABLATION.md](ABLATION.md), [HIERARCHICAL.md](HIERARCHICAL.md),
   [CHILD_PROFILE.md](CHILD_PROFILE.md), [INDICATOR_LINKING.md](INDICATOR_LINKING.md),
   [MODEL_DIRECTION_PROPOSAL.md](MODEL_DIRECTION_PROPOSAL.md), [probes/README.md](probes/README.md)

## 이 저장소가 주장하지 않는 것

- 독립 테스트 성능, 전체 말뭉치 성능, 음성 전사 오류에 대한 견고성.
- 실제 아동 데이터로 검증된 프로필. 아동 프로필 실험은 검증 지표 11개로 만든
  **합성 아동**이며, 같은 지표의 관찰은 같은 수준에서 뽑았다는 가정을 둔다.
- 활동 추천, LED 출력, 온디바이스 추론. 이 저장소는 분류·연결·수준 추정까지만
  다룬다.
