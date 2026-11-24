# 실행·재현 안내

모든 명령은 저장소 루트에서 실행한다. CPU만 사용하며 GPU는 필요 없다.

## 1. 환경

- Python 3.12 (`.python-version`). Windows fastText wheel이 3.12에 맞춰져 있다.
- [uv](https://docs.astral.sh/uv/) — `uv.lock`으로 의존성을 고정한다.
- 약 2 GB의 디스크 여유(가상환경 1 GB + 생성 산출물). fastText 산출물까지 만들면
  추가로 약 700 MB가 필요하다.

```sh
export UV_PROJECT_ENVIRONMENT=.venv-semantic
uv sync --locked --extra semantic
```

PowerShell에서는 `$env:UV_PROJECT_ENVIRONMENT = ".venv-semantic"`.

**이 환경 변수가 중요하다.** 기본 `.venv`에는 `fasttext`와 `torch`가 없으므로,
변수를 빼고 `uv run pytest`를 실행하면 `test_fasttext_metrics`,
`test_ablation_analysis`, `test_indicator_linking` 세 모듈이 수집 단계에서 실패한다
(`ModuleNotFoundError`). `scripts/verify.sh`는 이 변수를 스스로 설정한다.

`semantic` extra는 fastText, Pydantic, sentence-transformers와 **CPU 전용** PyTorch를
설치한다(`tool.uv.sources`에서 pytorch-cpu 인덱스를 고정).

## 2. 검증

```sh
bash scripts/verify.sh
```

수행 내용: `ruff check`, `ruff format --check`, `pytest -q`(49건), 그리고 임시 폴더에
TF-IDF 24개 설정 재실행 후 best 행 출력. 정상 종료 시 `VERIFY_OK`가 찍힌다.
기준 결과는 `results/metrics.csv` 첫 행과 같아야 한다.

## 3. 실험별 실행

| 명령 | 출력 | 필요 자산 | 대략 시간(로컬 CPU) |
|---|---|---|---|
| `uv run python experiment.py` | `results/` | 없음 | 1분 이내 |
| `uv run --extra semantic python prepare_corpus.py` | `artifacts/corpus/` | `data/TL1.zip` | 1–2분 |
| `uv run --extra semantic python fasttext_experiment.py` | `results_fasttext/`, `artifacts/fasttext/` | corpus 문장 | CBOW 131초 + 10설정 수 분 |
| `uv run --extra semantic python encoder_experiment.py` | `results_encoder/` | E5 가중치(자동 다운로드) | 인코딩 44초 |
| `uv run --extra semantic python ablation_experiment.py` | `results_ablation/`, `artifacts/ablation/` | `artifacts/fasttext/cbow.bin` | 수 분 |
| `uv run python analyze_ablation.py` | `results_ablation/` | 위 출력 | 수십 초 |
| `uv run python build_probes.py` | `probes/llm/`, `probes/human_probe.html` | `results/` | 수십 초 |
| `uv run python analyze_probes.py --llm … --llm … --merge-batches --human probes/human_probe_results.csv` | `probes/analysis.*` | 저장된 LLM 답변 + 사람 CSV | 수십 초 |
| `uv run python hierarchical_experiment.py` | `results_hierarchical/` | 없음 | 1–2분 |
| `uv run python child_profile_experiment.py --children 500` | `results_child_profile/` | 없음 | 1–2분 |
| `bash scripts/run_indicator_experiments.sh` | `results_indicator_linking/`, `results_indicator_profile/` | `cc.ko.300.bin` | mini에서 수십 분 |

일괄 실행: `bash scripts/run_all.sh`(cc.ko.300 불필요 단계 전체),
`STEPS="tfidf hierarchical" bash scripts/run_all.sh`(선택 실행). 프로브 채점은
저장된 판정이 없으면 자동으로 건너뛴다.

## 4. 데이터 배치

```text
data/
  train.csv   666행   ← 모든 실험의 기본 입력
  val.csv     158행
  source/     원자료 CSV 2종(참고용, 코드가 읽지 않음)
  TL1.zip     (선택) corpus 재생성용. 저장소에 없음
```

`nuri_data.DEFAULT_DATA_DIR`은 `data/`다. 다른 위치를 쓰려면 `--data-dir`로 지정한다.
분할은 다시 섞지 않으며, `group_id` 또는 정규화 텍스트가 학습/검증 사이에서 겹치면
`DatasetError`로 중단한다.

## 5. 외부 자산

| 자산 | 크기 | 이 저장소 | 비고 |
|---|---:|---|---|
| `intfloat/multilingual-e5-small` | 470 MB | 미포함 | `encoder_experiment.py`가 Hugging Face에서 자동 다운로드. 리비전 `614241f622f53c4eeff9890bdc4f31cfecc418b3` 고정 |
| `cc.ko.300.bin` (fastText 한국어 위키 300차원) | 7.2 GB | 미포함 | sha256 `a021ebbd5521ca4b3b33425fc25dacd60e4a795041d6f785997800d32a58acd7`, 300차원. **배포 URL 미확인(미결)** |
| `TL1.zip` (AI-Hub 학습 말뭉치) | 수 GB | 미포함 | `prepare_corpus.py` 입력. 추출 결과는 `artifacts/corpus/`에 포함 |
| fastText 학습 산출물 | 약 480 MB | 미포함 | `fasttext_experiment.py`로 재생성 |

로컬에서 cc.ko.300을 쓰지 않으면, 연결 실험(`results_indicator_linking/`)은 기존
결과 CSV를 읽는 것으로 대체한다. `indicator_report.py`는 결과 디렉터리만 있으면
다시 표를 만든다.

## 6. mini 호스트에서 연결 실험 실행

```sh
# 로컬에서 코드와 데이터 전송 (경로는 환경에 맞게)
rsync -av --exclude .venv --exclude .venv-semantic --exclude artifacts \
  ./ mini:~/nuri-exp/nuri-ml/

ssh mini
cd ~/nuri-exp/nuri-ml
uv sync --locked --extra semantic
NURI_CC=~/nuri-exp/artifacts/cc.ko.300.bin bash scripts/run_indicator_experiments.sh
```

- 스크립트는 `NURI_CC`, `PYTHON`, `NURI_EPOCHS`, `NURI_THREADS`, `OMP_NUM_THREADS`를
  환경 변수로 받는다. `PYTHON`을 주지 않으면 `uv run --extra semantic python`을 쓴다.
- 결과는 저장소 안 `results_indicator_linking/`, `results_indicator_profile/`에
  바로 쓰인다. 로컬로 되가져올 때 기존 파일과 diff로 비교한다.
- `cc.ko.300.bin`이 없다면 스크립트가 2번 종료 코드로 멈추고 경로를 알려준다.

## 7. 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| `ModuleNotFoundError: fasttext` / `torch` | `UV_PROJECT_ENVIRONMENT` 미설정 또는 `uv sync` 누락. `.venv-semantic`을 쓰고 `--extra semantic`을 붙인다 |
| `pytest`가 출력 없이 종료 | 메모리 부족. `OPENBLAS_NUM_THREADS=2`를 앞에 붙인다(스크립트는 기본 설정) |
| `filterwarnings = error`로 테스트 실패 | 경고를 오류로 승격하는 설정이다. 새 경고를 억제하지 말고 원인을 확인한다 |
| fastText 설치 실패 | Python 3.12가 아닌 경우. `.python-version`을 확인한다 |
| 결과 수치가 문서와 다름 | `results_*/metadata.json`의 입력 해시·패키지 버전과 현재 환경을 먼저 비교한다. 스레드 수(4 threads)에 따라 fastText 부동소수 결과가 달라질 수 있다 |
