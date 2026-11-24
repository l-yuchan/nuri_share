# 생성 산출물

이 디렉터리는 실험이 만드는 중간 산출물 자리다. 대부분은 `.gitignore`로 제외되어
있고, 저장소에는 `corpus/`만 들어 있다.

| 경로 | 포함 | 재생성 |
|---|---|---|
| `corpus/sentences.txt`, `corpus/manifest.json` | 예 (27.9 MB) | `uv run --extra semantic python prepare_corpus.py` (`data/TL1.zip` 필요) |
| `fasttext/cbow.bin`, `fasttext/*.vec`, `fasttext/supervised_*.bin` | 아니오 (약 480 MB) | `uv run --extra semantic python fasttext_experiment.py` |
| `encoder/embeddings.npz` | 아니오 (1.2 MB) | `uv run --extra semantic python encoder_experiment.py` |
| `ablation/*.vec` | 아니오 (약 160 MB) | `uv run --extra semantic python ablation_experiment.py` |
| `cc.ko.300.bin` | 아니오 (7.2 GB) | 외부 배포본. `RUNNING.md`의 "외부 자산" 참고 |
| `typechecker-repair/` | 아니오 | 과거 타입 검사 도구 복구용 사본이라 export에서 제외 |

각 산출물의 해시와 버전은 해당 `results_*/metadata.json`에 기록된다.
