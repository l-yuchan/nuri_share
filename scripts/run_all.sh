#!/usr/bin/env bash
# Local pipeline: every experiment that does not need the 7.2 GB cc.ko.300 model.
# Individual steps overwrite their results_* directory.
#
#   bash scripts/run_all.sh            # full local run (tens of minutes)
#   STEPS="tfidf" bash scripts/run_all.sh
#
# Observation-to-indicator linking is separate: scripts/run_indicator_experiments.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-semantic}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
RUN=(uv run --extra semantic python)
STEPS="${STEPS:-tfidf fasttext encoder ablation probes hierarchical profile}"

has() { case " $STEPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if has tfidf; then
  echo "== TF-IDF baseline =="
  "${RUN[@]}" experiment.py
fi

if has fasttext; then
  echo "== fastText =="
  # Needs artifacts/corpus/sentences.txt (shipped) or data/TL1.zip.
  "${RUN[@]}" fasttext_experiment.py
fi

if has encoder; then
  echo "== frozen multilingual E5 =="
  # Needs network on first run to fetch intfloat/multilingual-e5-small.
  "${RUN[@]}" encoder_experiment.py
fi

if has ablation; then
  echo "== initialization ablation =="
  # Needs artifacts/fasttext/cbow.bin from the fastText step.
  "${RUN[@]}" ablation_experiment.py
  "${RUN[@]}" analyze_ablation.py
fi

if has probes; then
  echo "== dataset probes =="
  "${RUN[@]}" build_probes.py
  # Grading needs saved LLM answers and the human CSV; see probes/README.md.
  if [ -d probes/llm/results/gpt ] && [ -f probes/human_probe_results.csv ]; then
    "${RUN[@]}" analyze_probes.py \
      --llm probes/llm/results/gpt \
      --llm probes/llm/results/gemini \
      --merge-batches \
      --human probes/human_probe_results.csv
  else
    echo "skip probe grading: saved judgments not present"
  fi
fi

if has hierarchical; then
  echo "== area + level heads =="
  "${RUN[@]}" hierarchical_experiment.py
fi

if has profile; then
  echo "== child profile =="
  "${RUN[@]}" child_profile_experiment.py --children 500
fi

echo "RUN_ALL_DONE"
