#!/usr/bin/env bash
# Observation-to-indicator linking: needs cc.ko.300.bin (7.2 GB) and torch.
# Intended to run on the `mini` host; it also works anywhere the model exists.
#
#   NURI_CC=/path/to/cc.ko.300.bin bash scripts/run_indicator_experiments.sh
#
# Env overrides: NURI_CC, PYTHON (defaults to `uv run --extra semantic python`),
# NURI_EPOCHS, NURI_THREADS, OMP_NUM_THREADS.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CC="${NURI_CC:-$ROOT/artifacts/cc.ko.300.bin}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NURI_THREADS="${NURI_THREADS:-$OMP_NUM_THREADS}"
export NURI_EPOCHS="${NURI_EPOCHS:-80}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-semantic}"

run() {
  if [ -n "${PYTHON:-}" ]; then
    "$PYTHON" "$@"
  else
    uv run --extra semantic python "$@"
  fi
}

if [ ! -f "$CC" ]; then
  echo "missing cc vectors: $CC" >&2
  echo "set NURI_CC to the 7.2 GB cc.ko.300.bin path" >&2
  exit 2
fi

run indicator_linking_experiment.py \
  --data-dir "$ROOT/data" \
  --cc-path "$CC" \
  --output-dir "$ROOT/results_indicator_linking"

run indicator_profile_experiment.py \
  --data-dir "$ROOT/data" \
  --predictions-path "$ROOT/results_indicator_linking/val_row_predictions.csv" \
  --output-dir "$ROOT/results_indicator_profile"

run indicator_maxsim_control.py \
  --data-dir "$ROOT/data" \
  --output "$ROOT/results_indicator_linking/maxsim_control.json"

run indicator_report.py --results "$ROOT/results_indicator_linking"

echo RUN_INDICATOR_DONE
