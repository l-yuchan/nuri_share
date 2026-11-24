#!/usr/bin/env bash
# Full check of this repository: lint, format, tests, and one reproduced run.
# Usage: bash scripts/verify.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-semantic}"
RUN=(uv run --extra semantic)

echo "== ruff check =="
"${RUN[@]}" ruff check .

echo "== ruff format --check =="
"${RUN[@]}" ruff format --check .

echo "== pytest =="
"${RUN[@]}" pytest -q

echo "== reproduced TF-IDF run =="
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
"${RUN[@]}" python experiment.py --output-dir "$OUT" >/dev/null
echo "best row:"
head -2 "$OUT/metrics.csv"
echo "VERIFY_OK"
