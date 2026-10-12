#!/usr/bin/env bash
# Regenerates every result table and figure of the article.
#
# Requires (see README.md): the Pyserini prebuilt msmarco-v1-passage index
# and, in data/, msmarco-test2019-queries.tsv, msmarco-test2020-queries.tsv,
# 2019qrels-pass.txt and 2020qrels-pass.txt. With --from-runs, retrieval is
# skipped and only the two qrels files are needed.
#
# Usage (from the repository root, inside the activated environment):
#   bash code/run_all.sh [--from-runs]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Pyserini constructs an OpenAI client at import time and raises without a
# key, although this pipeline never calls OpenAI.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-not-used}"

echo "=== unit tests (synthetic data) ==="
python code/tests_synthetic.py

if [ "${1:-}" != "--from-runs" ]; then
  echo "=== BM25 retrieval ==="
  python code/01_retrieve.py
fi

echo "=== evaluation and published-baseline checkpoint ==="
python code/02_evaluate_checkpoint1.py

echo "=== calibration frames ==="
python code/03_build_calibration_frames.py

echo "=== single-direction calibrators (superseded table) ==="
python code/04_fit_calibrators.py

echo "=== pooled risk analysis (superseded table) ==="
python code/05_risk_signal.py

echo "=== diagnostics: base rates, norm+ variants, sign checks ==="
python code/06_diagnostics.py

echo "=== fold-aware risk detection and depth-matched diagnostic (Tables 4-6) ==="
python code/07_fold_aware_risk.py

echo "=== two-fold calibration and Brier bootstrap (Table 3) ==="
python code/10_twofold_calibration.py

echo "=== figures (checked against the result tables) ==="
python code/08_final_figures.py

echo "=== DONE. See results/ ==="
