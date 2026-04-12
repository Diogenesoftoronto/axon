#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_URL="${BASE_URL:-https://api.synthetic.new/openai/v1/}"
API_KEY_ENV="${API_KEY_ENV:-SYNTHETIC_API_KEY}"
MODEL="${MODEL:-hf:MiniMaxAI/MiniMax-M2.5}"
RUNS="${RUNS:-1}"
ATTEMPTS_PER_RUN="${ATTEMPTS_PER_RUN:-1}"
TIMEOUT_S="${TIMEOUT_S:-600}"
MAX_TASKS="${MAX_TASKS:-100}"
OUT_DIR="${OUT_DIR:-benchmarks/hf-runs-$(date +%Y%m%d-%H%M%S)}"

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Missing API key in env var ${API_KEY_ENV}" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

DATASETS=(
  "benchmarks/hf_gsm8k_100.json"
  "benchmarks/hf_mbpp_calls_100.json"
  "benchmarks/hf_arc_challenge_100.json"
  "benchmarks/hf_boolq_100.json"
  "benchmarks/hf_hellaswag_100.json"
  "benchmarks/hf_mmlu_100.json"
  "benchmarks/hf_winogrande_100.json"
  "benchmarks/hf_commonsenseqa_100.json"
  "benchmarks/hf_truthfulqa_mc1_100.json"
)

for dataset in "${DATASETS[@]}"; do
  name="$(basename "$dataset" .json)"
  echo "=== Running $name ==="
  python3 scripts/benchmark_axon.py \
    --dataset "$dataset" \
    --base-url "$BASE_URL" \
    --api-key-env "$API_KEY_ENV" \
    --model "$MODEL" \
    --runs "$RUNS" \
    --attempts-per-run "$ATTEMPTS_PER_RUN" \
    --timeout "$TIMEOUT_S" \
    --max-tasks "$MAX_TASKS" \
    --pricing-from-models-api \
    --out "$OUT_DIR/results-$name.json" \
    --summary-md "$OUT_DIR/summary-$name.md"
done

echo "HF benchmark artifacts written to $OUT_DIR"
