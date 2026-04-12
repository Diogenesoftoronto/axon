#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_URL="${BASE_URL:-https://api.synthetic.new/openai/v1/}"
API_KEY_ENV="${API_KEY_ENV:-SYNTHETIC_API_KEY}"
MODEL="${MODEL:-hf:MiniMaxAI/MiniMax-M2.5}"
RUNS="${RUNS:-3}"
ATTEMPTS_PER_RUN="${ATTEMPTS_PER_RUN:-3}"
RETRY_BACKOFF_S="${RETRY_BACKOFF_S:-2}"
TIMEOUT_S="${TIMEOUT_S:-600}"
MAX_TASKS="${MAX_TASKS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-benchmarks/overall-${STAMP}}"

if [[ -z "${RUN_NON_CI_TASKS:-}" ]]; then
  if [[ "${CI:-}" == "true" ]]; then
    RUN_NON_CI_TASKS="0"
  else
    RUN_NON_CI_TASKS="1"
  fi
fi

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Missing API key in env var ${API_KEY_ENV}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

CORE_DATASETS=(
  "benchmarks/rlm_challenges.json"
  "benchmarks/rlm_hard_coding_planning.json"
  "benchmarks/hallucination_guardrails.json"
  "benchmarks/long_context_books_distractor.json"
  "benchmarks/information_dense_ledger.json"
)
NON_CI_DATASETS=(
  "benchmarks/mode_profile_targeted.json"
  "benchmarks/codeforces_hard_like.json"
  "benchmarks/hf_mmlu_100.json"
  "benchmarks/hf_winogrande_100.json"
  "benchmarks/hf_commonsenseqa_100.json"
  "benchmarks/hf_truthfulqa_mc1_100.json"
)

DATASETS=("${CORE_DATASETS[@]}")
if [[ "${RUN_NON_CI_TASKS}" == "1" ]]; then
  DATASETS+=("${NON_CI_DATASETS[@]}")
fi

run_dataset() {
  local dataset="$1"
  local name
  name="$(basename "${dataset}" .json)"
  local out_json="${OUT_DIR}/results-${name}.json"
  local out_md="${OUT_DIR}/summary-${name}.md"
  local log_path="${OUT_DIR}/log-${name}.txt"

  local cmd=(
    python3 scripts/benchmark_axon.py
    --dataset "${dataset}"
    --base-url "${BASE_URL}"
    --api-key-env "${API_KEY_ENV}"
    --model "${MODEL}"
    --runs "${RUNS}"
    --attempts-per-run "${ATTEMPTS_PER_RUN}"
    --retry-backoff-s "${RETRY_BACKOFF_S}"
    --timeout "${TIMEOUT_S}"
    --pricing-from-models-api
    --out "${out_json}"
    --summary-md "${out_md}"
  )

  if [[ "${MAX_TASKS}" != "0" ]]; then
    cmd+=(--max-tasks "${MAX_TASKS}")
  fi

  echo "=== Running ${name} (log: ${log_path}) ==="
  "${cmd[@]}" >"${log_path}" 2>&1
  echo "=== Finished ${name} ==="
}

running=0
failed=0
for dataset in "${DATASETS[@]}"; do
  run_dataset "${dataset}" &
  running=$((running + 1))

  if (( running >= MAX_PARALLEL )); then
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  fi
done

while (( running > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  running=$((running - 1))
done

if (( failed != 0 )); then
  echo "At least one dataset run failed. Check logs under ${OUT_DIR}." >&2
  exit 1
fi

python3 - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for p in sorted(out_dir.glob("results-*.json")):
    data = json.loads(p.read_text())
    dataset = Path(data.get("dataset", p.name)).name
    for s in data.get("summary", []):
        rows.append(
            (
                dataset,
                s.get("mode", ""),
                s.get("pass_rate", 0.0),
                s.get("pass_rate_ci95_low", 0.0),
                s.get("pass_rate_ci95_high", 0.0),
                s.get("avg_cost_usd", 0.0),
                s.get("avg_time_s", 0.0),
                s.get("readiness", ""),
            )
        )

lines = [
    "| Dataset | Mode | Pass rate | 95% CI | Avg cost (USD) | Avg time (s) | Readiness |",
    "|---|---|---:|---:|---:|---:|---|",
]
for dataset, mode, pass_rate, lo, hi, avg_cost, avg_time, readiness in rows:
    lines.append(
        f"| {dataset} | {mode} | {pass_rate:.2f}% | [{lo:.2f}, {hi:.2f}] | {avg_cost:.6f} | {avg_time:.3f} | {readiness} |"
    )

summary_path = out_dir / "overall-summary.md"
summary_path.write_text("\n".join(lines) + "\n")
print(f"Wrote combined summary: {summary_path}")
PY

echo "Overall test artifacts: ${OUT_DIR}"
