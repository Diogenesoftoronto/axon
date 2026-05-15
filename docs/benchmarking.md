# Benchmarking Altum Against Alternatives and Previous Code

This benchmark harness runs hard reasoning and algorithmic tasks (including Tower of Hanoi style tasks) across multiple Altum configurations and optionally a previous git ref.

## Marimo Workflows (Recommended)

Use marimo + pandas workflows for reproducible benchmark execution and paper plots.

Environment setup (uv):

```bash
uv venv .venv
uv pip install --python .venv/bin/python datasets matplotlib marimo plancraft
```

Notebook workflows:

- Runner workflow (multi-dataset benchmark execution, dataframe outputs, standardized artifacts):

```bash
uv run --python .venv/bin/python marimo edit notebooks/benchmark_run_workflow.py
```

- Analysis workflow (filter by evidence/runtime/scoring policy, dataframe-first plotting and table export):

```bash
uv run --python .venv/bin/python marimo edit notebooks/benchmark_results_analysis.py
```

- PlanCraft workflow (Altum mode profiling inside PlanCraft text environment):

```bash
uv run --python .venv/bin/python marimo edit notebooks/plancraft_altum_benchmark.py
```

The notebooks emit JSON/CSV/Markdown artifacts under `benchmarks/` so paper tables and figures can be regenerated from dataframes.

## What It Compares

Default modes in `scripts/benchmark_altum.py`:

- `current-default`: current binary, `--max-depth 0 --max-iterations 1`
- `current-no-recursion-best-of-3`: current binary, `--max-depth 0 --max-iterations 3`
- `current-depth6-single-pass`: current binary, `--max-depth 6 --max-iterations 1`
- `current-depth1-iter3`: current binary, `--max-depth 1 --max-iterations 3`
- `previous-default`: optional, built from `--prev-ref` using `git worktree`

For mode-profile comparisons, task prompt text is mode-neutral: the runner sends the same `query`/`context` to all modes and only changes runtime controls (`--max-depth`, `--max-iterations`).

Task suites:

- `benchmarks/rlm_challenges.json` (algorithmic)
- `benchmarks/rlm_hard_coding_planning.json` (hard coding/planning and long-horizon state tracking)
- `benchmarks/hallucination_guardrails.json` (insufficient-evidence and anti-fabrication probes)
- `benchmarks/codeforces_hard_like.json` (very hard Codeforces-style algorithmic problems)
  - This suite enforces Rust-only submissions via `check.type = rust_exec_exact`.
  - The benchmark runner auto-appends a strict Rust-only output contract for these tasks and injects diverse non-Rust negative examples (Python/C++/Java/Go/JavaScript) to reduce language-format drift.
- `benchmarks/long_context_sniah_xl.json` (large context S-NIAH style retrieval/counting; generated contexts)
- `benchmarks/long_context_books_distractor.json` (book-style long contexts with distractor narratives and QA)
- `benchmarks/information_dense_ledger.json` (dense aggregation over many rows)
- `benchmarks/mode_profile_targeted.json` (Gemini-designed tasks to distinguish the four mode profiles)

## Task Set

The included dataset has problems like:

- Tower of Hanoi minimum move counts
- Josephus survivor index
- 8-queens solution count
- LIS length
- Edit distance
- Coin change combinations

Each task has:

- `context` or `context_gen`
- `query`
- `check` rule (`exact`, `contains`, `regex`, `number_exact`, `choice_exact`, `yesno_exact`, `lines_in_order`, or `rust_exec_exact`)

`context_gen` currently supports:

- `log_haystack_v1` (large generated text with controlled insertions)
- `ledger_v1` (deterministic transaction ledgers for dense aggregation tasks)

## Run It

Set your API key and run:

```bash
export SYNTHETIC_API_KEY=...
python3 scripts/benchmark_altum.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/rlm_hard_coding_planning.json \
  --runs 3 \
  --attempts-per-run 3 \
  --timeout 600 \
  --prompt-cost-per-1m 0.30 \
  --completion-cost-per-1m 1.20
```

Run all core suites in one shot:

```bash
export SYNTHETIC_API_KEY=...
scripts/run_overall_tests.sh
```

With parallelism and CI-safe dataset scope:

```bash
MAX_PARALLEL=3 RUN_NON_CI_TASKS=0 scripts/run_overall_tests.sh
```

`RUN_NON_CI_TASKS=0` skips heavy/non-CI suites (`mode_profile_targeted`, `codeforces_hard_like`).

Recommended Python env setup with `uv`:

```bash
uv venv .venv
uv pip install --python .venv/bin/python datasets matplotlib marimo plancraft
```

Run across multiple models (root and sub-model default to the same id):

```bash
python3 scripts/benchmark_altum.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/long_context_books_distractor.json \
  --model-list benchmarks/models_synthetic_mix.txt \
  --max-tasks 3 \
  --runs 3 \
  --attempts-per-run 3 \
  --timeout 600 \
  --pricing-from-models-api \
  --summary-md benchmarks/results/summary-books.md \
  --out benchmarks/results/results-books-multimodel.json
```

Compare with previous code:

```bash
python3 scripts/benchmark_altum.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --prev-ref 76265b0 \
  --runs 3 \
  --attempts-per-run 3
```

The runner writes a JSON report to `benchmarks/results/results-<timestamp>.json` unless `--out` is provided.

Retry controls:

- `--attempts-per-run`: retries transient provider/transport failures while preserving one logical sample per run.
- `--retry-backoff-s`: linear backoff base between retry attempts.

Policy/depth controls forwarded to Altum:

- `--policy-profile`
- `--inject-policy-into-context`
- `--depth-enforcement {off,soft,strict}`
- `--require-min-depth`
- `--require-min-recursive-calls`

## Cost and Context Metrics

The runner captures and reports per run:

- context length (`context_chars`)
- query length (`query_chars`)
- LLM call count (`llm_calls`)
- token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`)
- logged prompt/response character totals
- estimated USD cost from:
  - `--prompt-cost-per-1m`
  - `--completion-cost-per-1m`
  - or exact per-token pricing from `/models` with `--pricing-from-models-api`

If pricing flags are omitted, cost fields are still emitted but remain `0`.

## How To Add Harder Problems

Edit `benchmarks/rlm_challenges.json` and add entries:

```json
{
  "id": "hanoi_moves_24",
  "context": "Classic Tower of Hanoi with 24 disks.",
  "query": "What is the minimum number of moves? Return just the integer.",
  "check": {"type": "exact", "value": "16777215"}
}
```

Tips:

- Keep checks deterministic.
- Favor integer or short-string answers for robust scoring.
- Use `regex` checks for multi-line outputs where exact matching is brittle.

## Reading Results

The summary includes per-mode:

- total runs
- process success count
- match count
- pass rate
- 95% confidence intervals for pass/ok rates (Wilson)
- average latency
- average tokens per task
- average cost per task
- hallucination rate on guardrail probes

Use this to compare strategy variants and regressions over time.

Optional markdown table output:

```bash
python3 scripts/benchmark_altum.py ... --summary-md benchmarks/summary.md
```

For stable comparisons, avoid single-shot (`runs=1`) reports. Use `--runs 3` or higher and include CI bounds.
Use `--timeout 600` or higher to avoid undercounting slow but valid completions.

## Hugging Face Adapters

Adapter scripts to generate deterministic Altum task JSON from HF datasets:

- `scripts/adapt_hf_gsm8k.py`
- `scripts/adapt_hf_mbpp_calls.py`
- `scripts/adapt_hf_arc_mcq.py`
- `scripts/adapt_hf_boolq.py`
- `scripts/adapt_hf_hellaswag.py`
- `scripts/adapt_hf_book_distractor.py`

Examples:

```bash
.venv/bin/python scripts/adapt_hf_gsm8k.py --split test --limit 100 --out benchmarks/hf_gsm8k_100.json
.venv/bin/python scripts/adapt_hf_mbpp_calls.py --split test --limit 100 --out benchmarks/hf_mbpp_calls_100.json
.venv/bin/python scripts/adapt_hf_arc_mcq.py --config ARC-Challenge --split validation --limit 100 --out benchmarks/hf_arc_challenge_100.json
.venv/bin/python scripts/adapt_hf_boolq.py --split validation --limit 100 --out benchmarks/hf_boolq_100.json
.venv/bin/python scripts/adapt_hf_hellaswag.py --split validation --limit 100 --out benchmarks/hf_hellaswag_100.json
.venv/bin/python scripts/adapt_hf_book_distractor.py --dataset bookcorpusopen --split train --limit 20 --out benchmarks/hf_books_distractor_20.json
```

Run the prepared HF benchmark pack:

```bash
export SYNTHETIC_API_KEY=...
scripts/run_hf_benchmarks.sh
```

## Plotting and Analysis

Generate summary plots from benchmark reports and HF task files:

```bash
.venv/bin/python scripts/plot_benchmark_findings.py \
  --results-glob "benchmarks/results/results-*.json" \
  --out-dir benchmarks/analysis
```

This writes:

- `benchmarks/analysis/task_dataset_profile.csv`
- `benchmarks/analysis/task_dataset_profile.png`
- `benchmarks/analysis/benchmark_results_flat.csv`
- `benchmarks/analysis/results_cost_vs_pass.png`
- `benchmarks/analysis/results_passrate_heatmap.png`

Interactive analysis with marimo:

```bash
.venv/bin/marimo edit notebooks/benchmark_results_analysis.py
```

Failure taltumomy analysis:

```bash
.venv/bin/python scripts/failure_taltumomy.py \
  benchmarks/overall-20260228-165727 \
  --out-md benchmarks/overall-20260228-165727/failure-taltumomy.md \
  --out-csv benchmarks/overall-20260228-165727/failure-taltumomy.csv
```

This writes a per-failure flat CSV and a Markdown report with:

- top failure categories,
- breakdown by mode and dataset,
- example failures per category.

Prompt-policy optimization:

```bash
.venv/bin/python scripts/optimize_prompt_policy.py \
  "benchmarks/results/results-*.json" \
  --out-md benchmarks/analysis/policy_optimization_report.md \
  --out-json benchmarks/analysis/policy_optimization_selection.json
```

## SWE-bench/ReBench Alignment Notes

This harness is closest to a reasoning and planning benchmark with deterministic scoring. It is useful for:

- comparing recursive vs non-recursive Altum modes,
- measuring token/cost tradeoffs,
- testing long-horizon state tracking and planning behavior.

It is not yet a full SWE-bench style patch benchmark. SWE-bench style evaluation needs:

- real repository tasks,
- generated code edits,
- test execution as ground truth,
- patch-level success metrics.

Recommended next step for SWE-bench alignment:

1. Add task adapters that provide a codebase snapshot in context.
2. Ask Altum for patch output in unified diff format.
3. Apply patch in sandbox clone.
4. Run project tests and score pass/fail.

## HLE Note

Humanity's Last Exam (HLE) is a strong source of very hard expert-level questions. In practice, the official dataset is distributed with access controls on Hugging Face, so this repo includes HLE-style tasks rather than redistributing HLE items directly.

If you have access, you can extend this harness by exporting a local JSON subset in the same schema (`id`, `context`, `query`, `check`) and running it with `--dataset`.
