# Testing Playbook

This document lists the standard scripts and commands used to test Axon locally.

## 1) Core Build and Checks

Run these first on every change:

```bash
cargo fmt --all
cargo test -q
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release -q
```

## 2) Functional Smoke Test (Synthetic)

```bash
timeout 120 ./target/release/axon \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --api-key "$SYNTHETIC_API_KEY" \
  --max-depth 0 \
  --max-iterations 3 \
  -v --trace-sandbox \
  query "What animal jumped over the dog?" \
  --context /tmp/test_context.txt
```

## 3) Benchmark Script (General)

Main script:

```bash
python3 scripts/benchmark_axon.py --help
```

Overall multi-suite wrapper:

```bash
scripts/run_overall_tests.sh
```

Standard run:

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --runs 3 \
  --attempts-per-run 3 \
  --timeout 600
```

Multi-model run (Synthetic model grid):

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --model-list benchmarks/models_synthetic_mix.txt \
  --pricing-from-models-api \
  --runs 3 \
  --attempts-per-run 3 \
  --timeout 600
```

With cost estimation:

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --prompt-cost-per-1m 0.30 \
  --completion-cost-per-1m 1.20 \
  --runs 3 \
  --attempts-per-run 3
```

## 4) Dataset-Specific Runs

### Algorithmic set

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/rlm_challenges.json \
  --runs 3 \
  --attempts-per-run 3
```

### Hard coding/planning set

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/rlm_hard_coding_planning.json \
  --runs 3 \
  --attempts-per-run 3
```

### Long-horizon subset

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/rlm_hard_coding_planning.json \
  --task-id long_horizon_incident_state \
  --task-id dependency_order \
  --runs 5 \
  --attempts-per-run 3
```

### Hallucination guardrail set

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/hallucination_guardrails.json \
  --runs 5 \
  --attempts-per-run 3
```

### Hard Codeforces-style set (Rust-only judging)

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/codeforces_hard_like.json \
  --runs 3 \
  --attempts-per-run 3
```

### Long-context S-NIAH-style set

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/long_context_sniah_xl.json \
  --runs 3 \
  --attempts-per-run 3 \
  --timeout 600
```

### Long-context books + distractor QA

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/long_context_books_distractor.json \
  --model-list benchmarks/models_synthetic_mix.txt \
  --pricing-from-models-api \
  --runs 3 \
  --attempts-per-run 3 \
  --timeout 600
```

### Information-dense ledger aggregation

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/information_dense_ledger.json \
  --model-list benchmarks/models_synthetic_mix.txt \
  --pricing-from-models-api \
  --runs 3 \
  --attempts-per-run 3 \
  --timeout 600
```

### Mode-profile targeted suite (Gemini-designed)

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --dataset benchmarks/mode_profile_targeted.json \
  --runs 3 \
  --attempts-per-run 3 \
  --timeout 600
```

## 5) Previous-Ref Regression Comparison

Compare current code against an earlier git ref:

```bash
python3 scripts/benchmark_axon.py \
  --base-url "https://api.synthetic.new/openai/v1/" \
  --prev-ref 76265b0 \
  --runs 3 \
  --attempts-per-run 3
```

## 6) Output Reports

Benchmark outputs are written to:

- `benchmarks/results/results-<timestamp>.json`

Or use explicit output path:

```bash
python3 scripts/benchmark_axon.py ... --out benchmarks/results/results-custom.json
```

Write markdown summary table too:

```bash
python3 scripts/benchmark_axon.py ... \
  --out benchmarks/results/results-custom.json \
  --summary-md benchmarks/results/summary-custom.md
```

Inspect summary quickly:

```bash
jq '.summary' benchmarks/results/results-custom.json
```

## 7) Notes

- Export API key before running benchmarks:

```bash
export SYNTHETIC_API_KEY=...
```

- For long suites, prefer `PYTHONUNBUFFERED=1` to stream progress:

```bash
env PYTHONUNBUFFERED=1 python3 scripts/benchmark_axon.py ...
```

- Use at least `--timeout 600` (10 minutes) for research-grade runs; shorter values can bias mode comparisons.
- For publishable comparisons, avoid `runs=1`; use at least `--runs 3` and report confidence intervals.
- The overall wrapper writes one folder per run, including per-suite JSON/markdown and `overall-summary.md`.

Overall wrapper controls:

```bash
MAX_PARALLEL=3 RUN_NON_CI_TASKS=1 scripts/run_overall_tests.sh
```

- `MAX_PARALLEL` runs suites concurrently.
- `RUN_NON_CI_TASKS=0` skips non-CI-heavy suites (`mode_profile_targeted`, `codeforces_hard_like`).

## 8) Hugging Face Dataset Adaptation

Generate deterministic Axon JSON tasks from HF datasets:

```bash
python3 scripts/adapt_hf_gsm8k.py --split test --limit 100 --out benchmarks/hf_gsm8k_100.json
python3 scripts/adapt_hf_mbpp_calls.py --split test --limit 100 --out benchmarks/hf_mbpp_calls_100.json
python3 scripts/adapt_hf_book_distractor.py --dataset bookcorpusopen --split train --limit 20 --out benchmarks/hf_books_distractor_20.json
```
