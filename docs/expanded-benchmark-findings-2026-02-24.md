# Expanded Benchmark Findings (2026-02-24)

This note summarizes the post-review benchmark expansion focused on:

- long-context books with distractors (`benchmarks/long_context_books_distractor.json`)
- information-dense aggregation (`benchmarks/information_dense_ledger.json`)
- multi-model runs on Synthetic-compatible models

## What Changed

1. Replaced purely marker-style long-context tasks with book-like distractor QA tasks.
2. Added deterministic generated context types in the harness:
   - `log_haystack_v1` with insertions and distributed markers
   - `ledger_v1` for dense aggregation tasks
3. Added model-grid execution to `scripts/benchmark_axon.py`:
   - `--models`, `--model-list`
   - `/models` pricing fetch via `--pricing-from-models-api`
   - Wilson 95% confidence intervals in summaries
   - optional markdown summary output via `--summary-md`

## Run Artifacts

- Books+distractors (small sample): `benchmarks/results/results-books-fast.json`
- Books+distractors (higher sample with retries): `benchmarks/results/results-books-minimax-n15-retry.json`
- Ledger dense aggregation (small sample): `benchmarks/results/results-ledger-fast.json`
- Ledger dense aggregation (higher sample with retries): `benchmarks/results/results-ledger-minimax-n18-retry.json`
- Summaries: `benchmarks/results/summary-books-fast.md`, `benchmarks/results/summary-books-minimax-n15-retry.md`, `benchmarks/results/summary-ledger-fast.md`, `benchmarks/results/summary-ledger-minimax-n18-retry.md`

Model set used:

- `hf:MiniMaxAI/MiniMax-M2.5`
- `hf:deepseek-ai/DeepSeek-V3-0324`
- `hf:meta-llama/Llama-3.3-70B-Instruct`

## Headline Results

Books+distractors (2 tasks, 1 run):

- MiniMax-M2.5: default `50%`, low-budget `50%`, no-recursion `0%`
- Llama-3.3-70B: default `0%`, low-budget `0%`, no-recursion `100%`
- DeepSeek-V3-0324: `0%` all modes due backend 404/503 errors

Books+distractors higher-sample rerun (MiniMax-M2.5, 5 tasks, 3 runs, retries enabled; 15 samples/mode):

- default: `73.33%` pass, CI95 `[48.05, 89.10]`
- low-budget: `66.67%` pass, CI95 `[41.71, 84.82]`
- no-recursion: `60.00%` pass, CI95 `[35.75, 80.18]`

Ledger dense aggregation (3 tasks, 1 run):

- MiniMax-M2.5: default `100%`, no-recursion `100%`, low-budget `33.33%`
- Llama-3.3-70B: default `0%`, no-recursion `33.33%`, low-budget `0%`
- DeepSeek-V3-0324: `0%` all modes due backend 404/503 errors

Ledger higher-sample rerun (MiniMax-M2.5, 6 tasks, 3 runs, retries enabled; 18 samples/mode):

- default: `61.11%` pass, CI95 `[38.62, 79.70]`
- low-budget: `44.44%` pass, CI95 `[24.56, 66.28]`
- no-recursion: `83.33%` pass, CI95 `[60.78, 94.16]`

## Interpretation

1. The new long-context suite is less trivially gameable than pure marker retrieval because it mixes target-book facts with distractor-book facts and asks semantic QA over target scope.
2. Mode performance is model-specific. No universal advantage for recursion appeared in these small samples.
3. Model availability and provider routing are currently a major confound:
   - DeepSeek-V3-0324 was listed in `/models` but failed at inference time in this run.
4. Confidence intervals are wide due small `N`; these results are directional, not conclusive.
5. Increasing sample size and adding retry logic removed the suspicious `50%` granularity pattern from books runs and exposed a clearer mode ranking.

## Next Improvements

1. Continue increasing sample size toward at least 30-50 tasks per suite before strong claims.
2. Add a model-availability preflight in the harness (`/models` plus a cheap probe call) to skip unavailable model ids.
3. Add a full-context baseline mode for apples-to-apples comparison with recursion.
4. Add time-to-first-token and sandbox-overhead metrics to better quantify recursion tax.
