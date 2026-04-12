# Axon vs. Recursive Language Models (RLM) Paper: Result Comparison

Reference paper in README: [Recursive Language Models](https://arxiv.org/abs/2512.24601v1).

## Scope

This note compares what our Axon benchmark results currently show against the key empirical claims in the RLM paper. It is a research comparison, not a replication. Our runs use different models, task sets, and infrastructure.

## Paper Claims We Can Compare

From the paper abstract and observations:

- RLM improves performance on difficult, context-heavy tasks by recursively decomposing work.
- RLM can scale to very large external context while keeping generation quality and cost favorable.
- Cost and performance depend on decomposition strategy and problem structure.

## Our Evidence (Current Repo Results)

Primary result files:

- `benchmarks/results/results-hard-planning-sample.json`
- `benchmarks/results/results-longhorizon-sample.json`
- `benchmarks/results/results-hard-vs-prev.json`
- `benchmarks/results/results-hard-scored.json`
- `benchmarks/results/results-hallucination-sample-short.json`
- `benchmarks/results/results-cf-rust-only-smoke.json`

### 1) Recursive depth helps on hard planning and long-horizon subsets

`current-default` (`max-depth=2`) vs `current-no-recursion` (`max-depth=0`):

- Hard planning sample: `100%` vs `50%` pass rate (`+50` points for recursion).
- Long-horizon sample: `100%` vs `50%` pass rate (`+50` points for recursion).

Interpretation: this supports the paper's direction that recursion can improve quality on tasks requiring decomposition and state tracking.

### 2) Lower budget can reduce cost, but quality can collapse on harder trajectories

`current-low-budget` vs `current-default`:

- Hard planning sample: same pass rate (`100%`), about `13.3%` lower avg cost per task.
- Long-horizon sample: avg cost reduced, but pass rate dropped to `0%`.

Interpretation: consistent with the paper's broader tradeoff framing; decomposition budget and iteration policy are critical.

### 3) Current implementation appears faster than prior code on sampled tasks, with parity on pass rate

From `results-hard-vs-prev.json`:

- `current-default` and `previous-default` both `50%` pass rate on the sampled set.
- Current code average runtime is about `25.9%` faster (`14.188s` vs `19.151s`).

Interpretation: engineering improvements improved latency in this slice, but no quality gain was observed on this specific sample.

### 4) Hardest coding regime is still weak

Rust-only Codeforces-style smoke (`results-cf-rust-only-smoke.json`):

- All tested modes had `0%` pass rate.

Interpretation: unlike the strongest claims of robustness in the paper's hardest settings, our current scaffold is not yet strong on very hard competitive programming with strict executable checks.

### 5) Hallucination robustness is not yet validated

`results-hallucination-sample-short.json` shows `0%` hallucination rate, but all modes also had `0%` pass/process reliability in that short run.

Interpretation: this is not strong evidence of hallucination resistance yet. Reliability and timeout handling must improve before drawing conclusions.

## Where Our Findings Match the Paper

- Directionally aligned: recursion helps on hard planning/long-horizon tasks.
- Directionally aligned: budget and decomposition policy materially change cost-quality outcomes.

## Where We Cannot Claim Alignment Yet

- No replication of paper scale (the paper discusses much larger context regimes, including million-token class settings).
- Different models and providers, so absolute numbers are not directly comparable.
- Our hardest-code benchmarks currently underperform.
- Hallucination evaluation is still underpowered/confounded by failures.

## Methodological Gaps to Close

1. Increase task count per suite and number of runs (confidence intervals, not single small samples).
2. Add significance testing for pass-rate and latency differences.
3. Stabilize hallucination harness to separate refusal quality from infra/runtime failures.
4. Add longer-context stress tests (progressively larger context windows) to directly test the paper's scaling claim.
5. Add SWE-bench style patch-and-test tasks for external validity on software engineering outcomes.

## Practical Conclusion

Current evidence supports the core qualitative RLM hypothesis in our setting: recursive decomposition can improve success on difficult planning tasks. But our evidence is still preliminary and does not yet support strong claims about extreme-context scaling or hard-code robustness at the level implied by the cited paper.
