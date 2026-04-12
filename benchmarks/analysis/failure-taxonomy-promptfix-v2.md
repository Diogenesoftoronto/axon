# Failure Taxonomy

Generated: 2026-02-28 23:56:39 UTC
Result files: 1
Total failures: 12

## Category Summary

| Category | Count | Share |
|---|---:|---:|
| format_trace_or_scratchpad_leak | 9 | 75.00% |
| wrong_exact_value | 1 | 8.33% |
| empty_output | 1 | 8.33% |
| runtime_nonzero_exit | 1 | 8.33% |

## Failures by Mode

| Mode | Failures | Top Categories |
|---|---:|---|
| d0-i1 | 5 | format_trace_or_scratchpad_leak (5) |
| d6-i1 | 4 | format_trace_or_scratchpad_leak (1), wrong_exact_value (1), empty_output (1) |
| d3-i1 | 3 | format_trace_or_scratchpad_leak (3) |

## Failures by Dataset

| Dataset | Failures | Top Categories |
|---|---:|---|
| rlm_hard_coding_planning.json | 12 | format_trace_or_scratchpad_leak (9), wrong_exact_value (1), empty_output (1) |

## Example Failures

### format_trace_or_scratchpad_leak
- `rlm_hard_coding_planning.json` | `d0-i1` | `critical_path_release_plan` | `exact`: returned scratchpad/tool trace
  answer: `Let me analyze this project scheduling problem:  ```repl # Parse the tasks and dependencies tasks = {     'A': (3, []),     'B': (5, ['A']),     'C': (4, ['A']),     'D': (6, ['...`
- `rlm_hard_coding_planning.json` | `d0-i1` | `knapsack_0_1` | `exact`: returned scratchpad/tool trace
  answer: `This is a 0/1 knapsack problem. Let me solve it:  - weights = [10, 20, 30] - values = [60, 100, 120] - capacity = 50  Possible combinations that fit within capacity 50: - Items ...`
- `rlm_hard_coding_planning.json` | `d0-i1` | `interval_scheduling` | `exact`: returned scratchpad/tool trace
  answer: ````repl import re  # Parse the intervals intervals_str = context.split("Intervals (start,end): ")[1].rstrip(".") intervals = eval(intervals_str)  # Greedy algorithm: sort by end...`

### wrong_exact_value
- `rlm_hard_coding_planning.json` | `d6-i1` | `knapsack_0_1` | `exact`: exact answer mismatch
  answer: `-6`

### empty_output
- `rlm_hard_coding_planning.json` | `d6-i1` | `interval_scheduling` | `exact`: empty answer

### runtime_nonzero_exit
- `rlm_hard_coding_planning.json` | `d6-i1` | `dependency_order` | `exact`: non-zero exit
  error: `[depth=0][iter=1/1] response: 247 chars [depth=0][iter=1/1] usage: prompt_tokens=1422 completion_tokens=102 total_tokens=1524 prompt_chars=5929  thread 'main' (375115) panicked ...`
