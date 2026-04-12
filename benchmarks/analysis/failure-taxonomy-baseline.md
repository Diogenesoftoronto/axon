# Failure Taxonomy

Generated: 2026-02-28 23:46:57 UTC
Result files: 1
Total failures: 53

## Category Summary

| Category | Count | Share |
|---|---:|---:|
| format_trace_or_scratchpad_leak | 27 | 50.94% |
| wrong_exact_value | 11 | 20.75% |
| format_extra_text_exact | 9 | 16.98% |
| placeholder_output | 6 | 11.32% |

## Failures by Mode

| Mode | Failures | Top Categories |
|---|---:|---|
| d3-i1 | 15 | format_trace_or_scratchpad_leak (12), placeholder_output (1), wrong_exact_value (1) |
| d0-i1 | 14 | format_trace_or_scratchpad_leak (6), format_extra_text_exact (3), wrong_exact_value (3) |
| d6-i1 | 12 | format_trace_or_scratchpad_leak (9), placeholder_output (1), wrong_exact_value (1) |
| d1-i3 | 7 | wrong_exact_value (3), placeholder_output (2), format_extra_text_exact (2) |
| d0-i3 | 5 | wrong_exact_value (3), format_extra_text_exact (2) |

## Failures by Dataset

| Dataset | Failures | Top Categories |
|---|---:|---|
| rlm_hard_coding_planning.json | 53 | format_trace_or_scratchpad_leak (27), wrong_exact_value (11), format_extra_text_exact (9) |

## Example Failures

### format_trace_or_scratchpad_leak
- `rlm_hard_coding_planning.json` | `d0-i1` | `critical_path_release_plan` | `exact`: returned scratchpad/tool trace
  answer: `I can solve this directly with Python - it's a critical path problem with task dependencies. Let me calculate it: ```repl # Parse tasks and calculate minimum duration with unlim...`
- `rlm_hard_coding_planning.json` | `d0-i1` | `critical_path_release_plan` | `exact`: returned scratchpad/tool trace
  answer: `This is a project scheduling problem. Let me calculate the minimum project duration with unlimited parallelism (tasks can run as soon as their dependencies are complete): ```rep...`
- `rlm_hard_coding_planning.json` | `d0-i1` | `interval_scheduling` | `exact`: returned scratchpad/tool trace
  answer: `I can solve this directly with Python - it's a classic interval scheduling problem. The context is a simple list of intervals, and I need to find the maximum number of non-overl...`

### wrong_exact_value
- `rlm_hard_coding_planning.json` | `d0-i1` | `long_horizon_incident_state` | `exact`: exact answer mismatch
  answer: `{'ALPHA': 'ON', 'BETA': 'ON'}`
- `rlm_hard_coding_planning.json` | `d0-i1` | `long_horizon_incident_state` | `exact`: exact answer mismatch
  answer: `{"ALPHA":"ON","BETA":"OFF"}`
- `rlm_hard_coding_planning.json` | `d0-i1` | `dependency_order` | `exact`: exact answer mismatch
  answer: `Looking at the context, I can see the dependency graph clearly:  - parser depends on lexer - typecheck depends on parser - codegen depends on typecheck - optimizer depends on pa...`

### format_extra_text_exact
- `rlm_hard_coding_planning.json` | `d0-i1` | `knapsack_0_1` | `exact`: contains expected value plus extra text
  answer: `Looking at the context, this is a 0/1 knapsack problem with: - weights = [10, 20, 30] - values = [60, 100, 120] - capacity = 50  Let me verify the maximum achievable value:  - I...`
- `rlm_hard_coding_planning.json` | `d0-i1` | `dependency_order` | `exact`: contains expected value plus extra text
  answer: `"lexer,parser,typecheck,optimizer,codegen,packager"`
- `rlm_hard_coding_planning.json` | `d0-i1` | `dependency_order` | `exact`: contains expected value plus extra text
  answer: `'lexer,parser,typecheck,optimizer,codegen,packager'`

### placeholder_output
- `rlm_hard_coding_planning.json` | `d0-i1` | `mini_swe_bug_localization` | `exact`: placeholder token instead of answer
  answer: `final_answer`
- `rlm_hard_coding_planning.json` | `d0-i1` | `long_horizon_incident_state` | `exact`: placeholder token instead of answer
  answer: `final_answer`
- `rlm_hard_coding_planning.json` | `d1-i3` | `mini_swe_bug_localization` | `exact`: placeholder token instead of answer
  answer: `final_answer`
