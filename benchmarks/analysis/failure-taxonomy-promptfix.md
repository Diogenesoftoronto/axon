# Failure Taxonomy

Generated: 2026-02-28 23:46:50 UTC
Result files: 1
Total failures: 12

## Category Summary

| Category | Count | Share |
|---|---:|---:|
| format_trace_or_scratchpad_leak | 8 | 66.67% |
| wrong_exact_value | 3 | 25.00% |
| placeholder_output | 1 | 8.33% |

## Failures by Mode

| Mode | Failures | Top Categories |
|---|---:|---|
| d0-i1 | 4 | format_trace_or_scratchpad_leak (2), placeholder_output (1), wrong_exact_value (1) |
| d3-i1 | 4 | format_trace_or_scratchpad_leak (4) |
| d6-i1 | 4 | wrong_exact_value (2), format_trace_or_scratchpad_leak (2) |

## Failures by Dataset

| Dataset | Failures | Top Categories |
|---|---:|---|
| rlm_hard_coding_planning.json | 12 | format_trace_or_scratchpad_leak (8), wrong_exact_value (3), placeholder_output (1) |

## Example Failures

### format_trace_or_scratchpad_leak
- `rlm_hard_coding_planning.json` | `d0-i1` | `interval_scheduling` | `exact`: returned scratchpad/tool trace
  answer: ````repl # This is a classic interval scheduling problem # When touching endpoints is allowed, intervals [a,b] and [b,c] can both be selected  # Extract intervals from context im...`
- `rlm_hard_coding_planning.json` | `d0-i1` | `long_horizon_incident_state` | `exact`: returned scratchpad/tool trace
  answer: `I can see this is an incident log tracking feature flags ALPHA and BETA over days. Let me look for Day 18 specifically to find the exact states: ```repl import re # Find all men...`
- `rlm_hard_coding_planning.json` | `d3-i1` | `critical_path_release_plan` | `exact`: returned scratchpad/tool trace
  answer: `I can see this is a project scheduling problem with tasks A-F, their durations, and dependencies. Let me analyze this to find the minimum project duration with unlimited paralle...`

### wrong_exact_value
- `rlm_hard_coding_planning.json` | `d0-i1` | `dependency_order` | `exact`: exact answer mismatch
  answer: `Looking at this dependency graph, I need to perform a topological sort to find a valid build order:  - lexer (no deps) - parser depends on lexer - typecheck depends on parser - ...`
- `rlm_hard_coding_planning.json` | `d6-i1` | `critical_path_release_plan` | `exact`: exact answer mismatch
  answer: `0`
- `rlm_hard_coding_planning.json` | `d6-i1` | `long_horizon_incident_state` | `exact`: exact answer mismatch
  answer: `{'ALPHA': 'ON', 'BETA': 'ON'}`

### placeholder_output
- `rlm_hard_coding_planning.json` | `d0-i1` | `mini_swe_bug_localization` | `exact`: placeholder token instead of answer
  answer: `answer`
