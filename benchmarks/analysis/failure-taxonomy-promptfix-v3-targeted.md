# Failure Taxonomy

Generated: 2026-03-01 00:01:40 UTC
Result files: 1
Total failures: 6

## Category Summary

| Category | Count | Share |
|---|---:|---:|
| format_trace_or_scratchpad_leak | 5 | 83.33% |
| placeholder_output | 1 | 16.67% |

## Failures by Mode

| Mode | Failures | Top Categories |
|---|---:|---|
| d0-i1 | 2 | format_trace_or_scratchpad_leak (1), placeholder_output (1) |
| d3-i1 | 2 | format_trace_or_scratchpad_leak (2) |
| d6-i1 | 2 | format_trace_or_scratchpad_leak (2) |

## Failures by Dataset

| Dataset | Failures | Top Categories |
|---|---:|---|
| rlm_hard_coding_planning.json | 6 | format_trace_or_scratchpad_leak (5), placeholder_output (1) |

## Example Failures

### format_trace_or_scratchpad_leak
- `rlm_hard_coding_planning.json` | `d0-i1` | `interval_scheduling` | `exact`: returned scratchpad/tool trace
  answer: ````repl import re # Parse intervals from context intervals_str = context pattern = r'\((\d+),(\d+)\)' matches = re.findall(pattern, intervals_str) intervals = [(int(m[0]), int(m...`
- `rlm_hard_coding_planning.json` | `d3-i1` | `interval_scheduling` | `exact`: returned scratchpad/tool trace
  answer: ````repl import re  # Parse the intervals from context text = context intervals = re.findall(r'\((\d+),(\d+)\)', text) intervals = [(int(s), int(e)) for s, e in intervals] print(...`
- `rlm_hard_coding_planning.json` | `d3-i1` | `long_horizon_incident_state` | `exact`: returned scratchpad/tool trace
  answer: `I can see the context is an incident log tracking feature flags ALPHA and BETA over 18 days. Let me parse all entries to verify the final states at the end of Day 18. ```repl im...`

### placeholder_output
- `rlm_hard_coding_planning.json` | `d0-i1` | `mini_swe_bug_localization` | `exact`: placeholder token instead of answer
  answer: `answer`
