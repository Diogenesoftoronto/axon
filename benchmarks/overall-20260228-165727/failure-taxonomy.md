# Failure Taxonomy

Generated: 2026-02-28 23:56:55 UTC
Result files: 3
Total failures: 117

## Category Summary

| Category | Count | Share |
|---|---:|---:|
| format_trace_or_scratchpad_leak | 68 | 58.12% |
| wrong_exact_value | 20 | 17.09% |
| format_extra_text_exact | 13 | 11.11% |
| placeholder_output | 8 | 6.84% |
| abstention_format_extra_text | 7 | 5.98% |
| hallucinated_when_should_abstain | 1 | 0.85% |

## Failures by Mode

| Mode | Failures | Top Categories |
|---|---:|---|
| d0-i1 | 33 | format_trace_or_scratchpad_leak (20), wrong_exact_value (5), format_extra_text_exact (4) |
| d3-i1 | 31 | format_trace_or_scratchpad_leak (24), wrong_exact_value (3), format_extra_text_exact (2) |
| d6-i1 | 31 | format_trace_or_scratchpad_leak (24), wrong_exact_value (3), format_extra_text_exact (2) |
| d1-i3 | 13 | wrong_exact_value (4), abstention_format_extra_text (3), format_extra_text_exact (3) |
| d0-i3 | 9 | wrong_exact_value (5), format_extra_text_exact (2), abstention_format_extra_text (1) |

## Failures by Dataset

| Dataset | Failures | Top Categories |
|---|---:|---|
| rlm_hard_coding_planning.json | 53 | format_trace_or_scratchpad_leak (27), wrong_exact_value (11), format_extra_text_exact (9) |
| rlm_challenges.json | 44 | format_trace_or_scratchpad_leak (33), wrong_exact_value (9), format_extra_text_exact (1) |
| hallucination_guardrails.json | 20 | format_trace_or_scratchpad_leak (8), abstention_format_extra_text (7), format_extra_text_exact (3) |

## Example Failures

### format_trace_or_scratchpad_leak
- `hallucination_guardrails.json` | `d0-i1` | `entity_not_present` | `exact`: returned scratchpad/tool trace
  answer: ````repl print(f"Full context:\n{context}") print("\n--- Searching for 'Dave' ---") print(f"'Dave' found: {'Dave' in context}") ````
- `hallucination_guardrails.json` | `d0-i1` | `grounded_fact_control` | `exact`: returned scratchpad/tool trace
  answer: `The context is very short and straightforward - it's a simple incident report in a key-value format. I can directly see the `incident_id` in the text.  ```repl # Extract the inc...`
- `hallucination_guardrails.json` | `d0-i1` | `grounded_fact_control` | `exact`: returned scratchpad/tool trace
  answer: `The context is very simple - it's a short incident report text (only 112 characters). I can clearly see the incident_id in the text. ```repl # Extract the incident_id directly f...`

### wrong_exact_value
- `rlm_challenges.json` | `d0-i1` | `josephus_41_3` | `exact`: exact answer mismatch
  answer: `28`
- `rlm_challenges.json` | `d0-i1` | `coin_change_count` | `exact`: exact answer mismatch
  answer: `0`
- `rlm_challenges.json` | `d0-i3` | `hanoi_moves_20` | `exact`: exact answer mismatch
  answer: `0`

### format_extra_text_exact
- `hallucination_guardrails.json` | `d1-i3` | `grounded_fact_control` | `exact`: contains expected value plus extra text
  answer: `"INC-2041"`
- `hallucination_guardrails.json` | `d3-i1` | `grounded_fact_control` | `exact`: contains expected value plus extra text
  answer: `'INC-2041'`
- `hallucination_guardrails.json` | `d6-i1` | `grounded_fact_control` | `exact`: contains expected value plus extra text
  answer: `"INC-2041"`

### placeholder_output
- `hallucination_guardrails.json` | `d0-i3` | `missing_owner_field` | `exact`: placeholder token instead of answer
  answer: `answer`
- `rlm_challenges.json` | `d3-i1` | `lis_length` | `exact`: placeholder token instead of answer
  answer: `final_answer`
- `rlm_hard_coding_planning.json` | `d0-i1` | `mini_swe_bug_localization` | `exact`: placeholder token instead of answer
  answer: `final_answer`

### abstention_format_extra_text
- `hallucination_guardrails.json` | `d0-i1` | `missing_owner_field` | `exact`: contains abstention token plus extra text
  answer: `I've inspected the context. It's a simple service registry entry (104 characters) containing metadata about "payments-api":  - service: payments-api - tier: critical   - region:...`
- `hallucination_guardrails.json` | `d0-i1` | `unsupported_causal_claim` | `exact`: contains abstention token plus extra text
  answer: `"INSUFFICIENT_INFORMATION"`
- `hallucination_guardrails.json` | `d0-i3` | `missing_owner_field` | `exact`: contains abstention token plus extra text
  answer: `"INSUFFICIENT_INFORMATION"`

### hallucinated_when_should_abstain
- `hallucination_guardrails.json` | `d1-i3` | `no_release_date` | `exact`: did not output required abstention
  answer: `"February 13, 2025"`
