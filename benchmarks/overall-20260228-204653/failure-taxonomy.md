# Failure Taxonomy

Generated: 2026-03-01 19:54:13 UTC
Result files: 11
Total failures: 4151

## Category Summary

| Category | Count | Share |
|---|---:|---:|
| runtime_nonzero_exit | 3123 | 75.23% |
| wrong_exact_value | 355 | 8.55% |
| wrong_choice | 340 | 8.19% |
| rust_exec_mismatch | 90 | 2.17% |
| regex_mismatch | 68 | 1.64% |
| rust_missing_code | 55 | 1.32% |
| wrong_numeric_value | 55 | 1.32% |
| empty_output | 53 | 1.28% |
| placeholder_output | 4 | 0.10% |
| format_trace_or_scratchpad_leak | 3 | 0.07% |
| missing_numeric_answer | 3 | 0.07% |
| hallucinated_when_should_abstain | 2 | 0.05% |

## Failures by Mode

| Mode | Failures | Top Categories |
|---|---:|---|
| d6-i1 | 1211 | runtime_nonzero_exit (1056), wrong_exact_value (71), rust_exec_mismatch (18) |
| d3-i1 | 851 | runtime_nonzero_exit (655), wrong_exact_value (78), wrong_choice (57) |
| d1-i3 | 806 | runtime_nonzero_exit (600), wrong_exact_value (75), wrong_choice (67) |
| d0-i3 | 729 | runtime_nonzero_exit (512), wrong_choice (79), wrong_exact_value (66) |
| d0-i1 | 554 | runtime_nonzero_exit (300), wrong_choice (119), wrong_exact_value (65) |

## Failures by Dataset

| Dataset | Failures | Top Categories |
|---|---:|---|
| hf_truthfulqa_mc1_100.json | 1500 | runtime_nonzero_exit (1500) |
| hf_commonsenseqa_100.json | 1176 | runtime_nonzero_exit (1112), wrong_choice (64) |
| hf_winogrande_100.json | 515 | runtime_nonzero_exit (355), wrong_choice (160) |
| hf_mmlu_100.json | 274 | runtime_nonzero_exit (156), wrong_choice (116), empty_output (2) |
| mode_profile_targeted.json | 172 | wrong_exact_value (97), wrong_numeric_value (55), empty_output (16) |
| codeforces_hard_like.json | 147 | rust_exec_mismatch (90), rust_missing_code (55), placeholder_output (2) |
| rlm_challenges.json | 111 | wrong_exact_value (99), empty_output (11), placeholder_output (1) |
| information_dense_ledger.json | 90 | wrong_exact_value (86), empty_output (2), format_trace_or_scratchpad_leak (2) |
| long_context_books_distractor.json | 75 | regex_mismatch (68), empty_output (7) |
| rlm_hard_coding_planning.json | 74 | wrong_exact_value (63), empty_output (10), format_trace_or_scratchpad_leak (1) |
| hallucination_guardrails.json | 17 | wrong_exact_value (10), empty_output (5), hallucinated_when_should_abstain (2) |

## Example Failures

### runtime_nonzero_exit
- `hf_commonsenseqa_100.json` | `d0-i3` | `commonsenseqa_validation_0029` | `choice_exact`: non-zero exit
  error: `Error: Web call failed for model 'hf:MiniMaxAI/MiniMax-M2.5 (adapter: OpenAI)'. Cause: Request failed with status code '402 Payment Required'. Response body: {"error":"Insuffici...`
- `hf_commonsenseqa_100.json` | `d0-i3` | `commonsenseqa_validation_0029` | `choice_exact`: non-zero exit
  error: `Error: Web call failed for model 'hf:MiniMaxAI/MiniMax-M2.5 (adapter: OpenAI)'. Cause: Request failed with status code '429 Too Many Requests'. Response body: {"error":"You've e...`
- `hf_commonsenseqa_100.json` | `d0-i3` | `commonsenseqa_validation_0030` | `choice_exact`: non-zero exit
  error: `Error: Web call failed for model 'hf:MiniMaxAI/MiniMax-M2.5 (adapter: OpenAI)'. Cause: Request failed with status code '429 Too Many Requests'. Response body: {"error":"You've e...`

### wrong_exact_value
- `hallucination_guardrails.json` | `d0-i1` | `grounded_fact_control` | `exact`: exact answer mismatch
  answer: `no context provided`
- `hallucination_guardrails.json` | `d0-i3` | `grounded_fact_control` | `exact`: exact answer mismatch
  answer: `<minimax:tool_call> <invoke name="meta_vim_ino"> <parameter name="expr">context</parameter> </invoke> </minimax:tool_call>`
- `hallucination_guardrails.json` | `d0-i3` | `grounded_fact_control` | `exact`: exact answer mismatch
  answer: `<minimax:tool_call> <invoke name="inspect_context"> </invoke> </minimax:tool_call>`

### wrong_choice
- `hf_commonsenseqa_100.json` | `d0-i1` | `commonsenseqa_validation_0004` | `choice_exact`: incorrect choice token
  answer: `D`
- `hf_commonsenseqa_100.json` | `d0-i1` | `commonsenseqa_validation_0004` | `choice_exact`: incorrect choice token
  answer: `B`
- `hf_commonsenseqa_100.json` | `d0-i1` | `commonsenseqa_validation_0009` | `choice_exact`: incorrect choice token
  answer: `E`

### rust_exec_mismatch
- `codeforces_hard_like.json` | `d0-i1` | `cf_tree_dp_weighted_independent_set` | `rust_exec_exact`: Rust code did not produce expected output
  answer: ````rust fn main() {     println!("0"); } ````
- `codeforces_hard_like.json` | `d0-i1` | `cf_bitmask_dp_count_toposorts` | `rust_exec_exact`: Rust code did not produce expected output
  answer: ````rust fn main() {     println!("0"); } ````
- `codeforces_hard_like.json` | `d0-i1` | `cf_bitmask_dp_count_toposorts` | `rust_exec_exact`: Rust code did not produce expected output
  answer: `I don't see any `context` variable provided in the conversation. The only information I've received is the original query asking for a Rust program.  Based on my previous respon...`

### regex_mismatch
- `long_context_books_distractor.json` | `d0-i1` | `books_distractor_compass_700k` | `regex`: regex check did not match
  answer: `I don't know`
- `long_context_books_distractor.json` | `d0-i1` | `books_distractor_compass_700k` | `regex`: regex check did not match
  answer: `I don't have the context about Lantern Shore to answer this question`
- `long_context_books_distractor.json` | `d0-i1` | `books_distractor_compass_700k` | `regex`: regex check did not match
  answer: `I cannot answer this question without the context variable containing the relevant information about Lantern Shore.`

### rust_missing_code
- `codeforces_hard_like.json` | `d0-i1` | `cf_tree_dp_weighted_independent_set` | `rust_exec_exact`: missing compilable Rust answer
  answer: `1304`
- `codeforces_hard_like.json` | `d0-i1` | `cf_tree_dp_weighted_independent_set` | `rust_exec_exact`: missing compilable Rust answer
  answer: `I appreciate the instruction, but I don't have sufficient information to provide an answer. You mentioned "Phase 1 - inspect the `context` variable" but no context variable or p...`
- `codeforces_hard_like.json` | `d0-i1` | `cf_linear_basis_max_xor` | `rust_exec_exact`: missing compilable Rust answer
  answer: `I don't have access to the `context` variable or any integer answer that needs to be printed. No context was provided in our conversation.  Please provide the integer answer tha...`

### wrong_numeric_value
- `mode_profile_targeted.json` | `d0-i1` | `multi_step_math_chain` | `number_exact`: numeric answer mismatch
  answer: `0`
- `mode_profile_targeted.json` | `d0-i1` | `distributional_ledger` | `number_exact`: numeric answer mismatch
  answer: `0`
- `mode_profile_targeted.json` | `d0-i1` | `distributional_ledger` | `number_exact`: numeric answer mismatch
  answer: `0`

### empty_output
- `hallucination_guardrails.json` | `d0-i1` | `grounded_fact_control` | `exact`: empty answer
- `hallucination_guardrails.json` | `d0-i1` | `grounded_fact_control` | `exact`: empty answer
- `hallucination_guardrails.json` | `d0-i3` | `grounded_fact_control` | `exact`: empty answer

### placeholder_output
- `codeforces_hard_like.json` | `d0-i1` | `cf_bitmask_dp_count_toposorts` | `rust_exec_exact`: placeholder token instead of answer
  answer: `None`
- `codeforces_hard_like.json` | `d3-i1` | `cf_cht_dp_value` | `rust_exec_exact`: placeholder token instead of answer
  answer: `answer`
- `mode_profile_targeted.json` | `d6-i1` | `cognitive_trap_bat_ball` | `exact`: placeholder token instead of answer
  answer: `none`

### format_trace_or_scratchpad_leak
- `information_dense_ledger.json` | `d0-i3` | `ledger_max_id_n600_s17` | `exact`: returned scratchpad/tool trace
  answer: `<FunctionCall> struct Tool {\n    tool: "inspect_context",\n    args: {\n        <param name="context_var">"context"</param>    } } </FunctionCall>`
- `information_dense_ledger.json` | `d1-i3` | `ledger_day20plus_sum_n900_s41` | `exact`: returned scratchpad/tool trace
  answer: `<FunctionCall> struct Tool {     tool: "inspect",     args: {         <!--__tool_arg_namespace: Any,-->         variable_name: "context"     } } </FunctionCall>`
- `rlm_hard_coding_planning.json` | `d1-i3` | `long_horizon_incident_state` | `exact`: returned scratchpad/tool trace
  answer: `Let me check for any variables in the environment that might contain the data about ALPHA and BETA states. <tool_call> <invoke name="repl">  <parameter name="command">dir()</par...`

### missing_numeric_answer
- `mode_profile_targeted.json` | `d0-i1` | `distributional_ledger_sum` | `number_exact`: no numeric token in answer
  answer: `<minimax:tool_call> <invoke name="read-variable"> <parameter name="name">context</parameter> </invoke> </minimax:tool_call>`
- `mode_profile_targeted.json` | `d3-i1` | `multi_step_math_chain` | `number_exact`: no numeric token in answer
  answer: `...`
- `mode_profile_targeted.json` | `d3-i1` | `distributional_ledger` | `number_exact`: no numeric token in answer
  answer: `<minimax:tool_call> <invoke name="filesystem_list_allowed_directories"> </invoke> </minimax:tool_call>`

### hallucinated_when_should_abstain
- `hallucination_guardrails.json` | `d1-i3` | `conflicting_serial_numbers` | `exact`: did not output required abstention
  answer: `<minimax:tool_call> <invoke name="filesystem_list_allowed_directories"> </invoke> </minimax:tool_call>`
- `hallucination_guardrails.json` | `d3-i1` | `unsupported_causal_claim` | `exact`: did not output required abstention
  answer: `EMPTY`
