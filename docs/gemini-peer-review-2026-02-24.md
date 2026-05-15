# Gemini Peer Review (2026-02-24)

This note captures a one-shot review of the Altum paper draft and benchmarking plan, generated using Gemini CLI with `gemini-3-pro-preview`.

Command used:

```bash
cat /tmp/gemini_peer_review_prompt.txt | gemini -m gemini-3-pro-preview -p "Follow the tasks A-D precisely. Respond in sections A-D."
```

## Output

Here is the peer review feedback and actionable benchmark expansion plan for Altum.

### A. Paper Peer Review

**General Assessment:**
The paper presents a solid systems engineering contribution (Altum) but currently fails as an empirical research paper due to critically small sample sizes ($N=2$) and lack of variance quantification. It frames itself as an evaluation of RLM physics (scaling, density), but the "Preliminary Ablations" are too thin to support the claims.

**Specific Recommendations:**

1.  **Reframing Contributions:**
    *   **Shift focus from "Science" to "Systems":** The strongest contribution is the open-source Rust engine with isolated sandboxing and telemetry. Frame Altum as a **platform** for RLM research rather than just a reproduction study.
    *   **Clarify "True Recursion":** The distinction between "context-stuffing" and "recursive spawning" is your moat. Explicitly define the state separation: does the child RLM inherit the parent's sandbox variables, or is it a fresh environment? (Code suggests fresh; Paper should justify this design choice regarding state isolation vs. context overhead).

2.  **Experimental Rigor (Critical):**
    *   **Sample Size:** $N=2$ is anecdotal. You must scale to at least $N=50$ per suite to claim any reliability.
    *   **Baselines:** You are comparing `max-depth 2` vs `max-depth 0` (No Recursion). You *must* add a **"Full Context" baseline**: attempting to stuff the entire prompt into the context window (up to model limits) without recursion. This proves recursion is necessary for *processing*, not just fitting context.
    *   **Metrics:** Add **"Time-to-First-Token" (TTFT)** and **"Wall-clock Overhead"**. RLM introduces latency (spawning sandboxes, multiple calls). Quantify the "tax" of recursion.

3.  **Missing Sections:**
    *   **System Profiling:** Add a section measuring the memory footprint and startup time of the Rust binary vs a Python equivalent. This justifies the "Rewriting in Rust" effort.
    *   **Cost Analysis:** The estimated cost section is generic. Add a chart: **Cost vs. Task Complexity**. Does recursion become cheaper on harder tasks because it fails faster or generates more targeted code?

4.  **Editorial Nits:**
    *   **Abstract:** Remove specific pass rate delta ("50 percentage points") unless calculating on a statistically significant set. Say "significant improvement" instead.
    *   **Related Work:** Cite **AutoGPT** or **OpenInterpreter** as non-recursive agentic baselines.

### B. Benchmark Expansions

To move beyond "toy" problems, implement these three generator-based suites.

#### Suite 1: Synthetic Long-Context Retrieval (S-NIAH-XL)
*Goal: Prove RLM handles context larger than the model window.*

**Strategy:** Do not check in 10MB text files. Use a deterministic Python generator in the harness to create the context at runtime.

**Tasks (5 variations):**
1.  **`needle_start`**: 1M chars of log data. UUID injected at index 1000.
    *   *Query:* "Find the UUID tagged 'CRITICAL_REF' in the logs."
    *   *Check:* `exact` match of UUID.
2.  **`needle_end`**: 1M chars. UUID injected at `len - 1000`.
3.  **`needle_middle_depth`**: 500k chars. UUID at 50% depth.
4.  **`multi_needle`**: 3 keys injected at 25%, 50%, 75%.
    *   *Query:* "Concatenate the values for keys Alpha, Beta, Gamma."
    *   *Check:* `exact` concatenation.
5.  **`distributed_counter`**: The string "ERROR_timeout" appears exactly $N$ times spread evenly.
    *   *Query:* "Count the occurrences of 'ERROR_timeout'."
    *   *Check:* `exact` integer string (e.g., "42").

#### Suite 2: Dense Information Aggregation (The "Accountant" Suite)
*Goal: Force the model to read every line, punishing "skimming" or simple search.*

**Context:** A CSV-like textual dump of 500 transaction records (generated).
`[2025-01-01] ID:992 TYPE:Credit AMT:45.00 REGION:US`

**Tasks:**
1.  **`sum_total`**: "Calculate the total sum of all AMT values." (Check: `exact` float/int).
2.  **`conditional_sum`**: "Sum AMT where TYPE is 'Debit'."
3.  **`temporal_filter`**: "Sum AMT for transactions after Jan 15th."
4.  **`variance_check`**: "Find the ID of the transaction with the highest AMT."
5.  **`format_transformation`**: "Convert the first 5 records to JSON format." (Check: `json` validity + key check).

#### Suite 3: Code Repository Understanding (Rust-Mini)
*Goal: Test dependency resolution and structural understanding.*

**Context:** A flat concatenation of 5 Rust files representing a small game logic library (structs, traits, impls).

**Tasks:**
1.  **`struct_deps`**: "List all structs that 'Player' struct depends on."
2.  **`fn_call_graph`**: "What function calls `init_physics`?"
3.  **`trait_impl`**: "Which structs implement the `Renderable` trait?"
4.  **`compile_fix`**: Context has a syntax error (missing semicolon). Query: "Fix the syntax error." (Check: `rust_exec_exact` compilation).
5.  **`feature_flag`**: "What code is gated behind `#[cfg(feature = \"server\")]`?"

### C. Hugging Face Dataset Adapters

I recommend adapting **GSM8K** (Math) and **MBPP** (Coding) for deterministic RLM evaluation.

**1. GSM8K (Math Reasoning)**
*Why:* High density, requires multi-step reasoning (good for recursion), easy regex check.
*Adapter Logic:*
- Download `main` split from `gsm8k`.
- Context: (None, or "You are a math solver").
- Query: The `question` field.
- Check: Extract the number after `####` in the `answer` field. Use `regex` check.

**2. MBPP (Sanitized)**
*Why:* Python coding problems. Altum uses a Python REPL, so it's a native fit.
*Adapter Logic:*
- Download `mbpp` (sanitized).
- Query: `prompt` + "Write a function to solve this. Return ONLY the code."
- Check: This is tricky. Altum harness supports `rust_exec_exact`.
    *   *Modification:* Add `python_exec_check` to harness (runs code + test cases from dataset).
    *   *Workaround:* Ask Altum to translate the Python problem to Rust. Use `test_list` field to generate a Rust `main` function that asserts the logic.
    *   *Check Type:* `rust_exec_exact` (Harness compiles the translation + test runner).

**Script snippet for GSM8K adaptation:**

```python
# scripts/adapt_gsm8k.py
import json
import re
from datasets import load_dataset

ds = load_dataset("gsm8k", "main", split="test")
tasks = []

for i, row in enumerate(ds.select(range(50))):  # Top 50
    # Extract numeric answer
    ans_match = re.search(r"####\\s*(-?\\d+\\.?\\d*)", row[\"answer\"])
    if not ans_match: continue
    
    tasks.append({
        \"id\": f\"gsm8k_{i}\",
        \"context\": \"Solve the math problem step by step.\",
        \"query\": row[\"question\"],
        \"check\": {
            \"type\": \"regex\",
            \"value\": f\"\\\\b{re.escape(ans_match.group(1))}\\\\b\"
        }
    })

print(json.dumps(tasks, indent=2))
```

### D. Multi-Model Evaluation Strategy

**Recommended Model Grid:**
Run the benchmark across these three tiers to test RLM vs. Raw Intelligence vs. Cost.

1.  **Reasoning Heavy:** `fireworks hf:deepseek-ai/DeepSeek-R1-0528` (The gold standard for planning).
2.  **Balanced/Open:** `fireworks hf:meta-llama/Llama-3.3-70B-Instruct` (Standard baseline).
3.  **Long-Context Efficient:** `synthetic hf:Qwen/Qwen3.5-397B-A17B` or `deepseek-ai/DeepSeek-V3` (Cost/performance sweet spot).

**Harness Modifications:**

1.  **Grid Search Support:**
    Update `benchmark_altum.py` to accept a `--model-list` file.
    ```python
    # In argument parsing
    parser.add_argument(\"--model-list\", type=Path, help=\"File with one model per line\")
    
    # In main loop
    models = [args.model] if args.model else []
    if args.model_list:
        models = args.model_list.read_text().splitlines()
    
    for model in models:
        # Run suite, save to results/{sanitized_model_name}/results.json
    ```

2.  **CI Reporting:**
    Output a Markdown summary table (`results/summary.md`) at the end of the run.
    ```markdown
    | Model | Task Suite | Pass Rate | Avg Cost ($) | Avg Time (s) |
    |-------|------------|-----------|--------------|--------------|
    | R1    | Planning   | 85%       | 0.12         | 45.2         |
    | Llama3| Planning   | 60%       | 0.04         | 22.1         |
    ```

This plan provides the necessary experimental volume ($N > 50$), strict validation (Generators/Rust checks), and model diversity to make the paper submission viable.
