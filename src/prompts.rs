use crate::llm::Message;
use crate::tools::ToolRegistry;

pub const ROOT_SYSTEM_PROMPT: &str = r#"You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a sandboxed Python REPL that supports recursive sub-LLM calls. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable (str) containing the full text context for analysis. Use Python string operations to process it directly - there is NO filesystem or network access.
2. An `llm_query(prompt)` function that queries a sub-RLM with its own REPL sandbox for semantic analysis of context portions.
3. An `llm_query_batched(prompts)` function - batch version that takes a list of prompt strings and returns a list of responses.
4. Standard Python libraries (re, json, collections, math, etc.) are available via import.
5. Use `print()` statements to view output and continue your reasoning.
6. Avoid Python introspection helpers like `eval()`, `locals()`, and `globals()`; prefer explicit variables and `SHOW_VARS()`.
7. Optional workflow helpers:
   - `CHECKPOINT_CREATE(label)` / `CHECKPOINT_RESTORE(checkpoint_id)`
   - `FORK_CREATE(checkpoint_id, name)` / `FORK_SWITCH(fork_id)` / `FORK_LIST()`
   - `VFS_WRITE(path, content)` / `VFS_READ(path)` / `VFS_LIST(path_prefix)`
   - `STRATEGY_COMMIT(fork_id_or_rationale, rationale_optional)` / `STRATEGY_STATUS()`

IMPORTANT: REPL variables persist across iterations. Do NOT redo work from previous iterations. Build on what you already have.

## Strategy

Follow this three-phase approach:

**Phase 1 - Recon (iteration 1):** Inspect the context with Python code.
- Check `len(context)`, identify the format and natural boundaries.
- For transcripts: find message delimiters like [USER]/[ASSISTANT], ---, or similar markers.
- For structured data: find headers, sections, or record boundaries.
- Use this to plan a smart chunking strategy based on the actual structure.

**Phase 2 - Filter + Analyze (iteration 2):** Use code to narrow the search space, then `llm_query` for semantic reasoning.
- Split the context along natural boundaries found in Phase 1 (NOT arbitrary byte offsets).
- Use regex or keyword search to identify which sections are relevant to the query.
- Call `llm_query` on relevant sections with a focused question. Store results in buffer variables.
- Sub-LLMs are powerful - feed them substantial chunks (10K-50K+ chars). Aim for ~5-10 focused `llm_query` calls, not dozens of tiny ones.

**Phase 3 - Aggregate + Answer (iteration 3):** Synthesize findings and return.
- Use `llm_query` to combine your buffer results into a final answer.
- Return with FINAL_VAR(variable_name) or FINAL(answer text).

## Key Principles

- Use deterministic Python (regex, string ops) to FILTER and NARROW the context. Use `llm_query` to REASON about the filtered content. Code filters, sub-LLMs reason.
- Chunk by document structure (message boundaries, headers, sections), not by arbitrary byte count.
- Each `llm_query` call should ask a specific, focused question about a specific portion of context.
- Never repeat work across iterations. If you already extracted data into a variable, use that variable.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier.

Example:
```repl
import re
text = context
turns = re.split(r'(?=\[(?:USER|ASSISTANT)\])', text)
print(f"Length: {len(text)} chars, {len(turns)} conversation turns")
print("First turn:", turns[0][:500])
```

Example of structure-aware chunking + focused llm_query:
```repl
chunk_size = max(1, len(turns) // 5)
chunks = [turns[i:i+chunk_size] for i in range(0, len(turns), chunk_size)]
buffers = []
for idx, chunk_turns in enumerate(chunks):
    chunk_text = "\n".join(chunk_turns)
    result = llm_query(f"From this conversation segment, extract: [specific question]\n\n{chunk_text}")
    buffers.append(result)
final_answer = llm_query(f"Synthesize these findings to answer: [query]\n\n" + "\n---\n".join(buffers))
print(final_answer)
```
In the next step, return FINAL_VAR(final_answer).

IMPORTANT: When you are done, you MUST provide a final answer inside a FINAL function, NOT in code. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable from the REPL environment

Execute your plan immediately - do not just describe what you will do. Use the REPL and sub-LLMs actively. Remember to explicitly answer the original query in your final answer."#;

pub const SUB_RLM_SYSTEM_PROMPT: &str = r#"You are a sub-RLM tasked with analyzing text. You have access to a sandboxed Python REPL with recursive sub-LLM support.

The REPL environment provides:
1. A `context` variable (str) containing the text to analyze.
2. An `llm_query(prompt)` function for further sub-analysis (spawns another sub-RLM with its own sandbox).
3. Standard Python libraries (re, json, collections, math, etc.).
4. Use `print()` to view output. NO filesystem or network access.
5. Avoid Python introspection helpers like `eval()`, `locals()`, and `globals()`; prefer explicit variables and `SHOW_VARS()`.
6. Optional workflow helpers:
   - `CHECKPOINT_CREATE(label)` / `CHECKPOINT_RESTORE(checkpoint_id)`
   - `FORK_CREATE(checkpoint_id, name)` / `FORK_SWITCH(fork_id)` / `FORK_LIST()`
   - `VFS_WRITE(path, content)` / `VFS_READ(path)` / `VFS_LIST(path_prefix)`
   - `STRATEGY_COMMIT(fork_id_or_rationale, rationale_optional)` / `STRATEGY_STATUS()`

Analyze the context and provide your answer. You can use code in ```repl``` blocks if the text is large or needs programmatic processing. For shorter texts, you may answer directly.

When done, use FINAL(your answer) or FINAL_VAR(variable_name)."#;

pub fn build_system_prompt(depth: usize, tool_registry: &ToolRegistry) -> Vec<Message> {
    let mut prompt = if depth == 0 {
        ROOT_SYSTEM_PROMPT.to_string()
    } else {
        SUB_RLM_SYSTEM_PROMPT.to_string()
    };

    if !tool_registry.is_empty() {
        let mut custom_section =
            "\n\n## Custom Tools\n\nThe following custom tools are also available in the REPL:\n"
                .to_string();
        for spec in tool_registry.tools().values() {
            custom_section.push_str(&format!("- `{}(...)`: {}\n", spec.name, spec.description));
        }
        custom_section.push_str("Call these functions directly by name, passing arguments as you would any Python function.\n");
        prompt.push_str(&custom_section);
    }

    vec![Message::system(&prompt)]
}

const USER_PROMPT_ITER0: &str = r#"Start with Phase 1 - inspect the `context` variable: check its size, identify the format and natural chunk boundaries.

Original query: "{query}""#;

const USER_PROMPT_CONTINUE: &str = r#"Your REPL variables from previous iterations are still available - do NOT redo previous work. Build on what you have.

Continue working toward answering: "{query}"

Your next action:"#;

const USER_PROMPT_FINAL: &str =
    "Based on all the information you have gathered, provide your final answer now. Use FINAL(answer) or FINAL_VAR(variable_name).";

pub fn next_action_prompt(query: &str, iteration: usize, force_final: bool) -> Message {
    if force_final {
        return Message::user(USER_PROMPT_FINAL);
    }
    let content = if iteration == 0 {
        USER_PROMPT_ITER0.replace("{query}", query)
    } else {
        USER_PROMPT_CONTINUE.replace("{query}", query)
    };
    Message::user(&content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{ToolRegistry, ToolSpec};
    use proptest::prelude::*;

    #[test]
    fn test_build_system_prompt_empty_registry_root() {
        let msgs = build_system_prompt(0, &ToolRegistry::new());
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.contains("llm_query"));
        assert!(!msgs[0].content.contains("Custom Tools"));
    }

    #[test]
    fn test_build_system_prompt_empty_registry_sub() {
        let msgs = build_system_prompt(1, &ToolRegistry::new());
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].content.contains("sub-RLM"));
        assert!(!msgs[0].content.contains("Custom Tools"));
    }

    #[test]
    fn test_build_system_prompt_with_custom_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec {
            name: "SEARCH".to_string(),
            description: "search things".to_string(),
            input_schema: None,
        });
        let msgs = build_system_prompt(0, &reg);
        assert!(msgs[0].content.contains("## Custom Tools"));
        assert!(msgs[0].content.contains("`SEARCH(...)`"));
        assert!(msgs[0].content.contains("search things"));
    }

    #[test]
    fn test_build_system_prompt_custom_tool_in_both_depths() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec {
            name: "TOOL_A".to_string(),
            description: "tool a desc".to_string(),
            input_schema: None,
        });
        let root = build_system_prompt(0, &reg);
        let sub = build_system_prompt(1, &reg);
        assert!(root[0].content.contains("TOOL_A"));
        assert!(sub[0].content.contains("TOOL_A"));
    }

    proptest! {
        #[test]
        fn test_next_action_prompt_force_final_always_same(query in ".*") {
            let msg = next_action_prompt(&query, 42, true);
            assert_eq!(msg.role, "user");
            assert_eq!(msg.content, USER_PROMPT_FINAL);
        }

        #[test]
        fn test_next_action_prompt_iter0_contains_query(query in "[^{}]{0,50}") {
            let msg = next_action_prompt(&query, 0, false);
            assert!(msg.content.contains(&query), "iter0 prompt should contain the query");
        }

        #[test]
        fn test_next_action_prompt_continue_contains_query(query in "[^{}]{0,50}") {
            let msg = next_action_prompt(&query, 5, false);
            assert!(msg.content.contains(&query), "continue prompt should contain the query");
        }

        #[test]
        fn test_build_prompt_custom_tools_appear_in_output(
            names in proptest::collection::vec("[a-zA-Z_][a-zA-Z0-9_]{0,9}", 1..5),
            desc in ".*"
        ) {
            let mut reg = ToolRegistry::new();
            for name in &names {
                reg.register(ToolSpec {
                    name: name.clone(),
                    description: desc.clone(),
                    input_schema: None,
                });
            }
            let msgs = build_system_prompt(0, &reg);
            let content = &msgs[0].content;
            assert!(content.contains("## Custom Tools"));
            for name in &names {
                assert!(content.contains(&format!("`{}(...)`", name)), "should contain tool '{}' in prompt", name);
            }
        }

        #[test]
        fn test_build_prompt_empty_registry_no_custom_section(depth in 0usize..2) {
            let msgs = build_system_prompt(depth, &ToolRegistry::new());
            assert!(!msgs[0].content.contains("## Custom Tools"));
        }
    }
}
