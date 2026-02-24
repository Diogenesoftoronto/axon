use anyhow::Result;
use genai::Client;
use ouros::{Object, ReplProgress};
use regex::Regex;

use crate::llm::{LlmClient, Message};
use crate::prompts;
use crate::sandbox::Sandbox;

pub struct RlmConfig {
    pub client: Client,
    pub model: String,
    pub sub_model: String,
    pub max_iterations: usize,
    pub depth: usize,
    pub max_depth: usize,
    pub verbose: bool,
}

pub struct Rlm {
    config: RlmConfig,
    llm: LlmClient,
}

impl Rlm {
    pub fn new(config: RlmConfig) -> Self {
        let llm = LlmClient::new(config.client.clone(), &config.model);
        Self { config, llm }
    }

    pub async fn completion(&self, query: &str, context: &str) -> Result<String> {
        let mut sandbox = Sandbox::new()?;

        if !context.is_empty() {
            sandbox.set_variable("context", Object::String(context.to_string()))?;
        }

        let mut messages = prompts::build_system_prompt(self.config.depth);

        for iteration in 0..self.config.max_iterations {
            messages.push(prompts::next_action_prompt(query, iteration, false));

            let response = self.llm.completion(&messages).await?;

            if self.config.verbose {
                eprintln!(
                    "[depth={}][iter={}/{}] response: {} chars",
                    self.config.depth,
                    iteration + 1,
                    self.config.max_iterations,
                    response.len()
                );
            }

            let code_blocks = find_code_blocks(&response);
            messages.push(Message::assistant(&response));

            if !code_blocks.is_empty() {
                for code in &code_blocks {
                    let result = self.execute_in_sandbox(&mut sandbox, code).await?;
                    let formatted = format_exec_result(code, &result);
                    messages.push(Message::user(&formatted));
                }
                continue;
            }

            if let Some(answer) = check_final_answer(&response, &sandbox) {
                if self.config.verbose {
                    eprintln!("[depth={}] FINAL: {}...", self.config.depth, &answer[..answer.len().min(120)]);
                }
                return Ok(answer);
            }
        }

        messages.push(prompts::next_action_prompt(query, self.config.max_iterations, true));
        self.llm.completion(&messages).await
    }

    async fn execute_in_sandbox(&self, sandbox: &mut Sandbox, code: &str) -> Result<String> {
        let output = sandbox.execute(code)?;
        let mut stdout = output.stdout;
        let mut progress = output.progress;

        loop {
            match progress {
                ReplProgress::FunctionCall {
                    function_name,
                    args,
                    call_id,
                    ..
                } => {
                    let ret = self.handle_external(sandbox, &function_name, &args).await?;
                    let next = sandbox.resume(call_id, ret)?;
                    stdout.push_str(&next.stdout);
                    progress = next.progress;
                }
                ReplProgress::Complete(_) => break,
                _ => break,
            }
        }

        Ok(stdout)
    }

    async fn handle_external(
        &self,
        sandbox: &Sandbox,
        function_name: &str,
        args: &[Object],
    ) -> Result<Object> {
        match function_name {
            "llm_query" => {
                let prompt = obj_to_string(args.first().unwrap_or(&Object::None));
                self.handle_llm_query(&prompt).await
            }
            "llm_query_batched" => {
                let prompts = obj_to_string_list(args.first().unwrap_or(&Object::None));
                let mut results = Vec::new();
                for p in &prompts {
                    results.push(self.handle_llm_query(p).await?);
                }
                Ok(Object::List(results))
            }
            "FINAL_VAR" => {
                let name = obj_to_string(args.first().unwrap_or(&Object::None));
                let name = name.trim().trim_matches('"').trim_matches('\'');
                Ok(Object::String(
                    sandbox
                        .get_variable(name)
                        .unwrap_or_else(|| format!("Error: Variable '{}' not found", name)),
                ))
            }
            "SHOW_VARS" => {
                let vars = sandbox.list_variables();
                let desc: Vec<String> = vars.iter().map(|(n, t)| format!("{}: {}", n, t)).collect();
                Ok(Object::String(format!(
                    "Available variables: [{}]",
                    desc.join(", ")
                )))
            }
            other => Ok(Object::String(format!(
                "Error: unknown function '{}'",
                other
            ))),
        }
    }

    async fn handle_llm_query(&self, prompt: &str) -> Result<Object> {
        if self.config.depth < self.config.max_depth {
            if self.config.verbose {
                eprintln!(
                    "[depth={}] → spawning sub-RLM at depth {}",
                    self.config.depth,
                    self.config.depth + 1
                );
            }
            let sub = Rlm::new(RlmConfig {
                client: self.config.client.clone(),
                model: self.config.sub_model.clone(),
                sub_model: self.config.sub_model.clone(),
                max_iterations: self.config.max_iterations.min(5),
                depth: self.config.depth + 1,
                max_depth: self.config.max_depth,
                verbose: self.config.verbose,
            });
            let result = Box::pin(sub.completion(
                "Analyze the context and answer the question within it.",
                prompt,
            ))
            .await?;
            Ok(Object::String(result))
        } else {
            if self.config.verbose {
                eprintln!("[depth={}] max depth — direct LLM call", self.config.depth);
            }
            let sub_llm = LlmClient::new(self.config.client.clone(), &self.config.sub_model);
            let result = sub_llm.completion_simple(prompt).await?;
            Ok(Object::String(result))
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn obj_to_string(obj: &Object) -> String {
    match obj {
        Object::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn obj_to_string_list(obj: &Object) -> Vec<String> {
    match obj {
        Object::List(items) | Object::Tuple(items) => items.iter().map(obj_to_string).collect(),
        other => vec![obj_to_string(other)],
    }
}

fn find_code_blocks(text: &str) -> Vec<String> {
    let re = Regex::new(r"(?s)```repl\s*\n(.*?)\n```").unwrap();
    re.captures_iter(text)
        .map(|cap| cap[1].trim().to_string())
        .collect()
}

enum FinalType {
    Direct,
    Variable,
}

fn find_final_answer(text: &str) -> Option<(FinalType, String)> {
    let re_var = Regex::new(r"(?m)^\s*FINAL_VAR\(").unwrap();
    if let Some(m) = re_var.find(text) {
        if let Some(content) = extract_balanced_parens(text, m.end() - 1) {
            return Some((FinalType::Variable, content.trim().to_string()));
        }
    }
    let re_final = Regex::new(r"(?m)^\s*FINAL\(").unwrap();
    if let Some(m) = re_final.find(text) {
        if let Some(content) = extract_balanced_parens(text, m.end() - 1) {
            return Some((FinalType::Direct, content.trim().to_string()));
        }
    }
    None
}

fn extract_balanced_parens(text: &str, start: usize) -> Option<String> {
    let bytes = text.as_bytes();
    if start >= bytes.len() || bytes[start] != b'(' {
        return None;
    }
    let mut depth: i32 = 0;
    for i in start..bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start + 1..i].to_string());
                }
            }
            _ => {}
        }
    }
    Some(text[start + 1..].to_string())
}

fn check_final_answer(response: &str, sandbox: &Sandbox) -> Option<String> {
    let (kind, content) = find_final_answer(response)?;
    Some(match kind {
        FinalType::Direct => content,
        FinalType::Variable => {
            let name = content.trim().trim_matches('"').trim_matches('\'');
            sandbox
                .get_variable(name)
                .unwrap_or_else(|| format!("Error: Variable '{}' not found", name))
        }
    })
}

const MAX_OUTPUT_CHARS: usize = 12_000;

fn format_exec_result(code: &str, stdout: &str) -> String {
    let display = if stdout.len() > MAX_OUTPUT_CHARS {
        format!(
            "{}...[TRUNCATED {} chars]",
            &stdout[..MAX_OUTPUT_CHARS],
            stdout.len() - MAX_OUTPUT_CHARS
        )
    } else if stdout.is_empty() {
        "No output".to_string()
    } else {
        stdout.to_string()
    };
    format!(
        "Code executed:\n```python\n{}\n```\n\nREPL output:\n{}",
        code, display
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_code_blocks_single() {
        let text = "Here is some code:\n```repl\nprint('hello')\n```\nDone.";
        let blocks = find_code_blocks(text);
        assert_eq!(blocks, vec!["print('hello')"]);
    }

    #[test]
    fn test_find_code_blocks_multiple() {
        let text = "```repl\nx = 1\n```\ntext\n```repl\ny = 2\n```";
        let blocks = find_code_blocks(text);
        assert_eq!(blocks, vec!["x = 1", "y = 2"]);
    }

    #[test]
    fn test_find_code_blocks_ignores_other_langs() {
        let text = "```python\nprint('hello')\n```";
        let blocks = find_code_blocks(text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_find_final_answer_direct() {
        let text = "The answer is:\nFINAL(42 is the answer)";
        let (kind, content) = find_final_answer(text).unwrap();
        assert!(matches!(kind, FinalType::Direct));
        assert_eq!(content, "42 is the answer");
    }

    #[test]
    fn test_find_final_answer_variable() {
        let text = "FINAL_VAR(result)";
        let (kind, content) = find_final_answer(text).unwrap();
        assert!(matches!(kind, FinalType::Variable));
        assert_eq!(content, "result");
    }

    #[test]
    fn test_find_final_answer_none() {
        let text = "No final answer here.";
        assert!(find_final_answer(text).is_none());
    }

    #[test]
    fn test_find_final_answer_nested_parens() {
        let text = "FINAL(f(x) + g(y))";
        let (kind, content) = find_final_answer(text).unwrap();
        assert!(matches!(kind, FinalType::Direct));
        assert_eq!(content, "f(x) + g(y)");
    }

    #[test]
    fn test_extract_balanced_parens() {
        assert_eq!(extract_balanced_parens("(hello)", 0), Some("hello".into()));
        assert_eq!(extract_balanced_parens("(a(b)c)", 0), Some("a(b)c".into()));
        assert_eq!(extract_balanced_parens("no parens", 0), None);
    }

    #[test]
    fn test_format_exec_result_normal() {
        let result = format_exec_result("print(1)", "1\n");
        assert!(result.contains("print(1)"));
        assert!(result.contains("1\n"));
    }

    #[test]
    fn test_format_exec_result_empty() {
        let result = format_exec_result("x = 1", "");
        assert!(result.contains("No output"));
    }

    #[test]
    fn test_format_exec_result_truncation() {
        let long_output = "x".repeat(MAX_OUTPUT_CHARS + 1000);
        let result = format_exec_result("code", &long_output);
        assert!(result.contains("TRUNCATED"));
    }
}
