use std::io::{self, BufRead, Write};

use anyhow::Result;
use genai::Client;
use serde_json::{json, Value};

use crate::rlm::{Rlm, RlmConfig};
use crate::store::ContextStore;

pub struct McpServer {
    client: Client,
    model: String,
    sub_model: String,
    max_iterations: usize,
    max_depth: usize,
    store: ContextStore,
    verbose: bool,
    trace_sandbox: bool,
}

impl McpServer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: Client,
        model: String,
        sub_model: String,
        max_iterations: usize,
        max_depth: usize,
        store: ContextStore,
        verbose: bool,
        trace_sandbox: bool,
    ) -> Self {
        Self {
            client,
            model,
            sub_model,
            max_iterations,
            max_depth,
            store,
            verbose,
            trace_sandbox,
        }
    }

    pub async fn run(&self) -> Result<()> {
        let stdin = io::stdin();
        let mut reader = io::BufReader::new(stdin.lock());
        let stdout = io::stdout();
        let mut writer = io::BufWriter::new(stdout.lock());

        loop {
            let msg = match read_message(&mut reader) {
                Ok(m) => m,
                Err(_) => break,
            };

            let method = msg["method"].as_str().unwrap_or("");
            let id = msg.get("id").cloned();

            let response = match method {
                "initialize" => Some(json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": { "tools": {} },
                    "serverInfo": {
                        "name": "axon",
                        "version": "0.1.0"
                    }
                })),
                "notifications/initialized" | "notifications/cancelled" => None,
                "ping" => Some(json!({})),
                "tools/list" => Some(json!({
                    "tools": [
                        {
                            "name": "chat_rlm_query",
                            "description": "Query the recursive language model with persistent thread context. The RLM handles arbitrarily large contexts via recursive reasoning with sandboxed Python execution.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": { "type": "string", "description": "The question to ask" },
                                    "thread_id": { "type": "string", "description": "Thread identifier - context accumulates per thread" }
                                },
                                "required": ["query", "thread_id"]
                            }
                        },
                        {
                            "name": "upload_context",
                            "description": "Upload a transcript to the RLM persistent memory. Stored under a thread so the RLM can reason over past sessions.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "transcript": { "type": "string", "description": "Full transcript text" },
                                    "session_id": { "type": "string", "description": "Session identifier" },
                                    "thread_id": { "type": "string", "description": "Thread to store under", "default": "transcripts" }
                                },
                                "required": ["transcript", "session_id"]
                            }
                        }
                    ]
                })),
                "tools/call" => {
                    let params = &msg["params"];
                    let tool_name = params["name"].as_str().unwrap_or("");
                    let args = &params["arguments"];
                    Some(self.handle_tool_call(tool_name, args).await)
                }
                _ => id
                    .as_ref()
                    .map(|_| json!({ "error": { "code": -32601, "message": "Method not found" } })),
            };

            if let (Some(id), Some(result)) = (id, response) {
                let resp = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": result
                });
                write_message(&mut writer, &resp)?;
            }
        }

        Ok(())
    }

    async fn handle_tool_call(&self, tool_name: &str, args: &Value) -> Value {
        match tool_name {
            "chat_rlm_query" => {
                let query = args["query"].as_str().unwrap_or("").trim();
                let thread_id = args["thread_id"].as_str().unwrap_or("default");

                if query.is_empty() {
                    return tool_error("query cannot be empty");
                }

                let context = self.store.read_context(thread_id);

                let rlm = Rlm::new(RlmConfig {
                    client: self.client.clone(),
                    model: self.model.clone(),
                    sub_model: self.sub_model.clone(),
                    max_iterations: self.max_iterations,
                    depth: 0,
                    max_depth: self.max_depth,
                    verbose: self.verbose,
                    trace_sandbox: self.trace_sandbox,
                });

                match rlm.completion(query, &context).await {
                    Ok(answer) => {
                        let _ = self.store.append_context(
                            thread_id,
                            &format!("\nUSER: {}\nASSISTANT: {}\n", query, answer),
                        );
                        tool_result(&answer)
                    }
                    Err(e) => tool_error(&format!("RLM error: {}", e)),
                }
            }
            "upload_context" => {
                let transcript = args["transcript"].as_str().unwrap_or("").trim();
                let session_id = args["session_id"].as_str().unwrap_or("").trim();
                let thread_id = args["thread_id"].as_str().unwrap_or("transcripts");

                if transcript.is_empty() {
                    return tool_error("transcript cannot be empty");
                }
                if session_id.is_empty() {
                    return tool_error("session_id cannot be empty");
                }

                let text = format!("\n[SESSION {}]\n{}\n", session_id, transcript);
                match self.store.append_context(thread_id, &text) {
                    Ok(()) => tool_result(&format!(
                        "Uploaded session {} to thread '{}'.",
                        session_id, thread_id
                    )),
                    Err(e) => tool_error(&format!("Store error: {}", e)),
                }
            }
            _ => tool_error(&format!("Unknown tool: {}", tool_name)),
        }
    }
}

fn tool_result(text: &str) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }]
    })
}

fn tool_error(text: &str) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": true
    })
}

// ---------------------------------------------------------------------------
// Content-Length framed JSON-RPC I/O
// ---------------------------------------------------------------------------

fn read_message(reader: &mut impl BufRead) -> Result<Value> {
    let mut content_length: usize = 0;
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            anyhow::bail!("EOF");
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            content_length = val.trim().parse()?;
        }
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok(serde_json::from_slice(&body)?)
}

fn write_message(writer: &mut impl Write, msg: &Value) -> Result<()> {
    let body = serde_json::to_string(msg)?;
    write!(writer, "Content-Length: {}\r\n\r\n{}", body.len(), body)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_write_and_read_message() {
        let msg = json!({"jsonrpc": "2.0", "method": "ping", "id": 1});
        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();

        let mut reader = Cursor::new(buf);
        let parsed = read_message(&mut reader).unwrap();
        assert_eq!(parsed["method"], "ping");
        assert_eq!(parsed["id"], 1);
    }

    #[test]
    fn test_read_message_eof() {
        let mut reader = Cursor::new(Vec::<u8>::new());
        assert!(read_message(&mut reader).is_err());
    }

    #[test]
    fn test_tool_result_format() {
        let result = tool_result("hello");
        assert_eq!(result["content"][0]["text"], "hello");
        assert!(result.get("isError").is_none());
    }

    #[test]
    fn test_tool_error_format() {
        let result = tool_error("bad input");
        assert_eq!(result["content"][0]["text"], "bad input");
        assert_eq!(result["isError"], true);
    }
}
