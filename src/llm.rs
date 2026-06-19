use anyhow::Result;
use genai::chat::{ChatMessage, ChatRequest, Usage};
use genai::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }

    fn to_chat_message(&self) -> ChatMessage {
        match self.role.as_str() {
            "system" => ChatMessage::system(self.content.as_str()),
            "assistant" => ChatMessage::assistant(self.content.as_str()),
            _ => ChatMessage::user(self.content.as_str()),
        }
    }
}

#[derive(Clone)]
pub struct LlmClient {
    client: Client,
    model: String,
}

pub struct CompletionResult {
    pub text: String,
    pub usage: Usage,
}

impl LlmClient {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    pub async fn completion(&self, messages: &[Message]) -> Result<CompletionResult> {
        let max_retries = 3;
        let mut delay = Duration::from_secs(2);

        for attempt in 0..=max_retries {
            let mut request = ChatRequest::default();
            for m in messages {
                if m.role == "system" {
                    request = request.with_system(m.content.as_str());
                } else {
                    request = request.append_message(m.to_chat_message());
                }
            }

            match self.client.exec_chat(&self.model, request, None).await {
                Ok(response) => {
                    let usage = response.usage.clone();
                    let text = response
                        .into_first_text()
                        .ok_or_else(|| anyhow::anyhow!("Empty response from LLM"))?;
                    return Ok(CompletionResult { text, usage });
                }
                Err(e) if attempt < max_retries => {
                    let err_str = e.to_string();
                    let retryable = ["429", "402", "500", "502", "503", "504"]
                        .iter()
                        .any(|code| err_str.contains(code));
                    if retryable {
                        eprintln!(
                            "LLM request failed (attempt {}/{}), retrying in {:?}: {}",
                            attempt + 1,
                            max_retries,
                            delay,
                            err_str
                        );
                        sleep(delay).await;
                        delay *= 2;
                        continue;
                    }
                    return Err(e.into());
                }
                Err(e) => return Err(e.into()),
            }
        }

        unreachable!()
    }

    pub async fn completion_simple(&self, prompt: &str) -> Result<String> {
        Ok(self.completion(&[Message::user(prompt)]).await?.text)
    }
}
