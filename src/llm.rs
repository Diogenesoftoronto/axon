use anyhow::Result;
use genai::chat::{ChatMessage, ChatRequest, Usage};
use genai::Client;

#[derive(Clone, Debug)]
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
        let mut request = ChatRequest::default();

        for m in messages {
            if m.role == "system" {
                request = request.with_system(m.content.as_str());
            } else {
                request = request.append_message(m.to_chat_message());
            }
        }

        let response = self.client.exec_chat(&self.model, request, None).await?;
        let usage = response.usage.clone();
        let text = response
            .into_first_text()
            .ok_or_else(|| anyhow::anyhow!("Empty response from LLM"))?;
        Ok(CompletionResult { text, usage })
    }

    pub async fn completion_simple(&self, prompt: &str) -> Result<String> {
        Ok(self.completion(&[Message::user(prompt)]).await?.text)
    }
}
