use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use genai::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::policy::RuntimePolicy;
use crate::service::{QueryRequest, RlmService};
use crate::store::ContextStore;

#[derive(Clone)]
pub struct OpenAiServer {
    service: RlmService,
}

#[derive(Clone)]
struct OpenAiState {
    service: RlmService,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsRequest {
    model: Option<String>,
    messages: Vec<OpenAiMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    metadata: Option<HashMap<String, Value>>,
    #[serde(default)]
    user: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    role: String,
    #[serde(default)]
    content: Option<MessageContent>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponsesRequest {
    model: Option<String>,
    input: ResponseInput,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    metadata: Option<HashMap<String, Value>>,
    #[serde(default)]
    user: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResponseInput {
    Text(String),
    Messages(Vec<OpenAiMessage>),
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

impl OpenAiServer {
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
        policy_catalog: crate::policy::PolicyCatalog,
        default_runtime_policy: RuntimePolicy,
    ) -> Self {
        Self {
            service: RlmService::new(
                client,
                model,
                sub_model,
                max_iterations,
                max_depth,
                store,
                verbose,
                trace_sandbox,
                policy_catalog,
                default_runtime_policy,
            ),
        }
    }

    pub async fn serve(self, bind_addr: SocketAddr) -> Result<()> {
        let state = Arc::new(OpenAiState {
            service: self.service,
        });
        let app = Router::new()
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/responses", post(responses))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind(bind_addr).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }
}

async fn list_models(State(state): State<Arc<OpenAiState>>) -> Response {
    let created = unix_timestamp();
    json_response(
        StatusCode::OK,
        &ModelsResponse {
            object: "list",
            data: vec![ModelInfo {
                id: state.service.model.clone(),
                object: "model",
                created,
                owned_by: "axon",
            }],
        },
    )
}

async fn chat_completions(
    State(state): State<Arc<OpenAiState>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionsRequest>,
) -> Response {
    let Some(query) = latest_user_message(&req.messages) else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "messages must include user content",
        );
    };
    let thread_id = thread_id(req.metadata.as_ref(), req.user.as_deref(), &headers);
    let request_context = render_messages(&req.messages);

    match run_rlm_query(
        &state,
        req.model.as_deref(),
        &thread_id,
        &query,
        &request_context,
    )
    .await
    {
        Ok(answer) if req.stream => chat_stream_response(&state, req.model.as_deref(), &answer),
        Ok(answer) => chat_json_response(&state, req.model.as_deref(), &answer),
        Err(err) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
    }
}

async fn responses(
    State(state): State<Arc<OpenAiState>>,
    headers: HeaderMap,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    let (query, rendered_input) = match response_input_parts(&req.input) {
        Some(parts) => parts,
        None => return error_response(StatusCode::BAD_REQUEST, "input must include user content"),
    };
    let thread_id = thread_id(req.metadata.as_ref(), req.user.as_deref(), &headers);
    let request_context = if let Some(instructions) = req.instructions.as_deref() {
        format!("SYSTEM: {}\n{}", instructions, rendered_input)
    } else {
        rendered_input
    };

    match run_rlm_query(
        &state,
        req.model.as_deref(),
        &thread_id,
        &query,
        &request_context,
    )
    .await
    {
        Ok(answer) if req.stream => {
            responses_stream_response(&state, req.model.as_deref(), &answer)
        }
        Ok(answer) => responses_json_response(&state, req.model.as_deref(), &answer),
        Err(err) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
    }
}

async fn run_rlm_query(
    state: &OpenAiState,
    model_override: Option<&str>,
    thread_id: &str,
    query: &str,
    request_context: &str,
) -> Result<String> {
    let req = QueryRequest {
        query: query.to_string(),
        thread_id: thread_id.to_string(),
        request_context: Some(request_context.to_string()),
        model_override: model_override.map(|s| s.to_string()),
        sub_model_override: None,
        policy_profile: None,
        inject_policy_into_context: None,
        depth_enforcement: None,
        require_min_depth: None,
        require_min_recursive_calls: None,
    };
    state.service.query(req).await
}

fn latest_user_message(messages: &[OpenAiMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(|message| message_text(message).filter(|text| !text.trim().is_empty()))
}

fn response_input_parts(input: &ResponseInput) -> Option<(String, String)> {
    match input {
        ResponseInput::Text(text) if !text.trim().is_empty() => {
            Some((text.clone(), format!("USER: {}", text)))
        }
        ResponseInput::Text(_) => None,
        ResponseInput::Messages(messages) => {
            let query = latest_user_message(messages)?;
            Some((query, render_messages(messages)))
        }
    }
}

fn render_messages(messages: &[OpenAiMessage]) -> String {
    messages
        .iter()
        .filter_map(|message| {
            let text = message_text(message)?;
            if text.trim().is_empty() {
                return None;
            }
            Some(format!("{}: {}", message.role.to_uppercase(), text))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn message_text(message: &OpenAiMessage) -> Option<String> {
    match message.content.as_ref()? {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Parts(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            Some(text)
        }
    }
}

fn thread_id(
    metadata: Option<&HashMap<String, Value>>,
    user: Option<&str>,
    headers: &HeaderMap,
) -> String {
    if let Some(value) = headers
        .get("x-axon-thread")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
    {
        return value.trim().to_string();
    }
    if let Some(value) = metadata
        .and_then(|metadata| metadata.get("thread_id"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        return value.trim().to_string();
    }
    user.filter(|value| !value.trim().is_empty())
        .unwrap_or("default")
        .trim()
        .to_string()
}

fn chat_json_response(state: &OpenAiState, request_model: Option<&str>, answer: &str) -> Response {
    let created = unix_timestamp();
    let model = request_model.unwrap_or(&state.service.model);
    json_response(
        StatusCode::OK,
        &json!({
            "id": response_id("chatcmpl"),
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }]
        }),
    )
}

fn chat_stream_response(
    state: &OpenAiState,
    request_model: Option<&str>,
    answer: &str,
) -> Response {
    let created = unix_timestamp();
    let id = response_id("chatcmpl");
    let model = request_model.unwrap_or(&state.service.model);
    let first = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": answer
            },
            "finish_reason": null
        }]
    });
    let done = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    });
    sse_response(vec![first, done])
}

fn responses_json_response(
    state: &OpenAiState,
    request_model: Option<&str>,
    answer: &str,
) -> Response {
    let created = unix_timestamp();
    let id = response_id("resp");
    let model = request_model.unwrap_or(&state.service.model);
    json_response(
        StatusCode::OK,
        &json!({
            "id": id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": model,
            "output": [{
                "id": response_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": answer,
                    "annotations": []
                }]
            }],
            "output_text": answer
        }),
    )
}

fn responses_stream_response(
    state: &OpenAiState,
    request_model: Option<&str>,
    answer: &str,
) -> Response {
    let created = unix_timestamp();
    let id = response_id("resp");
    let model = request_model.unwrap_or(&state.service.model);
    sse_response(vec![
        json!({
            "type": "response.output_text.delta",
            "response_id": id,
            "delta": answer
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": id,
                "object": "response",
                "created_at": created,
                "status": "completed",
                "model": model,
                "output_text": answer
            }
        }),
    ])
}

fn sse_response(events: Vec<Value>) -> Response {
    let mut body = String::new();
    for event in events {
        body.push_str("data: ");
        body.push_str(&event.to_string());
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    response_with_content_type(StatusCode::OK, "text/event-stream", body)
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Response {
    let body = serde_json::to_string(value).expect("JSON serialization should not fail");
    response_with_content_type(status, "application/json", body)
}

fn error_response(status: StatusCode, message: &str) -> Response {
    json_response(
        status,
        &json!({
            "error": {
                "message": message,
                "type": "server_error",
                "param": null,
                "code": null
            }
        }),
    )
}

fn response_with_content_type(
    status: StatusCode,
    content_type: &'static str,
    body: String,
) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(body))
        .expect("response builder should accept static headers")
}

fn response_id(prefix: &str) -> String {
    format!("{}-{}", prefix, unix_timestamp_nanos())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_timestamp_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_message(role: &str, content: &str) -> OpenAiMessage {
        OpenAiMessage {
            role: role.to_string(),
            content: Some(MessageContent::Text(content.to_string())),
        }
    }

    #[test]
    fn latest_user_message_uses_last_user_message() {
        let messages = vec![
            text_message("user", "first"),
            text_message("assistant", "middle"),
            text_message("user", "second"),
        ];

        assert_eq!(latest_user_message(&messages), Some("second".to_string()));
    }

    #[test]
    fn thread_id_prefers_header_then_metadata_then_user() {
        let mut headers = HeaderMap::new();
        headers.insert("x-axon-thread", "header-thread".parse().unwrap());
        let mut metadata = HashMap::new();
        metadata.insert(
            "thread_id".to_string(),
            Value::String("meta-thread".to_string()),
        );

        assert_eq!(
            thread_id(Some(&metadata), Some("user-thread"), &headers),
            "header-thread"
        );

        let headers = HeaderMap::new();
        assert_eq!(
            thread_id(Some(&metadata), Some("user-thread"), &headers),
            "meta-thread"
        );

        assert_eq!(
            thread_id(None, Some("user-thread"), &headers),
            "user-thread"
        );
        assert_eq!(thread_id(None, None, &headers), "default");
    }

    #[test]
    fn render_messages_includes_roles() {
        let messages = vec![
            text_message("system", "be terse"),
            text_message("user", "hello"),
        ];

        assert_eq!(render_messages(&messages), "SYSTEM: be terse\nUSER: hello");
    }
}
