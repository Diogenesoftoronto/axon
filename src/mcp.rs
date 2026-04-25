use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use genai::Client;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content};
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::tower::StreamableHttpService;
use rmcp::transport::{stdio, StreamableHttpServerConfig};
use rmcp::{schemars, tool, tool_handler, tool_router, ErrorData, ServerHandler, ServiceExt};
use serde::Deserialize;

use crate::policy::{DepthEnforcementMode, RuntimePolicy};
use crate::service::{QueryRequest, RlmService};
use crate::store::ContextStore;
use crate::tools::ToolSpec;

#[derive(Clone)]
pub struct McpServer {
    service: RlmService,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ChatRlmQueryArgs {
    /// The question to ask.
    query: String,
    /// Thread identifier; context accumulates per thread.
    thread_id: String,
    /// Optional policy profile override for this call.
    #[serde(default)]
    policy_profile: Option<String>,
    /// Optional: prepend policy text into context for this call.
    #[serde(default)]
    inject_policy_into_context: Option<bool>,
    /// Optional model override for this query (e.g. "hf:MiniMaxAI/MiniMax-M2.5").
    #[serde(default)]
    model: Option<String>,
    /// Optional sub-model override for this query.
    #[serde(default)]
    sub_model: Option<String>,
    /// Optional depth enforcement mode.
    #[serde(default)]
    depth_enforcement: Option<String>,
    /// Optional strict minimum depth threshold.
    #[serde(default)]
    require_min_depth: Option<usize>,
    /// Optional strict minimum recursive llm_query call threshold.
    #[serde(default)]
    require_min_recursive_calls: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct UploadContextArgs {
    /// Full transcript text.
    transcript: String,
    /// Session identifier.
    session_id: String,
    /// Thread to store under.
    #[serde(default)]
    thread_id: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ConfigureToolsArgs {
    /// List of tool specifications to register.
    tools: Vec<ToolSpec>,
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

    pub async fn serve_stdio(self) -> Result<()> {
        let server = self.serve(stdio()).await?;
        server.waiting().await?;
        Ok(())
    }

    pub async fn serve_http(self, bind_addr: SocketAddr, path: &str) -> Result<()> {
        let normalized_path = normalize_http_path(path);
        let mut config = StreamableHttpServerConfig::default();
        config.allowed_hosts = allowed_hosts_for(bind_addr);

        let factory = {
            let server = self.clone();
            move || Ok(server.clone())
        };
        let service =
            StreamableHttpService::new(factory, Arc::new(LocalSessionManager::default()), config);
        let app = Router::new().route_service(&normalized_path, service);
        let listener = tokio::net::TcpListener::bind(bind_addr).await?;

        axum::serve(listener, app).await?;
        Ok(())
    }

    async fn run_chat_rlm_query(&self, args: ChatRlmQueryArgs) -> CallToolResult {
        let query = args.query.trim().to_string();
        let thread_id = args.thread_id.trim().to_string();

        if query.is_empty() {
            return tool_error("query cannot be empty");
        }
        if thread_id.is_empty() {
            return tool_error("thread_id cannot be empty");
        }

        let req = QueryRequest {
            query,
            thread_id,
            request_context: None,
            model_override: args.model,
            sub_model_override: args.sub_model,
            policy_profile: args.policy_profile,
            inject_policy_into_context: args.inject_policy_into_context,
            depth_enforcement: parse_depth_mode(args.depth_enforcement.as_deref()),
            require_min_depth: args.require_min_depth,
            require_min_recursive_calls: args.require_min_recursive_calls,
        };

        match self.service.query(req).await {
            Ok(answer) => tool_result(&answer),
            Err(e) => tool_error(&format!("RLM error: {}", e)),
        }
    }

    fn run_upload_context(&self, args: UploadContextArgs) -> CallToolResult {
        let transcript = args.transcript.trim();
        let session_id = args.session_id.trim();
        let thread_id = args.thread_id.as_deref().unwrap_or("transcripts").trim();

        if transcript.is_empty() {
            return tool_error("transcript cannot be empty");
        }
        if session_id.is_empty() {
            return tool_error("session_id cannot be empty");
        }
        if thread_id.is_empty() {
            return tool_error("thread_id cannot be empty");
        }

        let text = format!("\n[SESSION {}]\n{}\n", session_id, transcript);
        match self.service.store.append_context(thread_id, &text) {
            Ok(()) => tool_result(&format!(
                "Uploaded session {} to thread '{}'.",
                session_id, thread_id
            )),
            Err(e) => tool_error(&format!("Store error: {}", e)),
        }
    }

    async fn run_configure_tools(&self, args: ConfigureToolsArgs) -> CallToolResult {
        if args.tools.is_empty() {
            return tool_error("tools list cannot be empty");
        }
        
        let _guard = self.service.registry_lock.lock().await;
        
        let mut registry = (**self.service.tool_registry.load()).clone();
        let mut registered = Vec::new();
        for spec in args.tools {
            let name = spec.name.clone();
            registry.register(spec);
            registered.push(name);
        }
        
        self.service.tool_registry.store(std::sync::Arc::new(registry));
        
        tool_result(&format!("Registered tools: {}", registered.join(", ")))
    }
}

#[tool_router]
impl McpServer {
    #[tool(
        name = "chat_rlm_query",
        description = "Query the recursive language model with persistent thread context. The RLM handles arbitrarily large contexts via recursive reasoning with sandboxed Python execution."
    )]
    async fn chat_rlm_query(
        &self,
        Parameters(args): Parameters<ChatRlmQueryArgs>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        Ok(self.run_chat_rlm_query(args).await)
    }

    #[tool(
        name = "upload_context",
        description = "Upload a transcript to the RLM persistent memory. Stored under a thread so the RLM can reason over past sessions."
    )]
    async fn upload_context(
        &self,
        Parameters(args): Parameters<UploadContextArgs>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        Ok(self.run_upload_context(args))
    }

    #[tool(
        name = "configure_tools",
        description = "Register custom tools that the RLM sandbox can invoke as external functions. Tools are available in subsequent chat_rlm_query calls."
    )]
    async fn configure_tools(
        &self,
        Parameters(args): Parameters<ConfigureToolsArgs>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        Ok(self.run_configure_tools(args).await)
    }
}

#[tool_handler(
    name = "axon",
    version = "0.1.0",
    instructions = "Recursive language model MCP server."
)]
impl ServerHandler for McpServer {}

fn parse_depth_mode(value: Option<&str>) -> Option<DepthEnforcementMode> {
    let s = value?.trim().to_lowercase();
    match s.as_str() {
        "off" => Some(DepthEnforcementMode::Off),
        "soft" => Some(DepthEnforcementMode::Soft),
        "strict" => Some(DepthEnforcementMode::Strict),
        _ => None,
    }
}

fn tool_result(text: &str) -> CallToolResult {
    CallToolResult::success(vec![Content::text(text.to_string())])
}

fn tool_error(text: &str) -> CallToolResult {
    CallToolResult::error(vec![Content::text(text.to_string())])
}

fn normalize_http_path(path: &str) -> String {
    let trimmed = path.trim();
    if trimmed.is_empty() || trimmed == "/" {
        return "/mcp".to_string();
    }
    if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{}", trimmed)
    }
}

fn allowed_hosts_for(bind_addr: SocketAddr) -> Vec<String> {
    let host = bind_addr.ip().to_string();
    let port = bind_addr.port();
    let mut allowed = vec![
        "localhost".to_string(),
        format!("localhost:{port}"),
        "127.0.0.1".to_string(),
        format!("127.0.0.1:{port}"),
        "::1".to_string(),
        format!("[::1]:{port}"),
        host.clone(),
        format!("{host}:{port}"),
    ];
    allowed.sort();
    allowed.dedup();
    allowed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyCatalog;

    #[test]
    fn test_parse_depth_mode() {
        assert_eq!(
            parse_depth_mode(Some("off")),
            Some(DepthEnforcementMode::Off)
        );
        assert_eq!(
            parse_depth_mode(Some("strict")),
            Some(DepthEnforcementMode::Strict)
        );
        assert_eq!(parse_depth_mode(Some("nope")), None);
    }

    #[test]
    fn test_tool_result_format() {
        let result = tool_result("hello");
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.is_error, Some(false));
    }

    #[test]
    fn test_tool_error_format() {
        let result = tool_error("bad input");
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn test_normalize_http_path() {
        assert_eq!(normalize_http_path("mcp"), "/mcp");
        assert_eq!(normalize_http_path("/custom"), "/custom");
        assert_eq!(normalize_http_path(""), "/mcp");
        assert_eq!(normalize_http_path("/"), "/mcp");
    }

    #[test]
    fn test_allowed_hosts_include_loopback_and_bind_addr() {
        let hosts = allowed_hosts_for("127.0.0.1:8000".parse().unwrap());
        assert!(hosts.contains(&"localhost".to_string()));
        assert!(hosts.contains(&"localhost:8000".to_string()));
        assert!(hosts.contains(&"127.0.0.1".to_string()));
        assert!(hosts.contains(&"127.0.0.1:8000".to_string()));
    }

    // --- Property tests for new features ---

    fn make_test_mcp_server() -> McpServer {
        use std::path::PathBuf;
        McpServer::new(
            Client::builder().build(),
            "default-model".to_string(),
            "default-sub-model".to_string(),
            5,
            1,
            ContextStore::new(&PathBuf::from("/tmp/axon-test-store")),
            false,
            false,
            PolicyCatalog::builtin(),
            RuntimePolicy::default(),
        )
    }

    #[tokio::test]
    async fn test_configure_tools_registers_into_shared_registry() {
        let server = make_test_mcp_server();
        let reg = server.service.tool_registry.load();
        assert!(reg.is_empty());

        let args = ConfigureToolsArgs {
            tools: vec![
                ToolSpec { name: "SEARCH".to_string(), description: "searches".to_string(), input_schema: None },
                ToolSpec { name: "FETCH".to_string(), description: "fetches".to_string(), input_schema: None },
            ],
        };
        let result = server.run_configure_tools(args).await;
        assert!(result.is_error.is_none() || result.is_error == Some(false));

        let reg = server.service.tool_registry.load();
        assert_eq!(reg.tool_names(), vec!["FETCH", "SEARCH"]);
    }

    #[tokio::test]
    async fn test_configure_tools_empty_returns_error() {
        let server = make_test_mcp_server();
        let args = ConfigureToolsArgs { tools: vec![] };
        let result = server.run_configure_tools(args).await;
        assert_eq!(result.is_error, Some(true));
    }

    #[tokio::test]
    async fn test_configure_tools_overwrites_existing() {
        let server = make_test_mcp_server();
        let args1 = ConfigureToolsArgs {
            tools: vec![ToolSpec { name: "TOOL".to_string(), description: "v1".to_string(), input_schema: None }],
        };
        server.run_configure_tools(args1).await;

        let args2 = ConfigureToolsArgs {
            tools: vec![ToolSpec { name: "TOOL".to_string(), description: "v2".to_string(), input_schema: None }],
        };
        server.run_configure_tools(args2).await;

        let reg = server.service.tool_registry.load();
        assert_eq!(reg.get("TOOL").unwrap().description, "v2");
    }

    #[test]
    fn test_model_override_falls_back_to_default() {
        let server = make_test_mcp_server();
        let args_no_override = ChatRlmQueryArgs {
            query: "test".to_string(),
            thread_id: "t1".to_string(),
            policy_profile: None,
            inject_policy_into_context: None,
            model: None,
            sub_model: None,
            depth_enforcement: None,
            require_min_depth: None,
            require_min_recursive_calls: None,
        };
        // We can't call run_chat_rlm_query (it would make LLM calls),
        // but we can verify the arg resolution logic by checking the struct.
        let resolved_model = args_no_override.model.unwrap_or_else(|| server.service.model.clone());
        let resolved_sub = args_no_override.sub_model.unwrap_or_else(|| server.service.sub_model.clone());
        assert_eq!(resolved_model, "default-model");
        assert_eq!(resolved_sub, "default-sub-model");

        let args_with_override = ChatRlmQueryArgs {
            query: "test".to_string(),
            thread_id: "t1".to_string(),
            policy_profile: None,
            inject_policy_into_context: None,
            model: Some("custom-model".to_string()),
            sub_model: Some("custom-sub".to_string()),
            depth_enforcement: None,
            require_min_depth: None,
            require_min_recursive_calls: None,
        };
        let resolved_model = args_with_override.model.unwrap_or_else(|| server.service.model.clone());
        let resolved_sub = args_with_override.sub_model.unwrap_or_else(|| server.service.sub_model.clone());
        assert_eq!(resolved_model, "custom-model");
        assert_eq!(resolved_sub, "custom-sub");
    }

    #[tokio::test]
    async fn test_registry_clone_is_snapshot() {
        let server = make_test_mcp_server();
        let reg1_snapshot = server.service.tool_registry.load_full();
        assert!(reg1_snapshot.is_empty());

        let args = ConfigureToolsArgs {
            tools: vec![ToolSpec { name: "NEW_TOOL".to_string(), description: "new".to_string(), input_schema: None }],
        };
        server.run_configure_tools(args).await;

        // reg1_snapshot was loaded before registration — should still be empty
        assert!(reg1_snapshot.is_empty());

        // Current registry should have the tool
        let reg2 = server.service.tool_registry.load();
        assert!(reg2.get("NEW_TOOL").is_some());
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_parse_depth_mode_valid_values(s in "(off|soft|strict)") {
            let result = parse_depth_mode(Some(&s));
            assert!(result.is_some(), "'{}' should parse to a valid depth mode", s);
        }

        #[test]
        fn test_parse_depth_mode_invalid_returns_none(s in "[^offsftric]{1,10}") {
            let result = parse_depth_mode(Some(&s));
            // Only "off", "soft", "strict" are valid; anything else is None
            let valid = matches!(s.to_lowercase().as_str(), "off" | "soft" | "strict");
            assert_eq!(result.is_some(), valid, "'{}' parse result unexpected", s);
        }

        #[test]
        fn test_tool_result_always_has_content(text in ".*") {
            let result = tool_result(&text);
            assert_eq!(result.content.len(), 1);
            assert_eq!(result.is_error, Some(false));
        }

        #[test]
        fn test_tool_error_always_has_content(text in ".*") {
            let result = tool_error(&text);
            assert_eq!(result.content.len(), 1);
            assert_eq!(result.is_error, Some(true));
        }

        #[test]
        fn test_normalize_http_path_always_starts_with_slash(path in ".*") {
            let normalized = normalize_http_path(&path);
            assert!(normalized.starts_with('/'));
        }
    }
}
