use arc_swap::ArcSwap;
use std::sync::Arc;
use tokio::sync::Mutex;

use anyhow::Result;
use genai::Client;

use crate::policy::{DepthEnforcementMode, PolicyCatalog, RuntimePolicy};
use crate::rlm::{Rlm, RlmConfig};
use crate::store::ContextStore;
use crate::tools::ToolRegistry;

#[derive(Clone)]
pub struct RlmService {
    pub client: Client,
    pub model: String,
    pub sub_model: String,
    pub max_iterations: usize,
    pub max_depth: usize,
    pub store: ContextStore,
    pub verbose: bool,
    pub trace_sandbox: bool,
    pub policy_catalog: PolicyCatalog,
    pub default_runtime_policy: RuntimePolicy,
    pub tool_registry: Arc<ArcSwap<ToolRegistry>>,
    pub registry_lock: Arc<Mutex<()>>,
}

pub struct QueryRequest {
    pub query: String,
    pub thread_id: String,
    pub request_context: Option<String>,
    pub model_override: Option<String>,
    pub sub_model_override: Option<String>,
    pub policy_profile: Option<String>,
    pub inject_policy_into_context: Option<bool>,
    pub depth_enforcement: Option<DepthEnforcementMode>,
    pub require_min_depth: Option<usize>,
    pub require_min_recursive_calls: Option<usize>,
}

impl RlmService {
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
        policy_catalog: PolicyCatalog,
        default_runtime_policy: RuntimePolicy,
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
            policy_catalog,
            default_runtime_policy,
            tool_registry: Arc::new(ArcSwap::from_pointee(ToolRegistry::new())),
            registry_lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn query(&self, req: QueryRequest) -> Result<String> {
        let query = req.query.trim();
        let thread_id = req.thread_id.trim();

        if query.is_empty() {
            anyhow::bail!("query cannot be empty");
        }
        if thread_id.is_empty() {
            anyhow::bail!("thread_id cannot be empty");
        }

        let stored_context = self.store.read_context(thread_id);

        let context = if let Some(req_ctx) = req.request_context {
            if stored_context.trim().is_empty() {
                req_ctx
            } else {
                format!("{stored_context}\n\n[CURRENT REQUEST]\n{req_ctx}")
            }
        } else {
            stored_context
        };

        let mut runtime_policy = if let Some(profile) = req.policy_profile.as_deref() {
            self.policy_catalog
                .build_runtime_policy(Some(profile), None, None, None, None)
        } else {
            self.default_runtime_policy.clone()
        };

        if let Some(v) = req.inject_policy_into_context {
            runtime_policy.inject_policy_into_context = v;
        }
        if let Some(v) = req.depth_enforcement {
            runtime_policy.depth_enforcement = v;
        }
        if let Some(v) = req.require_min_depth {
            runtime_policy.require_min_depth = Some(v);
        }
        if let Some(v) = req.require_min_recursive_calls {
            runtime_policy.require_min_recursive_calls = Some(v);
        }

        let model = req.model_override.unwrap_or_else(|| self.model.clone());
        let sub_model = req
            .sub_model_override
            .unwrap_or_else(|| self.sub_model.clone());
        let tool_registry = self.tool_registry.load_full();

        let rlm = Rlm::new(RlmConfig {
            client: self.client.clone(),
            model,
            sub_model,
            max_iterations: self.max_iterations,
            depth: 0,
            max_depth: self.max_depth,
            verbose: self.verbose,
            trace_sandbox: self.trace_sandbox,
            runtime_policy,
            tool_registry,
            trace_sink: None,
        });

        let answer = tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("RLM runtime");
            rt.block_on(rlm.completion(query, &context))
        })?;

        self.store.append_context(
            thread_id,
            &format!("\nUSER: {}\nASSISTANT: {}\n", query, answer),
        )?;

        Ok(answer)
    }
}
