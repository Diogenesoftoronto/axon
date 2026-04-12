use anyhow::Result;
use genai::Client;
use ouros::{Object, ReplProgress};
use regex::Regex;
use std::collections::{BTreeMap, HashMap};

use crate::llm::{LlmClient, Message};
use crate::policy::{DepthEnforcementMode, RuntimePolicy};
use crate::prompts;
use crate::sandbox::{Sandbox, MAIN_SESSION_ID};

pub struct RlmConfig {
    pub client: Client,
    pub model: String,
    pub sub_model: String,
    pub max_iterations: usize,
    pub depth: usize,
    pub max_depth: usize,
    pub verbose: bool,
    pub trace_sandbox: bool,
    pub runtime_policy: RuntimePolicy,
}

pub struct Rlm {
    config: RlmConfig,
    llm: LlmClient,
}

#[derive(Clone, Default)]
struct ForkState {
    session_id: String,
    code_blocks: Vec<String>,
    external_returns: Vec<Object>,
    vfs: BTreeMap<String, String>,
    strategy_commit: Option<String>,
}

#[derive(Clone)]
struct CheckpointState {
    fork_id: String,
    session_id: String,
    code_len: usize,
    return_len: usize,
    code_blocks: Vec<String>,
    external_returns: Vec<Object>,
    vfs_snapshot: BTreeMap<String, String>,
    label: String,
}

#[derive(Clone)]
enum PendingSandboxAction {
    CreateCheckpointSession {
        source_session_id: String,
        checkpoint_session_id: String,
    },
    RestoreFork {
        fork_id: String,
        target_session_id: String,
        source_session_id: String,
        code_blocks: Vec<String>,
        external_returns: Vec<Object>,
        vfs_snapshot: BTreeMap<String, String>,
    },
    SwitchFork {
        fork_id: String,
        session_id: String,
    },
}

struct RuntimeState {
    active_fork: String,
    forks: HashMap<String, ForkState>,
    checkpoints: HashMap<String, CheckpointState>,
    next_fork_id: usize,
    next_checkpoint_id: usize,
    pending_sandbox_actions: Vec<PendingSandboxAction>,
}

#[derive(Clone, Debug)]
struct DepthTelemetry {
    max_depth_reached: usize,
    recursive_call_count: usize,
    recursive_calls_by_depth: BTreeMap<usize, usize>,
}

impl DepthTelemetry {
    fn new(current_depth: usize) -> Self {
        Self {
            max_depth_reached: current_depth,
            recursive_call_count: 0,
            recursive_calls_by_depth: BTreeMap::new(),
        }
    }

    fn record_spawn(&mut self, child_depth: usize) {
        self.recursive_call_count += 1;
        self.max_depth_reached = self.max_depth_reached.max(child_depth);
        *self
            .recursive_calls_by_depth
            .entry(child_depth)
            .or_insert(0) += 1;
    }

    fn merge_child(&mut self, child: &DepthTelemetry) {
        self.max_depth_reached = self.max_depth_reached.max(child.max_depth_reached);
        self.recursive_call_count += child.recursive_call_count;
        for (depth, count) in &child.recursive_calls_by_depth {
            *self.recursive_calls_by_depth.entry(*depth).or_insert(0) += count;
        }
    }
}

impl RuntimeState {
    fn new(_context: &str) -> Self {
        let mut forks = HashMap::new();
        forks.insert(
            "main".to_string(),
            ForkState {
                session_id: MAIN_SESSION_ID.to_string(),
                ..ForkState::default()
            },
        );
        Self {
            active_fork: "main".to_string(),
            forks,
            checkpoints: HashMap::new(),
            next_fork_id: 1,
            next_checkpoint_id: 1,
            pending_sandbox_actions: Vec::new(),
        }
    }

    fn active_fork(&self) -> Option<&ForkState> {
        self.forks.get(&self.active_fork)
    }

    fn active_fork_mut(&mut self) -> Option<&mut ForkState> {
        self.forks.get_mut(&self.active_fork)
    }
}

impl Rlm {
    pub fn new(config: RlmConfig) -> Self {
        let llm = LlmClient::new(config.client.clone(), &config.model);
        Self { config, llm }
    }

    pub async fn completion(&self, query: &str, context: &str) -> Result<String> {
        let (answer, _) = self.completion_with_telemetry(query, context).await?;
        Ok(answer)
    }

    async fn completion_with_telemetry(
        &self,
        query: &str,
        context: &str,
    ) -> Result<(String, DepthTelemetry)> {
        self.validate_depth_policy_feasibility()?;
        let mut sandbox = Sandbox::new()?;
        let mut runtime = RuntimeState::new(context);
        let mut telemetry = DepthTelemetry::new(self.config.depth);

        let mut effective_context = context.to_string();
        if self.config.runtime_policy.inject_policy_into_context {
            effective_context = self.config.runtime_policy.prepend_to_context(context);
        }

        if !effective_context.is_empty() {
            sandbox.set_variable("context", Object::String(effective_context))?;
        }

        let mut messages = prompts::build_system_prompt(self.config.depth);
        if let Some(policy_block) = self.config.runtime_policy.prompt_instruction_block() {
            messages.push(Message::system(&policy_block));
        }

        for iteration in 0..self.config.max_iterations {
            messages.push(prompts::next_action_prompt(query, iteration, false));

            let prompt_chars: usize = messages.iter().map(|m| m.content.len()).sum();
            let completion = self.llm.completion(&messages).await?;
            let response = completion.text;

            if self.config.verbose {
                eprintln!(
                    "[depth={}][iter={}/{}] response: {} chars",
                    self.config.depth,
                    iteration + 1,
                    self.config.max_iterations,
                    response.len()
                );
                eprintln!(
                    "[depth={}][iter={}/{}] usage: prompt_tokens={} completion_tokens={} total_tokens={} prompt_chars={}",
                    self.config.depth,
                    iteration + 1,
                    self.config.max_iterations,
                    completion.usage.prompt_tokens.unwrap_or(0),
                    completion.usage.completion_tokens.unwrap_or(0),
                    completion.usage.total_tokens.unwrap_or(0),
                    prompt_chars
                );
            }

            let code_blocks = find_code_blocks(&response);
            messages.push(Message::assistant(&response));

            if !code_blocks.is_empty() {
                for (idx, code) in code_blocks.iter().enumerate() {
                    if self.config.trace_sandbox {
                        eprintln!(
                            "[depth={}][iter={}/{}][sandbox][block={}] code:\n{}",
                            self.config.depth,
                            iteration + 1,
                            self.config.max_iterations,
                            idx + 1,
                            code
                        );
                    }
                    let result = self
                        .execute_in_sandbox(&mut sandbox, code, &mut runtime, &mut telemetry)
                        .await?;
                    if let Some(fork) = runtime.active_fork_mut() {
                        fork.code_blocks.push(code.to_string());
                    }
                    let formatted = format_exec_result(code, &result);
                    messages.push(Message::user(&formatted));
                }
            }

            if let Some(answer) = check_final_answer(&response, &sandbox) {
                if let Some(reason) = self.depth_policy_unmet_reason(&telemetry) {
                    if self.config.runtime_policy.depth_enforcement == DepthEnforcementMode::Strict
                    {
                        if iteration + 1 < self.config.max_iterations {
                            messages.push(Message::user(&format!(
                                "Policy requirement not met: {}. Continue with recursive llm_query calls before FINAL(...).",
                                reason
                            )));
                            continue;
                        }
                        anyhow::bail!("Depth policy unmet: {}", reason);
                    }
                }
                if self.config.verbose {
                    eprintln!(
                        "[depth={}] FINAL: {}...",
                        self.config.depth,
                        &answer[..answer.len().min(120)]
                    );
                }
                return Ok((strip_final_wrapper(&answer), telemetry));
            }
        }

        messages.push(prompts::next_action_prompt(
            query,
            self.config.max_iterations,
            true,
        ));
        let completion = self.llm.completion(&messages).await?;
        let response = &completion.text;

        // Execute any repl code blocks in the fallback response before extracting.
        let fallback_code_blocks = find_code_blocks(response);
        for code in &fallback_code_blocks {
            let _ = self
                .execute_in_sandbox(&mut sandbox, code, &mut runtime, &mut telemetry)
                .await;
        }

        if let Some(answer) = check_final_answer(response, &sandbox) {
            if let Some(reason) = self.depth_policy_unmet_reason(&telemetry) {
                if self.config.runtime_policy.depth_enforcement == DepthEnforcementMode::Strict {
                    anyhow::bail!("Depth policy unmet: {}", reason);
                }
            }
            return Ok((strip_final_wrapper(&answer), telemetry));
        }
        if let Some(reason) = self.depth_policy_unmet_reason(&telemetry) {
            if self.config.runtime_policy.depth_enforcement == DepthEnforcementMode::Strict {
                anyhow::bail!("Depth policy unmet: {}", reason);
            }
        }
        Ok((strip_final_wrapper(&completion.text), telemetry))
    }

    fn validate_depth_policy_feasibility(&self) -> Result<()> {
        if self.config.runtime_policy.depth_enforcement != DepthEnforcementMode::Strict {
            return Ok(());
        }

        if let Some(min_depth) = self.config.runtime_policy.require_min_depth {
            if min_depth > self.config.max_depth {
                anyhow::bail!(
                    "Depth policy invalid: require_min_depth={} exceeds max_depth={}",
                    min_depth,
                    self.config.max_depth
                );
            }
        }

        if let Some(min_calls) = self.config.runtime_policy.require_min_recursive_calls {
            if min_calls > 0 && self.config.depth >= self.config.max_depth {
                anyhow::bail!(
                    "Depth policy invalid: require_min_recursive_calls={} but current depth {} has no remaining recursion budget (max_depth={})",
                    min_calls,
                    self.config.depth,
                    self.config.max_depth
                );
            }
        }

        Ok(())
    }

    fn depth_policy_unmet_reason(&self, telemetry: &DepthTelemetry) -> Option<String> {
        if self.config.runtime_policy.depth_enforcement == DepthEnforcementMode::Off {
            return None;
        }
        if let Some(min_depth) = self.config.runtime_policy.require_min_depth {
            if telemetry.max_depth_reached < min_depth {
                return Some(format!(
                    "max depth reached {} < required {}",
                    telemetry.max_depth_reached, min_depth
                ));
            }
        }
        if let Some(min_calls) = self.config.runtime_policy.require_min_recursive_calls {
            if telemetry.recursive_call_count < min_calls {
                return Some(format!(
                    "recursive call count {} < required {}",
                    telemetry.recursive_call_count, min_calls
                ));
            }
        }
        None
    }

    async fn execute_in_sandbox(
        &self,
        sandbox: &mut Sandbox,
        code: &str,
        runtime: &mut RuntimeState,
        telemetry: &mut DepthTelemetry,
    ) -> Result<String> {
        let output = match sandbox.execute(code) {
            Ok(out) => out,
            Err(err) => {
                // Keep the outer RLM loop alive so model-produced FINAL(...) can still be parsed.
                // This avoids turning single REPL mistakes into hard run failures.
                return Ok(format!("Sandbox execution error: {}", err));
            }
        };
        let mut stdout = output.stdout;
        let mut progress = output.progress;

        if self.config.trace_sandbox && !stdout.is_empty() {
            eprintln!(
                "[depth={}][sandbox] stdout: {}",
                self.config.depth,
                truncate_for_trace(&stdout, 600)
            );
        }

        loop {
            match progress {
                ReplProgress::FunctionCall {
                    function_name,
                    args,
                    call_id,
                    ..
                } => {
                    if self.config.trace_sandbox {
                        eprintln!(
                            "[depth={}][sandbox] external call: {}({})",
                            self.config.depth,
                            function_name,
                            describe_args(&args)
                        );
                    }
                    let ret = self
                        .handle_external(sandbox, &function_name, &args, runtime, telemetry)
                        .await?;
                    if self.config.trace_sandbox {
                        eprintln!(
                            "[depth={}][sandbox] external return: {}",
                            self.config.depth,
                            truncate_for_trace(&ret.to_string(), 300)
                        );
                    }
                    if let Some(fork) = runtime.active_fork_mut() {
                        fork.external_returns.push(ret.clone());
                    }
                    let next = match sandbox.resume(call_id, ret) {
                        Ok(next) => next,
                        Err(err) => {
                            let msg = format!("\nSandbox resume error: {}", err);
                            stdout.push_str(&msg);
                            if self.config.trace_sandbox {
                                eprintln!(
                                    "[depth={}][sandbox] {}",
                                    self.config.depth,
                                    truncate_for_trace(&msg, 300)
                                );
                            }
                            break;
                        }
                    };
                    stdout.push_str(&next.stdout);
                    if self.config.trace_sandbox && !next.stdout.is_empty() {
                        eprintln!(
                            "[depth={}][sandbox] stdout: {}",
                            self.config.depth,
                            truncate_for_trace(&next.stdout, 600)
                        );
                    }
                    progress = next.progress;
                }
                ReplProgress::Complete(_) => {
                    if self.config.trace_sandbox {
                        eprintln!(
                            "[depth={}][sandbox] complete; vars: {}",
                            self.config.depth,
                            describe_vars(sandbox)
                        );
                    }
                    break;
                }
                _ => {
                    if self.config.trace_sandbox {
                        eprintln!(
                            "[depth={}][sandbox] non-complete progress; vars: {}",
                            self.config.depth,
                            describe_vars(sandbox)
                        );
                    }
                    break;
                }
            }
        }

        self.apply_pending_sandbox_actions(sandbox, runtime)?;
        Ok(stdout)
    }

    fn apply_pending_sandbox_actions(
        &self,
        sandbox: &mut Sandbox,
        runtime: &mut RuntimeState,
    ) -> Result<()> {
        let pending = std::mem::take(&mut runtime.pending_sandbox_actions);
        for action in pending {
            match action {
                PendingSandboxAction::CreateCheckpointSession {
                    source_session_id,
                    checkpoint_session_id,
                } => {
                    sandbox.fork_session(&source_session_id, &checkpoint_session_id)?;
                }
                PendingSandboxAction::RestoreFork {
                    fork_id,
                    target_session_id,
                    source_session_id,
                    code_blocks,
                    external_returns,
                    vfs_snapshot,
                } => {
                    if let Some(fork) = runtime.forks.get_mut(&fork_id) {
                        fork.code_blocks = code_blocks;
                        fork.external_returns = external_returns;
                        fork.vfs = vfs_snapshot;
                    } else {
                        anyhow::bail!("fork '{}' not found during pending restore", fork_id);
                    }
                    runtime.active_fork = fork_id;
                    sandbox.replace_session_from(&target_session_id, &source_session_id)?;
                    sandbox.switch_session(&target_session_id)?;
                }
                PendingSandboxAction::SwitchFork {
                    fork_id,
                    session_id,
                } => {
                    runtime.active_fork = fork_id;
                    sandbox.switch_session(&session_id)?;
                }
            }
        }
        Ok(())
    }

    async fn handle_external(
        &self,
        sandbox: &mut Sandbox,
        function_name: &str,
        args: &[Object],
        runtime: &mut RuntimeState,
        telemetry: &mut DepthTelemetry,
    ) -> Result<Object> {
        match function_name {
            "llm_query" => {
                let prompt = obj_to_string(args.first().unwrap_or(&Object::None));
                self.handle_llm_query(&prompt, telemetry).await
            }
            "llm_query_batched" => {
                let prompts = obj_to_string_list(args.first().unwrap_or(&Object::None));
                let mut results = Vec::new();
                for p in &prompts {
                    results.push(self.handle_llm_query(p, telemetry).await?);
                }
                Ok(Object::List(results))
            }
            "FINAL_VAR" => {
                let name = obj_to_string(args.first().unwrap_or(&Object::None));
                let name = name.trim().trim_matches('"').trim_matches('\'');
                Ok(Object::String(sandbox.get_variable(name).unwrap_or_else(
                    || format!("Error: Variable '{}' not found", name),
                )))
            }
            "FINAL" => Ok(Object::String(obj_to_string(
                args.first().unwrap_or(&Object::None),
            ))),
            "SHOW_VARS" => {
                let vars = sandbox.list_variables();
                let desc: Vec<String> = vars.iter().map(|(n, t)| format!("{}: {}", n, t)).collect();
                Ok(Object::String(format!(
                    "Available variables: [{}]",
                    desc.join(", ")
                )))
            }
            "CHECKPOINT_CREATE" => {
                let label = obj_to_string(args.first().unwrap_or(&Object::None));
                let checkpoint_num = runtime.next_checkpoint_id;
                let checkpoint_id = format!("checkpoint-{}", checkpoint_num);
                let checkpoint_session_id = format!("checkpoint-session-{}", checkpoint_num);

                let Some(active_fork) = runtime.active_fork() else {
                    return Ok(Object::String("Error: active fork not found".to_string()));
                };
                let source_session_id = active_fork.session_id.clone();
                let code_blocks = active_fork.code_blocks.clone();
                let external_returns = active_fork.external_returns.clone();
                let vfs_snapshot = active_fork.vfs.clone();

                runtime.pending_sandbox_actions.push(
                    PendingSandboxAction::CreateCheckpointSession {
                        source_session_id,
                        checkpoint_session_id: checkpoint_session_id.clone(),
                    },
                );
                runtime.next_checkpoint_id += 1;

                runtime.checkpoints.insert(
                    checkpoint_id.clone(),
                    CheckpointState {
                        fork_id: runtime.active_fork.clone(),
                        session_id: checkpoint_session_id,
                        code_len: code_blocks.len(),
                        return_len: external_returns.len(),
                        code_blocks,
                        external_returns,
                        vfs_snapshot,
                        label,
                    },
                );
                Ok(Object::String(checkpoint_id))
            }
            "CHECKPOINT_RESTORE" => {
                let checkpoint_id = obj_to_string(args.first().unwrap_or(&Object::None));
                let checkpoint_id = checkpoint_id.trim().trim_matches('"').trim_matches('\'');
                let Some(checkpoint) = runtime.checkpoints.get(checkpoint_id).cloned() else {
                    return Ok(Object::String(format!(
                        "Error: checkpoint '{}' not found",
                        checkpoint_id
                    )));
                };
                let Some(target_session_id) = runtime
                    .forks
                    .get(&checkpoint.fork_id)
                    .map(|f| f.session_id.clone())
                else {
                    return Ok(Object::String(format!(
                        "Error: fork '{}' not found",
                        checkpoint.fork_id
                    )));
                };
                runtime
                    .pending_sandbox_actions
                    .push(PendingSandboxAction::RestoreFork {
                        fork_id: checkpoint.fork_id.clone(),
                        target_session_id: target_session_id.clone(),
                        source_session_id: checkpoint.session_id.clone(),
                        code_blocks: checkpoint.code_blocks.clone(),
                        external_returns: checkpoint.external_returns.clone(),
                        vfs_snapshot: checkpoint.vfs_snapshot.clone(),
                    });
                Ok(Object::String(format!(
                    "Restored checkpoint '{}'",
                    checkpoint_id
                )))
            }
            "FORK_CREATE" => {
                let checkpoint_id = obj_to_string(args.first().unwrap_or(&Object::None));
                let checkpoint_id = checkpoint_id.trim().trim_matches('"').trim_matches('\'');
                let _name = obj_to_string(args.get(1).unwrap_or(&Object::None));

                let checkpoint = if let Some(c) = runtime.checkpoints.get(checkpoint_id) {
                    c.clone()
                } else {
                    return Ok(Object::String(format!(
                        "Error: checkpoint '{}' not found",
                        checkpoint_id
                    )));
                };
                let fork_num = runtime.next_fork_id;
                let fork_id = format!("fork-{}", fork_num);
                let fork_session_id = format!("fork-session-{}", fork_num);
                if let Err(err) = sandbox.fork_session(&checkpoint.session_id, &fork_session_id) {
                    return Ok(Object::String(format!(
                        "Error: failed to create fork session: {}",
                        err
                    )));
                }
                runtime.next_fork_id += 1;
                let mut new_fork = ForkState {
                    session_id: fork_session_id,
                    ..ForkState::default()
                };
                new_fork.code_blocks = checkpoint.code_blocks.clone();
                new_fork.external_returns = checkpoint.external_returns.clone();
                new_fork.vfs = checkpoint.vfs_snapshot.clone();
                runtime.forks.insert(fork_id.clone(), new_fork);
                Ok(Object::String(fork_id))
            }
            "FORK_SWITCH" => {
                let fork_id = obj_to_string(args.first().unwrap_or(&Object::None));
                let fork_id = fork_id.trim().trim_matches('"').trim_matches('\'');
                let Some(target_session_id) =
                    runtime.forks.get(fork_id).map(|f| f.session_id.clone())
                else {
                    return Ok(Object::String(format!(
                        "Error: fork '{}' not found",
                        fork_id
                    )));
                };
                runtime
                    .pending_sandbox_actions
                    .push(PendingSandboxAction::SwitchFork {
                        fork_id: fork_id.to_string(),
                        session_id: target_session_id,
                    });
                Ok(Object::String(format!("Switched to fork '{}'", fork_id)))
            }
            "FORK_LIST" => {
                let mut lines = Vec::new();
                for (id, fork) in &runtime.forks {
                    let active = if *id == runtime.active_fork { "*" } else { " " };
                    let commit = fork.strategy_commit.as_deref().unwrap_or("-");
                    lines.push(format!(
                        "{} {} (code={}, returns={}, committed={})",
                        active,
                        id,
                        fork.code_blocks.len(),
                        fork.external_returns.len(),
                        truncate_for_trace(commit, 60)
                    ));
                }
                lines.sort();
                Ok(Object::String(lines.join("\n")))
            }
            "VFS_WRITE" => {
                let path = obj_to_string(args.first().unwrap_or(&Object::None));
                let content = obj_to_string(args.get(1).unwrap_or(&Object::None));
                let path = normalize_vfs_path(&path);
                if let Some(fork) = runtime.active_fork_mut() {
                    fork.vfs.insert(path.clone(), content);
                }
                Ok(Object::String(format!("Wrote {}", path)))
            }
            "VFS_READ" => {
                let path = obj_to_string(args.first().unwrap_or(&Object::None));
                let path = normalize_vfs_path(&path);
                if let Some(fork) = runtime.active_fork() {
                    return Ok(Object::String(
                        fork.vfs.get(&path).cloned().unwrap_or_default(),
                    ));
                }
                Ok(Object::String(String::new()))
            }
            "VFS_LIST" => {
                let prefix = obj_to_string(args.first().unwrap_or(&Object::None));
                let prefix = normalize_vfs_prefix(&prefix);
                let mut paths = Vec::new();
                if let Some(fork) = runtime.active_fork() {
                    for key in fork.vfs.keys() {
                        if key.starts_with(&prefix) {
                            paths.push(key.clone());
                        }
                    }
                }
                paths.sort();
                Ok(Object::String(paths.join("\n")))
            }
            "STRATEGY_COMMIT" => {
                let (fork_id, rationale) = if args.len() >= 2 {
                    (
                        obj_to_string(args.first().unwrap_or(&Object::None)),
                        obj_to_string(args.get(1).unwrap_or(&Object::None)),
                    )
                } else {
                    (
                        runtime.active_fork.clone(),
                        obj_to_string(args.first().unwrap_or(&Object::None)),
                    )
                };
                let fork_id = fork_id
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string();
                if let Some(fork) = runtime.forks.get_mut(&fork_id) {
                    fork.strategy_commit = Some(rationale.clone());
                    Ok(Object::String(format!(
                        "Committed strategy on '{}'",
                        fork_id
                    )))
                } else {
                    Ok(Object::String(format!(
                        "Error: fork '{}' not found",
                        fork_id
                    )))
                }
            }
            "STRATEGY_STATUS" => {
                let mut lines = vec![format!("active_fork={}", runtime.active_fork)];
                let mut checkpoints: Vec<_> = runtime.checkpoints.iter().collect();
                checkpoints.sort_by_key(|(id, _)| *id);
                for (id, cp) in checkpoints {
                    lines.push(format!(
                        "checkpoint {}: fork={}, code_len={}, return_len={}, label={}",
                        id, cp.fork_id, cp.code_len, cp.return_len, cp.label
                    ));
                }
                let mut forks: Vec<_> = runtime.forks.iter().collect();
                forks.sort_by_key(|(id, _)| *id);
                for (id, fork) in forks {
                    let committed = fork.strategy_commit.as_deref().unwrap_or("-");
                    lines.push(format!(
                        "fork {}: code={}, returns={}, committed={}",
                        id,
                        fork.code_blocks.len(),
                        fork.external_returns.len(),
                        truncate_for_trace(committed, 80)
                    ));
                }
                Ok(Object::String(lines.join("\n")))
            }
            other => Ok(Object::String(format!(
                "Error: unknown function '{}'",
                other
            ))),
        }
    }

    async fn handle_llm_query(
        &self,
        prompt: &str,
        telemetry: &mut DepthTelemetry,
    ) -> Result<Object> {
        if self.config.depth < self.config.max_depth {
            let child_depth = self.config.depth + 1;
            telemetry.record_spawn(child_depth);
            if self.config.verbose {
                eprintln!(
                    "[depth={}] → spawning sub-RLM at depth {}",
                    self.config.depth, child_depth
                );
            }
            let sub = Rlm::new(RlmConfig {
                client: self.config.client.clone(),
                model: self.config.sub_model.clone(),
                sub_model: self.config.sub_model.clone(),
                max_iterations: self.config.max_iterations.min(5),
                depth: child_depth,
                max_depth: self.config.max_depth,
                verbose: self.config.verbose,
                trace_sandbox: self.config.trace_sandbox,
                runtime_policy: self.config.runtime_policy.for_sub_rlm(),
            });
            let (result, child_telemetry) = Box::pin(sub.completion_with_telemetry(
                "Analyze the context and answer the question within it.",
                prompt,
            ))
            .await?;
            telemetry.merge_child(&child_telemetry);
            Ok(Object::String(result))
        } else {
            if self.config.verbose {
                eprintln!("[depth={}] max depth - direct LLM call", self.config.depth);
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
    if let Some(i) = text.find("FINAL_VAR(") {
        let start = i + "FINAL_VAR".len();
        if let Some(content) = extract_balanced_parens(text, start) {
            return Some((FinalType::Variable, content.trim().to_string()));
        }
    }
    if let Some(i) = text.find("FINAL(") {
        let start = i + "FINAL".len();
        if let Some(content) = extract_balanced_parens(text, start) {
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

fn strip_hallucinated_tool_calls(text: &str) -> String {
    let re = Regex::new(
        r"(?s)<minimax:tool_call>.*?</minimax:tool_call>|<FunctionCall>.*?</FunctionCall>|<invoke\s+name=[^>]*>.*?</invoke>",
    )
    .unwrap();
    re.replace_all(text, "").trim().to_string()
}

fn strip_final_wrapper(text: &str) -> String {
    let trimmed = text.trim();
    if let Some((_, content)) = find_final_answer(trimmed) {
        return strip_hallucinated_tool_calls(&content);
    }
    strip_hallucinated_tool_calls(trimmed)
}

fn check_final_answer(response: &str, sandbox: &Sandbox) -> Option<String> {
    let (kind, content) = find_final_answer(response)?;
    Some(match kind {
        FinalType::Direct => strip_hallucinated_tool_calls(&content),
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

fn truncate_for_trace(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        format!(
            "{}...[+{} chars]",
            &text[..max_chars],
            text.len() - max_chars
        )
    }
}

fn describe_args(args: &[Object]) -> String {
    args.iter()
        .map(|a| truncate_for_trace(&a.to_string(), 200))
        .collect::<Vec<_>>()
        .join(", ")
}

fn describe_vars(sandbox: &Sandbox) -> String {
    let vars = sandbox.list_variables();
    if vars.is_empty() {
        return "none".to_string();
    }
    let mut parts = Vec::new();
    for (name, ty) in vars.into_iter().take(12) {
        let value = sandbox
            .get_variable(&name)
            .unwrap_or_else(|| "<unavailable>".to_string());
        parts.push(format!(
            "{}:{}={}",
            name,
            ty,
            truncate_for_trace(&value, 120)
        ));
    }
    if parts.len() == 12 {
        format!("{} ...", parts.join(", "))
    } else {
        parts.join(", ")
    }
}

fn normalize_vfs_path(path: &str) -> String {
    let trimmed = path.trim().trim_matches('"').trim_matches('\'');
    if trimmed.is_empty() {
        return "/".to_string();
    }
    if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{}", trimmed)
    }
}

fn normalize_vfs_prefix(path: &str) -> String {
    let mut p = normalize_vfs_path(path);
    if !p.ends_with('/') && p != "/" {
        p.push('/');
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{DepthEnforcementMode, RuntimePolicy};
    use genai::Client;

    fn make_test_rlm() -> Rlm {
        let client = Client::builder().build();
        Rlm::new(RlmConfig {
            client,
            model: "test-model".to_string(),
            sub_model: "test-sub-model".to_string(),
            max_iterations: 1,
            depth: 0,
            max_depth: 0,
            verbose: false,
            trace_sandbox: false,
            runtime_policy: RuntimePolicy::default(),
        })
    }

    fn exec_block(
        rt: &tokio::runtime::Runtime,
        rlm: &Rlm,
        sandbox: &mut Sandbox,
        runtime: &mut RuntimeState,
        code: &str,
    ) {
        let mut telemetry = DepthTelemetry::new(0);
        let out = rt
            .block_on(rlm.execute_in_sandbox(sandbox, code, runtime, &mut telemetry))
            .unwrap();
        assert!(
            !out.contains("Sandbox execution error"),
            "unexpected sandbox execution error: {}",
            out
        );
    }

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
    fn test_find_final_answer_inline_text() {
        let text = "Here is my result: FINAL(the answer).";
        let (kind, content) = find_final_answer(text).unwrap();
        assert!(matches!(kind, FinalType::Direct));
        assert_eq!(content, "the answer");
    }

    #[test]
    fn test_extract_balanced_parens() {
        assert_eq!(extract_balanced_parens("(hello)", 0), Some("hello".into()));
        assert_eq!(extract_balanced_parens("(a(b)c)", 0), Some("a(b)c".into()));
        assert_eq!(extract_balanced_parens("no parens", 0), None);
    }

    #[test]
    fn test_strip_final_wrapper() {
        assert_eq!(strip_final_wrapper("FINAL(88492)"), "88492");
        assert_eq!(strip_final_wrapper("FINAL_VAR(x)"), "x");
        assert_eq!(strip_final_wrapper("plain text"), "plain text");
        assert_eq!(strip_final_wrapper("  FINAL(42)  "), "42");
        assert_eq!(strip_final_wrapper("The answer is FINAL(hello)"), "hello");
    }

    #[test]
    fn test_normalize_vfs_path() {
        assert_eq!(normalize_vfs_path("foo/bar"), "/foo/bar");
        assert_eq!(normalize_vfs_path("/foo/bar"), "/foo/bar");
    }

    #[test]
    fn test_depth_policy_unmet_reason_min_depth() {
        let client = Client::builder().build();
        let rlm = Rlm::new(RlmConfig {
            client,
            model: "test-model".to_string(),
            sub_model: "test-sub-model".to_string(),
            max_iterations: 1,
            depth: 0,
            max_depth: 2,
            verbose: false,
            trace_sandbox: false,
            runtime_policy: RuntimePolicy {
                profile_name: "depth_strict".to_string(),
                format_contract_template: "".to_string(),
                depth_strategy_template: "".to_string(),
                inject_policy_into_context: false,
                depth_enforcement: DepthEnforcementMode::Strict,
                require_min_depth: Some(2),
                require_min_recursive_calls: None,
            },
        });
        let telemetry = DepthTelemetry::new(0);
        let reason = rlm.depth_policy_unmet_reason(&telemetry);
        assert!(reason.is_some());
    }

    #[test]
    fn test_depth_policy_unmet_reason_min_recursive_calls() {
        let client = Client::builder().build();
        let rlm = Rlm::new(RlmConfig {
            client,
            model: "test-model".to_string(),
            sub_model: "test-sub-model".to_string(),
            max_iterations: 1,
            depth: 0,
            max_depth: 2,
            verbose: false,
            trace_sandbox: false,
            runtime_policy: RuntimePolicy {
                profile_name: "depth_strict".to_string(),
                format_contract_template: "".to_string(),
                depth_strategy_template: "".to_string(),
                inject_policy_into_context: false,
                depth_enforcement: DepthEnforcementMode::Strict,
                require_min_depth: None,
                require_min_recursive_calls: Some(1),
            },
        });
        let telemetry = DepthTelemetry::new(0);
        let reason = rlm.depth_policy_unmet_reason(&telemetry);
        assert!(reason.is_some());
    }

    #[test]
    fn test_validate_depth_policy_feasibility_rejects_min_depth_above_max() {
        let client = Client::builder().build();
        let rlm = Rlm::new(RlmConfig {
            client,
            model: "test-model".to_string(),
            sub_model: "test-sub-model".to_string(),
            max_iterations: 1,
            depth: 0,
            max_depth: 1,
            verbose: false,
            trace_sandbox: false,
            runtime_policy: RuntimePolicy {
                profile_name: "depth_strict".to_string(),
                format_contract_template: "".to_string(),
                depth_strategy_template: "".to_string(),
                inject_policy_into_context: false,
                depth_enforcement: DepthEnforcementMode::Strict,
                require_min_depth: Some(3),
                require_min_recursive_calls: None,
            },
        });
        let err = rlm.validate_depth_policy_feasibility().unwrap_err();
        assert!(
            err.to_string().contains("require_min_depth"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_validate_depth_policy_feasibility_rejects_no_recursion_budget() {
        let client = Client::builder().build();
        let rlm = Rlm::new(RlmConfig {
            client,
            model: "test-model".to_string(),
            sub_model: "test-sub-model".to_string(),
            max_iterations: 1,
            depth: 1,
            max_depth: 1,
            verbose: false,
            trace_sandbox: false,
            runtime_policy: RuntimePolicy {
                profile_name: "depth_strict".to_string(),
                format_contract_template: "".to_string(),
                depth_strategy_template: "".to_string(),
                inject_policy_into_context: false,
                depth_enforcement: DepthEnforcementMode::Strict,
                require_min_depth: None,
                require_min_recursive_calls: Some(1),
            },
        });
        let err = rlm.validate_depth_policy_feasibility().unwrap_err();
        assert!(
            err.to_string().contains("no remaining recursion budget"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_checkpoint_restore_restores_sandbox_state() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let rlm = make_test_rlm();
        let mut sandbox = Sandbox::new().unwrap();
        let mut runtime = RuntimeState::new("");

        exec_block(
            &rt,
            &rlm,
            &mut sandbox,
            &mut runtime,
            "x = 1\ncp = CHECKPOINT_CREATE('base')",
        );
        exec_block(&rt, &rlm, &mut sandbox, &mut runtime, "x = 2");
        assert_eq!(sandbox.get_variable("x"), Some("2".to_string()));

        exec_block(
            &rt,
            &rlm,
            &mut sandbox,
            &mut runtime,
            "CHECKPOINT_RESTORE(cp)",
        );
        assert_eq!(sandbox.get_variable("x"), Some("1".to_string()));
    }

    #[test]
    fn test_fork_switch_restores_sandbox_state() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let rlm = make_test_rlm();
        let mut sandbox = Sandbox::new().unwrap();
        let mut runtime = RuntimeState::new("");

        exec_block(
            &rt,
            &rlm,
            &mut sandbox,
            &mut runtime,
            "x = 1\ncp = CHECKPOINT_CREATE('base')",
        );
        exec_block(
            &rt,
            &rlm,
            &mut sandbox,
            &mut runtime,
            "x = 2\nfork_id = FORK_CREATE(cp, 'branch')",
        );
        exec_block(
            &rt,
            &rlm,
            &mut sandbox,
            &mut runtime,
            "FORK_SWITCH(fork_id)",
        );

        assert_eq!(runtime.active_fork, "fork-1");
        assert_eq!(sandbox.get_variable("x"), Some("1".to_string()));
    }

    #[test]
    fn test_fork_create_uses_checkpoint_vfs_snapshot() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let rlm = make_test_rlm();
        let mut sandbox = Sandbox::new().unwrap();
        let mut runtime = RuntimeState::new("");

        exec_block(
            &rt,
            &rlm,
            &mut sandbox,
            &mut runtime,
            "cp = CHECKPOINT_CREATE('base')",
        );
        exec_block(
            &rt,
            &rlm,
            &mut sandbox,
            &mut runtime,
            "VFS_WRITE('/after.txt', 'late')\nmain_len = len(VFS_READ('/after.txt'))",
        );
        assert_eq!(sandbox.get_variable("main_len"), Some("4".to_string()));

        exec_block(
            &rt,
            &rlm,
            &mut sandbox,
            &mut runtime,
            "fork_id = FORK_CREATE(cp, 'branch')",
        );
        exec_block(
            &rt,
            &rlm,
            &mut sandbox,
            &mut runtime,
            "FORK_SWITCH(fork_id)",
        );
        exec_block(
            &rt,
            &rlm,
            &mut sandbox,
            &mut runtime,
            "branch_len = len(VFS_READ('/after.txt'))",
        );

        assert_eq!(sandbox.get_variable("branch_len"), Some("0".to_string()));
    }

    #[test]
    fn test_fallback_repl_blocks_execute() {
        // Simulates the fallback path: a response contains ```repl blocks
        // and a FINAL_VAR() referencing a variable set by that code.
        // The repl block must execute so FINAL_VAR can resolve.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let rlm = make_test_rlm();
        let mut sandbox = Sandbox::new().unwrap();
        let mut runtime = RuntimeState::new("");

        let fallback_response = "Let me compute:\n```repl\nanswer = 5\n```\nFINAL_VAR(answer)";

        // Execute code blocks found in the fallback response.
        let code_blocks = find_code_blocks(fallback_response);
        assert_eq!(code_blocks.len(), 1);
        for code in &code_blocks {
            exec_block(&rt, &rlm, &mut sandbox, &mut runtime, code);
        }

        // Now check_final_answer should resolve FINAL_VAR(answer) -> "5".
        let answer = check_final_answer(fallback_response, &sandbox);
        assert_eq!(answer, Some("5".to_string()));
    }

    #[test]
    fn test_fallback_repl_direct_final() {
        // Fallback response has a repl block AND a direct FINAL().
        // The repl should execute (side effects), and FINAL() should extract directly.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let rlm = make_test_rlm();
        let mut sandbox = Sandbox::new().unwrap();
        let mut runtime = RuntimeState::new("");

        let fallback_response = "```repl\nx = 42\n```\nFINAL(5)";

        let code_blocks = find_code_blocks(fallback_response);
        assert_eq!(code_blocks.len(), 1);
        for code in &code_blocks {
            exec_block(&rt, &rlm, &mut sandbox, &mut runtime, code);
        }

        let answer = check_final_answer(fallback_response, &sandbox);
        assert_eq!(answer, Some("5".to_string()));
        // The repl block also ran (side effect):
        assert_eq!(sandbox.get_variable("x"), Some("42".to_string()));
    }

    #[test]
    fn test_fallback_no_repl_no_final() {
        // Fallback response has no repl blocks and no FINAL — returns None.
        let fallback_response = "The answer is 5.";
        let code_blocks = find_code_blocks(fallback_response);
        assert!(code_blocks.is_empty());

        let sandbox = Sandbox::new().unwrap();
        let answer = check_final_answer(fallback_response, &sandbox);
        assert!(answer.is_none());

        // strip_final_wrapper should pass through unchanged.
        assert_eq!(strip_final_wrapper(fallback_response), "The answer is 5.");
    }

    #[test]
    fn test_strip_hallucinated_tool_calls() {
        assert_eq!(
            strip_hallucinated_tool_calls("Hello <minimax:tool_call>stuff\nhere</minimax:tool_call> world"),
            "Hello  world"
        );
        assert_eq!(
            strip_hallucinated_tool_calls("Answer <FunctionCall>foo()</FunctionCall>"),
            "Answer"
        );
        assert_eq!(
            strip_hallucinated_tool_calls("Result <invoke name=\"bar\">body</invoke> done"),
            "Result  done"
        );
        assert_eq!(strip_hallucinated_tool_calls("no junk here"), "no junk here");
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
