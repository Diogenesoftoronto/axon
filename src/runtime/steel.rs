//! Experimental Steel-backed runtime.
//!
//! Steel does not expose Ouros-style continuations. External calls therefore
//! use deterministic journal replay: the first pass records a call and
//! returns a placeholder, and `resume` rebuilds the session while replaying
//! prior host responses at the same call sites.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use steel::rvals::SteelVal;
use steel::steel_vm::engine::Engine;
use steel::steel_vm::register_fn::RegisterFn;

use crate::runtime::repair::RepairEngine;
use crate::runtime::{
    ExternalCall, Runtime, RuntimeKind, RuntimeOutput, RuntimeProgress, RuntimeValue, VariableInfo,
};
use crate::tools::ToolRegistry;

#[derive(Clone)]
enum JournalEntry {
    Code(String),
    Set(String, RuntimeValue),
}

#[derive(Default)]
struct ReplayState {
    responses: Vec<RuntimeValue>,
    cursor: usize,
    pending: Option<ExternalCall>,
}

struct SteelSession {
    engine: Engine,
    journal: Vec<JournalEntry>,
    responses: Vec<RuntimeValue>,
    variables: BTreeSet<String>,
    replay: Arc<Mutex<ReplayState>>,
}

/// A sandboxed Steel interpreter with forkable journal-replayed sessions.
pub struct SteelRuntime {
    sessions: BTreeMap<String, SteelSession>,
    active_session: String,
    external_names: Vec<String>,
    repair: RepairEngine,
    repair_labels: Vec<String>,
}

impl SteelRuntime {
    pub fn new(tool_registry: &ToolRegistry) -> Result<Self> {
        let external_names = crate::sandbox::all_external_functions_for_read(tool_registry);
        let (main, _) = build_session(&[], &[], BTreeSet::new(), &external_names)?;
        Ok(Self {
            sessions: BTreeMap::from([(crate::sandbox::MAIN_SESSION_ID.to_string(), main)]),
            active_session: crate::sandbox::MAIN_SESSION_ID.to_string(),
            external_names,
            repair: RepairEngine::new(),
            repair_labels: Vec::new(),
        })
    }

    fn active(&self) -> Result<&SteelSession> {
        self.sessions
            .get(&self.active_session)
            .ok_or_else(|| anyhow!("active Steel session is missing"))
    }

    fn active_mut(&mut self) -> Result<&mut SteelSession> {
        self.sessions
            .get_mut(&self.active_session)
            .ok_or_else(|| anyhow!("active Steel session is missing"))
    }

    fn replace_active_with_rebuild(
        &mut self,
        responses: Vec<RuntimeValue>,
    ) -> Result<Vec<SteelVal>> {
        let active = self.active()?;
        let journal = active.journal.clone();
        let variables = active.variables.clone();
        let (rebuilt, values) =
            build_session(&journal, &responses, variables, &self.external_names)?;
        self.sessions.insert(self.active_session.clone(), rebuilt);
        Ok(values)
    }

    fn output_from_values(&self, values: Vec<SteelVal>) -> Result<RuntimeOutput> {
        let pending = self
            .active()?
            .replay
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?
            .pending
            .clone();
        let progress = match pending {
            Some(call) => RuntimeProgress::FunctionCall(call),
            None => RuntimeProgress::Complete,
        };
        let stdout = values
            .into_iter()
            .filter(|value| !matches!(value, SteelVal::Void))
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        Ok(RuntimeOutput { progress, stdout })
    }
}

fn runtime_to_steel(value: &RuntimeValue) -> SteelVal {
    match value {
        RuntimeValue::None => SteelVal::Void,
        RuntimeValue::Bool(value) => SteelVal::BoolV(*value),
        RuntimeValue::Int(value) => SteelVal::IntV(*value as isize),
        RuntimeValue::Float(value) => SteelVal::NumV(*value),
        RuntimeValue::String(value) => SteelVal::StringV(value.clone().into()),
        RuntimeValue::Bytes(value) => SteelVal::ListV(
            value
                .iter()
                .map(|byte| SteelVal::IntV(*byte as isize))
                .collect(),
        ),
        RuntimeValue::List(values) => {
            SteelVal::ListV(values.iter().map(runtime_to_steel).collect())
        }
    }
}

fn steel_to_runtime(value: &SteelVal) -> RuntimeValue {
    match value {
        SteelVal::Void => RuntimeValue::None,
        SteelVal::BoolV(value) => RuntimeValue::Bool(*value),
        SteelVal::IntV(value) => RuntimeValue::Int(*value as i64),
        SteelVal::NumV(value) => RuntimeValue::Float(*value),
        SteelVal::StringV(value) | SteelVal::SymbolV(value) => {
            RuntimeValue::String(value.to_string())
        }
        other => RuntimeValue::String(other.to_string()),
    }
}

fn register_external_functions(
    engine: &mut Engine,
    names: &[String],
    replay: &Arc<Mutex<ReplayState>>,
) -> Result<()> {
    let state = Arc::clone(replay);
    engine.register_fn(
        "__altum-external",
        move |function_name: String, args: SteelVal| -> SteelVal {
            let mut state = state.lock().expect("Steel replay lock poisoned");
            if let Some(response) = state.responses.get(state.cursor).cloned() {
                state.cursor += 1;
                return runtime_to_steel(&response);
            }
            if state.pending.is_none() {
                let call_id = state.cursor as u32;
                let args = match args {
                    SteelVal::ListV(values) => values.iter().map(steel_to_runtime).collect(),
                    value => vec![steel_to_runtime(&value)],
                };
                state.pending = Some(ExternalCall {
                    function_name,
                    args,
                    call_id,
                });
            }
            SteelVal::Void
        },
    );

    for name in names {
        let quoted_name = name.replace('\\', "\\\\").replace('"', "\\\"");
        engine
            .run(format!(
                "(define ({name} . args) (__altum-external \"{quoted_name}\" args))"
            ))
            .map_err(|e| anyhow!(e.to_string()))?;
    }
    Ok(())
}

fn build_session(
    journal: &[JournalEntry],
    responses: &[RuntimeValue],
    variables: BTreeSet<String>,
    external_names: &[String],
) -> Result<(SteelSession, Vec<SteelVal>)> {
    let replay = Arc::new(Mutex::new(ReplayState {
        responses: responses.to_vec(),
        cursor: 0,
        pending: None,
    }));
    let mut engine = Engine::new_sandboxed();
    register_external_functions(&mut engine, external_names, &replay)?;
    let mut last_values = Vec::new();
    for entry in journal {
        match entry {
            JournalEntry::Code(code) => {
                last_values = engine
                    .run(code.clone())
                    .map_err(|e| anyhow!(e.to_string()))?;
            }
            JournalEntry::Set(name, value) => {
                let value = runtime_to_steel(value);
                if engine.update_value(name, value.clone()).is_none() {
                    engine.register_value(name, value);
                }
            }
        }
    }
    Ok((
        SteelSession {
            engine,
            journal: journal.to_vec(),
            responses: responses.to_vec(),
            variables,
            replay,
        },
        last_values,
    ))
}

fn defined_names(code: &str) -> impl Iterator<Item = String> + '_ {
    code.match_indices("(define").filter_map(|(index, _)| {
        let tail = code[index + "(define".len()..].trim_start();
        let tail = tail.strip_prefix('(').unwrap_or(tail);
        let name: String = tail
            .chars()
            .take_while(|ch| !ch.is_whitespace() && *ch != ')' && *ch != '(')
            .collect();
        (!name.is_empty()).then_some(name)
    })
}

fn value_type(value: &SteelVal) -> &'static str {
    match value {
        SteelVal::Void => "void",
        SteelVal::BoolV(_) => "bool",
        SteelVal::IntV(_) => "int",
        SteelVal::NumV(_) => "float",
        SteelVal::StringV(_) => "string",
        SteelVal::ListV(_) => "list",
        SteelVal::VectorV(_) => "vector",
        _ => "value",
    }
}

fn value_string(value: &SteelVal) -> String {
    match value {
        SteelVal::StringV(value) | SteelVal::SymbolV(value) => value.to_string(),
        other => other.to_string(),
    }
}

#[async_trait(?Send)]
impl Runtime for SteelRuntime {
    fn kind(&self) -> RuntimeKind {
        RuntimeKind::Steel
    }

    fn fence_language(&self) -> &'static str {
        "scheme"
    }

    async fn execute(&mut self, code: &str) -> Result<RuntimeOutput> {
        let first = self.active_mut()?.engine.run(code.to_string());
        let (code, values) = match first {
            Ok(values) => (code.to_string(), values),
            Err(error) => {
                let message = error.to_string();
                let Some(repaired) = self.repair.try_repair(code, &message, "scheme")? else {
                    return Err(anyhow!(message));
                };
                self.repair_labels.extend(
                    self.repair
                        .take_log()
                        .into_iter()
                        .map(|entry| entry.rule.label().to_string()),
                );
                let responses = self.active()?.responses.clone();
                self.replace_active_with_rebuild(responses)?;
                let values = self
                    .active_mut()?
                    .engine
                    .run(repaired.clone())
                    .map_err(|e| anyhow!(e.to_string()))?;
                (repaired, values)
            }
        };
        let names: Vec<String> = defined_names(&code).collect();
        let active = self.active_mut()?;
        active.variables.extend(names);
        active.journal.push(JournalEntry::Code(code));
        self.output_from_values(values)
    }

    async fn resume(&mut self, call_id: u32, value: RuntimeValue) -> Result<RuntimeOutput> {
        let pending = self
            .active()?
            .replay
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?
            .pending
            .clone();
        match pending {
            Some(call) if call.call_id == call_id => {}
            Some(call) => bail!(
                "Steel call id mismatch: expected {}, got {call_id}",
                call.call_id
            ),
            None => bail!("Steel runtime has no pending external call"),
        }
        let mut responses = self.active()?.responses.clone();
        responses.push(value);
        let values = self.replace_active_with_rebuild(responses)?;
        self.output_from_values(values)
    }

    fn get_variable(&self, name: &str) -> Option<String> {
        self.active()
            .ok()?
            .engine
            .extract_value(name)
            .ok()
            .map(|value| value_string(&value))
    }

    async fn set_variable(&mut self, name: &str, value: RuntimeValue) -> Result<()> {
        let steel_value = runtime_to_steel(&value);
        let active = self.active_mut()?;
        if active
            .engine
            .update_value(name, steel_value.clone())
            .is_none()
        {
            active.engine.register_value(name, steel_value);
        }
        active.variables.insert(name.to_string());
        active
            .journal
            .push(JournalEntry::Set(name.to_string(), value));
        Ok(())
    }

    fn list_variables(&self) -> Vec<VariableInfo> {
        let Ok(active) = self.active() else {
            return Vec::new();
        };
        active
            .variables
            .iter()
            .filter_map(|name| {
                active
                    .engine
                    .extract_value(name)
                    .ok()
                    .map(|value| VariableInfo {
                        name: name.clone(),
                        type_name: value_type(&value).to_string(),
                        value_string: Some(value_string(&value)),
                    })
            })
            .collect()
    }

    async fn fork_session(&mut self, source: &str, new: &str) -> Result<()> {
        if self.sessions.contains_key(new) {
            bail!("session already exists: {new}");
        }
        let source = self
            .sessions
            .get(source)
            .ok_or_else(|| anyhow!("session not found: {source}"))?;
        let (fork, _) = build_session(
            &source.journal,
            &source.responses,
            source.variables.clone(),
            &self.external_names,
        )?;
        self.sessions.insert(new.to_string(), fork);
        Ok(())
    }

    async fn destroy_session(&mut self, id: &str) -> Result<()> {
        if id == crate::sandbox::MAIN_SESSION_ID {
            bail!("cannot destroy main session");
        }
        if self.sessions.remove(id).is_none() {
            bail!("session not found: {id}");
        }
        if self.active_session == id {
            self.active_session = crate::sandbox::MAIN_SESSION_ID.to_string();
        }
        Ok(())
    }

    async fn switch_session(&mut self, id: &str) -> Result<()> {
        if !self.sessions.contains_key(id) {
            bail!("session not found: {id}");
        }
        self.active_session = id.to_string();
        Ok(())
    }

    async fn replace_session_from(&mut self, target: &str, source: &str) -> Result<()> {
        if !self.sessions.contains_key(target) {
            bail!("session not found: {target}");
        }
        let source = self
            .sessions
            .get(source)
            .ok_or_else(|| anyhow!("session not found: {source}"))?;
        let (replacement, _) = build_session(
            &source.journal,
            &source.responses,
            source.variables.clone(),
            &self.external_names,
        )?;
        self.sessions.insert(target.to_string(), replacement);
        Ok(())
    }

    fn session_exists(&self, id: &str) -> bool {
        self.sessions.contains_key(id)
    }

    fn main_session_id(&self) -> &str {
        crate::sandbox::MAIN_SESSION_ID
    }

    fn list_sessions(&self) -> Vec<String> {
        self.sessions.keys().cloned().collect()
    }

    fn external_function_names(&self) -> Vec<String> {
        self.external_names.clone()
    }

    fn repair_description(&self) -> Option<String> {
        (!self.repair_labels.is_empty()).then(|| self.repair_labels.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> SteelRuntime {
        SteelRuntime::new(&ToolRegistry::new()).unwrap()
    }

    #[tokio::test]
    async fn executes_and_tracks_variables() {
        let mut runtime = runtime();
        runtime
            .execute("(define x 41) (set! x (+ x 1))")
            .await
            .unwrap();
        assert_eq!(runtime.get_variable("x").as_deref(), Some("42"));
        assert!(runtime
            .list_variables()
            .iter()
            .any(|value| value.name == "x"));
    }

    #[tokio::test]
    async fn forked_sessions_are_isolated() {
        let mut runtime = runtime();
        runtime.execute("(define x 1)").await.unwrap();
        runtime.fork_session("main", "fork").await.unwrap();
        runtime.switch_session("fork").await.unwrap();
        runtime.execute("(set! x 2)").await.unwrap();
        assert_eq!(runtime.get_variable("x").as_deref(), Some("2"));
        runtime.switch_session("main").await.unwrap();
        assert_eq!(runtime.get_variable("x").as_deref(), Some("1"));
    }

    #[tokio::test]
    async fn external_calls_resume_via_journal_replay() {
        let mut runtime = runtime();
        let output = runtime
            .execute("(define answer (llm_query \"question\"))")
            .await
            .unwrap();
        let RuntimeProgress::FunctionCall(call) = output.progress else {
            panic!("expected external call");
        };
        assert_eq!(call.function_name, "llm_query");
        let output = runtime
            .resume(call.call_id, RuntimeValue::String("response".into()))
            .await
            .unwrap();
        assert!(matches!(output.progress, RuntimeProgress::Complete));
        assert_eq!(runtime.get_variable("answer").as_deref(), Some("response"));
    }
}
