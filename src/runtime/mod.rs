//! Runtime-neutral sandbox interface.
//!
//! The RLM orchestration loop talks to sandboxes through [`Runtime`] rather
//! than to a concrete interpreter (Ouros, Steel, …). This lets us swap the
//! underlying execution engine to evaluate alternative VMs without touching
//! the orchestration code.
//!
//! Each runtime implementation owns:
//!
//! - code execution with first-class external calls (suspend/resume),
//! - variable get/set/list,
//! - session lifecycle (fork/destroy/switch/restore),
//! - external function registration table for sandbox-visible host calls.
//!
//! [`OurosRuntime`] wraps the existing `Sandbox` semantics (the previous
//! direct ouros access in `Rlm`). A `steel` feature flag enables
//! [`SteelRuntime`], a Scheme runtime based on `steel-core` that uses
//! journal replay for external-call suspension.

pub mod ouros;

#[cfg(feature = "steel")]
pub mod steel;
#[cfg(feature = "steel")]
pub use steel::SteelRuntime;

pub mod repair;

use std::fmt;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

/// One external-call invocation requested by sandbox code.
///
/// Mirrors ouros's `ReplProgress::FunctionCall` but is runtime-neutral.
/// Steeled equivalents (Steel has no native continuations) emerge from
/// `Runtime::execute`/`resume` via the same shape so the RLM loop can
/// dispatch uniformly.
#[derive(Debug, Clone)]
pub struct ExternalCall {
    /// Function name, e.g. `llm_query` or `VFS_WRITE`.
    pub function_name: String,
    /// Positional string-coercible args for the call.
    pub args: Vec<RuntimeValue>,
    /// Correlation id used to pair this call with a future `resume`.
    pub call_id: u32,
}

/// Result of [`Runtime::execute`] or [`Runtime::resume`].
#[derive(Debug, Clone)]
pub enum RuntimeProgress {
    /// Execution paused pending an external call.
    FunctionCall(ExternalCall),
    /// Execution finished; sandbox state is updated.
    Complete,
    /// Sandbox reached some other wait state. Tracked but not dispatched
    /// further by the RLM loop.
    Other(String),
}

/// Generic value passed across the runtime/host boundary.
///
/// Subset sufficient for Altum's external functions today (all of which
/// pass strings, None, ints, or lists thereof). Ouros-backed values are
/// converted via the [`From<ouros::Object>`] impls on the ouros runtime.
#[derive(Debug, Clone)]
pub enum RuntimeValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<RuntimeValue>),
}

impl RuntimeValue {
    /// Coerces any value into its `String` representation, mirroring how
    /// `obj_to_string` historically treated ouros objects in `rlm.rs`.
    pub fn to_value_string(&self) -> String {
        match self {
            RuntimeValue::None => "None".to_string(),
            RuntimeValue::Bool(b) => b.to_string(),
            RuntimeValue::Int(i) => i.to_string(),
            RuntimeValue::Float(f) => f.to_string(),
            RuntimeValue::String(s) => s.clone(),
            RuntimeValue::Bytes(b) => format!("{b:?}"),
            RuntimeValue::List(items) => {
                let parts: Vec<String> = items.iter().map(RuntimeValue::to_value_string).collect();
                format!("[{}]", parts.join(", "))
            }
        }
    }

    /// Coerces a single value to a list of strings, for callers that
    /// previously did `obj_to_string_list` over a single Ouros argument.
    /// Lists expand to their elements; atomic values wrap into a 1-element list.
    pub fn to_string_list(value: &RuntimeValue) -> Vec<String> {
        match value {
            RuntimeValue::List(items) => items.iter().map(RuntimeValue::to_value_string).collect(),
            other => vec![other.to_value_string()],
        }
    }
}

impl From<String> for RuntimeValue {
    fn from(s: String) -> Self {
        RuntimeValue::String(s)
    }
}

impl From<&str> for RuntimeValue {
    fn from(s: &str) -> Self {
        RuntimeValue::String(s.to_string())
    }
}

impl From<i64> for RuntimeValue {
    fn from(i: i64) -> Self {
        RuntimeValue::Int(i)
    }
}

/// Output of an `execute` or `resume` step.
#[derive(Debug, Clone)]
pub struct RuntimeOutput {
    pub progress: RuntimeProgress,
    pub stdout: String,
}

/// Information about a single variable in a runtime session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableInfo {
    pub name: String,
    pub type_name: String,
    pub value_string: Option<String>,
}

/// Runtime backend tag (e.g. `"ouros"`, `"steel"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeKind {
    Ouros,
    Steel,
}

impl RuntimeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RuntimeKind::Ouros => "ouros",
            RuntimeKind::Steel => "steel",
        }
    }
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "ouros" => Ok(RuntimeKind::Ouros),
            "steel" => Ok(RuntimeKind::Steel),
            other => Err(anyhow::anyhow!("unknown runtime: {other}")),
        }
    }
}

impl fmt::Display for RuntimeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Pluggable runtime trait.
///
/// All session-mutating methods take `&mut self`. Read access via `&self` is
/// allowed for variable lookup and listing where the underlying runtime
/// supports it (Ouros `get_variable` is `&self` already).
///
/// Methods use `#[async_trait(?Send)]` because the underlying interpreters
/// (Ouros `SessionManager`, future Steel `SteelCore`) are intentionally
/// single-threaded per RLM call. The RLM runtime wraps each completion in a
/// `tokio::runtime::Builder::new_current_thread` block already.
#[async_trait(?Send)]
pub trait Runtime {
    /// Backend kind for telemetry.
    fn kind(&self) -> RuntimeKind;

    /// Returns the code-fence language identifier (e.g. `repl`, `scheme`).
    fn fence_language(&self) -> &'static str {
        "repl"
    }

    /// Executes code in the active session.
    async fn execute(&mut self, code: &str) -> Result<RuntimeOutput>;

    /// Resumes a previously suspended external call.
    async fn resume(&mut self, call_id: u32, value: RuntimeValue) -> Result<RuntimeOutput>;

    /// Looks up a variable value (string form) from the active session.
    fn get_variable(&self, name: &str) -> Option<String>;

    /// Injects a variable into the active session.
    async fn set_variable(&mut self, name: &str, value: RuntimeValue) -> Result<()>;

    /// Lists variables visible in the active session.
    fn list_variables(&self) -> Vec<VariableInfo>;

    /// Forks an existing session into a new independent one.
    async fn fork_session(&mut self, source: &str, new: &str) -> Result<()>;

    /// Destroys a named session (other than main).
    async fn destroy_session(&mut self, id: &str) -> Result<()>;

    /// Switches the active session. Errors if the session does not exist.
    async fn switch_session(&mut self, id: &str) -> Result<()>;

    /// Replaces `target` with the state of `source`.
    async fn replace_session_from(&mut self, target: &str, source: &str) -> Result<()>;

    /// True if a session with the given id exists.
    fn session_exists(&self, id: &str) -> bool;

    /// Returns the main session identifier (e.g. `"main"`).
    fn main_session_id(&self) -> &str;

    /// Optional description of all sessions currently alive.
    fn list_sessions(&self) -> Vec<String>;

    /// External function names the runtime is configured to expose to code.
    fn external_function_names(&self) -> Vec<String>;

    /// Optional human-readable note about JIT repair / repairs that this
    /// runtime performs automatically. Tracked to telemetry.
    fn repair_description(&self) -> Option<String> {
        None
    }
}

/// Convenience type for Arc-shared runtime trait objects. Note that
/// runtimes are intentionally not `Send`; the RLM keeps a runtime pinned to
/// the calling thread by spawning its own current-thread executor.
pub type SharedRuntime = Arc<dyn Runtime>;

/// Marker for the Steel "experiment" feature flag.
pub const fn steel_enabled() -> bool {
    cfg!(feature = "steel")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_value_string_roundtrip() {
        let v = RuntimeValue::String("hello".to_string());
        assert_eq!(v.to_value_string(), "hello");
    }

    #[test]
    fn runtime_value_list_to_string() {
        let v = RuntimeValue::List(vec![
            RuntimeValue::String("a".into()),
            RuntimeValue::String("b".into()),
        ]);
        assert_eq!(v.to_value_string(), "[a, b]");
    }

    #[test]
    fn runtime_kind_parse_roundtrip() {
        assert_eq!(RuntimeKind::parse("ouros").unwrap(), RuntimeKind::Ouros);
        assert_eq!(RuntimeKind::parse("STEEL").unwrap(), RuntimeKind::Steel);
        assert!(RuntimeKind::parse("rust").is_err());
    }
}
