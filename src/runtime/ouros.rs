//! Ouros-backed runtime.
//!
//! Wraps the existing `Sandbox` (which itself wraps ouros's `SessionManager`)
//! behind the [`crate::runtime::Runtime`] trait. This is the default runtime
//! and preserves all historical semantics from `src/sandbox.rs` / `src/rlm.rs`:
//!
//! - the session model (`main` + auxiliary fork/checkpoint sessions),
//! - external function registration (builtins + tool registry),
//! - variable get/set/list,
//! - fork / switch / replace_session_from.
//!
//! The adapter is intentionally thin: every method delegates to `Sandbox`,
//! translating between ouros's `Object`/`ReplProgress` types and the
//! runtime-neutral [`RuntimeValue`]/[`RuntimeProgress`] declarations.

use anyhow::{bail, Result};
use async_trait::async_trait;
use ouros::{Object, ReplProgress};

use crate::runtime::{
    ExternalCall, Runtime, RuntimeKind, RuntimeOutput, RuntimeProgress, RuntimeValue, VariableInfo,
};
use crate::sandbox::Sandbox;
use crate::tools::ToolRegistry;

/// Default Ouros-backed runtime. `main` session is created on construction
/// and carries Altum's builtin external function table.
pub struct OurosRuntime {
    sandbox: Sandbox,
}

impl OurosRuntime {
    /// Constructs the runtime with builtin + tool-registry external functions
    /// registered, mirroring `Sandbox::new`.
    pub fn new(tool_registry: &ToolRegistry) -> Result<Self> {
        let sandbox = Sandbox::new(tool_registry)?;
        Ok(Self { sandbox })
    }

    /// Conversion helper: ouros `Object` -> runtime-neutral value.
    pub fn object_to_value(obj: &Object) -> RuntimeValue {
        match obj {
            Object::None => RuntimeValue::None,
            Object::Bool(b) => RuntimeValue::Bool(*b),
            Object::Int(i) => RuntimeValue::Int(*i),
            Object::Float(f) => RuntimeValue::Float(*f),
            Object::String(s) => RuntimeValue::String(s.clone()),
            Object::Bytes(b) => RuntimeValue::Bytes(b.clone()),
            Object::List(items) | Object::Tuple(items) => {
                RuntimeValue::List(items.iter().map(Self::object_to_value).collect())
            }
            // Steel-side and other Ouros variants land as string repr.
            other => RuntimeValue::String(other.to_string()),
        }
    }

    /// Conversion helper: runtime-neutral value -> ouros `Object`.
    pub fn value_to_object(value: RuntimeValue) -> Object {
        match value {
            RuntimeValue::None => Object::None,
            RuntimeValue::Bool(b) => Object::Bool(b),
            RuntimeValue::Int(i) => Object::Int(i),
            RuntimeValue::Float(f) => Object::Float(f),
            RuntimeValue::String(s) => Object::String(s),
            RuntimeValue::Bytes(b) => Object::Bytes(b),
            RuntimeValue::List(items) => {
                Object::List(items.into_iter().map(Self::value_to_object).collect())
            }
        }
    }

    /// Maps a ouros [`ReplProgress`] to the runtime-neutral variant.
    fn map_progress(progress: ReplProgress, stdout: String) -> RuntimeOutput {
        match progress {
            ReplProgress::FunctionCall {
                function_name,
                args,
                call_id,
                ..
            } => RuntimeOutput {
                progress: RuntimeProgress::FunctionCall(ExternalCall {
                    function_name,
                    args: args.iter().map(Self::object_to_value).collect(),
                    call_id,
                }),
                stdout,
            },
            ReplProgress::Complete(_) => RuntimeOutput {
                progress: RuntimeProgress::Complete,
                stdout,
            },
            other => RuntimeOutput {
                progress: RuntimeProgress::Other(format!("{other:?}")),
                stdout,
            },
        }
    }
}

#[async_trait(?Send)]
impl Runtime for OurosRuntime {
    fn kind(&self) -> RuntimeKind {
        RuntimeKind::Ouros
    }

    fn fence_language(&self) -> &'static str {
        "repl"
    }

    async fn execute(&mut self, code: &str) -> Result<RuntimeOutput> {
        let output = self.sandbox.execute(code)?;
        Ok(Self::map_progress(output.progress, output.stdout))
    }

    async fn resume(&mut self, call_id: u32, value: RuntimeValue) -> Result<RuntimeOutput> {
        let obj = Self::value_to_object(value);
        let output = self.sandbox.resume(call_id, obj)?;
        Ok(Self::map_progress(output.progress, output.stdout))
    }

    fn get_variable(&self, name: &str) -> Option<String> {
        self.sandbox.get_variable(name)
    }

    async fn set_variable(&mut self, name: &str, value: RuntimeValue) -> Result<()> {
        let obj = Self::value_to_object(value);
        self.sandbox.set_variable(name, obj)?;
        Ok(())
    }

    fn list_variables(&self) -> Vec<VariableInfo> {
        self.sandbox
            .list_variables()
            .into_iter()
            .map(|(name, type_name)| VariableInfo {
                name,
                type_name,
                value_string: None,
            })
            .collect()
    }

    async fn fork_session(&mut self, source: &str, new: &str) -> Result<()> {
        self.sandbox.fork_session(source, new)?;
        Ok(())
    }

    async fn destroy_session(&mut self, id: &str) -> Result<()> {
        if id == crate::sandbox::MAIN_SESSION_ID {
            bail!("cannot destroy main session");
        }
        let mgr_ref = self.sandbox.session_manager_mut();
        mgr_ref.destroy_session(id)?;
        Ok(())
    }

    async fn switch_session(&mut self, id: &str) -> Result<()> {
        self.sandbox.switch_session(id)?;
        Ok(())
    }

    async fn replace_session_from(&mut self, target: &str, source: &str) -> Result<()> {
        self.sandbox.replace_session_from(target, source)?;
        Ok(())
    }

    fn session_exists(&self, id: &str) -> bool {
        self.sandbox.session_exists(id)
    }

    fn main_session_id(&self) -> &str {
        crate::sandbox::MAIN_SESSION_ID
    }

    fn list_sessions(&self) -> Vec<String> {
        self.sandbox
            .session_manager()
            .list_sessions()
            .into_iter()
            .map(|info| info.id)
            .collect()
    }

    fn external_function_names(&self) -> Vec<String> {
        crate::sandbox::all_external_functions_for_read(self.sandbox.tool_registry())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolRegistry;

    fn empty_registry() -> ToolRegistry {
        ToolRegistry::new()
    }

    #[tokio::test]
    async fn execute_and_resume_complete() {
        let mut rt = OurosRuntime::new(&empty_registry()).unwrap();
        let out = rt.execute("x = 1\nprint(x)").await.unwrap();
        assert!(matches!(out.progress, RuntimeProgress::Complete));
        assert!(out.stdout.contains('1'));
        assert_eq!(rt.get_variable("x"), Some("1".to_string()));
    }

    #[tokio::test]
    async fn fork_and_switch_isolates_state() {
        let mut rt = OurosRuntime::new(&empty_registry()).unwrap();
        rt.execute("x = 1").await.unwrap();
        rt.fork_session("main", "fork-a").await.unwrap();
        rt.switch_session("fork-a").await.unwrap();
        rt.execute("x = 2").await.unwrap();
        assert_eq!(rt.get_variable("x"), Some("2".to_string()));
        rt.switch_session("main").await.unwrap();
        assert_eq!(rt.get_variable("x"), Some("1".to_string()));
    }

    #[tokio::test]
    async fn set_variable_roundtrip() {
        let mut rt = OurosRuntime::new(&empty_registry()).unwrap();
        rt.set_variable("ctx", RuntimeValue::String("hello".into()))
            .await
            .unwrap();
        let out = rt.execute("print(ctx)").await.unwrap();
        assert!(out.stdout.contains("hello"));
        // Ouros stores String values; get_variable returns the repr which
        // includes Python quoting.
        let v = rt.get_variable("ctx").unwrap();
        assert!(v == "hello" || v == "'hello'", "unexpected repr: {v}");
    }

    #[tokio::test]
    async fn list_variables_returns_typed_entries() {
        let mut rt = OurosRuntime::new(&empty_registry()).unwrap();
        rt.execute("x = 1\ny = 'a'").await.unwrap();
        let names: Vec<String> = rt.list_variables().into_iter().map(|v| v.name).collect();
        assert!(names.contains(&"x".to_string()));
        assert!(names.contains(&"y".to_string()));
    }

    #[tokio::test]
    async fn main_session_is_main_id() {
        let rt = OurosRuntime::new(&empty_registry()).unwrap();
        assert_eq!(rt.main_session_id(), "main");
        assert!(rt.session_exists("main"));
        assert!(!rt.session_exists("missing"));
    }

    #[tokio::test]
    async fn main_session_cannot_be_destroyed() {
        let mut rt = OurosRuntime::new(&empty_registry()).unwrap();
        assert!(rt.destroy_session("main").await.is_err());
        assert!(rt.session_exists("main"));
        let out = rt.execute("x = 1").await.unwrap();
        assert!(matches!(out.progress, RuntimeProgress::Complete));
    }

    #[tokio::test]
    async fn replace_session_restores_state() {
        let mut rt = OurosRuntime::new(&empty_registry()).unwrap();
        rt.execute("x = 5").await.unwrap();
        rt.fork_session("main", "checkpoint-a").await.unwrap();
        rt.execute("x = 99").await.unwrap();
        assert_eq!(rt.get_variable("x"), Some("99".to_string()));
        rt.replace_session_from("main", "checkpoint-a")
            .await
            .unwrap();
        rt.switch_session("main").await.unwrap();
        assert_eq!(rt.get_variable("x"), Some("5".to_string()));
    }

    #[test]
    fn object_to_value_basic_variants() {
        assert!(matches!(
            OurosRuntime::object_to_value(&Object::None),
            RuntimeValue::None
        ));
        assert!(matches!(
            OurosRuntime::object_to_value(&Object::Int(42)),
            RuntimeValue::Int(42)
        ));
        match OurosRuntime::object_to_value(&Object::List(vec![Object::Int(1), Object::Int(2)])) {
            RuntimeValue::List(items) => assert_eq!(items.len(), 2),
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn value_to_object_roundtrips_string() {
        let obj = OurosRuntime::value_to_object(RuntimeValue::String("hi".into()));
        match obj {
            Object::String(s) => assert_eq!(s, "hi"),
            other => panic!("expected string, got {other:?}"),
        }
    }
}
