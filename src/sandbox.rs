use anyhow::Result;
use ouros::session_manager::{ExecuteOutput, SessionManager};
use ouros::{ExternalResult, Object};

use crate::tools::ToolRegistry;

pub const MAIN_SESSION_ID: &str = "main";

pub struct Sandbox {
    mgr: SessionManager,
    active_session: String,
}

fn builtin_external_functions() -> Vec<String> {
    vec![
        "llm_query".into(),
        "llm_query_batched".into(),
        "FINAL".into(),
        "FINAL_VAR".into(),
        "SHOW_VARS".into(),
        "CHECKPOINT_CREATE".into(),
        "CHECKPOINT_RESTORE".into(),
        "FORK_CREATE".into(),
        "FORK_SWITCH".into(),
        "FORK_LIST".into(),
        "VFS_WRITE".into(),
        "VFS_READ".into(),
        "VFS_LIST".into(),
        "STRATEGY_COMMIT".into(),
        "STRATEGY_STATUS".into(),
    ]
}

fn all_external_functions(tool_registry: &ToolRegistry) -> Vec<String> {
    let mut funcs = builtin_external_functions();
    for name in tool_registry.tool_names() {
        if !funcs.contains(&name) {
            funcs.push(name);
        }
    }
    funcs
}

impl Sandbox {
    pub fn new(tool_registry: &ToolRegistry) -> Result<Self> {
        let mut mgr = SessionManager::new("repl.py");
        let funcs = all_external_functions(tool_registry);
        mgr.reset(None, funcs.clone())?;
        mgr.create_session(MAIN_SESSION_ID, funcs)?;
        Ok(Self {
            mgr,
            active_session: MAIN_SESSION_ID.to_string(),
        })
    }

    pub fn execute(&mut self, code: &str) -> Result<ExecuteOutput> {
        Ok(self.mgr.execute(Some(&self.active_session), code)?)
    }

    pub fn resume(&mut self, call_id: u32, value: Object) -> Result<ExecuteOutput> {
        Ok(self.mgr.resume(
            Some(&self.active_session),
            call_id,
            ExternalResult::Return(value),
        )?)
    }

    pub fn get_variable(&self, name: &str) -> Option<String> {
        self.mgr
            .get_variable(Some(&self.active_session), name)
            .ok()
            .map(|v| v.repr.unwrap_or_else(|| v.json_value.to_string()))
    }

    pub fn set_variable(&mut self, name: &str, value: Object) -> Result<()> {
        Ok(self
            .mgr
            .set_variable_obj(Some(&self.active_session), name, value)?)
    }

    pub fn list_variables(&self) -> Vec<(String, String)> {
        self.mgr
            .list_variables(Some(&self.active_session))
            .unwrap_or_default()
            .into_iter()
            .map(|v| (v.name, v.type_name))
            .collect()
    }

    pub fn fork_session(&mut self, source_session: &str, new_session: &str) -> Result<()> {
        Ok(self.mgr.fork_session(source_session, new_session)?)
    }

    pub fn switch_session(&mut self, session_id: &str) -> Result<()> {
        if !self.session_exists(session_id) {
            anyhow::bail!("session '{}' not found", session_id);
        }
        self.active_session = session_id.to_string();
        Ok(())
    }

    pub fn replace_session_from(
        &mut self,
        target_session: &str,
        source_session: &str,
    ) -> Result<()> {
        if !self.session_exists(source_session) {
            anyhow::bail!("source session '{}' not found", source_session);
        }
        if source_session == target_session {
            return Ok(());
        }
        if self.session_exists(target_session) {
            self.mgr.destroy_session(target_session)?;
        }
        Ok(self.mgr.fork_session(source_session, target_session)?)
    }

    fn session_exists(&self, session_id: &str) -> bool {
        self.mgr
            .list_sessions()
            .iter()
            .any(|info| info.id == session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{ToolRegistry, ToolSpec};

    fn empty_registry() -> ToolRegistry {
        ToolRegistry::new()
    }

    #[test]
    fn test_fork_and_switch_isolates_session_state() {
        let mut sandbox = Sandbox::new(&empty_registry()).unwrap();
        sandbox.execute("x = 1").unwrap();

        sandbox.fork_session(MAIN_SESSION_ID, "fork-a").unwrap();
        sandbox.switch_session("fork-a").unwrap();
        sandbox.execute("x = 2").unwrap();
        assert_eq!(sandbox.get_variable("x"), Some("2".to_string()));

        sandbox.switch_session(MAIN_SESSION_ID).unwrap();
        assert_eq!(sandbox.get_variable("x"), Some("1".to_string()));
    }

    #[test]
    fn test_replace_session_restores_previous_state() {
        let mut sandbox = Sandbox::new(&empty_registry()).unwrap();
        sandbox.execute("x = 5").unwrap();

        sandbox
            .fork_session(MAIN_SESSION_ID, "checkpoint-a")
            .unwrap();
        sandbox.execute("x = 99").unwrap();
        assert_eq!(sandbox.get_variable("x"), Some("99".to_string()));

        sandbox
            .replace_session_from(MAIN_SESSION_ID, "checkpoint-a")
            .unwrap();
        sandbox.switch_session(MAIN_SESSION_ID).unwrap();
        assert_eq!(sandbox.get_variable("x"), Some("5".to_string()));
    }

    #[test]
    fn test_all_external_functions_empty_registry() {
        let reg = ToolRegistry::new();
        let funcs = all_external_functions(&reg);
        assert_eq!(funcs, builtin_external_functions());
    }

    #[test]
    fn test_all_external_functions_with_custom_tools() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec {
            name: "my_tool".to_string(),
            description: "custom".to_string(),
            input_schema: None,
        });
        let funcs = all_external_functions(&reg);
        let builtins = builtin_external_functions();
        assert!(funcs.len() > builtins.len());
        assert!(funcs.contains(&"my_tool".to_string()));
        for b in &builtins {
            assert!(funcs.contains(b));
        }
    }

    #[test]
    fn test_all_external_functions_dedup_builtin_names() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec {
            name: "FINAL".to_string(),
            description: "shadow".to_string(),
            input_schema: None,
        });
        let funcs = all_external_functions(&reg);
        let final_count = funcs.iter().filter(|f| *f == "FINAL").count();
        assert_eq!(final_count, 1, "builtin name should not be duplicated");
    }

    #[test]
    fn test_sandbox_new_with_custom_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec {
            name: "SEARCH".to_string(),
            description: "search tool".to_string(),
            input_schema: None,
        });
        let sandbox = Sandbox::new(&reg);
        assert!(
            sandbox.is_ok(),
            "sandbox should initialize with custom tool registry"
        );
    }

    #[test]
    fn test_builtins_are_valid_identifiers() {
        for name in builtin_external_functions() {
            assert!(
                name.chars()
                    .next()
                    .map(|c| c.is_alphabetic() || c == '_')
                    .unwrap_or(false),
                "builtin '{}' must be a valid Python identifier",
                name
            );
        }
    }
}
