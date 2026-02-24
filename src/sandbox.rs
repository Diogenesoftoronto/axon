use anyhow::Result;
use ouros::session_manager::{ExecuteOutput, SessionManager};
use ouros::{ExternalResult, Object};

pub struct Sandbox {
    mgr: SessionManager,
}

impl Sandbox {
    pub fn new() -> Result<Self> {
        let mut mgr = SessionManager::new("repl.py");
        mgr.reset(
            None,
            vec![
                "llm_query".into(),
                "llm_query_batched".into(),
                "FINAL_VAR".into(),
                "SHOW_VARS".into(),
            ],
        )?;
        Ok(Self { mgr })
    }

    pub fn execute(&mut self, code: &str) -> Result<ExecuteOutput> {
        Ok(self.mgr.execute(None, code)?)
    }

    pub fn resume(&mut self, call_id: u32, value: Object) -> Result<ExecuteOutput> {
        Ok(self.mgr.resume(None, call_id, ExternalResult::Return(value))?)
    }

    pub fn get_variable(&self, name: &str) -> Option<String> {
        self.mgr
            .get_variable(None, name)
            .ok()
            .map(|v| v.repr.unwrap_or_else(|| v.json_value.to_string()))
    }

    pub fn set_variable(&mut self, name: &str, value: Object) -> Result<()> {
        Ok(self.mgr.set_variable_obj(None, name, value)?)
    }

    pub fn list_variables(&self) -> Vec<(String, String)> {
        self.mgr
            .list_variables(None)
            .unwrap_or_default()
            .into_iter()
            .map(|v| (v.name, v.type_name))
            .collect()
    }
}
