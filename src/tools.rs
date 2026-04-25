use rmcp::schemars;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub input_schema: Option<serde_json::Value>,
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolSpec>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, spec: ToolSpec) {
        self.tools.insert(spec.name.clone(), spec);
    }

    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.tools.get(name)
    }

    pub fn tool_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn tools(&self) -> &HashMap<String, ToolSpec> {
        &self.tools
    }
}

pub type SharedToolRegistry = Arc<Mutex<ToolRegistry>>;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    prop_compose! {
        fn arb_tool_spec_no_schema()(name in "[a-zA-Z_][a-zA-Z0-9_]{0,19}", desc in ".*") -> ToolSpec {
            ToolSpec { name, description: desc, input_schema: None }
        }
    }

    proptest! {
        #[test]
        fn register_then_get_returns_spec(spec in arb_tool_spec_no_schema()) {
            let mut reg = ToolRegistry::new();
            reg.register(spec.clone());
            let got = reg.get(&spec.name).unwrap();
            assert_eq!(got.name, spec.name);
            assert_eq!(got.description, spec.description);
        }

        #[test]
        fn get_unknown_returns_none(name in "[a-zA-Z_][a-zA-Z0-9_]{0,19}") {
            let reg = ToolRegistry::new();
            assert!(reg.get(&name).is_none());
        }

        #[test]
        fn is_empty_reflects_state(specs in proptest::collection::vec(arb_tool_spec_no_schema(), 0..10)) {
            let mut reg = ToolRegistry::new();
            assert!(reg.is_empty());
            for spec in &specs {
                reg.register(spec.clone());
            }
            assert_eq!(reg.is_empty(), specs.is_empty());
        }

        #[test]
        fn tool_names_sorted_and_complete(specs in proptest::collection::vec(arb_tool_spec_no_schema(), 1..10)) {
            let mut reg = ToolRegistry::new();
            let mut expected: Vec<String> = Vec::new();
            for spec in &specs {
                reg.register(spec.clone());
                if !expected.contains(&spec.name) {
                    expected.push(spec.name.clone());
                }
            }
            expected.sort();
            assert_eq!(reg.tool_names(), expected);
        }

        #[test]
        fn overwrite_replaces_previous(name in "[a-zA-Z_][a-zA-Z0-9_]{0,19}", desc1 in ".*", desc2 in ".*") {
            let mut reg = ToolRegistry::new();
            reg.register(ToolSpec { name: name.clone(), description: desc1, input_schema: None });
            reg.register(ToolSpec { name: name.clone(), description: desc2.clone(), input_schema: None });
            assert_eq!(reg.get(&name).unwrap().description, desc2);
            assert_eq!(reg.tool_names().len(), 1);
        }

        #[test]
        fn tools_reflects_internal_state(specs in proptest::collection::vec(arb_tool_spec_no_schema(), 0..10)) {
            let mut reg = ToolRegistry::new();
            let mut unique_names: Vec<String> = Vec::new();
            for spec in &specs {
                reg.register(spec.clone());
                if !unique_names.contains(&spec.name) {
                    unique_names.push(spec.name.clone());
                }
            }
            unique_names.sort();
            let mut actual: Vec<String> = reg.tools().keys().cloned().collect();
            actual.sort();
            assert_eq!(actual, unique_names);
        }

        #[test]
        fn clone_preserves_all_registrations(specs in proptest::collection::vec(arb_tool_spec_no_schema(), 1..5)) {
            let mut reg = ToolRegistry::new();
            for spec in &specs {
                reg.register(spec.clone());
            }
            let cloned = reg.clone();
            assert_eq!(reg.tool_names(), cloned.tool_names());
            for name in reg.tool_names() {
                assert_eq!(reg.get(&name).unwrap().description, cloned.get(&name).unwrap().description);
            }
        }
    }

    #[test]
    fn test_json_schema_roundtrip() {
        let spec = ToolSpec {
            name: "my_tool".to_string(),
            description: "does stuff".to_string(),
            input_schema: Some(
                serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}}),
            ),
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: ToolSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, spec.name);
        assert_eq!(back.description, spec.description);
    }
}
