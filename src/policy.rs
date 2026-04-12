use std::collections::HashMap;
use std::fs;
use std::path::Path;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum DepthEnforcementMode {
    #[default]
    Off,
    Soft,
    Strict,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyProfile {
    pub name: String,
    pub format_contract_template: String,
    pub depth_strategy_template: String,
    #[serde(default)]
    pub prepend_to_context: bool,
    #[serde(default)]
    pub default_depth_enforcement: DepthEnforcementMode,
    #[serde(default)]
    pub default_require_min_depth: Option<usize>,
    #[serde(default)]
    pub default_require_min_recursive_calls: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyCatalogFile {
    pub default_profile: String,
    pub profiles: Vec<PolicyProfile>,
}

#[derive(Clone, Debug)]
pub struct PolicyCatalog {
    default_profile: String,
    by_name: HashMap<String, PolicyProfile>,
}

#[derive(Clone, Debug)]
pub struct RuntimePolicy {
    pub profile_name: String,
    pub format_contract_template: String,
    pub depth_strategy_template: String,
    pub inject_policy_into_context: bool,
    pub depth_enforcement: DepthEnforcementMode,
    pub require_min_depth: Option<usize>,
    pub require_min_recursive_calls: Option<usize>,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        let catalog = PolicyCatalog::builtin();
        catalog.build_runtime_policy(
            Some(&catalog.default_profile),
            Some(false),
            Some(DepthEnforcementMode::Off),
            None,
            None,
        )
    }
}

impl RuntimePolicy {
    pub fn prompt_instruction_block(&self) -> Option<String> {
        if self.format_contract_template.trim().is_empty()
            && self.depth_strategy_template.trim().is_empty()
            && self.depth_enforcement == DepthEnforcementMode::Off
        {
            return None;
        }

        let mut lines = Vec::new();
        lines.push(format!("Active policy profile: {}", self.profile_name));

        if !self.format_contract_template.trim().is_empty() {
            lines.push("Format contract:".to_string());
            lines.push(self.format_contract_template.trim().to_string());
        }
        if !self.depth_strategy_template.trim().is_empty() {
            lines.push("Depth strategy:".to_string());
            lines.push(self.depth_strategy_template.trim().to_string());
        }
        if self.depth_enforcement != DepthEnforcementMode::Off {
            lines.push(format!("Depth enforcement: {:?}", self.depth_enforcement));
            if let Some(v) = self.require_min_depth {
                lines.push(format!("Minimum depth required: {}", v));
            }
            if let Some(v) = self.require_min_recursive_calls {
                lines.push(format!("Minimum recursive llm_query calls required: {}", v));
            }
        }
        Some(lines.join("\n"))
    }

    pub fn prepend_to_context(&self, context: &str) -> String {
        let mut policy_text = String::new();
        if !self.format_contract_template.trim().is_empty() {
            policy_text.push_str(self.format_contract_template.trim());
            policy_text.push('\n');
        }
        if !self.depth_strategy_template.trim().is_empty() {
            policy_text.push_str(self.depth_strategy_template.trim());
            policy_text.push('\n');
        }
        if self.depth_enforcement != DepthEnforcementMode::Off {
            policy_text.push_str(&format!(
                "Depth enforcement mode: {:?}\n",
                self.depth_enforcement
            ));
        }
        if let Some(v) = self.require_min_depth {
            policy_text.push_str(&format!("Required minimum depth: {}\n", v));
        }
        if let Some(v) = self.require_min_recursive_calls {
            policy_text.push_str(&format!("Required minimum recursive calls: {}\n", v));
        }
        if policy_text.trim().is_empty() {
            return context.to_string();
        }
        format!(
            "[[AXON_POLICY_BEGIN]]\n{}\n[[AXON_POLICY_END]]\n\n{}",
            policy_text.trim(),
            context
        )
    }

    pub fn for_sub_rlm(&self) -> Self {
        let mut next = self.clone();
        // Sub-RLMs should not hard-fail parent policy gates directly.
        next.depth_enforcement = DepthEnforcementMode::Off;
        next.inject_policy_into_context = false;
        next
    }
}

impl PolicyCatalog {
    pub fn load(path: &Path) -> Self {
        if path.exists() {
            let parsed = fs::read_to_string(path)
                .ok()
                .and_then(|raw| serde_json::from_str::<PolicyCatalogFile>(&raw).ok());
            if let Some(file) = parsed {
                return Self::from_file(file);
            }
        }
        Self::builtin()
    }

    pub fn builtin() -> Self {
        let file = PolicyCatalogFile {
            default_profile: "baseline".to_string(),
            profiles: vec![
                PolicyProfile {
                    name: "baseline".to_string(),
                    format_contract_template:
                        "Return the final answer in FINAL(...) with no extra text.".to_string(),
                    depth_strategy_template:
                        "Use recursive decomposition when needed, otherwise answer directly."
                            .to_string(),
                    prepend_to_context: false,
                    default_depth_enforcement: DepthEnforcementMode::Off,
                    default_require_min_depth: None,
                    default_require_min_recursive_calls: None,
                },
                PolicyProfile {
                    name: "format_strict".to_string(),
                    format_contract_template: "STRICT OUTPUT CONTRACT: one line FINAL(<answer>); no prose, no code blocks, no tool traces.".to_string(),
                    depth_strategy_template: "".to_string(),
                    prepend_to_context: true,
                    default_depth_enforcement: DepthEnforcementMode::Off,
                    default_require_min_depth: None,
                    default_require_min_recursive_calls: None,
                },
                PolicyProfile {
                    name: "depth_soft".to_string(),
                    format_contract_template:
                        "Return only FINAL(<answer>) in the final response.".to_string(),
                    depth_strategy_template: "Decompose into subproblems and call llm_query for independent/high-complexity parts before FINAL(...).".to_string(),
                    prepend_to_context: true,
                    default_depth_enforcement: DepthEnforcementMode::Soft,
                    default_require_min_depth: Some(1),
                    default_require_min_recursive_calls: Some(1),
                },
                PolicyProfile {
                    name: "depth_strict".to_string(),
                    format_contract_template:
                        "Return only FINAL(<answer>) in the final response.".to_string(),
                    depth_strategy_template: "You must use recursive llm_query calls and verify sub-results before FINAL(...).".to_string(),
                    prepend_to_context: true,
                    default_depth_enforcement: DepthEnforcementMode::Strict,
                    default_require_min_depth: Some(1),
                    default_require_min_recursive_calls: Some(1),
                },
                PolicyProfile {
                    name: "format_plus_depth".to_string(),
                    format_contract_template: "STRICT OUTPUT CONTRACT: exactly FINAL(<answer>) on one line; no markdown, no repl blocks, no scratchpad.".to_string(),
                    depth_strategy_template: "Break task into 2-4 subproblems, use llm_query for hard parts, merge, verify, then FINAL(...).".to_string(),
                    prepend_to_context: true,
                    default_depth_enforcement: DepthEnforcementMode::Soft,
                    default_require_min_depth: Some(1),
                    default_require_min_recursive_calls: Some(1),
                },
            ],
        };
        Self::from_file(file)
    }

    fn from_file(file: PolicyCatalogFile) -> Self {
        let mut by_name = HashMap::new();
        for p in file.profiles {
            by_name.insert(p.name.clone(), p);
        }
        let default_profile = if by_name.contains_key(&file.default_profile) {
            file.default_profile
        } else {
            "baseline".to_string()
        };
        Self {
            default_profile,
            by_name,
        }
    }

    pub fn build_runtime_policy(
        &self,
        profile_name: Option<&str>,
        inject_policy_into_context: Option<bool>,
        depth_enforcement: Option<DepthEnforcementMode>,
        require_min_depth: Option<usize>,
        require_min_recursive_calls: Option<usize>,
    ) -> RuntimePolicy {
        let name = profile_name.unwrap_or(&self.default_profile);
        let profile = self
            .by_name
            .get(name)
            .or_else(|| self.by_name.get(&self.default_profile))
            .expect("policy catalog must have at least one profile");

        RuntimePolicy {
            profile_name: profile.name.clone(),
            format_contract_template: profile.format_contract_template.clone(),
            depth_strategy_template: profile.depth_strategy_template.clone(),
            inject_policy_into_context: inject_policy_into_context
                .unwrap_or(profile.prepend_to_context),
            depth_enforcement: depth_enforcement.unwrap_or(profile.default_depth_enforcement),
            require_min_depth: require_min_depth.or(profile.default_require_min_depth),
            require_min_recursive_calls: require_min_recursive_calls
                .or(profile.default_require_min_recursive_calls),
        }
    }
}
