use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::trajectory::Trajectory;

/// Supported SFT output formats. Only OpenAI is implemented in this pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SftFormat {
    OpenaiMessages,
}

impl SftFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "openai" | "openai-messages" | "messages" => Some(SftFormat::OpenaiMessages),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportOptions {
    pub format: SftFormat,
    pub require_final: bool,
    pub min_messages: usize,
    pub max_messages: Option<usize>,
    pub max_chars_per_message: Option<usize>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: SftFormat::OpenaiMessages,
            require_final: true,
            min_messages: 2,
            max_messages: None,
            max_chars_per_message: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ExportStats {
    pub read: usize,
    pub kept: usize,
    pub skipped_no_final: usize,
    pub skipped_too_few_messages: usize,
    pub skipped_truncated_messages: usize,
}

/// A single OpenAI-format SFT example. `messages` is the actual training
/// conversation; sibling top-level fields are metadata that axolotl / Unsloth
/// recipes can key off (token counts, depth, etc).
#[derive(Clone, Debug, Serialize)]
pub struct OpenAiSftExample {
    pub messages: Vec<OpenAiMessage>,
    pub altum: AltumSftMetadata,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AltumSftMetadata {
    pub trace_id: String,
    pub model: String,
    pub sub_model: String,
    pub depth: usize,
    pub max_depth: usize,
    pub max_iterations: usize,
    pub policy_profile: String,
    pub recursive_call_count: usize,
    pub max_depth_reached: usize,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub final_answer: String,
}

/// Read JSONL trajectories from `input` and export SFT examples to `output`
/// according to `opts`. Returns aggregate stats.
pub fn export_sft<P1: AsRef<Path>, P2: AsRef<Path>>(
    input: P1,
    output: P2,
    opts: &ExportOptions,
) -> Result<ExportStats> {
    let input = input.as_ref();
    let output = output.as_ref();
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output parent dir {}", parent.display()))?;
        }
    }
    let body = std::fs::read_to_string(input)
        .with_context(|| format!("reading trajectories from {}", input.display()))?;
    let trajectories = parse_trajectory_jsonl(&body)?;
    let stats = ExportStats {
        read: trajectories.len(),
        kept: 0,
        skipped_no_final: 0,
        skipped_too_few_messages: 0,
        skipped_truncated_messages: 0,
    };
    let file = File::create(output)
        .with_context(|| format!("creating output file {}", output.display()))?;
    let mut writer = BufWriter::new(file);
    let mut stats = stats;
    for traj in &trajectories {
        if opts.require_final && traj.final_answer.trim().is_empty() {
            stats.skipped_no_final += 1;
            continue;
        }
        if traj.messages.len() < opts.min_messages {
            stats.skipped_too_few_messages += 1;
            continue;
        }
        let example = build_openai_example(traj, opts);
        if let Some(max) = opts.max_messages {
            if example.messages.len() > max {
                stats.skipped_truncated_messages += 1;
                continue;
            }
        }
        let line = serde_json::to_string(&example)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
        stats.kept += 1;
    }
    writer.flush()?;
    Ok(stats)
}

pub fn parse_trajectory_jsonl(body: &str) -> Result<Vec<Trajectory>> {
    let mut out = Vec::new();
    for (idx, line) in body.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let traj: Trajectory = serde_json::from_str(trimmed)
            .with_context(|| format!("parsing trajectory line {}", idx + 1))?;
        out.push(traj);
    }
    Ok(out)
}

/// Build a single OpenAI messages-format SFT example from a trajectory.
///
/// The key RLM-aware behavior: keep the full internal conversation log
/// (system, initial user query, assistant code blocks, sandbox observations
/// as user messages, recursive llm_query results, final assistant turn with
/// `FINAL(...)`) so the student model learns the agentic loop, not just the
/// input/output pair.
pub fn build_openai_example(traj: &Trajectory, opts: &ExportOptions) -> OpenAiSftExample {
    let mut messages: Vec<OpenAiMessage> = Vec::with_capacity(traj.messages.len() + 1);
    for (included, m) in traj.messages.iter().enumerate() {
        if let Some(max) = opts.max_messages {
            if included >= max {
                break;
            }
        }
        let content = match opts.max_chars_per_message {
            Some(max) if m.content.len() > max => truncate(&m.content, max),
            _ => m.content.clone(),
        };
        messages.push(OpenAiMessage {
            role: m.role.clone(),
            content,
        });
    }
    // Append the unwrapped final answer as a final assistant turn so trainers
    // that truncate tail messages still have a clean SFT target.
    if !traj.final_answer.trim().is_empty()
        && !messages
            .last()
            .map(|m| m.content.trim() == traj.final_answer.trim())
            .unwrap_or(false)
    {
        messages.push(OpenAiMessage {
            role: "assistant".into(),
            content: traj.final_answer.clone(),
        });
    }
    OpenAiSftExample {
        messages,
        altum: AltumSftMetadata {
            trace_id: traj.trace_id.clone(),
            model: traj.model.clone(),
            sub_model: traj.sub_model.clone(),
            depth: traj.depth,
            max_depth: traj.max_depth,
            max_iterations: traj.max_iterations,
            policy_profile: traj.policy_profile.clone(),
            recursive_call_count: traj.telemetry.recursive_call_count,
            max_depth_reached: traj.telemetry.max_depth_reached,
            prompt_tokens: traj.usage.prompt_tokens,
            completion_tokens: traj.usage.completion_tokens,
            total_tokens: traj.usage.total_tokens,
            final_answer: traj.final_answer.clone(),
        },
    }
}

fn truncate(s: &str, max: usize) -> String {
    let len = s.chars().count();
    if len <= max {
        return s.to_string();
    }
    format!(
        "{}...[+{} chars]",
        s.chars().take(max).collect::<String>(),
        len - max
    )
}

/// Convenience for the CLI: load trajectories then serialize to a generic
/// JSON array (used by `inspect` / debugging workflows).
pub fn to_json_array(trajectories: &[Trajectory]) -> Value {
    json!(trajectories)
}

/// Default sort order for SFT export: most-recent first, then by depth desc.
pub fn default_sort_key(t: &Trajectory) -> (i64, i64) {
    let ts = chrono_like_unix(&t.timestamp);
    (-ts, -(t.depth as i64))
}

fn chrono_like_unix(ts: &str) -> i64 {
    // Naive YYYY-MM-DDTHH:MM:SSZ parser; good enough for deterministic sort.
    let mut iter = ts.split('T');
    let date = iter.next().unwrap_or("");
    let time = iter.next().unwrap_or("Z");
    let time = time.trim_end_matches('Z');
    let mut dp = date.split('-');
    let y: i64 = dp.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let m: i64 = dp.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let d: i64 = dp.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut tp = time.split(':');
    let hh: i64 = tp.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let mm: i64 = tp.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let ss: i64 = tp
        .next()
        .and_then(|s| s.split('.').next().unwrap_or("").parse().ok())
        .unwrap_or(0);
    ((y - 1970) * 365 + m * 30 + d) * 86_400 + hh * 3600 + mm * 60 + ss
}

#[allow(dead_code)]
pub fn _unused_btreemap_marker() -> BTreeMap<(), ()> {
    BTreeMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Message;
    use crate::trajectory::{TrajectoryTelemetry, TrajectoryUsage};

    fn fixture() -> Trajectory {
        Trajectory {
            trace_id: "tr-1".into(),
            timestamp: "2026-06-19T00:00:00Z".into(),
            query: "What is 2+2?".into(),
            context_chars: 0,
            model: "m".into(),
            sub_model: "s".into(),
            depth: 0,
            max_depth: 1,
            max_iterations: 5,
            policy_profile: "baseline".into(),
            final_answer: "4".into(),
            messages: vec![
                Message::system("you are an rlm"),
                Message::user("What is 2+2?"),
                Message::assistant("```repl\nprint(2+2)\n```"),
                Message::user("REPL output:\n4\n"),
                Message::assistant("FINAL(4)"),
            ],
            telemetry: TrajectoryTelemetry::default(),
            usage: TrajectoryUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        }
    }

    #[test]
    fn build_openai_example_preserves_full_conversation() {
        let t = fixture();
        let ex = build_openai_example(&t, &ExportOptions::default());
        let roles: Vec<&str> = ex.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(
            roles,
            vec![
                "system",
                "user",
                "assistant",
                "user",
                "assistant",
                "assistant"
            ]
        );
        assert_eq!(ex.messages.last().unwrap().content, "4");
        assert_eq!(ex.altum.final_answer, "4");
        assert_eq!(ex.altum.total_tokens, 15);
    }

    #[test]
    fn build_openai_example_does_not_duplicate_final_when_already_wrapped() {
        let t = fixture();
        let ex = build_openai_example(&t, &ExportOptions::default());
        let last = ex.messages.last().unwrap().content.clone();
        assert_eq!(last, "4");
    }

    #[test]
    fn export_sft_skips_trajectories_without_final_answer() {
        let dir = tempfile::tempdir().unwrap();
        let inp = dir.path().join("in.jsonl");
        let out = dir.path().join("out.jsonl");
        let mut bad = fixture();
        bad.final_answer = "".into();
        std::fs::write(&inp, format!("{}\n", serde_json::to_string(&bad).unwrap())).unwrap();
        let stats = export_sft(&inp, &out, &ExportOptions::default()).unwrap();
        assert_eq!(stats.read, 1);
        assert_eq!(stats.kept, 0);
        assert_eq!(stats.skipped_no_final, 1);
    }

    #[test]
    fn export_sft_skips_trajectories_with_too_few_messages() {
        let dir = tempfile::tempdir().unwrap();
        let inp = dir.path().join("in.jsonl");
        let out = dir.path().join("out.jsonl");
        let mut small = fixture();
        small.messages.truncate(1);
        std::fs::write(
            &inp,
            format!("{}\n", serde_json::to_string(&small).unwrap()),
        )
        .unwrap();
        let stats = export_sft(&inp, &out, &ExportOptions::default()).unwrap();
        assert_eq!(stats.read, 1);
        assert_eq!(stats.kept, 0);
        assert_eq!(stats.skipped_too_few_messages, 1);
    }

    #[test]
    fn export_sft_keeps_valid_trajectories_and_writes_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let inp = dir.path().join("in.jsonl");
        let out = dir.path().join("out.jsonl");
        std::fs::write(
            &inp,
            format!("{}\n", serde_json::to_string(&fixture()).unwrap()),
        )
        .unwrap();
        let stats = export_sft(&inp, &out, &ExportOptions::default()).unwrap();
        assert_eq!(stats.kept, 1);
        let body = std::fs::read_to_string(&out).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 1);
        let parsed: Value = serde_json::from_str(lines[0]).unwrap();
        assert!(parsed["messages"].is_array());
        assert_eq!(parsed["messages"].as_array().unwrap().len(), 6);
        assert_eq!(parsed["altum"]["final_answer"], "4");
    }

    #[test]
    fn max_chars_per_message_truncates_long_content() {
        let mut t = fixture();
        t.messages[1].content = "x".repeat(2000);
        let ex = build_openai_example(
            &t,
            &ExportOptions {
                max_chars_per_message: Some(100),
                ..Default::default()
            },
        );
        assert!(ex.messages[1].content.len() < 2000);
        assert!(ex.messages[1].content.contains("[+"));
    }

    #[test]
    fn sft_format_parses_known_aliases() {
        assert_eq!(SftFormat::parse("openai"), Some(SftFormat::OpenaiMessages));
        assert_eq!(
            SftFormat::parse("openai-messages"),
            Some(SftFormat::OpenaiMessages)
        );
        assert_eq!(
            SftFormat::parse("messages"),
            Some(SftFormat::OpenaiMessages)
        );
        assert_eq!(SftFormat::parse("OPENAI"), Some(SftFormat::OpenaiMessages));
        assert_eq!(SftFormat::parse("bogus"), None);
    }

    #[test]
    fn default_sort_key_is_descending_by_timestamp() {
        let a = fixture();
        let mut b = fixture();
        b.timestamp = "2026-06-20T00:00:00Z".into();
        assert!(default_sort_key(&b) < default_sort_key(&a));
    }
}
