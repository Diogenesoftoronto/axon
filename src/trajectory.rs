use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::llm::Message;
use crate::rlm::DepthTelemetry;

/// A single completed RLM run, suitable for later conversion to SFT data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trajectory {
    pub trace_id: String,
    pub timestamp: String,
    pub query: String,
    pub context_chars: usize,
    pub model: String,
    pub sub_model: String,
    pub depth: usize,
    pub max_depth: usize,
    pub max_iterations: usize,
    pub policy_profile: String,
    pub final_answer: String,
    pub messages: Vec<Message>,
    pub telemetry: TrajectoryTelemetry,
    pub usage: TrajectoryUsage,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrajectoryTelemetry {
    pub max_depth_reached: usize,
    pub recursive_call_count: usize,
    pub recursive_calls_by_depth: std::collections::BTreeMap<usize, usize>,
}

impl From<&DepthTelemetry> for TrajectoryTelemetry {
    fn from(t: &DepthTelemetry) -> Self {
        Self {
            max_depth_reached: t.max_depth_reached,
            recursive_call_count: t.recursive_call_count,
            recursive_calls_by_depth: t.recursive_calls_by_depth.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrajectoryUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

impl TrajectoryUsage {
    pub fn add(&mut self, prompt: Option<i32>, completion: Option<i32>, total: Option<i32>) {
        self.prompt_tokens = self
            .prompt_tokens
            .saturating_add(prompt.unwrap_or(0).max(0) as u64);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(completion.unwrap_or(0).max(0) as u64);
        self.total_tokens = self
            .total_tokens
            .saturating_add(total.unwrap_or(0).max(0) as u64);
    }
}

/// Append-only JSONL sink for [`Trajectory`] records.
///
/// Off by default. Construct with [`TrajectoryRecorder::to_path`] when
/// `--trace-output` is set on the CLI.
pub struct TrajectoryRecorder {
    writer: Mutex<BufWriter<File>>,
    path: PathBuf,
}

impl TrajectoryRecorder {
    pub fn to_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
            path,
        })
    }

    /// Append a single trajectory as one JSONL line.
    pub fn record(&self, trajectory: &Trajectory) -> Result<()> {
        let line = serde_json::to_string(trajectory)?;
        let mut guard = self
            .writer
            .lock()
            .map_err(|_| anyhow::anyhow!("TrajectoryRecorder mutex poisoned"))?;
        guard.write_all(line.as_bytes())?;
        guard.write_all(b"\n")?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flush the underlying writer.
    pub fn flush(&self) -> Result<()> {
        let mut guard = self
            .writer
            .lock()
            .map_err(|_| anyhow::anyhow!("TrajectoryRecorder mutex poisoned"))?;
        guard.flush()?;
        Ok(())
    }
}

pub fn new_trace_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    format!("tr-{nanos:x}-{pid:x}")
}

pub fn rfc3339_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days_from_epoch = secs / 86_400;
    let mut year = 1970i64;
    let mut remaining = days_from_epoch as i64;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let year_days = if leap { 366 } else { 365 };
        if remaining < year_days {
            break;
        }
        remaining -= year_days;
        year += 1;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let months = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    let mut day = remaining + 1;
    for &md in &months {
        if day <= md {
            break;
        }
        day -= md;
        month += 1;
    }
    let tod = secs % 86_400;
    let hh = tod / 3600;
    let mm = (tod % 3600) / 60;
    let ss = tod % 60;
    format!("{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Message;

    fn fixture_trajectory() -> Trajectory {
        Trajectory {
            trace_id: "tr-abc".into(),
            timestamp: "2026-06-19T00:00:00Z".into(),
            query: "What is 2+2?".into(),
            context_chars: 0,
            model: "test-model".into(),
            sub_model: "test-sub".into(),
            depth: 0,
            max_depth: 1,
            max_iterations: 5,
            policy_profile: "baseline".into(),
            final_answer: "4".into(),
            messages: vec![
                Message::system("you are an rlm"),
                Message::user("What is 2+2?"),
                Message::assistant("```repl\nprint(2+2)\n```"),
                Message::user("REPL output: 4\n"),
                Message::assistant("FINAL(4)"),
            ],
            telemetry: TrajectoryTelemetry {
                max_depth_reached: 0,
                recursive_call_count: 0,
                recursive_calls_by_depth: Default::default(),
            },
            usage: TrajectoryUsage {
                prompt_tokens: 50,
                completion_tokens: 25,
                total_tokens: 75,
            },
        }
    }

    #[test]
    fn trajectory_serializes_roundtrip() {
        let t = fixture_trajectory();
        let s = serde_json::to_string(&t).unwrap();
        let back: Trajectory = serde_json::from_str(&s).unwrap();
        assert_eq!(back.query, t.query);
        assert_eq!(back.final_answer, t.final_answer);
        assert_eq!(back.messages.len(), t.messages.len());
        assert_eq!(back.usage.total_tokens, 75);
    }

    #[test]
    fn trajectory_recorder_appends_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let rec = TrajectoryRecorder::to_path(&path).unwrap();
        rec.record(&fixture_trajectory()).unwrap();
        rec.record(&fixture_trajectory()).unwrap();
        rec.flush().unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        let parsed: Trajectory = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed.trace_id, "tr-abc");
    }

    #[test]
    fn trajectory_recorder_creates_missing_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("trace.jsonl");
        let rec = TrajectoryRecorder::to_path(&path).unwrap();
        rec.record(&fixture_trajectory()).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn usage_accumulates_token_counts() {
        let mut u = TrajectoryUsage::default();
        u.add(Some(10), Some(5), Some(15));
        u.add(Some(20), None, None);
        assert_eq!(u.prompt_tokens, 30);
        assert_eq!(u.completion_tokens, 5);
        assert_eq!(u.total_tokens, 15);
    }

    #[test]
    fn rfc3339_now_is_well_formed() {
        let s = rfc3339_now();
        assert!(s.ends_with('Z'));
        assert_eq!(s.len(), 20);
    }

    #[test]
    fn new_trace_id_is_unique_per_call() {
        let a = new_trace_id();
        let b = new_trace_id();
        assert!(a.starts_with("tr-"));
        assert_ne!(a, b);
    }

    #[test]
    fn depth_telemetry_conversion_preserves_counts() {
        let mut t = DepthTelemetry::new(0);
        t.record_spawn(1);
        t.record_spawn(2);
        t.record_spawn(1);
        let conv: TrajectoryTelemetry = (&t).into();
        assert_eq!(conv.recursive_call_count, 3);
        assert_eq!(conv.max_depth_reached, 2);
        assert_eq!(conv.recursive_calls_by_depth.get(&1), Some(&2));
        assert_eq!(conv.recursive_calls_by_depth.get(&2), Some(&1));
    }
}
