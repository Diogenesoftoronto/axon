use altum::export::{build_openai_example, export_sft, parse_trajectory_jsonl, ExportOptions};
use altum::llm::Message;
use altum::trajectory::{Trajectory, TrajectoryRecorder, TrajectoryTelemetry, TrajectoryUsage};

fn fixture_trajectory() -> Trajectory {
    Trajectory {
        trace_id: "tr-int-1".into(),
        timestamp: "2026-06-19T00:00:00Z".into(),
        query: "What is 6*7?".into(),
        context_chars: 0,
        model: "m".into(),
        sub_model: "s".into(),
        depth: 0,
        max_depth: 1,
        max_iterations: 5,
        policy_profile: "baseline".into(),
        final_answer: "42".into(),
        messages: vec![
            Message::system("you are altum"),
            Message::user("What is 6*7?"),
            Message::assistant("```repl\nprint(6*7)\n```"),
            Message::user("Code executed:\n```python\nprint(6*7)\n```\n\nREPL output:\n42\n"),
            Message::assistant("FINAL(42)"),
        ],
        telemetry: TrajectoryTelemetry {
            max_depth_reached: 0,
            recursive_call_count: 0,
            recursive_calls_by_depth: Default::default(),
        },
        usage: TrajectoryUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        },
    }
}

#[test]
fn end_to_end_record_then_export_to_openai_messages() {
    let dir = tempfile::tempdir().unwrap();
    let trace_path = dir.path().join("trace.jsonl");
    let sft_path = dir.path().join("sft.jsonl");

    // 1. Record a trajectory via the recorder.
    let rec = TrajectoryRecorder::to_path(&trace_path).unwrap();
    rec.record(&fixture_trajectory()).unwrap();
    rec.flush().unwrap();
    let body = std::fs::read_to_string(&trace_path).unwrap();
    assert_eq!(body.lines().count(), 1);

    // 2. Parse the JSONL back.
    let trajectories = parse_trajectory_jsonl(&body).unwrap();
    assert_eq!(trajectories.len(), 1);
    assert_eq!(trajectories[0].final_answer, "42");

    // 3. Run the exporter.
    let stats = export_sft(&trace_path, &sft_path, &ExportOptions::default()).unwrap();
    assert_eq!(stats.read, 1);
    assert_eq!(stats.kept, 1);
    assert_eq!(stats.skipped_no_final, 0);

    // 4. Validate the output SFT example is OpenAI-messages compatible and
    //    RLM-aware (full conversation log preserved).
    let sft_body = std::fs::read_to_string(&sft_path).unwrap();
    let example: serde_json::Value =
        serde_json::from_str(sft_body.lines().next().unwrap()).unwrap();
    let messages = example["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
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
    assert_eq!(messages.last().unwrap()["content"], "42");
    assert!(messages[2]["content"]
        .as_str()
        .unwrap()
        .contains("print(6*7)"));
    assert!(messages[3]["content"]
        .as_str()
        .unwrap()
        .contains("REPL output"));

    // 5. axolotl-friendly metadata sidecar.
    let altum = &example["altum"];
    assert_eq!(altum["trace_id"], "tr-int-1");
    assert_eq!(altum["total_tokens"], 150);
    assert_eq!(altum["max_iterations"], 5);
}

#[test]
fn end_to_end_skips_failed_and_keeps_successful() {
    let dir = tempfile::tempdir().unwrap();
    let trace_path = dir.path().join("trace.jsonl");
    let sft_path = dir.path().join("sft.jsonl");

    let mut bad = fixture_trajectory();
    bad.trace_id = "tr-bad".into();
    bad.final_answer = "".into(); // simulate no FINAL produced

    let mut good = fixture_trajectory();
    good.trace_id = "tr-good".into();

    let rec = TrajectoryRecorder::to_path(&trace_path).unwrap();
    rec.record(&bad).unwrap();
    rec.record(&good).unwrap();
    rec.flush().unwrap();

    let stats = export_sft(&trace_path, &sft_path, &ExportOptions::default()).unwrap();
    assert_eq!(stats.read, 2);
    assert_eq!(stats.kept, 1);
    assert_eq!(stats.skipped_no_final, 1);
    let body = std::fs::read_to_string(&sft_path).unwrap();
    assert_eq!(body.lines().count(), 1);
    let example: serde_json::Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
    assert_eq!(example["altum"]["trace_id"], "tr-good");
}

#[test]
fn build_openai_example_keeps_code_block_and_observation() {
    let traj = fixture_trajectory();
    let ex = build_openai_example(&traj, &ExportOptions::default());
    assert!(ex.messages[2].content.contains("```repl"));
    assert!(ex.messages[2].content.contains("print(6*7)"));
    assert!(ex.messages[3].content.contains("REPL output"));
    assert!(ex.messages[3].content.contains("42"));
}

#[test]
fn max_messages_truncates_conversation_log() {
    let mut traj = fixture_trajectory();
    // Add filler assistant turns to exceed a small cap.
    for i in 0..5 {
        traj.messages
            .push(Message::assistant(&format!("filler {i}")));
    }
    let opts = ExportOptions {
        max_messages: Some(4),
        require_final: false,
        ..Default::default()
    };
    let ex = build_openai_example(&traj, &opts);
    // Internal log is truncated to max_messages (4), then the unwrapped final
    // answer "42" is appended as a final assistant turn (the internal last
    // message is the wrapper "FINAL(42)" so it does not dedup).
    assert_eq!(ex.messages.len(), 5);
    assert_eq!(ex.messages.last().unwrap().content, "42");
}
