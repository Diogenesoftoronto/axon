use std::io::Cursor;

use axon::store::ContextStore;

#[test]
fn test_store_read_empty() {
    let dir = tempfile::tempdir().unwrap();
    let store = ContextStore::new(dir.path());
    let ctx = store.read_context("nonexistent");
    assert_eq!(ctx, "");
}

#[test]
fn test_store_append_and_read() {
    let dir = tempfile::tempdir().unwrap();
    let store = ContextStore::new(dir.path());

    store.append_context("test-thread", "hello ").unwrap();
    store.append_context("test-thread", "world").unwrap();

    let ctx = store.read_context("test-thread");
    assert_eq!(ctx, "hello world");
}

#[test]
fn test_store_separate_threads() {
    let dir = tempfile::tempdir().unwrap();
    let store = ContextStore::new(dir.path());

    store.append_context("thread-a", "alpha").unwrap();
    store.append_context("thread-b", "beta").unwrap();

    assert_eq!(store.read_context("thread-a"), "alpha");
    assert_eq!(store.read_context("thread-b"), "beta");
}

#[test]
fn test_mcp_message_roundtrip() {
    // Test the Content-Length framed JSON-RPC message format
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 42
    });

    let body = serde_json::to_string(&msg).unwrap();
    let framed = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);

    let mut reader = Cursor::new(framed.as_bytes().to_vec());
    let mut line = String::new();
    let mut content_length: usize = 0;

    // Parse headers
    loop {
        line.clear();
        use std::io::BufRead;
        let n = reader.read_line(&mut line).unwrap();
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            content_length = val.trim().parse().unwrap();
        }
    }

    // Read body
    let mut body_buf = vec![0u8; content_length];
    use std::io::Read;
    reader.read_exact(&mut body_buf).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&body_buf).unwrap();

    assert_eq!(parsed["method"], "tools/list");
    assert_eq!(parsed["id"], 42);
}

#[test]
fn test_store_context_file_path() {
    let dir = tempfile::tempdir().unwrap();
    let store = ContextStore::new(dir.path());

    store.append_context("my-thread", "data").unwrap();

    // Verify the file is at the expected path
    let expected = dir.path().join("my-thread").join("context.txt");
    assert!(expected.exists());
    assert_eq!(std::fs::read_to_string(expected).unwrap(), "data");
}

#[test]
fn test_store_large_append() {
    let dir = tempfile::tempdir().unwrap();
    let store = ContextStore::new(dir.path());

    let large_text = "x".repeat(100_000);
    store.append_context("big", &large_text).unwrap();

    let ctx = store.read_context("big");
    assert_eq!(ctx.len(), 100_000);
}
