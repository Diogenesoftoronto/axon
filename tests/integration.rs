use altum::store::ContextStore;

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
