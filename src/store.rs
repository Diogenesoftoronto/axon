use anyhow::Result;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ContextStore {
    base_dir: PathBuf,
}

impl ContextStore {
    pub fn new(base_dir: &Path) -> Self {
        Self {
            base_dir: base_dir.to_path_buf(),
        }
    }

    pub fn read_context(&self, thread_id: &str) -> String {
        fs::read_to_string(self.context_path(thread_id)).unwrap_or_default()
    }

    pub fn append_context(&self, thread_id: &str, text: &str) -> Result<()> {
        let path = self.context_path(thread_id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        write!(file, "{}", text)?;
        Ok(())
    }

    fn context_path(&self, thread_id: &str) -> PathBuf {
        self.base_dir.join(thread_id).join("context.txt")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_append_empty_string_is_noop_for_length() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(dir.path());
        store.append_context("t", "").unwrap();
        assert_eq!(store.read_context("t"), "");
    }

    #[test]
    fn test_unicode_content_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(dir.path());
        let unicode = "こんにちは 🚀 ünïcödé";
        store.append_context("t", unicode).unwrap();
        assert_eq!(store.read_context("t"), unicode);
    }

    #[test]
    fn test_thread_id_with_slash_creates_nested_path() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(dir.path());
        store.append_context("team/alpha", "x").unwrap();
        assert!(dir
            .path()
            .join("team")
            .join("alpha")
            .join("context.txt")
            .exists());
    }

    proptest! {
        #[test]
        fn append_then_read_returns_concatenation(
            chunks in proptest::collection::vec("[a-zA-Z0-9 ]{0,20}", 1..10)
        ) {
            let dir = tempfile::tempdir().unwrap();
            let store = ContextStore::new(dir.path());
            let mut expected = String::new();
            for chunk in &chunks {
                store.append_context("p", chunk).unwrap();
                expected.push_str(chunk);
            }
            prop_assert_eq!(store.read_context("p"), expected);
        }
    }
}
