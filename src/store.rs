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
