//! Pluggable Virtual Filesystem boundary.
//!
//! The RLM exposes a small VFS surface to sandbox code: `VFS_WRITE`,
//! `VFS_READ`, `VFS_LIST`. The default in-memory backend stores entries in a
//! `BTreeMap<String, Vec<u8>>` and snapshots cheaply via deep clone. Two
//! additional backends are shipped behind feature flags:
//!
//! - `FsVfs` (always available) mirrors the VFS onto a host directory. Used for
//!   local persistence tests and as the cheapest persistent surface.
//! - `S3Vfs` (feature `s3`) stores content-addressed blobs and per-fork
//!   manifests on R2/MinIO/AWS S3, enabling copy-on-write forks across hosts.
//!
//! All backends implement [`VfsHandle`] via an `async` trait object. The
//! boundary is async because remote backends would otherwise block the
//! Tokio runtime inside the RLM loop. Fork semantics are provided by
//! [`VfsHandle::snapshot`], which produces an independent handle (deep clone
//! in-memory; manifest clone for remote) suitable for a new branch.
//!
//! Object layout for `S3Vfs`:
//!
//! ```text
//! {prefix}/blobs/{content_hash}
//! {prefix}/namespaces/{namespace}/manifests/current.json
//! ```
//!
//! `content_hash` is the SHA-256 of the VFS entry payload. Immutable blobs are
//! shared across namespaces; manifests carry a path→blob-hash table plus a
//! parent id for provenance. Mutations on one handle are serialized, while
//! snapshots write to a new manifest namespace.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// One path:content_address mapping inside a manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    pub path: String,
    pub blob_hash: String,
    pub size: u64,
}

/// Identifier for content-addressed VFS state. Cheap to clone for fork
/// semantics — the underlying blobs are immutable and shared.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VfsManifest {
    pub entries: Vec<ManifestEntry>,
    /// Optional parent manifest id, for COW provenance tracking.
    pub parent: Option<String>,
}

impl VfsManifest {
    /// Returns the entry matching `path`, if any.
    pub fn get(&self, path: &str) -> Option<&ManifestEntry> {
        self.entries.iter().find(|e| e.path == path)
    }

    /// Inserts or replaces the entry for `path`.
    pub fn upsert(&mut self, path: &str, hash: &str, size: u64) {
        if let Some(slot) = self.entries.iter_mut().find(|e| e.path == path) {
            slot.blob_hash = hash.to_string();
            slot.size = size;
        } else {
            self.entries.push(ManifestEntry {
                path: path.to_string(),
                blob_hash: hash.to_string(),
                size,
            });
        }
    }

    /// Removes the entry matching `path`. Returns true if it existed.
    pub fn remove(&mut self, path: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.path != path);
        self.entries.len() < before
    }

    /// Lists all paths whose key starts with `prefix`.
    pub fn list(&self, prefix: &str) -> Vec<String> {
        let mut paths: Vec<String> = self
            .entries
            .iter()
            .filter(|e| e.path.starts_with(prefix))
            .map(|e| e.path.clone())
            .collect();
        paths.sort();
        paths
    }
}

/// Normalizes a VFS path so that lookups are stable across backends.
///
/// Mirrors the in-memory semantics the RLM historically shipped (paths must
/// start with `/`, empty becomes `/`). Backends receive the canonical
/// string and need not re-normalize.
pub fn normalize_path(path: &str) -> String {
    let trimmed = path.trim().trim_matches('"').trim_matches('\'');
    if trimmed.is_empty() {
        return "/".to_string();
    }
    if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    }
}

/// Normalizes a list prefix to always end with `/`, mirroring historical
/// `VFS_LIST` semantics.
pub fn normalize_prefix(path: &str) -> String {
    let mut p = normalize_path(path);
    if !p.ends_with('/') && p != "/" {
        p.push('/');
    }
    p
}

/// Compute the content-addressed blob id for a payload.
pub fn content_hash(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// The pluggable per-fork VFS surface.
///
/// Implementations own a path→payload mapping for exactly one fork. Fork
/// creation is expressed by [`snapshot`], which returns an independent
/// handle preserving the current path set without sharing mutable state.
///
/// `Send + Sync` is mandated because RLM instances are passed across async
/// tasks on a multi-thread runtime.
#[async_trait]
pub trait VfsHandle: Send + Sync {
    /// Reads the payload stored at `path`, or `None` if missing.
    async fn read(&self, path: &str) -> Result<Option<Vec<u8>>>;

    /// Writes `data` to `path`. Replaces any existing payload.
    async fn write(&self, path: &str, data: Vec<u8>) -> Result<()>;

    /// Lists paths starting with `prefix`.
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Deletes `path`. Returns true if something was removed.
    async fn delete(&self, path: &str) -> Result<bool>;

    /// Creates an independent snapshot of the current state. Fork creation
    /// in the RLM uses this to seed a new branch from a checkpoint.
    async fn snapshot(&self) -> Result<Box<dyn VfsHandle>>;

    /// Returns a human-readable backend tag (e.g. `"memory"`, `"fs"`, `"s3"`).
    fn backend(&self) -> &'static str;
}

/// Factory that mints fresh [`VfsHandle`] instances, one per fork.
#[async_trait]
pub trait VfsFactory: Send + Sync {
    /// Creates a fresh, empty handle for a new fork or top-level session.
    async fn create(&self) -> Result<Box<dyn VfsHandle>>;
    /// Backend tag propagating to the RLM telemetry.
    fn backend(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------------

/// In-memory `BTreeMap` VFS. Snapshots deep-clone the map so each fork has
/// its own mutable state.
#[derive(Debug, Default, Clone)]
pub struct MemoryVfs {
    entries: Arc<RwLock<BTreeMap<String, Vec<u8>>>>,
}

impl MemoryVfs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.read().map(|m| m.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl VfsHandle for MemoryVfs {
    async fn read(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let p = normalize_path(path);
        let guard = self
            .entries
            .read()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?;
        Ok(guard.get(&p).cloned())
    }

    async fn write(&self, path: &str, data: Vec<u8>) -> Result<()> {
        let p = normalize_path(path);
        let mut guard = self
            .entries
            .write()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?;
        guard.insert(p, data);
        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let p = normalize_prefix(prefix);
        let guard = self
            .entries
            .read()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?;
        let mut out: Vec<String> = guard
            .keys()
            .filter(|k| k.starts_with(&p))
            .cloned()
            .collect();
        out.sort();
        Ok(out)
    }

    async fn delete(&self, path: &str) -> Result<bool> {
        let p = normalize_path(path);
        let mut guard = self
            .entries
            .write()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?;
        Ok(guard.remove(&p).is_some())
    }

    async fn snapshot(&self) -> Result<Box<dyn VfsHandle>> {
        let guard = self
            .entries
            .read()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?;
        let cloned: BTreeMap<String, Vec<u8>> = guard.clone();
        drop(guard);
        Ok(Box::new(MemoryVfs {
            entries: Arc::new(RwLock::new(cloned)),
        }))
    }

    fn backend(&self) -> &'static str {
        "memory"
    }
}

/// Factory producing empty [`MemoryVfs`] handles.
#[derive(Debug, Default, Clone, Copy)]
pub struct MemoryVfsFactory;

#[async_trait]
impl VfsFactory for MemoryVfsFactory {
    async fn create(&self) -> Result<Box<dyn VfsHandle>> {
        Ok(Box::new(MemoryVfs::new()))
    }
    fn backend(&self) -> &'static str {
        "memory"
    }
}

// ---------------------------------------------------------------------------
// Host-filesystem backend
// ---------------------------------------------------------------------------

/// VFS mirrored onto a host directory. One file per VFS entry. Snapshots
/// deep-copy the directory. Useful for local-persistence tests and as the
/// simplest non-memory backend.
#[derive(Debug, Clone)]
pub struct FsVfs {
    root: PathBuf,
    /// Subdirectory mirroring this instance's isolated namespace. Forks get
    /// their own subdirectory under the factory root.
    dir: PathBuf,
}

impl FsVfs {
    fn path_for(&self, raw_path: &str) -> Result<PathBuf> {
        let normalized = normalize_path(raw_path);
        // Strip the logical leading slash so the host join stays relative.
        // Validate components before sanitizing: preserving `..` here would
        // let sandbox-controlled paths escape this handle's namespace.
        let stripped = normalized.trim_start_matches('/');
        let mut safe = PathBuf::new();
        for component in Path::new(stripped).components() {
            match component {
                Component::Normal(part) => {
                    let sanitized: String = part
                        .to_string_lossy()
                        .chars()
                        .map(|c| {
                            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                                c
                            } else {
                                '_'
                            }
                        })
                        .collect();
                    safe.push(sanitized);
                }
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(anyhow!(
                        "invalid VFS path: parent or root components are forbidden"
                    ));
                }
            }
        }
        Ok(self.dir.join(safe))
    }
}

#[async_trait]
impl VfsHandle for FsVfs {
    async fn read(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let p = self.path_for(path)?;
        match tokio::fs::read(&p).await {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("VFS read failed: {}", p.display())),
        }
    }

    async fn write(&self, path: &str, data: Vec<u8>) -> Result<()> {
        let p = self.path_for(path)?;
        if let Some(parent) = p.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        tokio::fs::write(&p, data)
            .await
            .with_context(|| format!("VFS write failed: {}", p.display()))?;
        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let p = normalize_prefix(prefix);
        if !self.dir.exists() {
            return Ok(Vec::new());
        }
        let mut paths = Vec::new();
        collect_paths(&self.dir, &self.dir, &p, &mut paths).await?;
        paths.sort();
        Ok(paths)
    }

    async fn delete(&self, path: &str) -> Result<bool> {
        let p = self.path_for(path)?;
        match tokio::fs::remove_file(&p).await {
            Ok(_) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e).with_context(|| format!("VFS delete failed: {}", p.display())),
        }
    }

    async fn snapshot(&self) -> Result<Box<dyn VfsHandle>> {
        let new_dir = create_unique_dir(&self.root, "snapshot", &GLOBAL_FS_ID).await?;
        copy_dir_recursive(&self.dir, &new_dir).await?;
        Ok(Box::new(FsVfs {
            root: self.root.clone(),
            dir: new_dir,
        }))
    }

    fn backend(&self) -> &'static str {
        "fs"
    }
}

static GLOBAL_FS_ID: AtomicU64 = AtomicU64::new(0);

async fn create_unique_dir(root: &Path, prefix: &str, counter: &AtomicU64) -> Result<PathBuf> {
    tokio::fs::create_dir_all(root)
        .await
        .with_context(|| format!("create VFS root: {}", root.display()))?;
    loop {
        let id = counter.fetch_add(1, Ordering::Relaxed);
        let dir = root.join(format!("{prefix}-{id}"));
        match tokio::fs::create_dir(&dir).await {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(e).with_context(|| format!("create VFS namespace: {}", dir.display()));
            }
        }
    }
}

async fn collect_paths(
    base: &PathBuf,
    dir: &PathBuf,
    prefix: &str,
    out: &mut Vec<String>,
) -> Result<()> {
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e).context("list dir"),
    };
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.is_dir() {
            Box::pin(collect_paths(base, &path, prefix, out)).await?;
        } else if let Ok(rel) = path.strip_prefix(base) {
            let s = rel.to_string_lossy().into_owned();
            let normalized = format!("/{}", s);
            if normalized.starts_with(prefix) || prefix == "/" {
                out.push(normalized);
            }
        }
    }
    Ok(())
}

async fn copy_dir_recursive(src: &PathBuf, dst: &PathBuf) -> Result<()> {
    tokio::fs::create_dir_all(dst).await.ok();
    let mut entries = match tokio::fs::read_dir(src).await {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e).context("copy dir"),
    };
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if path.is_dir() {
            Box::pin(copy_dir_recursive(&path, &dst_path)).await?;
        } else {
            tokio::fs::copy(&path, &dst_path).await?;
        }
    }
    Ok(())
}

/// Factory producing isolated [`FsVfs`] handles under a common root.
#[derive(Debug, Clone)]
pub struct FsVfsFactory {
    root: PathBuf,
    next_id: Arc<AtomicU64>,
}

impl FsVfsFactory {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            next_id: Arc::new(AtomicU64::new(0)),
        }
    }

    async fn next_dir(&self) -> Result<PathBuf> {
        create_unique_dir(&self.root, "fork", &self.next_id).await
    }
}

#[async_trait]
impl VfsFactory for FsVfsFactory {
    async fn create(&self) -> Result<Box<dyn VfsHandle>> {
        let dir = self.next_dir().await?;
        Ok(Box::new(FsVfs {
            root: self.root.clone(),
            dir,
        }))
    }
    fn backend(&self) -> &'static str {
        "fs"
    }
}

// ---------------------------------------------------------------------------
// S3 / R2 / MinIO backend (optional)
// ---------------------------------------------------------------------------

#[cfg(feature = "s3")]
mod s3_backend;
#[cfg(feature = "s3")]
pub use s3_backend::{S3Vfs, S3VfsFactory};

/// Public `S3Vfs` symbol with a stable identity across feature configurations.
/// When `s3` is disabled, callers see this name but it is unconstructible.
#[cfg(not(feature = "s3"))]
pub enum S3VfsDisabled {}

#[cfg(not(feature = "s3"))]
pub type S3Vfs = S3VfsDisabled;

#[cfg(not(feature = "s3"))]
pub enum S3VfsFactoryDisabled {}

#[cfg(not(feature = "s3"))]
pub type S3VfsFactory = S3VfsFactoryDisabled;

/// CLI/programmatic configuration for any backend.
#[derive(Debug, Clone, Default)]
pub struct VfsConfig {
    pub backend: VfsBackendKind,
    pub root_dir: Option<PathBuf>,
    pub bucket: Option<String>,
    pub prefix: Option<String>,
    pub endpoint: Option<String>,
    pub region: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VfsBackendKind {
    #[default]
    Memory,
    Fs,
    S3,
}

impl VfsBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            VfsBackendKind::Memory => "memory",
            VfsBackendKind::Fs => "fs",
            VfsBackendKind::S3 => "s3",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "memory" | "mem" => Ok(VfsBackendKind::Memory),
            "fs" | "filesystem" => Ok(VfsBackendKind::Fs),
            "s3" | "r2" | "minio" => Ok(VfsBackendKind::S3),
            other => Err(anyhow!("unknown VFS backend: {other}")),
        }
    }
}

/// Builds a factory from configuration.
pub fn build_factory(config: &VfsConfig) -> Result<Box<dyn VfsFactory>> {
    match config.backend {
        VfsBackendKind::Memory => Ok(Box::new(MemoryVfsFactory)),
        VfsBackendKind::Fs => {
            let dir = config
                .root_dir
                .clone()
                .unwrap_or_else(|| std::env::temp_dir().join("altum-vfs"));
            std::fs::create_dir_all(&dir).ok();
            Ok(Box::new(FsVfsFactory::new(dir)))
        }
        VfsBackendKind::S3 => build_s3_factory(config),
    }
}

#[cfg(feature = "s3")]
fn build_s3_factory(config: &VfsConfig) -> Result<Box<dyn VfsFactory>> {
    use s3_backend::S3VfsFactoryInner;
    let bucket = config
        .bucket
        .clone()
        .ok_or_else(|| anyhow!("--vfs-bucket required for s3 backend"))?;
    let prefix = config.prefix.clone().unwrap_or_else(|| "altum".to_string());
    let endpoint = config.endpoint.clone();
    let region = config.region.clone().unwrap_or_else(|| "auto".to_string());
    let factory = S3VfsFactoryInner::new(bucket, prefix, endpoint, region)?;
    Ok(Box::new(factory))
}

#[cfg(not(feature = "s3"))]
fn build_s3_factory(_config: &VfsConfig) -> Result<Box<dyn VfsFactory>> {
    Err(anyhow!(
        "S3 VFS backend requires the `s3` Cargo feature: rebuild with --features s3"
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_backend_roundtrip<H: VfsHandle + ?Sized>(_handle: &H) -> (Vec<u8>, Vec<u8>) {
        // Helper for symmetric tests; runs no async ops itself.
        (Vec::new(), Vec::new())
    }

    #[tokio::test]
    async fn memory_roundtrip() {
        let vfs = MemoryVfs::new();
        vfs.write("/foo", b"hello".to_vec()).await.unwrap();
        let snap = vfs.snapshot().await.unwrap();

        vfs.write("/foo", b"world".to_vec()).await.unwrap();

        assert_eq!(vfs.read("/foo").await.unwrap(), Some(b"world".to_vec()));
        assert_eq!(snap.read("/foo").await.unwrap(), Some(b"hello".to_vec()));
        assert_eq!(snap.backend(), "memory");
        let _ = assert_backend_roundtrip(&*snap);
    }

    #[tokio::test]
    async fn memory_list_prefix() {
        let vfs = MemoryVfs::new();
        vfs.write("/plans/a", b"x".to_vec()).await.unwrap();
        vfs.write("/plans/b", b"y".to_vec()).await.unwrap();
        vfs.write("/other", b"z".to_vec()).await.unwrap();
        let paths = vfs.list("/plans").await.unwrap();
        assert_eq!(paths, vec!["/plans/a".to_string(), "/plans/b".to_string()]);
    }

    #[tokio::test]
    async fn memory_delete() {
        let vfs = MemoryVfs::new();
        vfs.write("/foo", b"hi".to_vec()).await.unwrap();
        assert!(vfs.delete("/foo").await.unwrap());
        assert!(!vfs.delete("/foo").await.unwrap());
        assert_eq!(vfs.read("/foo").await.unwrap(), None);
    }

    #[tokio::test]
    async fn fs_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let factory = FsVfsFactory::new(dir.path());
        let a = factory.create().await.unwrap();
        a.write("/foo/bar", b"data".to_vec()).await.unwrap();
        assert_eq!(a.read("/foo/bar").await.unwrap(), Some(b"data".to_vec()));

        let snap = a.snapshot().await.unwrap();
        a.write("/foo/bar", b"changed".to_vec()).await.unwrap();
        assert_eq!(snap.read("/foo/bar").await.unwrap(), Some(b"data".to_vec()));
        assert_eq!(a.read("/foo/bar").await.unwrap(), Some(b"changed".to_vec()));
        assert_eq!(snap.backend(), "fs");
    }

    #[tokio::test]
    async fn fs_list_keeps_logical_leading_slash() {
        let dir = tempfile::tempdir().unwrap();
        let factory = FsVfsFactory::new(dir.path());
        let vfs = factory.create().await.unwrap();
        vfs.write("/plans/a", b"a".to_vec()).await.unwrap();
        vfs.write("/other/b", b"b".to_vec()).await.unwrap();

        assert_eq!(vfs.list("/plans").await.unwrap(), vec!["/plans/a"]);
    }

    #[tokio::test]
    async fn fs_rejects_parent_components() {
        let dir = tempfile::tempdir().unwrap();
        let factory = FsVfsFactory::new(dir.path().join("root"));
        let vfs = factory.create().await.unwrap();
        let outside = dir.path().join("secret");
        std::fs::write(&outside, b"secret").unwrap();

        assert!(vfs.read("../../secret").await.is_err());
        assert!(vfs
            .write("../../secret", b"changed".to_vec())
            .await
            .is_err());
        assert!(vfs.delete("../../secret").await.is_err());
        assert_eq!(std::fs::read(outside).unwrap(), b"secret");
    }

    #[tokio::test]
    async fn fs_factory_does_not_reuse_existing_namespace() {
        let dir = tempfile::tempdir().unwrap();
        let old = dir.path().join("fork-0");
        std::fs::create_dir(&old).unwrap();
        std::fs::write(old.join("stale"), b"old").unwrap();

        let factory = FsVfsFactory::new(dir.path());
        let vfs = factory.create().await.unwrap();
        assert_eq!(vfs.read("/stale").await.unwrap(), None);
        vfs.write("/fresh", b"new".to_vec()).await.unwrap();
        assert_eq!(std::fs::read(old.join("stale")).unwrap(), b"old");
    }

    #[tokio::test]
    async fn factory_creates_isolated_forks() {
        let factory = MemoryVfsFactory;
        let a = factory.create().await.unwrap();
        let b = factory.create().await.unwrap();
        a.write("/x", b"v1".to_vec()).await.unwrap();
        assert_eq!(a.read("/x").await.unwrap(), Some(b"v1".to_vec()));
        assert_eq!(b.read("/x").await.unwrap(), None);
    }

    #[test]
    fn normalize_path_basic() {
        assert_eq!(normalize_path("foo/bar"), "/foo/bar");
        assert_eq!(normalize_path("/foo/bar"), "/foo/bar");
        assert_eq!(normalize_path(""), "/");
        assert_eq!(normalize_path("\"quoted\""), "/quoted");
    }

    #[test]
    fn manifest_upsert_replaces_existing() {
        let mut m = VfsManifest::default();
        m.upsert("/a", "hash1", 10);
        m.upsert("/a", "hash2", 12);
        assert_eq!(m.get("/a").unwrap().blob_hash, "hash2");
        assert_eq!(m.get("/a").unwrap().size, 12);
        assert_eq!(m.entries.len(), 1);
    }

    #[test]
    fn manifest_list_filters_sorted() {
        let mut m = VfsManifest::default();
        m.upsert("/plans/b", "h", 1);
        m.upsert("/plans/a", "h", 1);
        m.upsert("/other", "h", 1);
        assert_eq!(
            m.list("/plans"),
            vec!["/plans/a".to_string(), "/plans/b".to_string()]
        );
    }

    #[test]
    fn backend_kind_parse_accepts_aliases() {
        assert_eq!(
            VfsBackendKind::parse("memory").unwrap(),
            VfsBackendKind::Memory
        );
        assert_eq!(
            VfsBackendKind::parse("mem").unwrap(),
            VfsBackendKind::Memory
        );
        assert_eq!(VfsBackendKind::parse("fs").unwrap(), VfsBackendKind::Fs);
        assert_eq!(VfsBackendKind::parse("r2").unwrap(), VfsBackendKind::S3);
        assert!(VfsBackendKind::parse("foo").is_err());
    }

    #[test]
    fn content_hash_is_deterministic() {
        assert_eq!(content_hash(b"hi"), content_hash(b"hi"));
        assert_ne!(content_hash(b"hi"), content_hash(b"bye"));
    }

    #[test]
    fn build_factory_memory_default() {
        let cfg = VfsConfig::default();
        let f = build_factory(&cfg).unwrap();
        assert_eq!(f.backend(), "memory");
    }

    #[test]
    fn build_factory_fs_uses_root() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = VfsConfig {
            backend: VfsBackendKind::Fs,
            root_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        let f = build_factory(&cfg).unwrap();
        assert_eq!(f.backend(), "fs");
    }

    #[test]
    fn build_factory_s3_errors_without_feature() {
        let cfg = VfsConfig {
            backend: VfsBackendKind::S3,
            bucket: Some("test".to_string()),
            ..Default::default()
        };
        match build_factory(&cfg) {
            Ok(_) => {
                // Feature is enabled; nothing to assert about errors.
            }
            Err(e) => {
                let msg = e.to_string().to_ascii_lowercase();
                assert!(msg.contains("s3"), "unexpected error: {msg}");
            }
        }
    }
}
