//! S3-compatible content-addressed VFS backend.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use async_trait::async_trait;
use s3::creds::Credentials;
use s3::{Bucket, Region};
use tokio::sync::Mutex;

use super::{content_hash, normalize_path, normalize_prefix, VfsFactory, VfsHandle, VfsManifest};

static NAMESPACE_ID: AtomicU64 = AtomicU64::new(0);

fn new_namespace() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let id = NAMESPACE_ID.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{}-{id}", std::process::id())
}

fn clean_prefix(prefix: &str) -> String {
    prefix.trim_matches('/').to_string()
}

/// One isolated S3-backed VFS namespace.
#[derive(Clone)]
pub struct S3Vfs {
    bucket: Arc<Bucket>,
    prefix: String,
    namespace: String,
    manifest: Arc<Mutex<VfsManifest>>,
}

impl S3Vfs {
    fn blob_key(&self, hash: &str) -> String {
        format!("{}/blobs/{hash}", self.prefix)
    }

    fn manifest_key(&self) -> String {
        format!(
            "{}/namespaces/{}/manifests/current.json",
            self.prefix, self.namespace
        )
    }

    async fn put_manifest(&self, manifest: &VfsManifest) -> Result<()> {
        let body = serde_json::to_vec(manifest).context("serialize S3 VFS manifest")?;
        self.bucket
            .put_object(self.manifest_key(), &body)
            .await
            .context("write S3 VFS manifest")?;
        Ok(())
    }
}

#[async_trait]
impl VfsHandle for S3Vfs {
    async fn read(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let path = normalize_path(path);
        let entry = self.manifest.lock().await.get(&path).cloned();
        let Some(entry) = entry else {
            return Ok(None);
        };
        let response = self
            .bucket
            .get_object(self.blob_key(&entry.blob_hash))
            .await
            .with_context(|| format!("read S3 VFS blob for {path}"))?;
        Ok(Some(response.to_vec()))
    }

    async fn write(&self, path: &str, data: Vec<u8>) -> Result<()> {
        let path = normalize_path(path);
        let hash = content_hash(&data);
        self.bucket
            .put_object(self.blob_key(&hash), &data)
            .await
            .with_context(|| format!("write S3 VFS blob for {path}"))?;

        let mut manifest = self.manifest.lock().await;
        let mut next = manifest.clone();
        next.upsert(&path, &hash, data.len() as u64);
        self.put_manifest(&next).await?;
        *manifest = next;
        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let prefix = normalize_prefix(prefix);
        Ok(self.manifest.lock().await.list(&prefix))
    }

    async fn delete(&self, path: &str) -> Result<bool> {
        let path = normalize_path(path);
        let mut manifest = self.manifest.lock().await;
        let mut next = manifest.clone();
        if !next.remove(&path) {
            return Ok(false);
        }
        self.put_manifest(&next).await?;
        *manifest = next;
        Ok(true)
    }

    async fn snapshot(&self) -> Result<Box<dyn VfsHandle>> {
        let mut manifest = self.manifest.lock().await.clone();
        manifest.parent = Some(self.namespace.clone());
        let snapshot = S3Vfs {
            bucket: Arc::clone(&self.bucket),
            prefix: self.prefix.clone(),
            namespace: new_namespace(),
            manifest: Arc::new(Mutex::new(manifest.clone())),
        };
        snapshot.put_manifest(&manifest).await?;
        Ok(Box::new(snapshot))
    }

    fn backend(&self) -> &'static str {
        "s3"
    }
}

/// Factory for fresh S3-backed VFS namespaces.
#[derive(Clone)]
pub struct S3VfsFactoryInner {
    bucket: Arc<Bucket>,
    prefix: String,
}

impl S3VfsFactoryInner {
    pub fn new(
        bucket: String,
        prefix: String,
        endpoint: Option<String>,
        region: String,
    ) -> Result<Self> {
        let use_path_style = endpoint.is_some();
        let region = match endpoint {
            Some(endpoint) => Region::Custom { region, endpoint },
            None => region.parse().context("invalid S3 region")?,
        };
        let credentials = Credentials::default().context("load S3 credentials")?;
        let mut bucket =
            Bucket::new(&bucket, region, credentials).context("configure S3 bucket")?;
        if use_path_style {
            bucket = bucket.with_path_style();
        }
        Ok(Self {
            bucket: Arc::new(*bucket),
            prefix: clean_prefix(&prefix),
        })
    }
}

#[async_trait]
impl VfsFactory for S3VfsFactoryInner {
    async fn create(&self) -> Result<Box<dyn VfsHandle>> {
        let vfs = S3Vfs {
            bucket: Arc::clone(&self.bucket),
            prefix: self.prefix.clone(),
            namespace: new_namespace(),
            manifest: Arc::new(Mutex::new(VfsManifest::default())),
        };
        vfs.put_manifest(&VfsManifest::default()).await?;
        Ok(Box::new(vfs))
    }

    fn backend(&self) -> &'static str {
        "s3"
    }
}

pub type S3VfsFactory = S3VfsFactoryInner;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaces_are_unique() {
        assert_ne!(new_namespace(), new_namespace());
    }

    #[test]
    fn prefix_is_canonical() {
        assert_eq!(clean_prefix("/altum/workspaces/"), "altum/workspaces");
    }
}
