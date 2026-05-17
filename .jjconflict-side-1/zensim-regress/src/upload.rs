//! Resource uploading with shell-based backends.
//!
//! Mirrors [`fetch`](crate::fetch) — trait + shell implementation using
//! rclone, AWS CLI, or s3cmd (whichever is available).

use std::path::Path;
use std::process::Command;

use crate::error::RegressError;

/// Trait for uploading a local file to a remote destination.
pub trait ResourceUploader: Send + Sync {
    /// Upload `local` to `remote_url`. Returns `Ok(())` on success.
    fn upload(&self, local: &Path, remote_url: &str) -> Result<(), RegressError>;
}

/// Uploads using shell tools: rclone → aws cli → s3cmd (first available).
pub struct ShellUploader {
    /// Maximum time in seconds for the upload. 0 = no limit.
    pub timeout_secs: u32,
}

impl ShellUploader {
    /// Create a new uploader with default 300-second timeout.
    pub fn new() -> Self {
        Self { timeout_secs: 300 }
    }

    /// Set the upload timeout in seconds (0 = no limit).
    pub fn with_timeout(mut self, secs: u32) -> Self {
        self.timeout_secs = secs;
        self
    }

    fn try_rclone(&self, local: &Path, remote: &str) -> Result<(), String> {
        let mut cmd = Command::new("rclone");
        cmd.arg("copyto").arg(local).arg(remote);
        if self.timeout_secs > 0 {
            cmd.arg("--timeout").arg(format!("{}s", self.timeout_secs));
        }
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());

        let output = cmd.output().map_err(|e| format!("rclone not found: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("rclone exit {}: {}", output.status, stderr.trim()))
        }
    }

    fn try_aws_cli(&self, local: &Path, remote: &str) -> Result<(), String> {
        let mut cmd = Command::new("aws");
        cmd.args(["s3", "cp"]).arg(local).arg(remote);
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());

        let output = cmd
            .output()
            .map_err(|e| format!("aws cli not found: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("aws exit {}: {}", output.status, stderr.trim()))
        }
    }

    fn try_s3cmd(&self, local: &Path, remote: &str) -> Result<(), String> {
        let mut cmd = Command::new("s3cmd");
        cmd.arg("put").arg(local).arg(remote);
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());

        let output = cmd.output().map_err(|e| format!("s3cmd not found: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("s3cmd exit {}: {}", output.status, stderr.trim()))
        }
    }
}

impl Default for ShellUploader {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceUploader for ShellUploader {
    fn upload(&self, local: &Path, remote_url: &str) -> Result<(), RegressError> {
        let result = self
            .try_rclone(local, remote_url)
            .or_else(|_| self.try_aws_cli(local, remote_url))
            .or_else(|_| self.try_s3cmd(local, remote_url));

        match result {
            Ok(()) => Ok(()),
            Err(msg) => Err(RegressError::Upload {
                dest: remote_url.to_string(),
                message: msg,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_uploader_default_timeout() {
        let u = ShellUploader::new();
        assert_eq!(u.timeout_secs, 300);
    }

    #[test]
    fn shell_uploader_custom_timeout() {
        let u = ShellUploader::new().with_timeout(60);
        assert_eq!(u.timeout_secs, 60);
    }

    /// A mock uploader that records what was called.
    struct MockUploader {
        should_fail: bool,
    }

    impl ResourceUploader for MockUploader {
        fn upload(&self, _local: &Path, remote_url: &str) -> Result<(), RegressError> {
            if self.should_fail {
                Err(RegressError::Upload {
                    dest: remote_url.to_string(),
                    message: "mock failure".to_string(),
                })
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn mock_uploader_success() {
        let uploader = MockUploader { should_fail: false };
        let result = uploader.upload(Path::new("/tmp/test.png"), "s3://bucket/test.png");
        assert!(result.is_ok());
    }

    #[test]
    fn mock_uploader_failure() {
        let uploader = MockUploader { should_fail: true };
        let result = uploader.upload(Path::new("/tmp/test.png"), "s3://bucket/test.png");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("mock failure"));
    }
}
