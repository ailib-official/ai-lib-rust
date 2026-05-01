//! Credential resolution for protocol-backed transports.
//!
//! 凭证解析模块：按显式覆盖、manifest 环境变量、兼容环境变量、系统 keyring 的顺序解析。

use crate::protocol::{AuthConfig, ProtocolManifest};
#[cfg(feature = "keyring")]
use keyring::Entry;
use std::env;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialSourceKind {
    Explicit,
    ManifestEnv,
    ConventionalEnv,
    Keyring,
    None,
}

#[derive(Clone)]
pub struct ResolvedCredential {
    secret: Option<String>,
    pub source_kind: CredentialSourceKind,
    pub source_name: Option<String>,
    pub required_envs: Vec<String>,
    pub conventional_envs: Vec<String>,
}

impl fmt::Debug for ResolvedCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResolvedCredential")
            .field("secret", &self.secret.as_ref().map(|_| "<redacted>"))
            .field("source_kind", &self.source_kind)
            .field("source_name", &self.source_name)
            .field("required_envs", &self.required_envs)
            .field("conventional_envs", &self.conventional_envs)
            .finish()
    }
}

impl ResolvedCredential {
    pub fn secret(&self) -> Option<&str> {
        self.secret.as_deref()
    }

    pub fn missing(required_envs: Vec<String>, conventional_envs: Vec<String>) -> Self {
        Self {
            secret: None,
            source_kind: CredentialSourceKind::None,
            source_name: None,
            required_envs,
            conventional_envs,
        }
    }
}

/// Returns the active `AuthConfig` for credential resolution.
///
/// V2 manifests declare `endpoint.auth`; V1 manifests use top-level `auth`.
/// When both are present the endpoint-level config wins (single source of truth)
/// and the top-level entry is treated purely as a V1 compatibility fallback.
pub fn primary_auth(manifest: &ProtocolManifest) -> Option<&AuthConfig> {
    manifest.endpoint.auth.as_ref().or(manifest.auth.as_ref())
}

/// Returns the manifest-declared environment variable names that
/// `resolve_credential` should probe, in declaration order.
///
/// Only the winning [`primary_auth`] entry is scanned, so the list of
/// candidate env vars is always consistent with the auth attachment shape
/// applied by the transport. This avoids the V1/V2 ambiguity where two
/// auth blocks could otherwise contribute env names that no longer match
/// the active auth type.
pub fn required_envs(manifest: &ProtocolManifest) -> Vec<String> {
    let Some(auth) = primary_auth(manifest) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(env) = auth.token_env.as_ref().or(auth.key_env.as_ref()) {
        let trimmed = env.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
    }
    out
}

pub fn provider_id(manifest: &ProtocolManifest) -> &str {
    manifest.provider_id.as_deref().unwrap_or(&manifest.id)
}

/// Returns the conventional `${PROVIDER_ID}_API_KEY` fallback env var name.
///
/// The provider id is uppercased and `-` is normalized to `_`, matching the
/// industry convention used by virtually every provider's official docs
/// (e.g. `OPENAI_API_KEY`, `DEEP_SEEK_API_KEY`). A single canonical name is
/// returned; if a deployment uses a non-conventional alias they should declare
/// it via `auth.token_env` / `auth.key_env` in the manifest instead.
pub fn conventional_envs(provider_id: &str) -> Vec<String> {
    let normalized = provider_id.to_uppercase().replace('-', "_");
    vec![format!("{normalized}_API_KEY")]
}

pub fn resolve_credential(
    manifest: &ProtocolManifest,
    explicit: Option<&str>,
) -> ResolvedCredential {
    let required_envs = required_envs(manifest);
    let conventional_envs = conventional_envs(provider_id(manifest));

    if let Some(value) = explicit.map(str::trim).filter(|value| !value.is_empty()) {
        return ResolvedCredential {
            secret: Some(value.to_string()),
            source_kind: CredentialSourceKind::Explicit,
            source_name: Some("explicit".to_string()),
            required_envs,
            conventional_envs,
        };
    }

    for name in &required_envs {
        if let Some(value) = env_value(name) {
            return ResolvedCredential {
                secret: Some(value),
                source_kind: CredentialSourceKind::ManifestEnv,
                source_name: Some(name.clone()),
                required_envs,
                conventional_envs,
            };
        }
    }

    for name in &conventional_envs {
        if let Some(value) = env_value(name) {
            return ResolvedCredential {
                secret: Some(value),
                source_kind: CredentialSourceKind::ConventionalEnv,
                source_name: Some(name.clone()),
                required_envs,
                conventional_envs,
            };
        }
    }

    #[cfg(feature = "keyring")]
    {
        let id = provider_id(manifest);
        if let Some(value) = keyring_value(id) {
            return ResolvedCredential {
                secret: Some(value),
                source_kind: CredentialSourceKind::Keyring,
                source_name: Some(format!("ai-protocol/{id}")),
                required_envs,
                conventional_envs,
            };
        }
    }

    ResolvedCredential::missing(required_envs, conventional_envs)
}

fn env_value(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(feature = "keyring")]
fn keyring_value(provider_id: &str) -> Option<String> {
    Entry::new("ai-protocol", provider_id)
        .ok()
        .and_then(|entry| entry.get_password().ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        key: &'static str,
        old: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let old = env::var(key).ok();
            match value {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
            Self { key, old }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.old.as_ref() {
                Some(value) => env::set_var(self.key, value),
                None => env::remove_var(self.key),
            }
        }
    }

    fn manifest() -> ProtocolManifest {
        serde_yaml::from_str(
            r#"
id: replicate
protocol_version: "1.5"
name: "Replicate"
status: "stable"
category: "ai_provider"
official_url: "https://example.com"
support_contact: "https://example.com/support"
endpoint:
  base_url: "https://api.example.com/v1"
auth:
  type: "bearer"
  token_env: "REPLICATE_API_TOKEN"
capabilities: [chat]
"#,
        )
        .expect("manifest")
    }

    #[test]
    fn explicit_credential_wins() {
        let _lock = ENV_LOCK.lock().expect("env lock");
        let _token = EnvGuard::set("REPLICATE_API_TOKEN", Some("env-token"));
        let _key = EnvGuard::set("REPLICATE_API_KEY", Some("env-key"));

        let resolved = resolve_credential(&manifest(), Some(" explicit "));

        assert_eq!(resolved.secret(), Some("explicit"));
        assert_eq!(resolved.source_kind, CredentialSourceKind::Explicit);
        assert_eq!(resolved.source_name.as_deref(), Some("explicit"));
    }

    #[test]
    fn manifest_env_wins_over_conventional_env() {
        let _lock = ENV_LOCK.lock().expect("env lock");
        let _token = EnvGuard::set("REPLICATE_API_TOKEN", Some("manifest-token"));
        let _key = EnvGuard::set("REPLICATE_API_KEY", Some("conventional-key"));

        let resolved = resolve_credential(&manifest(), None);

        assert_eq!(resolved.secret(), Some("manifest-token"));
        assert_eq!(resolved.source_kind, CredentialSourceKind::ManifestEnv);
        assert_eq!(resolved.source_name.as_deref(), Some("REPLICATE_API_TOKEN"));
    }

    #[test]
    fn conventional_env_is_fallback() {
        let _lock = ENV_LOCK.lock().expect("env lock");
        let _token = EnvGuard::set("REPLICATE_API_TOKEN", None);
        let _key = EnvGuard::set("REPLICATE_API_KEY", Some("conventional-key"));

        let resolved = resolve_credential(&manifest(), None);

        assert_eq!(resolved.secret(), Some("conventional-key"));
        assert_eq!(resolved.source_kind, CredentialSourceKind::ConventionalEnv);
        assert_eq!(resolved.source_name.as_deref(), Some("REPLICATE_API_KEY"));
    }

    #[test]
    fn debug_redacts_secret() {
        let _lock = ENV_LOCK.lock().expect("env lock");
        let _token = EnvGuard::set("REPLICATE_API_TOKEN", Some("manifest-token"));

        let resolved = resolve_credential(&manifest(), None);
        let debug = format!("{resolved:?}");

        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("manifest-token"));
    }
}
