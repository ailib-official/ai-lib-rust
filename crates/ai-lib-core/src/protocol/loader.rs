//! Protocol loader with support for local files, embedded assets, and remote URLs
//! Heartbeat sync - 2026-01-06
//! Includes hot-reload capability using ArcSwap

use crate::protocol::{ProtocolError, ProtocolManifest};
use arc_swap::ArcSwap;
use lru::LruCache;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Protocol loader that supports multiple sources
pub struct ProtocolLoader {
    base_path: Option<PathBuf>,
    hot_reload: bool,
    validator: crate::protocol::validator::ProtocolValidator,
    cache: Mutex<LruCache<String, Arc<ProtocolManifest>>>,
}

impl ProtocolLoader {
    pub fn new() -> Self {
        Self {
            base_path: None,
            hot_reload: false,
            validator: crate::protocol::validator::ProtocolValidator::default(),
            // Use 100 as default cache size
            // NonZeroUsize::new(100) is guaranteed to be Some, but use expect for clarity
            cache: Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(100)
                    .expect("Cache size must be non-zero (this should never happen)"),
            )),
        }
    }

    /// Set base path for protocol files
    pub fn with_base_path(mut self, path: impl AsRef<Path>) -> Self {
        self.base_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Enable hot reload
    pub fn with_hot_reload(mut self, enable: bool) -> Self {
        self.hot_reload = enable;
        self
    }

    /// Load a model configuration
    /// Model identifier format: "provider/model-name"
    pub async fn load_model(&self, model: &str) -> Result<ProtocolManifest, ProtocolError> {
        // 1. Check Cache
        {
            let mut cache = self.cache.lock().map_err(|e| {
                ProtocolError::Internal(format!(
                    "Failed to acquire cache lock while loading model '{}': {}",
                    model, e
                ))
            })?;
            if let Some(manifest) = cache.get(model) {
                return Ok(manifest.as_ref().clone());
            }
        }

        // Allow "provider/model" or "provider/org/model-name" (e.g. nvidia/minimaxai/minimax-m2)
        let parts: Vec<&str> = model.split('/').collect();
        if parts.len() < 2 {
            return Err(ProtocolError::NotFound {
                id: model.to_string(),
                hint: Some("Ensure the model name follows the 'provider/model' format".to_string()),
            });
        }

        let provider = parts[0];
        let model_name = parts[1..].join("/");

        // First, try to load model registry to get provider reference.
        // Prefer the full logical id as registry key (e.g. nvidia/nemotron-mini-…),
        // then the stripped remainder (e.g. meta/llama-…). If neither is registered
        // (common for some providers), fall back to the provider segment.
        let manifest = match self.load_model_config(model).await {
            Ok(model_config) => self.load_provider(&model_config.provider).await?,
            Err(ProtocolError::NotFound { .. }) => match self.load_model_config(&model_name).await
            {
                Ok(model_config) => self.load_provider(&model_config.provider).await?,
                Err(ProtocolError::NotFound { .. }) => self.load_provider(provider).await?,
                Err(e) => return Err(e),
            },
            Err(e) => return Err(e),
        };

        // 2. Update Cache
        {
            let mut cache = self.cache.lock().map_err(|e| {
                ProtocolError::Internal(format!(
                    "Failed to acquire cache lock while caching model '{}': {}",
                    model, e
                ))
            })?;
            cache.put(model.to_string(), Arc::new(manifest.clone()));
        }

        Ok(manifest)
    }

    /// Resolve the OpenAI-compatible wire `model` field for a logical `provider/…` id.
    ///
    /// Lookup order (ALR-NIM-001):
    /// 1. v1 model registry by full logical id, then by `parts[1..]` — prefer `model_id`, else key
    /// 2. Fallback [`Self::wire_model_id_fallback`] (NIM nvidia-org keeps `nvidia/<name>`)
    pub async fn resolve_wire_model_id(&self, logical: &str) -> String {
        let trimmed = logical.trim();
        let parts: Vec<&str> = trimmed.split('/').filter(|p| !p.is_empty()).collect();
        let stripped = if parts.len() >= 2 {
            parts[1..].join("/")
        } else {
            trimmed.to_string()
        };

        for key in [trimmed, stripped.as_str()] {
            if key.is_empty() {
                continue;
            }
            if let Ok(cfg) = self.load_model_config(key).await {
                if let Some(id) = cfg.model_id.filter(|s| !s.trim().is_empty()) {
                    return id;
                }
                return key.to_string();
            }
        }

        Self::wire_model_id_fallback(trimmed)
    }

    /// Sync fallback when the v1 model registry has no entry (or protocol dir unset).
    ///
    /// For `nvidia/<single-segment>` NIM catalog ids, the wire body must keep the
    /// `nvidia/` prefix (bare id → HTTP 404 page not found). Org-qualified ids
    /// (`nvidia/meta/…`) still strip to `meta/…`.
    #[must_use]
    pub fn wire_model_id_fallback(logical: &str) -> String {
        let trimmed = logical.trim();
        let parts: Vec<&str> = trimmed.split('/').filter(|p| !p.is_empty()).collect();
        if parts.len() < 2 {
            return trimmed.to_string();
        }
        if parts.len() == 2 && parts[0].eq_ignore_ascii_case("nvidia") {
            return format!("{}/{}", parts[0], parts[1]);
        }
        parts[1..].join("/")
    }

    /// Load provider configuration.
    ///
    /// Resolution (PT-ARCH-005 / 005d / ALR-ID-001):
    /// 1. Exact match on `provider_id` as file stem / primary `id`
    /// 2. If missing: resolve via published `dist/provider-identity.json`
    ///    (`families[]`, with legacy single-family fallback) and retry
    /// 3. Else fail closed (`NotFound`)
    pub async fn load_provider(
        &self,
        provider_id: &str,
    ) -> Result<ProtocolManifest, ProtocolError> {
        match self.load_provider_exact(provider_id).await {
            Ok(manifest) => return Ok(manifest),
            // Only alias-resolve on missing files; do not mask ValidationError / LoadError.
            Err(ProtocolError::NotFound { .. }) => {}
            Err(other) => return Err(other),
        }

        if let Some(canonical) = self.resolve_canonical_provider_id(provider_id).await {
            if canonical != provider_id {
                return self.load_provider_exact(&canonical).await;
            }
        }

        Err(ProtocolError::NotFound {
            id: provider_id.to_string(),
            hint: Some(format!(
                "Check if the provider file '{}.json' or '{}.yaml' exists, or that '{}' is listed as an alias in dist/provider-identity.json / manifest.aliases",
                provider_id, provider_id, provider_id
            )),
        })
    }

    async fn load_provider_exact(
        &self,
        provider_id: &str,
    ) -> Result<ProtocolManifest, ProtocolError> {
        // Try multiple sources in order:
        // 1. Local file system (dist JSON) - PREFERRED
        // 2. Local file system (source YAML) - FALLBACK
        // 3. GitHub URL (if AI_PROTOCOL_DIR is a URL)
        // 4. Embedded assets (future)

        // Path prioritization helper
        let mut search_locations: Vec<(PathBuf, bool)> = Vec::new(); // (path_base, is_json_preferred)

        // 1. Check user-configured base_path
        if let Some(ref base_path) = self.base_path {
            // Priority 0: dist/v2/providers/{id}.json
            search_locations.push((base_path.join("dist").join("v2").join("providers"), true));
            // Priority 0b: v2/providers/{id}.yaml
            search_locations.push((base_path.join("v2").join("providers"), false));
            // Priority 1: dist/v1/providers/{id}.json
            search_locations.push((base_path.join("dist").join("v1").join("providers"), true));
            // Priority 2: v1/providers/{id}.yaml
            search_locations.push((base_path.join("v1").join("providers"), false));
        }

        // 2. Check AI_PROTOCOL_DIR Env Var
        if let Ok(root) =
            std::env::var("AI_PROTOCOL_DIR").or_else(|_| std::env::var("AI_PROTOCOL_PATH"))
        {
            if root.starts_with("http://") || root.starts_with("https://") {
                // Handling URL sources (Remote)
                // Try JSON first if it looks like a raw github url, but typically raw github urls are specific.
                // For simplicity, we stick to the existing logic for URLs but could enhance later to try .json
                let url = if root.ends_with('/') {
                    format!("{}dist/v1/providers/{}.json", root, provider_id)
                } else {
                    format!("{}/dist/v1/providers/{}.json", root, provider_id)
                };

                // Try JSON from remote
                if let Ok(manifest) = self.load_from_json_url(&url).await {
                    return Ok(manifest);
                }

                // Fallback to YAML from remote
                let url_yaml = if root.ends_with('/') {
                    format!("{}v1/providers/{}.yaml", root, provider_id)
                } else {
                    format!("{}/v1/providers/{}.yaml", root, provider_id)
                };
                return self.load_from_url(&url_yaml).await;
            } else {
                // Local Path from Env
                let root = PathBuf::from(root);
                search_locations.push((root.join("dist").join("v2").join("providers"), true));
                search_locations.push((root.join("v2").join("providers"), false));
                search_locations.push((root.join("dist").join("v1").join("providers"), true));
                search_locations.push((root.join("v1").join("providers"), false));
            }
        }

        // 3. Default dev locations
        let default_roots = vec![
            PathBuf::from("ai-protocol"),
            PathBuf::from("../ai-protocol"),
            PathBuf::from("../../ai-protocol"),
            PathBuf::from("D:\\ai-protocol"),
        ];

        for root in default_roots {
            search_locations.push((root.join("dist").join("v2").join("providers"), true));
            search_locations.push((root.join("v2").join("providers"), false));
            search_locations.push((root.join("dist").join("v1").join("providers"), true));
            search_locations.push((root.join("v1").join("providers"), false));
        }

        // Execute Search
        for (base, prefer_json) in search_locations {
            if prefer_json {
                let json_path = base.join(format!("{}.json", provider_id));
                if json_path.exists() {
                    return self.load_from_json_file(&json_path).await;
                }
            } else {
                let yaml_path = base.join(format!("{}.yaml", provider_id));
                if yaml_path.exists() {
                    return self.load_from_file(&yaml_path).await;
                }
            }
        }

        // Last resort: try GitHub raw URL (canonical source) - JSON (v2 first)
        let github_json_v2 = format!(
            "https://raw.githubusercontent.com/ailib-official/ai-protocol/main/dist/v2/providers/{}.json",
            provider_id
        );
        if let Ok(manifest) = self.load_from_json_url(&github_json_v2).await {
            return Ok(manifest);
        }

        // Last resort fallback: v1 JSON
        let github_json = format!(
            "https://raw.githubusercontent.com/ailib-official/ai-protocol/main/dist/v1/providers/{}.json",
            provider_id
        );
        if let Ok(manifest) = self.load_from_json_url(&github_json).await {
            return Ok(manifest);
        }

        // Last resort fallback: YAML
        let github_yaml = format!(
            "https://raw.githubusercontent.com/ailib-official/ai-protocol/main/v1/providers/{}.yaml",
            provider_id
        );
        if let Ok(manifest) = self.load_from_url(&github_yaml).await {
            return Ok(manifest);
        }

        Err(ProtocolError::NotFound {
            id: provider_id.to_string(),
            hint: Some(format!(
                "Check if the provider file '{}.json' or '{}.yaml' exists in your protocol directory",
                provider_id, provider_id
            )),
        })
    }

    /// Resolve an alias key to canonical provider id using published identity map
    /// (`dist/provider-identity.json`, PT-ARCH-005c).
    async fn resolve_canonical_provider_id(&self, key: &str) -> Option<String> {
        for map_path in self.identity_map_candidates() {
            if let Ok(raw) = std::fs::read_to_string(&map_path) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
                    if let Some(canonical) = canonical_from_identity_value(&value, key) {
                        return Some(canonical);
                    }
                }
            }
        }

        // Remote package surface (same pin as provider JSON last-resort).
        let url = "https://raw.githubusercontent.com/ailib-official/ai-protocol/main/dist/provider-identity.json";
        if let Ok(resp) = reqwest::get(url).await {
            if resp.status().is_success() {
                if let Ok(value) = resp.json::<serde_json::Value>().await {
                    if let Some(canonical) = canonical_from_identity_value(&value, key) {
                        return Some(canonical);
                    }
                }
            }
        }

        None
    }

    fn identity_map_candidates(&self) -> Vec<PathBuf> {
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Some(ref base) = self.base_path {
            roots.push(base.clone());
        }
        if let Ok(root) =
            std::env::var("AI_PROTOCOL_DIR").or_else(|_| std::env::var("AI_PROTOCOL_PATH"))
        {
            if !(root.starts_with("http://") || root.starts_with("https://")) {
                roots.push(PathBuf::from(root));
            }
        }
        roots.extend([
            PathBuf::from("ai-protocol"),
            PathBuf::from("../ai-protocol"),
            PathBuf::from("../../ai-protocol"),
            PathBuf::from("D:\\ai-protocol"),
        ]);

        let mut out = Vec::new();
        for root in roots {
            out.push(root.join("dist").join("provider-identity.json"));
            out.push(root.join("v2").join("provider-identity.fixture.json"));
        }
        out
    }
}

fn canonical_from_identity_value(value: &serde_json::Value, key: &str) -> Option<String> {
    // PT-ARCH-005d: multi-family map.
    if let Some(families) = value.get("families").and_then(|f| f.as_array()) {
        for family in families {
            if let Some(canonical) = canonical_from_family(family, key) {
                return Some(canonical);
            }
        }
        return None;
    }

    // Legacy single-family document (pre-005d).
    canonical_from_family(value, key)
}

fn canonical_from_family(family: &serde_json::Value, key: &str) -> Option<String> {
    let canonical = family.get("canonical_id")?.as_str()?;
    if key == canonical {
        return Some(canonical.to_string());
    }
    let aliases = family.get("aliases")?.as_array()?;
    if aliases.iter().any(|a| a.as_str() == Some(key)) {
        return Some(canonical.to_string());
    }
    None
}

impl ProtocolLoader {
    /// Load protocol from local JSON file (Fast Path)
    async fn load_from_json_file(&self, path: &Path) -> Result<ProtocolManifest, ProtocolError> {
        let content = tokio::fs::read(path)
            .await
            .map_err(|e| ProtocolError::LoadError {
                path: path.to_string_lossy().to_string(),
                reason: e.to_string(),
                hint: Some("Check file permissions.".to_string()),
            })?;

        let manifest: ProtocolManifest = serde_json::from_slice(&content)
            .map_err(|e| ProtocolError::ValidationError(format!("Invalid JSON manifest: {}", e)))?;

        // Validate against JSON Schema (Optional but recommended even for dist)
        // For max speed, we might skip this in release, but keeping for safety now.
        self.validator.validate(&manifest)?;

        Ok(manifest)
    }

    /// Load protocol from local YAML file (Legacy/Dev Path)
    async fn load_from_file(&self, path: &Path) -> Result<ProtocolManifest, ProtocolError> {
        // Read as bytes first to handle different encodings
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| ProtocolError::LoadError {
                path: path.to_string_lossy().to_string(),
                reason: e.to_string(),
                hint: Some("Check if the file exists and you have read permissions.".to_string()),
            })?;

        // ... (encoding detection remains same)
        let content = if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE {
            // UTF-16 LE with BOM
            let utf16_bytes = &bytes[2..];
            let mut utf16_chars = Vec::new();
            for i in (0..utf16_bytes.len()).step_by(2) {
                if i + 1 < utf16_bytes.len() {
                    let code_unit = u16::from_le_bytes([utf16_bytes[i], utf16_bytes[i + 1]]);
                    utf16_chars.push(code_unit);
                }
            }
            String::from_utf16(&utf16_chars).map_err(|e| ProtocolError::LoadError {
                path: path.to_string_lossy().to_string(),
                reason: format!("Invalid UTF-16: {}", e),
                hint: None,
            })?
        } else if bytes.len() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF {
            String::from_utf8(bytes[3..].to_vec()).map_err(|e| ProtocolError::LoadError {
                path: path.to_string_lossy().to_string(),
                reason: format!("Invalid UTF-8 (after BOM): {}", e),
                hint: None,
            })?
        } else {
            String::from_utf8(bytes).map_err(|e| ProtocolError::LoadError {
                path: path.to_string_lossy().to_string(),
                reason: format!("Invalid UTF-8: {}", e),
                hint: None,
            })?
        };

        let manifest: ProtocolManifest = Self::parse_manifest_yaml(&content)?;
        self.validator.validate(&manifest)?;
        Ok(manifest)
    }

    /// Load protocol from remote JSON URL
    async fn load_from_json_url(&self, url: &str) -> Result<ProtocolManifest, ProtocolError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| ProtocolError::Internal(format!("Failed to create HTTP client: {}", e)))?;

        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| ProtocolError::LoadError {
                path: url.to_string(),
                reason: format!("HTTP request failed: {}", e),
                hint: None,
            })?;

        if !response.status().is_success() {
            return Err(ProtocolError::LoadError {
                path: url.to_string(),
                reason: format!("HTTP {}", response.status()),
                hint: None,
            });
        }

        let content = response
            .bytes()
            .await
            .map_err(|e| ProtocolError::LoadError {
                path: url.to_string(),
                reason: format!("Failed to read bytes: {}", e),
                hint: None,
            })?;

        let manifest: ProtocolManifest = serde_json::from_slice(&content).map_err(|e| {
            ProtocolError::ValidationError(format!("Invalid JSON manifest from URL: {}", e))
        })?;

        self.validator.validate(&manifest)?;
        Ok(manifest)
    }

    /// Load protocol from remote URL (GitHub raw URL)
    async fn load_from_url(&self, url: &str) -> Result<ProtocolManifest, ProtocolError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| ProtocolError::Internal(format!("Failed to create HTTP client: {}", e)))?;

        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| ProtocolError::LoadError {
                path: url.to_string(),
                reason: format!("HTTP request failed: {}", e),
                hint: Some(
                    "Check your internet connection and verify the URL is accessible.".to_string(),
                ),
            })?;

        if !response.status().is_success() {
            return Err(ProtocolError::LoadError {
                path: url.to_string(),
                reason: format!(
                    "HTTP {}: {}",
                    response.status(),
                    response.text().await.unwrap_or_default()
                ),
                hint: Some(
                    "Verify the remote registry URL and your API permissions if any.".to_string(),
                ),
            });
        }

        let content = response
            .text()
            .await
            .map_err(|e| ProtocolError::LoadError {
                path: url.to_string(),
                reason: format!("Failed to read response: {}", e),
                hint: None,
            })?;

        let manifest: ProtocolManifest = Self::parse_manifest_yaml(&content)?;

        // Validate against JSON Schema
        self.validator.validate(&manifest)?;

        Ok(manifest)
    }

    /// Parse YAML into a ProtocolManifest with better error classification.
    ///
    /// Rationale:
    /// - YAML syntax/encoding issues are "load" errors.
    /// - Structural mismatches (missing required fields, wrong types) are "validation" errors.
    fn parse_manifest_yaml(content: &str) -> Result<ProtocolManifest, ProtocolError> {
        serde_yaml::from_str::<ProtocolManifest>(content).map_err(|e| {
            let msg = e.to_string();
            // Heuristic classification based on serde error messages.
            // This keeps public error categories stable without pulling in serde internals.
            let looks_structural = msg.contains("missing field")
                || msg.contains("unknown field")
                || msg.contains("invalid type")
                || msg.contains("invalid value")
                || msg.contains("expected");

            if looks_structural {
                ProtocolError::ValidationError(format!("Invalid manifest structure: {}", msg))
            } else {
                ProtocolError::YamlError(msg)
            }
        })
    }

    /// Load model configuration from registry
    async fn load_model_config(&self, model_name: &str) -> Result<ModelConfig, ProtocolError> {
        // Try to find model, scanning registries.
        // Priority: dist/v1/models/*.json -> v1/models/*.yaml

        let mut search_locations: Vec<(PathBuf, bool)> = Vec::new(); // (path_base, is_json_preferred)

        // 0. Explicit loader base_path (AiClientBuilder / tests)
        if let Some(ref root) = self.base_path {
            search_locations.push((root.join("dist").join("v1").join("models"), true));
            search_locations.push((root.join("v1").join("models"), false));
        }

        // 1. Env Var AI_PROTOCOL_DIR
        if let Ok(root) =
            std::env::var("AI_PROTOCOL_DIR").or_else(|_| std::env::var("AI_PROTOCOL_PATH"))
        {
            // If HTTP, skipped here as typically model config loading implies local or full repo clone access.
            // If we really need remote model loading, we'd need a different strategy (scanning a remote index).
            // For now, assume local model registry for this heuristic.
            if !root.starts_with("http://") && !root.starts_with("https://") {
                let root = PathBuf::from(root);
                search_locations.push((root.join("dist").join("v1").join("models"), true));
                search_locations.push((root.join("v1").join("models"), false));
            }
        }

        // 2. Default paths
        let default_roots = vec![
            PathBuf::from("ai-protocol"),
            PathBuf::from("../ai-protocol"),
            PathBuf::from("../../ai-protocol"),
            PathBuf::from("D:\\ai-protocol"),
        ];

        for root in default_roots {
            search_locations.push((root.join("dist").join("v1").join("models"), true));
            search_locations.push((root.join("v1").join("models"), false));
        }

        for (base, prefer_json) in search_locations {
            if !base.exists() {
                continue;
            }
            let mut rd = match tokio::fs::read_dir(&base).await {
                Ok(rd) => rd,
                Err(_) => continue,
            };

            while let Ok(Some(entry)) = rd.next_entry().await {
                let path = entry.path();
                let extension = path.extension().and_then(|s| s.to_str());

                let is_match = if prefer_json {
                    extension.map(|s| s.eq_ignore_ascii_case("json")) == Some(true)
                } else {
                    extension
                        .map(|s| s.eq_ignore_ascii_case("yaml") || s.eq_ignore_ascii_case("yml"))
                        == Some(true)
                };

                if !is_match {
                    continue;
                }

                if prefer_json {
                    if let Ok(config) = self.load_model_registry_json(&path).await {
                        if let Some(model) = config.models.get(model_name) {
                            return Ok(model.clone());
                        }
                    }
                } else if let Ok(config) = self.load_model_registry_yaml(&path).await {
                    if let Some(model) = config.models.get(model_name) {
                        return Ok(model.clone());
                    }
                }
            }
        }

        Err(ProtocolError::NotFound {
            id: model_name.to_string(),
            hint: Some(
                "Check if the model is registered in the manifests/v1/models/ directory"
                    .to_string(),
            ),
        })
    }

    async fn load_model_registry_json(&self, path: &Path) -> Result<ModelRegistry, ProtocolError> {
        let content = tokio::fs::read(path)
            .await
            .map_err(|e| ProtocolError::LoadError {
                path: path.to_string_lossy().to_string(),
                reason: e.to_string(),
                hint: None,
            })?;
        let registry: ModelRegistry = serde_json::from_slice(&content).map_err(|e| {
            ProtocolError::ValidationError(format!("Invalid JSON model registry: {}", e))
        })?;
        Ok(registry)
    }

    async fn load_model_registry_yaml(&self, path: &Path) -> Result<ModelRegistry, ProtocolError> {
        let content =
            tokio::fs::read_to_string(path)
                .await
                .map_err(|e| ProtocolError::LoadError {
                    path: path.to_string_lossy().to_string(),
                    reason: format!("Failed to read model registry: {}", e),
                    hint: None,
                })?;

        let registry: ModelRegistry = serde_yaml::from_str(&content).map_err(|e| {
            ProtocolError::YamlError(format!("Failed to parse model registry: {}", e))
        })?;

        Ok(registry)
    }
}

impl Default for ProtocolLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Model registry structure
#[derive(Debug, Clone, serde::Deserialize)]
struct ModelRegistry {
    models: std::collections::HashMap<String, ModelConfig>,
}

/// Model configuration from registry
#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
struct ModelConfig {
    provider: String,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    context_window: Option<u32>,
    #[serde(default)]
    capabilities: Vec<String>,
}

/// Hot-reloadable protocol registry
pub struct ProtocolRegistry {
    manifests: ArcSwap<std::collections::HashMap<String, Arc<ProtocolManifest>>>,
    loader: ProtocolLoader,
}

impl ProtocolRegistry {
    pub fn new() -> Self {
        Self {
            manifests: ArcSwap::from_pointee(std::collections::HashMap::new()),
            loader: ProtocolLoader::new(),
        }
    }

    /// Get or load a protocol manifest
    pub async fn get_manifest(
        &self,
        provider_id: &str,
    ) -> Result<Arc<ProtocolManifest>, ProtocolError> {
        // Check cache first
        let current = self.manifests.load();
        if let Some(manifest) = current.get(provider_id) {
            return Ok(Arc::clone(manifest));
        }

        // Load and cache
        let manifest = self.loader.load_provider(provider_id).await?;
        let manifest_arc = Arc::new(manifest);

        // Update cache atomically
        let mut updated_map = std::collections::HashMap::new();
        for (k, v) in current.iter() {
            updated_map.insert(k.clone(), v.clone());
        }
        updated_map.insert(provider_id.to_string(), manifest_arc.clone());
        self.manifests.store(Arc::new(updated_map));

        Ok(manifest_arc)
    }
}

impl Default for ProtocolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod identity_alias_tests {
    use super::*;

    #[test]
    fn wire_model_id_fallback_keeps_nvidia_org_prefix() {
        assert_eq!(
            ProtocolLoader::wire_model_id_fallback("nvidia/nemotron-mini-4b-instruct"),
            "nvidia/nemotron-mini-4b-instruct"
        );
    }

    #[test]
    fn wire_model_id_fallback_strips_org_qualified_nvidia() {
        assert_eq!(
            ProtocolLoader::wire_model_id_fallback("nvidia/meta/llama-3.1-8b-instruct"),
            "meta/llama-3.1-8b-instruct"
        );
    }

    #[test]
    fn wire_model_id_fallback_strips_openai_style() {
        assert_eq!(
            ProtocolLoader::wire_model_id_fallback("openai/gpt-4o-mini"),
            "gpt-4o-mini"
        );
    }

    #[tokio::test]
    async fn resolve_wire_model_id_uses_registry_when_present() {
        let protocol_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ai-protocol");
        let nvidia_models = protocol_root.join("dist/v1/models/nvidia.json");
        if !nvidia_models.exists() {
            return;
        }
        let loader = ProtocolLoader::new().with_base_path(&protocol_root);
        let wire = loader
            .resolve_wire_model_id("nvidia/nemotron-mini-4b-instruct")
            .await;
        assert_eq!(wire, "nvidia/nemotron-mini-4b-instruct");
        let wire_meta = loader
            .resolve_wire_model_id("nvidia/meta/llama-3.1-8b-instruct")
            .await;
        assert_eq!(wire_meta, "meta/llama-3.1-8b-instruct");
    }

    #[test]
    fn canonical_from_identity_map_resolves_google_to_gemini() {
        let value = serde_json::json!({
            "canonical_id": "gemini",
            "aliases": ["google"]
        });
        assert_eq!(
            canonical_from_identity_value(&value, "google").as_deref(),
            Some("gemini")
        );
        assert_eq!(
            canonical_from_identity_value(&value, "gemini").as_deref(),
            Some("gemini")
        );
        assert_eq!(canonical_from_identity_value(&value, "openai"), None);
    }

    #[test]
    fn canonical_from_multi_family_map_resolves_kimi_and_glm() {
        let value = serde_json::json!({
            "families": [
                { "canonical_id": "gemini", "aliases": ["google"] },
                { "canonical_id": "moonshot", "aliases": ["kimi"] },
                { "canonical_id": "zhipu", "aliases": ["glm"] }
            ]
        });
        assert_eq!(
            canonical_from_identity_value(&value, "kimi").as_deref(),
            Some("moonshot")
        );
        assert_eq!(
            canonical_from_identity_value(&value, "glm").as_deref(),
            Some("zhipu")
        );
        assert_eq!(
            canonical_from_identity_value(&value, "google").as_deref(),
            Some("gemini")
        );
        assert_eq!(canonical_from_identity_value(&value, "openai"), None);
    }

    #[tokio::test]
    async fn load_provider_resolves_google_alias_to_gemini_manifest() {
        let protocol_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ai-protocol");
        if !protocol_root.join("dist/v2/providers/gemini.json").exists() {
            // Workspace without sibling ai-protocol checkout — skip.
            return;
        }
        let loader = ProtocolLoader::new().with_base_path(&protocol_root);
        let manifest = loader
            .load_provider("google")
            .await
            .expect("google alias should resolve via provider-identity map");
        assert_eq!(manifest.id, "gemini");
        let aliases = manifest.aliases.as_ref().expect("aliases present");
        assert!(aliases.iter().any(|a| a == "google"));
    }

    #[tokio::test]
    async fn load_provider_resolves_kimi_and_glm_when_multi_family_map_present() {
        let protocol_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ai-protocol");
        let identity = protocol_root.join("dist/provider-identity.json");
        if !identity.exists() {
            return;
        }
        let raw = std::fs::read_to_string(&identity).expect("read identity map");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("parse identity map");
        // Skip when PROTO-PIN still has legacy single-family map (pre-005d).
        if value.get("families").is_none() {
            return;
        }
        if !protocol_root
            .join("dist/v2/providers/moonshot.json")
            .exists()
        {
            return;
        }

        let loader = ProtocolLoader::new().with_base_path(&protocol_root);
        let moonshot = loader
            .load_provider("kimi")
            .await
            .expect("kimi alias should resolve to moonshot");
        assert_eq!(moonshot.id, "moonshot");

        let zhipu = loader
            .load_provider("glm")
            .await
            .expect("glm alias should resolve to zhipu");
        assert_eq!(zhipu.id, "zhipu");
    }
}
