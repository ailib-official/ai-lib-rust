use crate::protocol::ProtocolManifest;
use crate::{BoxStream, Result};
use bytes::Bytes;
use futures::TryStreamExt;
use reqwest::Proxy;
use std::collections::HashSet;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Tracks `auth_type` strings already warned about by `apply_auth`, so the
/// "unknown auth_type, falling back to bearer" warn fires once per process
/// per offending value rather than once per request.
fn unknown_auth_type_seen() -> &'static Mutex<HashSet<String>> {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
}

fn warn_unknown_auth_type_once(auth_type: &str) {
    let Ok(mut seen) = unknown_auth_type_seen().lock() else {
        return;
    };
    if seen.insert(auth_type.to_string()) {
        tracing::warn!(
            auth_type,
            "unknown manifest auth.type; falling back to Bearer Authorization header"
        );
    }
}

struct TransportRoute {
    label: String,
    client: reqwest::Client,
}

fn auth_header_value(prefix: &str, secret: &str) -> String {
    let trimmed = prefix.trim();
    if trimmed.is_empty() {
        secret.to_string()
    } else {
        format!("{trimmed} {secret}")
    }
}

pub struct HttpTransport {
    routes: Vec<TransportRoute>,
    preferred_route: AtomicUsize,
    base_url: String,
    model: String,
    credential: crate::credentials::ResolvedCredential,
    auth: Option<crate::protocol::AuthConfig>,
}

impl HttpTransport {
    /// Create a new HttpTransport from a manifest.
    ///
    /// If `base_url_override` is provided, it will be used instead of the manifest's base_url.
    /// This is useful for testing with mock servers.
    pub fn new(manifest: &ProtocolManifest, model: &str) -> Result<Self> {
        Self::new_with_base_url(manifest, model, None)
    }

    /// Create a new HttpTransport with an optional base_url override.
    ///
    /// This is primarily for testing, allowing injection of mock server URLs.
    pub fn new_with_base_url(
        manifest: &ProtocolManifest,
        model: &str,
        base_url_override: Option<&str>,
    ) -> Result<Self> {
        Self::new_with_base_url_and_credential(manifest, model, base_url_override, None)
    }

    pub fn new_with_base_url_and_credential(
        manifest: &ProtocolManifest,
        model: &str,
        base_url_override: Option<&str>,
        credential_override: Option<&str>,
    ) -> Result<Self> {
        let credential = crate::credentials::resolve_credential(manifest, credential_override);
        let auth = crate::credentials::primary_auth(manifest).cloned();
        if let Some(shadowed) = crate::credentials::shadowed_auth(manifest) {
            tracing::warn!(
                provider = crate::credentials::provider_id(manifest),
                primary_auth_type = auth.as_ref().map(|a| a.auth_type.as_str()).unwrap_or(""),
                primary_token_env = auth
                    .as_ref()
                    .and_then(|a| a.token_env.as_deref().or(a.key_env.as_deref()))
                    .unwrap_or(""),
                shadowed_auth_type = shadowed.auth_type.as_str(),
                shadowed_token_env = shadowed
                    .token_env
                    .as_deref()
                    .or(shadowed.key_env.as_deref())
                    .unwrap_or(""),
                "manifest declares both endpoint.auth and top-level auth with different credentials; \
                 endpoint.auth wins, top-level auth is ignored. Update the manifest to remove the \
                 redundant top-level block."
            );
        }
        if credential.secret().is_some() {
            tracing::debug!(
                provider = crate::credentials::provider_id(manifest),
                source_kind = ?credential.source_kind,
                source_name = credential.source_name.as_deref().unwrap_or("unknown"),
                "resolved provider credential"
            );
        } else {
            tracing::warn!(
                provider = crate::credentials::provider_id(manifest),
                required_envs = ?credential.required_envs,
                conventional_envs = ?credential.conventional_envs,
                "no provider credential found"
            );
        }

        // Use override if provided, otherwise use manifest endpoint.base_url
        let base_url = base_url_override
            .map(|s| s.to_string())
            .unwrap_or_else(|| manifest.get_base_url().to_string());

        let routes = Self::build_routes()?;

        Ok(Self {
            routes,
            preferred_route: AtomicUsize::new(0),
            base_url,
            model: model.to_string(),
            credential,
            auth,
        })
    }

    fn proxy_candidates() -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for key in ["AI_PROXY_URL", "HTTPS_PROXY", "HTTP_PROXY"] {
            if let Ok(value) = env::var(key) {
                let trimmed = value.trim();
                if !trimmed.is_empty() && seen.insert(trimmed.to_string()) {
                    out.push(trimmed.to_string());
                }
            }
        }
        out
    }

    fn client_builder(has_failover_routes: bool) -> reqwest::ClientBuilder {
        let timeout_secs = env::var("AI_HTTP_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .or_else(|| {
                env::var("AI_TIMEOUT_SECS")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
            })
            .unwrap_or(300);
        let connect_timeout_ms = env::var("AI_HTTP_CONNECT_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(if has_failover_routes { 2500 } else { 10000 });

        reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .connect_timeout(Duration::from_millis(connect_timeout_ms))
            .pool_max_idle_per_host(
                env::var("AI_HTTP_POOL_MAX_IDLE_PER_HOST")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(32),
            )
            .pool_idle_timeout(Some(Duration::from_secs(
                env::var("AI_HTTP_POOL_IDLE_TIMEOUT_SECS")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(90),
            )))
            .http2_adaptive_window(true)
            .http2_keep_alive_interval(Some(Duration::from_secs(30)))
            .http2_keep_alive_timeout(Duration::from_secs(10))
    }

    fn build_client(proxy_url: Option<&str>, has_failover_routes: bool) -> Result<reqwest::Client> {
        let mut builder = Self::client_builder(has_failover_routes);
        if let Some(proxy_url) = proxy_url {
            let proxy = Proxy::all(proxy_url).map_err(|e| {
                crate::Error::Transport(crate::transport::TransportError::Other(e.to_string()))
            })?;
            builder = builder.proxy(proxy);
        } else {
            builder = builder.no_proxy();
        }

        builder.build().map_err(|e| {
            crate::Error::Transport(crate::transport::TransportError::Other(e.to_string()))
        })
    }

    fn build_routes() -> Result<Vec<TransportRoute>> {
        let proxies = Self::proxy_candidates();
        let has_failover_routes = !proxies.is_empty();
        let mut routes = Vec::with_capacity(1 + proxies.len());
        routes.push(TransportRoute {
            label: "direct".to_string(),
            client: Self::build_client(None, has_failover_routes)?,
        });
        for proxy_url in proxies {
            routes.push(TransportRoute {
                label: format!("proxy:{}", proxy_url),
                client: Self::build_client(Some(&proxy_url), has_failover_routes)?,
            });
        }
        Ok(routes)
    }

    fn preferred_route_indices(&self) -> Vec<usize> {
        let len = self.routes.len();
        let start = self
            .preferred_route
            .load(Ordering::Relaxed)
            .min(len.saturating_sub(1));
        (0..len).map(|offset| (start + offset) % len).collect()
    }

    fn should_try_alternate_route(status: u16) -> bool {
        matches!(status, 403 | 407 | 451 | 502 | 503 | 504)
    }

    fn apply_auth(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let Some(secret) = self.credential.secret() else {
            return request;
        };
        let Some(auth) = self.auth.as_ref() else {
            return request.bearer_auth(secret);
        };
        match auth.auth_type.as_str() {
            "query_param" => {
                let param = auth.param_name.as_deref().unwrap_or("api_key");
                request.query(&[(param, secret)])
            }
            "api_key" | "custom_header" | "header" => {
                let header = auth.header_name.as_deref().unwrap_or("X-API-Key");
                request.header(header, secret)
            }
            "bearer" => {
                let header = auth.header_name.as_deref().unwrap_or("Authorization");
                let prefix = auth.prefix.as_deref().unwrap_or("Bearer");
                request.header(header, auth_header_value(prefix, secret))
            }
            other => {
                warn_unknown_auth_type_once(other);
                let header = auth.header_name.as_deref().unwrap_or("Authorization");
                let prefix = auth.prefix.as_deref().unwrap_or("Bearer");
                request.header(header, auth_header_value(prefix, secret))
            }
        }
    }

    pub async fn execute_stream_response(
        &self,
        method: &str,
        path: &str,
        request_body: &serde_json::Value,
        client_request_id: Option<&str>,
        accept_event_stream: bool,
    ) -> Result<reqwest::Response> {
        let interpolated_path = path.replace("{model}", &self.model);
        let url = format!("{}{}", self.base_url, interpolated_path);
        let mut last_err = None;
        for idx in self.preferred_route_indices() {
            let route = &self.routes[idx];
            let mut req = match method.to_uppercase().as_str() {
                "POST" => route.client.post(&url).json(request_body),
                "PUT" => route.client.put(&url).json(request_body),
                "DELETE" => route.client.delete(&url),
                _ => route.client.get(&url),
            };

            req = self.apply_auth(req);
            if accept_event_stream {
                req = req.header("accept", "text/event-stream");
            } else {
                req = req.header("accept", "application/json");
            }
            if let Some(id) = client_request_id {
                req = req.header("x-ai-protocol-request-id", id);
            }

            match req.send().await {
                Ok(resp) => {
                    if self.routes.len() > 1
                        && Self::should_try_alternate_route(resp.status().as_u16())
                    {
                        tracing::debug!(
                            route = route.label.as_str(),
                            url = url.as_str(),
                            status = resp.status().as_u16(),
                            "http route returned retryable route status, trying alternate route"
                        );
                        continue;
                    }
                    self.preferred_route.store(idx, Ordering::Relaxed);
                    tracing::debug!(
                        route = route.label.as_str(),
                        url = url.as_str(),
                        "http route selected"
                    );
                    return Ok(resp);
                }
                Err(e) => last_err = Some(e),
            }
        }

        Err(crate::Error::Transport(
            crate::transport::TransportError::Http(last_err.expect("at least one route exists")),
        ))
    }

    pub async fn execute_stream<'a>(
        &'a self,
        method: &str,
        path: &str,
        request_body: &serde_json::Value,
    ) -> Result<BoxStream<'a, Bytes>> {
        let resp = self
            .execute_stream_response(method, path, request_body, None, true)
            .await?;

        // Convert reqwest bytes stream to our unified BoxStream
        let byte_stream = resp
            .bytes_stream()
            .map_err(|e| crate::Error::Transport(crate::transport::TransportError::Http(e)));
        Ok(Box::pin(byte_stream))
    }

    pub async fn execute_get(&self, path: &str) -> Result<serde_json::Value> {
        self.execute_service(path, "GET", None, None).await
    }

    pub async fn execute_service(
        &self,
        path: &str,
        method: &str,
        headers: Option<&std::collections::HashMap<String, String>>,
        query_params: Option<&std::collections::HashMap<String, String>>,
    ) -> Result<serde_json::Value> {
        let interpolated_path = path.replace("{model}", &self.model);
        let url = format!("{}{}", self.base_url, interpolated_path);
        let mut last_err = None;
        for idx in self.preferred_route_indices() {
            let route = &self.routes[idx];
            let mut request = match method.to_uppercase().as_str() {
                "POST" => route.client.post(&url),
                "PUT" => route.client.put(&url),
                "DELETE" => route.client.delete(&url),
                _ => route.client.get(&url),
            };

            request = self.apply_auth(request);
            if let Some(headers) = headers {
                for (k, v) in headers {
                    request = request.header(k, v);
                }
            }
            if let Some(params) = query_params {
                request = request.query(params);
            }

            match request.send().await {
                Ok(response) => {
                    if self.routes.len() > 1
                        && Self::should_try_alternate_route(response.status().as_u16())
                    {
                        tracing::debug!(
                            route = route.label.as_str(),
                            url = url.as_str(),
                            status = response.status().as_u16(),
                            "service route returned retryable route status, trying alternate route"
                        );
                        continue;
                    }
                    self.preferred_route.store(idx, Ordering::Relaxed);
                    tracing::debug!(
                        route = route.label.as_str(),
                        url = url.as_str(),
                        "service route selected"
                    );
                    let json = response.json().await.map_err(|e| {
                        crate::Error::Transport(crate::transport::TransportError::Http(e))
                    })?;
                    return Ok(json);
                }
                Err(e) => last_err = Some(e),
            }
        }

        Err(crate::Error::Transport(
            crate::transport::TransportError::Http(last_err.expect("at least one route exists")),
        ))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Transport error: {0}")]
    Other(String),
}
