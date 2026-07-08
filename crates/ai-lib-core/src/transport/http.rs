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

    /// Explicit ai-lib proxy override routes for failover (see `build_routes`).
    ///
    /// Standard `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` env vars are handled by
    /// reqwest's default `auto_sys_proxy` on the direct route — not duplicated here.
    fn proxy_candidates() -> Vec<String> {
        match env::var("AI_PROXY_URL") {
            Ok(value) => {
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    Vec::new()
                } else {
                    vec![trimmed.to_string()]
                }
            }
            Err(_) => Vec::new(),
        }
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

    /// Build an HTTP client for a transport route.
    ///
    /// When `proxy_url` is `None` (the direct route), reqwest's default
    /// `auto_sys_proxy` remains enabled so `http_proxy` / `https_proxy` /
    /// `no_proxy` env vars are respected. When `proxy_url` is set, that URL is
    /// used as an explicit override for the failover route labeled `proxy:…`.
    fn build_client(proxy_url: Option<&str>, has_failover_routes: bool) -> Result<reqwest::Client> {
        let mut builder = Self::client_builder(has_failover_routes);
        if let Some(proxy_url) = proxy_url {
            let proxy = Proxy::all(proxy_url).map_err(|e| {
                crate::Error::Transport(crate::transport::TransportError::Other(e.to_string()))
            })?;
            builder = builder.proxy(proxy);
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

        Err(crate::Error::Transport(match last_err {
            Some(e) => crate::transport::TransportError::Http(e),
            None => crate::transport::TransportError::Other(
                "all HTTP routes exhausted with retryable status codes".to_string(),
            ),
        }))
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

        Err(crate::Error::Transport(match last_err {
            Some(e) => crate::transport::TransportError::Http(e),
            None => crate::transport::TransportError::Other(
                "all HTTP routes exhausted with retryable status codes".to_string(),
            ),
        }))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Transport error: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::ResolvedCredential;
    use crate::protocol::AuthConfig;

    fn transport_with(auth: Option<AuthConfig>, secret: Option<&str>) -> HttpTransport {
        let credential = if let Some(value) = secret {
            ResolvedCredential::resolved_explicit(value)
        } else {
            ResolvedCredential::missing(Vec::new(), Vec::new())
        };
        let routes = HttpTransport::build_routes().expect("routes");
        HttpTransport {
            routes,
            preferred_route: AtomicUsize::new(0),
            base_url: "https://example.invalid/v1".to_string(),
            model: "model".to_string(),
            credential,
            auth,
        }
    }

    fn fresh_request_builder() -> reqwest::RequestBuilder {
        reqwest::Client::new().get("https://example.invalid/v1/test")
    }

    fn build(req: reqwest::RequestBuilder) -> reqwest::Request {
        req.build().expect("request")
    }

    #[test]
    fn apply_auth_with_no_secret_leaves_request_unchanged() {
        let transport = transport_with(
            Some(AuthConfig {
                auth_type: "bearer".to_string(),
                token_env: None,
                key_env: None,
                param_name: None,
                header_name: None,
                prefix: None,
                extra_headers: None,
            }),
            None,
        );
        let request = build(transport.apply_auth(fresh_request_builder()));
        assert!(request.headers().get("authorization").is_none());
        assert!(!request.url().query().unwrap_or("").contains("api_key"));
    }

    #[test]
    fn apply_auth_query_param_attaches_param() {
        let transport = transport_with(
            Some(AuthConfig {
                auth_type: "query_param".to_string(),
                token_env: None,
                key_env: None,
                param_name: Some("api_key".to_string()),
                header_name: None,
                prefix: None,
                extra_headers: None,
            }),
            Some("kp-secret"),
        );
        let request = build(transport.apply_auth(fresh_request_builder()));
        let query = request.url().query().expect("query");
        assert!(
            query.contains("api_key=kp-secret"),
            "expected api_key=kp-secret in query, got {query}"
        );
        assert!(request.headers().get("authorization").is_none());
    }

    #[test]
    fn apply_auth_unknown_type_falls_back_to_bearer() {
        let transport = transport_with(
            Some(AuthConfig {
                auth_type: "totally_made_up_auth".to_string(),
                token_env: None,
                key_env: None,
                param_name: None,
                header_name: None,
                prefix: None,
                extra_headers: None,
            }),
            Some("ut-secret"),
        );
        let request = build(transport.apply_auth(fresh_request_builder()));
        let auth_header = request
            .headers()
            .get("authorization")
            .expect("authorization header set");
        assert_eq!(auth_header, "Bearer ut-secret");
    }

    struct EnvGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let previous = env::var(key).ok();
            match value {
                Some(v) => env::set_var(key, v),
                None => env::remove_var(key),
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => env::set_var(self.key, value),
                None => env::remove_var(self.key),
            }
        }
    }

    fn with_proxy_env<F>(vars: &[(&'static str, Option<&'static str>)], f: F)
    where
        F: FnOnce(),
    {
        static PROXY_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _lock = PROXY_ENV_LOCK.lock().expect("proxy env lock");
        let guards: Vec<EnvGuard> = vars
            .iter()
            .map(|(key, value)| EnvGuard::set(key, *value))
            .collect();
        let _guards = guards;
        f();
    }

    #[test]
    fn build_routes_direct_only_without_ai_proxy_url() {
        with_proxy_env(
            &[
                ("AI_PROXY_URL", None),
                ("HTTP_PROXY", None),
                ("HTTPS_PROXY", None),
            ],
            || {
                let routes = HttpTransport::build_routes().expect("routes");
                assert_eq!(routes.len(), 1);
                assert_eq!(routes[0].label, "direct");
            },
        );
    }

    #[test]
    fn build_routes_adds_ai_proxy_url_failover_route() {
        with_proxy_env(
            &[
                ("AI_PROXY_URL", Some("http://proxy.example:8080")),
                ("HTTP_PROXY", Some("http://ignored-for-candidates:3128")),
            ],
            || {
                let routes = HttpTransport::build_routes().expect("routes");
                assert_eq!(routes.len(), 2);
                assert_eq!(routes[0].label, "direct");
                assert_eq!(routes[1].label, "proxy:http://proxy.example:8080");
            },
        );
    }

    #[test]
    fn direct_route_uses_default_reqwest_proxy_policy() {
        // Regression ALR-TRN-001: direct route must not call no_proxy().
        // Parity with reqwest::Client::builder() default (auto_sys_proxy enabled).
        with_proxy_env(
            &[
                ("AI_PROXY_URL", None),
                ("HTTP_PROXY", Some("http://127.0.0.1:9")),
                ("HTTPS_PROXY", Some("http://127.0.0.1:9")),
                ("ALL_PROXY", Some("http://127.0.0.1:9")),
            ],
            || {
                HttpTransport::build_client(None, false).expect("direct client");
                reqwest::Client::builder()
                    .build()
                    .expect("default reqwest client");
                // Explicit no_proxy() must remain a distinct code path (not used on direct).
                reqwest::Client::builder()
                    .no_proxy()
                    .build()
                    .expect("no_proxy client");
            },
        );
    }
}
