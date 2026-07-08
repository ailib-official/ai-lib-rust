//! HTTP transport for provider API calls.
//!
//! # Proxy policy (ALR-TRN-001)
//!
//! - **Direct route**: reqwest `ClientBuilder` defaults (`auto_sys_proxy = true`) —
//!   honors `http_proxy`, `https_proxy`, and `no_proxy` from the process environment.
//! - **Explicit override**: `AI_PROXY_URL` adds an optional failover route with a
//!   dedicated client; it supplements (does not replace) system proxy env vars.
//! - **Failover**: when `AI_PROXY_URL` is set, retryable HTTP statuses on the direct
//!   route can fall through to the explicit proxy route.
//! - Application code does not implement provider-region proxy heuristics; use env vars
//!   or `AI_PROXY_URL` instead.
//!
//! Host applications (e.g. VelaClaw `[proxy]` in `config.toml`) may configure proxies
//! for their own HTTP clients; LLM traffic via `HttpTransport` follows the rules above
//! unless the host sets matching process env vars before initializing `AiClient`.

pub mod http;
pub mod middleware;

pub use http::{HttpTransport, TransportError};
