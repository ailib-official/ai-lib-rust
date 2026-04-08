# ai-lib-rust

**Protocol Runtime for AI-Protocol** — High-performance Rust reference implementation

`ai-lib-rust` is the Rust runtime implementation for the [AI-Protocol](https://github.com/ailib-official/ai-protocol) specification, embodying the core design principle:

> **All logic is operators, all configuration is protocol**

## 🎯 Design Philosophy

Unlike traditional adapter libraries that hardcode provider-specific logic, `ai-lib-rust` is a **protocol-driven runtime**:

- **Zero Hardcoding** — All behavior is driven by protocol manifests (YAML/JSON)
- **Operator Pipeline** — Decoder → Selector → Accumulator → FanOut → EventMapper
- **Hot Reload** — Protocol configurations can be updated at runtime without restart
- **Unified Interface** — Single API for all providers, no provider-specific code needed

## 🏗️ v0.9 Architecture: E/P Separation

Starting from v0.9.0, `ai-lib-rust` adopts an **Execution/Policy separation** architecture, decoupling core execution from policy decisions:

```
┌─────────────────────────────────────────────────────────────┐
│                      ai-lib-rust (Facade)                    │
│         Backward-compatible entry point, re-exports core    │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────┐
│    ai-lib-core      │               │   ai-lib-contact    │
│   (E: Execution)    │               │    (P: Policy)      │
├─────────────────────┤               ├─────────────────────┤
│ • protocol loader   │               │ • routing           │
│ • client            │               │ • cache             │
│ • transport         │               │ • batch             │
│ • pipeline          │◄──────────────│ • plugins           │
│ • drivers           │               │ • interceptors      │
│ • types             │               │ • guardrails        │
│ • structured output │               │ • telemetry         │
│ • mcp/computer_use  │               │ • tokens            │
│ • embeddings        │               │ • resilience        │
└─────────────────────┘               └─────────────────────┘
          │
          ▼
┌─────────────────────┐
│     ai-lib-wasm     │
│   (WASI Exports)    │
├─────────────────────┤
│ • 6 export functions│
│ • wasm32-wasip1     │
│ • ~1.2 MB binary    │
│ • No P-layer deps   │
└─────────────────────┘
```

### Benefits of E/P Separation

| Aspect | E Layer (ai-lib-core) | P Layer (ai-lib-contact) |
|--------|----------------------|--------------------------|
| **Responsibility** | Deterministic execution, protocol loading, type conversion | Policy decisions, caching, routing, telemetry |
| **Dependencies** | Minimal, stateless | Depends on E layer, may be stateful |
| **WASM** | Compiles to wasm32-wasip1 | Not supported (policy logic unsuitable) |
| **Use Case** | Edge devices, browsers, serverless | Server-side, full applications |

### Cargo Workspace Structure

| Crate | Path | Role | Published |
|-------|------|------|-----------|
| `ai-lib-core` | `crates/ai-lib-core` | Execution layer: protocol loading, client, transport, pipeline, types | crates.io |
| `ai-lib-contact` | `crates/ai-lib-contact` | Policy layer: cache, batch, routing, plugins, telemetry, resilience | crates.io |
| `ai-lib-wasm` | `crates/ai-lib-wasm` | WASI exports: 6 functions, < 2 MB | Not published |
| `ai-lib-rust` | `crates/ai-lib-rust` | Facade: re-exports core + contact, maintains backward compatibility | crates.io |
| `ai-lib-wasmtime-harness` | `crates/ai-lib-wasmtime-harness` | WASM integration tests (optional, heavy deps) | Not published |

## 📦 Installation

### Basic Installation (Facade)

```toml
[dependencies]
ai-lib-rust = "0.9"
```

### Direct Dependency on Sub-Crates

If you only need execution layer capabilities (smaller dependency graph):

```toml
[dependencies]
ai-lib-core = "0.9"
```

If you need policy layer capabilities:

```toml
[dependencies]
ai-lib-contact = "0.9"
```

### Feature Flags

```toml
[dependencies]
# Lean core (default)
ai-lib-rust = "0.9"

# Enable specific capabilities
ai-lib-rust = { version = "0.9", features = ["embeddings", "telemetry"] }

# Enable all capabilities
ai-lib-rust = { version = "0.9", features = ["full"] }
```

**Execution Layer Features** (ai-lib-core):
- `embeddings` — Embedding vector generation
- `mcp` — MCP tool bridging
- `computer_use` — Computer Use abstraction
- `multimodal` — Extended multimodal support
- `reasoning` — Reasoning/chain-of-thought support
- `stt` / `tts` — Speech recognition/synthesis
- `reranking` — Re-ranking

**Policy Layer Features** (ai-lib-contact):
- `batch` — Batch processing
- `cache` — Cache management
- `routing_mvp` — Model routing
- `guardrails` — Input/output guardrails
- `tokens` — Token counting and cost estimation
- `telemetry` — Telemetry and observability
- `interceptors` — Call interceptors

## 🚀 Quick Start

### Basic Usage

```rust
use ai_lib_rust::{AiClient, Message};

#[tokio::main]
async fn main() -> ai_lib_rust::Result<()> {
    // Protocol-driven: supports any provider defined in ai-protocol manifests
    let client = AiClient::new("anthropic/claude-3-5-sonnet").await?;
    
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello!"),
    ];
    
    // Non-streaming call
    let response = client
        .chat()
        .messages(messages)
        .temperature(0.7)
        .execute()
        .await?;
    
    println!("{}", response.content);
    Ok(())
}
```

### Streaming Response

```rust
use ai_lib_rust::{AiClient, Message};
use ai_lib_rust::types::events::StreamingEvent;
use futures::StreamExt;

#[tokio::main]
async fn main() -> ai_lib_rust::Result<()> {
    let client = AiClient::new("openai/gpt-4o").await?;
    
    let mut stream = client
        .chat()
        .messages(vec![Message::user("Tell me a joke")])
        .stream()
        .execute_stream()
        .await?;
    
    while let Some(event) = stream.next().await {
        match event? {
            StreamingEvent::PartialContentDelta { content, .. } => print!("{content}"),
            StreamingEvent::StreamEnd { .. } => break,
            _ => {}
        }
    }
    
    Ok(())
}
```

### Sharing Client Across Tasks

`AiClient` intentionally does not implement `Clone` (API key compliance). Use `Arc` to share:

```rust
use std::sync::Arc;

let client = Arc::new(AiClient::new("deepseek/deepseek-chat").await?);

// Pass to multiple async tasks
let handle = tokio::spawn({
    let c = Arc::clone(&client);
    async move {
        c.chat().messages(vec![Message::user("Hi")]).execute().await
    }
});
```

## 🔧 Configuration

### Protocol Manifest Search Path

The runtime searches for protocol configurations in the following order:

1. Custom path set via `ProtocolLoader::with_base_path()`
2. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH` environment variable
3. Common development paths: `ai-protocol/`, `../ai-protocol/`, `../../ai-protocol/`
4. Final fallback: GitHub raw `ailib-official/ai-protocol`

### API Keys

**Recommended** (production):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
```

**Optional** (local development): OS keyring (macOS Keychain / Windows Credential Manager / Linux Secret Service)

### Production Configuration

```bash
# Proxy
export AI_PROXY_URL="http://user:pass@host:port"

# Timeout
export AI_HTTP_TIMEOUT_SECS=30

# Concurrency limit
export AI_LIB_MAX_INFLIGHT=10

# Rate limiting
export AI_LIB_RPS=5  # or AI_LIB_RPM=300

# Circuit breaker
export AI_LIB_BREAKER_FAILURE_THRESHOLD=5
export AI_LIB_BREAKER_COOLDOWN_SECS=30
```

## 🧪 Testing

### Unit Tests

```bash
cargo test
```

### Compliance Tests (Cross-Runtime Consistency)

```bash
# Default run
cargo test --test compliance

# Specify compliance directory
COMPLIANCE_DIR=../ai-protocol/tests/compliance cargo test --test compliance

# Run from ai-lib-core (shared test suite)
COMPLIANCE_DIR=../ai-protocol/tests/compliance cargo test -p ai-lib-core --test compliance_from_core
```

### WASM Integration Tests

```bash
# Build WASM
cargo build -p ai-lib-wasm --target wasm32-wasip1 --release

# Run wasmtime tests
cargo test -p ai-lib-wasmtime-harness --test wasm_compliance
```

### Testing with Mock Server

```bash
# Start ai-protocol-mock
docker-compose up -d

# Run tests with mock
MOCK_HTTP_URL=http://localhost:4010 cargo test -- --ignored --nocapture
```

## 🌐 WASM Support

`ai-lib-wasm` provides server-side WASM support (wasmtime, Wasmer, etc.):

```bash
# Build WASM binary
cargo build -p ai-lib-wasm --target wasm32-wasip1 --release

# Output: target/wasm32-wasip1/release/ai_lib_wasm.wasm (~1.2 MB)
```

**WASM Export Functions**:
- `chat_sync` — Synchronous chat
- `chat_stream_init` / `chat_stream_poll` / `chat_stream_end` — Streaming chat
- `embed_sync` — Synchronous embedding

**Limitations**: WASM version only includes E-layer capabilities, without caching, routing, telemetry, etc.

## 📊 Observability

### Call Statistics

```rust
let (response, stats) = client.call_model_with_stats(request).await?;
println!("request_id: {}", stats.client_request_id);
println!("latency_ms: {:?}", stats.latency_ms);
```

### Telemetry Feedback (opt-in)

```rust
use ai_lib_rust::telemetry::{FeedbackEvent, ChoiceSelectionFeedback};

client.report_feedback(FeedbackEvent::ChoiceSelection(
    ChoiceSelectionFeedback {
        request_id: stats.client_request_id,
        chosen_index: 0,
        ..Default::default()
    }
)).await?;
```

## 🔄 Error Codes (V2 Specification)

All provider errors are normalized to 13 standard error codes:

| Code | Name | Retryable | Fallbackable |
|------|------|-----------|--------------|
| E1001 | `invalid_request` | No | No |
| E1002 | `authentication` | No | Yes |
| E1003 | `permission_denied` | No | No |
| E1004 | `not_found` | No | No |
| E1005 | `request_too_large` | No | No |
| E2001 | `rate_limited` | Yes | Yes |
| E2002 | `quota_exhausted` | No | Yes |
| E3001 | `server_error` | Yes | Yes |
| E3002 | `overloaded` | Yes | Yes |
| E3003 | `timeout` | Yes | Yes |
| E4001 | `conflict` | Yes | No |
| E4002 | `cancelled` | No | No |
| E9999 | `unknown` | No | No |

## 🤝 Community & Contributing

### Use Cases

- **Server Applications** — Use `ai-lib-rust` (facade) or `ai-lib-contact` directly for full capabilities
- **Edge/Embedded** — Use `ai-lib-core` for minimal dependencies and deterministic execution
- **Browser/WASM** — Use `ai-lib-wasm` for WebAssembly environments

### Contributing Guidelines

1. All protocol configurations must follow AI-Protocol specification (v1.5 / V2)
2. New features must include tests; compliance tests must pass
3. Code must pass `cargo clippy` checks
4. Follow [Rust API design principles](https://rust-lang.github.io/api-guidelines/)

### Code of Conduct

- Respect all contributors
- Welcome participants from all backgrounds
- Focus on technical discussions, avoid personal attacks
- Report issues via GitHub Issues

## 🔗 Related Projects

| Project | Description |
|---------|-------------|
| [AI-Protocol](https://github.com/ailib-official/ai-protocol) | Protocol specification (v1.5 / V2) |
| [ai-lib-python](https://github.com/ailib-official/ai-lib-python) | Python runtime |
| [ai-lib-ts](https://github.com/ailib-official/ai-lib-ts) | TypeScript runtime |
| [ai-lib-go](https://github.com/ailib-official/ai-lib-go) | Go runtime |
| [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock) | Mock server |

## 📄 License

This project is dual-licensed:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

You may choose either.

---

**ai-lib-rust** — Where protocol meets performance 🚀
