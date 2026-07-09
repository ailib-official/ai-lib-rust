# ai-lib-rust

**Protocol runtime for [AI-Protocol](https://github.com/ailib-official/ai-protocol)** — high-performance Rust reference implementation (v**1.0.1**).

`ai-lib-rust` is the umbrella crate most applications depend on. It re-exports **`ai-lib-core`** (execution) and **`ai-lib-contact`** (policy) so existing `ai_lib_rust::…` import paths stay stable.

## How it works

**Default chat path:** `AiClient` loads a provider manifest → builds a **`Pipeline`** from manifest operators → sends HTTP via **`HttpTransport`**. Streaming frames are normalized to **`StreamingEvent`**.

This is protocol-driven for chat, but not “zero provider code”: the repo also ships provider-specific **decoders/mappers**, optional **`ProviderDriver`** implementations (advanced / WASM / tests), and standalone HTTP clients for embeddings, STT, TTS, and rerank.

| Layer | Crate | Responsibility |
|-------|-------|----------------|
| Execution (E) | `ai-lib-core` | `AiClient`, `pipeline`, `protocol`, `transport`, `types`, `structured`, optional capability modules |
| Policy (P) | `ai-lib-contact` | `resilience`, `cache`, `routing`, `plugins`, `guardrails`, `batch`, `telemetry`, `tokens` |
| Facade | `ai-lib-rust` | Re-exports + examples, integration tests, CLI bins |

Published on [crates.io](https://crates.io/crates/ai-lib-rust): **`ai-lib-core`**, **`ai-lib-contact`**, **`ai-lib-rust`** (all **1.0.1**). `ai-lib-wasm` is built for `wasm32-wasip1` and is not published.

## Quick start

```toml
[dependencies]
ai-lib-rust = "1.0.1"
tokio = { version = "1", features = ["full"] }
```

```bash
export DEEPSEEK_API_KEY="your-key"
```

```rust
use ai_lib_rust::{AiClient, Message};

#[tokio::main]
async fn main() -> ai_lib_rust::Result<()> {
    let client = AiClient::new("deepseek/deepseek-chat").await?;

    let response = client
        .chat()
        .messages(vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello!"),
        ])
        .temperature(0.7)
        .max_tokens(500)
        .execute()
        .await?;

    println!("{}", response.content);
    Ok(())
}
```

Same example: `cargo run --example basic_usage` (requires `DEEPSEEK_API_KEY`).

### Streaming

```rust
use ai_lib_rust::{AiClient, Message, StreamingEvent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> ai_lib_rust::Result<()> {
    let client = AiClient::new("deepseek/deepseek-chat").await?;

    let mut stream = client
        .chat()
        .messages(vec![Message::user("Write a haiku about Rust.")])
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

### Share across tasks

`AiClient` is not `Clone` (API key / ToS boundary). Use `Arc<AiClient>`:

```rust
use ai_lib_rust::AiClient;
use std::sync::Arc;

let client = Arc::new(AiClient::new("openai/gpt-4o").await?);
```

## Public API (crate root)

Always available (non-WASM):

- **Client:** `AiClient`, `AiClientBuilder`, `ChatBatchRequest`, `CancelHandle`, `CallStats`, `EndpointExt`
- **Types:** `Message`, `MessageRole`, `StreamingEvent`, `ToolCall`, `ExecutionMetadata`, `ExecutionResult`, `ExecutionUsage`
- **Errors:** `Result`, `Error`, `ErrorContext`, `StandardErrorCode`
- **Feedback:** `FeedbackEvent`, `FeedbackSink`
- **Structured output:** `structured` module (`JsonModeConfig`, `OutputValidator`, …)
- **Text-tool / TTC:** `StandardTextToolParser`, `ToolCallingPolicy`, `TextToolConfig`, …
- **Policy (always re-exported):** `cache`, `context`, `plugins`, `resilience`

Feature-gated re-exports from `ai-lib-contact`: `batch`, `guardrails`, `interceptors`, `routing` (`routing_mvp`), `telemetry`, `tokens`.

Feature-gated modules in `ai-lib-core`: `embeddings`, `mcp`, `computer_use`, `multimodal`, `stt`, `tts`, `rerank`.

### What features actually do

| Feature | What you get | Notes |
|---------|--------------|-------|
| `embeddings` | `EmbeddingClient` | Standalone OpenAI-style HTTP client |
| `stt` / `tts` / `reranking` | `SttClient`, `TtsClient`, `RerankerClient` | Standalone service clients |
| `mcp` | `McpToolBridge` | Wire-format conversion / filtering; **no** built-in MCP transport client |
| `computer_use` | `ComputerAction`, `SafetyPolicy` | Schema + validation; **no** action execution runtime |
| `multimodal` | `MultimodalCapabilities` | Modality detection / format checks |
| `reasoning` | Registry flag only | Reasoning deltas work in core pipeline without enabling this |
| `batch` | `BatchExecutor` (contact) | `AiClient::chat_batch` / `chat_batch_smart` are **always** available |
| `telemetry` | `InMemoryFeedbackSink`, `report_feedback`, … | Core exports `FeedbackEvent` / `FeedbackSink` without this feature |
| `routing_mvp` | `CustomModelManager`, `ModelArray`, … | Pure routing helpers |
| `full` | All features above | |

Enable features in `Cargo.toml`:

```toml
ai-lib-rust = { version = "1.0.1", features = ["embeddings", "telemetry"] }
```

## Advanced: `ProviderDriver`

`ai_lib_rust::drivers` exposes `ProviderDriver`, `create_driver`, and OpenAI / Anthropic / Gemini drivers. **`AiClient` does not use this path** for chat; it uses `Pipeline::from_manifest`. Drivers are for WASM targets, compliance tests, and custom integrations.

## Resilience

- **Built into `AiClient`:** `max_inflight` backpressure (`AiClientBuilder::max_inflight` or `AI_LIB_MAX_INFLIGHT`).
- **Opt-in policy layer:** `ai_lib_rust::resilience` (retry, rate limiter, circuit breaker) — wire beside the client; not auto-enabled on `AiClient::new`.
- **Batch concurrency:** `AI_LIB_BATCH_CONCURRENCY`.

## Protocol manifests

Resolution order for provider manifests:

1. `ProtocolLoader::with_base_path(...)`
2. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH` (local dir or GitHub raw URL)
3. Dev paths: `ai-protocol/`, `../ai-protocol/`, …
4. Fallback: GitHub raw `ailib-official/ai-protocol` (`main`)

Per base path: `dist/v2/providers/<id>.json` → `v2/providers/<id>.yaml` → `dist/v1/providers/<id>.json` → `v1/providers/<id>.yaml`.

Manifest cache: in-memory only. `with_hot_reload(true)` stores a flag but **does not watch files** — call `ProtocolLoader::clear_cache()` or rebuild the client after manifest changes.

## API keys

1. OS keyring (optional, `keyring` feature, desktop)
2. `<PROVIDER_ID>_API_KEY` env var (recommended for CI/containers)

## Standard error codes (V2)

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

## Testing

```bash
# Unit + integration (facade crate)
cargo test

# Cross-runtime YAML compliance
COMPLIANCE_DIR=../ai-protocol/tests/compliance cargo test --test compliance

# All optional features
cargo test --features full
```

Mock server integration (requires [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock)):

```bash
MOCK_HTTP_URL=http://localhost:4010 cargo test -- --ignored
```

## Examples

| Example | Features |
|---------|----------|
| `basic_usage` | — |
| `deepseek_chat_stream` | streaming collect |
| `custom_protocol` | manifest paths |
| `resilience_patterns` | policy layer |
| `batch_processing` | `batch` |
| `embeddings_similarity` | `embeddings` |
| `guardrails_usage` | `guardrails` |
| `multi_provider` | `routing_mvp` |
| `tavily_tool_calling` | tools |
| … | see `crates/ai-lib-rust/Cargo.toml` `[[example]]` |

CLI bins: `cargo run --bin validate_protocols`, `cargo run --bin ai-protocol-cli`.

## WASM

```bash
cargo build -p ai-lib-wasm --target wasm32-wasip1 --release
# → target/wasm32-wasip1/release/ai_lib_wasm.wasm
```

## Related

- [AI-Protocol](https://github.com/ailib-official/ai-protocol) — specification & manifests
- [ai-lib-python](https://github.com/ailib-official/ai-lib-python) — Python runtime
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — workspace layout

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
