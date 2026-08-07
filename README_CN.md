# ai-lib-rust

**[AI-Protocol](https://github.com/ailib-official/ai-protocol) 协议运行时** — 高性能 Rust 参考实现（v**1.3.0**）。

[English](README.md)

`ai-lib-rust` 是应用侧最常用的聚合 crate：再导出 **`ai-lib-core`**（执行层）与 **`ai-lib-contact`**（策略层），保持 `ai_lib_rust::…` 导入路径稳定。

## 工作原理

**默认聊天路径：** `AiClient` 加载 provider manifest → 由 manifest 算子构建 **`Pipeline`** → 经 **`HttpTransport`** 发 HTTP。流式帧归一为 **`StreamingEvent`**。

聊天路径由协议驱动，但并非“零 provider 代码”：仓库仍包含 provider 专用 **decoder/mapper**、可选 **`ProviderDriver`**（WASM / 合规测试 / 高级集成），以及 embeddings / STT / TTS / rerank 的独立 HTTP 客户端。

| 层 | Crate | 职责 |
|----|-------|------|
| 执行 (E) | `ai-lib-core` | `AiClient`、`pipeline`、`protocol`、`transport`、`types`、`structured`、可选能力模块 |
| 策略 (P) | `ai-lib-contact` | `context`（分层拼装）、`resilience`、`cache`、`routing`、`plugins`、`guardrails`、`batch`、`telemetry`、`tokens` |
| 门面 | `ai-lib-rust` | 再导出 + 示例、集成测试、CLI |

已发布到 [crates.io](https://crates.io/crates/ai-lib-rust)：**`ai-lib-core`**、**`ai-lib-contact`**、**`ai-lib-rust`**（均为 **1.3.0**）。`ai-lib-wasm` 面向 `wasm32-wasip1` 构建，不发布。

> **钉版本：** 优先 crates.io **1.3.0**（标签 `v1.3.0`）。CI 钉住 `ai-protocol` **v1.2.0**。见 [CHANGELOG](CHANGELOG.md)。

## 快速开始

```toml
[dependencies]
ai-lib-rust = "1.3.0"
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

同款示例：`cargo run --example basic_usage`（需 `DEEPSEEK_API_KEY`）。

### 流式

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

### 跨任务共享

`AiClient` 不可 `Clone`（API key / ToS 边界）。使用 `Arc<AiClient>`：

```rust
use ai_lib_rust::AiClient;
use std::sync::Arc;

let client = Arc::new(AiClient::new("openai/gpt-4o").await?);
```

## 公共 API（crate 根）

始终可用（非 WASM）：

- **Client：** `AiClient`、`AiClientBuilder`、`ChatBatchRequest`、`CancelHandle`、`CallStats`、`ClientMetrics`、`EndpointExt`
- **Types：** `Message`、`MessageRole`、`StreamingEvent`、`ToolCall`、`ExecutionMetadata`、`ExecutionResult`、`ExecutionUsage`
- **Errors：** `Result`、`Error`、`ErrorContext`、`StandardErrorCode`
- **Feedback：** `FeedbackEvent`、`FeedbackSink`
- **Structured output：** `structured` 模块（`JsonModeConfig`、`OutputValidator` 等）
- **Text-tool / TTC：** `StandardTextToolParser`、`ToolCallingPolicy`、`TextToolConfig` 等
- **策略（始终再导出）：** `cache`、`context`、`plugins`、`resilience`

`ai_lib_rust::context`（策略层，默认可用）提供分层上下文拼装：`MessageChunk`、`ContextLayer`、`MessageAssembler::assemble_layered`、`AssemblePool`（带并发/超时的异步门面）；当 System+Active 超出 token 预算时返回 `AssembleError::HardBudgetViolation`。

来自 `ai-lib-contact` 的 feature 门控再导出：`batch`、`guardrails`、`interceptors`、`routing`（`routing_mvp`）、`telemetry`、`tokens`。

`ai-lib-core` 中 feature 门控模块：`embeddings`、`mcp`、`computer_use`、`multimodal`、`stt`、`tts`、`rerank`。

### Feature 实际含义

| Feature | 得到什么 | 说明 |
|---------|----------|------|
| `keyring`（**默认**） | 操作系统密钥环回退 | 精简/CI 构建用 `default-features = false` |
| `embeddings` | `EmbeddingClient` | 协议化 builder：`from_model` / `from_manifest`（无厂商 URL 默认值） |
| `stt` / `tts` / `reranking` | `SttClient`、`TtsClient`、`RerankerClient` | 独立服务客户端；rerank 支持 `from_model` / `from_manifest` |
| `mcp` | `McpToolBridge` | 线格式转换/过滤；**不含**内置 MCP 传输客户端 |
| `computer_use` | `ComputerAction`、`SafetyPolicy` | Schema + 校验；**不含**动作执行运行时 |
| `multimodal` | `MultimodalCapabilities` | 模态检测 / 格式检查 |
| `reasoning` | 仅注册表标志 | 推理 delta 在核心流水线中可用，无需开此 feature |
| `batch` | `BatchExecutor`（contact） | `AiClient::chat_batch` / `chat_batch_smart` **始终可用** |
| `telemetry` | `InMemoryFeedbackSink`、`report_feedback` 等 | 无此 feature 时核心仍导出 `FeedbackEvent` / `FeedbackSink` |
| `routing_mvp` | `CustomModelManager`、`ModelArray` 等 | 纯路由辅助 |
| `full` | 以上全部 | |

在 `Cargo.toml` 中启用：

```toml
ai-lib-rust = { version = "1.3.0", features = ["embeddings", "telemetry"] }
```

## 进阶：`ProviderDriver`

`ai_lib_rust::drivers` 暴露 `ProviderDriver`、`create_driver` 以及 OpenAI / Anthropic / Gemini 驱动。**聊天路径上 `AiClient` 不用这套接口**，而是 `Pipeline::from_manifest`。驱动用于 WASM、合规测试与自定义集成。

## 弹性

- **内置于 `AiClient`：** `max_inflight` 背压（`AiClientBuilder::max_inflight` 或 `AI_LIB_MAX_INFLIGHT`）。
- **可选策略层：** `ai_lib_rust::resilience`（重试、限流、熔断）— 需自行挂到客户端旁；`AiClient::new` 不会自动启用。
- **批量并发：** `AI_LIB_BATCH_CONCURRENCY`。
- **HTTP 代理：** 直连路由遵循 reqwest 系统代理（`HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`）；设置 `AI_PROXY_URL` 可增加显式故障转移候选路由。

## 协议 Manifest

Provider manifest 解析顺序：

1. `ProtocolLoader::with_base_path(...)`
2. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH`（本地目录或 GitHub raw URL）
3. 开发路径：`ai-protocol/`、`../ai-protocol/`、…
4. 兜底：GitHub raw `ailib-official/ai-protocol`（`main`）

每个 base path：`dist/v2/providers/<id>.json` → `v2/providers/<id>.yaml` → `dist/v1/providers/<id>.json` → `v1/providers/<id>.yaml`。

**身份 / 别名：** `load_provider` 通过 `dist/provider-identity.json`（多家族映射）解析市场别名，例如 `google` → `gemini`、`kimi` → `moonshot`。别名查找不会掩盖校验错误。

**Wire model id：** 聊天请求经 v1 模型注册表解析 OpenAI 兼容的 `model` 字段（`resolve_wire_model_id`），并对 `nvidia/<name>` 做 NIM 感知回退。

**Endpoints：** 操作名 `"chat"` 在缺少规范 `chat` 键时回退到 `endpoints.chat_openai`（DeepSeek 双 API manifest）。

**模型模态（Experimental）：** 若存在 `metadata.models.<id>`，优先采用其中的模态事实，而非 provider 级广告字段（ALR-ME-001）。

Manifest 缓存仅为内存。`with_hot_reload(true)` 只存标志，**不会监视文件** — 变更后请调用 `ProtocolLoader::clear_cache()` 或重建客户端。

## API 密钥

1. Builder 覆盖（若设置）
2. Manifest 声明的环境变量（`auth.token_env` / `auth.key_env`）
3. 约定式 `<PROVIDER_ID>_API_KEY` 环境变量
4. 操作系统密钥环（可选，`keyring` feature，桌面）

CI/容器推荐环境变量；用 `default-features = false` 去掉 keyring。

## 标准错误码（V2）

| 错误码 | 名称 | 可重试 | 可回退 |
|--------|------|--------|--------|
| E1001 | `invalid_request` | 否 | 否 |
| E1002 | `authentication` | 否 | 是 |
| E1003 | `permission_denied` | 否 | 否 |
| E1004 | `not_found` | 否 | 否 |
| E1005 | `request_too_large` | 否 | 否 |
| E2001 | `rate_limited` | 是 | 是 |
| E2002 | `quota_exhausted` | 否 | 是 |
| E3001 | `server_error` | 是 | 是 |
| E3002 | `overloaded` | 是 | 是 |
| E3003 | `timeout` | 是 | 是 |
| E4001 | `conflict` | 是 | 否 |
| E4002 | `cancelled` | 否 | 否 |
| E9999 | `unknown` | 否 | 否 |

## 测试

```bash
# 门面 crate 单元 + 集成
cargo test

# 跨运行时 YAML 合规
COMPLIANCE_DIR=../ai-protocol/tests/compliance cargo test --test compliance

# 全部可选 feature
cargo test --features full
```

Mock 服务集成（需 [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock)）：

```bash
MOCK_HTTP_URL=http://localhost:4010 cargo test -- --ignored
```

## 示例

| 示例 | Features |
|------|----------|
| `basic_usage` | — |
| `deepseek_chat_stream` | 流式收集 |
| `custom_protocol` | manifest 路径 |
| `resilience_patterns` | 策略层 |
| `batch_processing` | `batch` |
| `embeddings_similarity` | `embeddings` |
| `guardrails_usage` | `guardrails` |
| `multi_provider` | `routing_mvp` |
| `tavily_tool_calling` | tools |
| … | 见 `crates/ai-lib-rust/Cargo.toml` 的 `[[example]]` 与 `examples/` |

CLI：`cargo run --bin validate_protocols`、`cargo run --bin ai-protocol-cli`。

## WASM

```bash
cargo build -p ai-lib-wasm --target wasm32-wasip1 --release
# → target/wasm32-wasip1/release/ai_lib_wasm.wasm
```

## 相关

- [AI-Protocol](https://github.com/ailib-official/ai-protocol) — 规范与 manifest
- [ai-lib-python](https://github.com/ailib-official/ai-lib-python) — Python 运行时
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 工作区布局

## 许可证

双许可：[Apache-2.0](LICENSE-APACHE) 或 [MIT](LICENSE-MIT)。
