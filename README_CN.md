# ai-lib-rust

**AI-Protocol 协议运行时** - 高性能 Rust 参考实现

`ai-lib-rust` 是 [AI-Protocol](https://github.com/ailib-official/ai-protocol) 规范的 Rust 运行时实现。它体现了核心设计原则：**一切逻辑皆算子，一切配置皆协议** (All logic is operators, all configuration is protocol)。

## 🎯 设计哲学

与硬编码 provider 特定逻辑的传统适配器库不同，`ai-lib-rust` 是一个**协议驱动的运行时**，执行 AI-Protocol 规范。这意味着：

- **零硬编码 provider 逻辑**：所有行为都由协议 manifest 驱动（source YAML 或 dist JSON）
- **基于算子的架构**：通过可组合的算子处理（Decoder → Selector → Accumulator → FanOut → EventMapper）
- **热重载**：协议配置可以在不重启应用的情况下更新
- **统一接口**：开发者使用单一、一致的 API，无论底层 provider 是什么

## 🏗️ 架构

库分为三层：

### 1. 协议规范层 (`protocol/`)
- **Loader**: 从本地文件系统、嵌入式资源或远程 URL 加载协议文件
- **Validator**: 根据 JSON Schema 验证协议
- **Schema**: 协议结构定义

### 2. 流水线解释器层 (`pipeline/`)
- **Decoder**: 将原始字节解析为协议帧（SSE、JSON Lines 等）
- **Selector**: 使用 JSONPath 表达式过滤帧
- **Accumulator**: 累积有状态数据（例如，工具调用参数）
- **FanOut**: 处理多候选场景
- **EventMapper**: 将协议帧转换为统一事件

### 3. 用户接口层 (`client/`, `types/`)
- **Client**: 统一客户端接口
- **Types**: 基于 AI-Protocol `standard_schema` 的标准类型系统

## 🔄 V2 协议对齐

从 v0.7.0 开始，`ai-lib-rust` 与 **AI-Protocol V2** 规范对齐。V0.8.0 新增完整 V2 运行时支持，包括 V2 manifest 解析、Provider 驱动、MCP、Computer Use 及扩展多模态。

### 标准错误码（V2）

所有 provider 错误被分类为 13 个标准错误码，具有统一的重试/回退语义：

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

分类遵循优先级管道：provider 特定错误码 → HTTP 状态码覆盖 → 标准 HTTP 映射 → `E9999`。

### 兼容性测试

跨运行时行为一致性通过 `ai-protocol` 仓库中的共享 YAML 测试套件验证：

```bash
# 运行兼容性测试
cargo test --test compliance

# 指定兼容性测试目录
COMPLIANCE_DIR=../ai-protocol/tests/compliance cargo test --test compliance
```

详细信息请参阅 [CROSS_RUNTIME.md](https://github.com/ailib-official/ai-protocol/blob/main/docs/CROSS_RUNTIME.md)。

### 使用 ai-protocol-mock 进行测试

在无需真实 API 调用的集成和 MCP 测试中，可使用 [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock)：

```bash
# 启动 mock 服务（在 ai-protocol-mock 仓库中）
docker-compose up -d

# 使用 mock 运行测试
MOCK_HTTP_URL=http://localhost:4010 MOCK_MCP_URL=http://localhost:4010/mcp cargo test -- --ignored --nocapture

# 运行指定 mock 集成测试
MOCK_HTTP_URL=http://localhost:4010 cargo test test_sse_streaming_via_mock test_error_classification_via_mock -- --ignored --nocapture
```

或在代码中：`AiClientBuilder::new().base_url_override("http://localhost:4010").build(...)`

## 🧩 Feature 与 re-export（对外便利入口）

`ai-lib-rust` 的 runtime 核心保持精简；一些“更上层、更偏应用”的工具通过 feature opt-in 暴露，并在 crate root 做 re-export 以提升易用性。

更深入的架构说明见：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

- **默认可用的 crate root re-export**：
  - `AiClient`, `AiClientBuilder`, `CancelHandle`, `CallStats`, `ChatBatchRequest`, `EndpointExt`
  - `Message`, `MessageRole`, `StreamingEvent`, `ToolCall`
  - `Result<T>`, `Error`, `ErrorContext`
  - `FeedbackEvent`, `FeedbackSink`（核心反馈类型）
- **Capability features（V2 对齐）**：
  - **`embeddings`**：嵌入向量生成（`EmbeddingClient`）
  - **`batch`**：批量 API 处理（`BatchExecutor`）
  - **`guardrails`**：输入/输出校验
  - **`tokens`**：Token 计数与成本估算
  - **`telemetry`**：可观测性 Sink（`InMemoryFeedbackSink`, `ConsoleFeedbackSink` 等）
  - **`mcp`**：MCP（Model Context Protocol）工具桥接 — 基于命名空间的工具转换与过滤
  - **`computer_use`**：Computer Use 抽象 — 安全策略、域名白名单、动作校验
  - **`multimodal`**：扩展多模态 — 视觉、音频、视频模态校验与格式检查
  - **`reasoning`**：扩展推理 / 思维链支持
- **Infrastructure features**：
  - **`routing_mvp`**：纯逻辑模型管理工具（`CustomModelManager`, `ModelArray` 等）
  - **`interceptors`**：应用层调用钩子（`InterceptorPipeline`, `Interceptor`, `RequestContext`）
- **Meta-feature**：
  - **`full`**：启用所有 capability 与 infrastructure features

启用方式：

```toml
[dependencies]
ai-lib-rust = "0.8.4"

# 启用特定能力
ai-lib-rust = { version = "0.8.4", features = ["embeddings", "telemetry"] }

# 全部启用
ai-lib-rust = { version = "0.8.4", features = ["full"] }
```

## 🗺️ 能力结构清单（按层次划分）

下面是面向开发者的“能力地图”，按 runtime 的分层来组织：

### 1）协议层（`src/protocol/`）
- **`ProtocolLoader`**：从本地路径 / 环境变量路径 / GitHub raw URL 加载 provider manifest
- **`ProtocolValidator`**：JSON Schema 验证（发布后也支持离线：内置 v1 schema 兜底）
- **`ProtocolManifest`**：provider manifest 的强类型结构
- **`UnifiedRequest`**：运行时内部的统一请求结构（provider 无关）

### 2）传输层（`src/transport/`）
- **`HttpTransport`**：基于 reqwest 的传输实现（支持 `AI_PROXY_URL`、timeout 等生产 knobs）
- **API key 解析**：keyring → 环境变量 `<PROVIDER_ID>_API_KEY`

### 3）流水线解释器层（`src/pipeline/`）
- **算子流水线**：decoder → selector → accumulator → fanout → event mapper
- **流式归一化**：把 provider 的 frame 映射为统一的 `StreamingEvent`

### 4）客户端层（`src/client/`）
- **`AiClient`**：runtime 入口（`"provider/model"`）
- **Chat builder**：`client.chat().messages(...).stream().execute_stream()`
- **Batch**：`chat_batch`, `chat_batch_smart`
- **可观测性**：`call_model_with_stats` → `CallStats`
- **取消流**：`execute_stream_with_cancel()` → `CancelHandle`
- **服务发现/服务调用**：`EndpointExt` 调用 protocol `services` 声明的管理接口

### 5）弹性/策略层（`src/resilience/` + `client/policy`）
- **策略引擎**：capability 校验 + retry/fallback 决策
- **Rate limiter**：token bucket +（可选）基于 headers 的自适应模式
- **Circuit breaker**：最小熔断器（env 或 builder 默认值）
- **Backpressure**：max in-flight 并发许可

### 6）类型系统层（`src/types/`）
- **消息**：`Message`, `MessageRole`, `MessageContent`, `ContentBlock`
- **工具**：`ToolDefinition`, `FunctionDefinition`, `ToolCall`
- **事件**：`StreamingEvent`

### 7）Telemetry 层（`src/telemetry/`）
- **`FeedbackSink` / `FeedbackEvent`**：可选的反馈上报能力（opt-in）
- **扩展反馈类型**：`RatingFeedback`、`ThumbsFeedback`、`TextFeedback`、`CorrectionFeedback`、`RegenerateFeedback`、`StopFeedback`
- **多种 Sink**：`InMemoryFeedbackSink`、`ConsoleFeedbackSink`、`CompositeFeedbackSink`
- **全局 Sink 管理**：`get_feedback_sink()`、`set_feedback_sink()`、`report_feedback()`

### 8）Embedding 层（`src/embeddings/`）- v0.6.5 新增
- **`EmbeddingClient` / `EmbeddingClientBuilder`**：从文本生成嵌入向量
- **类型**：`Embedding`、`EmbeddingRequest`、`EmbeddingResponse`、`EmbeddingUsage`
- **向量运算**：`cosine_similarity`、`dot_product`、`euclidean_distance`、`manhattan_distance`
- **工具函数**：`normalize_vector`、`average_vectors`、`weighted_average_vectors`、`find_most_similar`

### 9）Cache 层（`src/cache/`）- v0.6.5 新增
- **`CacheBackend`** trait：`MemoryCache` 和 `NullCache` 实现
- **`CacheManager`**：基于 TTL 的缓存管理，支持统计
- **`CacheKey` / `CacheKeyGenerator`**：确定性缓存键生成

### 10）Token 层（`src/tokens/`）- v0.6.5 新增
- **`TokenCounter`** trait：`CharacterEstimator`、`AnthropicEstimator`、`CachingCounter`
- **`ModelPricing`**：预配置 GPT-4o、Claude 等模型定价
- **`CostEstimate`**：请求成本估算

### 11）Batch 层（`src/batch/`）- v0.6.5 新增
- **`BatchCollector` / `BatchConfig`**：请求收集与批处理配置
- **`BatchExecutor`**：可配置策略的批量执行器
- **`BatchResult`**：结构化的批量执行结果

### 12）Plugin 层（`src/plugins/`）- v0.6.5 新增
- **`Plugin`** trait：带生命周期钩子的插件接口
- **`PluginRegistry`**：集中式插件管理
- **钩子系统**：`HookType`、`Hook`、`HookManager`
- **中间件**：`Middleware`、`MiddlewareChain` 用于请求/响应转换

### 13）工具层（`src/utils/`）
- JSONPath/路径映射、tool-call assembler 等运行时小工具

### 14）可选上层工具（feature-gated）
- **`routing_mvp`**（`src/routing/`）：模型选择 + endpoint array 负载均衡（纯逻辑）
- **`interceptors`**（`src/interceptors/`）：调用前后钩子（日志/指标/审计）

## 🚀 快速开始

### 基本用法（非流式）

```rust
use ai_lib_rust::{AiClient, Message};

#[tokio::main]
async fn main() -> ai_lib_rust::Result<()> {
    // 直接使用 provider/model 字符串创建客户端
    // 这完全由协议驱动，支持 ai-protocol manifest 中定义的任何 provider
    let client = AiClient::new("deepseek/deepseek-chat").await?;

    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello! Explain the runtime briefly."),
    ];

    // 非流式：返回完整响应
    let resp = client
        .chat()
        .messages(messages)
        .temperature(0.7)
        .max_tokens(500)
        .execute()
        .await?;

    println!("Response:\n{}", resp.content);
    if let Some(usage) = resp.usage {
        println!("\nUsage: {usage:?}");
    }

    Ok(())
}
```

### 流式用法

```rust
use ai_lib_rust::{AiClient, Message};
use ai_lib_rust::types::events::StreamingEvent;
use futures::StreamExt;

#[tokio::main]
async fn main() -> ai_lib_rust::Result<()> {
    let client = AiClient::new("deepseek/deepseek-chat").await?;

    let messages = vec![Message::user("你好！")];

    // 流式：返回事件流
    let mut stream = client
        .chat()
        .messages(messages)
        .temperature(0.7)
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

### 多模态（图像 / 音频）

多模态输入表示为 `MessageContent::Blocks(Vec<ContentBlock>)`。

```rust
use ai_lib_rust::{Message, MessageRole};
use ai_lib_rust::types::message::{MessageContent, ContentBlock};

fn multimodal_message(image_path: &str) -> ai_lib_rust::Result<Message> {
    let blocks = vec![
        ContentBlock::text("简要描述这张图片。"),
        ContentBlock::image_from_file(image_path)?,
    ];
    Ok(Message::with_content(
        MessageRole::User,
        MessageContent::blocks(blocks),
    ))
}
```

### 有用的环境变量

- `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH`: 本地 `ai-protocol` 仓库根目录路径（包含 `v1/`）
- `AI_LIB_ATTEMPT_TIMEOUT_MS`: 统一策略引擎使用的每次尝试超时保护
- `AI_LIB_BATCH_CONCURRENCY`: 批量操作的并发限制覆盖

### 自定义协议

```rust
use ai_lib_rust::protocol::ProtocolLoader;

let loader = ProtocolLoader::new()
    .with_base_path("./ai-protocol")
    .with_hot_reload(true);

let manifest = loader.load_provider("openai").await?;
```

## 📦 安装

添加到 `Cargo.toml`：

```toml
[dependencies]
ai-lib-rust = "0.8.4"
tokio = { version = "1.0", features = ["full"] }
futures = "0.3"
```

## 🔧 配置

库自动在以下位置查找协议 manifest（按顺序）：

1. 通过 `ProtocolLoader::with_base_path()` 设置的自定义路径
2. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH`（本地路径或 GitHub raw URL）
3. 常见开发路径：`ai-protocol/`、`../ai-protocol/`、`../../ai-protocol/`
4. 最终兜底：GitHub raw `hiddenpath/ai-protocol`（main）

对每个 base path，provider manifest 的解析顺序为（向后兼容）：
`dist/v1/providers/<id>.json` → `v1/providers/<id>.yaml`。

协议 manifest 应遵循 AI-Protocol 规范（v1.5 / V2）结构。运行时根据 AI-Protocol 仓库中的官方 JSON Schema 验证 manifest。

## 🔐 Provider 要求（API 密钥）

大多数 provider 需要 API 密钥。运行时按以下顺序读取密钥：

1. **操作系统密钥环**（可选，便利功能）
   - **Windows**: 使用 Windows 凭据管理器
   - **macOS**: 使用 Keychain
   - **Linux**: 使用 Secret Service API
   - 服务：`ai-protocol`，用户名：provider id
   - **注意**：密钥环是可选的，在容器/WSL 中可能无法工作。会自动回退到环境变量。

2. **环境变量**（生产环境推荐）
   - 格式：`<PROVIDER_ID>_API_KEY`（例如 `DEEPSEEK_API_KEY`、`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`）
   - **推荐用于**：CI/CD、容器、WSL、生产部署

**示例**：
```bash
# 通过环境变量设置 API 密钥（推荐）
export DEEPSEEK_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# 或使用密钥环（可选，用于本地开发）
# Windows: 存储在凭据管理器中
# macOS: 存储在 Keychain 中
```

Provider 特定细节各不相同，但 `ai-lib-rust` 在统一客户端 API 后面将它们标准化。

## 🌐 代理 / 超时 / 背压（生产环境配置）

- **代理**：设置 `AI_PROXY_URL`（例如 `http://user:pass@host:port`）
- **HTTP 超时**：设置 `AI_HTTP_TIMEOUT_SECS`（后备：`AI_TIMEOUT_SECS`）
- **并发限制**：设置 `AI_LIB_MAX_INFLIGHT` 或使用 `AiClientBuilder::max_inflight(n)`
- **速率限制**（可选）：设置以下之一
  - `AI_LIB_RPS`（每秒请求数），或
  - `AI_LIB_RPM`（每分钟请求数）
- **熔断器**（可选）：通过 `AiClientBuilder::circuit_breaker_default()` 或环境变量启用
  - `AI_LIB_BREAKER_FAILURE_THRESHOLD`（默认 5）
  - `AI_LIB_BREAKER_COOLDOWN_SECS`（默认 30）

## 📊 可观测性：CallStats

如果需要每次调用的统计信息（延迟、重试、请求 ID、端点），请使用：

```rust
let (resp, stats) = client.call_model_with_stats(unified_req).await?;
println!("client_request_id={}", stats.client_request_id);
```

## 🛑 可取消的流式响应

```rust
let (mut stream, cancel) = client.chat().messages(messages).stream().execute_stream_with_cancel().await?;
// cancel.cancel(); // 发出 StreamEnd{finish_reason:"cancelled"}，丢弃底层网络流，并释放并发许可
```

## 🧾 可选反馈（Choice Selection）

遥测是**选择加入**的。您可以注入 `FeedbackSink` 并显式报告反馈：

```rust
use ai_lib_rust::telemetry::{FeedbackEvent, ChoiceSelectionFeedback};

client.report_feedback(FeedbackEvent::ChoiceSelection(ChoiceSelectionFeedback {
    request_id: stats.client_request_id.clone(),
    chosen_index: 0,
    rejected_indices: None,
    latency_to_select_ms: None,
    ui_context: None,
    candidate_hashes: None,
})).await?;
```

## 🎨 核心特性

### 协议驱动架构

没有 `match provider` 语句。所有逻辑都来自协议配置：

```rust
// 流水线从协议 manifest 动态构建
let pipeline = Pipeline::from_manifest(&manifest)?;

// 算子通过 manifest（YAML/JSON）配置，而不是硬编码
// 添加新 provider 需要零代码更改
```

### 多候选支持

通过 `FanOut` 算子自动处理多候选场景：

```yaml
streaming:
  candidate:
    candidate_id_path: "$.choices[*].index"
    fan_out: true
```

### 工具累积

工具调用参数的有状态累积：

```yaml
streaming:
  accumulator:
    stateful_tool_parsing: true
    key_path: "$.delta.partial_json"
    flush_on: "$.type == 'content_block_stop'"
```

### 热重载

协议配置可以在运行时更新：

```rust
let loader = ProtocolLoader::new().with_hot_reload(true);
// 协议更改会自动拾取
```

## 📚 示例

查看 `examples/` 目录：

- `basic_usage.rs`: 简单的非流式聊天完成
- `deepseek_chat_stream.rs`: 流式聊天示例
- `deepseek_tool_call_stream.rs`: 流式工具调用
- `custom_protocol.rs`: 加载自定义协议配置
- `list_models.rs`: 列出 provider 的可用模型
- `service_discovery.rs`: 服务发现和自定义服务调用
- `test_protocol_loading.rs`: 协议加载自检

## 🧪 测试

```bash
cargo test
```

## 📦 批量（聊天）

对于批量执行（保持顺序），请使用：

```rust
use ai_lib_rust::{AiClient, ChatBatchRequest, Message};

let client = AiClient::new("deepseek/deepseek-chat").await?;

let reqs = vec![
    ChatBatchRequest::new(vec![Message::user("你好")]),
    ChatBatchRequest::new(vec![Message::user("用一句话解释 SSE")])
        .temperature(0.2),
];

let results = client.chat_batch(reqs, Some(5)).await;
```

### 智能批量调优

如果您更喜欢保守的默认启发式，请使用：

```rust
let results = client.chat_batch_smart(reqs).await;
```

通过以下方式覆盖并发：
- `AI_LIB_BATCH_CONCURRENCY`

## 🤝 贡献

欢迎贡献！请确保：

1. 所有协议配置遵循 AI-Protocol 规范（v1.5 / V2）
2. 新算子有适当文档
3. 新功能包含测试
4. 兼容性测试通过（`cargo test --test compliance`）
5. 代码遵循 Rust 最佳实践并通过 `cargo clippy`

## 📄 许可证

本项目采用以下许可证之一：

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

您可以选择其中一种。

## 🔗 相关项目

- [AI-Protocol](https://github.com/ailib-official/ai-protocol): 协议规范（v1.5 / V2）
- [ai-lib-python](https://github.com/ailib-official/ai-lib-python): Python 运行时实现

---

**ai-lib-rust** - 协议与性能的完美结合。🚀
