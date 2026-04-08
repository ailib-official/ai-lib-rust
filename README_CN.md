# ai-lib-rust

**AI-Protocol 协议运行时** — 高性能 Rust 参考实现

`ai-lib-rust` 是 [AI-Protocol](https://github.com/ailib-official/ai-protocol) 规范的 Rust 运行时实现，体现了核心设计原则：

> **一切逻辑皆算子，一切配置皆协议**

## 🎯 设计哲学

与传统硬编码 Provider 逻辑的适配器库不同，`ai-lib-rust` 是一个**协议驱动的运行时**：

- **零硬编码** — 所有行为由协议 Manifest（YAML/JSON）驱动
- **算子流水线** — Decoder → Selector → Accumulator → FanOut → EventMapper
- **热重载** — 协议配置可在运行时更新，无需重启应用
- **统一接口** — 单一 API 适配所有 Provider，开发者无需关心底层差异

## 🏗️ v0.9 架构：E/P 分层

从 v0.9.0 开始，`ai-lib-rust` 采用 **执行层/策略层分离** 架构，将核心执行能力与策略决策解耦：

```
┌─────────────────────────────────────────────────────────────┐
│                      ai-lib-rust (Facade)                    │
│              向后兼容的统一入口，重新导出 core + contact        │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────┐
│    ai-lib-core      │               │   ai-lib-contact    │
│     (E 执行层)       │               │     (P 策略层)       │
├─────────────────────┤               ├─────────────────────┤
│ • protocol 加载     │               │ • routing 路由       │
│ • client 客户端     │               │ • cache 缓存         │
│ • transport 传输    │               │ • batch 批处理       │
│ • pipeline 流水线   │◄──────────────│ • plugins 插件       │
│ • drivers 驱动      │               │ • interceptors 拦截器│
│ • types 类型系统    │               │ • guardrails 守卫    │
│ • structured 结构化 │               │ • telemetry 遥测     │
│ • mcp/computer_use  │               │ • tokens Token计算   │
│ • embeddings 嵌入   │               │ • resilience 弹性    │
└─────────────────────┘               └─────────────────────┘
          │
          ▼
┌─────────────────────┐
│     ai-lib-wasm     │
│   (WASI 导出层)      │
├─────────────────────┤
│ • 6 个导出函数       │
│ • wasm32-wasip1     │
│ • ~1.2 MB 二进制     │
│ • 不含 P 层依赖      │
└─────────────────────┘
```

### E/P 分层的优势

| 维度 | E 层 (ai-lib-core) | P 层 (ai-lib-contact) |
|------|-------------------|----------------------|
| **职责** | 确定性执行、协议加载、类型转换 | 策略决策、缓存、路由、遥测 |
| **依赖** | 最小化，无状态 | 依赖 E 层，可有状态 |
| **WASM** | 可编译到 wasm32-wasip1 | 不支持（策略逻辑不适合 WASM） |
| **适用场景** | 边缘设备、浏览器、Serverless | 服务端、完整应用 |

### Cargo Workspace 结构

| Crate | 路径 | 角色 | 发布 |
|-------|------|------|------|
| `ai-lib-core` | `crates/ai-lib-core` | 执行层：协议加载、客户端、传输、流水线、类型 | crates.io |
| `ai-lib-contact` | `crates/ai-lib-contact` | 策略层：缓存、批处理、路由、插件、遥测、弹性 | crates.io |
| `ai-lib-wasm` | `crates/ai-lib-wasm` | WASI 导出：6 个函数，< 2 MB | 不发布 |
| `ai-lib-rust` | `crates/ai-lib-rust` | 门面层：重新导出 core + contact，保持向后兼容 | crates.io |
| `ai-lib-wasmtime-harness` | `crates/ai-lib-wasmtime-harness` | WASM 集成测试（可选，依赖较重） | 不发布 |

## 📦 安装

### 基础安装（门面层）

```toml
[dependencies]
ai-lib-rust = "0.9"
```

### 直接依赖子 Crate

如果只需要执行层能力（更小的依赖图）：

```toml
[dependencies]
ai-lib-core = "0.9"
```

如果需要策略层能力：

```toml
[dependencies]
ai-lib-contact = "0.9"
```

### Feature Flags

```toml
[dependencies]
# 精简核心（默认）
ai-lib-rust = "0.9"

# 启用特定能力
ai-lib-rust = { version = "0.9", features = ["embeddings", "telemetry"] }

# 启用全部能力
ai-lib-rust = { version = "0.9", features = ["full"] }
```

**执行层 Features**（ai-lib-core）：
- `embeddings` — 嵌入向量生成
- `mcp` — MCP 工具桥接
- `computer_use` — Computer Use 抽象
- `multimodal` — 扩展多模态支持
- `reasoning` — 推理/思维链支持
- `stt` / `tts` — 语音识别/合成
- `reranking` — 重排序

**策略层 Features**（ai-lib-contact）：
- `batch` — 批处理执行
- `cache` — 缓存管理
- `routing_mvp` — 模型路由
- `guardrails` — 输入/输出守卫
- `tokens` — Token 计数与成本估算
- `telemetry` — 遥测与可观测性
- `interceptors` — 调用拦截器

## 🚀 快速开始

### 基本用法

```rust
use ai_lib_rust::{AiClient, Message};

#[tokio::main]
async fn main() -> ai_lib_rust::Result<()> {
    // 协议驱动：支持 ai-protocol manifest 中定义的任何 provider
    let client = AiClient::new("anthropic/claude-3-5-sonnet").await?;
    
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello!"),
    ];
    
    // 非流式调用
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

### 流式响应

```rust
use ai_lib_rust::{AiClient, Message};
use ai_lib_rust::types::events::StreamingEvent;
use futures::StreamExt;

#[tokio::main]
async fn main() -> ai_lib_rust::Result<()> {
    let client = AiClient::new("openai/gpt-4o").await?;
    
    let mut stream = client
        .chat()
        .messages(vec![Message::user("讲一个笑话")])
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

### 跨任务共享客户端

`AiClient` 故意不实现 `Clone`（API 密钥合规），使用 `Arc` 共享：

```rust
use std::sync::Arc;

let client = Arc::new(AiClient::new("deepseek/deepseek-chat").await?);

// 传递给多个异步任务
let handle = tokio::spawn({
    let c = Arc::clone(&client);
    async move {
        c.chat().messages(vec![Message::user("Hi")]).execute().await
    }
});
```

## 🔧 配置

### 协议 Manifest 搜索路径

运行时按以下顺序查找协议配置：

1. `ProtocolLoader::with_base_path()` 设置的自定义路径
2. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH` 环境变量
3. 常见开发路径：`ai-protocol/`、`../ai-protocol/`、`../../ai-protocol/`
4. 最终兜底：GitHub raw `ailib-official/ai-protocol`

### API 密钥

**推荐方式**（生产环境）：

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
```

**可选方式**（本地开发）：操作系统密钥环（macOS Keychain / Windows Credential Manager / Linux Secret Service）

### 生产环境配置

```bash
# 代理
export AI_PROXY_URL="http://user:pass@host:port"

# 超时
export AI_HTTP_TIMEOUT_SECS=30

# 并发限制
export AI_LIB_MAX_INFLIGHT=10

# 速率限制
export AI_LIB_RPS=5  # 或 AI_LIB_RPM=300

# 熔断器
export AI_LIB_BREAKER_FAILURE_THRESHOLD=5
export AI_LIB_BREAKER_COOLDOWN_SECS=30
```

## 🧪 测试

### 单元测试

```bash
cargo test
```

### 兼容性测试（跨运行时一致性）

```bash
# 默认运行
cargo test --test compliance

# 指定兼容性测试目录
COMPLIANCE_DIR=../ai-protocol/tests/compliance cargo test --test compliance

# 从 ai-lib-core 运行（共享测试套件）
COMPLIANCE_DIR=../ai-protocol/tests/compliance cargo test -p ai-lib-core --test compliance_from_core
```

### WASM 集成测试

```bash
# 构建 WASM
cargo build -p ai-lib-wasm --target wasm32-wasip1 --release

# 运行 wasmtime 测试
cargo test -p ai-lib-wasmtime-harness --test wasm_compliance
```

### 使用 Mock 服务器测试

```bash
# 启动 ai-protocol-mock
docker-compose up -d

# 使用 Mock 运行测试
MOCK_HTTP_URL=http://localhost:4010 cargo test -- --ignored --nocapture
```

## 🌐 WASM 支持

`ai-lib-wasm` 提供服务器端 WASM 支持（wasmtime、Wasmer 等）：

```bash
# 构建 WASM 二进制
cargo build -p ai-lib-wasm --target wasm32-wasip1 --release

# 输出：target/wasm32-wasip1/release/ai_lib_wasm.wasm (~1.2 MB)
```

**WASM 导出函数**：
- `chat_sync` — 同步聊天
- `chat_stream_init` / `chat_stream_poll` / `chat_stream_end` — 流式聊天
- `embed_sync` — 同步嵌入

**限制**：WASM 版本仅包含 E 层能力，不含缓存、路由、遥测等 P 层功能。

## 📊 可观测性

### 调用统计

```rust
let (response, stats) = client.call_model_with_stats(request).await?;
println!("request_id: {}", stats.client_request_id);
println!("latency_ms: {:?}", stats.latency_ms);
```

### 遥测反馈（opt-in）

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

## 🔄 错误码（V2 规范）

所有 Provider 错误归一化为 13 个标准错误码：

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

## 🤝 社区与贡献

### 适用场景

- **服务端应用** — 使用 `ai-lib-rust`（门面）或直接依赖 `ai-lib-contact` 获得完整能力
- **边缘计算/嵌入式** — 使用 `ai-lib-core` 获得最小依赖和确定性执行
- **浏览器/WASM** — 使用 `ai-lib-wasm` 在 WebAssembly 环境运行

### 贡献指南

1. 所有协议配置遵循 AI-Protocol 规范（v1.5 / V2）
2. 新功能需包含测试，兼容性测试必须通过
3. 代码通过 `cargo clippy` 检查
4. 遵循 [Rust API 设计原则](https://rust-lang.github.io/api-guidelines/)

### 行为准则

- 尊重所有贡献者
- 欢迎不同背景的参与者
- 专注于技术讨论，避免人身攻击
- 发现问题请通过 GitHub Issues 反馈

## 🔗 相关项目

| 项目 | 说明 |
|------|------|
| [AI-Protocol](https://github.com/ailib-official/ai-protocol) | 协议规范（v1.5 / V2） |
| [ai-lib-python](https://github.com/ailib-official/ai-lib-python) | Python 运行时 |
| [ai-lib-ts](https://github.com/ailib-official/ai-lib-ts) | TypeScript 运行时 |
| [ai-lib-go](https://github.com/ailib-official/ai-lib-go) | Go 运行时 |
| [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock) | Mock 服务器 |

## 📄 许可证

本项目采用双许可证：

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

您可任选其一。

---

**ai-lib-rust** — 协议与性能的完美结合 🚀
