use ai_lib_core::types::message::{ContentBlock, Message, MessageContent, MessageRole};

use super::budget::{ContextBudget, ModelCapacity};
use super::envelope::{AssembleStrategy, ContextLayer, MessageChunk};
use super::error::AssembleError;
use super::token_estimate::estimate_message_tokens;

/// Options for deterministic context assembly (no LLM summarization).
#[derive(Debug, Clone)]
pub struct AssembleOptions {
    pub budget: ContextBudget,
    pub capacity: ModelCapacity,
    /// Replace tool payloads larger than this (chars) with `tool_placeholder`.
    pub tool_fold_threshold_chars: usize,
    pub tool_placeholder: String,
}

impl Default for AssembleOptions {
    fn default() -> Self {
        Self {
            budget: ContextBudget::from_capacity(ModelCapacity::UNKNOWN, 2),
            capacity: ModelCapacity::UNKNOWN,
            tool_fold_threshold_chars: 8_192,
            tool_placeholder: "[tool output truncated]".to_string(),
        }
    }
}

/// Options for layer-aware envelope assembly (CR-L1-001).
#[derive(Debug, Clone)]
pub struct LayeredAssembleOptions {
    pub budget: ContextBudget,
    pub strategy: AssembleStrategy,
    pub tool_fold_threshold_chars: usize,
    pub tool_placeholder: String,
}

impl Default for LayeredAssembleOptions {
    fn default() -> Self {
        Self {
            budget: ContextBudget::from_capacity(ModelCapacity::UNKNOWN, 2),
            strategy: AssembleStrategy::Chat,
            tool_fold_threshold_chars: 8_192,
            tool_placeholder: "[tool output truncated]".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AssembleReport {
    pub messages: Vec<Message>,
    /// Flat path: suffix start index. Layered path: count of omitted chunks.
    pub dropped_prefix: usize,
    pub folded_tool_segments: usize,
}

pub struct MessageAssembler;

impl MessageAssembler {
    pub fn assemble(
        messages: &[Message],
        options: &AssembleOptions,
    ) -> Result<AssembleReport, AssembleError> {
        if messages.is_empty() {
            return Err(AssembleError::EmptyInput);
        }

        let mut working: Vec<Message> = messages.to_vec();
        let folded_tool_segments = fold_oversized_tool_content(
            &mut working,
            options.tool_fold_threshold_chars,
            &options.tool_placeholder,
        );

        let budget = options.budget.max_input_tokens;
        let min_tail = options.budget.min_tail_messages;
        let start = select_suffix_start(&working, budget, min_tail);
        let dropped_prefix = start;

        Ok(AssembleReport {
            messages: working[start..].to_vec(),
            dropped_prefix,
            folded_tool_segments,
        })
    }

    /// News-style Layer 0–5 fill. Archive (L5) is never expanded into the payload.
    ///
    /// Critical layers (System+Active) must fit; otherwise [`AssembleError::HardBudgetViolation`].
    pub fn assemble_layered(
        chunks: &[MessageChunk],
        options: &LayeredAssembleOptions,
    ) -> Result<AssembleReport, AssembleError> {
        if chunks.is_empty() {
            return Err(AssembleError::EmptyInput);
        }

        let mut working: Vec<MessageChunk> = chunks.to_vec();
        let mut folded_tool_segments = 0usize;
        for chunk in &mut working {
            folded_tool_segments += fold_oversized_tool_content(
                std::slice::from_mut(&mut chunk.message),
                options.tool_fold_threshold_chars,
                &options.tool_placeholder,
            );
        }

        let budget = options.budget.max_input_tokens;
        let critical: Vec<&MessageChunk> =
            working.iter().filter(|c| c.layer.is_critical()).collect();
        let critical_tokens: u32 = critical
            .iter()
            .map(|c| estimate_message_tokens(&c.message))
            .sum();
        if critical_tokens > budget {
            return Err(AssembleError::HardBudgetViolation {
                critical_tokens,
                budget,
            });
        }

        let mut selected: Vec<MessageChunk> = critical.into_iter().cloned().collect();
        let mut used = critical_tokens;

        // Soft layers: Relevant → Summary → Background (newest Background first when picking).
        // Archive omitted. CodeFix prefers Relevant over Summary when both compete (fill order).
        let soft_order = match options.strategy {
            AssembleStrategy::Chat | AssembleStrategy::CodeFix => [
                ContextLayer::Relevant,
                ContextLayer::Summary,
                ContextLayer::Background,
            ],
        };

        for layer in soft_order {
            let mut candidates: Vec<MessageChunk> = working
                .iter()
                .filter(|c| c.layer == layer)
                .cloned()
                .collect();

            match layer {
                ContextLayer::Background => {
                    // Newest first when deciding what to keep; output still chronological later.
                    candidates.sort_by_key(|b| std::cmp::Reverse(b.timestamp));
                }
                ContextLayer::Summary | ContextLayer::Relevant => {
                    // Prefer is_summary=true under soft pressure (kept earlier in list).
                    candidates.sort_by(|a, b| {
                        b.is_summary
                            .cmp(&a.is_summary)
                            .then_with(|| a.timestamp.cmp(&b.timestamp))
                    });
                }
                _ => {}
            }

            for chunk in candidates {
                let cost = estimate_message_tokens(&chunk.message);
                if used.saturating_add(cost) > budget {
                    continue;
                }
                used = used.saturating_add(cost);
                selected.push(chunk);
            }
        }

        let omitted = working.len().saturating_sub(selected.len());
        selected.sort_by(|a, b| {
            a.timestamp
                .cmp(&b.timestamp)
                .then_with(|| a.layer.cmp(&b.layer))
                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
        });

        Ok(AssembleReport {
            messages: selected.into_iter().map(|c| c.message).collect(),
            dropped_prefix: omitted,
            folded_tool_segments,
        })
    }
}

fn fold_oversized_tool_content(
    messages: &mut [Message],
    threshold: usize,
    placeholder: &str,
) -> usize {
    let mut folded = 0usize;

    for message in messages.iter_mut() {
        match &mut message.content {
            MessageContent::Text(text) if message.role == MessageRole::Tool => {
                if text.len() > threshold {
                    *text = placeholder.to_string();
                    folded += 1;
                }
            }
            MessageContent::Blocks(blocks) => {
                for block in blocks.iter_mut() {
                    if let ContentBlock::ToolResult { content, .. } = block {
                        let serialized = content.to_string();
                        if serialized.len() > threshold {
                            *content = serde_json::Value::String(placeholder.to_string());
                            folded += 1;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    folded
}

fn select_suffix_start(messages: &[Message], budget: u32, min_tail: usize) -> usize {
    let n = messages.len();
    if n == 0 {
        return 0;
    }

    let mut start = n;
    let mut used = 0u32;

    for i in (0..n).rev() {
        let cost = estimate_message_tokens(&messages[i]);
        let kept = n - start;

        if kept >= min_tail && start < n && used.saturating_add(cost) > budget {
            break;
        }

        if start == n && cost > budget && i + 1 == n {
            start = i;
            break;
        }

        used = used.saturating_add(cost);
        start = i;
    }

    start = trim_leading_orphan_tools(messages, start, n);
    start = extend_for_tool_chain(messages, start, n, budget);

    start.min(n)
}

fn trim_leading_orphan_tools(messages: &[Message], start: usize, end: usize) -> usize {
    let mut s = start;
    while s < end && messages[s].role == MessageRole::Tool {
        s += 1;
    }
    s
}

/// If the kept window ends with tool results, walk backward to include the initiating assistant.
fn extend_for_tool_chain(messages: &[Message], start: usize, end: usize, budget: u32) -> usize {
    if start == 0 || start >= end {
        return start;
    }

    let s = start;
    if messages[end - 1].role != MessageRole::Tool {
        return s;
    }

    let mut i = start;
    while i < end && messages[i].role == MessageRole::Tool {
        i += 1;
    }

    if i < end {
        return s;
    }

    for j in (0..start).rev() {
        if messages[j].role == MessageRole::Assistant {
            let candidate = j;
            let slice = &messages[candidate..end];
            let cost: u32 = slice.iter().map(estimate_message_tokens).sum();
            if cost <= budget {
                return candidate;
            }
            break;
        }
    }

    trim_leading_orphan_tools(messages, start, end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_lib_core::types::message::Message;

    fn opts(budget: u32, min_tail: usize) -> AssembleOptions {
        AssembleOptions {
            budget: ContextBudget::new(budget, 0, min_tail),
            ..Default::default()
        }
    }

    #[test]
    fn drops_oldest_when_over_budget() {
        let messages: Vec<Message> = (0..20)
            .map(|i| Message::user(format!("msg-{i}-{}", "x".repeat(40))))
            .collect();

        let report = MessageAssembler::assemble(&messages, &opts(120, 1)).unwrap();
        assert!(report.dropped_prefix > 0);
        assert!(!report.messages.is_empty());
        let tokens: u32 = report.messages.iter().map(estimate_message_tokens).sum();
        assert!(tokens <= 200);
    }

    #[test]
    fn keeps_minimum_tail_messages() {
        let messages = vec![
            Message::user("old"),
            Message::user("mid"),
            Message::assistant("newest"),
        ];

        let report = MessageAssembler::assemble(&messages, &opts(10, 2)).unwrap();
        assert!(report.messages.len() >= 2);
    }

    #[test]
    fn folds_oversized_tool_text() {
        let huge = "x".repeat(20_000);
        let messages = vec![
            Message::user("q"),
            Message::tool("call_1", huge),
            Message::assistant("done"),
        ];

        let report = MessageAssembler::assemble(&messages, &opts(50_000, 1)).unwrap();
        assert_eq!(report.folded_tool_segments, 1);
        let tool = report
            .messages
            .iter()
            .find(|m| m.role == MessageRole::Tool)
            .unwrap();
        if let MessageContent::Text(text) = &tool.content {
            assert_eq!(text, "[tool output truncated]");
        } else {
            panic!("expected text tool content");
        }
    }

    #[test]
    fn does_not_start_with_orphan_tool() {
        let messages = vec![
            Message::user("u1"),
            Message::assistant("a1"),
            Message::tool("call_1", "result"),
            Message::user("u2"),
            Message::assistant("a2"),
        ];

        let report = MessageAssembler::assemble(&messages, &opts(30, 1)).unwrap();
        assert_ne!(report.messages.first().unwrap().role, MessageRole::Tool);
    }

    #[test]
    fn empty_input_errors() {
        let err = MessageAssembler::assemble(&[], &opts(100, 1)).unwrap_err();
        assert_eq!(err, AssembleError::EmptyInput);
    }

    #[test]
    fn budget_from_capacity_subtracts_output_reserve() {
        let budget = ContextBudget::from_capacity(ModelCapacity::new(128_000, 8_192), 2);
        assert_eq!(budget.max_input_tokens, 119_808);
        assert_eq!(budget.reserve_output_tokens, 8_192);
    }

    #[test]
    fn token_estimate_heuristic() {
        assert_eq!(crate::context::estimate_tokens("abcd"), 1);
        assert_eq!(crate::context::estimate_tokens("abcdefgh"), 2);
    }

    fn layered_opts(budget: u32, strategy: AssembleStrategy) -> LayeredAssembleOptions {
        LayeredAssembleOptions {
            budget: ContextBudget::new(budget, 0, 1),
            strategy,
            ..Default::default()
        }
    }

    #[test]
    fn assemble_layered_empty_errors() {
        let err =
            MessageAssembler::assemble_layered(&[], &layered_opts(100, AssembleStrategy::Chat))
                .unwrap_err();
        assert_eq!(err, AssembleError::EmptyInput);
    }

    #[test]
    fn assemble_layered_hard_budget_violation() {
        let chunks = vec![
            MessageChunk::new(
                ContextLayer::System,
                1,
                Message::system("S".repeat(200)),
                "sys",
            ),
            MessageChunk::new(
                ContextLayer::Active,
                2,
                Message::user("A".repeat(200)),
                "act",
            ),
        ];
        // tokens ≫ small budget
        let err =
            MessageAssembler::assemble_layered(&chunks, &layered_opts(5, AssembleStrategy::Chat))
                .unwrap_err();
        match err {
            AssembleError::HardBudgetViolation {
                critical_tokens,
                budget,
            } => {
                assert!(critical_tokens > budget);
                assert_eq!(budget, 5);
            }
            other => panic!("expected HardBudgetViolation, got {other:?}"),
        }
    }

    #[test]
    fn assemble_layered_keeps_critical_and_drops_background_before_relevant() {
        let chunks = vec![
            MessageChunk::new(ContextLayer::System, 1, Message::system("sys"), "s"),
            MessageChunk::new(ContextLayer::Active, 2, Message::user("ask"), "a"),
            MessageChunk::new(
                ContextLayer::Relevant,
                3,
                Message::user(format!("rel-{}", "r".repeat(40))),
                "r1",
            ),
            MessageChunk::new(
                ContextLayer::Background,
                4,
                Message::user(format!("bg-{}", "b".repeat(40))),
                "b1",
            ),
            MessageChunk::new(
                ContextLayer::Archive,
                5,
                Message::user("archive-should-omit"),
                "arch",
            ),
        ];
        // Budget fits critical + one soft chunk roughly; Relevant is filled before Background.
        let report = MessageAssembler::assemble_layered(
            &chunks,
            &layered_opts(40, AssembleStrategy::CodeFix),
        )
        .unwrap();
        let texts: Vec<String> = report
            .messages
            .iter()
            .map(|m| match &m.content {
                MessageContent::Text(t) => t.clone(),
                _ => String::new(),
            })
            .collect();
        assert!(texts.iter().any(|t| t == "sys"));
        assert!(texts.iter().any(|t| t == "ask"));
        assert!(
            texts.iter().any(|t| t.starts_with("rel-")),
            "relevant should be preferred over background under tight budget: {texts:?}"
        );
        assert!(
            !texts.iter().any(|t| t.contains("archive")),
            "archive must not expand"
        );
    }

    #[test]
    fn assemble_layered_under_budget_includes_soft_layers() {
        let chunks = vec![
            MessageChunk::new(ContextLayer::System, 1, Message::system("sys"), "s"),
            MessageChunk::new(ContextLayer::Active, 2, Message::user("ask"), "a"),
            MessageChunk::new(ContextLayer::Relevant, 3, Message::user("rel"), "r")
                .with_summary(true),
            MessageChunk::new(ContextLayer::Background, 4, Message::user("old-bg"), "b"),
        ];
        let report = MessageAssembler::assemble_layered(
            &chunks,
            &layered_opts(10_000, AssembleStrategy::Chat),
        )
        .unwrap();
        assert_eq!(report.dropped_prefix, 0);
        assert_eq!(report.messages.len(), 4);
    }
}
