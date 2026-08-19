//! Soft-layer promotion heuristics before layered assembly (EOS-CX-LAYER23-001 Phase A).
//!
//! Promotes Background → Relevant for recent pre-active turns and tool outputs that
//! overlap the current user query. Does not fork assembly — called from
//! [`super::MessageAssembler::assemble_layered`] only.

use std::collections::HashSet;

use ai_lib_core::types::message::{Message, MessageContent, MessageRole};

use super::envelope::{ContextLayer, MessageChunk};

/// Heuristics applied before soft-layer fill (Phase A).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftLayerPromotionOptions {
    /// Pre-active user turns to promote from Background → Relevant. `0` disables.
    pub recent_turns: u8,
    /// Minimum shared normalized tokens to promote a tool chunk. `0` disables overlap pass.
    pub tool_token_overlap_min: u8,
}

impl Default for SoftLayerPromotionOptions {
    fn default() -> Self {
        Self {
            recent_turns: 3,
            tool_token_overlap_min: 2,
        }
    }
}

impl SoftLayerPromotionOptions {
    pub const fn disabled() -> Self {
        Self {
            recent_turns: 0,
            tool_token_overlap_min: 0,
        }
    }

    pub fn is_enabled(self) -> bool {
        self.recent_turns > 0 || self.tool_token_overlap_min > 0
    }
}

/// Mutates `chunks` in place: never demotes; skips critical / archive layers.
pub fn promote_soft_layers(chunks: &mut [MessageChunk], opts: SoftLayerPromotionOptions) {
    if chunks.is_empty() || !opts.is_enabled() {
        return;
    }
    let Some(last_user) = chunks
        .iter()
        .rposition(|c| c.message.role == MessageRole::User)
    else {
        return;
    };

    if opts.recent_turns > 0 {
        promote_recent_pre_active_turns(chunks, last_user, opts.recent_turns);
    }
    if opts.tool_token_overlap_min > 0 {
        let query = message_plain_text(&chunks[last_user].message);
        let query_tokens = normalized_tokens(&query);
        if !query_tokens.is_empty() {
            promote_tool_overlap(chunks, &query_tokens, opts.tool_token_overlap_min as usize);
        }
    }
}

fn promote_recent_pre_active_turns(chunks: &mut [MessageChunk], last_user: usize, n: u8) {
    if last_user == 0 || n == 0 {
        return;
    }
    let mut turn_count = 0u8;
    let mut i = last_user;
    while i > 0 {
        i -= 1;
        if chunks[i].message.role == MessageRole::User {
            turn_count = turn_count.saturating_add(1);
        }
        if turn_count > 0 && turn_count <= n {
            promote_background_to_relevant(&mut chunks[i]);
        }
        if turn_count > n {
            break;
        }
    }
}

fn promote_tool_overlap(
    chunks: &mut [MessageChunk],
    query_tokens: &HashSet<String>,
    min_overlap: usize,
) {
    for chunk in chunks.iter_mut() {
        if chunk.message.role != MessageRole::Tool {
            continue;
        }
        let text = message_plain_text(&chunk.message);
        let overlap = normalized_tokens(&text).intersection(query_tokens).count();
        if overlap >= min_overlap {
            promote_background_to_relevant(chunk);
        }
    }
}

fn promote_background_to_relevant(chunk: &mut MessageChunk) {
    if chunk.layer == ContextLayer::Background {
        chunk.layer = ContextLayer::Relevant;
    }
}

fn message_plain_text(message: &Message) -> String {
    match &message.content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                ai_lib_core::types::message::ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn normalized_tokens(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_lib_core::types::message::Message;

    fn bg_user(content: &str, ts: u64) -> MessageChunk {
        MessageChunk::new(
            ContextLayer::Background,
            ts,
            Message::user(content),
            format!("u{ts}"),
        )
    }

    fn bg_tool(content: &str, ts: u64) -> MessageChunk {
        MessageChunk::new(
            ContextLayer::Background,
            ts,
            Message::tool(format!("call-{ts}"), content),
            format!("t{ts}"),
        )
    }

    #[test]
    fn promotes_last_three_pre_active_turns() {
        let mut chunks = vec![
            MessageChunk::new(ContextLayer::System, 0, Message::system("sys"), "s"),
            bg_user("turn1", 1),
            MessageChunk::new(
                ContextLayer::Background,
                2,
                Message::assistant("turn1-a"),
                "a1",
            ),
            bg_user("turn2", 3),
            MessageChunk::new(
                ContextLayer::Background,
                4,
                Message::assistant("turn2-a"),
                "a2",
            ),
            bg_user("turn3", 5),
            MessageChunk::new(
                ContextLayer::Background,
                6,
                Message::assistant("turn3-a"),
                "a3",
            ),
            bg_user("turn4-old", 7),
            MessageChunk::new(ContextLayer::Active, 8, Message::user("now"), "active"),
        ];
        promote_soft_layers(
            &mut chunks,
            SoftLayerPromotionOptions {
                recent_turns: 3,
                tool_token_overlap_min: 0,
            },
        );
        assert_eq!(chunks[1].layer, ContextLayer::Background, "turn1 too old");
        assert_eq!(chunks[7].layer, ContextLayer::Relevant, "turn4 promoted");
        assert_eq!(chunks[6].layer, ContextLayer::Relevant);
        assert_eq!(chunks[8].layer, ContextLayer::Active);
    }

    #[test]
    fn promotes_tool_with_query_overlap() {
        let mut chunks = vec![
            MessageChunk::new(ContextLayer::System, 0, Message::system("sys"), "s"),
            bg_tool("weather api returned rainfall data", 1),
            MessageChunk::new(
                ContextLayer::Active,
                2,
                Message::user("what rainfall data do we have?"),
                "q",
            ),
        ];
        promote_soft_layers(
            &mut chunks,
            SoftLayerPromotionOptions {
                recent_turns: 0,
                tool_token_overlap_min: 2,
            },
        );
        assert_eq!(chunks[1].layer, ContextLayer::Relevant);
    }
}
