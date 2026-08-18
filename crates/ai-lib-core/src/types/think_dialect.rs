//! Experimental opt-in think-dialect splitter (ALR-RSN-002).
//!
//! 中文：可选正文思考方言剥离（标签 / 围栏 / 孤立闭合）；**默认不进 Client 热路径**。
//!
//! Wire-level `ThinkingDelta` / `reasoning_content` remains the canonical channel
//! (ALR-RSN-001). This helper only cleans dialects that leaked into `content`.
//!
//! **GOV-007**: separate from [`crate::types::text_tool::TextToolParser`] — do not
//! unify into one dialect parser. Thin span-scan reuse only if justified later.
//!
//! **GOV-006**: Experimental Public API — hosts call explicitly; Client does not
//! invoke this unless an opt-in hook is added later.

use regex::Regex;
use std::sync::OnceLock;

/// Result of splitting leaked think dialects from assistant text.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ThinkSplit {
    pub thinking: String,
    pub content: String,
}

fn think_stem_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)^(?:redacted[_-]?)?(?:think(?:ing)?|reason(?:ing)?|thoughts?|reflection|scratch(?:pad)?|analysis|chain[_-]?of[_-]?thought|cot)$",
        )
        .expect("think stem regex")
    })
}

fn open_tag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<\s*([A-Za-z][\w:.-]*)\b[^>]*>").expect("open tag"))
}

fn close_tag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<\s*/\s*([A-Za-z][\w:.-]*)\s*>").expect("close tag"))
}

fn fence_open_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // No lookaround — `regex` crate is RE2-style.
        Regex::new(r"(?m)^[ \t]*```[ \t]*([A-Za-z][\w-]*)[ \t]*$").expect("fence open")
    })
}

/// Returns true if `name` is a known think/reasoning dialect stem.
pub fn is_think_dialect_name(name: &str) -> bool {
    let n = name.trim();
    if n.is_empty() {
        return false;
    }
    let bare = n.rsplit(':').next().unwrap_or(n);
    think_stem_re().is_match(bare)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkerKind {
    Open,
    Close,
    FenceOpen,
    FenceClose,
}

#[derive(Debug, Clone)]
struct Marker {
    kind: MarkerKind,
    index: usize,
    end: usize,
    name: String,
}

/// Match a lone ``` line (optional leading whitespace) starting at `from` or after a newline.
fn find_fence_close_line(src: &str, from: usize) -> Option<(usize, usize)> {
    let bytes = src.as_bytes();
    let mut i = from;
    while i < src.len() {
        let line_start = i;
        // Skip optional spaces/tabs
        while i < src.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
            i += 1;
        }
        if i + 3 <= src.len() && &src[i..i + 3] == "```" {
            let mut j = i + 3;
            while j < src.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            if j == src.len() || bytes[j] == b'\n' || bytes[j] == b'\r' {
                return Some((line_start, j));
            }
        }
        // Advance to next line
        if let Some(nl) = src[i..].find('\n') {
            i = i + nl + 1;
        } else {
            break;
        }
    }
    None
}

fn find_next_marker(src: &str, from: usize) -> Option<Marker> {
    let slice = &src[from..];
    let mut cands: Vec<Marker> = Vec::new();

    for caps in open_tag_re().captures_iter(slice) {
        let m = caps.get(0)?;
        let name = caps.get(1)?.as_str();
        if is_think_dialect_name(name) {
            cands.push(Marker {
                kind: MarkerKind::Open,
                index: from + m.start(),
                end: from + m.end(),
                name: name.to_string(),
            });
            break;
        }
    }

    for caps in close_tag_re().captures_iter(slice) {
        let m = caps.get(0)?;
        let name = caps.get(1)?.as_str();
        if is_think_dialect_name(name) {
            cands.push(Marker {
                kind: MarkerKind::Close,
                index: from + m.start(),
                end: from + m.end(),
                name: name.to_string(),
            });
            break;
        }
    }

    for caps in fence_open_re().captures_iter(slice) {
        let m = caps.get(0)?;
        let name = caps.get(1)?.as_str();
        if is_think_dialect_name(name) {
            cands.push(Marker {
                kind: MarkerKind::FenceOpen,
                index: from + m.start(),
                end: from + m.end(),
                name: name.to_string(),
            });
            break;
        }
    }

    if let Some((index, end)) = find_fence_close_line(src, from) {
        cands.push(Marker {
            kind: MarkerKind::FenceClose,
            index,
            end,
            name: String::new(),
        });
    }

    cands.sort_by(|a, b| a.index.cmp(&b.index).then(a.end.cmp(&b.end)));
    cands.into_iter().next()
}

fn find_closing_tag(src: &str, from: usize, open_name: &str) -> Option<(usize, usize)> {
    let open_lc = open_name.to_ascii_lowercase();
    let mut any: Option<(usize, usize)> = None;
    for caps in close_tag_re().captures_iter(&src[from..]) {
        let m = caps.get(0)?;
        let name = caps.get(1)?.as_str();
        if !is_think_dialect_name(name) {
            continue;
        }
        let index = from + m.start();
        let end = from + m.end();
        if name.to_ascii_lowercase() == open_lc {
            return Some((index, end));
        }
        if any.is_none() {
            any = Some((index, end));
        }
    }
    any
}

fn find_fence_close(src: &str, from: usize) -> Option<(usize, usize)> {
    find_fence_close_line(src, from)
}

fn append_thinking(thinking: &mut String, chunk: &str) {
    let t = chunk;
    if t.is_empty() {
        return;
    }
    if !thinking.is_empty() {
        thinking.push_str("\n\n");
    }
    thinking.push_str(t);
}

/// Split leaked think dialects from assistant text into thinking + visible content.
///
/// Experimental helper — **not** called by default Client stream/non-stream paths.
pub fn split_think_blocks(text: &str) -> ThinkSplit {
    let src = text;
    if src.is_empty() {
        return ThinkSplit::default();
    }

    let mut thinking = String::new();
    let mut content = String::new();
    let mut i = 0usize;

    while i < src.len() {
        let Some(marker) = find_next_marker(src, i) else {
            content.push_str(&src[i..]);
            break;
        };

        match marker.kind {
            MarkerKind::Open => {
                content.push_str(&src[i..marker.index]);
                if let Some((ci, ce)) = find_closing_tag(src, marker.end, &marker.name) {
                    append_thinking(&mut thinking, &src[marker.end..ci]);
                    i = ce;
                } else {
                    append_thinking(&mut thinking, &src[marker.end..]);
                    i = src.len();
                }
            }
            MarkerKind::FenceOpen => {
                content.push_str(&src[i..marker.index]);
                if let Some((ci, ce)) = find_fence_close(src, marker.end) {
                    append_thinking(&mut thinking, &src[marker.end..ci]);
                    i = ce;
                } else {
                    append_thinking(&mut thinking, &src[marker.end..]);
                    i = src.len();
                }
            }
            MarkerKind::Close => {
                append_thinking(&mut thinking, &src[i..marker.index]);
                i = marker.end;
            }
            MarkerKind::FenceClose => {
                content.push_str(&src[i..marker.end]);
                i = marker.end;
            }
        }
    }

    ThinkSplit {
        thinking: thinking.trim().to_string(),
        content: content
            .trim_start_matches(['\n', '\r', ' ', '\t'])
            .trim()
            .to_string(),
    }
}

/// Remove all think dialects; keep visible answer only.
pub fn strip_think_dialects(text: &str) -> String {
    split_think_blocks(text).content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_names() {
        assert!(is_think_dialect_name("think"));
        assert!(is_think_dialect_name("thinking"));
        assert!(is_think_dialect_name("redacted_reasoning"));
        assert!(is_think_dialect_name("ns:think"));
        assert!(!is_think_dialect_name("body"));
        assert!(!is_think_dialect_name("tool_call"));
    }

    #[test]
    fn plain_text() {
        let r = split_think_blocks("hello");
        assert_eq!(r.thinking, "");
        assert_eq!(r.content, "hello");
    }

    #[test]
    fn paired_think_tags() {
        let r = split_think_blocks("<think>plan A</think>\nFinal answer");
        assert_eq!(r.thinking, "plan A");
        assert_eq!(r.content, "Final answer");
    }

    #[test]
    fn unclosed_think() {
        let r = split_think_blocks("<Think attr=\"x\">only think");
        assert_eq!(r.thinking, "only think");
        assert_eq!(r.content, "");
    }

    #[test]
    fn orphan_close() {
        let r = split_think_blocks("pre\n</think>\nbody");
        assert_eq!(r.thinking, "pre");
        assert_eq!(r.content, "body");
    }

    #[test]
    fn thinking_tag() {
        let r = split_think_blocks("<thinking>x</thinking>y");
        assert_eq!(r.thinking, "x");
        assert_eq!(r.content, "y");
    }

    #[test]
    fn redacted_reasoning() {
        let r = split_think_blocks("<redacted_reasoning>hidden</redacted_reasoning>\nVisible");
        assert_eq!(r.thinking, "hidden");
        assert_eq!(r.content, "Visible");
    }

    #[test]
    fn namespaced_reasoning() {
        let r = split_think_blocks("<ns:reasoning>z</ns:reasoning>\nout");
        assert_eq!(r.thinking, "z");
        assert_eq!(r.content, "out");
    }

    #[test]
    fn thinking_fence() {
        let r = split_think_blocks("```thinking\nstep1\n```\nAnswer");
        assert_eq!(r.thinking, "step1");
        assert_eq!(r.content, "Answer");
    }

    #[test]
    fn chain_of_thought_fence() {
        let r = split_think_blocks("```chain-of-thought\na\nb\n```\nC");
        assert_eq!(r.thinking, "a\nb");
        assert_eq!(r.content, "C");
    }

    #[test]
    fn analysis_not_body() {
        let r = split_think_blocks("<analysis>a1</analysis>\n<body>nope</body>\nOK");
        assert_eq!(r.thinking, "a1");
        assert!(r.content.contains("<body>nope</body>"));
        assert!(r.content.contains("OK"));
    }

    #[test]
    fn strip_helper() {
        assert_eq!(strip_think_dialects("<think>x</think>\nY"), "Y");
    }
}
