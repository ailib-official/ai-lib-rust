//! Provider-specific content block encoding for multimodal messages.
//!
//! 多模态内容块按厂商 API 形状编码（Anthropic Messages、Gemini generateContent）。

use serde_json::{json, Value};

use crate::error::Error;
use crate::protocol::ProtocolError;
use crate::types::message::{ContentBlock, DocumentSource, ImageSource};

fn validation(msg: impl Into<String>) -> Error {
    Error::Protocol(ProtocolError::ValidationError(msg.into()))
}

/// Encode unified content blocks into Anthropic Messages API `content` array items.
pub fn encode_blocks_for_anthropic(blocks: &[ContentBlock]) -> Result<Vec<Value>, Error> {
    blocks.iter().map(encode_block_for_anthropic).collect()
}

fn encode_block_for_anthropic(block: &ContentBlock) -> Result<Value, Error> {
    match block {
        ContentBlock::Text { text } => Ok(json!({ "type": "text", "text": text })),
        ContentBlock::Image { source } => Ok(json!({
            "type": "image",
            "source": encode_anthropic_media_source(source, "image")?,
        })),
        ContentBlock::Document { source } => Ok(json!({
            "type": "document",
            "source": encode_anthropic_document_source(source)?,
        })),
        ContentBlock::Audio { .. } => Err(validation(
            "Anthropic Messages driver does not encode audio content blocks yet",
        )),
        ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => Err(validation(
            "tool blocks must be encoded via Anthropic tool_use/tool_result paths",
        )),
    }
}

fn encode_anthropic_media_source(source: &ImageSource, kind: &str) -> Result<Value, Error> {
    match source.source_type.as_str() {
        "base64" => {
            let media_type = source
                .media_type
                .as_deref()
                .ok_or_else(|| validation(format!("{kind} base64 block requires media_type")))?;
            Ok(json!({
                "type": "base64",
                "media_type": media_type,
                "data": source.data,
            }))
        }
        "url" => Ok(json!({
            "type": "url",
            "url": source.data,
        })),
        other => Err(validation(format!(
            "unsupported {kind} source type: {other}"
        ))),
    }
}

fn encode_anthropic_document_source(source: &DocumentSource) -> Result<Value, Error> {
    match source.source_type.as_str() {
        "base64" => {
            let media_type = source.mime_type.as_deref().unwrap_or("application/pdf");
            Ok(json!({
                "type": "base64",
                "media_type": media_type,
                "data": source.data,
            }))
        }
        "url" => Ok(json!({
            "type": "url",
            "url": source.data,
        })),
        "ref" => Err(validation(
            "document ref must be resolved to base64 or url before sending to Anthropic",
        )),
        other => Err(validation(format!(
            "unsupported document source type: {other}"
        ))),
    }
}

/// Encode unified content blocks into Gemini `parts` array.
pub fn encode_blocks_for_gemini(blocks: &[ContentBlock]) -> Result<Value, Error> {
    let parts: Vec<Value> = blocks
        .iter()
        .map(encode_block_for_gemini)
        .collect::<Result<_, _>>()?;
    Ok(Value::Array(parts))
}

fn encode_block_for_gemini(block: &ContentBlock) -> Result<Value, Error> {
    match block {
        ContentBlock::Text { text } => Ok(json!({ "text": text })),
        ContentBlock::Image { source } => encode_gemini_inline_data(source, "image"),
        ContentBlock::Document { source } => encode_gemini_document_inline(source),
        ContentBlock::Audio { .. } => Err(validation(
            "Gemini generateContent driver does not encode audio content blocks yet",
        )),
        ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => Err(validation(
            "tool blocks must be encoded via Gemini functionCall/functionResponse paths",
        )),
    }
}

fn encode_gemini_inline_data(source: &ImageSource, kind: &str) -> Result<Value, Error> {
    if source.source_type != "base64" {
        return Err(validation(format!(
            "Gemini {kind} blocks require base64 inline data (got {})",
            source.source_type
        )));
    }
    let mime_type = source
        .media_type
        .as_deref()
        .ok_or_else(|| validation(format!("{kind} base64 block requires media_type")))?;
    Ok(json!({
        "inlineData": {
            "mimeType": mime_type,
            "data": source.data,
        }
    }))
}

fn encode_gemini_document_inline(source: &DocumentSource) -> Result<Value, Error> {
    if source.source_type != "base64" {
        return Err(validation(
            "Gemini document blocks require base64 inline data; resolve ref before send",
        ));
    }
    let mime_type = source.mime_type.as_deref().unwrap_or("application/pdf");
    Ok(json!({
        "inlineData": {
            "mimeType": mime_type,
            "data": source.data,
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::ContentBlock;

    const PDF_B64: &str = "JVBERi0xLjQK";

    #[test]
    fn anthropic_document_base64_shape() {
        let blocks = vec![ContentBlock::document_base64(
            PDF_B64.into(),
            Some("application/pdf".into()),
            Some("paper.pdf".into()),
        )];
        let encoded = encode_blocks_for_anthropic(&blocks).unwrap();
        assert_eq!(encoded[0]["type"], "document");
        assert_eq!(encoded[0]["source"]["type"], "base64");
        assert_eq!(encoded[0]["source"]["media_type"], "application/pdf");
        assert_eq!(encoded[0]["source"]["data"], PDF_B64);
    }

    #[test]
    fn anthropic_document_ref_rejected() {
        let blocks = vec![ContentBlock::document_ref(
            "upload://abc".into(),
            Some("application/pdf".into()),
            Some("paper.pdf".into()),
        )];
        assert!(encode_blocks_for_anthropic(&blocks).is_err());
    }

    #[test]
    fn gemini_document_inline_data_shape() {
        let blocks = vec![
            ContentBlock::text("Summarize"),
            ContentBlock::document_base64(PDF_B64.into(), Some("application/pdf".into()), None),
        ];
        let parts = encode_blocks_for_gemini(&blocks).unwrap();
        let arr = parts.as_array().unwrap();
        assert_eq!(arr[0]["text"], "Summarize");
        assert_eq!(arr[1]["inlineData"]["mimeType"], "application/pdf");
        assert_eq!(arr[1]["inlineData"]["data"], PDF_B64);
    }

    #[test]
    fn gemini_document_ref_rejected() {
        let blocks = vec![ContentBlock::document_ref(
            "upload://abc".into(),
            Some("application/pdf".into()),
            None,
        )];
        assert!(encode_blocks_for_gemini(&blocks).is_err());
    }
}
