//! Conformance helpers for ai-protocol compliance YAML runners.
//! 合规测试辅助：同步 SSE 解码与 event_map 映射。

use crate::pipeline::event_map::RuleBasedEventMapper;
use crate::pipeline::{decode::SseDecoder, Decoder, PipelineError};
use crate::protocol::{DecoderConfig, EventMapRule};
use crate::types::events::StreamingEvent;
use crate::utils::{JsonPathEvaluator, PathMapper};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;
use std::collections::HashMap;

/// Decode raw SSE chunks synchronously using the production `SseDecoder`.
pub fn decode_sse_chunks_sync(
    chunks: &[String],
    prefix: &str,
    done_signal: &str,
) -> Result<(usize, bool), PipelineError> {
    let mut done_received = false;
    for chunk in chunks {
        for line in chunk.lines() {
            if !line.starts_with(prefix) {
                continue;
            }
            let payload = line[prefix.len()..].trim();
            if payload == done_signal {
                done_received = true;
            }
        }
    }

    let cfg = DecoderConfig {
        format: "sse".to_string(),
        strategy: None,
        delimiter: None,
        prefix: Some(prefix.to_string()),
        done_signal: Some(done_signal.to_string()),
    };
    let decoder = SseDecoder::from_config(&cfg)?;
    let body = chunks.join("");
    let input = futures::stream::iter(vec![Ok(Bytes::from(body))]);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PipelineError::Decoder(e.to_string()))?;

    let mut decoded = rt
        .block_on(decoder.decode_stream(Box::pin(input)))
        .map_err(|e| PipelineError::Decoder(e.to_string()))?;
    let mut frame_count = 0usize;
    while let Some(item) = rt.block_on(decoded.next()) {
        item.map_err(|e| PipelineError::Decoder(e.to_string()))?;
        frame_count += 1;
    }
    Ok((frame_count, done_received))
}

/// Map a decoded frame to compliance YAML events using manifest `event_map` rules.
pub fn map_frame_to_compliance_events(
    frame: &YamlValue,
    rules: &[EventMapRule],
) -> Result<Vec<YamlValue>, PipelineError> {
    let json_frame: JsonValue = serde_json::to_value(frame)
        .map_err(|e| PipelineError::EventMapper(format!("frame yaml→json: {e}")))?;

    let mapper = RuleBasedEventMapper::new(rules)?;
    let mut out = Vec::new();

    for rule in rules {
        let matcher = JsonPathEvaluator::new(&rule.match_expr).map_err(|e| {
            PipelineError::InvalidJsonPath {
                path: rule.match_expr.clone(),
                error: e.to_string(),
                hint: None,
            }
        })?;
        if !matcher.matches(&json_frame) {
            continue;
        }

        if rule.emit == "PartialToolCall" {
            if let Some(fields) = &rule.fields {
                if let Some(path) = fields.get("tool_calls") {
                    if let Some(tc) = PathMapper::get_path(&json_frame, path) {
                        let mut event = HashMap::new();
                        event.insert(
                            "type".to_string(),
                            YamlValue::String("PartialToolCall".to_string()),
                        );
                        event.insert(
                            "tool_calls".to_string(),
                            serde_yaml::to_value(tc).map_err(|e| {
                                PipelineError::EventMapper(format!("tool_calls: {e}"))
                            })?,
                        );
                        out.push(
                            serde_yaml::to_value(event).map_err(|e| {
                                PipelineError::EventMapper(format!("yaml event: {e}"))
                            })?,
                        );
                        continue;
                    }
                }
            }
        }

        let extract: Vec<(String, String)> = rule
            .fields
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        if let Some(ev) = mapper.build_event_for_frame(&rule.emit, &json_frame, &extract) {
            if let Some(yaml_ev) = streaming_event_to_compliance_yaml(&ev, &json_frame, rule) {
                out.push(yaml_ev);
            }
        }
    }

    Ok(out)
}

fn streaming_event_to_compliance_yaml(
    ev: &StreamingEvent,
    frame: &JsonValue,
    rule: &EventMapRule,
) -> Option<YamlValue> {
    match ev {
        StreamingEvent::PartialContentDelta { content, .. } => {
            let mut event = HashMap::new();
            event.insert(
                "type".to_string(),
                YamlValue::String("PartialContentDelta".to_string()),
            );
            event.insert("content".to_string(), YamlValue::String(content.clone()));
            serde_yaml::to_value(event).ok()
        }
        StreamingEvent::StreamEnd { finish_reason } => {
            let mut event = HashMap::new();
            event.insert(
                "type".to_string(),
                YamlValue::String("StreamEnd".to_string()),
            );
            if let Some(fr) = finish_reason {
                event.insert("finish_reason".to_string(), YamlValue::String(fr.clone()));
            } else if let Some(fields) = &rule.fields {
                if let Some(path) = fields.get("finish_reason") {
                    if let Some(fr) = PathMapper::get_path(frame, path) {
                        event.insert("finish_reason".to_string(), serde_yaml::to_value(fr).ok()?);
                    }
                }
            }
            serde_yaml::to_value(event).ok()
        }
        _ => None,
    }
}

/// Parse `event_map` rules from a compliance YAML input value (`extract` or `fields` keys).
pub fn event_map_rules_from_yaml(value: &YamlValue) -> Result<Vec<EventMapRule>, PipelineError> {
    let Some(seq) = value.as_sequence() else {
        return Ok(Vec::new());
    };

    let mut rules = Vec::with_capacity(seq.len());
    for item in seq {
        let Some(map) = item.as_mapping() else {
            continue;
        };
        let match_expr = map
            .get(YamlValue::String("match".to_string()))
            .and_then(YamlValue::as_str)
            .unwrap_or_default()
            .to_string();
        let emit = map
            .get(YamlValue::String("emit".to_string()))
            .and_then(YamlValue::as_str)
            .unwrap_or_default()
            .to_string();
        let fields_src = map
            .get(YamlValue::String("extract".to_string()))
            .or_else(|| map.get(YamlValue::String("fields".to_string())));
        let fields = fields_src.and_then(|v| {
            let m = v.as_mapping()?;
            let mut out = HashMap::new();
            for (k, v) in m {
                let key = k.as_str()?.to_string();
                let val = v.as_str()?.to_string();
                out.insert(key, val);
            }
            Some(out)
        });
        rules.push(EventMapRule {
            match_expr,
            emit,
            fields,
        });
    }
    Ok(rules)
}
