//! 从 `metadata.models` 读取 Experimental 模型级能力事实（ME-001 / ALR-ME-001）。
//!
//! # Experimental model-level capability facts (ME-001)
//!
//! Parses `metadata.models.<id>` entries for optional `model_capabilities` and
//! `modalities`. Omitted boolean fields mean **unknown** — never coerce to
//! `false`. When known facts exist for a model id, prefer them over provider
//! `capabilities.required` / `optional` advertisements.

use serde::Deserialize;
use serde_json::Value;

/// Tri-state support: omitted / unknown must not act like false.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityKnown {
    Yes,
    No,
    Unknown,
}

impl CapabilityKnown {
    pub fn from_option(v: Option<bool>) -> Self {
        match v {
            Some(true) => Self::Yes,
            Some(false) => Self::No,
            None => Self::Unknown,
        }
    }

    /// Prefer known model fact; otherwise fall back to provider-level boolean.
    pub fn or_provider(self, provider_allows: bool) -> bool {
        match self {
            Self::Yes => true,
            Self::No => false,
            Self::Unknown => provider_allows,
        }
    }
}

/// Experimental per-model capability booleans (`Option` = omit = unknown).
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct ModelCapabilityFacts {
    #[serde(default)]
    pub tool_call: Option<bool>,
    #[serde(default)]
    pub structured_output: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<bool>,
    #[serde(default)]
    pub attachment: Option<bool>,
}

/// Experimental per-model modalities.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct ModelModalities {
    #[serde(default)]
    pub input: Option<Vec<String>>,
    #[serde(default)]
    pub output: Option<Vec<String>>,
}

/// Subset of `metadata-model-entry.json` used by runtime capability checks.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct MetadataModelEntry {
    #[serde(default)]
    pub model_capabilities: Option<ModelCapabilityFacts>,
    #[serde(default)]
    pub modalities: Option<ModelModalities>,
}

impl MetadataModelEntry {
    pub fn from_value(value: &Value) -> Option<Self> {
        serde_json::from_value(value.clone()).ok()
    }

    /// Whether this model accepts a given input modality tag (`image`, `audio`, …).
    ///
    /// Preference: `modalities.input` when present; else `model_capabilities.attachment`
    /// for non-text attachments; else Unknown.
    pub fn supports_input_modality(&self, modality: &str) -> CapabilityKnown {
        if let Some(modalities) = self.modalities.as_ref() {
            if let Some(input) = modalities.input.as_ref() {
                let ok = input.iter().any(|m| m.eq_ignore_ascii_case(modality));
                return if ok {
                    CapabilityKnown::Yes
                } else {
                    CapabilityKnown::No
                };
            }
        }

        if modality.eq_ignore_ascii_case("text") {
            return CapabilityKnown::Unknown;
        }

        if let Some(caps) = self.model_capabilities.as_ref() {
            return CapabilityKnown::from_option(caps.attachment);
        }

        CapabilityKnown::Unknown
    }
}

/// Look up `metadata.models.<model_id>` from a flattened manifest `extra` map
/// (or any JSON object that may contain top-level `metadata`).
pub fn model_entry_from_extra(
    extra: &std::collections::HashMap<String, Value>,
    model_id: &str,
) -> Option<MetadataModelEntry> {
    let metadata = extra.get("metadata")?;
    let models = metadata.get("models")?;
    let entry = models.get(model_id)?;
    MetadataModelEntry::from_value(entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn omit_capabilities_is_unknown_not_false() {
        let entry = MetadataModelEntry::from_value(&json!({
            "context_window": 128000,
            "max_output_tokens": 4096,
            "status": "active"
        }))
        .expect("parse");
        assert_eq!(
            entry.supports_input_modality("image"),
            CapabilityKnown::Unknown
        );
        assert!(entry.model_capabilities.is_none());
    }

    #[test]
    fn modalities_input_text_only_rejects_image() {
        let entry = MetadataModelEntry::from_value(&json!({
            "modalities": { "input": ["text"], "output": ["text"] }
        }))
        .expect("parse");
        assert_eq!(entry.supports_input_modality("image"), CapabilityKnown::No);
        assert_eq!(entry.supports_input_modality("text"), CapabilityKnown::Yes);
    }

    #[test]
    fn attachment_false_without_modalities_rejects_image() {
        let entry = MetadataModelEntry::from_value(&json!({
            "model_capabilities": { "attachment": false, "tool_call": true }
        }))
        .expect("parse");
        assert_eq!(entry.supports_input_modality("image"), CapabilityKnown::No);
        assert_eq!(
            CapabilityKnown::from_option(entry.model_capabilities.as_ref().unwrap().tool_call),
            CapabilityKnown::Yes
        );
    }

    #[test]
    fn lookup_from_extra_map() {
        let mut extra = HashMap::new();
        extra.insert(
            "metadata".into(),
            json!({
                "models": {
                    "m1": {
                        "modalities": { "input": ["text", "image"] }
                    }
                }
            }),
        );
        let entry = model_entry_from_extra(&extra, "m1").expect("entry");
        assert_eq!(entry.supports_input_modality("image"), CapabilityKnown::Yes);
        assert!(model_entry_from_extra(&extra, "missing").is_none());
    }

    #[test]
    fn or_provider_prefers_known_model_fact() {
        assert!(CapabilityKnown::Yes.or_provider(false));
        assert!(!CapabilityKnown::No.or_provider(true));
        assert!(CapabilityKnown::Unknown.or_provider(true));
        assert!(!CapabilityKnown::Unknown.or_provider(false));
    }
}
