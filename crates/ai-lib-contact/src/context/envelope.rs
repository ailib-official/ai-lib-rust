//! Context envelope types for layered assembly (CR-L1-001 / ADR-2026-07).

use ai_lib_core::types::message::Message;

/// History / context priority layer (Eos CONTEXT_ARCHITECTURE_V2 Layer 0–5).
///
/// Lower discriminant = higher priority (kept first under budget pressure).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ContextLayer {
    /// System prompt, tools, constraints — critical with Active.
    System = 0,
    /// Current round / active task — critical with System.
    Active = 1,
    /// Retrieved / referenced relevant material.
    Relevant = 2,
    /// External or compacted summaries.
    Summary = 3,
    /// Ordinary background history (drop oldest first).
    Background = 4,
    /// Archive index / refs only — not expanded into the payload by default.
    Archive = 5,
}

impl ContextLayer {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::System),
            1 => Some(Self::Active),
            2 => Some(Self::Relevant),
            3 => Some(Self::Summary),
            4 => Some(Self::Background),
            5 => Some(Self::Archive),
            _ => None,
        }
    }

    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Layers that must fit or assembly fails with [`super::AssembleError::HardBudgetViolation`].
    pub const fn is_critical(self) -> bool {
        matches!(self, Self::System | Self::Active)
    }
}

/// Single envelope unit fed to [`super::MessageAssembler::assemble_layered`].
#[derive(Debug, Clone)]
pub struct MessageChunk {
    pub layer: ContextLayer,
    /// Ordering key within a layer (higher = newer for Background drop order).
    pub timestamp: u64,
    pub message: Message,
    /// Prefer keeping this chunk when thinning Summary/Relevant under soft pressure.
    pub is_summary: bool,
    pub chunk_id: String,
}

impl MessageChunk {
    pub fn new(
        layer: ContextLayer,
        timestamp: u64,
        message: Message,
        chunk_id: impl Into<String>,
    ) -> Self {
        Self {
            layer,
            timestamp,
            message,
            is_summary: false,
            chunk_id: chunk_id.into(),
        }
    }

    pub fn with_summary(mut self, is_summary: bool) -> Self {
        self.is_summary = is_summary;
        self
    }
}

/// Fill strategy for layered assembly (minimal v1 set).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssembleStrategy {
    /// Default chat: L0→L1 hard, then L2→L3→L4.
    #[default]
    Chat,
    /// Code-fix: same layer order; when thinning soft layers, prefer Relevant over Summary.
    CodeFix,
}
