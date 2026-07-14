//! Deterministic context budget assembly (ALR-P2-001 + CR-L1-001 + CR-L3-001).
//! No network, no LLM summarization.
//!
//! Sync [`MessageAssembler::assemble_layered`] is the source of truth.
//! [`AssemblePool`] / [`MessageAssembler::assemble_layered_async`] only schedule that
//! algorithm under concurrency and timeout limits (Cadence §5.1; Experimental Envelope).

mod assemble_async;
mod assembler;
mod budget;
mod envelope;
mod error;
mod token_estimate;

pub use assemble_async::{AssemblePool, AssemblePoolConfig};
pub use assembler::{AssembleOptions, AssembleReport, LayeredAssembleOptions, MessageAssembler};
pub use budget::{ContextBudget, ModelCapacity};
pub use envelope::{AssembleStrategy, ContextLayer, MessageChunk};
pub use error::AssembleError;
pub use token_estimate::{estimate_message_tokens, estimate_tokens, CHARS_PER_TOKEN};
