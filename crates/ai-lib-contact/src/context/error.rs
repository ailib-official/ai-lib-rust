use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssembleError {
    EmptyInput,
    /// Tokens required by Layer 0+1 exceed the input budget — do not silently strip critical layers.
    HardBudgetViolation {
        critical_tokens: u32,
        budget: u32,
    },
    /// Async pool has no free permit (`try_acquire` failed). Fail-closed; does not queue unboundedly.
    QueueFull {
        max_in_flight: usize,
    },
    /// Per-job wall-clock deadline exceeded while scheduling or running sync assemble.
    Timeout {
        timeout_ms: u64,
    },
    /// `spawn_blocking` join failed (panic / cancellation). Fail-closed.
    WorkerFailed,
}

impl fmt::Display for AssembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "no messages to assemble"),
            Self::HardBudgetViolation {
                critical_tokens,
                budget,
            } => write!(
                f,
                "hard budget violation: critical layers need {critical_tokens} tokens but budget is {budget}"
            ),
            Self::QueueFull { max_in_flight } => {
                write!(f, "assemble queue full (max_in_flight={max_in_flight})")
            }
            Self::Timeout { timeout_ms } => {
                write!(f, "assemble timed out after {timeout_ms}ms")
            }
            Self::WorkerFailed => write!(f, "assemble worker failed"),
        }
    }
}

impl std::error::Error for AssembleError {}
