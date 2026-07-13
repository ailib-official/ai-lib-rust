use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssembleError {
    EmptyInput,
    /// Tokens required by Layer 0+1 exceed the input budget — do not silently strip critical layers.
    HardBudgetViolation {
        critical_tokens: u32,
        budget: u32,
    },
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
        }
    }
}

impl std::error::Error for AssembleError {}
