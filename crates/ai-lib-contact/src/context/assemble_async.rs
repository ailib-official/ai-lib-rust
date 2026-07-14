//! CR-L3-001: async schedule façade over sync [`super::MessageAssembler::assemble_layered`].
//!
//! Sync assemble remains the source of truth. This module only bounds concurrency and
//! applies per-job timeouts (fail-closed). It does **not** introduce a second assemble
//! algorithm or promote Experimental Envelope schemas to a stable Facade.
//!
//! Cadence §5.1: schedule-only slice under Experimental Envelope boundary
//! (ai-protocol context-envelope / Tag mapping).

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::assembler::{AssembleReport, LayeredAssembleOptions, MessageAssembler};
use super::envelope::MessageChunk;
use super::error::AssembleError;

/// Configuration for bounded async assemble scheduling.
#[derive(Debug, Clone)]
pub struct AssemblePoolConfig {
    /// Maximum concurrent assemble jobs (semaphore permits).
    pub max_in_flight: usize,
    /// Per-job wall-clock timeout (covers blocking sync assemble work).
    pub timeout: Duration,
}

impl Default for AssemblePoolConfig {
    fn default() -> Self {
        Self {
            max_in_flight: 8,
            timeout: Duration::from_secs(5),
        }
    }
}

/// Bounded worker façade for layered assemble (CR-L3-001).
///
/// - [`Self::assemble_layered`] calls [`MessageAssembler::assemble_layered`] unchanged.
/// - No free permit → [`AssembleError::QueueFull`] (does not queue unboundedly).
/// - Exceeds `timeout` → [`AssembleError::Timeout`].
/// - [`AssembleError::HardBudgetViolation`] propagates without stripping critical layers.
#[derive(Debug, Clone)]
pub struct AssemblePool {
    semaphore: Arc<Semaphore>,
    max_in_flight: usize,
    timeout: Duration,
    /// Test-only: sleep inside the blocking worker so timeout paths are deterministic.
    #[cfg(test)]
    test_block: Duration,
}

impl AssemblePool {
    pub fn new(config: AssemblePoolConfig) -> Self {
        let max_in_flight = config.max_in_flight.max(1);
        Self {
            semaphore: Arc::new(Semaphore::new(max_in_flight)),
            max_in_flight,
            timeout: config.timeout,
            #[cfg(test)]
            test_block: Duration::ZERO,
        }
    }

    #[cfg(test)]
    fn with_test_block(mut self, block: Duration) -> Self {
        self.test_block = block;
        self
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Schedule layered assemble under the pool's concurrency and timeout limits.
    ///
    /// Owns `chunks` / `options` so work can move into `spawn_blocking`. On timeout the
    /// caller receives [`AssembleError::Timeout`] immediately; the blocking task may still
    /// finish and release its permit afterward (fail-closed to the caller).
    pub async fn assemble_layered(
        &self,
        chunks: Vec<MessageChunk>,
        options: LayeredAssembleOptions,
    ) -> Result<AssembleReport, AssembleError> {
        let permit: OwnedSemaphorePermit =
            self.semaphore
                .clone()
                .try_acquire_owned()
                .map_err(|_| AssembleError::QueueFull {
                    max_in_flight: self.max_in_flight,
                })?;

        let timeout = self.timeout;
        let timeout_ms = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
        #[cfg(test)]
        let test_block = self.test_block;

        let result = tokio::time::timeout(
            timeout,
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                #[cfg(test)]
                if !test_block.is_zero() {
                    std::thread::sleep(test_block);
                }
                MessageAssembler::assemble_layered(&chunks, &options)
            }),
        )
        .await;

        match result {
            Ok(Ok(assemble_result)) => assemble_result,
            Ok(Err(_)) => Err(AssembleError::WorkerFailed),
            Err(_) => Err(AssembleError::Timeout { timeout_ms }),
        }
    }
}

impl MessageAssembler {
    /// Async façade: same semantics as [`Self::assemble_layered`], scheduled via `pool`.
    pub async fn assemble_layered_async(
        chunks: Vec<MessageChunk>,
        options: LayeredAssembleOptions,
        pool: &AssemblePool,
    ) -> Result<AssembleReport, AssembleError> {
        pool.assemble_layered(chunks, options).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{
        AssembleStrategy, ContextBudget, ContextLayer, LayeredAssembleOptions, MessageChunk,
    };
    use ai_lib_core::types::message::Message;

    fn layered_opts(budget: u32) -> LayeredAssembleOptions {
        LayeredAssembleOptions {
            budget: ContextBudget::new(budget, 0, 1),
            strategy: AssembleStrategy::Chat,
            ..Default::default()
        }
    }

    fn under_budget_chunks() -> Vec<MessageChunk> {
        vec![
            MessageChunk::new(ContextLayer::System, 1, Message::system("sys"), "s"),
            MessageChunk::new(ContextLayer::Active, 2, Message::user("ask"), "a"),
            MessageChunk::new(ContextLayer::Relevant, 3, Message::user("rel"), "r"),
        ]
    }

    #[tokio::test]
    async fn assemble_layered_async_matches_sync_under_budget() {
        let chunks = under_budget_chunks();
        let opts = layered_opts(10_000);
        let sync = MessageAssembler::assemble_layered(&chunks, &opts).unwrap();

        let pool = AssemblePool::new(AssemblePoolConfig {
            max_in_flight: 2,
            timeout: Duration::from_secs(2),
        });
        let async_report = MessageAssembler::assemble_layered_async(chunks, opts, &pool)
            .await
            .unwrap();

        assert_eq!(async_report.dropped_prefix, sync.dropped_prefix);
        assert_eq!(async_report.folded_tool_segments, sync.folded_tool_segments);
        assert_eq!(async_report.messages.len(), sync.messages.len());
        for (a, b) in async_report.messages.iter().zip(sync.messages.iter()) {
            assert_eq!(a.role, b.role);
            assert_eq!(format!("{:?}", a.content), format!("{:?}", b.content));
        }
    }

    #[tokio::test]
    async fn assemble_layered_async_hard_budget_fail_closed() {
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
        let pool = AssemblePool::new(AssemblePoolConfig::default());
        let err = MessageAssembler::assemble_layered_async(chunks, layered_opts(5), &pool)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            AssembleError::HardBudgetViolation { budget: 5, .. }
        ));
    }

    #[tokio::test]
    async fn assemble_pool_queue_full_when_saturated() {
        let pool = AssemblePool::new(AssemblePoolConfig {
            max_in_flight: 1,
            timeout: Duration::from_secs(5),
        });
        let _hold = pool.semaphore.clone().try_acquire_owned().unwrap();
        let err = pool
            .assemble_layered(under_budget_chunks(), layered_opts(10_000))
            .await
            .unwrap_err();
        assert_eq!(err, AssembleError::QueueFull { max_in_flight: 1 });
    }

    #[tokio::test]
    async fn assemble_pool_timeout_fail_closed() {
        let pool = AssemblePool::new(AssemblePoolConfig {
            max_in_flight: 1,
            timeout: Duration::from_millis(30),
        })
        .with_test_block(Duration::from_millis(200));

        let err = pool
            .assemble_layered(under_budget_chunks(), layered_opts(10_000))
            .await
            .unwrap_err();
        assert_eq!(err, AssembleError::Timeout { timeout_ms: 30 });
    }
}
