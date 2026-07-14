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

    // --- CR-L3-002: sync ≡ async + stress / back-pressure ---

    fn assert_report_eq(sync: &AssembleReport, async_report: &AssembleReport) {
        assert_eq!(async_report.dropped_prefix, sync.dropped_prefix);
        assert_eq!(async_report.folded_tool_segments, sync.folded_tool_segments);
        assert_eq!(async_report.messages.len(), sync.messages.len());
        for (a, b) in async_report.messages.iter().zip(sync.messages.iter()) {
            assert_eq!(a.role, b.role);
            assert_eq!(a.tool_call_id, b.tool_call_id);
            assert_eq!(format!("{:?}", a.content), format!("{:?}", b.content));
        }
    }

    fn assert_sync_async_eq(
        sync: &Result<AssembleReport, AssembleError>,
        async_result: &Result<AssembleReport, AssembleError>,
    ) {
        match (sync, async_result) {
            (Ok(s), Ok(a)) => assert_report_eq(s, a),
            (Err(s), Err(a)) => assert_eq!(s, a),
            (Ok(_), Err(e)) => panic!("async erred while sync ok: {e:?}"),
            (Err(e), Ok(_)) => panic!("async ok while sync erred: {e:?}"),
        }
    }

    fn hard_budget_chunks() -> Vec<MessageChunk> {
        vec![
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
        ]
    }

    fn soft_pressure_chunks() -> Vec<MessageChunk> {
        vec![
            MessageChunk::new(ContextLayer::System, 1, Message::system("sys"), "s"),
            MessageChunk::new(ContextLayer::Active, 2, Message::user("ask"), "a"),
            MessageChunk::new(
                ContextLayer::Relevant,
                3,
                Message::user(format!("rel-{}", "r".repeat(40))),
                "r1",
            ),
            MessageChunk::new(
                ContextLayer::Background,
                4,
                Message::user(format!("bg-{}", "b".repeat(40))),
                "b1",
            ),
            MessageChunk::new(
                ContextLayer::Archive,
                5,
                Message::user("archive-should-omit"),
                "arch",
            ),
        ]
    }

    #[tokio::test]
    async fn sync_async_equivalence_matrix() {
        let pool = AssemblePool::new(AssemblePoolConfig {
            max_in_flight: 4,
            timeout: Duration::from_secs(2),
        });

        let cases: Vec<(&str, Vec<MessageChunk>, LayeredAssembleOptions)> = vec![
            ("empty", vec![], layered_opts(100)),
            ("under_budget", under_budget_chunks(), layered_opts(10_000)),
            ("hard_budget", hard_budget_chunks(), layered_opts(5)),
            (
                "soft_pressure_codefix",
                soft_pressure_chunks(),
                LayeredAssembleOptions {
                    budget: ContextBudget::new(40, 0, 1),
                    strategy: AssembleStrategy::CodeFix,
                    ..Default::default()
                },
            ),
            (
                "under_budget_with_summary",
                vec![
                    MessageChunk::new(ContextLayer::System, 1, Message::system("sys"), "s"),
                    MessageChunk::new(ContextLayer::Active, 2, Message::user("ask"), "a"),
                    MessageChunk::new(ContextLayer::Relevant, 3, Message::user("rel"), "r")
                        .with_summary(true),
                    MessageChunk::new(ContextLayer::Background, 4, Message::user("old-bg"), "b"),
                ],
                layered_opts(10_000),
            ),
        ];

        for (name, chunks, opts) in cases {
            let sync = MessageAssembler::assemble_layered(&chunks, &opts);
            let async_result = pool.assemble_layered(chunks, opts).await;
            assert_sync_async_eq(&sync, &async_result);
            let _ = name;
        }
    }

    #[tokio::test]
    async fn assemble_async_stress_concurrent_ok_matches_sync() {
        let pool = AssemblePool::new(AssemblePoolConfig {
            max_in_flight: 4,
            timeout: Duration::from_secs(5),
        });
        let chunks = under_budget_chunks();
        let opts = layered_opts(10_000);
        let sync = MessageAssembler::assemble_layered(&chunks, &opts).unwrap();

        let mut joins = Vec::with_capacity(64);
        for _ in 0..64 {
            let pool = pool.clone();
            let chunks = chunks.clone();
            let opts = opts.clone();
            joins.push(tokio::spawn(async move {
                pool.assemble_layered(chunks, opts).await
            }));
        }

        let mut ok = 0usize;
        let mut queue_full = 0usize;
        for join in joins {
            match join.await.expect("join") {
                Ok(report) => {
                    assert_report_eq(&sync, &report);
                    ok += 1;
                }
                Err(AssembleError::QueueFull { max_in_flight }) => {
                    assert_eq!(max_in_flight, 4);
                    queue_full += 1;
                }
                Err(other) => panic!("unexpected under contention: {other:?}"),
            }
        }
        assert!(ok >= 4, "expected some successes, got ok={ok}");
        assert!(
            queue_full > 0,
            "expected back-pressure QueueFull, got ok={ok} queue_full={queue_full}"
        );
    }

    #[tokio::test]
    async fn assemble_async_stress_hard_budget_under_queue_pressure() {
        // Hold all permits with slow workers so new submits see QueueFull;
        // any job that still acquires must preserve HardBudgetViolation.
        let pool = AssemblePool::new(AssemblePoolConfig {
            max_in_flight: 2,
            timeout: Duration::from_millis(80),
        })
        .with_test_block(Duration::from_millis(40));

        let chunks = hard_budget_chunks();
        let opts = layered_opts(5);
        let sync_err = MessageAssembler::assemble_layered(&chunks, &opts).unwrap_err();
        assert!(matches!(
            sync_err,
            AssembleError::HardBudgetViolation { budget: 5, .. }
        ));

        let mut joins = Vec::with_capacity(24);
        for _ in 0..24 {
            let pool = pool.clone();
            let chunks = chunks.clone();
            let opts = opts.clone();
            joins.push(tokio::spawn(async move {
                pool.assemble_layered(chunks, opts).await
            }));
        }

        let mut hard_budget = 0usize;
        let mut queue_or_timeout = 0usize;
        for join in joins {
            match join.await.expect("join") {
                Err(AssembleError::HardBudgetViolation { budget: 5, .. }) => {
                    hard_budget += 1;
                }
                Err(AssembleError::QueueFull { .. } | AssembleError::Timeout { .. }) => {
                    queue_or_timeout += 1;
                }
                Ok(_) => panic!("HardBudget must not be stripped to Ok under pressure"),
                Err(other) => panic!("unexpected: {other:?}"),
            }
        }
        assert!(
            hard_budget >= 1,
            "expected at least one HardBudgetViolation through the pool"
        );
        assert!(
            queue_or_timeout >= 1,
            "expected QueueFull/Timeout under pressure"
        );
    }

    #[tokio::test]
    async fn assemble_async_stress_retry_until_n_successes_match_sync() {
        let pool = AssemblePool::new(AssemblePoolConfig {
            max_in_flight: 2,
            timeout: Duration::from_secs(2),
        });
        let chunks = soft_pressure_chunks();
        let opts = LayeredAssembleOptions {
            budget: ContextBudget::new(40, 0, 1),
            strategy: AssembleStrategy::CodeFix,
            ..Default::default()
        };
        let sync = MessageAssembler::assemble_layered(&chunks, &opts).unwrap();

        let mut matched = 0usize;
        let mut attempts = 0usize;
        while matched < 16 && attempts < 200 {
            let mut batch = Vec::with_capacity(4);
            for _ in 0..4 {
                let pool = pool.clone();
                let chunks = chunks.clone();
                let opts = opts.clone();
                batch.push(tokio::spawn(async move {
                    pool.assemble_layered(chunks, opts).await
                }));
            }
            for join in batch {
                attempts += 1;
                match join.await.expect("join") {
                    Ok(report) => {
                        assert_report_eq(&sync, &report);
                        matched += 1;
                        if matched >= 16 {
                            break;
                        }
                    }
                    Err(AssembleError::QueueFull { .. }) => {}
                    Err(other) => panic!("unexpected: {other:?}"),
                }
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            matched, 16,
            "only matched {matched} after {attempts} attempts"
        );
    }
}
