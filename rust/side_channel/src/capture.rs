/// Hot-path capture module — receives `LogprobSignalEvent` structs from the
/// inference engine and queues them into a bounded SPSC ring buffer.
///
/// # Design goals
/// - `capture_event` must complete in **< 2 µs** on a typical laptop core.
/// - **Zero heap allocations** on the hot path after initial setup.
/// - Drop semantics: when the ring buffer is full the *new* event is dropped
///   (never block).  A counter tracks every dropped event.
/// - `CaptureStats` is the single authority for capture-side metrics; it is
///   updated via atomic operations and read by the Prometheus scrape thread.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

// ── Event schema ─────────────────────────────────────────────────────────────

/// Per-token position where the model's log-probability crossed an uncertainty
/// threshold.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenUncertaintySpike {
    /// Zero-based token position in the output sequence.
    pub position: i64,
    /// Raw log-probability at this position (always ≤ 0).
    pub logprob: f64,
}

/// Percentile summary of per-token log-probabilities across the output sequence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogprobPercentiles {
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p99: f64,
}

/// Canonical event emitted once per inference request.
///
/// Field names and JSON keys match the Python `LogprobSignalEvent` dataclass
/// defined in `shared/schemas/events.py` exactly so that downstream consumers
/// can use the same schema without translation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogprobSignalEvent {
    /// Globally unique request identifier (UUIDv4 string).
    pub request_id: String,
    /// Wall-clock time of the inference request, nanoseconds since Unix epoch.
    pub timestamp_ns: u64,
    /// Model version tag, e.g. `"adapter-v2"`.
    pub model_version: String,
    /// Opaque tenant identifier.
    pub tenant_id: String,
    /// Number of tokens in the prompt.
    pub input_token_count: u32,
    /// Number of tokens generated.
    pub output_token_count: u32,
    /// Arithmetic mean of per-token log-probabilities (≤ 0).
    pub mean_logprob: f64,
    /// Minimum log-probability across all output tokens (most uncertain token).
    pub min_logprob: f64,
    /// Mean of the per-token entropy values (`-sum(p * log p)`).
    pub logprob_entropy_mean: f64,
    /// Sample variance of per-token log-probabilities.
    pub logprob_variance: f64,
    /// Positions where the model's uncertainty exceeded a configured threshold.
    pub token_uncertainty_spikes: Vec<TokenUncertaintySpike>,
    /// Percentile summary over the output sequence's log-probabilities.
    pub sequence_logprob_percentiles: LogprobPercentiles,
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Atomic counters and latency histogram for the capture path.
///
/// All fields are updated only via `Relaxed` stores on the writer side and
/// read via `Acquire` loads on the reader side.  The weak ordering is
/// acceptable because we only need *eventual* consistency for metrics.
#[derive(Debug, Default)]
pub struct CaptureStats {
    /// Total events successfully enqueued to the ring buffer.
    pub events_captured: AtomicU64,
    /// Total events discarded because the ring buffer was full.
    pub events_dropped: AtomicU64,

    // Latency histogram — 8 buckets matching the Prometheus spec:
    //   ≤500ns, ≤1µs, ≤2µs, ≤5µs, ≤10µs, ≤50µs, ≤100µs, +Inf
    //
    // Indexed as:
    //   [0] ≤ 500 ns
    //   [1] ≤ 1 000 ns
    //   [2] ≤ 2 000 ns
    //   [3] ≤ 5 000 ns
    //   [4] ≤ 10 000 ns
    //   [5] ≤ 50 000 ns
    //   [6] ≤ 100 000 ns
    //   [7] +Inf
    pub latency_buckets: [AtomicU64; 8],
    /// Cumulative sum of all captured latencies in nanoseconds.
    pub latency_sum_ns: AtomicU64,
    /// Total latency samples recorded.
    pub latency_count: AtomicU64,
}

impl CaptureStats {
    /// Convenience constructor that creates a heap-allocated `Arc<CaptureStats>`.
    pub fn new_shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a single capture latency sample (nanoseconds) into the histogram.
    ///
    /// Complexity: O(1), branch-free after the match dispatch.
    #[inline]
    pub fn record_latency_ns(&self, ns: u64) {
        let bucket = match ns {
            n if n <= 500 => 0,
            n if n <= 1_000 => 1,
            n if n <= 2_000 => 2,
            n if n <= 5_000 => 3,
            n if n <= 10_000 => 4,
            n if n <= 50_000 => 5,
            n if n <= 100_000 => 6,
            _ => 7,
        };
        self.latency_buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.latency_sum_ns.fetch_add(ns, Ordering::Relaxed);
        self.latency_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Estimate the p99 capture latency from the histogram buckets.
    ///
    /// Uses linear interpolation within the p99 bucket.  Returns `0` if no
    /// samples have been recorded yet.
    pub fn p99_latency_ns(&self) -> u64 {
        let total = self.latency_count.load(Ordering::Acquire);
        if total == 0 {
            return 0;
        }
        let threshold = (total as f64 * 0.99).ceil() as u64;
        let bucket_upper = [500u64, 1_000, 2_000, 5_000, 10_000, 50_000, 100_000, u64::MAX];
        let mut cumulative = 0u64;
        for (i, &upper) in bucket_upper.iter().enumerate() {
            let count = self.latency_buckets[i].load(Ordering::Acquire);
            cumulative += count;
            if cumulative >= threshold {
                return upper;
            }
        }
        u64::MAX
    }
}

// ── Ring buffer handle ────────────────────────────────────────────────────────

/// Owns the send-side of the bounded channel used as a ring buffer.
///
/// `clone()` is cheap — both ends share the underlying channel via `Arc`.
#[derive(Clone, Debug)]
pub struct CaptureSender {
    tx: Sender<LogprobSignalEvent>,
}

/// Owns the receive-side of the ring buffer.  Consumed by the publisher task.
#[derive(Debug)]
pub struct CaptureReceiver {
    pub rx: Receiver<LogprobSignalEvent>,
}

/// Create a paired `(CaptureSender, CaptureReceiver)` with `capacity` slots.
///
/// # Arguments
/// - `capacity`: Number of `LogprobSignalEvent` slots in the ring buffer.
///   Under-sizing this relative to your burst rate will cause drops.
pub fn make_channel(capacity: usize) -> (CaptureSender, CaptureReceiver) {
    let (tx, rx) = bounded(capacity);
    (CaptureSender { tx }, CaptureReceiver { rx })
}

// ── Hot-path capture function ─────────────────────────────────────────────────

/// Attempt to push `event` into the ring buffer.
///
/// # Performance contract
/// - No heap allocations after the event struct is constructed.
/// - `TrySend` on a crossbeam bounded channel is a CAS + notify; the
///   fast-path (buffer not full) completes in < 200 ns on modern hardware.
/// - On drop: increments `stats.events_dropped` with a single `Relaxed` store.
///
/// # Arguments
/// - `sender`: Shared sender handle; call-site should clone once and reuse.
/// - `stats`: Shared capture statistics; updated atomically.
/// - `event`: The event to enqueue.  Ownership is transferred; no copy occurs.
#[inline]
pub fn capture_event(
    sender: &CaptureSender,
    stats: &CaptureStats,
    event: LogprobSignalEvent,
) {
    let t0 = Instant::now();
    match sender.tx.try_send(event) {
        Ok(()) => {
            let elapsed = t0.elapsed().as_nanos() as u64;
            stats.events_captured.fetch_add(1, Ordering::Relaxed);
            stats.record_latency_ns(elapsed);
        }
        Err(TrySendError::Full(_)) => {
            stats.events_dropped.fetch_add(1, Ordering::Relaxed);
        }
        Err(TrySendError::Disconnected(_)) => {
            // Publisher has gone away; this is fatal in production but
            // can happen during test teardown.
            warn!("capture channel disconnected — event discarded");
            stats.events_dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ── Mock source ───────────────────────────────────────────────────────────────

/// Synthetic event generator that mimics the vLLM inference loop.
///
/// Spawns a Tokio task that emits `LogprobSignalEvent` structs at
/// `rate_per_second` events per second.  Call `stop()` to terminate it
/// gracefully (the spawned task will drain its current batch and exit).
#[derive(Debug)]
pub struct MockVllmSource {
    stop_tx: tokio::sync::oneshot::Sender<()>,
}

impl MockVllmSource {
    /// Start the mock source.
    ///
    /// # Arguments
    /// - `sender`: Where to send generated events.
    /// - `stats`: Shared statistics that `capture_event` updates.
    /// - `rate_per_second`: Target throughput in events per second.
    ///   Values up to ~200 k/s are achievable without spinning; the
    ///   implementation uses `sleep_until` so actual rate will be slightly
    ///   lower due to OS scheduler granularity.
    /// - `model_version`: Stamped on every synthetic event.
    /// - `tenant_id`: Stamped on every synthetic event.
    ///
    /// # Returns
    /// A `MockVllmSource` handle.  Drop or call `stop()` to terminate.
    pub fn start(
        sender: CaptureSender,
        stats: Arc<CaptureStats>,
        rate_per_second: u64,
        model_version: String,
        tenant_id: String,
    ) -> Self {
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();

        tokio::spawn(async move {
            // Interval between events (nanoseconds).  We use tokio::time::interval
            // which has ~1 ms granularity; for high-rate sources we batch events
            // per tick instead of sleeping per event.
            let ns_per_event: u64 = 1_000_000_000 / rate_per_second.max(1);
            // Batch at most 1 ms worth of events per wakeup.
            let tick_ns: u64 = 1_000_000; // 1 ms tick
            let events_per_tick = (tick_ns / ns_per_event).max(1) as u32;
            let tick_dur = Duration::from_nanos(tick_ns);
            let mut interval = tokio::time::interval(tick_dur);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        for _ in 0..events_per_tick {
                            let event = Self::make_event(&model_version, &tenant_id);
                            capture_event(&sender, &stats, event);
                        }
                    }
                    _ = &mut stop_rx => break,
                }
            }
        });

        Self { stop_tx }
    }

    /// Signal the mock source to stop.  Returns immediately; the background
    /// task exits asynchronously.
    pub fn stop(self) {
        // Ignore send errors — task may have already exited.
        let _ = self.stop_tx.send(());
    }

    /// Build a single synthetic `LogprobSignalEvent`.
    ///
    /// All numeric fields use deterministic but non-trivial values so that
    /// downstream schema validation can exercise every field.
    fn make_event(model_version: &str, tenant_id: &str) -> LogprobSignalEvent {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;

        LogprobSignalEvent {
            request_id: Uuid::new_v4().to_string(),
            timestamp_ns,
            model_version: model_version.to_string(),
            tenant_id: tenant_id.to_string(),
            input_token_count: 128,
            output_token_count: 64,
            mean_logprob: -1.5,
            min_logprob: -2.5,
            logprob_entropy_mean: 2.0,
            logprob_variance: 0.3,
            token_uncertainty_spikes: vec![TokenUncertaintySpike {
                position: 42,
                logprob: -3.1,
            }],
            sequence_logprob_percentiles: LogprobPercentiles {
                p10: -2.0,
                p25: -1.8,
                p50: -1.5,
                p75: -1.3,
                p90: -1.1,
                p99: -0.7,
            },
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_and_stat_increment() {
        let (sender, receiver) = make_channel(10);
        let stats = CaptureStats::new_shared();
        let event = MockVllmSource::make_event("v1", "tenant-a");

        capture_event(&sender, &stats, event.clone());

        assert_eq!(stats.events_captured.load(Ordering::Relaxed), 1);
        assert_eq!(stats.events_dropped.load(Ordering::Relaxed), 0);
        let received = receiver.rx.try_recv().expect("should have event");
        assert_eq!(received.model_version, "v1");
    }

    #[test]
    fn test_drop_when_full() {
        let capacity = 4;
        let (sender, _receiver) = make_channel(capacity);
        let stats = CaptureStats::new_shared();

        for _ in 0..capacity {
            let e = MockVllmSource::make_event("v1", "t");
            capture_event(&sender, &stats, e);
        }
        // One more — must be dropped.
        let extra = MockVllmSource::make_event("v1", "t");
        capture_event(&sender, &stats, extra);

        assert_eq!(stats.events_captured.load(Ordering::Relaxed), capacity as u64);
        assert_eq!(stats.events_dropped.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_latency_histogram_p99_bucket_assignment() {
        let stats = CaptureStats::default();
        stats.record_latency_ns(300);   // bucket 0
        stats.record_latency_ns(800);   // bucket 1
        stats.record_latency_ns(1500);  // bucket 2
        stats.record_latency_ns(3000);  // bucket 3
        stats.record_latency_ns(7000);  // bucket 4
        stats.record_latency_ns(20000); // bucket 5
        stats.record_latency_ns(80000); // bucket 6
        stats.record_latency_ns(200000);// bucket 7

        assert_eq!(stats.latency_buckets[0].load(Ordering::Relaxed), 1);
        assert_eq!(stats.latency_buckets[7].load(Ordering::Relaxed), 1);
        assert_eq!(stats.latency_count.load(Ordering::Relaxed), 8);
    }
}
