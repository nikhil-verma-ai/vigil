/// Kafka publish pipeline with in-process back-pressure buffer.
///
/// # Architecture
/// ```text
///  capture ring buffer (crossbeam)
///        │ drain_batch()
///        ▼
///  KafkaPublisher::run()
///        │ try_publish()
///        ├─── Kafka available ──► rdkafka FutureProducer
///        └─── Kafka unavailable ► back_pressure_buf (VecDeque, drop-oldest)
/// ```
///
/// # Back-pressure semantics
/// - While Kafka is reachable, batches are sent directly.
/// - When Kafka is unavailable, events accumulate in `back_pressure_buf` up
///   to `config.buffer_max_events`.
/// - When `back_pressure_buf` is at capacity, the **oldest** event is evicted
///   (front of the deque) before the newest is pushed.  This preserves
///   recency: consumers always see the most recent signals even under
///   sustained Kafka outages.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use async_trait::async_trait;
use crossbeam_channel::TryRecvError;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::capture::{CaptureReceiver, LogprobSignalEvent};
use crate::config::AppConfig;
use crate::metrics::Metrics;

// ── Producer abstraction ──────────────────────────────────────────────────────

/// Abstraction over a Kafka producer so integration tests can inject a mock
/// without standing up a real broker.
///
/// # Contract
/// - `send` must be non-blocking from the caller's perspective; the
///   implementation may complete the I/O asynchronously.
/// - `send` returns `Ok(())` when the broker has acknowledged the message.
/// - `send` returns `Err(_)` on any permanent or transient failure.
#[async_trait]
pub trait KafkaProducerBackend: Send + Sync + 'static {
    /// Publish `payload` bytes to `topic`.
    ///
    /// # Arguments
    /// - `topic`: Kafka topic name.
    /// - `payload`: Raw JSON bytes to send.
    async fn send(&self, topic: &str, payload: &str) -> Result<()>;
}

// ── Real rdkafka backend ──────────────────────────────────────────────────────

/// Production Kafka backend backed by `rdkafka::FutureProducer`.
pub struct RdKafkaBackend {
    producer: rdkafka::producer::FutureProducer,
}

impl RdKafkaBackend {
    /// Construct a `FutureProducer` from the broker list in `config`.
    ///
    /// # Errors
    /// Propagates `rdkafka` creation errors.
    pub fn new(config: &AppConfig) -> Result<Self> {
        use rdkafka::config::ClientConfig;
        let producer: rdkafka::producer::FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.kafka_brokers)
            .set("message.timeout.ms", "5000")
            .set("queue.buffering.max.messages", "100000")
            .set("queue.buffering.max.kbytes", "1048576") // 1 GiB
            .set("batch.num.messages", "500")
            .set("linger.ms", "5")
            .create()
            .context("failed to create rdkafka FutureProducer")?;
        Ok(Self { producer })
    }
}

#[async_trait]
impl KafkaProducerBackend for RdKafkaBackend {
    async fn send(&self, topic: &str, payload: &str) -> Result<()> {
        use rdkafka::producer::FutureRecord;
        use std::time::Duration;

        self.producer
            .send(
                FutureRecord::<(), str>::to(topic).payload(payload),
                Duration::from_secs(5),
            )
            .await
            .map_err(|(e, _)| anyhow::anyhow!("rdkafka send failed: {:?}", e))?;
        Ok(())
    }
}

// ── Mock backend (tests only) ─────────────────────────────────────────────────

/// In-memory Kafka backend for unit and integration tests.
///
/// Every message published via `send` is appended to the inner `Vec<String>`.
/// Tests can clone the `Arc<Mutex<Vec<String>>>` before construction and read
/// it directly.
pub struct MockKafkaBackend {
    /// Shared storage of published JSON payloads, in arrival order.
    pub messages: Arc<Mutex<Vec<String>>>,
}

impl MockKafkaBackend {
    /// Create a mock backend with a freshly-allocated message store.
    ///
    /// # Returns
    /// `(backend, messages_handle)` — `messages_handle` can be cloned and
    /// held by the test to inspect published messages.
    pub fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
        let messages = Arc::new(Mutex::new(Vec::new()));
        let backend = Self { messages: Arc::clone(&messages) };
        (backend, messages)
    }
}

#[async_trait]
impl KafkaProducerBackend for MockKafkaBackend {
    async fn send(&self, _topic: &str, payload: &str) -> Result<()> {
        self.messages.lock().await.push(payload.to_owned());
        Ok(())
    }
}

// ── Publisher statistics ──────────────────────────────────────────────────────

/// Counters and latency tracking for the publisher path.
#[derive(Debug, Default)]
pub struct PublisherStats {
    /// Total events successfully acknowledged by the Kafka broker.
    pub events_published: AtomicU64,
    /// Total Kafka send errors (retried via back-pressure buffer).
    pub kafka_errors: AtomicU64,
    /// Cumulative publish latency sum in microseconds.
    pub latency_sum_us: AtomicU64,
    /// Total latency samples.
    pub latency_count: AtomicU64,
    /// Events dropped from the back-pressure buffer due to overflow.
    pub events_dropped_kafka_backpressure: AtomicU64,
}

impl PublisherStats {
    /// Create a new shared `PublisherStats`.
    pub fn new_shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a single end-to-end publish latency sample in microseconds.
    #[inline]
    pub fn record_publish_latency_us(&self, us: u64) {
        self.latency_sum_us.fetch_add(us, Ordering::Relaxed);
        self.latency_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Estimate p99 publish latency in microseconds.
    ///
    /// This is a simple mean-based estimate; for production use wire this into
    /// the Prometheus `Histogram` in `metrics.rs` for accurate percentiles.
    pub fn mean_latency_us(&self) -> f64 {
        let count = self.latency_count.load(Ordering::Acquire);
        if count == 0 {
            return 0.0;
        }
        self.latency_sum_us.load(Ordering::Acquire) as f64 / count as f64
    }
}

// ── Publisher ────────────────────────────────────────────────────────────────

/// Drains the capture ring buffer and publishes events to Kafka.
///
/// Holds a `VecDeque` back-pressure buffer for the Kafka-unavailable case.
pub struct KafkaPublisher<P: KafkaProducerBackend> {
    config: AppConfig,
    producer: P,
    back_pressure_buf: VecDeque<LogprobSignalEvent>,
    pub stats: Arc<PublisherStats>,
    metrics: Arc<Metrics>,
}

impl<P: KafkaProducerBackend> KafkaPublisher<P> {
    /// Construct a new publisher.
    ///
    /// # Arguments
    /// - `config`: Application configuration.
    /// - `producer`: Kafka backend (real or mock).
    /// - `stats`: Shared publisher statistics.
    /// - `metrics`: Prometheus metrics handle.
    pub fn new(
        config: AppConfig,
        producer: P,
        stats: Arc<PublisherStats>,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            back_pressure_buf: VecDeque::with_capacity(config.buffer_max_events),
            config,
            producer,
            stats,
            metrics,
        }
    }

    /// Run the publisher event loop until `receiver` is closed and the
    /// back-pressure buffer is drained.
    ///
    /// The loop:
    /// 1. Drains up to 500 events from the ring buffer in a batch.
    /// 2. Flushes the back-pressure buffer first (oldest-first ordering).
    /// 3. Sends each new batch event; on error, enqueues to back-pressure buf.
    /// 4. Updates Prometheus gauges.
    /// 5. Yields to the Tokio scheduler if both buffers are momentarily empty.
    pub async fn run(mut self, receiver: CaptureReceiver) -> Result<()> {
        info!("publisher started");
        let batch_size = 500usize;
        let mut batch = Vec::with_capacity(batch_size);

        // Track whether the sender side of the ring buffer has been dropped.
        // We detect this via `TryRecvError::Disconnected` from `try_recv`.
        let mut source_done = false;

        loop {
            // Drain a batch from the ring buffer without blocking.
            for _ in 0..batch_size {
                match receiver.rx.try_recv() {
                    Ok(event) => batch.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        // Channel closed — no more events will arrive.
                        source_done = true;
                        break;
                    }
                }
            }

            let drained_count = batch.len();

            // Flush back-pressure buffer first (preserves send ordering).
            self.flush_backpressure().await;

            // Publish newly drained events.
            for event in batch.drain(..) {
                self.publish_one(event).await;
            }

            // Update Prometheus buffer depth gauge.
            self.metrics
                .kafka_buffer_depth
                .set(self.back_pressure_buf.len() as f64);

            // Exit when: source is closed, ring buffer is drained, and the
            // back-pressure buffer has been flushed.
            if source_done && drained_count == 0 && self.back_pressure_buf.is_empty() {
                info!("publisher: ring buffer closed and all events drained — exiting");
                break;
            }

            // Yield on every iteration so Tokio timers and other tasks (including
            // tokio::time::timeout pollers) can run.  The unconditional yield
            // replaces the previous conditional yield that only fired when both
            // buffers were empty; without it, 20 sequential synchronous-error
            // send() calls on the single-threaded test runtime would never give
            // the timeout future a chance to be polled.
            tokio::task::yield_now().await;
        }

        info!(
            events_published = self.stats.events_published.load(Ordering::Relaxed),
            "publisher shut down cleanly"
        );
        Ok(())
    }

    /// Attempt to flush the back-pressure buffer.
    ///
    /// Stops flushing on the first Kafka error to avoid thrashing during an
    /// outage.
    async fn flush_backpressure(&mut self) {
        while let Some(event) = self.back_pressure_buf.front().cloned() {
            match self.send_event(&event).await {
                Ok(()) => {
                    self.back_pressure_buf.pop_front();
                }
                Err(e) => {
                    warn!("Kafka still unavailable while flushing back-pressure: {e}");
                    self.stats.kafka_errors.fetch_add(1, Ordering::Relaxed);
                    break;
                }
            }
        }
    }

    /// Publish a single event.  On Kafka failure, enqueue to the back-pressure
    /// buffer, evicting the oldest event if at capacity.
    async fn publish_one(&mut self, event: LogprobSignalEvent) {
        match self.send_event(&event).await {
            Ok(()) => { /* already counted inside send_event */ }
            Err(e) => {
                debug!("Kafka send failed, buffering event: {e}");
                self.stats.kafka_errors.fetch_add(1, Ordering::Relaxed);
                self.enqueue_backpressure(event);
            }
        }
    }

    /// Serialize and send a single event to Kafka, measuring latency.
    async fn send_event(&self, event: &LogprobSignalEvent) -> Result<()> {
        let payload = serde_json::to_string(event)
            .context("failed to serialise LogprobSignalEvent")?;

        let t0 = Instant::now();
        self.producer.send(&self.config.kafka_topic, &payload).await?;
        let elapsed_us = t0.elapsed().as_micros() as u64;

        self.stats.events_published.fetch_add(1, Ordering::Relaxed);
        self.stats.record_publish_latency_us(elapsed_us);
        self.metrics
            .publish_latency_us
            .observe(elapsed_us as f64);
        self.metrics
            .events_total
            .with_label_values(&["captured"])
            .inc();

        Ok(())
    }

    /// Push `event` to the back-pressure buffer, dropping the oldest entry if
    /// at capacity.
    ///
    /// # Invariant
    /// After this call, `back_pressure_buf.len() <= config.buffer_max_events`.
    fn enqueue_backpressure(&mut self, event: LogprobSignalEvent) {
        if self.back_pressure_buf.len() >= self.config.buffer_max_events {
            // Drop oldest — preserves recency.
            self.back_pressure_buf.pop_front();
            self.stats
                .events_dropped_kafka_backpressure
                .fetch_add(1, Ordering::Relaxed);
            error!(
                "Kafka back-pressure buffer at capacity ({}); dropped oldest event",
                self.config.buffer_max_events
            );
        }
        self.back_pressure_buf.push_back(event);
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::MockVllmSource;
    use crate::metrics::Metrics;

    fn make_test_config() -> AppConfig {
        AppConfig::from_overrides([
            ("kafka_brokers", "localhost:9092"),
            ("tenant_id", "test-tenant"),
            ("model_version", "v1"),
            ("buffer_max_events", "10"),
        ])
        .unwrap()
    }

    #[tokio::test]
    async fn test_backpressure_evicts_oldest() {
        // Set capacity to 3 so we can observe eviction easily.
        let mut config = make_test_config();
        config.buffer_max_events = 3;

        let (mock, messages) = MockKafkaBackend::new();
        let stats = PublisherStats::new_shared();
        let metrics = Arc::new(Metrics::new().unwrap());

        let mut publisher = KafkaPublisher::new(config, mock, stats.clone(), metrics);

        // Simulate Kafka being down: fill back-pressure buffer beyond capacity.
        let ids: Vec<String> = (0..5).map(|i| format!("req-{i}")).collect();
        for id in &ids {
            publisher.enqueue_backpressure(MockVllmSource::make_event_with_id(id, "v1", "t"));
        }

        // Only 3 events should remain; first two were evicted.
        assert_eq!(publisher.back_pressure_buf.len(), 3);
        let remaining_ids: Vec<&str> = publisher
            .back_pressure_buf
            .iter()
            .map(|e| e.request_id.as_str())
            .collect();
        assert_eq!(remaining_ids, vec!["req-2", "req-3", "req-4"]);
        assert_eq!(
            stats
                .events_dropped_kafka_backpressure
                .load(Ordering::Relaxed),
            2
        );

        // Suppress unused warning on messages handle.
        drop(messages);
    }

    #[tokio::test]
    async fn test_serialisation_matches_python_schema() {
        use crate::capture::{LogprobPercentiles, TokenUncertaintySpike};
        let event = LogprobSignalEvent {
            request_id: "abc-123".to_string(),
            timestamp_ns: 1_700_000_000_000_000_000,
            model_version: "adapter-v1".to_string(),
            tenant_id: "test-tenant".to_string(),
            input_token_count: 128,
            output_token_count: 64,
            mean_logprob: -1.5,
            min_logprob: -2.5,
            logprob_entropy_mean: 2.0,
            logprob_variance: 0.3,
            token_uncertainty_spikes: vec![TokenUncertaintySpike {
                position: 5,
                logprob: -3.0,
            }],
            sequence_logprob_percentiles: LogprobPercentiles {
                p10: -2.0,
                p25: -1.8,
                p50: -1.5,
                p75: -1.3,
                p90: -1.1,
                p99: -0.7,
            },
        };

        let json = serde_json::to_string(&event).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Verify every field present in the Python schema is present.
        for key in &[
            "request_id",
            "timestamp_ns",
            "model_version",
            "tenant_id",
            "input_token_count",
            "output_token_count",
            "mean_logprob",
            "min_logprob",
            "logprob_entropy_mean",
            "logprob_variance",
            "token_uncertainty_spikes",
            "sequence_logprob_percentiles",
        ] {
            assert!(parsed.get(key).is_some(), "missing field: {key}");
        }

        let spikes = parsed["token_uncertainty_spikes"].as_array().unwrap();
        assert_eq!(spikes.len(), 1);
        assert!(spikes[0].get("position").is_some());
        assert!(spikes[0].get("logprob").is_some());

        let percentiles = &parsed["sequence_logprob_percentiles"];
        for p in &["p10", "p25", "p50", "p75", "p90", "p99"] {
            assert!(percentiles.get(p).is_some(), "missing percentile: {p}");
        }
    }
}

// Expose a test helper on MockVllmSource that avoids polluting the main API.
#[cfg(test)]
impl crate::capture::MockVllmSource {
    /// Create a synthetic event with a specific `request_id` for test assertions.
    pub fn make_event_with_id(
        request_id: &str,
        model_version: &str,
        tenant_id: &str,
    ) -> LogprobSignalEvent {
        use std::time::{Duration, SystemTime, UNIX_EPOCH};
        use crate::capture::{LogprobPercentiles, TokenUncertaintySpike};
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;

        LogprobSignalEvent {
            request_id: request_id.to_string(),
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
                position: 1,
                logprob: -3.0,
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
