"""
Central Prometheus metrics registry for the autonomous fine-tuning platform.

All services import PlatformMetrics and REGISTRY from here.
A single CollectorRegistry instance avoids duplicate-metric registration
errors when multiple modules import this file.

Thread-safety: prometheus_client metrics are goroutine-safe by design;
the same guarantees hold in CPython due to the GIL + internal locking
in the C implementation of counters/gauges.
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
)


# ---------------------------------------------------------------------------
# Single shared registry — never instantiate a second one.
# ---------------------------------------------------------------------------
REGISTRY = CollectorRegistry()


class PlatformMetrics:
    """
    Namespace class for all platform-wide Prometheus metrics.

    Metrics are class-level attributes so they are registered exactly once
    at import time.  Do NOT instantiate this class; use it as a pure
    namespace (PlatformMetrics.drift_events_total.labels(...).inc()).
    """

    # ------------------------------------------------------------------
    # Side-channel (Rust inference probe) metrics
    # ------------------------------------------------------------------

    side_channel_events_total = Counter(
        "side_channel_events_total",
        "Total logprob signal events captured by the side-channel probe",
        ["status"],          # label: ok | error | dropped
        registry=REGISTRY,
    )

    side_channel_kafka_buffer_depth = Gauge(
        "side_channel_kafka_buffer_depth",
        "Current number of events buffered in the Kafka producer queue",
        registry=REGISTRY,
    )

    # ------------------------------------------------------------------
    # Drift detection metrics
    # ------------------------------------------------------------------

    drift_events_total = Counter(
        "drift_events_total",
        "Drift events emitted by the drift detector",
        ["level", "drift_type"],   # level: WARNING/CRITICAL/EMERGENCY; drift_type: enum
        registry=REGISTRY,
    )

    anomaly_score_current = Gauge(
        "anomaly_score_current",
        "Current composite anomaly score for the active model version",
        ["model_version"],
        registry=REGISTRY,
    )

    baseline_events_processed = Counter(
        "baseline_events_processed_total",
        "Total logprob events ingested for baseline statistics computation",
        registry=REGISTRY,
    )

    # ------------------------------------------------------------------
    # Synthesis metrics
    # ------------------------------------------------------------------

    synthesis_jobs_total = Counter(
        "synthesis_jobs_total",
        "Synthesis jobs completed",
        ["status"],    # COMPLETED | FAILED | SKIPPED
        registry=REGISTRY,
    )

    amplification_factor = Gauge(
        "synthesis_amplification_factor",
        "Amplification factor of the most recently completed synthesis job",
        registry=REGISTRY,
    )

    pairs_synthesized_total = Counter(
        "pairs_synthesized_total",
        "Cumulative DPO preference pairs synthesized across all jobs",
        registry=REGISTRY,
    )

    # ------------------------------------------------------------------
    # Training metrics
    # ------------------------------------------------------------------

    training_cycles_total = Counter(
        "training_cycles_total",
        "Fine-tuning training cycles",
        ["status"],    # COMPLETED | FAILED | CANCELLED
        registry=REGISTRY,
    )

    training_cycle_cost_usd = Gauge(
        "training_cycle_cost_usd",
        "GPU compute cost in USD for the last completed training cycle",
        registry=REGISTRY,
    )

    training_cycle_duration_seconds = Histogram(
        "training_cycle_duration_seconds",
        "Wall-clock duration of a training cycle from start to completion",
        buckets=[300, 600, 1800, 3600, 7200, 14400],
        registry=REGISTRY,
    )

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    evaluation_gate_decisions_total = Counter(
        "evaluation_gate_decisions_total",
        "Decisions emitted by the evaluation gate (PASS / FAIL)",
        ["result"],    # PASS | FAIL
        registry=REGISTRY,
    )

    benchmark_score_delta = Gauge(
        "benchmark_score_delta",
        "Benchmark score delta of the candidate adapter versus production",
        ["benchmark"],
        registry=REGISTRY,
    )

    # ------------------------------------------------------------------
    # Deployment metrics
    # ------------------------------------------------------------------

    adapter_promotions_total = Counter(
        "adapter_promotions_total",
        "Adapter promotion events",
        ["result"],    # SUCCESS | FAILED
        registry=REGISTRY,
    )

    adapter_rollbacks_total = Counter(
        "adapter_rollbacks_total",
        "Adapter rollback events",
        ["reason"],    # REGRESSION | SAFETY | MANUAL | CANARY_FAIL
        registry=REGISTRY,
    )

    promotion_duration_ms = Histogram(
        "promotion_duration_ms",
        "Wall-clock time in milliseconds to complete an adapter promotion",
        buckets=[100, 500, 1000, 2000, 3000, 5000],
        registry=REGISTRY,
    )

    rollback_duration_ms = Histogram(
        "rollback_duration_ms",
        "Wall-clock time in milliseconds to complete an adapter rollback",
        buckets=[50, 100, 200, 500, 1000],
        registry=REGISTRY,
    )

    current_adapter_version = Gauge(
        "current_adapter_version_info",
        "Info gauge: value=1 when this adapter/cluster pair is active",
        ["adapter_id", "cluster_id"],
        registry=REGISTRY,
    )

    # ------------------------------------------------------------------
    # End-to-end loop metrics
    # ------------------------------------------------------------------

    loop_cycle_end_to_end_seconds = Histogram(
        "loop_cycle_end_to_end_seconds",
        "Full autonomous loop cycle duration from drift detection to promotion",
        buckets=[1800, 3600, 7200, 14400, 21600],
        registry=REGISTRY,
    )

    loop_state = Gauge(
        "loop_state_info",
        "Info gauge: value=1 for the currently active loop state",
        ["state"],
        registry=REGISTRY,
    )

    # ------------------------------------------------------------------
    # Scrape helper
    # ------------------------------------------------------------------

    @classmethod
    def get_metrics_output(cls) -> bytes:
        """
        Render all registered metrics to the Prometheus text exposition format.

        Returns:
            bytes: UTF-8 encoded Prometheus text format payload, suitable for
                   serving on a /metrics HTTP endpoint.
        Complexity: O(n) in number of registered time-series.
        Side effects: None (read-only snapshot of current metric state).
        """
        return generate_latest(REGISTRY)
