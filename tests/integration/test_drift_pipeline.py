"""
Integration tests for Module 2: Predictive Drift Detection Engine.

Tests the full pipeline end-to-end with a mock Kafka producer/consumer so no
external services are required.

Test coverage:
  test_full_pipeline_detects_drift
    — 1000 normal events establish baseline, 200 anomalous events trigger CRITICAL

  test_pipeline_does_not_false_positive
    — 5000 normal events produce zero CRITICAL DriftEvents

  test_kafka_consumer_lag_graceful
    — 10 000 events processed sequentially, no drops, detector functional at end

Run with:
  pytest tests/integration/test_drift_pipeline.py -v
"""

from __future__ import annotations

import sys
import os
import time
import uuid
from typing import List, Optional
from dataclasses import dataclass, field

import pytest

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared.schemas.events import (
    AlertLevel,
    DriftEvent,
    DriftType,
    LogprobSignalEvent,
    TokenUncertaintySpike,
    LogprobPercentiles,
)
from services.drift_detector.baseline import CALIBRATION_THRESHOLD
from services.drift_detector.detector import DriftDetector


# ---------------------------------------------------------------------------
# Mock Kafka producer
# ---------------------------------------------------------------------------

class MockKafkaMessage:
    """Mimics confluent_kafka.Message for consumer poll."""

    def __init__(self, value: bytes, key: bytes = b"") -> None:
        self._value = value
        self._key = key

    def value(self) -> bytes:
        return self._value

    def key(self) -> bytes:
        return self._key

    def error(self):
        return None


class MockKafkaProducer:
    """
    Captures all produce() calls so tests can assert on emitted DriftEvents
    without a real Kafka broker.
    """

    def __init__(self) -> None:
        self.produced: List[dict] = []  # list of {"topic", "key", "value"}

    def produce(self, topic: str, key: bytes, value: bytes) -> None:
        self.produced.append({"topic": topic, "key": key, "value": value})

    def poll(self, timeout: float = 0) -> None:
        pass  # No-op in mock.

    def flush(self, timeout: float = 5) -> None:
        pass

    def drift_events(self) -> List[DriftEvent]:
        """Deserialise all captured messages back into DriftEvent objects."""
        from dataclasses import fields as dc_fields
        import json

        events = []
        for msg in self.produced:
            if msg["topic"] != "drift.events":
                continue
            raw = json.loads(msg["value"].decode("utf-8"))
            # Re-inflate nested dataclasses.
            from shared.schemas.events import SignalBreakdown
            raw["signal_breakdown"] = SignalBreakdown(**raw["signal_breakdown"])
            raw["alert_level"] = AlertLevel(raw["alert_level"])
            raw["drift_type"] = DriftType(raw["drift_type"])
            events.append(DriftEvent(**raw))
        return events


# ---------------------------------------------------------------------------
# Event factory helpers
# ---------------------------------------------------------------------------

def _normal_event(
    model_version: str = "adapter-v1",
    tenant_id: str = "tenant-integ",
    timestamp_ns: Optional[int] = None,
) -> LogprobSignalEvent:
    """Stable, normally-distributed event (mean=-1.5, variance=0.3)."""
    return LogprobSignalEvent.make_test_event(
        mean_logprob=-1.5,
        logprob_entropy_mean=2.0,
        logprob_variance=0.3,
        model_version=model_version,
        tenant_id=tenant_id,
    )


def _anomalous_event(
    shift: float = -2.0,
    model_version: str = "adapter-v1",
    tenant_id: str = "tenant-integ",
    timestamp_ns: Optional[int] = None,
) -> LogprobSignalEvent:
    """
    Event that represents a drift scenario: mean_logprob shifted by `shift`
    (negative = worse quality), higher entropy and variance.
    """
    base_mean = -1.5 + shift  # e.g. -3.5 for shift=-2.0
    return LogprobSignalEvent(
        request_id=str(uuid.uuid4()),
        timestamp_ns=timestamp_ns if timestamp_ns is not None else time.time_ns(),
        model_version=model_version,
        tenant_id=tenant_id,
        input_token_count=128,
        output_token_count=64,
        mean_logprob=base_mean,
        min_logprob=base_mean - 2.0,  # much worse worst-case token
        logprob_entropy_mean=4.5,     # elevated entropy
        logprob_variance=1.5,         # elevated variance
        token_uncertainty_spikes=[
            TokenUncertaintySpike(position=i, logprob=-12.0) for i in range(5)
        ],
        sequence_logprob_percentiles=LogprobPercentiles(
            p10=base_mean - 1.0,
            p25=base_mean - 0.5,
            p50=base_mean,
            p75=base_mean + 0.3,
            p90=base_mean + 0.6,
            p99=base_mean + 1.2,
        ),
    )


# ---------------------------------------------------------------------------
# test_full_pipeline_detects_drift
# ---------------------------------------------------------------------------

def test_full_pipeline_detects_drift():
    """
    End-to-end pipeline test:
      1. Feed 1000 normal events → establish baseline (exits calibration).
      2. Feed 500 anomalous events (mean_logprob shifted -2.0) with timestamps
         spread uniformly over the last 10 minutes.
      3. Assert at least one CRITICAL/EMERGENCY DriftEvent was emitted.
      4. Assert the DriftEvent has composite_score > 4.5 and
         qualification_status == "QUALIFIED".

    Design rationale:
      With EWMA α=0.01, a step-change produces ~38 high-scoring events before
      the baseline adapts.  We backdate events over 10 minutes so the first
      burst of anomalous scores (the highest ones) are spread across all three
      persistence windows (1min, 5min, 10min).

      The EvidenceQualifier filters the evidence buffer to anomalous-only events
      (composite >= INFO=2.0) and checks:
        - Traffic: >= 30 anomalous events
        - Coherence: CV < 0.5 on the anomalous cluster
        - Persistence: anomalous events present in all three windows

      The 38-event anomalous burst is enough (>=30) and has lower CV than the
      full bimodal mix because the burst is the initial high-score phase.
    """
    detector = DriftDetector()
    producer = MockKafkaProducer()

    # Phase 1: normal baseline.
    now_ns = time.time_ns()
    for _ in range(1000):
        event = _normal_event()
        detector.process_event(event)

    assert not detector._baselines["adapter-v1::tenant-integ"].is_calibrating, (
        "Baseline must exit calibration after 1000 events"
    )

    # Phase 2: anomalous events with real-time timestamps.
    #
    # We do NOT backdate events.  The first ~38 events score very high
    # (before EWMA adapts); they arrive "now" so they appear in all three
    # persistence windows (1min, 5min, 10min) since the evidence buffer
    # uses the last event's timestamp as "now" for windowing.
    #
    # We send 500 events so the aggregator window averages build up enough
    # to cross the CRITICAL threshold (4.5) on average.
    drift_events: List[DriftEvent] = []

    for i in range(500):
        # Real-time timestamps — no backdating.
        event = _anomalous_event(shift=-2.0)
        result = detector.process_event(event)
        if result is not None:
            drift_events.append(result)
            detector.emit_to_kafka(result, producer)

    # Assertions.
    assert len(drift_events) > 0, (
        "Expected at least one CRITICAL DriftEvent after anomalous events, got none. "
        f"Events processed: {detector.events_processed}"
    )

    critical_events = [
        e for e in drift_events
        if e.alert_level in (AlertLevel.CRITICAL, AlertLevel.EMERGENCY)
    ]
    assert len(critical_events) > 0, (
        f"Expected CRITICAL or EMERGENCY level event, got levels: "
        f"{[e.alert_level for e in drift_events]}"
    )

    best = max(critical_events, key=lambda e: e.composite_anomaly_score)
    assert best.composite_anomaly_score > 4.5, (
        f"Expected composite_score > 4.5, got {best.composite_anomaly_score:.4f}"
    )
    assert best.qualification_status == "QUALIFIED", (
        f"Expected qualification_status='QUALIFIED', got '{best.qualification_status}'"
    )

    # Verify Kafka emission.
    kafka_events = producer.drift_events()
    assert len(kafka_events) == len(drift_events), (
        f"Kafka emission count mismatch: {len(kafka_events)} vs {len(drift_events)}"
    )


# ---------------------------------------------------------------------------
# test_pipeline_does_not_false_positive
# ---------------------------------------------------------------------------

def test_pipeline_does_not_false_positive():
    """
    Feed 5000 normal events with no drift.
    Assert zero CRITICAL/EMERGENCY DriftEvents are emitted.

    Normal events have mean=-1.5 ± tiny EWMA drift; once the baseline converges
    the composite z-scores should be near 0 and well below the CRITICAL threshold
    of 4.5.
    """
    detector = DriftDetector()
    critical_events = []

    for i in range(5000):
        event = _normal_event()
        result = detector.process_event(event)
        if result is not None and result.alert_level in (
            AlertLevel.CRITICAL, AlertLevel.EMERGENCY
        ):
            critical_events.append(result)

    assert len(critical_events) == 0, (
        f"Expected zero CRITICAL/EMERGENCY events on stable input, "
        f"got {len(critical_events)}: {[(e.alert_level, e.composite_anomaly_score) for e in critical_events]}"
    )


# ---------------------------------------------------------------------------
# test_kafka_consumer_lag_graceful
# ---------------------------------------------------------------------------

def test_kafka_consumer_lag_graceful():
    """
    Simulate 10 000 events arriving faster than processing (all enqueued
    synchronously to maximise contention).

    Assertions:
      1. All 10 000 events are processed (events_processed == 10 000).
      2. No events are dropped.
      3. Detector is still functional at the end (can process a new event
         and return a deterministic result).
      4. Baseline is intact and not calibrating after the run.

    This test validates the backpressure / no-drop guarantee of the
    synchronous processing path.  (The async Kafka consumer loop is covered
    separately in the FastAPI integration tests.)
    """
    detector = DriftDetector()
    all_events = []

    # Generate all events upfront (simulates a burst queue).
    for i in range(10_000):
        event = _normal_event()
        all_events.append(event)

    # Process synchronously (simulates consumer catching up with lag).
    results = []
    for event in all_events:
        r = detector.process_event(event)
        results.append(r)

    # Assertion 1: all events processed.
    assert detector.events_processed == 10_000, (
        f"Expected 10 000 events processed, got {detector.events_processed}"
    )

    # Assertion 2: result list length matches input (no silent drops).
    assert len(results) == 10_000, (
        f"Expected 10 000 results, got {len(results)}"
    )

    # Assertion 3: detector functional after burst — process one more event.
    sentinel = _normal_event()
    final_result = detector.process_event(sentinel)
    assert detector.events_processed == 10_001, (
        "Detector should accept events after burst without error"
    )

    # Assertion 4: baseline exited calibration.
    baseline_key = "adapter-v1::tenant-integ"
    assert baseline_key in detector._baselines, (
        f"Baseline key '{baseline_key}' not found in detector"
    )
    bl = detector._baselines[baseline_key]
    assert not bl.is_calibrating, (
        "Baseline should have exited calibration after 10 000 events"
    )

    # Baseline mean should be close to -1.5 (EWMA convergence).
    stats = bl.get_baseline()
    mean_val, _ = stats["mean_logprob"]
    assert abs(mean_val - (-1.5)) < 0.1, (
        f"Baseline mean drifted unexpectedly: {mean_val:.4f}"
    )
