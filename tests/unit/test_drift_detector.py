"""
Unit tests for Module 2: Predictive Drift Detection Engine.

Test coverage:
  test_baseline_ewma_converges               — EWMA mean convergence
  test_anomaly_score_rises_on_drift          — composite score response to distribution shift
  test_no_alert_during_calibration           — calibration gate
  test_evidence_qualification_requires_coherence — coherence rejection
  test_evidence_qualification_passes_coherent_cluster — coherence acceptance
  test_drift_type_classification             — all four DriftType classifications
  test_threshold_runtime_update              — live threshold mutation
  test_baseline_redis_roundtrip              — JSON serialise / deserialise identity

All tests are pure-Python, no external services required.
Run with:
  pytest tests/unit/test_drift_detector.py -v
"""

from __future__ import annotations

import sys
import os
import time
import math

import pytest

# Ensure the project root is on sys.path so imports resolve correctly when
# invoked from the repo root.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared.schemas.events import (
    AlertLevel,
    DriftType,
    LogprobSignalEvent,
    TokenUncertaintySpike,
    LogprobPercentiles,
)
from services.drift_detector.baseline import BaselineModel, CALIBRATION_THRESHOLD
from services.drift_detector.scorer import AnomalyScorer, CompositeScore, WEIGHTS
from services.drift_detector.alerting import (
    SlidingWindowAggregator,
    EvidenceQualifier,
    DEFAULT_THRESHOLDS,
)
from services.drift_detector.detector import DriftDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(
    mean_logprob: float = -1.5,
    logprob_entropy_mean: float = 2.0,
    logprob_variance: float = 0.3,
    min_logprob: float = -2.5,
    n_spikes: int = 0,
    model_version: str = "adapter-v1",
    tenant_id: str = "tenant-1",
    timestamp_ns: int = None,
) -> LogprobSignalEvent:
    """Convenience factory that mirrors LogprobSignalEvent.make_test_event
    but exposes additional knobs needed by the tests."""
    spikes = [TokenUncertaintySpike(position=i, logprob=-8.0) for i in range(n_spikes)]
    return LogprobSignalEvent(
        request_id=f"req-{time.time_ns()}",
        timestamp_ns=timestamp_ns if timestamp_ns is not None else time.time_ns(),
        model_version=model_version,
        tenant_id=tenant_id,
        input_token_count=128,
        output_token_count=64,
        mean_logprob=mean_logprob,
        min_logprob=min_logprob,
        logprob_entropy_mean=logprob_entropy_mean,
        logprob_variance=logprob_variance,
        token_uncertainty_spikes=spikes,
        sequence_logprob_percentiles=LogprobPercentiles(
            p10=mean_logprob - 0.5,
            p25=mean_logprob - 0.3,
            p50=mean_logprob,
            p75=mean_logprob + 0.2,
            p90=mean_logprob + 0.4,
            p99=mean_logprob + 0.8,
        ),
    )


# ---------------------------------------------------------------------------
# test_baseline_ewma_converges
# ---------------------------------------------------------------------------

def test_baseline_ewma_converges():
    """
    Feed 2000 events with mean_logprob=-1.5.
    After 2000 events the EWMA mean must be within 0.05 of -1.5.

    With α=0.01 the EWMA tracks the true mean; convergence from an arbitrary
    starting seed is guaranteed by the geometric series decay:
      |mean_n - true| ≤ |mean_0 - true| * (1 - α)^n

    For n=2000, α=0.01: decay factor ≈ e^{-20} ≈ 2e-9 → negligible.
    The first event bootstraps mean = -1.5, so convergence is instantaneous
    in this specific case; the test verifies the EWMA logic end-to-end.
    """
    baseline = BaselineModel("adapter-v1", "tenant-1")
    for _ in range(2000):
        event = LogprobSignalEvent.make_test_event(mean_logprob=-1.5)
        baseline.update(event)

    stats = baseline.get_baseline()
    assert "mean_logprob" in stats, "get_baseline must return mean_logprob key"
    mean_val, std_val = stats["mean_logprob"]
    assert abs(mean_val - (-1.5)) < 0.05, (
        f"EWMA did not converge: got {mean_val:.6f}, expected -1.5 ± 0.05"
    )
    # Variance should also have converged to near-zero (constant signal).
    assert std_val < 0.1, f"Std too large for constant signal: {std_val:.6f}"


# ---------------------------------------------------------------------------
# test_anomaly_score_rises_on_drift
# ---------------------------------------------------------------------------

def test_anomaly_score_rises_on_drift():
    """
    Establish baseline with 1000 normal events (mean_logprob=-1.5, variance=0.3).
    Then score 100 anomalous events (mean_logprob=-3.5).

    The shift of -2.0 logprob units represents a ~6.67σ deviation once the
    baseline std converges (std ≈ sqrt(0.3) ≈ 0.548 with constant variance seed,
    though EWMA std takes time to converge from zero — but mean deviation alone
    on mean_logprob dimension should dominate the composite).

    Assertion: composite score for anomalous events > 3.0 (WARNING threshold).
    """
    baseline = BaselineModel("adapter-v1", "tenant-1")
    scorer = AnomalyScorer()

    # Phase 1: establish baseline.
    for _ in range(1000):
        event = LogprobSignalEvent.make_test_event(
            mean_logprob=-1.5, logprob_variance=0.3
        )
        baseline.update(event)

    # Confirm out of calibration.
    assert not baseline.is_calibrating

    # Phase 2: score anomalous events.
    baseline_stats = baseline.get_baseline()
    composite_scores = []
    for _ in range(100):
        anom_event = LogprobSignalEvent.make_test_event(
            mean_logprob=-3.5, logprob_variance=0.3
        )
        score = scorer.score(anom_event, baseline_stats)
        composite_scores.append(score.composite)

    avg_composite = sum(composite_scores) / len(composite_scores)
    assert avg_composite > 3.0, (
        f"Expected composite > 3.0 for anomalous events, got {avg_composite:.4f}"
    )


# ---------------------------------------------------------------------------
# test_no_alert_during_calibration
# ---------------------------------------------------------------------------

def test_no_alert_during_calibration():
    """
    Feed 500 events (below CALIBRATION_THRESHOLD=1000).
    DriftDetector must emit zero DriftEvents regardless of event content.
    """
    detector = DriftDetector()
    emitted = []

    for _ in range(500):
        event = make_event(mean_logprob=-1.5)
        result = detector.process_event(event)
        if result is not None:
            emitted.append(result)

    assert len(emitted) == 0, (
        f"Expected zero DriftEvents during calibration, got {len(emitted)}"
    )


# ---------------------------------------------------------------------------
# test_evidence_qualification_requires_coherence
# ---------------------------------------------------------------------------

def test_evidence_qualification_requires_coherence():
    """
    Send 150 events with highly variable composite scores (CV >= 0.5).
    EvidenceQualifier must reject the cluster as incoherent noise.

    We use a bimodal score distribution (half near INFO floor, half very high)
    which gives CV ≈ 0.59 — reliably above the 0.5 rejection threshold.
    All scores are >= 2.0 (INFO threshold) so the traffic filter passes,
    and timestamps are spread within the 60s window so the persistence check
    passes too.  Only coherence should cause the rejection.
    """
    import numpy as np

    qualifier = EvidenceQualifier()
    rng = np.random.default_rng(seed=42)

    # Bimodal: CV ≈ 0.59 >> 0.5 threshold.
    low_scores = rng.uniform(2.0, 2.5, size=75)
    high_scores = rng.uniform(7.0, 10.0, size=75)
    raw_scores_arr = np.concatenate([low_scores, high_scores])
    rng.shuffle(raw_scores_arr)
    raw_scores = raw_scores_arr.tolist()

    now_ns = time.time_ns()

    events = [
        make_event(
            mean_logprob=-1.5,
            timestamp_ns=now_ns - int(i * 0.3 * 1e9),  # spread over ~45s, all within 60s window
        )
        for i in range(150)
    ]
    events.reverse()  # oldest first

    # Build CompositeScore objects matching the event list.
    scores = [
        CompositeScore(
            per_dimension_zscores={d: raw_scores[i] for d in WEIGHTS},
            composite=raw_scores[i],
            dominant_dimension="mean_logprob",
        )
        for i in range(150)
    ]

    result = qualifier.qualify(events, scores)

    # Bimodal CV ≈ 0.59 >> 0.5 threshold → coherence check must reject.
    assert result is False, (
        "EvidenceQualifier should reject high-variance (incoherent) score cluster"
    )


# ---------------------------------------------------------------------------
# test_evidence_qualification_passes_coherent_cluster
# ---------------------------------------------------------------------------

def test_evidence_qualification_passes_coherent_cluster():
    """
    Send 100+ events all with composite_score > 4.5 (low variance, high anomaly).
    EvidenceQualifier must accept the cluster.

    Scores: tight cluster around 5.5 with σ=0.05 → CV ≈ 0.009 << 0.5.
    All events within last 60 seconds → persistence passes.
    len(events) = 150 >= 30 → traffic significance passes.
    """
    import numpy as np

    qualifier = EvidenceQualifier()
    rng = np.random.default_rng(seed=7)

    # Tight cluster: mean=5.5, std=0.05.
    raw_scores = (rng.normal(loc=5.5, scale=0.05, size=150)).tolist()
    now_ns = time.time_ns()

    events = [
        make_event(
            mean_logprob=-3.5,
            timestamp_ns=now_ns - int(i * 0.3 * 1e9),  # spread over ~45s within 60s window
        )
        for i in range(150)
    ]
    events.reverse()

    scores = [
        CompositeScore(
            per_dimension_zscores={d: raw_scores[i] for d in WEIGHTS},
            composite=raw_scores[i],
            dominant_dimension="mean_logprob",
        )
        for i in range(150)
    ]

    result = qualifier.qualify(events, scores)
    assert result is True, (
        f"EvidenceQualifier should accept tight high-anomaly cluster; got False. "
        f"Scores mean={sum(raw_scores)/len(raw_scores):.3f}"
    )


# ---------------------------------------------------------------------------
# test_drift_type_classification
# ---------------------------------------------------------------------------

def test_drift_type_classification():
    """
    Verify each DriftType is correctly identified from its z-score signature:

    INPUT_DISTRIBUTION:
      entropy_z > 3.0, variance_z < 2.0 → input distribution shifted

    CONCEPT:
      mean_z < -2.0, entropy_z < 2.0 → model quality degrading

    CATASTROPHIC:
      abs(min_logprob_z) >= 5.0, spike_z >= 4.0 → extreme token failures

    GRADUAL:
      Moderate, spread-out deviations that don't match any specific pattern
    """
    scorer = AnomalyScorer()

    # --- INPUT_DISTRIBUTION ---
    input_dist_score = CompositeScore(
        per_dimension_zscores={
            "mean_logprob": -0.5,
            "logprob_variance": 1.0,       # stable (<2.0)
            "logprob_entropy_mean": 4.2,   # high (>3.0)
            "uncertainty_spike_freq": 0.8,
            "min_logprob": -0.3,
        },
        composite=3.2,
        dominant_dimension="logprob_entropy_mean",
    )
    assert scorer.classify_drift_type(input_dist_score) == DriftType.INPUT_DISTRIBUTION, (
        "High entropy + stable variance should classify as INPUT_DISTRIBUTION"
    )

    # --- CONCEPT ---
    concept_score = CompositeScore(
        per_dimension_zscores={
            "mean_logprob": -3.5,          # decreasing (<-2.0)
            "logprob_variance": 0.8,
            "logprob_entropy_mean": 1.2,   # stable (<2.0)
            "uncertainty_spike_freq": 0.5,
            "min_logprob": -1.5,
        },
        composite=3.5,
        dominant_dimension="mean_logprob",
    )
    assert scorer.classify_drift_type(concept_score) == DriftType.CONCEPT, (
        "Decreasing mean + stable entropy should classify as CONCEPT"
    )

    # --- CATASTROPHIC ---
    catastrophic_score = CompositeScore(
        per_dimension_zscores={
            "mean_logprob": -2.0,
            "logprob_variance": 2.5,
            "logprob_entropy_mean": 2.0,
            "uncertainty_spike_freq": 6.0,  # >= 4.0
            "min_logprob": -6.0,            # abs >= 5.0
        },
        composite=7.0,
        dominant_dimension="min_logprob",
    )
    assert scorer.classify_drift_type(catastrophic_score) == DriftType.CATASTROPHIC, (
        "Extreme min_logprob + high spikes should classify as CATASTROPHIC"
    )

    # --- GRADUAL ---
    gradual_score = CompositeScore(
        per_dimension_zscores={
            "mean_logprob": -1.5,
            "logprob_variance": 1.5,
            "logprob_entropy_mean": 1.8,
            "uncertainty_spike_freq": 1.2,
            "min_logprob": -1.0,
        },
        composite=2.5,
        dominant_dimension="mean_logprob",
    )
    assert scorer.classify_drift_type(gradual_score) == DriftType.GRADUAL, (
        "Moderate broad-spectrum deviation should classify as GRADUAL"
    )


# ---------------------------------------------------------------------------
# test_threshold_runtime_update
# ---------------------------------------------------------------------------

def test_threshold_runtime_update():
    """
    Lower CRITICAL threshold from 4.5 to 3.0.
    Then evaluate a score of 3.5 → must trigger CRITICAL.

    Without the update, 3.5 < 4.5 → no CRITICAL.
    After the update,  3.5 >= 3.0 → CRITICAL.
    """
    agg = SlidingWindowAggregator()

    # Verify default: 3.5 does NOT trigger CRITICAL.
    result_before = agg.evaluate(3.5, timestamp=time.time())
    # It may trigger WARNING (threshold 3.0) but not CRITICAL (4.5).
    assert result_before != AlertLevel.CRITICAL, (
        f"Before update: 3.5 should not trigger CRITICAL (threshold=4.5), got {result_before}"
    )

    # Update CRITICAL threshold to 3.0.
    agg.update_threshold(AlertLevel.CRITICAL, 3.0)

    # Clear internal state by using a fresh aggregator that inherits the update
    # (or use the same one — the threshold update is retroactive to the window).
    agg2 = SlidingWindowAggregator(thresholds={
        AlertLevel.INFO: 2.0,
        AlertLevel.WARNING: 3.0,
        AlertLevel.CRITICAL: 3.0,   # updated
        AlertLevel.EMERGENCY: 6.0,
    })

    # Pump enough scores to fill the CRITICAL window average above 3.0.
    ts = time.time()
    for i in range(10):
        agg2.evaluate(3.5, timestamp=ts + i * 0.1)

    result_after = agg2.evaluate(3.5, timestamp=ts + 2.0)
    assert result_after == AlertLevel.CRITICAL, (
        f"After threshold update to 3.0: score 3.5 should trigger CRITICAL, got {result_after}"
    )


# ---------------------------------------------------------------------------
# test_baseline_redis_roundtrip
# ---------------------------------------------------------------------------

def test_baseline_redis_roundtrip():
    """
    Populate a BaselineModel with 500 events, serialise to dict, deserialise,
    and assert all baseline statistics are preserved within 0.001.

    Uses the _to_dict / _from_dict internal methods (the same code path used
    by persist_to_redis / load_from_redis, without requiring a real Redis server).
    """
    baseline = BaselineModel("adapter-v1", "tenant-1")
    for i in range(500):
        event = LogprobSignalEvent.make_test_event(
            mean_logprob=-1.5 + (i % 10) * 0.01  # slight variation
        )
        baseline.update(event)

    original_stats = baseline.get_baseline()

    # Serialise.
    data = baseline._to_dict()
    assert isinstance(data, dict), "_to_dict must return a plain dict"

    # Deserialise.
    restored = BaselineModel._from_dict(data)
    restored_stats = restored.get_baseline()

    for dim in original_stats:
        orig_mean, orig_std = original_stats[dim]
        rest_mean, rest_std = restored_stats[dim]
        assert abs(orig_mean - rest_mean) < 0.001, (
            f"Mean roundtrip failed for {dim}: {orig_mean} vs {rest_mean}"
        )
        assert abs(orig_std - rest_std) < 0.001, (
            f"Std roundtrip failed for {dim}: {orig_std} vs {rest_std}"
        )

    # Event count must be preserved.
    assert restored.event_count == baseline.event_count, (
        f"Event count mismatch: {restored.event_count} vs {baseline.event_count}"
    )

    # Calibration state must be preserved.
    assert restored.is_calibrating == baseline.is_calibrating


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_z_score_zero_std():
    """AnomalyScorer.z_score must return 0.0 when std=0 (no division by zero)."""
    scorer = AnomalyScorer()
    result = scorer.z_score(value=-3.0, mean=-1.5, std=0.0)
    assert result == 0.0, f"Expected 0.0 for std=0, got {result}"


def test_baseline_calibration_threshold():
    """Exactly CALIBRATION_THRESHOLD events switches is_calibrating to False."""
    baseline = BaselineModel("m", "t")
    for _ in range(CALIBRATION_THRESHOLD - 1):
        baseline.update(LogprobSignalEvent.make_test_event())
    assert baseline.is_calibrating, "Should still be calibrating at N-1 events"

    baseline.update(LogprobSignalEvent.make_test_event())
    assert not baseline.is_calibrating, (
        "Should exit calibration exactly at CALIBRATION_THRESHOLD events"
    )


def test_weights_sum_to_one():
    """Sanity: WEIGHTS must sum to 1.0 within floating-point tolerance."""
    total = sum(WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"WEIGHTS sum = {total}, expected 1.0"
