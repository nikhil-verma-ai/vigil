"""
DriftDetector: top-level orchestrator wiring baseline, scorer, and alerting.

Lifecycle per event:
  1. Retrieve or create BaselineModel for (model_version, tenant_id).
  2. If calibrating → update baseline, return None.
  3. Retrieve baseline stats → score event.
  4. Evaluate alert level via SlidingWindowAggregator.
  5. On CRITICAL+ → run EvidenceQualifier against evidence buffer.
  6. If qualified → build and emit DriftEvent.

Evidence buffer: per model_version deque of last 500 (event, score) pairs.
Kafka emission: publishes serialised DriftEvent JSON to drift.events topic.
"""

from __future__ import annotations

import datetime
import uuid
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import structlog

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.schemas.events import (
    AlertLevel,
    DriftEvent,
    DriftType,
    LogprobSignalEvent,
    SignalBreakdown,
    TOPIC_DRIFT_EVENTS,
)
from .baseline import BaselineModel
from .scorer import AnomalyScorer, CompositeScore
from .alerting import (
    DEFAULT_THRESHOLDS,
    EvidenceQualifier,
    SlidingWindowAggregator,
    _ALERT_PRIORITY,
)

log = structlog.get_logger(__name__)

# Levels that require evidence qualification before emitting a DriftEvent.
_QUALIFICATION_REQUIRED_LEVELS = {AlertLevel.CRITICAL, AlertLevel.EMERGENCY}

# Maximum events to hold per model_version evidence buffer.
_EVIDENCE_BUFFER_SIZE: int = 500


class AlertThresholdManager:
    """
    Thin facade that holds one SlidingWindowAggregator per
    (model_version, tenant_id) and exposes runtime threshold updates.
    """

    def __init__(self) -> None:
        self._aggregators: Dict[str, SlidingWindowAggregator] = {}

    def _key(self, model_version: str, tenant_id: str) -> str:
        return f"{model_version}::{tenant_id}"

    def update_threshold(self, level: AlertLevel, value: float) -> None:
        """Propagate threshold update to all existing aggregators."""
        for agg in self._aggregators.values():
            agg.update_threshold(level, value)
        # Store for aggregators created after this call.
        self._pending_threshold_overrides: Dict[AlertLevel, float] = getattr(
            self, "_pending_threshold_overrides", {}
        )
        self._pending_threshold_overrides[level] = value

    def get_aggregator(
        self, model_version: str, tenant_id: str
    ) -> SlidingWindowAggregator:
        key = self._key(model_version, tenant_id)
        if key not in self._aggregators:
            agg = SlidingWindowAggregator()
            # Apply any pending threshold overrides.
            overrides = getattr(self, "_pending_threshold_overrides", {})
            for lvl, val in overrides.items():
                agg.update_threshold(lvl, val)
            self._aggregators[key] = agg
        return self._aggregators[key]


class DriftDetector:
    """
    Top-level orchestrator.

    Attributes
    ----------
    _baselines : {model_version::tenant_id → BaselineModel}
    _threshold_manager : AlertThresholdManager
    _scorer : AnomalyScorer (stateless)
    _qualifier : EvidenceQualifier (stateless)
    _evidence_buffers : {model_version → deque[(event, score)]}
    _events_processed : int
    """

    def __init__(self, redis_client=None) -> None:
        self._baselines: Dict[str, BaselineModel] = {}
        self._threshold_manager = AlertThresholdManager()
        self._scorer = AnomalyScorer()
        self._qualifier = EvidenceQualifier()
        self._evidence_buffers: Dict[
            str, Deque[Tuple[LogprobSignalEvent, CompositeScore]]
        ] = {}
        self._events_processed: int = 0
        self._redis_client = redis_client

    # ------------------------------------------------------------------
    # Core processing path
    # ------------------------------------------------------------------

    def process_event(
        self, event: LogprobSignalEvent
    ) -> Optional[DriftEvent]:
        """
        Ingest one LogprobSignalEvent and return a DriftEvent if qualified,
        else None.

        Complexity: O(evidence_buffer_size) in the worst case (qualification
        path); O(1) for the fast path (calibrating or below threshold).
        """
        self._events_processed += 1
        baseline = self._get_or_create_baseline(event.model_version, event.tenant_id)

        if baseline.is_calibrating:
            # During calibration: absorb into baseline but do not score.
            # We update FIRST here so the calibration counter advances.
            baseline.update(event)
            return None

        # Score BEFORE updating the baseline.
        # Invariant: the event is evaluated against the distribution learned
        # from all PRIOR events — not the distribution that includes itself.
        # This prevents the EWMA from absorbing the anomaly before we can
        # detect it, which would cause systematic miss of sustained drift.
        baseline_stats = baseline.get_baseline()

        # Score the event.
        composite_score = self._scorer.score(event, baseline_stats)

        # Only adapt baseline for non-anomalous observations.
        # If the event scores above the INFO threshold it may represent drift;
        # absorbing it into the baseline would mask the signal and prevent
        # the evidence qualifier from building a coherent anomalous cluster.
        # The baseline is intentionally frozen during active drift — once the
        # autonomous loop retrains and deploys a new adapter, a fresh baseline
        # is initialised for the new model_version.
        _INFO_THRESHOLD = DEFAULT_THRESHOLDS[AlertLevel.INFO]
        if composite_score.composite < _INFO_THRESHOLD:
            baseline.update(event)

        # Maintain evidence buffer (model_version scoped — intentional:
        # we want cross-tenant evidence for the same model version to detect
        # systematic model degradation, not per-tenant noise).
        self._append_evidence(event.model_version, event, composite_score)

        # Evaluate alert level.
        aggregator = self._threshold_manager.get_aggregator(
            event.model_version, event.tenant_id
        )
        alert_level = aggregator.evaluate(composite_score.composite)

        if alert_level is None:
            return None

        # Below CRITICAL: emit informational signal only if needed (callers
        # can observe the alert level from metrics).
        if alert_level not in _QUALIFICATION_REQUIRED_LEVELS:
            log.info(
                "drift_below_critical",
                model_version=event.model_version,
                tenant_id=event.tenant_id,
                alert_level=alert_level,
                composite=composite_score.composite,
            )
            return None

        # CRITICAL / EMERGENCY: run evidence qualification.
        recent_events, recent_scores = self._get_evidence(event.model_version)
        qualified = self._qualifier.qualify(recent_events, recent_scores)

        if not qualified:
            log.warning(
                "drift_qualification_rejected",
                model_version=event.model_version,
                alert_level=alert_level,
                composite=composite_score.composite,
            )
            return None

        # Build DriftEvent.
        drift_type = self._scorer.classify_drift_type(composite_score)
        drift_event = self._build_drift_event(
            event, composite_score, alert_level, drift_type, recent_events
        )

        log.info(
            "drift_event_emitted",
            event_id=drift_event.event_id,
            model_version=event.model_version,
            alert_level=alert_level,
            drift_type=drift_type,
            composite=composite_score.composite,
        )

        return drift_event

    # ------------------------------------------------------------------
    # Kafka emission
    # ------------------------------------------------------------------

    def emit_to_kafka(self, drift_event: DriftEvent, producer) -> None:
        """
        Publish a DriftEvent to the drift.events Kafka topic.

        Parameters
        ----------
        drift_event : DriftEvent to serialise and publish.
        producer    : confluent_kafka.Producer instance.

        The message key is set to model_version to ensure all drift events
        for the same model land on the same partition (preserving order).
        """
        producer.produce(
            topic=TOPIC_DRIFT_EVENTS,
            key=drift_event.model_version.encode("utf-8"),
            value=drift_event.to_json().encode("utf-8"),
        )
        producer.poll(0)  # Trigger delivery callbacks without blocking.

    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------

    def update_threshold(self, level: AlertLevel, value: float) -> None:
        """Propagate threshold update to all aggregators (live, no restart)."""
        self._threshold_manager.update_threshold(level, value)

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @property
    def baselines_loaded(self) -> int:
        return len(self._baselines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_baseline(
        self, model_version: str, tenant_id: str
    ) -> BaselineModel:
        key = f"{model_version}::{tenant_id}"
        if key not in self._baselines:
            # Attempt Redis restore on first encounter.
            baseline = None
            if self._redis_client is not None:
                try:
                    baseline = BaselineModel.load_from_redis(
                        self._redis_client, model_version, tenant_id
                    )
                except Exception as exc:
                    log.warning("redis_load_failed", error=str(exc))
            if baseline is None:
                baseline = BaselineModel(model_version, tenant_id)
            self._baselines[key] = baseline
        return self._baselines[key]

    def _append_evidence(
        self,
        model_version: str,
        event: LogprobSignalEvent,
        score: CompositeScore,
    ) -> None:
        if model_version not in self._evidence_buffers:
            self._evidence_buffers[model_version] = deque(
                maxlen=_EVIDENCE_BUFFER_SIZE
            )
        self._evidence_buffers[model_version].append((event, score))

    def _get_evidence(
        self, model_version: str
    ) -> Tuple[list, list]:
        buf = self._evidence_buffers.get(model_version, deque())
        events = [e for e, _ in buf]
        scores = [s for _, s in buf]
        return events, scores

    @staticmethod
    def _build_drift_event(
        trigger: LogprobSignalEvent,
        score: CompositeScore,
        alert_level: AlertLevel,
        drift_type: DriftType,
        evidence_events: list,
    ) -> DriftEvent:
        z = score.per_dimension_zscores
        breakdown = SignalBreakdown(
            mean_logprob_zscore=z.get("mean_logprob", 0.0),
            variance_zscore=z.get("logprob_variance", 0.0),
            entropy_zscore=z.get("logprob_entropy_mean", 0.0),
            spike_frequency_zscore=z.get("uncertainty_spike_freq", 0.0),
        )

        # Fraction of evidence buffer that contributed to qualification.
        affected_fraction = (
            len(evidence_events) / _EVIDENCE_BUFFER_SIZE
            if evidence_events
            else 0.0
        )

        evidence_ids = [e.request_id for e in evidence_events[-50:]]

        return DriftEvent(
            event_id=str(uuid.uuid4()),
            detected_at=datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
            model_version=trigger.model_version,
            alert_level=alert_level,
            drift_type=drift_type,
            composite_anomaly_score=score.composite,
            signal_breakdown=breakdown,
            affected_request_fraction=min(affected_fraction, 1.0),
            evidence_request_ids=evidence_ids,
            qualification_status="QUALIFIED",
        )
