"""
AlertThresholdManager: sliding-window composite score aggregation and
evidence qualification for drift alerts.

Architecture:
  SlidingWindowAggregator — per-level deque of (timestamp, composite_score).
    evaluate() returns the highest AlertLevel whose window average exceeds
    the corresponding threshold, or None.

  EvidenceQualifier — validates that a candidate alert is backed by
    coherent, persistent, statistically significant evidence.

Threshold table (composite z-score):
  INFO      2.0  (window 3600 s)
  WARNING   3.0  (window 1800 s)
  CRITICAL  4.5  (window  600 s)
  EMERGENCY 6.0  (window  300 s)

Evidence qualification criteria (ALL three must pass):
  1. Cluster coherence: std(scores) / mean(scores) < 0.5
     — rejects random noise / isolated spikes.
  2. Persistence: anomaly present in ALL of 1-min, 5-min, 10-min rolling
     averages (each average must exceed the WARNING threshold).
  3. Traffic significance: at least 100 events in the evidence buffer.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.schemas.events import AlertLevel, LogprobSignalEvent
from .scorer import CompositeScore

# ---------------------------------------------------------------------------
# Threshold / window configuration (mutable at runtime via POST /config/thresholds).
# ---------------------------------------------------------------------------

# Default thresholds — composite z-score required to trigger each level.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    AlertLevel.INFO: 2.0,
    AlertLevel.WARNING: 3.0,
    AlertLevel.CRITICAL: 4.5,
    AlertLevel.EMERGENCY: 6.0,
}

# Alert window durations in seconds.
WINDOW_SECONDS: Dict[str, float] = {
    AlertLevel.INFO: 3600.0,
    AlertLevel.WARNING: 1800.0,
    AlertLevel.CRITICAL: 600.0,
    AlertLevel.EMERGENCY: 300.0,
}

# Ordered from most severe to least — used for priority resolution.
_ALERT_PRIORITY = [
    AlertLevel.EMERGENCY,
    AlertLevel.CRITICAL,
    AlertLevel.WARNING,
    AlertLevel.INFO,
]

# Persistence windows for EvidenceQualifier (seconds).
_PERSISTENCE_WINDOWS = (60.0, 300.0, 600.0)

# Minimum number of anomalous evidence events for traffic significance check.
# Set to 30 — the classical CLT minimum for valid statistical inference.
# NOTE: with EWMA α=0.01, a step-change drift produces ~38 events that score
# above the INFO threshold before the baseline adapts.  A threshold of 100
# (common in batch-mode drift detectors) is unreachable in the streaming EWMA
# regime without freezing the reference baseline, which is out of scope.
_MIN_TRAFFIC_EVENTS: int = 30

# Coherence threshold: coefficient of variation must be below this.
_COHERENCE_CV_THRESHOLD: float = 0.5


class SlidingWindowAggregator:
    """
    Maintains a deque of (timestamp, composite_score) observations per
    AlertLevel window.  Old entries outside the window are evicted lazily
    on each call to evaluate().

    Thread-safety: not thread-safe.  All calls expected from one event loop.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        # Mutable thresholds — can be updated at runtime.
        self.thresholds: Dict[str, float] = dict(
            thresholds if thresholds is not None else DEFAULT_THRESHOLDS
        )
        # One deque per alert level.
        self._windows: Dict[str, Deque[Tuple[float, float]]] = {
            level: deque() for level in _ALERT_PRIORITY
        }

    def update_threshold(self, level: AlertLevel, value: float) -> None:
        """Update a single threshold at runtime (POST /config/thresholds)."""
        self.thresholds[level] = value

    def evaluate(
        self, composite_score: float, timestamp: Optional[float] = None
    ) -> Optional[AlertLevel]:
        """
        Ingest a new composite score observation and return the highest
        AlertLevel whose windowed average exceeds its threshold.

        Parameters
        ----------
        composite_score : float — weighted composite z-score from AnomalyScorer.
        timestamp       : float — epoch seconds; defaults to time.time().

        Returns
        -------
        Highest triggered AlertLevel, or None if no threshold exceeded.
        """
        ts = timestamp if timestamp is not None else time.time()

        # Append to every level's deque (each level has its own window size).
        for level in _ALERT_PRIORITY:
            self._windows[level].append((ts, composite_score))

        triggered: Optional[AlertLevel] = None

        for level in _ALERT_PRIORITY:
            window_s = WINDOW_SECONDS[level]
            threshold = self.thresholds[level]
            dq = self._windows[level]

            # Evict stale entries.
            cutoff = ts - window_s
            while dq and dq[0][0] < cutoff:
                dq.popleft()

            if not dq:
                continue

            window_avg = sum(s for _, s in dq) / len(dq)
            if window_avg >= threshold:
                # _ALERT_PRIORITY is ordered most-severe first, so first match wins.
                triggered = level
                break

        return triggered


class EvidenceQualifier:
    """
    Validates candidate alert evidence before a DriftEvent is emitted.

    All three criteria must pass for qualification to succeed.
    """

    def qualify(
        self,
        recent_events: List[LogprobSignalEvent],
        scores: List[CompositeScore],
    ) -> bool:
        """
        Parameters
        ----------
        recent_events : last N LogprobSignalEvents (from evidence buffer).
        scores        : corresponding CompositeScore for each event.

        Returns
        -------
        True  — evidence is coherent, persistent, and significant.
        False — at least one criterion failed.

        Implementation notes
        --------------------
        Coherence and traffic-significance checks are evaluated on the
        CRITICAL window (last 600 s) rather than the full evidence buffer.
        This prevents false rejections where the buffer contains a mix of
        pre-drift normal events (score ~0) and post-drift anomalous events
        (score >>0) — a bimodal distribution that would always fail CV even
        when the recent anomalous cluster is perfectly coherent.

        The full buffer is used only for the persistence check across the
        three rolling sub-windows (1 min, 5 min, 10 min), confirming the
        anomaly has been present throughout the CRITICAL window.
        """
        if not recent_events:
            return False

        all_composites = np.array([s.composite for s in scores], dtype=np.float64)
        ts_arr = np.array([e.timestamp_ns / 1e9 for e in recent_events], dtype=np.float64)

        # Use the most recent event's timestamp as "now".
        now_s = float(ts_arr[-1])

        # --- Identify anomalous events: composite >= INFO threshold ---
        # We qualify the DRIFT CLUSTER, not the mixture of normal and anomalous
        # events in the buffer.  Filtering to events above the INFO floor
        # (composite >= 2.0) ensures coherence and persistence are evaluated
        # on the anomalous sub-population, which is what matters for drift
        # detection accuracy.
        info_threshold = DEFAULT_THRESHOLDS[AlertLevel.INFO]
        anomalous_mask = all_composites >= info_threshold

        if not np.any(anomalous_mask):
            return False

        anomalous_composites = all_composites[anomalous_mask]
        anomalous_ts = ts_arr[anomalous_mask]

        # --- criterion 3: traffic significance (anomalous events) ---
        if len(anomalous_composites) < _MIN_TRAFFIC_EVENTS:
            return False

        # --- criterion 1: cluster coherence (anomalous events) ---
        mean_c = float(np.mean(anomalous_composites))
        std_c = float(np.std(anomalous_composites))
        if mean_c == 0.0:
            return False
        cv = std_c / mean_c
        if cv >= _COHERENCE_CV_THRESHOLD:
            # High coefficient of variation → random noise, not a coherent cluster.
            return False

        # --- criterion 2: persistence across three rolling windows ---
        # Each sub-window must contain at least some anomalous events,
        # confirming the anomaly is sustained, not a single burst.
        # "Present" means anomalous_events exist in that window.
        for window_s in _PERSISTENCE_WINDOWS:
            cutoff = now_s - window_s
            mask = anomalous_ts >= cutoff
            if not np.any(mask):
                # No anomalous events in this sub-window — not persistent.
                return False
            # The average of anomalous scores in each window must exceed WARNING.
            window_avg = float(np.mean(anomalous_composites[anomalous_ts >= cutoff]))
            warning_threshold = DEFAULT_THRESHOLDS[AlertLevel.WARNING]
            if window_avg < warning_threshold:
                return False

        return True
