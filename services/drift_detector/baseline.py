"""
BaselineModel: per-(model_version, tenant_id) rolling statistics tracker.



























Uses exponentially weighted moving averages (EWMA) for mean and variance
across five signal dimensions. Operates in calibration mode for the first
1000 events — during calibration no drift signals are emitted.

EWMA update equations:
  mean_new  = α_m * x + (1 - α_m) * mean_old          (α_m = 0.01)
  var_new   = α_v * (x - mean_new)^2 + (1 - α_v) * var_old  (α_v = 0.005)

Variance is tracked as the EWMA of squared deviations from the current mean.
std = sqrt(var).

Redis serialisation: JSON blob stored at key
  "baseline:{model_version}:{tenant_id}"
"""

from __future__ import annotations

import json
import math
from typing import Dict, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.schemas.events import LogprobSignalEvent

# Dimensions we track — order matters for serialisation.
SIGNAL_DIMENSIONS = (
    "mean_logprob",
    "logprob_variance",
    "logprob_entropy_mean",
    "uncertainty_spike_freq",
    "min_logprob",
)

# EWMA smoothing factors.
ALPHA_MEAN: float = 0.01
ALPHA_VAR: float = 0.005

# Number of events required before leaving calibration mode.
CALIBRATION_THRESHOLD: int = 1000


class BaselineModel:
    """
    Maintains rolling EWMA statistics for a single (model_version, tenant_id)
    key.

    Thread-safety: not thread-safe; callers must serialise updates if shared
    across threads (the owning DriftDetector uses a per-key instance with an
    asyncio event loop, so this is fine).
    """

    def __init__(self, model_version: str, tenant_id: str) -> None:
        self.model_version = model_version
        self.tenant_id = tenant_id

        # EWMA state: {dimension: [mean, variance]}
        # Initialised to None until the first observation bootstraps the state.
        self._mean: Dict[str, Optional[float]] = {d: None for d in SIGNAL_DIMENSIONS}
        self._var: Dict[str, Optional[float]] = {d: None for d in SIGNAL_DIMENSIONS}

        # Event counter — used to track calibration phase.
        self._event_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_calibrating(self) -> bool:
        """True while we have fewer than CALIBRATION_THRESHOLD events."""
        return self._event_count < CALIBRATION_THRESHOLD

    @property
    def event_count(self) -> int:
        return self._event_count

    def update(self, event: LogprobSignalEvent) -> None:
        """
        Ingest one LogprobSignalEvent and update all EWMA statistics.

        Complexity: O(|SIGNAL_DIMENSIONS|) = O(5) — constant.
        Side effects: updates internal EWMA state, increments event counter.
        """
        values = _extract_dimensions(event)
        self._event_count += 1

        for dim in SIGNAL_DIMENSIONS:
            x = values[dim]
            if self._mean[dim] is None:
                # Bootstrap: first observation seeds both mean and variance.
                self._mean[dim] = x
                self._var[dim] = 0.0
            else:
                old_mean = self._mean[dim]
                new_mean = ALPHA_MEAN * x + (1.0 - ALPHA_MEAN) * old_mean
                # Variance EWMA: deviation computed against new mean to avoid
                # systematic bias accumulation.
                deviation_sq = (x - new_mean) ** 2
                new_var = ALPHA_VAR * deviation_sq + (1.0 - ALPHA_VAR) * self._var[dim]
                self._mean[dim] = new_mean
                self._var[dim] = new_var

    def get_baseline(self) -> Dict[str, Tuple[float, float]]:
        """
        Returns {dimension: (mean, std)} for z-score computation.

        std is derived as sqrt(var).  A minimum floor is applied so that
        even when the observed signal has been perfectly constant (e.g. in
        unit tests with identical events, or early in deployment), any
        deviation from the mean still produces a non-zero z-score.

        Floor formula:
          std_floor = max(|mean| * 0.01, 0.01)

        Rationale: for log-probability signals (typically in [-10, 0]), a
        1% relative floor corresponds to 0.015–0.10 logprob units — a
        physically meaningful minimum sensitivity.  The absolute floor of
        0.01 covers the near-zero mean case (e.g. uncertainty_spike_freq
        starting at 0).

        Returns empty dict if no events have been observed yet.
        """
        result: Dict[str, Tuple[float, float]] = {}
        for dim in SIGNAL_DIMENSIONS:
            if self._mean[dim] is None:
                result[dim] = (0.0, 0.01)
            else:
                mean = self._mean[dim]
                raw_std = math.sqrt(max(self._var[dim], 0.0))
                # Apply minimum std floor: 1% of |mean| or 0.01, whichever larger.
                std_floor = max(abs(mean) * 0.01, 0.01)
                std = max(raw_std, std_floor)
                result[dim] = (mean, std)
        return result

    # ------------------------------------------------------------------
    # Redis persistence
    # ------------------------------------------------------------------

    def persist_to_redis(self, redis_client) -> None:
        """
        Serialise state to a JSON blob and write to Redis.

        Key format: "baseline:{model_version}:{tenant_id}"
        No TTL set — state must survive across restarts indefinitely.
        """
        key = self._redis_key()
        payload = self._to_dict()
        redis_client.set(key, json.dumps(payload))

    @classmethod
    def load_from_redis(
        cls, redis_client, model_version: str, tenant_id: str
    ) -> Optional["BaselineModel"]:
        """
        Attempt to hydrate a BaselineModel from Redis.

        Returns None if the key does not exist (first start, or evicted).
        Raises ValueError on schema mismatch (corrupt data).
        """
        key = f"baseline:{model_version}:{tenant_id}"
        raw = redis_client.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        return cls._from_dict(data)

    # ------------------------------------------------------------------
    # Serialisation helpers (also used for test round-trips)
    # ------------------------------------------------------------------

    def _to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "tenant_id": self.tenant_id,
            "event_count": self._event_count,
            "mean": {d: self._mean[d] for d in SIGNAL_DIMENSIONS},
            "var": {d: self._var[d] for d in SIGNAL_DIMENSIONS},
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "BaselineModel":
        instance = cls(data["model_version"], data["tenant_id"])
        instance._event_count = data["event_count"]
        for dim in SIGNAL_DIMENSIONS:
            instance._mean[dim] = data["mean"].get(dim)
            instance._var[dim] = data["var"].get(dim)
        return instance

    def _redis_key(self) -> str:
        return f"baseline:{self.model_version}:{self.tenant_id}"


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def _extract_dimensions(event: LogprobSignalEvent) -> Dict[str, float]:
    """
    Map a LogprobSignalEvent onto the five tracked dimensions.

    uncertainty_spike_freq is computed as
      len(token_uncertainty_spikes) / output_token_count
    clamped to [0, 1] to normalise across variable-length sequences.
    A zero output_token_count is treated as 1 to avoid division by zero.
    """
    denom = max(event.output_token_count, 1)
    spike_freq = len(event.token_uncertainty_spikes) / denom
    return {
        "mean_logprob": event.mean_logprob,
        "logprob_variance": event.logprob_variance,
        "logprob_entropy_mean": event.logprob_entropy_mean,
        "uncertainty_spike_freq": spike_freq,
        "min_logprob": event.min_logprob,
    }
