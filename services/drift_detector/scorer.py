"""
AnomalyScorer: converts raw signal dimensions into a weighted composite
z-score and classifies the drift type.

Dimension weights (must sum to 1.0):
  mean_logprob          0.35  — primary quality signal
  logprob_variance      0.25  — stability signal
  logprob_entropy_mean  0.20  — input distribution signal
  uncertainty_spike_freq 0.15 — token-level uncertainty
  min_logprob           0.05  — worst-case single token

DriftType classification heuristics (applied in priority order):
  CATASTROPHIC      — abs(min_logprob_z) >= 5.0 AND spike_z >= 4.0
  INPUT_DISTRIBUTION — entropy_z > 3.0 AND abs(variance_z) < 2.0
  CONCEPT           — mean_z < -2.0 AND abs(entropy_z) < 2.0
  GRADUAL           — fallback (slow, broad-spectrum drift)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.schemas.events import DriftType, LogprobSignalEvent

# ---------------------------------------------------------------------------
# Weights — must sum to exactly 1.0.
# ---------------------------------------------------------------------------
WEIGHTS: Dict[str, float] = {
    "mean_logprob": 0.35,
    "logprob_variance": 0.25,
    "logprob_entropy_mean": 0.20,
    "uncertainty_spike_freq": 0.15,
    "min_logprob": 0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"


@dataclass
class CompositeScore:
    """
    Full scoring result for a single event against a baseline.

    per_dimension_zscores: {dim: signed_z_score}
      Positive z means the current value is higher than baseline mean.
      Negative z means lower.  The composite uses absolute values.
    composite: weighted sum of abs(z_score) per dimension.
    dominant_dimension: dimension with the highest abs(z_score).
    """
    per_dimension_zscores: Dict[str, float]
    composite: float
    dominant_dimension: str


class AnomalyScorer:
    """
    Stateless scorer.  All state lives in the baseline dict passed to score().
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        event: LogprobSignalEvent,
        baseline: Dict[str, Tuple[float, float]],
    ) -> CompositeScore:
        """
        Compute per-dimension z-scores and weighted composite.

        Parameters
        ----------
        event:
            Incoming LogprobSignalEvent to evaluate.
        baseline:
            Dict returned by BaselineModel.get_baseline():
            {dimension: (mean, std)}.

        Returns
        -------
        CompositeScore with signed z-scores, weighted composite, and dominant dim.

        Complexity: O(|WEIGHTS|) = O(5).
        """
        values = _extract_dimensions(event)
        zscores: Dict[str, float] = {}

        for dim, (mean, std) in baseline.items():
            if dim not in WEIGHTS:
                continue
            zscores[dim] = self.z_score(values[dim], mean, std)

        # Ensure every weight dimension is present (guard against missing baseline dims).
        for dim in WEIGHTS:
            if dim not in zscores:
                zscores[dim] = 0.0

        # Composite = weighted sum of absolute z-scores.
        # We use abs because large deviations in either direction are anomalous.
        composite = sum(WEIGHTS[dim] * abs(zscores[dim]) for dim in WEIGHTS)

        dominant = max(WEIGHTS.keys(), key=lambda d: abs(zscores.get(d, 0.0)))

        return CompositeScore(
            per_dimension_zscores=zscores,
            composite=composite,
            dominant_dimension=dominant,
        )

    @staticmethod
    def z_score(value: float, mean: float, std: float) -> float:
        """
        Standard score with zero-std guard.

        When std == 0 (no variance observed yet) we return 0.0 rather than
        raising ZeroDivisionError.  This is correct: if we've never seen
        variance, we have no basis to call the observation anomalous.

        Parameters
        ----------
        value : observed value
        mean  : baseline EWMA mean
        std   : baseline EWMA std (sqrt of variance)

        Returns
        -------
        (value - mean) / std, or 0.0 if std < 1e-10.
        """
        if abs(std) < 1e-10:
            return 0.0
        return (value - mean) / std

    @staticmethod
    def classify_drift_type(score: CompositeScore) -> DriftType:
        """
        Heuristic classification based on z-score signature.

        Priority order (most severe first):
          1. CATASTROPHIC : extreme min_logprob deviation + high spike frequency
          2. INPUT_DISTRIBUTION : high entropy anomaly with stable variance
          3. CONCEPT : degrading mean on otherwise stable inputs
          4. GRADUAL : catch-all for slow, multi-dimensional drift

        The thresholds are intentionally conservative to minimise false
        classification (wrong type is less harmful than a false positive alert).
        """
        z = score.per_dimension_zscores

        min_logprob_z = abs(z.get("min_logprob", 0.0))
        spike_z = abs(z.get("uncertainty_spike_freq", 0.0))
        entropy_z = abs(z.get("logprob_entropy_mean", 0.0))
        variance_z = abs(z.get("logprob_variance", 0.0))
        mean_z = z.get("mean_logprob", 0.0)  # signed — decreasing mean is negative

        # 1. CATASTROPHIC: very bad individual tokens + many uncertainty spikes.
        if min_logprob_z >= 5.0 and spike_z >= 4.0:
            return DriftType.CATASTROPHIC

        # 2. INPUT_DISTRIBUTION: inputs changed but model internals stable.
        #    High entropy deviation while variance is low (model is still
        #    "confident" on whatever it is generating, inputs just shifted).
        if entropy_z > 3.0 and variance_z < 2.0:
            return DriftType.INPUT_DISTRIBUTION

        # 3. CONCEPT: model quality degrading on structurally similar inputs.
        #    Mean logprob decreasing while entropy is not spiking.
        if mean_z < -2.0 and abs(z.get("logprob_entropy_mean", 0.0)) < 2.0:
            return DriftType.CONCEPT

        # 4. GRADUAL: slow, broad-spectrum drift — default.
        return DriftType.GRADUAL


# ---------------------------------------------------------------------------
# Module-level helper (mirrors baseline._extract_dimensions to avoid circular
# import while keeping scorer self-contained).
# ---------------------------------------------------------------------------

def _extract_dimensions(event: LogprobSignalEvent) -> Dict[str, float]:
    denom = max(event.output_token_count, 1)
    spike_freq = len(event.token_uncertainty_spikes) / denom
    return {
        "mean_logprob": event.mean_logprob,
        "logprob_variance": event.logprob_variance,
        "logprob_entropy_mean": event.logprob_entropy_mean,
        "uncertainty_spike_freq": spike_freq,
        "min_logprob": event.min_logprob,
    }
