"""
Alert budget management and multi-channel dispatch.

Design decisions:
  - AlertBudget implements a sliding-window rate limiter (1-hour window) per
    level.  Timestamps older than 3600 s are evicted on each check, keeping
    memory bounded to O(limit) entries per level.
  - INFO alerts are never dispatched (rate limit = 0 means "never send").
  - EMERGENCY alerts bypass rate limiting (limit = 100 which is effectively
    unbounded for any realistic alert burst).
  - AlertDispatcher is intentionally synchronous because alert channels
    (Slack webhooks, PagerDuty, email) are typically fast and the volume of
    dispatched alerts is low by design.  Async channels can be wrapped in
    a lambda that schedules a coroutine if needed.

Invariants:
  - should_send(level) is idempotent — it does NOT modify state.
  - record_sent(level) MUST be called after a successful dispatch.
  - dispatch() calls both in the correct order and is the preferred entry point.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
import time
import uuid


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """
    Immutable alert payload propagated to all registered channels.

    Fields:
        alert_id:      Unique identifier (UUID recommended).
        level:         Severity — INFO | WARNING | CRITICAL | EMERGENCY.
        title:         Short human-readable summary (< 120 chars).
        body:          Detailed description / remediation guidance.
        evidence:      Arbitrary key-value evidence dict (anomaly scores,
                       request IDs, metric snapshots, etc.).
        timestamp:     Unix epoch float at creation time.
        acknowledged:  Set to True by an operator acknowledgement system.
    """

    alert_id: str
    level: str       # INFO | WARNING | CRITICAL | EMERGENCY
    title: str
    body: str
    evidence: dict
    timestamp: float
    acknowledged: bool = False


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class AlertBudget:
    """
    Per-level sliding-window rate limiter for alert dispatch.

    Rate limits (max alerts per rolling 1-hour window):
        INFO      →  0   (never dispatched)
        WARNING   → 10
        CRITICAL  →  5   (acknowledgement strongly recommended)
        EMERGENCY → 100  (effectively unlimited)

    The window is computed lazily on each call: timestamps older than 3600 s
    are evicted before the count is evaluated.  This avoids a background
    thread / timer and keeps the implementation lock-free (safe under the GIL).

    Memory: O(sum of limits) = O(115) entries maximum.
    """

    # Maximum alerts dispatched per 1-hour rolling window, per level.
    RATE_LIMITS: Dict[str, int] = {
        "INFO": 0,          # never alert — noisy, low value
        "WARNING": 10,      # moderate volume, suppress duplicates
        "CRITICAL": 5,      # high-severity, require ack, suppress storm
        "EMERGENCY": 100,   # always alert — safety-critical
    }

    # Window duration in seconds.
    WINDOW_SECONDS: float = 3600.0

    def __init__(self):
        # Maps level → list of dispatch timestamps within the current window.
        # Using a plain list is efficient: eviction is O(n) but n ≤ limit.
        self._sent: Dict[str, List[float]] = {}

    def _evict_stale(self, level: str, now: float) -> List[float]:
        """
        Remove timestamps outside the rolling window and return the remaining
        list.  Mutates self._sent[level] in place for efficiency.

        Args:
            level: Alert level key.
            now:   Current Unix timestamp.
        Returns:
            Pruned list of timestamps within the window.
        Complexity: O(k) where k is the current window count for this level.
        Side effects: mutates self._sent.
        """
        cutoff = now - self.WINDOW_SECONDS
        recent = [t for t in self._sent.get(level, []) if t > cutoff]
        self._sent[level] = recent
        return recent

    def should_send(self, level: str) -> bool:
        """
        Determine if an alert at this level should be dispatched without
        modifying budget state.

        Args:
            level: Alert level string.
        Returns:
            True if dispatch is permitted under the current budget.
        Complexity: O(k) — eviction scan.
        Side effects: evicts stale timestamps from internal state (benign).
        """
        limit = self.RATE_LIMITS.get(level, 100)
        if limit == 0:
            return False
        now = time.time()
        recent = self._evict_stale(level, now)
        return len(recent) < limit

    def record_sent(self, level: str):
        """
        Record that one alert of the given level was successfully dispatched.
        Must be called after every successful dispatch to keep the budget
        accurate.

        Args:
            level: Alert level that was dispatched.
        Complexity: O(1) amortised.
        Side effects: appends current timestamp to self._sent[level].
        """
        self._sent.setdefault(level, []).append(time.time())


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class AlertDispatcher:
    """
    Dispatch alerts through registered channels, gated by an AlertBudget.

    Channel callables receive a single Alert argument and are expected to
    be synchronous.  Exceptions in channels are caught and logged to stderr
    to prevent a faulty channel from blocking others.

    Attributes:
        budget:       AlertBudget instance controlling rate limits.
        channels:     List of callables (Slack, PagerDuty, email, etc.).
        sent_alerts:  In-memory audit log of successfully dispatched alerts.
                      In production this would be persisted to a database.
    """

    def __init__(
        self,
        budget: AlertBudget,
        channels: Optional[List[Callable[[Alert], None]]] = None,
    ):
        """
        Args:
            budget:   AlertBudget instance.
            channels: List of dispatch callables.  Defaults to empty list
                      (useful for testing budget logic in isolation).
        """
        self.budget = budget
        self.channels: List[Callable[[Alert], None]] = channels or []
        self.sent_alerts: List[Alert] = []

    def dispatch(self, alert: Alert) -> bool:
        """
        Attempt to dispatch an alert through all registered channels.

        Flow:
          1. Check budget.should_send(alert.level) — if denied, return False.
          2. Record the send in the budget (consume one slot).
          3. Invoke each channel in registration order.  Channel failures are
             caught and do not prevent subsequent channels from receiving the
             alert.
          4. Append to sent_alerts audit log.
          5. Return True.

        Args:
            alert: Alert payload to dispatch.
        Returns:
            True if the alert was dispatched to at least the budget/channel
            layer; False if rate-limited.
        Complexity: O(c) where c is the number of registered channels.
        Side effects:
            - Modifies AlertBudget internal state.
            - Appends to self.sent_alerts.
            - Calls all channel callables.
        """
        if not self.budget.should_send(alert.level):
            return False

        # Consume one budget slot BEFORE calling channels so that a channel
        # exception does not allow unlimited retries to bypass the budget.
        self.budget.record_sent(alert.level)

        for channel in self.channels:
            try:
                channel(alert)
            except Exception as exc:
                # Log to stderr; do not re-raise — a broken Slack webhook
                # must not suppress a PagerDuty page.
                import sys
                print(
                    f"[AlertDispatcher] channel {channel!r} raised: {exc}",
                    file=sys.stderr,
                )

        self.sent_alerts.append(alert)
        return True

    def add_channel(self, channel: Callable[[Alert], None]):
        """
        Register a new dispatch channel at runtime.

        Args:
            channel: Callable accepting a single Alert argument.
        Side effects: appends to self.channels.
        """
        self.channels.append(channel)
