"""
Trigger conditions that initiate a training cycle.

Three trigger types are supported:

  DriftTrigger   — reacts to qualifying drift events arriving from Kafka.
                   Only CRITICAL/EMERGENCY events that have passed the
                   evidence-qualification gate AND exceed the composite
                   anomaly score threshold produce a TriggerEvent.

  ScheduleTrigger — fires at a fixed cadence (default: weekly).
                    Does NOT fire immediately at startup; the caller must
                    set an initial reference time via record_trigger() or
                    let the first scheduled tick set it.

  ManualTrigger  — thin wrapper so operator-initiated cycles go through
                   the same TriggerEvent path as automated ones.

All three return the same TriggerEvent dataclass so downstream consumers
(AutonomousLoop) do not need to branch on trigger source.
"""

from dataclasses import dataclass
from typing import Optional
import time
import uuid


@dataclass
class TriggerEvent:
    """
    Immutable record of a single trigger firing.

    Fields:
        trigger_id:      UUID generated at fire time.
        trigger_type:    "DRIFT" | "SCHEDULE" | "MANUAL".
        triggered_at:    Unix epoch float.
        source_event_id: drift.events event_id when trigger_type == "DRIFT".
        operator_id:     Identity of the requesting operator for MANUAL triggers.
        budget_cap_usd:  Maximum training cost authorised for this cycle.
    """

    trigger_id: str
    trigger_type: str       # DRIFT | SCHEDULE | MANUAL
    triggered_at: float
    source_event_id: Optional[str] = None   # set for DRIFT triggers
    operator_id: Optional[str] = None       # set for MANUAL triggers
    budget_cap_usd: float = 30.0


class DriftTrigger:
    """
    Evaluates incoming drift.events payloads and fires when warranted.

    A trigger fires when ALL of the following are true:
      1. alert_level in {"CRITICAL", "EMERGENCY"}
      2. qualification_status == "QUALIFIED"
      3. composite_anomaly_score >= critical_score_threshold

    Attributes:
        threshold: Minimum composite_anomaly_score required to fire.
        triggered_events: Ordered log of every TriggerEvent this instance
                          has emitted (useful for auditing / dedup).
    """

    def __init__(self, critical_score_threshold: float = 4.5):
        self.threshold = critical_score_threshold
        self.triggered_events: list = []

    def evaluate(self, drift_event: dict) -> Optional[TriggerEvent]:
        """
        Evaluate a single drift event dict and return a TriggerEvent or None.

        Parameters:
            drift_event: Deserialised DriftEvent dict (keys: alert_level,
                         qualification_status, composite_anomaly_score, event_id).

        Returns:
            TriggerEvent if all conditions are met; None otherwise.

        Side-effects:
            Appends to self.triggered_events when a TriggerEvent is produced.

        Complexity: O(1).
        """
        alert_level = drift_event.get("alert_level")
        if alert_level not in ("CRITICAL", "EMERGENCY"):
            return None

        if drift_event.get("qualification_status") != "QUALIFIED":
            return None

        score = drift_event.get("composite_anomaly_score", 0)
        if score < self.threshold:
            return None

        t = TriggerEvent(
            trigger_id=str(uuid.uuid4()),
            trigger_type="DRIFT",
            triggered_at=time.time(),
            source_event_id=drift_event.get("event_id"),
        )
        self.triggered_events.append(t)
        return t


class ScheduleTrigger:
    """
    Time-based trigger that fires after a fixed inter-cycle interval.

    Design choice: does NOT fire on the very first should_trigger() call
    when _last_triggered is None.  This prevents a thundering-herd on
    service restart where all instances would immediately try to schedule
    a training cycle.  Operators who want an immediate cycle use
    ManualTrigger instead.

    Attributes:
        interval: Minimum seconds between scheduled fires.
    """

    def __init__(self, interval_seconds: int = 604800):  # default: weekly
        self.interval = interval_seconds
        self._last_triggered: Optional[float] = None

    def should_trigger(self, now: float = None) -> bool:
        """
        Return True iff the interval has elapsed since the last trigger.

        Returns False if record_trigger() has never been called (startup guard).
        """
        now = now or time.time()
        if self._last_triggered is None:
            return False
        return (now - self._last_triggered) >= self.interval

    def record_trigger(self, now: float = None) -> None:
        """
        Stamp the last-triggered timestamp.

        Call this both when a scheduled cycle fires and after service restart
        if a recent cycle is already recorded in persistent state.
        """
        self._last_triggered = now or time.time()

    def fire(self) -> TriggerEvent:
        """
        Produce a TriggerEvent and record the fire timestamp atomically.

        Callers should verify should_trigger() before calling fire().
        """
        self.record_trigger()
        return TriggerEvent(
            trigger_id=str(uuid.uuid4()),
            trigger_type="SCHEDULE",
            triggered_at=time.time(),
        )


class ManualTrigger:
    """
    Operator-initiated training cycle trigger.

    Produces a TriggerEvent immediately without any rate-limit or threshold
    check — those checks live in LoopStateMachine.can_start().
    """

    @staticmethod
    def fire(operator_id: str = "unknown", budget_cap_usd: float = 30.0) -> TriggerEvent:
        """
        Produce a MANUAL TriggerEvent.

        Parameters:
            operator_id:    Identifier of the requesting operator (for audit log).
            budget_cap_usd: Maximum training budget authorised for this cycle.

        Returns:
            TriggerEvent with trigger_type == "MANUAL".
        """
        return TriggerEvent(
            trigger_id=str(uuid.uuid4()),
            trigger_type="MANUAL",
            triggered_at=time.time(),
            operator_id=operator_id,
            budget_cap_usd=budget_cap_usd,
        )
