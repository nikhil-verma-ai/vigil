"""
Integration tests for Module 8: Autonomous Loop Orchestrator.

Test coverage:
  - LoopStateMachine: full happy path, rejection path, rollback path,
    rate limiting (block + allow after interval), transition callbacks.
  - DriftTrigger: fires on CRITICAL/EMERGENCY qualified events above threshold,
    ignores WARNING, ignores unqualified CRITICAL.
  - ScheduleTrigger: fires after interval, does not fire before interval.

All tests are pure in-process — no Kafka, no Redis, no network I/O.
"""

import pytest
import time
import uuid

from services.orchestrator.state_machine import (
    LoopStateMachine,
    LoopStateMachineConfig,
    LoopState,
    LoopExecution,
)
from services.orchestrator.triggers import DriftTrigger, ScheduleTrigger


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_execution(trigger_type: str = "DRIFT") -> LoopExecution:
    return LoopExecution(
        execution_id=str(uuid.uuid4()),
        trigger_type=trigger_type,
        trigger_event_id=str(uuid.uuid4()),
        started_at=time.time(),
        current_state=LoopState.IDLE,
    )


# ── State machine — happy path ────────────────────────────────────────────────

def test_loop_state_machine_full_happy_path():
    """Full successful cycle: IDLE → ... → IDLE."""
    sm = LoopStateMachine()
    exec = LoopExecution(str(uuid.uuid4()), "DRIFT", None, time.time(), LoopState.IDLE)
    sm.start(exec)
    sm.transition(LoopState.EVIDENCE_QUALIFYING)
    sm.transition(LoopState.SYNTHESIZING)
    sm.transition(LoopState.TRAINING_SFT)
    sm.transition(LoopState.TRAINING_DPO)
    sm.transition(LoopState.EVALUATING)
    sm.transition(LoopState.DEPLOYING_CANARY)
    sm.transition(LoopState.PROMOTING_FULL)
    sm.transition(LoopState.IDLE)

    history = sm.get_history()
    assert len(history) == 1
    assert history[0].completed_at is not None
    assert len(history[0].state_history) == 9


# ── State machine — rejection path ────────────────────────────────────────────

def test_loop_state_machine_rejection_path():
    """Evaluation failure: EVALUATING → REJECTED → IDLE."""
    sm = LoopStateMachine()
    exec = LoopExecution(str(uuid.uuid4()), "DRIFT", None, time.time(), LoopState.IDLE)
    sm.start(exec)
    for state in [
        LoopState.EVIDENCE_QUALIFYING,
        LoopState.SYNTHESIZING,
        LoopState.TRAINING_SFT,
        LoopState.TRAINING_DPO,
        LoopState.EVALUATING,
    ]:
        sm.transition(state)
    sm.transition(LoopState.REJECTED)
    sm.transition(LoopState.IDLE)

    history = sm.get_history()
    assert len(history) == 1
    assert LoopState.REJECTED.value in [h["to"] for h in history[0].state_history]


# ── State machine — rollback path ─────────────────────────────────────────────

def test_loop_state_machine_rollback_path():
    """Canary failure: DEPLOYING_CANARY → ROLLED_BACK → IDLE."""
    sm = LoopStateMachine()
    exec = LoopExecution(str(uuid.uuid4()), "DRIFT", None, time.time(), LoopState.IDLE)
    sm.start(exec)
    for state in [
        LoopState.EVIDENCE_QUALIFYING,
        LoopState.SYNTHESIZING,
        LoopState.TRAINING_SFT,
        LoopState.TRAINING_DPO,
        LoopState.EVALUATING,
        LoopState.DEPLOYING_CANARY,
    ]:
        sm.transition(state)
    sm.transition(LoopState.ROLLED_BACK)
    sm.transition(LoopState.IDLE)

    history = sm.get_history()
    assert LoopState.ROLLED_BACK.value in [h["to"] for h in history[0].state_history]


# ── Rate limiting ─────────────────────────────────────────────────────────────

def test_loop_rate_limiting():
    """Second cycle within 12 h window must be blocked."""
    config = LoopStateMachineConfig(min_cycle_interval_seconds=43200)
    sm = LoopStateMachine(config)
    exec1 = LoopExecution(str(uuid.uuid4()), "DRIFT", None, time.time(), LoopState.IDLE)
    sm.start(exec1)
    sm.transition(LoopState.EVIDENCE_QUALIFYING)
    sm.transition(LoopState.IDLE)   # complete it

    can_start, reason = sm.can_start()
    assert can_start is False
    assert "Rate limited" in reason


def test_loop_rate_limit_allows_after_interval():
    """After min interval passes, new cycle must be allowed."""
    config = LoopStateMachineConfig(min_cycle_interval_seconds=1)  # 1 second for test speed
    sm = LoopStateMachine(config)
    exec1 = LoopExecution(str(uuid.uuid4()), "DRIFT", None, time.time(), LoopState.IDLE)
    sm.start(exec1)
    sm.transition(LoopState.EVIDENCE_QUALIFYING)
    sm.transition(LoopState.IDLE)

    time.sleep(1.1)
    can_start, reason = sm.can_start()
    assert can_start is True


# ── DriftTrigger ──────────────────────────────────────────────────────────────

def test_drift_trigger_fires_on_critical():
    """CRITICAL qualified drift event must produce a TriggerEvent."""
    trigger = DriftTrigger(critical_score_threshold=4.5)
    event = {
        "event_id": str(uuid.uuid4()),
        "alert_level": "CRITICAL",
        "qualification_status": "QUALIFIED",
        "composite_anomaly_score": 5.2,
    }
    result = trigger.evaluate(event)
    assert result is not None
    assert result.trigger_type == "DRIFT"
    assert result.source_event_id == event["event_id"]


def test_drift_trigger_fires_on_emergency():
    """EMERGENCY qualified drift event must also produce a TriggerEvent."""
    trigger = DriftTrigger(critical_score_threshold=4.5)
    event = {
        "event_id": str(uuid.uuid4()),
        "alert_level": "EMERGENCY",
        "qualification_status": "QUALIFIED",
        "composite_anomaly_score": 6.0,
    }
    result = trigger.evaluate(event)
    assert result is not None
    assert result.trigger_type == "DRIFT"


def test_drift_trigger_ignores_warning():
    """WARNING level must not trigger training."""
    trigger = DriftTrigger()
    event = {
        "event_id": "x",
        "alert_level": "WARNING",
        "qualification_status": "QUALIFIED",
        "composite_anomaly_score": 3.5,
    }
    assert trigger.evaluate(event) is None


def test_drift_trigger_ignores_unqualified():
    """CRITICAL but PENDING qualification must not trigger."""
    trigger = DriftTrigger()
    event = {
        "event_id": "x",
        "alert_level": "CRITICAL",
        "qualification_status": "PENDING",
        "composite_anomaly_score": 5.0,
    }
    assert trigger.evaluate(event) is None


def test_drift_trigger_ignores_below_threshold():
    """CRITICAL, QUALIFIED, but score below threshold must not trigger."""
    trigger = DriftTrigger(critical_score_threshold=4.5)
    event = {
        "event_id": "x",
        "alert_level": "CRITICAL",
        "qualification_status": "QUALIFIED",
        "composite_anomaly_score": 4.4,
    }
    assert trigger.evaluate(event) is None


# ── ScheduleTrigger ───────────────────────────────────────────────────────────

def test_schedule_trigger_fires_after_interval():
    trigger = ScheduleTrigger(interval_seconds=1)
    trigger.record_trigger(time.time() - 2)   # last trigger was 2 seconds ago
    assert trigger.should_trigger() is True


def test_schedule_trigger_does_not_fire_before_interval():
    trigger = ScheduleTrigger(interval_seconds=604800)
    trigger.record_trigger(time.time())        # just triggered
    assert trigger.should_trigger() is False


def test_schedule_trigger_does_not_fire_on_startup():
    """schedule_trigger must not fire before record_trigger() is ever called."""
    trigger = ScheduleTrigger(interval_seconds=1)
    # No record_trigger() call — simulates fresh startup.
    assert trigger.should_trigger() is False


# ── Transition callbacks ──────────────────────────────────────────────────────

def test_transition_callbacks_fired():
    """on_transition callbacks must be called on every state change."""
    sm = LoopStateMachine()
    transitions_seen = []
    sm.on_transition(lambda exec, state: transitions_seen.append(state))

    exec = LoopExecution(str(uuid.uuid4()), "MANUAL", None, time.time(), LoopState.IDLE)
    sm.start(exec)
    sm.transition(LoopState.EVIDENCE_QUALIFYING)
    sm.transition(LoopState.IDLE)

    assert LoopState.EVIDENCE_QUALIFYING in transitions_seen
    assert LoopState.IDLE in transitions_seen


def test_multiple_callbacks_all_fired():
    """Multiple registered callbacks must all fire on each transition."""
    sm = LoopStateMachine()
    seen_a = []
    seen_b = []
    sm.on_transition(lambda e, s: seen_a.append(s))
    sm.on_transition(lambda e, s: seen_b.append(s))

    exec = LoopExecution(str(uuid.uuid4()), "MANUAL", None, time.time(), LoopState.IDLE)
    sm.start(exec)
    sm.transition(LoopState.EVIDENCE_QUALIFYING)
    sm.transition(LoopState.IDLE)

    assert seen_a == seen_b
    assert LoopState.EVIDENCE_QUALIFYING in seen_a


# ── Invalid transitions ───────────────────────────────────────────────────────

def test_invalid_transition_raises():
    """Attempting an illegal state jump must raise ValueError."""
    sm = LoopStateMachine()
    exec = make_execution()
    sm.start(exec)
    sm.transition(LoopState.EVIDENCE_QUALIFYING)

    with pytest.raises(ValueError, match="Invalid transition"):
        sm.transition(LoopState.TRAINING_SFT)   # must go through SYNTHESIZING first


def test_transition_without_active_execution_raises():
    """transition() without a prior start() must raise RuntimeError."""
    sm = LoopStateMachine()
    with pytest.raises(RuntimeError, match="No active execution"):
        sm.transition(LoopState.EVIDENCE_QUALIFYING)


# ── History ───────────────────────────────────────────────────────────────────

def test_history_accumulates_across_cycles():
    """Multiple completed cycles should all appear in history."""
    config = LoopStateMachineConfig(min_cycle_interval_seconds=0)  # no rate limit
    sm = LoopStateMachine(config)

    for _ in range(3):
        exec = make_execution()
        sm.start(exec)
        sm.transition(LoopState.EVIDENCE_QUALIFYING)
        sm.transition(LoopState.IDLE)

    history = sm.get_history(last_n=10)
    assert len(history) == 3


def test_history_respects_last_n():
    """get_history(last_n=2) returns only the 2 most recent executions."""
    config = LoopStateMachineConfig(min_cycle_interval_seconds=0)
    sm = LoopStateMachine(config)

    for _ in range(5):
        exec = make_execution()
        sm.start(exec)
        sm.transition(LoopState.EVIDENCE_QUALIFYING)
        sm.transition(LoopState.IDLE)

    assert len(sm.get_history(last_n=2)) == 2
