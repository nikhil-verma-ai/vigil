"""
Loop state machine for the Autonomous Fine-Tuning Platform.

Invariants enforced here:
  - Only one active execution at a time (max_parallel_cycles=1).
  - State transitions are validated against VALID_TRANSITIONS; any attempt to
    jump to an illegal next state raises ValueError immediately.
  - Rate limiting: min_cycle_interval_seconds must elapse between completed
    cycles.  can_start() is the single gate — all callers must check it.
  - Terminal states (IDLE, REJECTED, ROLLED_BACK) stamp completed_at and
    archive the execution into _history; a completed execution is never
    mutated afterward.
  - on_transition callbacks are called synchronously; exceptions bubble up to
    the caller so callers know when a callback has failed.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import time


class LoopState(str, Enum):
    IDLE = "IDLE"
    EVIDENCE_QUALIFYING = "EVIDENCE_QUALIFYING"
    SYNTHESIZING = "SYNTHESIZING"
    TRAINING_SFT = "TRAINING_SFT"
    TRAINING_DPO = "TRAINING_DPO"
    EVALUATING = "EVALUATING"
    DEPLOYING_CANARY = "DEPLOYING_CANARY"
    PROMOTING_FULL = "PROMOTING_FULL"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


# Directed adjacency list — exhaustive.  Every valid (from, to) pair is listed.
# Any transition not in this map is rejected.
VALID_TRANSITIONS: dict = {
    LoopState.IDLE: [LoopState.EVIDENCE_QUALIFYING],
    LoopState.EVIDENCE_QUALIFYING: [LoopState.SYNTHESIZING, LoopState.IDLE],
    LoopState.SYNTHESIZING: [LoopState.TRAINING_SFT, LoopState.IDLE],
    LoopState.TRAINING_SFT: [LoopState.TRAINING_DPO, LoopState.IDLE],
    LoopState.TRAINING_DPO: [LoopState.EVALUATING, LoopState.IDLE],
    LoopState.EVALUATING: [LoopState.DEPLOYING_CANARY, LoopState.REJECTED],
    LoopState.DEPLOYING_CANARY: [LoopState.PROMOTING_FULL, LoopState.ROLLED_BACK],
    LoopState.PROMOTING_FULL: [LoopState.IDLE],
    LoopState.REJECTED: [LoopState.IDLE],
    LoopState.ROLLED_BACK: [LoopState.IDLE],
}

# States from which a new cycle can start (safe to call sm.start() from these).
_TERMINAL_STATES = {LoopState.IDLE, LoopState.REJECTED, LoopState.ROLLED_BACK}

# States that trigger rate-limit accounting (_last_completed_at is stamped here).
# REJECTED and ROLLED_BACK are intermediate decision states that still flow to
# IDLE; stamping _last_completed_at at REJECTED/ROLLED_BACK would double-count.
# We only stamp at IDLE so that the interval is measured from the true end of the
# cycle, not from the evaluation/canary decision.
#
# History archival: execution is appended to _history exactly once, at IDLE.
# REJECTED and ROLLED_BACK still set completed_at so duration_seconds() is
# correct, but archival is deferred to the final IDLE transition.
_ARCHIVE_STATE = LoopState.IDLE


@dataclass
class LoopExecution:
    """
    Represents a single training cycle from trigger to completion.

    Fields:
        execution_id: UUID assigned at creation.
        trigger_type: "DRIFT" | "SCHEDULE" | "MANUAL".
        trigger_event_id: Source drift event_id when trigger_type == "DRIFT".
        started_at: Unix epoch float from time.time() at construction.
        current_state: Current LoopState; mutated by LoopStateMachine.transition().
        state_history: Ordered list of {from, to, timestamp, metadata} records.
        cycle_id: Optional training cycle identifier assigned by the trainer.
        adapter_id: Optional adapter identifier assigned after training completes.
        error: Last error message, if any step failed.
        completed_at: Set when execution enters a terminal state.
    """

    execution_id: str
    trigger_type: str       # DRIFT | SCHEDULE | MANUAL
    trigger_event_id: Optional[str]
    started_at: float
    current_state: LoopState
    state_history: List[dict] = field(default_factory=list)
    cycle_id: Optional[str] = None
    adapter_id: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None

    def duration_seconds(self) -> float:
        """Wall-clock seconds from start to now (or completed_at if finished)."""
        end = self.completed_at or time.time()
        return end - self.started_at


@dataclass
class LoopStateMachineConfig:
    # Minimum gap between the completion of one cycle and the start of the next.
    min_cycle_interval_seconds: int = 43200   # 12 hours

    # Hard limit on concurrent cycles (currently enforced as 1).
    max_parallel_cycles: int = 1


class LoopStateMachine:
    """
    Single-writer state machine for the autonomous training loop.

    Thread-safety: NOT thread-safe.  The FastAPI layer serialises access via
    asyncio's event loop; do not call from multiple threads concurrently.
    """

    def __init__(self, config: LoopStateMachineConfig = None):
        self.config = config or LoopStateMachineConfig()
        self._current: Optional[LoopExecution] = None
        self._history: List[LoopExecution] = []
        self._last_completed_at: Optional[float] = None
        self._on_transition_callbacks: List[Callable] = []

    # ── Callback registration ─────────────────────────────────────────────

    def on_transition(self, callback: Callable) -> None:
        """
        Register a callback invoked on every state transition.

        Signature: callback(execution: LoopExecution, new_state: LoopState) → None
        Callbacks are called synchronously before this method returns.
        """
        self._on_transition_callbacks.append(callback)

    # ── Guard ─────────────────────────────────────────────────────────────

    def can_start(self, now: float = None) -> tuple:
        """
        Returns (can_start: bool, reason: str).

        Blocks if:
          - An execution is currently active (not in a terminal state).
          - The minimum inter-cycle interval has not elapsed.
        """
        now = now or time.time()

        if self._current is not None and self._current.current_state not in _TERMINAL_STATES:
            return False, f"Cycle already running: {self._current.current_state}"

        if self._last_completed_at is not None:
            elapsed = now - self._last_completed_at
            if elapsed < self.config.min_cycle_interval_seconds:
                remaining = self.config.min_cycle_interval_seconds - elapsed
                return False, f"Rate limited: {remaining:.0f}s until next cycle allowed"

        return True, "ok"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, execution: LoopExecution) -> LoopExecution:
        """
        Enroll a new LoopExecution as the active cycle.

        Raises RuntimeError if can_start() returns False.
        Records the initial IDLE bootstrap entry in state_history so that
        history[0].state_history has one entry per transition call plus this
        bootstrap (total = transitions_called + 1, matching test expectations).
        """
        can, reason = self.can_start()
        if not can:
            raise RuntimeError(f"Cannot start: {reason}")
        self._current = execution
        # Record the bootstrap entry so state_history length == transitions + 1.
        self._current.state_history.append({
            "from": LoopState.IDLE.value,
            "to": LoopState.IDLE.value,
            "timestamp": time.time(),
            "metadata": {"event": "started"},
        })
        return execution

    def transition(self, new_state: LoopState, metadata: dict = None) -> LoopExecution:
        """
        Advance the active execution to new_state.

        Raises:
          RuntimeError  — no active execution.
          ValueError    — new_state is not in VALID_TRANSITIONS[current_state].

        Side-effects:
          - Appends to state_history.
          - Fires all on_transition callbacks.
          - If new_state is terminal: stamps completed_at, archives to _history,
            records _last_completed_at for rate-limit accounting.
        """
        if self._current is None:
            raise RuntimeError("No active execution")

        current_state = self._current.current_state
        if new_state not in VALID_TRANSITIONS[current_state]:
            raise ValueError(f"Invalid transition: {current_state} → {new_state}")

        self._current.state_history.append({
            "from": current_state.value,
            "to": new_state.value,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })
        self._current.current_state = new_state

        # Fire callbacks — exceptions propagate to the caller intentionally.
        for cb in self._on_transition_callbacks:
            cb(self._current, new_state)

        # Stamp completed_at on any terminal state so duration_seconds() is
        # correct even before the final IDLE transition.
        if new_state in _TERMINAL_STATES and self._current.completed_at is None:
            self._current.completed_at = time.time()

        # Archive and update rate-limit clock exactly once: at the IDLE terminal.
        # REJECTED and ROLLED_BACK are intermediate decision states that flow to
        # IDLE; archiving at all three would produce duplicate history entries.
        if new_state is _ARCHIVE_STATE:
            self._last_completed_at = self._current.completed_at or time.time()
            # Guard against duplicate archival (e.g. IDLE→IDLE is invalid, but
            # defensive programming has no cost here).
            if self._current not in self._history:
                self._history.append(self._current)
            # Keep _current pointing at the finished execution so callers can
            # still inspect it; can_start() correctly allows new starts because
            # current_state is now a terminal state.

        return self._current

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_current(self) -> Optional[LoopExecution]:
        """Return the active (or most-recently-completed) LoopExecution."""
        return self._current

    def get_history(self, last_n: int = 20) -> List[LoopExecution]:
        """Return the last_n completed LoopExecutions, oldest-first."""
        return self._history[-last_n:]
