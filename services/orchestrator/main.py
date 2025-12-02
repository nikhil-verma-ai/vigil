"""
FastAPI service entry point for the Autonomous Loop Orchestrator.

Endpoints:
  GET  /healthz                  — liveness probe
  GET  /metrics                  — Prometheus scrape endpoint
  GET  /api/v1/loop/status       — current state + active execution
  POST /api/v1/loop/trigger      — manual training cycle trigger
  GET  /api/v1/loop/history      — last N completed executions

The AutonomousLoop and all its dependencies are wired at startup in
_build_loop().  For local development with mocks, set the env var
MOCK_DEPENDENCIES=1.

Metrics emitted:
  loop_cycles_total{trigger_type, terminal_state}
  loop_cycle_duration_seconds{trigger_type}
  loop_current_state{state}
"""

import os
import time
import uuid
import structlog
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from services.orchestrator.state_machine import (
    LoopStateMachine,
    LoopStateMachineConfig,
    LoopState,
    LoopExecution,
)
from services.orchestrator.triggers import DriftTrigger, ScheduleTrigger, TriggerEvent
from services.orchestrator.loop import AutonomousLoop, LoopConfig

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

CYCLES_TOTAL = Counter(
    "loop_cycles_total",
    "Total completed training cycles",
    ["trigger_type", "terminal_state"],
)
CYCLE_DURATION = Histogram(
    "loop_cycle_duration_seconds",
    "Wall-clock duration of completed training cycles",
    ["trigger_type"],
    buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800],
)
CURRENT_STATE = Gauge(
    "loop_current_state_info",
    "Current loop state (label carries state name, value always 1)",
    ["state"],
)

# ── Global service state ──────────────────────────────────────────────────────
# Module-level so tests can introspect.

_loop: Optional[AutonomousLoop] = None
_sm: Optional[LoopStateMachine] = None


# ── Dependency wiring ─────────────────────────────────────────────────────────

def _build_loop() -> AutonomousLoop:
    """
    Construct AutonomousLoop with real or mock dependencies.

    When MOCK_DEPENDENCIES=1 (CI / local dev without GPU), every injected
    component is replaced with a deterministic in-process stub that returns
    success immediately.
    """
    mock_mode = os.environ.get("MOCK_DEPENDENCIES", "0") == "1"

    sm_config = LoopStateMachineConfig(
        min_cycle_interval_seconds=int(os.environ.get("MIN_CYCLE_INTERVAL_SECONDS", "43200")),
    )
    sm = LoopStateMachine(sm_config)

    # Register Prometheus state-tracking callback.
    def _on_transition(execution: LoopExecution, new_state: LoopState) -> None:
        CURRENT_STATE.labels(state=new_state.value).set(1)
        if new_state in (LoopState.IDLE, LoopState.REJECTED, LoopState.ROLLED_BACK):
            CYCLES_TOTAL.labels(
                trigger_type=execution.trigger_type,
                terminal_state=new_state.value,
            ).inc()
            CYCLE_DURATION.labels(trigger_type=execution.trigger_type).observe(
                execution.duration_seconds()
            )

    sm.on_transition(_on_transition)

    drift_trigger = DriftTrigger(
        critical_score_threshold=float(os.environ.get("DRIFT_SCORE_THRESHOLD", "4.5"))
    )
    schedule_trigger = ScheduleTrigger(
        interval_seconds=int(os.environ.get("SCHEDULE_INTERVAL_SECONDS", "604800"))
    )

    if mock_mode:
        synthesis_pipeline = _MockSynthesisPipeline()
        training_orchestrator = _MockTrainingOrchestrator()
        evaluation_gate = _MockEvaluationGate()
        deployment_engine = _MockDeploymentEngine()
        kafka_producer = None
    else:
        # Real implementations — imported lazily to keep startup fast in mock mode.
        from confluent_kafka import Producer  # type: ignore[import]
        kafka_producer = Producer(
            {"bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")}
        )
        # Real service clients must be wired here per deployment config.
        raise NotImplementedError(
            "Real dependency wiring not implemented — set MOCK_DEPENDENCIES=1 for local dev"
        )

    loop_config = LoopConfig(
        cluster_id=os.environ.get("CLUSTER_ID", "default"),
        max_cost_usd=float(os.environ.get("MAX_COST_USD", "30.0")),
    )

    return AutonomousLoop(
        state_machine=sm,
        drift_trigger=drift_trigger,
        schedule_trigger=schedule_trigger,
        synthesis_pipeline=synthesis_pipeline,
        training_orchestrator=training_orchestrator,
        evaluation_gate=evaluation_gate,
        deployment_engine=deployment_engine,
        kafka_producer=kafka_producer,
        config=loop_config,
    ), sm


# ── Mock stubs (used when MOCK_DEPENDENCIES=1) ────────────────────────────────

class _SynthesisResult:
    def __init__(self, success: bool, job_id: str, error: str = ""):
        self.success = success
        self.job_id = job_id
        self.error = error


class _TrainingResult:
    def __init__(self, success: bool, cycle_id: str, adapter_id: str, error: str = ""):
        self.success = success
        self.cycle_id = cycle_id
        self.adapter_id = adapter_id
        self.error = error


class _GateDecision:
    def __init__(self, passed: bool):
        self.passed = passed


class _PromotionResult:
    def __init__(self, success: bool, error: str = ""):
        self.success = success
        self.error = error


class _MockSynthesisPipeline:
    def run(self, failures, job_id, trigger_id):
        return _SynthesisResult(success=True, job_id=job_id)


class _MockTrainingOrchestrator:
    def run_cycle(self, config):
        return _TrainingResult(
            success=True,
            cycle_id=str(uuid.uuid4()),
            adapter_id=f"adapter-{uuid.uuid4().hex[:8]}",
        )


class _MockEvaluationGate:
    def evaluate(self, adapter_id, prompts, cluster_id):
        return _GateDecision(passed=True)


class _MockDeploymentEngine:
    def promote(self, config):
        return _PromotionResult(success=True)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop, _sm
    try:
        result = _build_loop()
        _loop, _sm = result
        logger.info("orchestrator.started")
    except NotImplementedError:
        # Real dependencies not wired — only usable with MOCK_DEPENDENCIES=1.
        logger.warning("orchestrator.mock_only", msg="Set MOCK_DEPENDENCIES=1")
    yield
    logger.info("orchestrator.shutdown")


app = FastAPI(
    title="Autonomous Loop Orchestrator",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / response models ─────────────────────────────────────────────────

class ManualTriggerRequest(BaseModel):
    operator_id: str = Field(default="unknown", description="Identity of the triggering operator")
    budget_cap_usd: float = Field(default=30.0, ge=0.0, description="Max training budget for this cycle")
    reason: Optional[str] = Field(default=None, description="Free-text reason for manual trigger")


class ExecutionSummary(BaseModel):
    execution_id: str
    trigger_type: str
    current_state: str
    started_at: float
    completed_at: Optional[float]
    duration_seconds: float
    cycle_id: Optional[str]
    adapter_id: Optional[str]
    error: Optional[str]
    state_history: list


def _summarise(ex: LoopExecution) -> ExecutionSummary:
    return ExecutionSummary(
        execution_id=ex.execution_id,
        trigger_type=ex.trigger_type,
        current_state=ex.current_state.value,
        started_at=ex.started_at,
        completed_at=ex.completed_at,
        duration_seconds=ex.duration_seconds(),
        cycle_id=ex.cycle_id,
        adapter_id=ex.adapter_id,
        error=ex.error,
        state_history=ex.state_history,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/healthz", tags=["ops"])
async def healthz():
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@app.get("/metrics", tags=["ops"])
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/loop/status", tags=["loop"])
async def loop_status():
    """
    Return current loop state, active execution details, and rate-limit info.

    Response shape:
      {
        "current_state": "IDLE" | ...,
        "active_execution": ExecutionSummary | null,
        "can_start": bool,
        "can_start_reason": str,
        "last_completed_at": float | null,
      }
    """
    if _sm is None:
        raise HTTPException(503, "Orchestrator not initialised")

    can_start, reason = _sm.can_start()
    current = _sm.get_current()

    return {
        "current_state": current.current_state.value if current else LoopState.IDLE.value,
        "active_execution": _summarise(current) if current else None,
        "can_start": can_start,
        "can_start_reason": reason,
        "last_completed_at": _sm._last_completed_at,
    }


@app.post("/api/v1/loop/trigger", tags=["loop"], status_code=202)
async def manual_trigger(body: ManualTriggerRequest):
    """
    Manually trigger a training cycle.

    Returns 202 Accepted with the execution_id if started.
    Returns 409 Conflict if the state machine is rate-limited or busy.
    """
    if _loop is None:
        raise HTTPException(503, "Orchestrator not initialised")

    can_start, reason = _sm.can_start()
    if not can_start:
        raise HTTPException(409, detail=f"Cannot start: {reason}")

    try:
        execution_id = _loop.trigger_manual(
            operator_id=body.operator_id,
            budget_cap_usd=body.budget_cap_usd,
        )
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))

    logger.info(
        "loop.manual_triggered",
        execution_id=execution_id,
        operator_id=body.operator_id,
        reason=body.reason,
    )
    return {"execution_id": execution_id, "status": "started"}


@app.get("/api/v1/loop/history", tags=["loop"])
async def loop_history(last_n: int = 20):
    """
    Return the last N completed training cycle executions.

    Query param: last_n (default 20, max 100).
    """
    if _sm is None:
        raise HTTPException(503, "Orchestrator not initialised")

    last_n = min(last_n, 100)
    history = _sm.get_history(last_n)
    return {
        "count": len(history),
        "executions": [_summarise(ex) for ex in history],
    }
