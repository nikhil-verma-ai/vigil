"""
FastAPI service entrypoint for the Training Orchestrator (Module 4b).

Endpoints:
  POST /cycles/run          — trigger a full SFT→DPO training cycle
  GET  /cycles/{cycle_id}   — fetch status of an in-progress or completed cycle
  GET  /healthz             — liveness probe
  GET  /readyz              — readiness probe
  GET  /metrics             — Prometheus text-format metrics

Background:
  Cycle runs are executed in a thread-pool executor so the event loop stays
  responsive.  A single in-progress cycle is tracked in module state; a real
  deployment would use a job queue (Celery / Ray / Temporal) for multiple
  concurrent cycles.

Prometheus metrics:
  training_cycles_total{status}          — counter per terminal status
  training_cycle_duration_seconds        — histogram of wall-clock cycle time
  training_cycle_cost_usd                — histogram of per-cycle USD cost
  gpu_provisioning_total                 — counter of provision calls
"""
from __future__ import annotations

import asyncio
import datetime
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

import structlog
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from services.trainer.orchestrator import TrainingOrchestrator, TrainingCycleConfig, OrchestratorResult
from services.trainer.gpu_provisioner import MockGPUProvisioner, GPUProvisioner
from services.trainer.cost_tracker import CostTracker
from services.trainer.checkpointing import CheckpointManager
from services.trainer.sft import MockSFTJob, SFTTrainingJob
from services.trainer.dpo import MockDPOJob, DPOTrainingJob
from services.trainer.qlora_config import SFTTrainingConfig
from services.trainer.dpo import DPOConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

CYCLES_TOTAL = Counter(
    "training_cycles_total",
    "Total training cycles by terminal status",
    labelnames=["status"],
)

CYCLE_DURATION = Histogram(
    "training_cycle_duration_seconds",
    "Wall-clock duration of training cycles",
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400, 28800],
)

CYCLE_COST = Histogram(
    "training_cycle_cost_usd",
    "Per-cycle GPU cost in USD",
    buckets=[1, 5, 10, 15, 20, 25, 30, 40, 60],
)

GPU_PROVISIONS = Counter(
    "gpu_provisioning_total",
    "Total GPU provisioning attempts",
)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Completed and in-progress cycle results indexed by cycle_id
_cycle_results: Dict[str, OrchestratorResult] = {}
_in_progress: Dict[str, bool] = {}

_orchestrator: Optional[TrainingOrchestrator] = None
_output_base_dir: str = "/tmp/training-cycles"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise orchestrator components on startup; flush on shutdown."""
    global _orchestrator, _output_base_dir

    log.info("trainer_service_starting")

    # Output directory for cycle artifacts
    _output_base_dir = os.environ.get("TRAINING_OUTPUT_DIR", "/tmp/training-cycles")
    os.makedirs(_output_base_dir, exist_ok=True)

    # Decide whether to run in mock mode (for local dev / CI)
    mock_mode = os.environ.get("TRAINING_MOCK_MODE", "true").lower() == "true"

    # Build Kafka producer (optional — graceful degradation)
    kafka_producer = _try_init_kafka_producer()

    # Budget
    max_budget = float(os.environ.get("MAX_CYCLE_BUDGET_USD", "30.0"))

    cost_tracker = CostTracker(max_cycle_budget_usd=max_budget)
    checkpoint_manager = CheckpointManager(_output_base_dir)

    if mock_mode:
        log.info("trainer_using_mock_mode")
        provisioner = MockGPUProvisioner()

        def sft_job_factory(config, out_dir):
            return MockSFTJob(config, out_dir)

        def dpo_job_factory(config, out_dir):
            return MockDPOJob(config, out_dir)
    else:
        cloud = os.environ.get("CLOUD_PROVIDER", "aws")
        provisioner = GPUProvisioner(cloud_provider=cloud)

        def sft_job_factory(config, out_dir):
            return SFTTrainingJob(config, out_dir)

        def dpo_job_factory(config, out_dir):
            return DPOTrainingJob(config, out_dir)

    _orchestrator = TrainingOrchestrator(
        provisioner=provisioner,
        cost_tracker=cost_tracker,
        sft_job_factory=sft_job_factory,
        dpo_job_factory=dpo_job_factory,
        checkpoint_manager=checkpoint_manager,
        kafka_producer=kafka_producer,
    )

    log.info("trainer_service_ready", mock_mode=mock_mode)
    yield

    log.info("trainer_service_shutting_down")
    if kafka_producer is not None:
        try:
            kafka_producer.flush(timeout=5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Training Orchestrator",
    description="Module 4b — SFT→DPO training pipeline service",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RunCycleRequest(BaseModel):
    """
    Payload to trigger a new training cycle.

    All path fields must be accessible from the service container.
    """
    base_model_id: str = Field(
        ...,
        description="HuggingFace model ID or local path; 'mock://' prefix enables mock mode",
        examples=["mock://llama-7b", "meta-llama/Llama-2-7b-hf"],
    )
    sft_dataset_path: str = Field(
        ...,
        description="Absolute path to {prompt, response} JSONL for SFT phase",
    )
    dpo_dataset_path: str = Field(
        ...,
        description="Absolute path to {prompt, chosen, rejected} JSONL for DPO phase",
    )
    max_cost_usd: float = Field(
        default=30.0,
        gt=0,
        description="Hard budget ceiling in USD; cycle aborts if exceeded",
    )
    trigger_event_id: Optional[str] = Field(
        default=None,
        description="Drift event ID that triggered this cycle (for tracing)",
    )
    triggered_by: str = Field(
        default="MANUAL",
        description="TriggerType: DRIFT | SCHEDULE | MANUAL",
    )


class CycleStatusResponse(BaseModel):
    cycle_id: str
    status: str
    total_cost_usd: float
    gpu_hours: float
    candidate_adapter_path: str
    sft_steps: Optional[int] = None
    dpo_steps: Optional[int] = None
    dpo_reward_accuracy: Optional[float] = None
    error: Optional[str] = None
    in_progress: bool = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> dict:
    """Liveness probe — always 200 if the process is alive."""
    return {
        "status": "ok",
        "cycles_completed": len(_cycle_results),
        "cycles_in_progress": sum(_in_progress.values()),
    }


@app.get("/readyz")
async def readyz() -> dict:
    """Readiness probe — 503 until orchestrator is initialised."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")
    return {"status": "ready"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus text-format metrics endpoint."""
    return generate_latest().decode("utf-8")


@app.post("/cycles/run", status_code=202)
async def run_cycle(req: RunCycleRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Trigger a new SFT→DPO training cycle.

    The cycle runs asynchronously in a background thread.
    Poll GET /cycles/{cycle_id} to track progress.
    Returns 202 Accepted immediately with the assigned cycle_id.
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")

    cycle_id = f"cycle-{uuid.uuid4().hex[:12]}"
    _in_progress[cycle_id] = True

    cycle_config = TrainingCycleConfig(
        cycle_id=cycle_id,
        base_model_id=req.base_model_id,
        sft_dataset_path=req.sft_dataset_path,
        dpo_dataset_path=req.dpo_dataset_path,
        output_base_dir=_output_base_dir,
        max_cost_usd=req.max_cost_usd,
        trigger_event_id=req.trigger_event_id,
        triggered_by=req.triggered_by,
    )

    background_tasks.add_task(_run_cycle_background, cycle_config)

    log.info("cycle_enqueued", cycle_id=cycle_id)
    return {"cycle_id": cycle_id, "status": "RUNNING"}


@app.get("/cycles/{cycle_id}", response_model=CycleStatusResponse)
async def get_cycle_status(cycle_id: str) -> CycleStatusResponse:
    """
    Fetch the current status of a training cycle.

    Returns 404 if the cycle_id is unknown.
    Returns in_progress=True if the cycle is still running.
    """
    in_progress = _in_progress.get(cycle_id, False)
    result = _cycle_results.get(cycle_id)

    if result is None and not in_progress:
        raise HTTPException(status_code=404, detail=f"Cycle '{cycle_id}' not found")

    if result is None:
        # Still running — return minimal status
        return CycleStatusResponse(
            cycle_id=cycle_id,
            status="RUNNING",
            total_cost_usd=0.0,
            gpu_hours=0.0,
            candidate_adapter_path="",
            in_progress=True,
        )

    return CycleStatusResponse(
        cycle_id=result.cycle_id,
        status=result.status,
        total_cost_usd=result.total_cost_usd,
        gpu_hours=result.gpu_hours,
        candidate_adapter_path=result.candidate_adapter_path,
        sft_steps=(result.sft_result.training_steps if result.sft_result else None),
        dpo_steps=(result.dpo_result.training_steps if result.dpo_result else None),
        dpo_reward_accuracy=(
            result.dpo_result.reward_accuracy if result.dpo_result else None
        ),
        error=result.error,
        in_progress=in_progress,
    )


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

async def _run_cycle_background(config: TrainingCycleConfig) -> None:
    """
    Run a training cycle in the default ThreadPoolExecutor.

    Purpose:  keep the asyncio event loop unblocked during CPU/IO-heavy training.
    Invariant: _in_progress[cycle_id] is always cleaned up, even on exception.
    """
    loop = asyncio.get_event_loop()
    try:
        t0 = loop.time()
        result = await loop.run_in_executor(None, _orchestrator.run_cycle, config)
        elapsed = loop.time() - t0

        _cycle_results[config.cycle_id] = result

        CYCLES_TOTAL.labels(status=result.status).inc()
        CYCLE_DURATION.observe(elapsed)
        CYCLE_COST.observe(result.total_cost_usd)

        log.info(
            "cycle_background_complete",
            cycle_id=config.cycle_id,
            status=result.status,
            cost_usd=result.total_cost_usd,
            duration_s=elapsed,
        )
    except Exception as exc:
        log.error(
            "cycle_background_exception",
            cycle_id=config.cycle_id,
            error=str(exc),
        )
        CYCLES_TOTAL.labels(status="FAILED").inc()
    finally:
        _in_progress.pop(config.cycle_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_init_kafka_producer():
    """Attempt to build a Kafka producer; return None if unavailable."""
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
    if not bootstrap:
        log.info("kafka_disabled_no_bootstrap_servers")
        return None
    try:
        from confluent_kafka import Producer
        producer = Producer({
            "bootstrap.servers": bootstrap,
            "linger.ms": 5,
            "compression.type": "lz4",
            "acks": "all",
        })
        log.info("kafka_producer_initialised", bootstrap=bootstrap)
        return producer
    except Exception as exc:
        log.warning("kafka_producer_init_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "services.trainer.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8004)),
        log_level="info",
        workers=1,  # Single worker — cycle state is in-process
    )
