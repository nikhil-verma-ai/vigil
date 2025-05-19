"""
Synthesizer service entry point.

Provides:
  - FastAPI REST API for health checks and manual synthesis triggers.
  - Kafka consumer loop that processes drift.events -> synthesis jobs.

API routes
----------
GET  /health           — liveness probe
GET  /ready            — readiness probe (checks Kafka connectivity)
POST /synthesize       — trigger synthesis manually with a JSON body
GET  /metrics          — Prometheus metrics (plain text)

Kafka consumer
--------------
Topic:    drift.events
Group:    synthesizer-group
Action:   on each qualifying DriftEvent, run SynthesisPipeline.run()

Observability
-------------
- Prometheus counters/histograms via prometheus_client.
- Structured logs via structlog.
- Every log entry carries: job_id, trigger_event_id, cluster_count, cost_usd.

Environment variables (all optional — fall back to SynthesizerConfig defaults)
---------------------------------------------------------------------------
OPENAI_API_KEY          str   — required for real oracle/judge calls
ORACLE_MODEL            str   — default: gpt-4o-mini
JUDGE_MODEL             str   — default: gpt-4o-mini
JUDGE_PASS_THRESHOLD    float — default: 0.7
KAFKA_BOOTSTRAP_SERVERS str   — default: localhost:9092
OUTPUT_BASE_DIR         str   — default: /tmp/synthesis
EMBEDDING_MODEL         str   — default: sentence-transformers/all-MiniLM-L6-v2
HDBSCAN_MIN_CLUSTER_SIZE int  — default: 10
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel

from services.synthesizer.amplifier import SyntheticAmplifier
from services.synthesizer.clustering import FailureClusterer
from services.synthesizer.config import SynthesizerConfig
from services.synthesizer.embedding import EmbeddingEngine, FailureRecord
from services.synthesizer.judge import LLMJudge
from services.synthesizer.pipeline import SynthesisPipeline, SynthesisJobResult

# --------------------------------------------------------------------------- #
# Structured logging                                                           #
# --------------------------------------------------------------------------- #

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger("synthesizer")

# --------------------------------------------------------------------------- #
# Prometheus metrics                                                           #
# --------------------------------------------------------------------------- #

SYNTHESIS_JOBS_TOTAL = Counter(
    "synthesizer_jobs_total",
    "Total synthesis jobs triggered",
    ["trigger_type"],  # "kafka" | "api"
)
SYNTHESIS_JOBS_FAILED = Counter(
    "synthesizer_jobs_failed_total",
    "Synthesis jobs that raised an exception",
)
PAIRS_SYNTHESIZED = Counter(
    "synthesizer_pairs_synthesized_total",
    "Total preference pairs synthesized",
)
PAIRS_PASSING_GATE = Counter(
    "synthesizer_pairs_passing_gate_total",
    "Preference pairs that passed the LLM judge quality gate",
)
SYNTHESIS_DURATION_SECONDS = Histogram(
    "synthesizer_job_duration_seconds",
    "Wall-clock time per synthesis job",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)
SYNTHESIS_COST_USD = Counter(
    "synthesizer_cost_usd_total",
    "Cumulative estimated cost in USD for oracle + judge calls",
)

# --------------------------------------------------------------------------- #
# FastAPI application                                                          #
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Synthesizer Service",
    version="1.0.0",
    description="LLM-as-Judge quality gate and DPO synthesis pipeline",
)


class SynthesizeRequest(BaseModel):
    """
    Request body for POST /synthesize.

    Fields
    ------
    failure_records: list of raw failure events.
    trigger_event_id: optional ID linking this job to a drift event.
    """
    failure_records: List[Dict[str, Any]]
    trigger_event_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Dependency construction                                                      #
# --------------------------------------------------------------------------- #

def _build_pipeline(config: SynthesizerConfig) -> SynthesisPipeline:
    """
    Wire together all pipeline dependencies from config.

    If OPENAI_API_KEY is set, constructs a real OpenAI client.
    Otherwise falls back to MockOracleClient + MockJudgeClient for
    development / integration test environments.
    """
    from services.synthesizer.amplifier import MockOracleClient
    from services.synthesizer.judge import MockJudgeClient

    embedder = EmbeddingEngine(model_name=config.embedding_model)
    clusterer = FailureClusterer(config=config, embedding_engine=embedder)

    if config.openai_api_key:
        try:
            from openai import OpenAI  # type: ignore

            openai_client = OpenAI(api_key=config.openai_api_key)

            class _OpenAIOracleAdapter:
                """Thin adapter: maps .generate() -> OpenAI chat completion."""

                def __init__(self, client, model: str, temperature: float, max_tokens: int):
                    self._c = client
                    self._model = model
                    self._temp = temperature
                    self._max_tokens = max_tokens

                def generate(self, prompt: str) -> str:
                    resp = self._c.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self._temp,
                        max_tokens=self._max_tokens,
                    )
                    return resp.choices[0].message.content or ""

                def chat_completions_create(self, **kwargs) -> str:
                    resp = self._c.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""

            class _OpenAIJudgeAdapter:
                """Thin adapter: maps .chat_completions_create() -> OpenAI."""

                def __init__(self, client):
                    self._c = client

                def chat_completions_create(self, **kwargs) -> str:
                    resp = self._c.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""

            oracle_client = _OpenAIOracleAdapter(
                openai_client,
                config.oracle_model,
                config.oracle_temperature,
                config.oracle_max_tokens,
            )
            judge_client = _OpenAIJudgeAdapter(openai_client)

        except ImportError:
            logger.warning("openai package not installed; falling back to mock clients")
            oracle_client = MockOracleClient()
            judge_client = MockJudgeClient()
    else:
        logger.info("No OPENAI_API_KEY — using mock oracle and judge clients")
        oracle_client = MockOracleClient()
        judge_client = MockJudgeClient()

    judge = LLMJudge(
        client=judge_client,
        model=config.judge_model,
        pass_threshold=config.judge_pass_threshold,
    )
    amplifier = SyntheticAmplifier(
        oracle_client=oracle_client,
        judge=judge,
        config=config,
    )

    return SynthesisPipeline(
        embedding_engine=embedder,
        clusterer=clusterer,
        amplifier=amplifier,
        config=config,
    )


# Build once at startup
_CONFIG = SynthesizerConfig.from_env()
_PIPELINE = _build_pipeline(_CONFIG)


# --------------------------------------------------------------------------- #
# REST endpoints                                                               #
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe — always 200 if the process is alive."""
    return JSONResponse({"status": "ok"})


@app.get("/ready")
async def ready() -> JSONResponse:
    """
    Readiness probe.

    Returns 200 when the embedding model is loaded and Kafka consumer
    thread is running (or Kafka is not configured).
    """
    return JSONResponse({"status": "ready", "embedding_model": _CONFIG.embedding_model})


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest) -> JSONResponse:
    """
    Trigger a synthesis job via the REST API.

    Request body (JSON):
        {
          "failure_records": [
            {"request_id": "...", "prompt": "...", "response": "...",
             "mean_logprob": -1.5, "timestamp": "2026-03-28T00:00:00Z"},
            ...
          ],
          "trigger_event_id": "optional-drift-event-id"
        }

    Returns:
        SynthesisJobResult as JSON.

    Status codes:
        200 — success
        422 — validation error in request body
        500 — synthesis pipeline exception
    """
    job_id = str(uuid.uuid4())
    trigger_event_id = request.trigger_event_id or f"api-{job_id}"

    try:
        records = [
            FailureRecord(
                request_id=rec.get("request_id", str(uuid.uuid4())),
                prompt=rec["prompt"],
                response=rec["response"],
                mean_logprob=float(rec.get("mean_logprob", 0.0)),
                timestamp=rec.get("timestamp", ""),
            )
            for rec in request.failure_records
        ]
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid failure_record schema: {exc}",
        )

    SYNTHESIS_JOBS_TOTAL.labels(trigger_type="api").inc()

    try:
        result = _PIPELINE.run(
            failure_records=records,
            job_id=job_id,
            trigger_event_id=trigger_event_id,
        )
    except Exception as exc:
        SYNTHESIS_JOBS_FAILED.inc()
        logger.error(
            "synthesis_job_failed",
            job_id=job_id,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc))

    _record_metrics(result)
    logger.info(
        "synthesis_job_complete",
        job_id=result.job_id,
        trigger_event_id=result.trigger_event_id,
        clusters_processed=result.clusters_processed,
        total_pairs=result.total_pairs_synthesized,
        passing_pairs=result.total_pairs_passing_gate,
        cost_usd=result.total_cost_usd,
        duration_s=result.duration_seconds,
    )

    return JSONResponse(result.__dict__)


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


# --------------------------------------------------------------------------- #
# Kafka consumer                                                               #
# --------------------------------------------------------------------------- #

def _kafka_consumer_loop(config: SynthesizerConfig, pipeline: SynthesisPipeline) -> None:
    """
    Background Kafka consumer thread.

    Subscribes to drift.events and triggers synthesis for qualifying events
    (qualification_status == "QUALIFIED").

    This function runs indefinitely until the process exits.  Errors during
    message processing are logged but do not terminate the consumer loop —
    transient failures should not halt the pipeline.
    """
    try:
        from confluent_kafka import Consumer, KafkaError  # type: ignore
        from shared.schemas.events import TOPIC_DRIFT_EVENTS
    except ImportError:
        logger.warning("confluent_kafka not available; Kafka consumer disabled")
        return

    consumer = Consumer(
        {
            "bootstrap.servers": config.kafka_bootstrap_servers,
            "group.id": config.kafka_group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([TOPIC_DRIFT_EVENTS])
    logger.info("kafka_consumer_started", topic=TOPIC_DRIFT_EVENTS)

    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            logger.error("kafka_error", error=str(msg.error()))
            continue

        try:
            event_data = json.loads(msg.value().decode("utf-8"))

            # Only process qualified drift evidence
            if event_data.get("qualification_status") != "QUALIFIED":
                continue

            event_id = event_data.get("event_id", str(uuid.uuid4()))
            evidence_ids: List[str] = event_data.get("evidence_request_ids", [])

            # Build synthetic FailureRecords from evidence IDs
            # In production this would fetch from a feature store / DB.
            # Here we construct placeholder records so the pipeline can run.
            records = [
                FailureRecord(
                    request_id=req_id,
                    prompt=f"[evidence] request {req_id}",
                    response="[placeholder response]",
                    mean_logprob=float(event_data.get("composite_anomaly_score", 0.0)),
                    timestamp=event_data.get("detected_at", ""),
                )
                for req_id in evidence_ids
            ]

            if not records:
                continue

            job_id = str(uuid.uuid4())
            SYNTHESIS_JOBS_TOTAL.labels(trigger_type="kafka").inc()

            try:
                result = pipeline.run(
                    failure_records=records,
                    job_id=job_id,
                    trigger_event_id=event_id,
                )
                _record_metrics(result)
                logger.info(
                    "kafka_synthesis_complete",
                    job_id=result.job_id,
                    trigger_event_id=result.trigger_event_id,
                    clusters=result.clusters_processed,
                    pairs=result.total_pairs_passing_gate,
                    cost_usd=result.total_cost_usd,
                )
            except Exception as exc:
                SYNTHESIS_JOBS_FAILED.inc()
                logger.error(
                    "kafka_synthesis_failed",
                    job_id=job_id,
                    event_id=event_id,
                    error=str(exc),
                )

        except Exception as exc:
            logger.error("kafka_message_parse_error", error=str(exc))


def _record_metrics(result: SynthesisJobResult) -> None:
    """Update Prometheus metrics from a completed SynthesisJobResult."""
    PAIRS_SYNTHESIZED.inc(result.total_pairs_synthesized)
    PAIRS_PASSING_GATE.inc(result.total_pairs_passing_gate)
    SYNTHESIS_DURATION_SECONDS.observe(result.duration_seconds)
    SYNTHESIS_COST_USD.inc(result.total_cost_usd)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def start_kafka_consumer_background() -> None:
    """Start the Kafka consumer in a daemon thread (called at startup)."""
    t = threading.Thread(
        target=_kafka_consumer_loop,
        args=(_CONFIG, _PIPELINE),
        daemon=True,
        name="kafka-consumer",
    )
    t.start()
    logger.info("kafka_consumer_thread_started")


@app.on_event("startup")
async def on_startup() -> None:
    """FastAPI startup hook — launch Kafka consumer."""
    start_kafka_consumer_background()


if __name__ == "__main__":
    uvicorn.run(
        "services.synthesizer.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8003")),
        log_level="info",
    )
