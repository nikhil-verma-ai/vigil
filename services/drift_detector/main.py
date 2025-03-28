"""
FastAPI application + Kafka consumer entrypoint for the Drift Detector service.

Endpoints:
  GET  /healthz              — liveness probe
  GET  /readyz               — readiness probe (Kafka connectivity)
  GET  /metrics              — Prometheus metrics (text format)
  POST /config/thresholds    — runtime threshold update (no restart required)

Background tasks:
  _kafka_consumer_loop — polls logprob.signals, calls detector.process_event(),
                         optionally emits DriftEvents to drift.events.

Prometheus metrics:
  drift_events_emitted_total{level, drift_type}
  events_processed_total
  baseline_update_latency_seconds (histogram)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.schemas.events import AlertLevel, LogprobSignalEvent, TOPIC_LOGPROB_SIGNALS
from .detector import DriftDetector
from .alerting import DEFAULT_THRESHOLDS

# ---------------------------------------------------------------------------
# Structured logging setup
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
DRIFT_EVENTS_EMITTED = Counter(
    "drift_events_emitted_total",
    "Total drift events emitted",
    labelnames=["level", "drift_type"],
)

EVENTS_PROCESSED = Counter(
    "events_processed_total",
    "Total logprob signal events processed",
)

BASELINE_UPDATE_LATENCY = Histogram(
    "baseline_update_latency_seconds",
    "Latency of baseline.update() calls",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)

# ---------------------------------------------------------------------------
# Application state (module-level singletons, injected into app.state)
# ---------------------------------------------------------------------------
_detector: Optional[DriftDetector] = None
_kafka_consumer = None          # confluent_kafka.Consumer instance
_kafka_producer = None          # confluent_kafka.Producer instance
_consumer_ready: bool = False
_consumer_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise detector, Redis (optional), Kafka.  Shutdown: flush."""
    global _detector, _kafka_consumer, _kafka_producer, _consumer_task, _consumer_ready

    log.info("drift_detector_starting")

    # Initialise detector (Redis optional — graceful degradation).
    redis_client = _try_init_redis()
    _detector = DriftDetector(redis_client=redis_client)

    # Kafka setup (graceful — service can start without Kafka for local dev).
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    group_id = os.environ.get("KAFKA_CONSUMER_GROUP", "drift-detector-v1")

    try:
        from confluent_kafka import Consumer, Producer

        _kafka_consumer = Consumer({
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "max.poll.interval.ms": 300000,
        })
        _kafka_consumer.subscribe([TOPIC_LOGPROB_SIGNALS])

        _kafka_producer = Producer({
            "bootstrap.servers": bootstrap,
            "linger.ms": 5,
            "compression.type": "lz4",
            "acks": "all",
        })

        _consumer_task = asyncio.create_task(_kafka_consumer_loop())
        _consumer_ready = True
        log.info("kafka_consumer_started", topic=TOPIC_LOGPROB_SIGNALS)
    except Exception as exc:
        log.warning("kafka_init_failed", error=str(exc))
        _consumer_ready = False

    yield

    # Shutdown.
    log.info("drift_detector_shutting_down")
    if _consumer_task is not None:
        _consumer_task.cancel()
        try:
            await _consumer_task
        except asyncio.CancelledError:
            pass
    if _kafka_consumer is not None:
        _kafka_consumer.close()
    if _kafka_producer is not None:
        _kafka_producer.flush(timeout=5)
    log.info("drift_detector_stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Drift Detector",
    description="Predictive Drift Detection Engine — Module 2",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> dict:
    """Liveness probe.  Always 200 if the process is alive."""
    return {
        "status": "ok",
        "baselines_loaded": _detector.baselines_loaded if _detector else 0,
        "events_processed": _detector.events_processed if _detector else 0,
    }


@app.get("/readyz")
async def readyz() -> dict:
    """
    Readiness probe.
    Returns 200 if Kafka consumer is connected, 503 otherwise.
    """
    if not _consumer_ready:
        raise HTTPException(status_code=503, detail="Kafka consumer not ready")
    return {"status": "ready"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus text-format metrics endpoint."""
    return generate_latest().decode("utf-8")


class ThresholdUpdateRequest(BaseModel):
    """
    Runtime threshold update payload.

    Example:
      {"level": "CRITICAL", "value": 3.0}
    """
    level: str = Field(..., description="AlertLevel name: INFO|WARNING|CRITICAL|EMERGENCY")
    value: float = Field(..., gt=0.0, description="New composite z-score threshold")


@app.post("/config/thresholds", status_code=200)
async def update_thresholds(req: ThresholdUpdateRequest) -> dict:
    """
    Update a drift alert threshold at runtime without restarting the service.

    The update is applied immediately to all existing SlidingWindowAggregators
    and to any newly-created aggregators.
    """
    try:
        level = AlertLevel(req.level)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown alert level '{req.level}'. Valid: INFO, WARNING, CRITICAL, EMERGENCY",
        )

    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialised")

    _detector.update_threshold(level, req.value)
    log.info("threshold_updated", level=req.level, value=req.value)
    return {"status": "updated", "level": req.level, "new_value": req.value}


# ---------------------------------------------------------------------------
# Kafka consumer loop
# ---------------------------------------------------------------------------

async def _kafka_consumer_loop() -> None:
    """
    Background task: poll logprob.signals, process events, emit DriftEvents.

    Runs as asyncio task — uses run_in_executor to avoid blocking the event
    loop during Kafka poll (which has a configurable timeout).

    Backpressure: if the processing queue falls behind, messages are still
    consumed (preventing Kafka consumer group rebalance), but processing is
    performed sequentially — there is no message drop.
    """
    loop = asyncio.get_event_loop()

    log.info("kafka_consumer_loop_started")

    while True:
        try:
            # Poll in executor to keep event loop unblocked.
            msg = await loop.run_in_executor(
                None, lambda: _kafka_consumer.poll(timeout=0.1)
            )

            if msg is None:
                continue

            if msg.error():
                log.warning("kafka_message_error", error=str(msg.error()))
                continue

            raw = msg.value()
            if raw is None:
                continue

            # Deserialise.
            try:
                event = LogprobSignalEvent.from_json(raw.decode("utf-8"))
            except Exception as exc:
                log.warning("event_deserialisation_failed", error=str(exc))
                continue

            # Process — measure latency for the baseline update portion.
            t0 = time.monotonic()
            drift_event = _detector.process_event(event)
            elapsed = time.monotonic() - t0

            BASELINE_UPDATE_LATENCY.observe(elapsed)
            EVENTS_PROCESSED.inc()

            # Emit if we got a qualified DriftEvent.
            if drift_event is not None and _kafka_producer is not None:
                _detector.emit_to_kafka(drift_event, _kafka_producer)
                DRIFT_EVENTS_EMITTED.labels(
                    level=drift_event.alert_level.value,
                    drift_type=drift_event.drift_type.value,
                ).inc()

        except asyncio.CancelledError:
            log.info("kafka_consumer_loop_cancelled")
            break
        except Exception as exc:
            log.error("kafka_consumer_loop_error", error=str(exc))
            # Brief backoff to avoid tight error loops.
            await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_init_redis():
    """Attempt Redis connection; return client or None on failure."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis

        client = redis.from_url(redis_url, socket_connect_timeout=2)
        client.ping()
        log.info("redis_connected", url=redis_url)
        return client
    except Exception as exc:
        log.warning("redis_unavailable", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "services.drift_detector.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8002)),
        log_level="info",
        workers=1,  # Single worker — shared in-process state (detector, baselines).
    )
