"""
FastAPI REST API for the Observability & Lineage Platform.

Endpoints serve dashboard data to the UI, CI/CD gate checks, and
operator tooling.  All endpoints are async and non-blocking.

Architecture notes:
  - A single LineageStore instance is shared across requests via FastAPI's
    dependency injection / lifespan machinery.  The store is initialised
    during application startup so the schema is always ready before requests
    arrive.
  - The /metrics endpoint returns raw Prometheus text format; this is
    scraped by the Prometheus server, NOT by the UI.
  - Dashboard health aggregates anomaly score (from Prometheus in-process
    gauge) + recent alert history from AlertDispatcher.
  - All list endpoints accept a `limit` parameter capped at 200 to prevent
    unbounded queries.

Invariants:
  - /healthz always returns 200 while the process is alive (no external deps
    checked — that is the job of /api/v1/dashboard/health).
  - adapter_id values are treated as opaque strings; no URL-unsafe characters
    should be used by upstream producers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional, List, Any, Dict

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import PlainTextResponse

from .metrics import PlatformMetrics
from .lineage import LineageStore, LineageNode
from .alerting import AlertBudget, AlertDispatcher, Alert


# ---------------------------------------------------------------------------
# Application lifespan — initialise shared state once at startup
# ---------------------------------------------------------------------------

# Module-level singletons shared by all request handlers.
_lineage_store: LineageStore = LineageStore()
_alert_budget: AlertBudget = AlertBudget()
_alert_dispatcher: AlertDispatcher = AlertDispatcher(_alert_budget)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialise the SQLite lineage schema before accepting requests."""
    await _lineage_store.initialize()
    yield
    # No explicit teardown needed — aiosqlite connections are closed per-op.


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Observability API",
    version="1.0.0",
    description=(
        "Model Observability & Lineage Platform — Module 7. "
        "Serves adapter lineage, Prometheus metrics, training cycle history, "
        "and production health data to dashboards and CI/CD gates."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_to_dict(node: LineageNode) -> Dict[str, Any]:
    """
    Serialise a LineageNode to a JSON-safe dict.

    Args:
        node: LineageNode to serialise.
    Returns:
        dict suitable for JSON response serialisation.
    Complexity: O(1).
    Side effects: None.
    """
    return {
        "adapter_id": node.adapter_id,
        "version": node.version,
        "base_model_id": node.base_model_id,
        "training_cycle_id": node.training_cycle_id,
        "created_at": node.created_at,
        "status": node.status,
        "parent_adapter_id": node.parent_adapter_id,
        "target_failure_cluster_ids": node.target_failure_cluster_ids,
        "evaluation_scores": node.evaluation_scores,
        "deployment_record": node.deployment_record,
    }


# ---------------------------------------------------------------------------
# Infrastructure endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["infra"])
async def healthz():
    """
    Liveness probe.

    Returns 200 {"status": "ok"} while the process is alive.
    Does not check external dependencies (that is /api/v1/dashboard/health).

    Returns:
        dict: {"status": "ok"}
    """
    return {"status": "ok"}


@app.get("/metrics", tags=["infra"], response_class=PlainTextResponse)
async def metrics():
    """
    Prometheus text-format metrics scrape endpoint.

    Returns all platform-wide metrics registered in the shared REGISTRY in
    Prometheus text exposition format (UTF-8).

    Returns:
        PlainTextResponse: Raw Prometheus text payload.
    """
    return PlainTextResponse(PlatformMetrics.get_metrics_output().decode())


# ---------------------------------------------------------------------------
# Lineage endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/lineage/{adapter_id}", tags=["lineage"])
async def get_lineage(adapter_id: str):
    """
    Return the full ancestry chain for a given adapter.

    Walks the parent_adapter_id chain from the given adapter back to the
    root, returning an ordered list [root, …, adapter_id].

    Args:
        adapter_id: Target adapter identifier.
    Returns:
        dict: {
            "adapter_id": str,
            "depth": int,
            "ancestors": [LineageNode dict, …]   # ordered root → adapter
        }
    Raises:
        404 if adapter_id is not found in the lineage store.
    """
    # Verify the node exists before walking ancestors.
    node = await _lineage_store.get_node(adapter_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Adapter '{adapter_id}' not found")

    ancestors = await _lineage_store.get_ancestors(adapter_id)
    return {
        "adapter_id": adapter_id,
        "depth": len(ancestors),
        "ancestors": [_node_to_dict(n) for n in ancestors],
    }


@app.get("/api/v1/adapters", tags=["lineage"])
async def list_adapters(
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """
    List adapters with an optional status filter.

    Args:
        status: If provided, only adapters with this status are returned
                (e.g. "PRODUCTION", "CANDIDATE", "ROLLED_BACK").
        limit:  Maximum number of adapters to return (1–200, default 50).
    Returns:
        dict: {
            "count": int,
            "adapters": [LineageNode dict, …]
        }
    """
    all_nodes = await _lineage_store.get_all_nodes(status_filter=status)
    # Slice to limit after filtering — full table scan is acceptable here
    # because the lineage DAG is bounded by the number of training cycles.
    nodes = all_nodes[:limit]
    return {
        "count": len(nodes),
        "adapters": [_node_to_dict(n) for n in nodes],
    }


@app.get("/api/v1/adapters/{adapter_id}/compare/{other_adapter_id}", tags=["lineage"])
async def compare_adapters(adapter_id: str, other_adapter_id: str):
    """
    Side-by-side comparison of two adapter versions.

    Fetches both adapters and computes per-benchmark score deltas
    (other - base).

    Args:
        adapter_id:       Base adapter identifier.
        other_adapter_id: Comparison adapter identifier.
    Returns:
        dict: {
            "base": LineageNode dict,
            "other": LineageNode dict,
            "score_deltas": {benchmark: delta, …}
        }
    Raises:
        404 if either adapter is not found.
    """
    base = await _lineage_store.get_node(adapter_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"Adapter '{adapter_id}' not found")

    other = await _lineage_store.get_node(other_adapter_id)
    if other is None:
        raise HTTPException(
            status_code=404, detail=f"Adapter '{other_adapter_id}' not found"
        )

    # Compute benchmark score deltas for benchmarks present in both adapters.
    base_scores = base.evaluation_scores or {}
    other_scores = other.evaluation_scores or {}
    all_benchmarks = set(base_scores) | set(other_scores)
    score_deltas = {
        bm: round(
            other_scores.get(bm, 0.0) - base_scores.get(bm, 0.0), 6
        )
        for bm in sorted(all_benchmarks)
    }

    return {
        "base": _node_to_dict(base),
        "other": _node_to_dict(other),
        "score_deltas": score_deltas,
    }


# ---------------------------------------------------------------------------
# Training cycle endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/cycles", tags=["training"])
async def list_training_cycles(limit: int = Query(20, ge=1, le=100)):
    """
    List recent training cycles with cost and status, derived from the
    lineage store.

    Each adapter node embeds the training_cycle_id and deployment_record
    (which carries cost and timing).  This endpoint pivots that data into
    a per-cycle summary.

    Args:
        limit: Maximum number of cycles to return (1–100, default 20).
    Returns:
        dict: {
            "count": int,
            "cycles": [
                {
                    "cycle_id": str,
                    "adapter_id": str,
                    "status": str,
                    "created_at": str,
                    "cost_usd": float | None,
                    "evaluation_scores": dict
                },
                …
            ]
        }
    """
    all_nodes = await _lineage_store.get_all_nodes()
    # Most-recent first: sort by created_at descending, then take limit.
    sorted_nodes = sorted(all_nodes, key=lambda n: n.created_at, reverse=True)
    sliced = sorted_nodes[:limit]

    cycles = []
    for node in sliced:
        cost = None
        if node.deployment_record and "cost_usd" in node.deployment_record:
            cost = node.deployment_record["cost_usd"]
        cycles.append(
            {
                "cycle_id": node.training_cycle_id,
                "adapter_id": node.adapter_id,
                "status": node.status,
                "created_at": node.created_at,
                "cost_usd": cost,
                "evaluation_scores": node.evaluation_scores,
            }
        )

    return {"count": len(cycles), "cycles": cycles}


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/dashboard/health", tags=["dashboard"])
async def production_health():
    """
    Current production model health snapshot.

    Aggregates:
      - Current anomaly score (read from the in-process Prometheus gauge).
      - Count of production adapters.
      - Recent alert summary from the in-process AlertDispatcher.

    Returns:
        dict: {
            "production_adapters": int,
            "recent_alerts": {level: count, …},
            "total_alerts_dispatched": int
        }

    Note: anomaly scores are labelled per model_version in the Prometheus
    gauge; this endpoint returns the total count of production adapters and
    alert statistics rather than duplicating the metric scrape.
    """
    production_nodes = await _lineage_store.get_all_nodes(status_filter="PRODUCTION")

    # Summarise dispatched alerts by level.
    alert_counts: Dict[str, int] = {}
    for alert in _alert_dispatcher.sent_alerts:
        alert_counts[alert.level] = alert_counts.get(alert.level, 0) + 1

    return {
        "production_adapters": len(production_nodes),
        "recent_alerts": alert_counts,
        "total_alerts_dispatched": len(_alert_dispatcher.sent_alerts),
    }


@app.get("/api/v1/dashboard/loop-status", tags=["dashboard"])
async def loop_status():
    """
    Current autonomous loop state and active job counts.

    Reads loop_state and adapter counts from the lineage store.
    The loop_state_info Prometheus gauge is set by the orchestrator;
    this endpoint provides a human-readable summary suitable for a
    status badge in the dashboard.

    Returns:
        dict: {
            "adapter_counts": {status: count, …},
            "total_adapters": int
        }
    """
    all_nodes = await _lineage_store.get_all_nodes()

    status_counts: Dict[str, int] = {}
    for node in all_nodes:
        status_counts[node.status] = status_counts.get(node.status, 0) + 1

    return {
        "adapter_counts": status_counts,
        "total_adapters": len(all_nodes),
    }
