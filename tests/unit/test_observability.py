"""
Unit tests for Module 7: Model Observability & Lineage Platform.

Test coverage:
  - PlatformMetrics: counter increments, gauge sets, histogram observations
  - LineageStore: node add/retrieve, ancestry chain traversal
  - AlertBudget / AlertDispatcher: rate limiting, INFO suppression, EMERGENCY passthrough
  - FastAPI endpoints: /healthz, /metrics

All tests are self-contained.  Lineage tests use isolated temporary SQLite
databases so they do not interfere with each other or with any running service.
"""

import asyncio
import tempfile
import time
import uuid

import pytest

# ---------------------------------------------------------------------------
# Module imports — absolute paths from project root.
# Add the project root to sys.path so imports resolve correctly when running
# pytest from any directory.
# ---------------------------------------------------------------------------
import sys
import os

# Insert the project root (two levels up from tests/unit/) so that
# `services.observability` is importable.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from services.observability.metrics import PlatformMetrics
from services.observability.lineage import LineageStore, LineageNode, LineageEdge
from services.observability.alerting import Alert, AlertBudget, AlertDispatcher
from services.observability.api import app


# ===========================================================================
# Prometheus metrics tests
# ===========================================================================


def test_prometheus_metrics_increment():
    """Counter increments must be reflected in /metrics output."""
    PlatformMetrics.drift_events_total.labels(
        level="CRITICAL", drift_type="CONCEPT"
    ).inc(3)
    output = PlatformMetrics.get_metrics_output().decode()
    assert "drift_events_total" in output
    assert 'level="CRITICAL"' in output


def test_prometheus_gauge_set():
    """Gauge set value must appear in the rendered metrics output."""
    PlatformMetrics.anomaly_score_current.labels(model_version="adapter-v3").set(4.7)
    output = PlatformMetrics.get_metrics_output().decode()
    assert "anomaly_score_current" in output
    assert "4.7" in output


def test_prometheus_histogram_observe():
    """Histogram observation must increment the _count series."""
    PlatformMetrics.promotion_duration_ms.observe(450)
    output = PlatformMetrics.get_metrics_output().decode()
    assert "promotion_duration_ms_count" in output


# ===========================================================================
# LineageStore tests
# ===========================================================================


def test_lineage_store_add_and_retrieve():
    """Nodes added to the store must be retrievable with all fields intact."""

    async def run():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = LineageStore(db_path)
        await store.initialize()

        node = LineageNode(
            adapter_id="adapter-v2",
            version="1.2.0",
            base_model_id="llama-7b",
            training_cycle_id="cycle-001",
            created_at="2026-03-28T00:00:00Z",
            status="PRODUCTION",
            parent_adapter_id="adapter-v1",
            target_failure_cluster_ids=[0, 1],
            evaluation_scores={"arithmetic": 0.92, "instruction": 0.88},
            deployment_record={"promoted_at": "2026-03-28T04:00:00Z"},
        )
        await store.add_node(node)
        retrieved = await store.get_node("adapter-v2")

        assert retrieved is not None
        assert retrieved.adapter_id == "adapter-v2"
        assert retrieved.version == "1.2.0"
        assert retrieved.target_failure_cluster_ids == [0, 1]
        assert retrieved.evaluation_scores["arithmetic"] == 0.92

    asyncio.run(run())


def test_lineage_ancestry_chain():
    """get_ancestors must return ordered chain from root to node."""

    async def run():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = LineageStore(db_path)
        await store.initialize()

        # Build chain: v1 (root) → v2 → v3
        for i, parent in enumerate([None, "adapter-v1", "adapter-v2"], start=1):
            await store.add_node(
                LineageNode(
                    adapter_id=f"adapter-v{i}",
                    version=f"1.{i}.0",
                    base_model_id="llama",
                    training_cycle_id=f"cycle-00{i}",
                    created_at="2026-03-28T00:00:00Z",
                    status="PRODUCTION",
                    parent_adapter_id=parent,
                    target_failure_cluster_ids=[],
                    evaluation_scores={},
                    deployment_record=None,
                )
            )

        ancestors = await store.get_ancestors("adapter-v3")
        assert len(ancestors) == 3
        assert ancestors[0].adapter_id == "adapter-v1"  # root first
        assert ancestors[-1].adapter_id == "adapter-v3"  # requested node last

    asyncio.run(run())


def test_lineage_get_node_not_found():
    """get_node must return None for unknown adapter_id."""

    async def run():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = LineageStore(db_path)
        await store.initialize()
        result = await store.get_node("does-not-exist")
        assert result is None

    asyncio.run(run())


def test_lineage_get_all_nodes_status_filter():
    """get_all_nodes with a status_filter must return only matching nodes."""

    async def run():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = LineageStore(db_path)
        await store.initialize()

        statuses = ["PRODUCTION", "CANDIDATE", "PRODUCTION", "SUPERSEDED"]
        for i, status in enumerate(statuses):
            await store.add_node(
                LineageNode(
                    adapter_id=f"adapter-{i}",
                    version=f"1.{i}.0",
                    base_model_id="llama",
                    training_cycle_id=f"cycle-{i:03d}",
                    created_at=f"2026-03-28T0{i}:00:00Z",
                    status=status,
                    parent_adapter_id=None,
                    target_failure_cluster_ids=[],
                    evaluation_scores={},
                    deployment_record=None,
                )
            )

        production = await store.get_all_nodes(status_filter="PRODUCTION")
        assert len(production) == 2
        assert all(n.status == "PRODUCTION" for n in production)

    asyncio.run(run())


def test_lineage_update_status():
    """update_status must mutate the status column of an existing node."""

    async def run():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = LineageStore(db_path)
        await store.initialize()

        await store.add_node(
            LineageNode(
                adapter_id="adapter-x",
                version="1.0.0",
                base_model_id="llama",
                training_cycle_id="cycle-x",
                created_at="2026-03-28T00:00:00Z",
                status="CANDIDATE",
                parent_adapter_id=None,
                target_failure_cluster_ids=[],
                evaluation_scores={},
                deployment_record=None,
            )
        )

        await store.update_status("adapter-x", "PRODUCTION")
        node = await store.get_node("adapter-x")
        assert node is not None
        assert node.status == "PRODUCTION"

    asyncio.run(run())


def test_lineage_add_edge():
    """add_edge must persist an edge retrievable by the database."""

    async def run():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = LineageStore(db_path)
        await store.initialize()

        edge = LineageEdge(
            parent_id="adapter-v1",
            child_id="adapter-v2",
            improvement_target="arithmetic cluster 0",
            cost_usd=12.50,
            created_at="2026-03-28T01:00:00Z",
        )
        # add_edge must not raise
        await store.add_edge(edge)

        # Verify via raw query that the edge was persisted.
        import aiosqlite

        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT * FROM edges WHERE parent_id = ? AND child_id = ?",
                ("adapter-v1", "adapter-v2"),
            ) as cursor:
                row = await cursor.fetchone()
        assert row is not None
        assert row[3] == 12.50  # cost_usd column

    asyncio.run(run())


def test_lineage_ancestors_root_node():
    """A root node (no parent) must return a chain of length 1."""

    async def run():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = LineageStore(db_path)
        await store.initialize()

        await store.add_node(
            LineageNode(
                adapter_id="adapter-root",
                version="1.0.0",
                base_model_id="llama",
                training_cycle_id="cycle-000",
                created_at="2026-03-28T00:00:00Z",
                status="SUPERSEDED",
                parent_adapter_id=None,
                target_failure_cluster_ids=[],
                evaluation_scores={},
                deployment_record=None,
            )
        )

        ancestors = await store.get_ancestors("adapter-root")
        assert len(ancestors) == 1
        assert ancestors[0].adapter_id == "adapter-root"

    asyncio.run(run())


# ===========================================================================
# AlertBudget / AlertDispatcher tests
# ===========================================================================


def _make_alert(level: str = "WARNING") -> Alert:
    return Alert(
        alert_id=str(uuid.uuid4()),
        level=level,
        title=f"Test alert [{level}]",
        body="Test body",
        evidence={"metric": 42},
        timestamp=time.time(),
    )


def test_alert_budget_rate_limiting():
    """WARNING alerts must be capped at 10 per hour."""
    budget = AlertBudget()
    dispatcher = AlertDispatcher(budget, channels=[])

    sent_count = sum(
        1 for _ in range(12) if dispatcher.dispatch(_make_alert("WARNING"))
    )

    assert sent_count == 10, f"Expected 10 sent (rate limit), got {sent_count}"


def test_info_alerts_never_sent():
    """INFO level alerts must always be suppressed."""
    budget = AlertBudget()
    dispatcher = AlertDispatcher(budget, channels=[])
    result = dispatcher.dispatch(_make_alert("INFO"))
    assert result is False
    assert len(dispatcher.sent_alerts) == 0


def test_emergency_alerts_always_sent():
    """EMERGENCY alerts must bypass rate limiting (limit=100)."""
    budget = AlertBudget()
    dispatcher = AlertDispatcher(budget, channels=[])
    for _ in range(20):
        dispatcher.dispatch(_make_alert("EMERGENCY"))
    assert len(dispatcher.sent_alerts) == 20


def test_critical_alerts_rate_limited_to_5():
    """CRITICAL alerts must be capped at 5 per hour."""
    budget = AlertBudget()
    dispatcher = AlertDispatcher(budget, channels=[])

    sent_count = sum(
        1 for _ in range(10) if dispatcher.dispatch(_make_alert("CRITICAL"))
    )

    assert sent_count == 5, f"Expected 5 sent (rate limit), got {sent_count}"


def test_channel_called_on_dispatch():
    """Registered channel callable must be invoked for each dispatched alert."""
    received: list = []
    budget = AlertBudget()
    dispatcher = AlertDispatcher(budget, channels=[received.append])

    alert = _make_alert("WARNING")
    result = dispatcher.dispatch(alert)

    assert result is True
    assert len(received) == 1
    assert received[0].alert_id == alert.alert_id


def test_channel_exception_does_not_block_dispatch():
    """A faulty channel must not prevent the alert from being recorded."""

    def bad_channel(a: Alert):
        raise RuntimeError("channel failure")

    budget = AlertBudget()
    dispatcher = AlertDispatcher(budget, channels=[bad_channel])
    result = dispatcher.dispatch(_make_alert("WARNING"))

    assert result is True
    assert len(dispatcher.sent_alerts) == 1


def test_budget_window_eviction():
    """
    Timestamps older than 1 hour must be evicted so that fresh alerts are
    allowed through again.  We monkey-patch time.time to simulate the window.
    """
    import unittest.mock as mock

    budget = AlertBudget()
    dispatcher = AlertDispatcher(budget, channels=[])

    # Exhaust the WARNING budget with timestamps in the past (> 1 hr ago).
    old_time = time.time() - 3700.0
    with mock.patch("time.time", return_value=old_time):
        for _ in range(10):
            dispatcher.dispatch(_make_alert("WARNING"))

    assert len(dispatcher.sent_alerts) == 10

    # Now at current time the window should be clear — 10 more should go through.
    for _ in range(10):
        dispatcher.dispatch(_make_alert("WARNING"))

    assert len(dispatcher.sent_alerts) == 20


# ===========================================================================
# FastAPI endpoint tests
# ===========================================================================


def test_api_healthz():
    """GET /healthz must return 200 and {"status": "ok"}."""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_metrics_endpoint():
    """GET /metrics must return 200 and contain Prometheus metric names."""
    from fastapi.testclient import TestClient

    PlatformMetrics.training_cycles_total.labels(status="COMPLETED").inc(1)
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "training_cycles_total" in response.text


def test_api_list_adapters_empty():
    """GET /api/v1/adapters on a fresh store must return an empty list."""
    from fastapi.testclient import TestClient

    # Override the module-level lineage store with a fresh isolated one.
    import services.observability.api as api_module

    original_store = api_module._lineage_store
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    fresh_store = LineageStore(tmp_db)
    asyncio.run(fresh_store.initialize())
    api_module._lineage_store = fresh_store

    try:
        client = TestClient(app)
        response = client.get("/api/v1/adapters")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["adapters"] == []
    finally:
        api_module._lineage_store = original_store


def test_api_lineage_not_found():
    """GET /api/v1/lineage/{unknown} must return 404."""
    from fastapi.testclient import TestClient
    import services.observability.api as api_module

    original_store = api_module._lineage_store
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name
    fresh_store = LineageStore(tmp_db)
    asyncio.run(fresh_store.initialize())
    api_module._lineage_store = fresh_store

    try:
        client = TestClient(app)
        response = client.get("/api/v1/lineage/does-not-exist")
        assert response.status_code == 404
    finally:
        api_module._lineage_store = original_store


def test_api_dashboard_health():
    """GET /api/v1/dashboard/health must return expected shape."""
    from fastapi.testclient import TestClient
    import services.observability.api as api_module

    original_store = api_module._lineage_store
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name
    fresh_store = LineageStore(tmp_db)
    asyncio.run(fresh_store.initialize())
    api_module._lineage_store = fresh_store

    try:
        client = TestClient(app)
        response = client.get("/api/v1/dashboard/health")
        assert response.status_code == 200
        data = response.json()
        assert "production_adapters" in data
        assert "recent_alerts" in data
        assert "total_alerts_dispatched" in data
    finally:
        api_module._lineage_store = original_store


def test_api_loop_status():
    """GET /api/v1/dashboard/loop-status must return expected shape."""
    from fastapi.testclient import TestClient
    import services.observability.api as api_module

    original_store = api_module._lineage_store
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name
    fresh_store = LineageStore(tmp_db)
    asyncio.run(fresh_store.initialize())
    api_module._lineage_store = fresh_store

    try:
        client = TestClient(app)
        response = client.get("/api/v1/dashboard/loop-status")
        assert response.status_code == 200
        data = response.json()
        assert "adapter_counts" in data
        assert "total_adapters" in data
    finally:
        api_module._lineage_store = original_store
