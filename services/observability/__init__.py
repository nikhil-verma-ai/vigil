"""
Observability & Lineage Platform — Module 7

Exports:
  - PlatformMetrics: central Prometheus registry
  - LineageStore / LineageNode / LineageEdge: adapter lineage DAG
  - AlertBudget / AlertDispatcher / Alert: alert rate-limiting and dispatch
  - app: FastAPI application (REST endpoints for dashboards)
"""

from .metrics import PlatformMetrics, REGISTRY
from .lineage import LineageStore, LineageNode, LineageEdge
from .alerting import Alert, AlertBudget, AlertDispatcher
from .api import app

__all__ = [
    "PlatformMetrics",
    "REGISTRY",
    "LineageStore",
    "LineageNode",
    "LineageEdge",
    "Alert",
    "AlertBudget",
    "AlertDispatcher",
    "app",
]
