"""
Observability Service entry point.

Launches the FastAPI application under uvicorn.  All configuration is driven
by environment variables with sane production defaults.

Environment variables:
    HOST            Bind address (default: 0.0.0.0)
    PORT            Bind port (default: 8004)
    LOG_LEVEL       Uvicorn log level (default: info)
    LINEAGE_DB_PATH SQLite path for lineage store (default: /tmp/lineage.db)
    WORKERS         Number of uvicorn worker processes (default: 1)
                    Keep at 1 unless the lineage store is backed by a
                    shared database, because in-process Prometheus metrics
                    are not shared across processes.

Usage:
    python -m services.observability.main
    # or
    uvicorn services.observability.main:app --host 0.0.0.0 --port 8004
"""

import os
import uvicorn
import structlog

from .api import app  # noqa: F401 — re-exported for uvicorn string reference

logger = structlog.get_logger(__name__)


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8004"))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    workers = int(os.environ.get("WORKERS", "1"))

    logger.info(
        "observability_service_starting",
        host=host,
        port=port,
        log_level=log_level,
        workers=workers,
    )

    uvicorn.run(
        # Use import string so uvicorn can reload in development mode.
        "services.observability.main:app",
        host=host,
        port=port,
        log_level=log_level,
        workers=workers,
    )


if __name__ == "__main__":
    main()
