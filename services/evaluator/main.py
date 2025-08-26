"""FastAPI service for the Safety Evaluation & Regression Gate."""
import structlog
import uvicorn
from fastapi import FastAPI
from prometheus_client import make_asgi_app

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Safety Evaluation & Regression Gate",
    version="1.0.0",
)

# Mount Prometheus ASGI app at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    return {"status": "ready"}


if __name__ == "__main__":
    uvicorn.run("services.evaluator.main:app", host="0.0.0.0", port=8004, reload=False)
