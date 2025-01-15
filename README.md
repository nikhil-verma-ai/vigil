# Vigil

Autonomous fine-tuning infrastructure for LLMs. Vigil closes the loop between inference and training — it watches your model in production, detects when it's drifting, synthesizes new training data from failure cases, fine-tunes, validates safety, and promotes the new weights. Zero human intervention required.

**This is v1.** The core pipeline is here — signal capture, drift detection, data synthesis, QLoRA training, safety eval, and deployment. We've shipped a lot since this snapshot that we're keeping close to the chest for now. Think of this as the foundation.

## Architecture

```
vLLM inference
    └── Rust side-channel (zero-copy logit capture)
            └── Kafka
                    ├── Drift Detector (HDBSCAN)
                    ├── Data Synthesizer (SFT/DPO dataset gen)
                    ├── Trainer (QLoRA via PEFT/TRL)
                    ├── Safety Evaluator (regression gate)
                    └── Deployer (canary + rollback)
```

Central orchestrator drives the whole cycle as a state machine. Redis for coordination, PostgreSQL for lineage, Prometheus for metrics.

## Stack

- **Rust** — side-channel signal capture (Tokio, rdkafka)
- **Python** — all services (FastAPI, transformers, peft, trl, sentence-transformers)
- **Kafka** — event backbone
- **Redis** — state coordination
- **PostgreSQL** — training lineage
- **Docker Compose** — local orchestration

## Quickstart

```bash
# Spin up infra
docker compose -f infra/docker-compose.yml up -d

# Run tests
make test
```

## Project Structure

```
rust/side_channel/    # Signal capture agent
services/             # Python microservices (one per pipeline stage)
shared/               # Common config, models, utils
infra/                # Docker, Kafka, Redis config
tests/                # Integration + unit tests
```

---

Built by the vigil-research-and-development team.
