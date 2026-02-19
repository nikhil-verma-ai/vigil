# Autonomous LLM Continuous Fine-Tuning Platform — Implementation Plan

**Goal:** Build a fully autonomous LLM self-improvement infrastructure with 8 modules: side-channel signal capture, drift detection, synthetic data amplification, QLoRA training, safety evaluation, zero-downtime deployment, observability, and autonomous loop orchestration.

**Architecture:** Co-located sidecar services alongside vLLM inference; Kafka event backbone; Redis coordination; ephemeral spot GPU training jobs; all modules communicate via typed Kafka events and REST APIs.

**Tech Stack:** Rust (side-channel), Python 3.11 (all services), FastAPI, Kafka (confluent-kafka), Redis, PostgreSQL (lineage), HDBSCAN (hdbscan/cuml), sentence-transformers, transformers+peft+trl (training), Prometheus, Docker Compose (local env)

**CRITICAL TEST REQUIREMENT:** Every test must be a FUNCTIONAL test — it exercises real logic paths, real data flows, real state transitions. NO tests that only verify "it compiled" or "function exists". Every test must assert on actual behavior, actual state, actual outputs.

---

## Module Breakdown

### Module 1 — Rust Side-Channel
- `rust/side_channel/Cargo.toml`
- `rust/side_channel/src/main.rs`
- `rust/side_channel/src/capture.rs`
- `rust/side_channel/src/publisher.rs`
- `rust/side_channel/src/metrics.rs`
- `rust/side_channel/src/config.rs`
- `rust/side_channel/tests/`

### Module 2 — Drift Detector
- `services/drift_detector/main.py`
- `services/drift_detector/detector.py`
- `services/drift_detector/baseline.py`
- `services/drift_detector/scorer.py`
- `services/drift_detector/alerting.py`
- `services/drift_detector/requirements.txt`
- `tests/unit/test_drift_detector.py`
- `tests/integration/test_drift_pipeline.py`

### Module 3 — Synthesizer: Clustering
- `services/synthesizer/clustering.py`
- `services/synthesizer/embedding.py`
- `services/synthesizer/config.py`
- `tests/unit/test_clustering.py`

### Module 4 — Synthesizer: LLM Judge + Pipeline
- `services/synthesizer/judge.py`
- `services/synthesizer/amplifier.py`
- `services/synthesizer/pipeline.py`
- `services/synthesizer/main.py`
- `services/synthesizer/requirements.txt`
- `tests/unit/test_synthesizer.py`
- `tests/integration/test_synthesis_pipeline.py`

### Module 5 — Trainer: SFT
- `services/trainer/sft.py`
- `services/trainer/qlora_config.py`
- `services/trainer/dataset.py`
- `services/trainer/checkpointing.py`
- `tests/unit/test_sft.py`

### Module 6 — Trainer: DPO + Orchestrator
- `services/trainer/dpo.py`
- `services/trainer/orchestrator.py`
- `services/trainer/gpu_provisioner.py`
- `services/trainer/cost_tracker.py`
- `services/trainer/main.py`
- `services/trainer/requirements.txt`
- `tests/unit/test_dpo.py`
- `tests/integration/test_training_orchestrator.py`

### Module 7 — Safety Evaluator
- `services/evaluator/gate.py`
- `services/evaluator/benchmarks.py`
- `services/evaluator/behavioral.py`
- `services/evaluator/safety.py`
- `services/evaluator/main.py`
- `services/evaluator/requirements.txt`
- `tests/unit/test_evaluator.py`
- `tests/integration/test_evaluation_gate.py`

### Module 8 — Deployment Engine
- `services/deployer/engine.py`
- `services/deployer/state_machine.py`
- `services/deployer/redis_coordinator.py`
- `services/deployer/vllm_client.py`
- `services/deployer/rollback.py`
- `services/deployer/main.py`
- `services/deployer/requirements.txt`
- `tests/unit/test_deployer.py`
- `tests/integration/test_deployment_engine.py`

### Module 9 — Observability
- `services/observability/metrics.py`
- `services/observability/dashboards.py`
- `services/observability/lineage.py`
- `services/observability/api.py`
- `services/observability/main.py`
- `services/observability/requirements.txt`
- `infra/grafana/dashboards/`
- `tests/unit/test_observability.py`

### Module 10 — Orchestrator + Infrastructure
- `services/orchestrator/loop.py`
- `services/orchestrator/state_machine.py`
- `services/orchestrator/triggers.py`
- `services/orchestrator/main.py`
- `services/orchestrator/requirements.txt`
- `infra/docker-compose.yml`
- `infra/docker-compose.test.yml`
- `infra/kafka/topics.sh`
- `infra/redis/redis.conf`
- `Makefile`
- `tests/integration/test_loop_orchestrator.py`
