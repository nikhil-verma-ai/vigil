.PHONY: test test-unit test-integration test-e2e test-simulation test-rust \
        lint build infra-up infra-down infra-test-up infra-test-down \
        install-deps build-rust

ROOT := $(CURDIR)

# ── Test targets ──────────────────────────────────────────────────────────────

test-unit:
	cd $(ROOT) && \
	python -m pytest tests/unit/ -v --tb=short -x

test-integration:
	cd $(ROOT) && \
	python -m pytest tests/integration/ -v --tb=short

test-e2e:
	cd $(ROOT) && \
	python -m pytest tests/e2e/ -v --tb=short

test-simulation:
	cd $(ROOT) && \
	python -m pytest tests/simulation/ -v --tb=short -s

# Default: unit + integration
test: test-unit test-integration

test-rust:
	cd $(ROOT)/rust/side_channel && \
	cargo test -- --test-threads=4

# ── Infrastructure ────────────────────────────────────────────────────────────

infra-up:
	docker compose -f $(ROOT)/infra/docker-compose.yml up -d
	docker compose -f $(ROOT)/infra/docker-compose.yml wait kafka-init

infra-down:
	docker compose -f $(ROOT)/infra/docker-compose.yml down -v

infra-test-up:
	docker compose -f $(ROOT)/infra/docker-compose.test.yml up -d

infra-test-down:
	docker compose -f $(ROOT)/infra/docker-compose.test.yml down -v

# ── Lint ──────────────────────────────────────────────────────────────────────

lint:
	python -m ruff check $(ROOT)/services/ $(ROOT)/tests/
	python -m mypy $(ROOT)/services/ --ignore-missing-imports

# ── Build ─────────────────────────────────────────────────────────────────────

build-rust:
	cd $(ROOT)/rust/side_channel && cargo build --release

install-deps:
	pip install -r $(ROOT)/services/drift_detector/requirements.txt
	pip install -r $(ROOT)/services/synthesizer/requirements.txt
	pip install -r $(ROOT)/services/trainer/requirements.txt
	pip install -r $(ROOT)/services/evaluator/requirements.txt
	pip install -r $(ROOT)/services/deployer/requirements.txt
	pip install -r $(ROOT)/services/observability/requirements.txt
	pip install -r $(ROOT)/services/orchestrator/requirements.txt
