#!/usr/bin/env bash
# init-topics.sh — Idempotent Kafka topic creation for the autonomous fine-tuning platform.
#
# Usage (run inside the kafka container or against a reachable broker):
#   BOOTSTRAP_SERVERS=kafka:29092 ./init-topics.sh
#
# Topics match shared/schemas/events.py constants exactly.  Never rename
# a topic here without updating the constants file — they are the single
# source of truth for topic names.
#
# Partitioning rationale:
#   logprob.signals    — 3 partitions: high-volume per-request events;
#                         3 allows parallel consumers across 3 drift-detector
#                         replicas without rebalance overhead.
#   drift.events       — 1 partition: low-volume, order matters for the
#                         orchestrator's sequential cycle logic.
#   synthesis.jobs     — 1 partition: 1 job at a time per design.
#   training.jobs      — 1 partition: serialised training cycles.
#   evaluation.results — 1 partition: serialised evaluation gate decisions.
#   adapter.promotions — 1 partition: ordered promotion history is critical
#                         for rollback forensics.

set -euo pipefail

BOOTSTRAP="${BOOTSTRAP_SERVERS:-kafka:29092}"
REPLICATION="${REPLICATION_FACTOR:-1}"

echo "[init-topics] Bootstrap: ${BOOTSTRAP}"

create_topic() {
    local topic="$1"
    local partitions="$2"
    kafka-topics \
        --create \
        --if-not-exists \
        --bootstrap-server "${BOOTSTRAP}" \
        --partitions "${partitions}" \
        --replication-factor "${REPLICATION}" \
        --topic "${topic}"
    echo "[init-topics] OK: ${topic} (${partitions}p)"
}

create_topic "logprob.signals"     3
create_topic "drift.events"        1
create_topic "synthesis.jobs"      1
create_topic "training.jobs"       1
create_topic "evaluation.results"  1
create_topic "adapter.promotions"  1

echo "[init-topics] All topics created."
