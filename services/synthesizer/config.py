"""
SynthesizerConfig — single source of truth for all tunable parameters in the
embedding + clustering pipeline (Module 3a) and synthesis pipeline (Module 3b).



























All fields carry explicit types and defaults so the service can be instantiated
with zero env-var wiring in tests.  The from_env() constructor reads a curated
subset of overridable variables for production deployments.

Complexity: O(1) construction; no I/O at import time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SynthesizerConfig:
    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    # Sentence-transformer model used for embedding failure prompts.
    # all-MiniLM-L6-v2 is ~80 MB, 384-dim, fast enough for CI.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Batch size for encoding calls — reduces GPU round-trips
    embedding_batch_size: int = 64

    # ------------------------------------------------------------------
    # HDBSCAN clustering
    # ------------------------------------------------------------------
    # Minimum number of points required to form a cluster.
    # Below this threshold, points are labelled as noise (-1).
    hdbscan_min_cluster_size: int = 5

    # Controls how conservative the clustering is.  Higher value -> more
    # noise points, fewer spurious micro-clusters.
    hdbscan_min_samples: int = 3

    # Intended similarity space (stored for documentation; at runtime we
    # pass a precomputed cosine-distance matrix to HDBSCAN).
    hdbscan_metric: str = "cosine"

    # "eom" (Excess of Mass) discovers variable-density clusters.
    # "leaf" produces smaller, more homogeneous clusters.
    hdbscan_cluster_selection_method: str = "eom"

    # Minimum persistence score for a cluster to be retained.
    # Clusters below this are discarded as unstable density peaks.
    min_persistence_score: float = 0.0

    # ------------------------------------------------------------------
    # Archetype / quality
    # ------------------------------------------------------------------
    # Maximum representative FailureRecord samples kept per cluster.
    max_archetype_samples: int = 10

    # Noise-point classifier threshold (used by Module 3b; stored here as
    # single config source).
    noise_classifier_threshold: float = 0.7

    # ------------------------------------------------------------------
    # Oracle / Judge models (Module 3b)
    # ------------------------------------------------------------------
    oracle_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"

    # All judge dimensions must be >= this to pass the quality gate
    judge_pass_threshold: float = 0.7

    # Temperature settings
    oracle_temperature: float = 0.2
    judge_temperature: float = 0.0

    # Max tokens for oracle responses
    oracle_max_tokens: int = 512
    # Max tokens for variant-prompt generation (batch call)
    variant_prompt_max_tokens: int = 1024

    # ------------------------------------------------------------------
    # Amplification (Module 3b)
    # ------------------------------------------------------------------
    # Target number of synthetic pairs to generate per cluster
    amplification_target_per_cluster: int = 20

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------
    # Rough per-pair cost estimate (oracle call + judge call)
    cost_per_pair_usd: float = 0.01

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_base_dir: str = "/tmp/synthesis"

    # ------------------------------------------------------------------
    # Kafka
    # ------------------------------------------------------------------
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_group_id: str = "synthesizer-group"

    # Optional; None means no real API calls (tests inject mock clients)
    openai_api_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Environment-variable constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "SynthesizerConfig":
        """
        Construct config from environment variables, falling back to the
        class-level defaults for any variable that is absent.

        Supported env vars:
            EMBEDDING_MODEL             str
            HDBSCAN_MIN_CLUSTER_SIZE    int
            ORACLE_MODEL                str
            JUDGE_MODEL                 str
            JUDGE_PASS_THRESHOLD        float
            OPENAI_API_KEY              str
            KAFKA_BOOTSTRAP_SERVERS     str
            OUTPUT_BASE_DIR             str
        """
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            hdbscan_min_cluster_size=int(
                os.getenv(
                    "HDBSCAN_MIN_CLUSTER_SIZE",
                    str(cls.hdbscan_min_cluster_size),
                )
            ),
            oracle_model=os.getenv("ORACLE_MODEL", cls.oracle_model),
            judge_model=os.getenv("JUDGE_MODEL", cls.judge_model),
            judge_pass_threshold=float(
                os.getenv("JUDGE_PASS_THRESHOLD", str(cls.judge_pass_threshold))
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            kafka_bootstrap_servers=os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", cls.kafka_bootstrap_servers
            ),
            output_base_dir=os.getenv("OUTPUT_BASE_DIR", cls.output_base_dir),
        )
