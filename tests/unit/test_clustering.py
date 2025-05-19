"""
Functional tests for Module 3a: embedding + clustering components.

All tests run end-to-end against real sentence-transformer inference and
real HDBSCAN clustering — no mocking.  The model (all-MiniLM-L6-v2, ~80 MB)
is downloaded on first run and cached by HuggingFace locally.

Run with:
    pytest tests/unit/test_clustering.py -v
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from services.synthesizer.config import SynthesizerConfig
from services.synthesizer.embedding import EmbeddingEngine, EmbeddedFailure, FailureRecord
from services.synthesizer.clustering import FailureClusterer, FailureCluster


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def make_failures_with_two_clusters(n_per_cluster: int = 20) -> List[FailureRecord]:
    """Create synthetic failures with 2 clear semantic clusters."""
    cluster1 = [
        FailureRecord(
            request_id=f"req-math-{i}",
            prompt=f"What is {i}*{i+1}? Show your work step by step",
            response=f"The answer is {i*(i+1)+1}",  # wrong answer
            mean_logprob=-2.5,
            timestamp="2026-03-28T00:00:00Z",
        )
        for i in range(n_per_cluster)
    ]
    cluster2 = [
        FailureRecord(
            request_id=f"req-code-{i}",
            prompt=f"Write a Python function to sort a list using bubble sort variant {i}",
            response=f"def sort(lst): return sorted(lst)  # wrong implementation",
            mean_logprob=-2.8,
            timestamp="2026-03-28T00:00:00Z",
        )
        for i in range(n_per_cluster)
    ]
    return cluster1 + cluster2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_embedding_produces_correct_shape():
    """Embeddings must be non-zero vectors of consistent dimension."""
    engine = EmbeddingEngine()
    failures = make_failures_with_two_clusters(5)
    embedded = engine.embed_batch(failures)
    assert len(embedded) == 10
    dim = embedded[0].embedding.shape[0]
    assert dim > 0
    for e in embedded:
        assert e.embedding.shape == (dim,)
        assert not np.all(e.embedding == 0), "Embedding must be non-zero"
        # L2 norm should be ~1.0 for cosine similarity
        norm = np.linalg.norm(e.embedding)
        assert 0.99 < norm < 1.01, f"Embedding not normalized: norm={norm}"


def test_clustering_separates_semantic_groups():
    """Math failures and code failures must land in different clusters."""
    config = SynthesizerConfig(hdbscan_min_cluster_size=5, hdbscan_min_samples=2)
    engine = EmbeddingEngine()
    clusterer = FailureClusterer(config, engine)
    failures = make_failures_with_two_clusters(15)
    embedded = engine.embed_batch(failures)
    clusters, noise = clusterer.cluster(embedded)

    # Must find at least 2 clusters
    assert len(clusters) >= 2, f"Expected >=2 clusters, got {len(clusters)}"

    # Math cluster and code cluster should not overlap
    math_ids = {f"req-math-{i}" for i in range(15)}
    code_ids = {f"req-code-{i}" for i in range(15)}

    for cluster in clusters:
        cluster_req_ids = {s.request_id for s in cluster.representative_samples}
        math_in_cluster = cluster_req_ids & math_ids
        code_in_cluster = cluster_req_ids & code_ids
        # A cluster should be dominated by one type
        if len(math_in_cluster) > 0 and len(code_in_cluster) > 0:
            dominant = max(len(math_in_cluster), len(code_in_cluster))
            total = len(math_in_cluster) + len(code_in_cluster)
            purity = dominant / total
            assert purity > 0.7, f"Cluster not pure enough: purity={purity}"


def test_faiss_similarity_search():
    """find_similar must return correct nearest neighbors."""
    engine = EmbeddingEngine()
    failures = make_failures_with_two_clusters(10)
    embedded = engine.embed_batch(failures)

    # Query with a math-like embedding (first math failure)
    query = embedded[0].embedding
    results = engine.find_similar(query, top_k=3)

    assert len(results) == 3
    # Top result should be itself or very similar (math failures)
    top_idx, top_score = results[0]
    assert top_score > 0.9, f"Top result similarity too low: {top_score}"


def test_noise_points_handled():
    """HDBSCAN noise points must be returned separately, not silently dropped."""
    config = SynthesizerConfig(hdbscan_min_cluster_size=100)  # high threshold = more noise
    engine = EmbeddingEngine()
    clusterer = FailureClusterer(config, engine)
    failures = make_failures_with_two_clusters(5)  # 10 total, min_cluster=100 -> all noise
    embedded = engine.embed_batch(failures)
    clusters, noise = clusterer.cluster(embedded)

    total = sum(c.member_count for c in clusters) + len(noise)
    assert total == len(embedded), f"Events lost: {len(embedded)} in, {total} accounted for"


def test_cluster_quality_filter():
    """Low persistence clusters must be filtered out."""
    config = SynthesizerConfig(hdbscan_min_cluster_size=5, hdbscan_min_samples=2)
    engine = EmbeddingEngine()
    clusterer = FailureClusterer(config, engine)
    failures = make_failures_with_two_clusters(15)
    embedded = engine.embed_batch(failures)
    clusters, _ = clusterer.cluster(embedded)

    filtered = clusterer.filter_low_quality(clusters, min_persistence=0.0)
    assert len(filtered) == len(clusters)  # 0.0 threshold keeps all

    filtered_strict = clusterer.filter_low_quality(clusters, min_persistence=99.0)
    assert len(filtered_strict) == 0  # 99.0 threshold removes all


def test_archetype_description_non_empty():
    """Every cluster must have a non-empty archetype description."""
    config = SynthesizerConfig(hdbscan_min_cluster_size=5, hdbscan_min_samples=2)
    engine = EmbeddingEngine()
    clusterer = FailureClusterer(config, engine)
    failures = make_failures_with_two_clusters(15)
    embedded = engine.embed_batch(failures)
    clusters, _ = clusterer.cluster(embedded)

    for cluster in clusters:
        assert cluster.archetype_description, f"Cluster {cluster.cluster_id} has empty description"
        assert len(cluster.archetype_description) > 10
