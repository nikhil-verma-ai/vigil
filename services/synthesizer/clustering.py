"""
FailureClusterer: HDBSCAN clustering + archetype extraction for Module 3a.

Key types
---------
FailureCluster  — a dense group of semantically related failures with a
                  centroid embedding, HDBSCAN persistence score,
                  representative FailureRecord samples, and a keyword-based
                  archetype description.

Algorithm
---------
1. Accept a List[EmbeddedFailure] (already embedded by EmbeddingEngine).
2. Build an (N x N) cosine-distance matrix (precomputed, float64) and run HDBSCAN.
   Precomputed distances avoid HDBSCAN's internal cosine-metric buffer dtype check
   that rejects float32 inputs.
3. For each discovered cluster (label != -1):
   - Collect member EmbeddedFailures.
   - Compute mean-centroid of member embeddings (float32).
   - Select up to max_archetype_samples representatives closest to centroid.
   - Record HDBSCAN cluster_persistence_ entry for quality filtering.
   - Generate a human-readable archetype description via keyword extraction.
4. Return (clusters, noise_points) where noise_points holds all points
   labelled -1 by HDBSCAN.

Invariant: sum(c.member_count for c in clusters) + len(noise_points) == len(input)

Complexity: O(N^2) for the precomputed distance matrix + HDBSCAN.
Memory:     O(N^2) for the float64 distance matrix.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_distances  # type: ignore

from services.synthesizer.config import SynthesizerConfig
from services.synthesizer.embedding import EmbeddingEngine, EmbeddedFailure, FailureRecord


# ---------------------------------------------------------------------------
# Stop-word set for lightweight keyword extraction (no NLTK dependency)
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "what", "which", "who", "whom", "whose", "how", "when",
    "where", "why", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "and", "but", "or", "nor", "so", "yet", "both",
    "either", "neither", "not", "only", "own", "same", "than", "too",
    "very", "your", "my", "his", "her", "its", "our", "their", "this",
    "that", "these", "those", "show", "write", "using", "each", "use",
    "give", "make", "get", "let", "take", "put", "set", "go", "come",
    "see", "look", "want", "know", "think", "say", "tell", "ask", "work",
    "seem", "feel", "try", "leave", "call", "keep", "run", "move", "live",
    "believe", "hold", "bring", "happen", "must", "also", "just", "like",
    "new", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "step", "steps", "correct", "wrong", "right", "answer",
    "output", "input", "result", "return", "returns", "please", "implement",
})


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class FailureCluster:
    """
    A dense semantic cluster of failure events.

    Fields
    ------
    cluster_id              HDBSCAN integer label (0-indexed).
    member_count            total EmbeddedFailures assigned to this cluster.
    persistence_score       HDBSCAN branch persistence in [0, inf).
                            Higher value = more stable / denser density peak.
    centroid_embedding      mean of all member embeddings (float32, shape (dim,)).
    representative_samples  FailureRecord objects closest to the centroid,
                            up to config.max_archetype_samples.
    archetype_description   human-readable summary of the cluster failure pattern.
    """
    cluster_id: int
    member_count: int
    persistence_score: float
    centroid_embedding: np.ndarray
    representative_samples: List[FailureRecord]
    archetype_description: str = ""


# ---------------------------------------------------------------------------
# FailureClusterer
# ---------------------------------------------------------------------------

class FailureClusterer:
    """
    Groups a pre-embedded batch of failure events into FailureClusters.

    Parameters
    ----------
    config:
        SynthesizerConfig controlling HDBSCAN hyper-parameters and archetype
        sample count.
    embedding_engine:
        EmbeddingEngine instance.  Accepted for interface symmetry with the
        broader pipeline; clustering operates on the already-computed embedding
        matrix from List[EmbeddedFailure] — no re-embedding is performed.

    Invariants
    ----------
    * cluster() always returns (clusters, noise) such that
      sum(c.member_count for c in clusters) + len(noise) == len(input).
    * HDBSCAN label -1 points are returned as noise; never included in clusters.
    * representative_samples are FailureRecord objects reconstructed from the
      EmbeddedFailure input.
    """

    def __init__(
        self,
        config: SynthesizerConfig,
        embedding_engine: EmbeddingEngine,
    ) -> None:
        self._config = config
        # embedding_engine retained for interface completeness (Module 3b may use it)
        self._embedder = embedding_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        embedded_failures: List[EmbeddedFailure],
    ) -> Tuple[List[FailureCluster], List[EmbeddedFailure]]:
        """
        Run HDBSCAN on the embedding matrix and partition results.

        Parameters
        ----------
        embedded_failures:
            List[EmbeddedFailure] as produced by EmbeddingEngine.embed_batch().

        Returns
        -------
        (clusters, noise_points)
            clusters:     List[FailureCluster] sorted descending by persistence.
            noise_points: List[EmbeddedFailure] for all HDBSCAN label == -1 points.

        Invariants
        ----------
        sum(c.member_count for c in clusters) + len(noise_points) == len(embedded_failures)

        Complexity: O(N^2)
        """
        import hdbscan  # type: ignore

        n = len(embedded_failures)
        if n == 0:
            return [], []

        # Build (N, dim) float64 matrix.
        # HDBSCAN's precomputed path requires float64 (C-level buffer dtype check).
        matrix_f32: np.ndarray = np.stack(
            [e.embedding for e in embedded_failures], axis=0
        )  # (N, dim) float32
        matrix_f64 = matrix_f32.astype(np.float64)

        # Cosine distance matrix: values in [0, 2].
        # sklearn handles edge cases (zero-norm vectors) correctly.
        dist_matrix: np.ndarray = cosine_distances(matrix_f64)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._config.hdbscan_min_cluster_size,
            min_samples=self._config.hdbscan_min_samples,
            metric="precomputed",
            cluster_selection_method=self._config.hdbscan_cluster_selection_method,
        )
        labels: np.ndarray = clusterer.fit_predict(dist_matrix)

        # cluster_persistence_: 1-D array indexed by cluster label (hdbscan >= 0.8).
        raw_persistence: np.ndarray = np.asarray(
            getattr(clusterer, "cluster_persistence_", []),
            dtype=np.float64,
        )

        # Partition into noise and clustered points.
        noise_points: List[EmbeddedFailure] = [
            embedded_failures[i] for i in range(n) if labels[i] == -1
        ]

        unique_labels = sorted(set(int(lbl) for lbl in labels) - {-1})
        clusters: List[FailureCluster] = []

        for label in unique_labels:
            mask = labels == label
            member_ef: List[EmbeddedFailure] = [
                embedded_failures[i] for i in range(n) if mask[i]
            ]
            member_embeddings = matrix_f32[mask]  # (k, dim) float32

            # Raw mean centroid (not re-normalised — preserves geometric center).
            centroid = member_embeddings.mean(axis=0).astype(np.float32)

            # Select representatives closest to centroid (Euclidean distance in
            # embedding space is a good proxy for semantic typicality).
            k_reps = min(self._config.max_archetype_samples, len(member_ef))
            dists_to_centroid = np.linalg.norm(member_embeddings - centroid, axis=1)
            closest_idx = np.argsort(dists_to_centroid)[:k_reps]
            reps: List[FailureRecord] = [
                FailureRecord(
                    request_id=member_ef[int(i)].request_id,
                    prompt=member_ef[int(i)].prompt,
                    response=member_ef[int(i)].response,
                    mean_logprob=member_ef[int(i)].mean_logprob,
                    timestamp="",
                )
                for i in closest_idx
            ]

            # Guard: label may exceed raw_persistence length when all points are
            # noise (cluster_persistence_ is empty in that case).
            persistence = (
                float(raw_persistence[label])
                if label < len(raw_persistence)
                else 0.0
            )

            cluster = FailureCluster(
                cluster_id=label,
                member_count=int(mask.sum()),
                persistence_score=persistence,
                centroid_embedding=centroid,
                representative_samples=reps,
                archetype_description="",
            )
            # Populate archetype description without an LLM (Module 3b enriches later).
            cluster.archetype_description = self.describe_cluster(cluster)
            clusters.append(cluster)

        clusters.sort(key=lambda c: c.persistence_score, reverse=True)
        return clusters, noise_points

    def describe_cluster(self, cluster: FailureCluster) -> str:
        """
        Generate a human-readable archetype description from representative prompts.

        Strategy: extract the most frequent non-stop-word lowercase tokens from
        representative sample prompts, then format as:
        "Cluster of N failures involving: <kw1>, <kw2>, ..."

        No LLM is used — this is a lightweight keyword heuristic.
        Module 3b overlays richer LLM-generated descriptions downstream.

        Parameters
        ----------
        cluster:
            FailureCluster whose representative_samples contain the prompts
            to analyse.

        Returns
        -------
        Non-empty string of length > 10.

        Complexity: O(total_token_count_in_prompts)
        """
        if not cluster.representative_samples:
            return f"Cluster of {cluster.member_count} failures (no representative samples)"

        all_tokens: List[str] = []
        for rec in cluster.representative_samples:
            tokens = re.findall(r"[a-z][a-z0-9]*", rec.prompt.lower())
            all_tokens.extend(
                tok for tok in tokens
                if tok not in _STOPWORDS and len(tok) > 2
            )

        counter = Counter(all_tokens)
        top_keywords = [kw for kw, _ in counter.most_common(5)]

        if top_keywords:
            kw_str = ", ".join(top_keywords)
            return f"Cluster of {cluster.member_count} failures involving: {kw_str}"

        # Fallback for very short or stop-word-only prompts.
        return (
            f"Cluster of {cluster.member_count} failures"
            f" (cluster_id={cluster.cluster_id})"
        )

    def compute_cluster_quality(
        self,
        clusters: List[FailureCluster],
    ) -> Dict[str, float]:
        """
        Return a quality score dict for each cluster.

        Current quality metric: HDBSCAN persistence_score — a higher value
        indicates a more stable, well-separated density peak.

        Parameters
        ----------
        clusters:
            List[FailureCluster] as returned by cluster().

        Returns
        -------
        Dict mapping str(cluster_id) -> persistence_score (float).

        Complexity: O(len(clusters))
        """
        return {str(c.cluster_id): c.persistence_score for c in clusters}

    def filter_low_quality(
        self,
        clusters: List[FailureCluster],
        min_persistence: float = 0.1,
    ) -> List[FailureCluster]:
        """
        Remove clusters with persistence_score < min_persistence.

        Clusters below the threshold correspond to density peaks that HDBSCAN
        deemed transient — likely noise artefacts rather than genuine failure modes.

        Parameters
        ----------
        clusters:
            List[FailureCluster] to filter.
        min_persistence:
            Inclusive lower bound.  Pass 0.0 to retain all; pass a large value
            (e.g. 99.0) to discard all.

        Returns
        -------
        List[FailureCluster] with persistence_score >= min_persistence,
        preserving the original sort order.

        Complexity: O(len(clusters))
        """
        return [c for c in clusters if c.persistence_score >= min_persistence]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_persistences(clusterer) -> Dict[int, float]:
        """
        Safe extraction of cluster_persistence_ from a fitted HDBSCAN object.

        Returns dict mapping cluster_label -> persistence_score.
        Falls back to empty dict on AttributeError (e.g. mock clusterers).
        """
        try:
            arr = clusterer.cluster_persistence_
            return {i: float(p) for i, p in enumerate(arr)}
        except AttributeError:
            return {}
