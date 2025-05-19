"""
EmbeddingEngine: sentence embedding + FAISS index for Module 3a.

Wraps sentence-transformers for batched L2-normalised float32 embeddings and
builds a FAISS IndexFlatIP for cosine-similarity nearest-neighbour search.

Because each embedding is L2-normalised to unit length, inner-product (IP)
search over FAISS is equivalent to cosine similarity — no separate distance
conversion is required.

Public types
------------
FailureRecord   — raw failure event (prompt + response + metadata)
EmbeddedFailure — FailureRecord augmented with its embedding vector

Public class
------------
EmbeddingEngine
    embed_batch(failures)            -> List[EmbeddedFailure]
    get_embedding_matrix(embedded)   -> np.ndarray  (N, dim) float32
    find_similar(query, top_k)       -> List[Tuple[int, float]]

Complexity
----------
  embed_batch:           O(N * seq_len * model_params)
  get_embedding_matrix:  O(N * dim)
  find_similar:          O(N * dim)  (exact exhaustive IndexFlatIP)

Memory
------
  Embedding matrix:  O(N * dim * 4 bytes)  (float32)
  FAISS index:       O(N * dim * 4 bytes)  (flat index, no compression)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    """
    A single failure event from the evidence pipeline.

    Fields
    ------
    request_id  : globally unique identifier for the inference request
    prompt      : the input text sent to the model
    response    : the model's (incorrect) output
    mean_logprob: mean per-token log-probability from the logprob side-channel
    timestamp   : ISO-8601 wall-clock time of the failure event
    """
    request_id: str
    prompt: str
    response: str
    mean_logprob: float
    timestamp: str


@dataclass
class EmbeddedFailure:
    """
    FailureRecord augmented with its dense embedding vector.

    Fields
    ------
    request_id  : passthrough from FailureRecord
    prompt      : passthrough from FailureRecord
    response    : passthrough from FailureRecord
    embedding   : L2-normalised float32 vector of shape (dim,); norm ~= 1.0
    mean_logprob: passthrough from FailureRecord
    """
    request_id: str
    prompt: str
    response: str
    embedding: np.ndarray   # shape (dim,), dtype float32
    mean_logprob: float


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Sentence embedding + FAISS nearest-neighbour index.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to all-MiniLM-L6-v2 (384-dim,
        ~80 MB, good general-purpose quality for English failure messages).

    Invariants
    ----------
    * Every embedding stored in the FAISS index is L2-normalised (norm ~= 1.0).
    * The internal FAISS index is reset and repopulated on each embed_batch call,
      so find_similar always searches over the most recently embedded batch.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)
        self._dim: int = self._model.get_sentence_embedding_dimension()

        # IndexFlatIP: exact exhaustive inner-product search.
        # For unit-norm vectors inner-product equals cosine similarity.
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_batch(self, failures: List[FailureRecord]) -> List[EmbeddedFailure]:
        """
        Encode each FailureRecord's prompt into a normalised embedding.

        Also resets and repopulates the internal FAISS index so that
        find_similar can be called immediately after.

        Parameters
        ----------
        failures:
            List of FailureRecord objects (may be empty).

        Returns
        -------
        List of EmbeddedFailure objects in the same order as the input.
        Each .embedding is float32 with L2-norm ~= 1.0.

        Side effects:
            Clears and refills the internal FAISS index.

        Complexity: O(N * seq_len * model_params)
        """
        if not failures:
            return []

        prompts: List[str] = [f.prompt for f in failures]

        # sentence_transformers' normalize_embeddings=True applies in-place
        # L2 normalisation before returning — avoids a second pass over the matrix.
        raw: np.ndarray = self._model.encode(
            prompts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)  # shape (N, dim)

        # Rebuild FAISS index from the fresh embedding matrix.
        self._index.reset()
        self._index.add(raw)

        return [
            EmbeddedFailure(
                request_id=f.request_id,
                prompt=f.prompt,
                response=f.response,
                embedding=raw[i],
                mean_logprob=f.mean_logprob,
            )
            for i, f in enumerate(failures)
        ]

    def get_embedding_matrix(self, embedded: List[EmbeddedFailure]) -> np.ndarray:
        """
        Stack embeddings into a contiguous (N, dim) float32 matrix.

        Parameters
        ----------
        embedded:
            List of EmbeddedFailure objects as returned by embed_batch.

        Returns
        -------
        np.ndarray of shape (N, dim), dtype float32.
        Returns shape (0, dim) for an empty list.

        Complexity: O(N * dim)
        """
        if not embedded:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([e.embedding for e in embedded], axis=0)

    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Return the top-k nearest neighbours of query_embedding in the FAISS index.

        The index must have been populated via embed_batch before calling this.

        Parameters
        ----------
        query_embedding:
            1-D float32 array of shape (dim,).  L2-normalised internally if needed.
        top_k:
            Number of nearest neighbours to return.  Clamped to index size.

        Returns
        -------
        List of (index, similarity_score) tuples sorted descending by score.
        similarity_score is in [-1, 1] (cosine similarity for unit vectors).

        Complexity: O(N * dim) — exact exhaustive search.
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)

        # Ensure the query is float32 and L2-normalised.
        q = query_embedding.astype(np.float32).copy()
        norm = float(np.linalg.norm(q))
        if norm > 0.0:
            q /= norm

        distances, indices = self._index.search(q.reshape(1, -1), k)
        # distances[0]: inner-product scores (~= cosine similarity)
        # indices[0]:   0-based positions in the index
        return [
            (int(indices[0][i]), float(distances[0][i]))
            for i in range(k)
        ]
