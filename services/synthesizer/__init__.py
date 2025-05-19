"""
Synthesizer service — Module 3 of the Autonomous Fine-Tuning Platform.

Module 3a (embedding + clustering):
  Embeds production failure records with sentence-transformers, clusters
  them via HDBSCAN, and yields stable FailureCluster objects.

Module 3b (synthesis + judge):
  Amplifies each cluster 20x via an oracle LLM, evaluates every preference
  pair through an LLM-as-Judge quality gate, and emits DPO + SFT datasets.

Public API surface (consumed by orchestrator and downstream training jobs):
"""
from services.synthesizer.config import SynthesizerConfig
from services.synthesizer.embedding import EmbeddingEngine, EmbeddedFailure, FailureRecord
from services.synthesizer.clustering import FailureClusterer, FailureCluster

__all__ = [
    "SynthesizerConfig",
    "EmbeddingEngine",
    "EmbeddedFailure",
    "FailureRecord",
    "FailureClusterer",
    "FailureCluster",
]
