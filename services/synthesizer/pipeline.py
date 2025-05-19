"""
SynthesisPipeline: orchestrates the full failure-to-dataset synthesis loop.

End-to-end flow
---------------
1. Receive List[FailureRecord] from the Kafka consumer (or direct call).
2. Embed all records via EmbeddingEngine.embed_batch().
3. Cluster with HDBSCAN via FailureClusterer.cluster().
4. For each cluster: amplify with SyntheticAmplifier.amplify_cluster().
5. Assemble SFT dataset: {prompt, response} — one record per passing pair.
6. Assemble DPO dataset: {prompt, chosen, rejected} — one record per passing pair.
7. Write both datasets as newline-delimited JSON (JSONL) under:
     {config.output_base_dir}/{job_id}/sft_data.jsonl
     {config.output_base_dir}/{job_id}/dpo_data.jsonl
8. Return SynthesisJobResult with paths and gate statistics.

Invariants
----------
- Files are always written (empty if 0 pairs pass) so downstream jobs can
  do existence checks without conditional logic.
- total_cost_usd sums AmplificationResult.synthesis_cost_estimate_usd across
  all clusters.
- duration_seconds is wall-clock time of the run() call.

Design notes
------------
- SynthesisPipeline is stateless after construction; run() is re-entrant.
- Dependency injection for all sub-systems (embedder, clusterer, amplifier)
  makes this fully testable without real API calls or model downloads.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from services.synthesizer.amplifier import AmplificationResult, SyntheticAmplifier
from services.synthesizer.clustering import FailureCluster, FailureClusterer
from services.synthesizer.config import SynthesizerConfig
from services.synthesizer.embedding import EmbeddingEngine, FailureRecord
from services.synthesizer.judge import LLMJudge, PreferencePair


# --------------------------------------------------------------------------- #
# Result type                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class SynthesisJobResult:
    """
    Summary of a completed synthesis job.

    Fields
    ------
    job_id:                      unique identifier for this run.
    trigger_event_id:            drift event or manual trigger ID.
    clusters_processed:          number of stable clusters found and amplified.
    total_pairs_synthesized:     sum of len(result.synthesized_pairs) across clusters.
    total_pairs_passing_gate:    sum of result.pairs_passing_gate across clusters.
    overall_amplification_factor: total_pairs_passing_gate / total_input_failures
                                  (0.0 if no failures).
    sft_dataset_path:            absolute path to the SFT JSONL file.
    dpo_dataset_path:            absolute path to the DPO JSONL file.
    total_cost_usd:              sum of cost estimates across all clusters.
    duration_seconds:            wall-clock time of the run() call.
    """
    job_id: str
    trigger_event_id: str
    clusters_processed: int
    total_pairs_synthesized: int
    total_pairs_passing_gate: int
    overall_amplification_factor: float
    sft_dataset_path: str
    dpo_dataset_path: str
    total_cost_usd: float
    duration_seconds: float


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #

class SynthesisPipeline:
    """
    Orchestrates the full failure record -> DPO dataset synthesis pipeline.

    Parameters
    ----------
    embedding_engine:
        EmbeddingEngine for embedding FailureRecords.
    clusterer:
        FailureClusterer for HDBSCAN clustering.
    amplifier:
        SyntheticAmplifier for 20x cluster amplification.
    config:
        SynthesizerConfig.  Defaults to SynthesizerConfig().
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        clusterer: FailureClusterer,
        amplifier: SyntheticAmplifier,
        config: SynthesizerConfig = None,
    ) -> None:
        self._embedder = embedding_engine
        self._clusterer = clusterer
        self._amplifier = amplifier
        self._config = config or SynthesizerConfig()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        failure_records: List[FailureRecord],
        job_id: str,
        trigger_event_id: str,
    ) -> SynthesisJobResult:
        """
        Execute the full synthesis pipeline.

        Parameters
        ----------
        failure_records:
            Raw FailureRecord objects from the evidence qualification layer.
        job_id:
            Unique string identifier for this synthesis job.
        trigger_event_id:
            ID of the DriftEvent or manual trigger that initiated this job.

        Returns
        -------
        SynthesisJobResult with dataset paths and gate statistics.

        Side effects
        ------------
        - Creates directory {config.output_base_dir}/{job_id}/.
        - Writes sft_data.jsonl and dpo_data.jsonl to that directory.
          Files are always created (may be empty if no clusters found).
        """
        start_time = time.monotonic()

        # 1. Embed all records
        embedded = self._embedder.embed_batch(failure_records)

        # 2. Cluster (takes List[EmbeddedFailure], returns Tuple[clusters, noise])
        clusters, _noise = self._clusterer.cluster(embedded)

        # 3. Amplify each cluster
        amplification_results: List[AmplificationResult] = []
        for cluster in clusters:
            # Extract production model responses from representative samples
            prod_responses = [r.response for r in cluster.representative_samples]

            # Pad to amplification target if needed (cycle through available responses)
            target = self._config.amplification_target_per_cluster
            if not prod_responses:
                prod_responses = ["(no response)"]
            while len(prod_responses) < target:
                prod_responses = (prod_responses * ((target // len(prod_responses)) + 1))
            prod_responses = prod_responses[:target]

            result = self._amplifier.amplify_cluster(
                cluster=cluster,
                production_model_responses=prod_responses,
                target_count=target,
            )
            amplification_results.append(result)

        # 4. Collect passing pairs across all clusters
        passing_pairs: List[PreferencePair] = [
            pair
            for ar in amplification_results
            for pair in ar.synthesized_pairs
            if pair.passed_gate
        ]

        # 5. Write datasets
        output_dir = os.path.join(self._config.output_base_dir, job_id)
        os.makedirs(output_dir, exist_ok=True)

        sft_path = os.path.join(output_dir, "sft_data.jsonl")
        dpo_path = os.path.join(output_dir, "dpo_data.jsonl")

        self._write_sft_jsonl(sft_path, passing_pairs)
        self._write_dpo_jsonl(dpo_path, passing_pairs)

        # 6. Aggregate statistics
        total_synthesized = sum(
            len(ar.synthesized_pairs) for ar in amplification_results
        )
        total_passing = sum(ar.pairs_passing_gate for ar in amplification_results)
        total_input = len(failure_records)
        overall_factor = (
            float(total_passing) / float(total_input) if total_input > 0 else 0.0
        )
        total_cost = sum(ar.synthesis_cost_estimate_usd for ar in amplification_results)
        duration = time.monotonic() - start_time

        return SynthesisJobResult(
            job_id=job_id,
            trigger_event_id=trigger_event_id,
            clusters_processed=len(clusters),
            total_pairs_synthesized=total_synthesized,
            total_pairs_passing_gate=total_passing,
            overall_amplification_factor=overall_factor,
            sft_dataset_path=sft_path,
            dpo_dataset_path=dpo_path,
            total_cost_usd=total_cost,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_sft_jsonl(path: str, pairs: List[PreferencePair]) -> None:
        """
        Write SFT dataset: one JSON object per line, schema {prompt, response}.

        The "response" field is the chosen (oracle) response — SFT trains the
        model to produce the correct output given the prompt.

        Parameters
        ----------
        path:   absolute path to the output file.
        pairs:  list of passing PreferencePairs.
        """
        with open(path, "w", encoding="utf-8") as fh:
            for pair in pairs:
                record = {
                    "prompt": pair.prompt,
                    "response": pair.chosen,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_dpo_jsonl(path: str, pairs: List[PreferencePair]) -> None:
        """
        Write DPO dataset: one JSON object per line, schema {prompt, chosen, rejected}.

        Parameters
        ----------
        path:   absolute path to the output file.
        pairs:  list of passing PreferencePairs.
        """
        with open(path, "w", encoding="utf-8") as fh:
            for pair in pairs:
                record = {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
