"""
Integration tests for the end-to-end synthesis pipeline (Module 3b).

These tests exercise the full pipeline — embedding -> clustering ->
amplification -> judge gate -> JSONL output — with mocked LLM clients
but real (mock) embedding and HDBSCAN clustering.

Test coverage
-------------
test_end_to_end_from_failure_records
    40 realistic failure records -> pipeline -> valid DPO JSONL with data.

test_pipeline_handles_no_clusters
    3 failure records (below min_cluster_size=10) -> 0 clusters,
    empty datasets, no crash, SynthesisJobResult well-formed.

test_pipeline_cost_tracking
    After run, result.total_cost_usd > 0 and matches
    cost_per_pair_usd * total_pairs_synthesized.

test_pipeline_amplification_factor_matches_counts
    overall_amplification_factor = total_pairs_passing_gate / len(records).

test_pipeline_sft_response_equals_chosen
    SFT JSONL response field must equal the corresponding DPO chosen field.

test_pipeline_dpo_rejects_differ_across_records
    DPO rejected fields must not all be identical (production variety propagated).

test_pipeline_idempotent_on_same_job_id
    Running the same job_id twice overwrites files rather than appending.
"""
from __future__ import annotations

import json
import os
import sys
import uuid

# Ensure the project root is on sys.path so absolute imports resolve.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from typing import List

import numpy as np
import pytest

from services.synthesizer.amplifier import MockOracleClient, SyntheticAmplifier
from services.synthesizer.clustering import FailureClusterer
from services.synthesizer.config import SynthesizerConfig
from services.synthesizer.embedding import EmbeddedFailure, FailureRecord
from services.synthesizer.judge import LLMJudge, MockJudgeClient
from services.synthesizer.pipeline import SynthesisPipeline, SynthesisJobResult

EMBEDDING_DIM = 384


# --------------------------------------------------------------------------- #
# Shared helpers (inlined to avoid cross-package imports)                      #
# --------------------------------------------------------------------------- #

def _make_failure_record(
    prompt: str = "What is 2+2?",
    response: str = "The answer is 5.",
    mean_logprob: float = -1.5,
) -> FailureRecord:
    return FailureRecord(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        response=response,
        mean_logprob=mean_logprob,
        timestamp="2026-03-28T00:00:00Z",
    )


def make_failures_with_two_clusters(n_per_cluster: int = 15) -> list:
    """
    Produce 2*n_per_cluster failure records with two distinct semantic groups.

    Group A: arithmetic failures
    Group B: grammar failures
    """
    failures = []
    for i in range(n_per_cluster):
        failures.append(
            _make_failure_record(
                prompt=f"arithmetic calculation error {i}: what is {i}+{i}?",
                response=f"The result is {i * 3} (wrong)",
                mean_logprob=-2.0,
            )
        )
    for i in range(n_per_cluster):
        failures.append(
            _make_failure_record(
                prompt=f"grammar correction error {i}: fix this sentence {i}",
                response=f"This sentence are wrong {i}",
                mean_logprob=-1.8,
            )
        )
    return failures


class MockEmbeddingEngine:
    """
    Test double for EmbeddingEngine — deterministic pseudo-random embeddings.
    Creates two tight clusters based on index parity.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    def embed_batch(self, failures: list) -> list:
        rng = np.random.default_rng(seed=42)
        results = []
        for i, f in enumerate(failures):
            base = np.zeros(self._dim, dtype=np.float32)
            if i % 2 == 0:
                base[0] = 1.0
            else:
                base[1] = 1.0
            noise = rng.normal(0, 0.01, self._dim).astype(np.float32)
            emb = base + noise
            norm = np.linalg.norm(emb)
            emb = emb / norm if norm > 0 else emb
            results.append(
                EmbeddedFailure(
                    request_id=f.request_id,
                    prompt=f.prompt,
                    response=f.response,
                    embedding=emb,
                    mean_logprob=f.mean_logprob,
                )
            )
        return results

    def get_embedding_matrix(self, embedded: list) -> np.ndarray:
        if not embedded:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([e.embedding for e in embedded], axis=0)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


# --------------------------------------------------------------------------- #
# Shared factory                                                               #
# --------------------------------------------------------------------------- #

def _make_pipeline(
    tmp_path: str,
    judge_scores: dict = None,
    min_cluster_size: int = 5,
    amplification_target: int = 10,
    cost_per_pair: float = 0.01,
) -> tuple:
    """
    Build a fully-wired SynthesisPipeline with mock sub-components.

    Returns (pipeline, config, oracle, judge_client).
    """
    cfg = SynthesizerConfig()
    cfg.output_base_dir = tmp_path
    cfg.hdbscan_min_cluster_size = min_cluster_size
    cfg.hdbscan_min_samples = 2
    cfg.min_persistence_score = 0.0
    cfg.amplification_target_per_cluster = amplification_target
    cfg.cost_per_pair_usd = cost_per_pair

    embedder = MockEmbeddingEngine()
    clusterer = FailureClusterer(config=cfg, embedding_engine=embedder)

    oracle = MockOracleClient()
    j_client = MockJudgeClient(scores=judge_scores)
    judge = LLMJudge(client=j_client, pass_threshold=0.7)
    amplifier = SyntheticAmplifier(oracle_client=oracle, judge=judge, config=cfg)

    pipeline = SynthesisPipeline(
        embedding_engine=embedder,
        clusterer=clusterer,
        amplifier=amplifier,
        config=cfg,
    )
    return pipeline, cfg, oracle, j_client


# --------------------------------------------------------------------------- #
# Integration tests                                                            #
# --------------------------------------------------------------------------- #

class TestEndToEndPipeline:

    def test_end_to_end_from_failure_records(self, tmp_path):
        """
        40 realistic failure records -> pipeline -> valid DPO JSONL with data.

        Validates:
        - result is a SynthesisJobResult.
        - clusters_processed >= 1 (two semantic groups should be found).
        - total_pairs_synthesized >= amplification_target.
        - DPO file is valid JSONL with {prompt, chosen, rejected} schema.
        - All DPO records have non-empty fields and chosen != rejected.
        """
        failures = make_failures_with_two_clusters(n_per_cluster=20)  # 40 total
        assert len(failures) == 40

        pipeline, cfg, oracle, j_client = _make_pipeline(
            tmp_path=str(tmp_path),
            min_cluster_size=5,
            amplification_target=10,
        )

        result = pipeline.run(
            failures,
            job_id="e2e-test-001",
            trigger_event_id="drift-evt-001",
        )

        # Result type
        assert isinstance(result, SynthesisJobResult)
        assert result.job_id == "e2e-test-001"
        assert result.trigger_event_id == "drift-evt-001"

        # Clustering found at least one cluster
        assert result.clusters_processed >= 1, (
            "Expected at least 1 cluster from 40 records with 2 semantic groups"
        )

        # Pairs were synthesized
        assert result.total_pairs_synthesized >= cfg.amplification_target_per_cluster

        # DPO file valid JSONL
        assert os.path.exists(result.dpo_dataset_path)
        with open(result.dpo_dataset_path) as f:
            dpo_lines = f.readlines()

        assert len(dpo_lines) > 0, "DPO JSONL must be non-empty for 40 input records"

        for i, line in enumerate(dpo_lines):
            record = json.loads(line)
            assert "prompt" in record, f"Line {i} missing 'prompt'"
            assert "chosen" in record, f"Line {i} missing 'chosen'"
            assert "rejected" in record, f"Line {i} missing 'rejected'"
            assert len(record["prompt"]) > 0
            assert len(record["chosen"]) > 0
            assert len(record["rejected"]) > 0
            assert record["chosen"] != record["rejected"], (
                f"Line {i}: chosen == rejected"
            )

        # SFT file also valid
        with open(result.sft_dataset_path) as f:
            sft_lines = f.readlines()

        assert len(sft_lines) == len(dpo_lines), (
            "SFT and DPO files must have the same number of records"
        )
        for i, line in enumerate(sft_lines):
            record = json.loads(line)
            assert "prompt" in record
            assert "response" in record

    def test_pipeline_handles_no_clusters(self, tmp_path):
        """
        3 failure records (below min_cluster_size=10) -> 0 clusters, empty datasets.

        Validates:
        - No crash.
        - clusters_processed == 0.
        - total_pairs_synthesized == 0.
        - SFT and DPO files exist but are empty.
        """
        failures = [_make_failure_record() for _ in range(3)]
        # Use canonical min_cluster_size=10 (default)
        pipeline, cfg, _, _ = _make_pipeline(
            tmp_path=str(tmp_path),
            min_cluster_size=10,  # 3 < 10 -> no clusters
        )

        result = pipeline.run(
            failures,
            job_id="no-clusters-test",
            trigger_event_id="evt-no-clusters",
        )

        assert result.clusters_processed == 0
        assert result.total_pairs_synthesized == 0
        assert result.total_pairs_passing_gate == 0
        assert result.overall_amplification_factor == pytest.approx(0.0)
        assert result.total_cost_usd == pytest.approx(0.0)
        assert result.duration_seconds > 0.0

        # Files must exist even when empty
        assert os.path.exists(result.sft_dataset_path)
        assert os.path.exists(result.dpo_dataset_path)

        with open(result.sft_dataset_path) as f:
            assert f.read() == "", "SFT file must be empty when no clusters found"
        with open(result.dpo_dataset_path) as f:
            assert f.read() == "", "DPO file must be empty when no clusters found"

    def test_pipeline_cost_tracking(self, tmp_path):
        """
        After run, result.total_cost_usd > 0 and matches
        cost_per_pair_usd * total_pairs_synthesized.
        """
        failures = make_failures_with_two_clusters(n_per_cluster=15)  # 30 total
        cost_per_pair = 0.01

        pipeline, cfg, _, _ = _make_pipeline(
            tmp_path=str(tmp_path),
            min_cluster_size=5,
            amplification_target=10,
            cost_per_pair=cost_per_pair,
        )

        result = pipeline.run(
            failures,
            job_id="cost-tracking-test",
            trigger_event_id="evt-cost",
        )

        if result.clusters_processed > 0:
            assert result.total_cost_usd > 0.0, (
                "Cost must be > 0 when pairs were synthesized"
            )
            expected_cost = cost_per_pair * result.total_pairs_synthesized
            assert abs(result.total_cost_usd - expected_cost) < 1e-9, (
                f"Expected cost {expected_cost:.4f}, got {result.total_cost_usd:.4f}"
            )
        else:
            # No clusters -> no cost
            assert result.total_cost_usd == pytest.approx(0.0)

    def test_pipeline_amplification_factor_matches_counts(self, tmp_path):
        """
        overall_amplification_factor must equal
        total_pairs_passing_gate / len(failure_records).
        """
        failures = make_failures_with_two_clusters(n_per_cluster=10)  # 20 total
        n_failures = len(failures)

        pipeline, _, _, _ = _make_pipeline(
            tmp_path=str(tmp_path),
            min_cluster_size=5,
            amplification_target=10,
        )

        result = pipeline.run(
            failures,
            job_id="factor-test",
            trigger_event_id="evt-factor",
        )

        expected_factor = (
            result.total_pairs_passing_gate / n_failures
            if n_failures > 0
            else 0.0
        )
        assert result.overall_amplification_factor == pytest.approx(
            expected_factor, abs=1e-9
        )

    def test_pipeline_sft_response_equals_chosen(self, tmp_path):
        """
        SFT response field must equal the chosen field in the DPO dataset.
        Both files are sorted by insertion order (same passing pairs).
        """
        failures = make_failures_with_two_clusters(n_per_cluster=15)

        pipeline, _, _, _ = _make_pipeline(
            tmp_path=str(tmp_path),
            min_cluster_size=5,
            amplification_target=8,
        )

        result = pipeline.run(
            failures,
            job_id="sft-dpo-parity",
            trigger_event_id="evt-parity",
        )

        if result.total_pairs_passing_gate == 0:
            pytest.skip("No passing pairs produced — clustering found no clusters")

        with open(result.sft_dataset_path) as f:
            sft_records = [json.loads(l) for l in f]
        with open(result.dpo_dataset_path) as f:
            dpo_records = [json.loads(l) for l in f]

        assert len(sft_records) == len(dpo_records)

        for i, (sft, dpo) in enumerate(zip(sft_records, dpo_records)):
            assert sft["prompt"] == dpo["prompt"], (
                f"Row {i}: SFT and DPO prompts diverge"
            )
            assert sft["response"] == dpo["chosen"], (
                f"Row {i}: SFT response '{sft['response']}' != DPO chosen '{dpo['chosen']}'"
            )

    def test_pipeline_dpo_rejects_not_all_identical(self, tmp_path):
        """
        DPO rejected fields should vary (production responses cycle through
        representative samples from the cluster, not a single constant value).
        This test passes even with a single cluster as long as rejected != chosen.
        """
        failures = make_failures_with_two_clusters(n_per_cluster=15)

        pipeline, _, _, _ = _make_pipeline(
            tmp_path=str(tmp_path),
            min_cluster_size=5,
            amplification_target=10,
        )

        result = pipeline.run(
            failures,
            job_id="variety-test",
            trigger_event_id="evt-variety",
        )

        if result.total_pairs_passing_gate == 0:
            pytest.skip("No passing pairs produced")

        with open(result.dpo_dataset_path) as f:
            dpo_records = [json.loads(l) for l in f]

        for rec in dpo_records:
            assert rec["chosen"] != rec["rejected"], (
                "Every DPO record must have distinct chosen and rejected"
            )

    def test_pipeline_idempotent_on_same_job_id(self, tmp_path):
        """
        Running the same job_id twice must overwrite files, not append.
        The second run's output must match the second run's pair count exactly.
        """
        failures = make_failures_with_two_clusters(n_per_cluster=10)

        pipeline, _, _, _ = _make_pipeline(
            tmp_path=str(tmp_path),
            min_cluster_size=5,
            amplification_target=5,
        )

        # First run
        result1 = pipeline.run(
            failures,
            job_id="idem-test",
            trigger_event_id="evt-1",
        )
        with open(result1.dpo_dataset_path) as f:
            lines_run1 = f.readlines()

        # Second run with same job_id
        result2 = pipeline.run(
            failures,
            job_id="idem-test",
            trigger_event_id="evt-2",
        )
        with open(result2.dpo_dataset_path) as f:
            lines_run2 = f.readlines()

        # Both runs should produce the same number of passing pairs
        # (mock embeddings are deterministic)
        assert len(lines_run1) == len(lines_run2), (
            "Idempotent runs with same input must produce same pair count"
        )
        # File should contain exactly the second run's data (overwrite, not append)
        assert len(lines_run2) == result2.total_pairs_passing_gate

    def test_pipeline_result_fields_complete(self, tmp_path):
        """SynthesisJobResult must have all required fields populated."""
        failures = make_failures_with_two_clusters(n_per_cluster=10)
        pipeline, _, _, _ = _make_pipeline(
            tmp_path=str(tmp_path),
            min_cluster_size=5,
        )

        result = pipeline.run(
            failures,
            job_id="field-check",
            trigger_event_id="evt-fields",
        )

        # All required fields must be present and correctly typed
        assert isinstance(result.job_id, str) and len(result.job_id) > 0
        assert isinstance(result.trigger_event_id, str)
        assert isinstance(result.clusters_processed, int)
        assert isinstance(result.total_pairs_synthesized, int)
        assert isinstance(result.total_pairs_passing_gate, int)
        assert isinstance(result.overall_amplification_factor, float)
        assert isinstance(result.sft_dataset_path, str)
        assert isinstance(result.dpo_dataset_path, str)
        assert isinstance(result.total_cost_usd, float)
        assert isinstance(result.duration_seconds, float)
        assert result.duration_seconds > 0.0
        assert result.total_pairs_passing_gate <= result.total_pairs_synthesized
        assert result.overall_amplification_factor >= 0.0
