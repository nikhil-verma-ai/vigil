"""
Unit tests for Module 3b: LLM-as-Judge quality gate and synthesis pipeline.

All tests are fully self-contained:
  - No real API keys required (MockJudgeClient / MockOracleClient).
  - No sentence-transformer model download required (MockEmbeddingEngine).
  - No HDBSCAN clustering required for judge/amplifier unit tests.

Test coverage
-------------
test_judge_passes_clear_preference      — passing pair scores -> gate passes
test_judge_rejects_ambiguous_pair       — low clarity score -> gate fails
test_judge_evaluate_batch_mutates_pairs — batch eval updates judge_scores + passed_gate
test_judge_parse_robust_json            — markdown-fenced and padded JSON parses
test_amplifier_generates_target_count   — amplify_cluster produces target_count pairs
test_amplifier_all_pairs_judged         — every pair has judge_scores set
test_amplification_factor_calculation   — factor = passing / input_count
test_cost_estimate_scales_linearly      — cost is linear in n_pairs
test_pipeline_writes_valid_sft_jsonl    — SFT JSONL schema {prompt, response}
test_pipeline_writes_valid_dpo_jsonl    — DPO JSONL schema {prompt, chosen, rejected}
test_pipeline_empty_clusters_no_crash   — 0 records -> 0 clusters, empty files
test_mock_judge_client_call_count       — MockJudgeClient increments call_count
test_mock_oracle_client_call_count      — MockOracleClient increments call_count
test_judge_alternating_50pct_pass       — MockJudgeClientAlternating gives ~50% pass
"""
from __future__ import annotations

import json
import os
import sys
import uuid

# Ensure the project root is on sys.path so imports resolve correctly when
# invoked from the repo root via `pytest tests/unit/`.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from services.synthesizer.amplifier import (
    AmplificationResult,
    MockOracleClient,
    SyntheticAmplifier,
)
from services.synthesizer.clustering import FailureCluster, FailureClusterer
from services.synthesizer.config import SynthesizerConfig
from services.synthesizer.embedding import (
    EmbeddedFailure,
    EmbeddingEngine,
    FailureRecord,
)
from services.synthesizer.judge import (
    JudgeScores,
    LLMJudge,
    MockJudgeClient,
    MockJudgeClientAlternating,
    PreferencePair,
)
from services.synthesizer.pipeline import SynthesisPipeline, SynthesisJobResult


# --------------------------------------------------------------------------- #
# Shared fixtures and helpers                                                  #
# --------------------------------------------------------------------------- #

EMBEDDING_DIM = 384


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


def _make_preference_pair(
    pair_id: str = None,
    prompt: str = "What is 2+2?",
    chosen: str = "2+2=4. Adding 2 to 2 gives 4.",
    rejected: str = "2+2=5",
    cluster_id: int = 0,
) -> PreferencePair:
    return PreferencePair(
        pair_id=pair_id or str(uuid.uuid4()),
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        source_cluster_id=cluster_id,
        generation_timestamp="2026-03-28T00:00:00Z",
        oracle_model="mock",
    )


def _make_cluster(
    cluster_id: int = 0,
    member_count: int = 20,
    archetype: str = "arithmetic calculation failures",
    n_reps: int = 3,
) -> FailureCluster:
    reps = [
        _make_failure_record(
            prompt=f"arithmetic prompt {i}",
            response=f"wrong answer {i}",
        )
        for i in range(n_reps)
    ]
    return FailureCluster(
        cluster_id=cluster_id,
        member_count=member_count,
        persistence_score=0.5,
        centroid_embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
        representative_samples=reps,
        archetype_description=archetype,
    )


class MockEmbeddingEngine:
    """
    Test double for EmbeddingEngine.

    Returns deterministic pseudo-random embeddings that create two tight
    clusters when the prompts contain "arithmetic" vs "grammar".
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    def embed_batch(self, failures: List[FailureRecord]) -> List[EmbeddedFailure]:
        rng = np.random.default_rng(seed=42)
        results = []
        for i, f in enumerate(failures):
            # Assign cluster A or B based on index parity for test isolation
            base = np.zeros(self._dim, dtype=np.float32)
            if i % 2 == 0:
                base[0] = 1.0  # cluster A centroid direction
            else:
                base[1] = 1.0  # cluster B centroid direction
            noise = rng.normal(0, 0.01, self._dim).astype(np.float32)
            emb = base + noise
            # L2 normalise
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

    def get_embedding_matrix(self, embedded: List[EmbeddedFailure]) -> np.ndarray:
        if not embedded:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([e.embedding for e in embedded], axis=0)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


def make_failures_with_two_clusters(n_per_cluster: int = 15) -> List[FailureRecord]:
    """
    Produce 2*n_per_cluster failure records with two distinct semantic groups.

    Group A: arithmetic failures (prompt: "arithmetic ...")
    Group B: grammar failures (prompt: "grammar ...")
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


def _build_test_pipeline(
    mock_judge_scores: dict = None,
    n_per_cluster: int = 15,
    config: SynthesizerConfig = None,
) -> tuple:
    """
    Build a SynthesisPipeline with all mocked dependencies.

    Returns (pipeline, mock_oracle, mock_judge_client).
    """
    cfg = config or SynthesizerConfig()
    # Lower thresholds for tests to ensure clustering works with mock embeddings
    cfg.hdbscan_min_cluster_size = 5
    cfg.hdbscan_min_samples = 2
    cfg.min_persistence_score = 0.0
    cfg.amplification_target_per_cluster = 5  # keep tests fast

    mock_embedder = MockEmbeddingEngine()
    clusterer = FailureClusterer(config=cfg, embedding_engine=mock_embedder)

    oracle = MockOracleClient()
    judge_client = MockJudgeClient(scores=mock_judge_scores)
    judge = LLMJudge(client=judge_client, pass_threshold=0.7)
    amplifier = SyntheticAmplifier(oracle_client=oracle, judge=judge, config=cfg)

    pipeline = SynthesisPipeline(
        embedding_engine=mock_embedder,
        clusterer=clusterer,
        amplifier=amplifier,
        config=cfg,
    )
    return pipeline, oracle, judge_client


# --------------------------------------------------------------------------- #
# LLMJudge tests                                                               #
# --------------------------------------------------------------------------- #

class TestLLMJudge:

    def test_judge_passes_clear_preference(self):
        """A pair where chosen is clearly better must pass the quality gate."""
        mock_client = MockJudgeClient(scores={
            "preference_clarity": 0.9,
            "prompt_validity": 0.95,
            "response_quality": 0.85,
            "amplification_fidelity": 0.88,
        })
        judge = LLMJudge(client=mock_client, pass_threshold=0.7)
        pair = _make_preference_pair(
            prompt="What is 2+2?",
            chosen="2+2=4. Adding 2 to 2 gives 4.",
            rejected="2+2=5",
        )

        scores = judge.evaluate_pair(pair, "arithmetic failures")

        assert scores.overall_pass is True
        assert scores.preference_clarity >= 0.7
        assert scores.prompt_validity >= 0.7
        assert scores.response_quality >= 0.7
        assert scores.amplification_fidelity >= 0.7
        assert mock_client.call_count == 1

    def test_judge_rejects_ambiguous_pair(self):
        """A pair where preference_clarity < threshold must fail the gate."""
        mock_client = MockJudgeClient(scores={
            "preference_clarity": 0.3,
            "prompt_validity": 0.9,
            "response_quality": 0.8,
            "amplification_fidelity": 0.85,
        })
        judge = LLMJudge(client=mock_client, pass_threshold=0.7)
        pair = _make_preference_pair(
            prompt="Describe quantum entanglement.",
            chosen="answer A with nuance",
            rejected="answer A but rephrased",
        )

        scores = judge.evaluate_pair(pair, "test")

        assert scores.overall_pass is False
        # Confirm the low dimension caused the failure
        assert scores.preference_clarity == pytest.approx(0.3)

    def test_judge_rejects_when_single_dimension_below_threshold(self):
        """Even a single sub-threshold dimension must fail the gate."""
        # response_quality = 0.5 < 0.7 threshold
        mock_client = MockJudgeClient(scores={
            "preference_clarity": 0.9,
            "prompt_validity": 0.9,
            "response_quality": 0.5,
            "amplification_fidelity": 0.9,
        })
        judge = LLMJudge(client=mock_client, pass_threshold=0.7)
        pair = _make_preference_pair()

        scores = judge.evaluate_pair(pair, "test")

        assert scores.overall_pass is False
        assert scores.response_quality == pytest.approx(0.5)

    def test_judge_evaluate_batch_mutates_pairs(self):
        """evaluate_batch must set judge_scores and passed_gate on every pair."""
        mock_client = MockJudgeClient()  # default all-passing
        judge = LLMJudge(client=mock_client, pass_threshold=0.7)
        pairs = [_make_preference_pair(pair_id=str(i)) for i in range(5)]

        result = judge.evaluate_batch(pairs, "arithmetic failures")

        assert len(result) == 5
        assert mock_client.call_count == 5
        for pair in result:
            assert pair.judge_scores is not None
            assert pair.judge_scores.overall_pass is True
            assert pair.passed_gate is True

    def test_judge_evaluate_batch_returns_all_pairs(self):
        """evaluate_batch must return ALL pairs, not just passing ones."""
        mock_client = MockJudgeClient(scores={
            "preference_clarity": 0.2,  # below threshold
            "prompt_validity": 0.9,
            "response_quality": 0.9,
            "amplification_fidelity": 0.9,
        })
        judge = LLMJudge(client=mock_client, pass_threshold=0.7)
        pairs = [_make_preference_pair() for _ in range(3)]

        result = judge.evaluate_batch(pairs, "test")

        # All 3 pairs returned even though all fail
        assert len(result) == 3
        for pair in result:
            assert pair.passed_gate is False

    def test_mock_judge_client_call_count(self):
        """MockJudgeClient.call_count increments on each call."""
        client = MockJudgeClient()
        assert client.call_count == 0

        judge = LLMJudge(client=client)
        pair = _make_preference_pair()
        judge.evaluate_pair(pair, "test")
        assert client.call_count == 1

        judge.evaluate_pair(pair, "test")
        assert client.call_count == 2

    def test_judge_parse_robust_markdown_fenced_json(self):
        """LLMJudge must parse JSON wrapped in markdown code fences."""
        fenced_response = """```json
{"preference_clarity": 0.8, "prompt_validity": 0.9, "response_quality": 0.85, "amplification_fidelity": 0.88}
```"""

        class FencedMockClient:
            def __init__(self):
                self.call_count = 0

            def chat_completions_create(self, **kwargs) -> str:
                self.call_count += 1
                return fenced_response

        judge = LLMJudge(client=FencedMockClient(), pass_threshold=0.7)
        pair = _make_preference_pair()
        scores = judge.evaluate_pair(pair, "test")

        assert scores.preference_clarity == pytest.approx(0.8)
        assert scores.overall_pass is True

    def test_judge_parse_invalid_json_raises(self):
        """LLMJudge must raise ValueError for completely unparseable responses."""
        class BrokenMockClient:
            def __init__(self):
                self.call_count = 0

            def chat_completions_create(self, **kwargs) -> str:
                self.call_count += 1
                return "I cannot score this pair."

        judge = LLMJudge(client=BrokenMockClient())
        pair = _make_preference_pair()

        with pytest.raises(ValueError, match="unparseable"):
            judge.evaluate_pair(pair, "test")


# --------------------------------------------------------------------------- #
# SyntheticAmplifier tests                                                     #
# --------------------------------------------------------------------------- #

class TestSyntheticAmplifier:

    def _make_amplifier(
        self,
        judge_scores: dict = None,
        config: SynthesizerConfig = None,
    ) -> tuple:
        cfg = config or SynthesizerConfig()
        oracle = MockOracleClient()
        judge_client = MockJudgeClient(scores=judge_scores)
        judge = LLMJudge(client=judge_client, pass_threshold=0.7)
        amplifier = SyntheticAmplifier(
            oracle_client=oracle, judge=judge, config=cfg
        )
        return amplifier, oracle, judge_client

    def test_amplifier_generates_target_count(self):
        """amplify_cluster must produce exactly target_count pairs before filtering."""
        amplifier, oracle, judge_client = self._make_amplifier()
        cluster = _make_cluster(member_count=20)
        production_responses = ["wrong answer"] * 20

        result = amplifier.amplify_cluster(
            cluster=cluster,
            production_model_responses=production_responses,
            target_count=20,
        )

        assert result.input_failure_count == 20
        assert len(result.synthesized_pairs) == 20
        # MockJudgeClient passes all by default (scores >= 0.85 > 0.7 threshold)
        assert result.pairs_passing_gate == 20

    def test_amplifier_all_pairs_judged(self):
        """Every returned pair must have judge_scores set."""
        amplifier, _, _ = self._make_amplifier()
        cluster = _make_cluster()
        result = amplifier.amplify_cluster(
            cluster=cluster,
            production_model_responses=["wrong"] * 20,
            target_count=5,
        )

        for pair in result.synthesized_pairs:
            assert pair.judge_scores is not None
            assert isinstance(pair.judge_scores, JudgeScores)

    def test_amplifier_sets_source_cluster_id(self):
        """All synthesized pairs must reference the correct source cluster."""
        amplifier, _, _ = self._make_amplifier()
        cluster = _make_cluster(cluster_id=7)
        result = amplifier.amplify_cluster(
            cluster=cluster,
            production_model_responses=["x"] * 10,
            target_count=5,
        )

        for pair in result.synthesized_pairs:
            assert pair.source_cluster_id == 7

    def test_amplification_factor_all_pass(self):
        """amplification_factor = pairs_passing_gate / input_failure_count."""
        amplifier, _, _ = self._make_amplifier()  # all pass
        cluster = _make_cluster(member_count=20)
        result = amplifier.amplify_cluster(
            cluster=cluster,
            production_model_responses=["wrong"] * 20,
            target_count=20,
        )

        assert result.pairs_passing_gate == 20
        assert result.amplification_factor == pytest.approx(20.0 / 20.0)

    def test_amplification_factor_half_pass(self):
        """amplification_factor reflects 50% pass rate via alternating judge."""
        cfg = SynthesizerConfig()
        oracle = MockOracleClient()
        judge_client = MockJudgeClientAlternating()
        judge = LLMJudge(client=judge_client, pass_threshold=0.7)
        amplifier = SyntheticAmplifier(
            oracle_client=oracle, judge=judge, config=cfg
        )
        cluster = _make_cluster(member_count=20)

        result = amplifier.amplify_cluster(
            cluster=cluster,
            production_model_responses=["wrong"] * 20,
            target_count=20,
        )

        # Alternating: calls 1,3,5,...19 pass (odd) -> 10 out of 20
        assert result.pairs_passing_gate == 10
        assert result.amplification_factor == pytest.approx(10.0 / 20.0)

    def test_cost_estimate_scales_linearly(self):
        """estimate_cost must be linear in n_pairs."""
        amplifier, _, _ = self._make_amplifier()

        cost_10 = amplifier.estimate_cost(10)
        cost_100 = amplifier.estimate_cost(100)

        assert cost_10 > 0.0
        assert abs(cost_100 / cost_10 - 10.0) < 0.01

    def test_cost_estimate_zero_for_zero_pairs(self):
        """estimate_cost(0) must return 0.0."""
        amplifier, _, _ = self._make_amplifier()
        assert amplifier.estimate_cost(0) == pytest.approx(0.0)

    def test_mock_oracle_client_call_count(self):
        """MockOracleClient.call_count increments on each generate() call."""
        oracle = MockOracleClient()
        assert oracle.call_count == 0

        oracle.generate("test prompt")
        assert oracle.call_count == 1

        oracle.generate("another prompt")
        assert oracle.call_count == 2

    def test_mock_oracle_response_contains_prompt(self):
        """MockOracleClient response must embed a prefix of the prompt."""
        oracle = MockOracleClient()
        prompt = "What is the speed of light?"
        response = oracle.generate(prompt)

        # Default template: "This is a correct response to: {prompt[:50]}"
        assert "This is a correct response to:" in response

    def test_amplifier_pairs_have_distinct_chosen_rejected(self):
        """chosen and rejected must differ for every pair."""
        amplifier, _, _ = self._make_amplifier()
        cluster = _make_cluster()
        result = amplifier.amplify_cluster(
            cluster=cluster,
            production_model_responses=["wrong answer"] * 20,
            target_count=10,
        )

        for pair in result.synthesized_pairs:
            assert pair.chosen != pair.rejected, (
                f"pair {pair.pair_id}: chosen == rejected"
            )

    def test_generate_variant_prompts_returns_list(self):
        """generate_variant_prompts must return a list of strings."""
        amplifier, _, _ = self._make_amplifier()
        cluster = _make_cluster(archetype="arithmetic failures")

        prompts = amplifier.generate_variant_prompts(cluster, count=5)

        assert isinstance(prompts, list)
        assert len(prompts) > 0
        for p in prompts:
            assert isinstance(p, str)
            assert len(p) > 0


# --------------------------------------------------------------------------- #
# SynthesisPipeline tests                                                      #
# --------------------------------------------------------------------------- #

class TestSynthesisPipeline:

    def test_pipeline_writes_valid_sft_jsonl(self, tmp_path):
        """Pipeline SFT output must be valid JSONL with {prompt, response} schema."""
        cfg = SynthesizerConfig()
        cfg.output_base_dir = str(tmp_path)
        cfg.hdbscan_min_cluster_size = 5
        cfg.hdbscan_min_samples = 2
        cfg.min_persistence_score = 0.0
        cfg.amplification_target_per_cluster = 5

        pipeline, _, _ = _build_test_pipeline(config=cfg)
        failures = make_failures_with_two_clusters(n_per_cluster=10)

        result = pipeline.run(failures, job_id="test-sft", trigger_event_id="evt-1")

        assert os.path.exists(result.sft_dataset_path)
        with open(result.sft_dataset_path) as f:
            lines = f.readlines()

        # If clusters were found, there must be at least one SFT record
        if result.clusters_processed > 0:
            assert len(lines) > 0

        for line in lines:
            record = json.loads(line)
            assert "prompt" in record, f"SFT record missing 'prompt': {record}"
            assert "response" in record, f"SFT record missing 'response': {record}"
            assert len(record["prompt"]) > 0
            assert len(record["response"]) > 0

    def test_pipeline_writes_valid_dpo_jsonl(self, tmp_path):
        """Pipeline DPO output must be valid JSONL with {prompt, chosen, rejected} schema."""
        cfg = SynthesizerConfig()
        cfg.output_base_dir = str(tmp_path)
        cfg.hdbscan_min_cluster_size = 5
        cfg.hdbscan_min_samples = 2
        cfg.min_persistence_score = 0.0
        cfg.amplification_target_per_cluster = 5

        pipeline, _, _ = _build_test_pipeline(config=cfg)
        failures = make_failures_with_two_clusters(n_per_cluster=10)

        result = pipeline.run(failures, job_id="test-dpo", trigger_event_id="evt-2")

        assert os.path.exists(result.dpo_dataset_path)
        with open(result.dpo_dataset_path) as f:
            lines = f.readlines()

        if result.clusters_processed > 0:
            assert len(lines) > 0

        for line in lines:
            record = json.loads(line)
            assert "prompt" in record
            assert "chosen" in record
            assert "rejected" in record
            assert record["chosen"] != record["rejected"], (
                "DPO record has identical chosen and rejected"
            )

    def test_pipeline_empty_clusters_no_crash(self, tmp_path):
        """
        0 failure records -> 0 clusters -> empty datasets -> no crash.
        Files must still be created (even if empty).
        """
        cfg = SynthesizerConfig()
        cfg.output_base_dir = str(tmp_path)

        pipeline, _, _ = _build_test_pipeline(config=cfg)

        result = pipeline.run([], job_id="test-empty", trigger_event_id="evt-3")

        assert result.clusters_processed == 0
        assert result.total_pairs_synthesized == 0
        assert result.total_pairs_passing_gate == 0
        assert result.overall_amplification_factor == pytest.approx(0.0)

        # Files must be created even for empty results
        assert os.path.exists(result.sft_dataset_path)
        assert os.path.exists(result.dpo_dataset_path)

        with open(result.sft_dataset_path) as f:
            assert f.read() == ""
        with open(result.dpo_dataset_path) as f:
            assert f.read() == ""

    def test_pipeline_below_min_cluster_size_no_clusters(self, tmp_path):
        """
        Fewer records than hdbscan_min_cluster_size must yield 0 clusters.
        """
        cfg = SynthesizerConfig()
        cfg.output_base_dir = str(tmp_path)
        cfg.hdbscan_min_cluster_size = 10  # canonical default

        pipeline, _, _ = _build_test_pipeline(config=cfg)
        # Only 3 records — below min_cluster_size=10
        failures = [_make_failure_record() for _ in range(3)]

        result = pipeline.run(failures, job_id="test-no-clusters", trigger_event_id="evt-4")

        assert result.clusters_processed == 0
        assert result.total_pairs_synthesized == 0

    def test_pipeline_result_has_correct_job_metadata(self, tmp_path):
        """SynthesisJobResult must carry the provided job_id and trigger_event_id."""
        cfg = SynthesizerConfig()
        cfg.output_base_dir = str(tmp_path)

        pipeline, _, _ = _build_test_pipeline(config=cfg)

        result = pipeline.run(
            [], job_id="job-xyz-123", trigger_event_id="drift-event-abc"
        )

        assert result.job_id == "job-xyz-123"
        assert result.trigger_event_id == "drift-event-abc"

    def test_pipeline_duration_positive(self, tmp_path):
        """duration_seconds must be a positive float."""
        cfg = SynthesizerConfig()
        cfg.output_base_dir = str(tmp_path)
        pipeline, _, _ = _build_test_pipeline(config=cfg)

        result = pipeline.run([], job_id="dur-test", trigger_event_id="e")

        assert result.duration_seconds > 0.0

    def test_pipeline_cost_zero_for_no_clusters(self, tmp_path):
        """total_cost_usd must be 0.0 when no clusters are amplified."""
        cfg = SynthesizerConfig()
        cfg.output_base_dir = str(tmp_path)
        pipeline, _, _ = _build_test_pipeline(config=cfg)

        result = pipeline.run([], job_id="cost-test", trigger_event_id="e")

        assert result.total_cost_usd == pytest.approx(0.0)

    def test_pipeline_dataset_paths_under_job_dir(self, tmp_path):
        """Dataset paths must live under {output_base_dir}/{job_id}/."""
        cfg = SynthesizerConfig()
        cfg.output_base_dir = str(tmp_path)
        pipeline, _, _ = _build_test_pipeline(config=cfg)

        result = pipeline.run([], job_id="path-test", trigger_event_id="e")

        expected_dir = os.path.join(str(tmp_path), "path-test")
        assert result.sft_dataset_path.startswith(expected_dir)
        assert result.dpo_dataset_path.startswith(expected_dir)
        assert result.sft_dataset_path.endswith("sft_data.jsonl")
        assert result.dpo_dataset_path.endswith("dpo_data.jsonl")


# --------------------------------------------------------------------------- #
# MockJudgeClientAlternating tests                                             #
# --------------------------------------------------------------------------- #

class TestMockJudgeClientAlternating:

    def test_alternating_50pct_pass_rate(self):
        """MockJudgeClientAlternating must yield ~50% pass rate over even N pairs."""
        client = MockJudgeClientAlternating()
        judge = LLMJudge(client=client, pass_threshold=0.7)

        n = 20
        pairs = [_make_preference_pair() for _ in range(n)]
        judged = judge.evaluate_batch(pairs, "test")

        passing = sum(1 for p in judged if p.passed_gate)
        failing = sum(1 for p in judged if not p.passed_gate)

        assert passing == 10  # calls 1,3,5,...19 pass
        assert failing == 10
        assert client.call_count == n
