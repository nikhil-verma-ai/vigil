"""
SyntheticAmplifier: amplifies a FailureCluster into DPO preference pairs.

For each cluster the amplifier:
  1. Generates `target_count` semantically-similar prompt variants using
     the oracle model (batch call).
  2. For each variant: uses the oracle to generate the "chosen" (correct)
     response; the production model's response for that cluster becomes
     the "rejected" response.
  3. Runs every pair through LLMJudge and records pass/fail.
  4. Returns AmplificationResult with all pairs and gate statistics.

The 20x amplification target means a cluster of 20 failures becomes 20
synthetic pairs — the amplification_factor is pairs_passing_gate /
input_failure_count, not raw pair count / input count.

Design notes
------------
* Oracle client is injected — works with real OpenAI clients or MockOracleClient.
* All oracle calls are synchronous; async is deferred to the pipeline layer
  which can parallelize across clusters.
* Cost estimate is a rough heuristic: $0.01 per pair regardless of token count.
  Production deployments should replace this with a token-counting model.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from services.synthesizer.clustering import FailureCluster
from services.synthesizer.config import SynthesizerConfig
from services.synthesizer.judge import JudgeScores, LLMJudge, PreferencePair


# --------------------------------------------------------------------------- #
# Domain types                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class AmplificationResult:
    """
    Result of amplifying a single FailureCluster.

    Fields
    ------
    cluster_id:                HDBSCAN cluster label.
    input_failure_count:       number of raw failure records in the cluster.
    synthesized_pairs:         all generated PreferencePairs (judge-scored).
    pairs_passing_gate:        count of pairs where passed_gate == True.
    amplification_factor:      pairs_passing_gate / input_failure_count.
                               > 1.0 means we generated more quality pairs
                               than there were raw failures.
    synthesis_cost_estimate_usd: rough cost of oracle + judge calls.
    """
    cluster_id: int
    input_failure_count: int
    synthesized_pairs: List[PreferencePair]
    pairs_passing_gate: int
    amplification_factor: float
    synthesis_cost_estimate_usd: float


# --------------------------------------------------------------------------- #
# Mock oracle client (test double)                                             #
# --------------------------------------------------------------------------- #

class MockOracleClient:
    """
    Deterministic test double for oracle model calls.

    Parameters
    ----------
    response_template:
        Python format string accepting a single positional arg ``{prompt}``
        (first 50 chars of the prompt).  Defaults to a simple acknowledgement.

    Exposes the same interface as real oracle wrappers:
      .generate(prompt: str) -> str
    """

    def __init__(
        self,
        response_template: str = "This is a correct response to: {prompt}",
    ) -> None:
        self.template = response_template
        self.call_count: int = 0

    def generate(self, prompt: str) -> str:
        """
        Return a deterministic response string.

        Parameters
        ----------
        prompt: user prompt (first 50 chars substituted into template).

        Returns
        -------
        Formatted response string.
        """
        self.call_count += 1
        return self.template.format(prompt=prompt[:50])

    def chat_completions_create(self, **kwargs) -> str:
        """
        OpenAI-compatible shim used by generate_variant_prompts.

        Parses the first user message to extract the count and returns a
        JSON list of that many synthetic prompts.
        """
        self.call_count += 1
        # Extract count from messages if possible; default to 10
        count = 10
        try:
            messages = kwargs.get("messages", [])
            for msg in messages:
                content = msg.get("content", "")
                m = re.search(r"Generate (\d+)", content)
                if m:
                    count = int(m.group(1))
                    break
        except Exception:
            pass

        prompts = [
            f"Synthetic variant prompt {i + 1}" for i in range(count)
        ]
        return json.dumps(prompts)


# --------------------------------------------------------------------------- #
# SyntheticAmplifier                                                           #
# --------------------------------------------------------------------------- #

class SyntheticAmplifier:
    """
    Amplifies a FailureCluster into judged DPO preference pairs.

    Parameters
    ----------
    oracle_client:
        Object with .generate(prompt: str) -> str and optionally
        .chat_completions_create(**kwargs) -> str for variant generation.
        Accepts MockOracleClient for tests.
    judge:
        LLMJudge instance.
    config:
        SynthesizerConfig.  Defaults to SynthesizerConfig().
    """

    VARIANT_PROMPT_TEMPLATE = (
        "Generate {count} diverse prompts that test the same capability as: "
        "{archetype}\n\n"
        "Each prompt should be a realistic user question that would expose the "
        "same type of model failure. Return a JSON array of strings only, no "
        "explanation."
    )

    def __init__(
        self,
        oracle_client,
        judge: LLMJudge,
        config: SynthesizerConfig = None,
    ) -> None:
        self._oracle = oracle_client
        self._judge = judge
        self._config = config or SynthesizerConfig()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def amplify_cluster(
        self,
        cluster: FailureCluster,
        production_model_responses: List[str],
        target_count: int = 20,
    ) -> AmplificationResult:
        """
        Amplify a failure cluster into DPO preference pairs.

        Algorithm
        ---------
        1. Generate target_count variant prompts from the cluster archetype.
        2. For each variant prompt:
           a. Get oracle (chosen) response.
           b. Assign a production model response as the rejected response
              (cycling through production_model_responses).
        3. Run LLMJudge on all pairs.
        4. Compute gate statistics.

        Parameters
        ----------
        cluster:
            FailureCluster with archetype_description set.
        production_model_responses:
            List of bad model responses from the cluster.  Used as `rejected`
            in the preference pairs.  Cycled if fewer than target_count.
        target_count:
            Number of pairs to synthesize.  Defaults to 20.

        Returns
        -------
        AmplificationResult with all synthesized pairs (judge-scored).

        Invariants
        ----------
        - len(result.synthesized_pairs) == target_count (before filtering).
        - result.pairs_passing_gate <= target_count.
        - result.amplification_factor = pairs_passing_gate / input_failure_count.
        """
        input_count = len(production_model_responses)

        # Step 1: generate variant prompts
        variant_prompts = self.generate_variant_prompts(cluster, target_count)

        # Ensure we have exactly target_count prompts (pad with archetype if needed)
        while len(variant_prompts) < target_count:
            variant_prompts.append(
                cluster.archetype_description
                or (cluster.representative_samples[0].prompt
                    if cluster.representative_samples else f"prompt_{len(variant_prompts)}")
            )
        variant_prompts = variant_prompts[:target_count]

        # Step 2: generate oracle (chosen) responses + build PreferencePairs
        now_iso = datetime.now(timezone.utc).isoformat()
        pairs: List[PreferencePair] = []

        for i, vp in enumerate(variant_prompts):
            chosen = self.get_oracle_response(vp)
            # Cycle through production responses for rejected
            rejected = production_model_responses[i % len(production_model_responses)]

            pairs.append(
                PreferencePair(
                    pair_id=str(uuid.uuid4()),
                    prompt=vp,
                    chosen=chosen,
                    rejected=rejected,
                    source_cluster_id=cluster.cluster_id,
                    generation_timestamp=now_iso,
                    oracle_model=self._config.oracle_model,
                )
            )

        # Step 3: judge all pairs
        cluster_desc = (
            cluster.archetype_description
            or f"cluster {cluster.cluster_id} failure mode"
        )
        judged_pairs = self._judge.evaluate_batch(pairs, cluster_desc)

        # Step 4: gate statistics
        n_passing = sum(1 for p in judged_pairs if p.passed_gate)
        factor = (
            float(n_passing) / float(input_count) if input_count > 0 else 0.0
        )
        cost = self.estimate_cost(len(judged_pairs))

        return AmplificationResult(
            cluster_id=cluster.cluster_id,
            input_failure_count=input_count,
            synthesized_pairs=judged_pairs,
            pairs_passing_gate=n_passing,
            amplification_factor=factor,
            synthesis_cost_estimate_usd=cost,
        )

    def generate_variant_prompts(
        self,
        cluster: FailureCluster,
        count: int,
    ) -> List[str]:
        """
        Use oracle model to generate `count` semantically similar prompts.

        Parameters
        ----------
        cluster:
            Source cluster; archetype_description is used as the reference
            capability description.  Falls back to the first representative
            sample's prompt if archetype_description is empty.
        count:
            Number of variant prompts to request.

        Returns
        -------
        List of prompt strings.  Length may differ from count if the oracle
        returns malformed JSON; callers must handle this.

        Raises
        ------
        Does not raise — returns best-effort list (may be shorter than count).
        """
        archetype = cluster.archetype_description
        if not archetype and cluster.representative_samples:
            archetype = cluster.representative_samples[0].prompt

        request_content = self.VARIANT_PROMPT_TEMPLATE.format(
            count=count,
            archetype=archetype,
        )

        try:
            raw = self._oracle.chat_completions_create(
                model=self._config.oracle_model,
                messages=[{"role": "user", "content": request_content}],
                temperature=self._config.oracle_temperature,
                max_tokens=self._config.variant_prompt_max_tokens,
            )
            return self._parse_json_list(raw)
        except Exception:
            # Graceful degradation: fall back to archetype as the only prompt
            return [archetype] * count

    def get_oracle_response(self, prompt: str) -> str:
        """
        Call oracle model to generate the "chosen" (correct) response.

        Parameters
        ----------
        prompt:
            User prompt string.

        Returns
        -------
        Oracle response string.
        """
        return self._oracle.generate(prompt)

    def estimate_cost(self, n_pairs: int) -> float:
        """
        Rough cost estimate for n_pairs oracle + judge calls.

        Uses the configured cost_per_pair_usd rate.  This is intentionally
        a heuristic — replace with actual token-counting in production.

        Parameters
        ----------
        n_pairs:
            Number of preference pairs synthesized.

        Returns
        -------
        Estimated cost in USD (float).

        Complexity: O(1).
        """
        return float(n_pairs) * self._config.cost_per_pair_usd

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json_list(raw: str) -> List[str]:
        """
        Robustly parse a JSON array of strings from the oracle response.

        Handles:
        - Clean JSON arrays.
        - Arrays embedded in markdown code fences.
        - Partial responses (extracts the first valid [...] block).

        Returns
        -------
        List[str].  Empty list if parsing fails entirely.
        """
        cleaned = raw.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            m = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
            if m:
                cleaned = m.group(1).strip()

        # Find the first [...] block
        m = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)

        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return [str(item) for item in result]
        except (json.JSONDecodeError, TypeError):
            pass

        return []
