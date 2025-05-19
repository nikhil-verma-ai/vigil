"""
LLMJudge: evaluates preference pairs through an LLM-as-Judge quality gate.

A preference pair (prompt, chosen, rejected) passes the gate when every
scored dimension meets the configured pass_threshold.  This prevents low-
quality DPO data from entering the training pipeline.

Dimensions scored (each 0.0 - 1.0):
  preference_clarity    — is chosen clearly better than rejected?
  prompt_validity       — is the prompt a realistic user query?
  response_quality      — is chosen actually correct and helpful?
  amplification_fidelity— does this pair exercise the target failure mode?

Gate rule: overall_pass = ALL dimensions >= pass_threshold.

Design notes
------------
* The judge is stateless — no mutable fields after __init__.
* evaluate_batch returns ALL pairs (not just passing ones) so callers can
  compute pass rates without a separate filter pass.
* MockJudgeClient allows deterministic test control without an API key.

Complexity
----------
  evaluate_pair:  1 LLM call, O(1) parse
  evaluate_batch: N LLM calls, O(N) — no batching to preserve per-pair
                  reasoning quality (judge model benefits from per-pair context)
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Domain types                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class JudgeScores:
    """
    Quality scores for a single preference pair.

    All float fields are in [0.0, 1.0].
    overall_pass is True iff ALL dimensions >= the judge's pass_threshold.
    """
    preference_clarity: float       # 0-1, is chosen clearly better?
    prompt_validity: float          # 0-1, is prompt realistic?
    response_quality: float         # 0-1, is chosen actually correct?
    amplification_fidelity: float   # 0-1, does it exercise the target failure?
    overall_pass: bool


@dataclass
class PreferencePair:
    """
    A single DPO training example with optional quality gate metadata.

    Fields
    ------
    pair_id:               UUID string, unique within a synthesis job.
    prompt:                the user query.
    chosen:                the oracle's (correct) response.
    rejected:              the production model's (bad) response.
    source_cluster_id:     HDBSCAN cluster label this pair was amplified from.
    generation_timestamp:  ISO-8601 UTC timestamp of synthesis.
    oracle_model:          model name used to generate the chosen response.
    judge_scores:          set by LLMJudge.evaluate_pair / evaluate_batch.
    passed_gate:           True once judge_scores is set and overall_pass is True.
    """
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    source_cluster_id: int
    generation_timestamp: str
    oracle_model: str
    judge_scores: Optional[JudgeScores] = None
    passed_gate: bool = False


# --------------------------------------------------------------------------- #
# Mock client (test double — no API key required)                              #
# --------------------------------------------------------------------------- #

class MockJudgeClient:
    """
    Deterministic test double for an OpenAI-compatible chat client.

    Parameters
    ----------
    scores:
        Dict with keys matching JudgeScores fields (excluding overall_pass).
        Defaults to all-passing scores (>= 0.7 threshold).

    Usage
    -----
    client = MockJudgeClient(scores={"preference_clarity": 0.3, ...})
    judge = LLMJudge(client=client, pass_threshold=0.7)
    """

    def __init__(self, scores: Dict[str, float] = None) -> None:
        # Default: all dimensions comfortably above the 0.7 pass threshold
        self.scores: Dict[str, float] = scores or {
            "preference_clarity": 0.85,
            "prompt_validity": 0.90,
            "response_quality": 0.80,
            "amplification_fidelity": 0.85,
        }
        self.call_count: int = 0

    def chat_completions_create(self, **kwargs) -> str:
        """
        Simulate an OpenAI chat completion and return a JSON score string.

        Parameters
        ----------
        **kwargs: ignored (model, messages, temperature, etc.)

        Returns
        -------
        JSON string matching the judge prompt response format.
        """
        self.call_count += 1
        return json.dumps(self.scores)


class MockJudgeClientAlternating:
    """
    Test double that alternates between passing and failing on consecutive calls.

    Useful for testing amplification_factor calculations (50% pass rate).
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self._passing_scores: Dict[str, float] = {
            "preference_clarity": 0.9,
            "prompt_validity": 0.9,
            "response_quality": 0.9,
            "amplification_fidelity": 0.9,
        }
        self._failing_scores: Dict[str, float] = {
            "preference_clarity": 0.3,
            "prompt_validity": 0.9,
            "response_quality": 0.9,
            "amplification_fidelity": 0.9,
        }

    def chat_completions_create(self, **kwargs) -> str:
        self.call_count += 1
        # Odd calls pass, even calls fail
        if self.call_count % 2 == 1:
            return json.dumps(self._passing_scores)
        return json.dumps(self._failing_scores)


# --------------------------------------------------------------------------- #
# LLM Judge                                                                    #
# --------------------------------------------------------------------------- #

class LLMJudge:
    """
    Evaluates preference pairs for DPO dataset quality.

    Supports either a real OpenAI client or MockJudgeClient for testing.

    Parameters
    ----------
    client:
        Object with a .chat_completions_create(**kwargs) -> str method.
        Accepts both real OpenAI client wrappers and MockJudgeClient.
        If None, a MockJudgeClient with default scores is used.
    model:
        Judge model identifier.  Defaults to "gpt-4o-mini".
    pass_threshold:
        All dimensions must be >= this float to set overall_pass = True.
        Defaults to 0.7.
    """

    JUDGE_PROMPT_TEMPLATE = """You are evaluating a preference pair for LLM fine-tuning.

PROMPT: {prompt}
CHOSEN RESPONSE: {chosen}
REJECTED RESPONSE: {rejected}

Score each dimension 0.0-1.0:
1. preference_clarity: Is chosen clearly better than rejected? (1.0=clearly better, 0.0=ambiguous)
2. prompt_validity: Is the prompt a realistic user query? (1.0=realistic, 0.0=artificial/adversarial)
3. response_quality: Is chosen actually correct and helpful? (1.0=correct, 0.0=also wrong)
4. amplification_fidelity: Does this pair exercise the same failure mode as described? Target: {cluster_description}

Respond with JSON only: {{"preference_clarity": float, "prompt_validity": float, "response_quality": float, "amplification_fidelity": float}}"""

    def __init__(
        self,
        client=None,
        model: str = "gpt-4o-mini",
        pass_threshold: float = 0.7,
    ) -> None:
        self._client = client if client is not None else MockJudgeClient()
        self._model = model
        self._threshold = pass_threshold

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def evaluate_pair(
        self,
        pair: PreferencePair,
        cluster_description: str,
    ) -> JudgeScores:
        """
        Call the judge model and parse the scored dimensions.

        Parameters
        ----------
        pair:
            PreferencePair to evaluate.
        cluster_description:
            One-sentence description of the target failure mode, passed
            verbatim into the prompt for amplification_fidelity scoring.

        Returns
        -------
        JudgeScores with all dimensions and overall_pass set.

        Side effects: increments client.call_count (for mock clients).

        Raises
        ------
        ValueError: if the judge response cannot be parsed as valid JSON
                    with all required keys.
        """
        prompt_text = self.JUDGE_PROMPT_TEMPLATE.format(
            prompt=pair.prompt,
            chosen=pair.chosen,
            rejected=pair.rejected,
            cluster_description=cluster_description,
        )

        raw_response: str = self._client.chat_completions_create(
            model=self._model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
            max_tokens=256,
        )

        scores_dict = self._parse_judge_response(raw_response)

        dims = [
            "preference_clarity",
            "prompt_validity",
            "response_quality",
            "amplification_fidelity",
        ]

        overall_pass = all(
            float(scores_dict.get(d, 0.0)) >= self._threshold for d in dims
        )

        return JudgeScores(
            preference_clarity=float(scores_dict["preference_clarity"]),
            prompt_validity=float(scores_dict["prompt_validity"]),
            response_quality=float(scores_dict["response_quality"]),
            amplification_fidelity=float(scores_dict["amplification_fidelity"]),
            overall_pass=overall_pass,
        )

    def evaluate_batch(
        self,
        pairs: List[PreferencePair],
        cluster_description: str,
    ) -> List[PreferencePair]:
        """
        Evaluate all pairs and update their judge_scores + passed_gate fields.

        All pairs are returned (not just passing ones) so callers can compute
        pass rates and audit failures without a separate list comprehension.

        Parameters
        ----------
        pairs:
            List of PreferencePairs to evaluate in order.
        cluster_description:
            Passed verbatim to every evaluate_pair call.

        Returns
        -------
        The same list of pairs, each mutated in-place with:
          pair.judge_scores  = JudgeScores(...)
          pair.passed_gate   = judge_scores.overall_pass

        Complexity: O(N) LLM calls — sequential (preserves per-pair context).
        """
        for pair in pairs:
            scores = self.evaluate_pair(pair, cluster_description)
            pair.judge_scores = scores
            pair.passed_gate = scores.overall_pass
        return pairs

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_judge_response(raw: str) -> Dict[str, float]:
        """
        Robustly parse the judge's JSON response.

        Handles:
        - Clean JSON strings.
        - JSON embedded in markdown code fences (```json ... ```).
        - Trailing/leading whitespace.

        Raises
        ------
        ValueError: if valid JSON with all required keys cannot be extracted.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Extract content between first and last ```
            match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        # Find first {...} JSON object in the string (handles trailing text)
        json_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM judge returned unparseable response: {raw!r}"
            ) from exc

        required_keys = {
            "preference_clarity",
            "prompt_validity",
            "response_quality",
            "amplification_fidelity",
        }
        missing = required_keys - set(parsed.keys())
        if missing:
            raise ValueError(
                f"Judge response missing required keys: {missing}. Got: {parsed}"
            )

        return {k: float(parsed[k]) for k in required_keys}
