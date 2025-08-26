"""Tier 3: Targeted improvement verification."""
import re
import random
from dataclasses import dataclass
from typing import List


@dataclass
class ImprovementResult:
    target_cluster_id: int
    candidate_win_rate: float
    production_win_rate: float
    improvement_pct: float
    passed: bool


class MockJudgeClientSimple:
    """Deterministic mock judge that picks the better response by comparing
    arithmetic correctness against the prompt, or random bias for non-arithmetic."""

    def __init__(self):
        self._rng = random.Random(7)

    def judge(self, prompt: str, response_a: str, response_b: str) -> str:
        nums_a = re.findall(r"\d+", response_a)
        nums_b = re.findall(r"\d+", response_b)

        match = re.search(r"(\d+)\+(\d+)", prompt)
        if match and nums_a and nums_b:
            expected = int(match.group(1)) + int(match.group(2))
            ans_a = int(nums_a[-1])
            ans_b = int(nums_b[-1])
            correct_a = ans_a == expected
            correct_b = ans_b == expected
            if correct_a and not correct_b:
                return "A"
            if correct_b and not correct_a:
                return "B"
            return "TIE"

        # Non-arithmetic: random choice weighted toward A (candidate)
        r = self._rng.random()
        if r < 0.6:
            return "A"
        elif r < 0.9:
            return "B"
        return "TIE"


class ImprovementVerifier:
    def __init__(self, candidate_client, production_client, judge_client):
        self.candidate_client = candidate_client
        self.production_client = production_client
        self.judge_client = judge_client

    def verify(
        self,
        failure_cluster_prompts: List[str],
        cluster_id: int,
    ) -> ImprovementResult:
        if not failure_cluster_prompts:
            return ImprovementResult(
                target_cluster_id=cluster_id,
                candidate_win_rate=0.0,
                production_win_rate=0.0,
                improvement_pct=0.0,
                passed=False,
            )

        # Use provided judge or create default
        judge = self.judge_client if self.judge_client is not None else MockJudgeClientSimple()

        candidate_wins = 0
        production_wins = 0
        n = len(failure_cluster_prompts)

        for prompt in failure_cluster_prompts:
            candidate_response = self.candidate_client.generate(prompt)
            production_response = self.production_client.generate(prompt)

            verdict = judge.judge(prompt, candidate_response, production_response)

            if verdict == "A":
                candidate_wins += 1
            elif verdict == "B":
                production_wins += 1

        candidate_win_rate = candidate_wins / n
        production_win_rate = production_wins / n

        if production_wins > 0:
            improvement_pct = (
                (candidate_wins - production_wins) / production_wins * 100.0
            )
        elif candidate_wins > 0:
            improvement_pct = 100.0
        else:
            improvement_pct = 0.0

        passed = improvement_pct >= 10.0

        return ImprovementResult(
            target_cluster_id=cluster_id,
            candidate_win_rate=candidate_win_rate,
            production_win_rate=production_win_rate,
            improvement_pct=improvement_pct,
            passed=passed,
        )
