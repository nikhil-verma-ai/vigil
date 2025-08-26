"""Tier 2: Behavioral drift detection."""
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class BehavioralCheckResult:
    check_name: str
    passed: bool
    candidate_value: float
    production_value: float
    delta_pct: float
    threshold: float


class BehavioralDriftChecker:
    def __init__(self, candidate_client, production_client, test_prompts: List[str]):
        self.candidate_client = candidate_client
        self.production_client = production_client
        self.test_prompts = test_prompts

    def check_response_length_distribution(self) -> BehavioralCheckResult:
        candidate_lengths = [
            len(self.candidate_client.generate(p).split())
            for p in self.test_prompts
        ]
        production_lengths = [
            len(self.production_client.generate(p).split())
            for p in self.test_prompts
        ]

        _ks_stat, p_value = ks_2samp(candidate_lengths, production_lengths)

        mean_candidate = float(np.mean(candidate_lengths))
        mean_production = float(np.mean(production_lengths))

        delta_pct = (
            (mean_candidate - mean_production) / mean_production * 100.0
            if mean_production > 0.0
            else 0.0
        )

        return BehavioralCheckResult(
            check_name="response_length_ks_test",
            passed=p_value >= 0.01,
            candidate_value=mean_candidate,
            production_value=mean_production,
            delta_pct=delta_pct,
            threshold=0.01,
        )

    def check_lexical_diversity(self) -> BehavioralCheckResult:
        candidate_responses = [
            self.candidate_client.generate(p) for p in self.test_prompts
        ]
        production_responses = [
            self.production_client.generate(p) for p in self.test_prompts
        ]

        candidate_ttr = self._compute_ttr(candidate_responses)
        production_ttr = self._compute_ttr(production_responses)

        delta_pct = (
            (candidate_ttr - production_ttr) / production_ttr * 100.0
            if production_ttr > 0.0
            else 0.0
        )

        passed = abs(delta_pct) < 15.0

        return BehavioralCheckResult(
            check_name="lexical_diversity_ttr",
            passed=passed,
            candidate_value=candidate_ttr,
            production_value=production_ttr,
            delta_pct=delta_pct,
            threshold=15.0,
        )

    def run_all_checks(self) -> List[BehavioralCheckResult]:
        return [
            self.check_response_length_distribution(),
            self.check_lexical_diversity(),
        ]

    @staticmethod
    def _compute_ttr(responses: List[str]) -> float:
        all_tokens = []
        for response in responses:
            all_tokens.extend(response.lower().split())
        if not all_tokens:
            return 0.0
        unique_tokens = set(all_tokens)
        return len(unique_tokens) / len(all_tokens)
