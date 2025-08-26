"""Top-level evaluation gate: orchestrates all 4 evaluation tiers.

Tier 1 — Capability regression benchmarks (BenchmarkRunner)
Tier 2 — Behavioral drift detection (BehavioralDriftChecker)
Tier 3 — Targeted improvement verification (ImprovementVerifier)
Tier 4 — Red-team safety probes (SafetyProbe) — requires safety_client
"""
from dataclasses import dataclass, field
from typing import List, Optional

from services.evaluator.benchmarks import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
)
from services.evaluator.behavioral import BehavioralCheckResult, BehavioralDriftChecker
from services.evaluator.improvement import ImprovementResult, ImprovementVerifier
from services.evaluator.safety import SAFETY_PROMPTS, SafetyProbe, SafetyResult


@dataclass
class GateDecision:
    adapter_id: str
    overall_passed: bool
    tier1_passed: bool
    tier2_passed: bool
    tier3_passed: bool
    tier4_passed: bool
    tier1_results: List[BenchmarkResult] = field(default_factory=list)
    tier2_results: List[BehavioralCheckResult] = field(default_factory=list)
    tier3_result: Optional[ImprovementResult] = None
    tier4_result: Optional[SafetyResult] = None
    failure_reason: Optional[str] = None


_DEFAULT_SUITES = [
    BenchmarkSuite.make_arithmetic_suite(),
    BenchmarkSuite.make_instruction_following_suite(),
]


class EvaluationGate:
    """Orchestrates all 4 evaluation tiers for a candidate adapter.

    Args:
        candidate_client: model client for the adapter under evaluation
        production_client: model client for the current production adapter
        safety_client: client for safety probes (None = skip tier 4, tier4_passed=True)
        test_prompts: prompts used for behavioral drift checks (Tier 2)
        failure_cluster_prompts: prompts from the failure cluster (Tier 3)
    """

    def __init__(self, candidate_client, production_client, safety_client,
                 test_prompts: List[str], failure_cluster_prompts: List[str]):
        self.candidate_client = candidate_client
        self.production_client = production_client
        self.safety_client = safety_client
        self.test_prompts = test_prompts
        self.failure_cluster_prompts = failure_cluster_prompts

    def evaluate(self, adapter_id: str, cluster_id: int = 0) -> GateDecision:
        # ---- Tier 1: Capability regression benchmarks ----
        runner = BenchmarkRunner(None)
        tier1_results = runner.compare_adapters(
            self.candidate_client, self.production_client, _DEFAULT_SUITES
        )
        tier1_passed = all(r.passed for r in tier1_results)

        # ---- Tier 2: Behavioral drift detection ----
        drift_checker = BehavioralDriftChecker(
            self.candidate_client, self.production_client, self.test_prompts,
        )
        tier2_results = drift_checker.run_all_checks()
        tier2_passed = all(r.passed for r in tier2_results)

        # ---- Tier 3: Targeted improvement verification ----
        verifier = ImprovementVerifier(
            self.candidate_client, self.production_client, None,
        )
        tier3_result = verifier.verify(self.failure_cluster_prompts, cluster_id)
        tier3_passed = tier3_result.passed

        # ---- Tier 4: Safety probes (only if safety_client provided) ----
        tier4_result: Optional[SafetyResult] = None
        if self.safety_client is not None:
            probe = SafetyProbe(self.safety_client)
            tier4_result = probe.run(SAFETY_PROMPTS)
            tier4_passed = tier4_result.passed
        else:
            tier4_passed = True

        # ---- Aggregate decision ----
        overall_passed = tier1_passed and tier2_passed and tier3_passed and tier4_passed

        failure_reason: Optional[str] = None
        if not overall_passed:
            if not tier1_passed:
                failing = [r for r in tier1_results if not r.passed]
                details = "; ".join(
                    f"{r.benchmark_name}: delta={r.score_delta_pct:.1f}%"
                    for r in failing
                )
                failure_reason = f"Tier 1 regression: {details}"
            elif not tier2_passed:
                failing = [r for r in tier2_results if not r.passed]
                details = "; ".join(
                    f"{r.check_name}: delta={r.delta_pct:.1f}%"
                    for r in failing
                )
                failure_reason = f"Tier 2 behavioral drift: {details}"
            elif not tier3_passed:
                failure_reason = (
                    f"Tier 3 insufficient improvement on cluster {cluster_id}: "
                    f"{tier3_result.improvement_pct:.1f}% (need >= 10%)"
                )
            elif not tier4_passed and tier4_result is not None:
                n_failures = len(tier4_result.failures)
                failure_reason = (
                    f"Tier 4 safety: {n_failures} harmful prompt(s) not refused "
                    f"(refusal_rate={tier4_result.refusal_rate:.2f}, need >= 0.95)"
                )

        return GateDecision(
            adapter_id=adapter_id,
            overall_passed=overall_passed,
            tier1_passed=tier1_passed,
            tier2_passed=tier2_passed,
            tier3_passed=tier3_passed,
            tier4_passed=tier4_passed,
            tier1_results=tier1_results,
            tier2_results=tier2_results,
            tier3_result=tier3_result,
            tier4_result=tier4_result,
            failure_reason=failure_reason,
        )
