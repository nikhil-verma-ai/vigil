"""Integration tests for the evaluation gate pipeline."""
import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from services.evaluator.benchmarks import MockModelClient
from services.evaluator.safety import MockSafetyModelClient
from services.evaluator.gate import EvaluationGate


def test_gate_full_evaluation_pipeline():
    """Wire up good candidate, production, safety. Assert all 4 tiers pass."""
    candidate = MockModelClient(accuracy=0.95, seed=10)
    production = MockModelClient(accuracy=0.50, seed=20)
    safety = MockSafetyModelClient(refusal_rate=1.0)
    prompts = [f"What is {i}+{i}?" for i in range(20)]
    failure_prompts = prompts[:10]

    gate = EvaluationGate(candidate, production, safety, prompts, failure_prompts)
    decision = gate.evaluate("adapter-v3", cluster_id=0)

    assert decision.overall_passed == True
    assert decision.tier1_passed == True
    assert decision.tier2_passed == True
    assert decision.tier3_passed == True
    assert decision.tier4_passed == True
    # All 4 tiers recorded
    assert len(decision.tier1_results) > 0
    assert len(decision.tier2_results) > 0
    assert decision.tier3_result is not None
    assert decision.tier4_result is not None


def test_gate_rejects_unsafe_adapter():
    """Candidate with 0.3 refusal rate must fail tier 4."""
    candidate = MockModelClient(accuracy=0.92)
    production = MockModelClient(accuracy=0.85)
    safety = MockSafetyModelClient(refusal_rate=0.3)
    prompts = [f"What is {i}+{i}?" for i in range(20)]
    failure_prompts = prompts[:10]

    gate = EvaluationGate(candidate, production, safety, prompts, failure_prompts)
    decision = gate.evaluate("adapter-unsafe", cluster_id=0)

    assert decision.tier4_passed == False
    assert decision.overall_passed == False
