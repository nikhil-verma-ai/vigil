"""Unit tests for Module 5: Safety Evaluation & Regression Gate."""
import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import random
import pytest
from services.evaluator.benchmarks import BenchmarkRunner, BenchmarkSuite, MockModelClient, BenchmarkResult
from services.evaluator.safety import SafetyProbe, SafetyResult, MockSafetyModelClient, SAFETY_PROMPTS
from services.evaluator.behavioral import BehavioralDriftChecker
from services.evaluator.improvement import ImprovementVerifier
from services.evaluator.gate import EvaluationGate, GateDecision


# ---------------------------------------------------------------------------
# Spec-mandated standalone tests
# ---------------------------------------------------------------------------

def test_benchmark_perfect_model_scores_1():
    client = MockModelClient(accuracy=1.0)
    runner = BenchmarkRunner(client)
    suite = BenchmarkSuite.make_arithmetic_suite()
    score = runner.run_suite(suite)
    assert score == 1.0


def test_benchmark_zero_accuracy_scores_0():
    client = MockModelClient(accuracy=0.0)
    runner = BenchmarkRunner(client)
    suite = BenchmarkSuite.make_arithmetic_suite()
    score = runner.run_suite(suite)
    assert score == 0.0


def test_benchmark_regression_fails_when_10pct_worse():
    candidate = MockModelClient(accuracy=0.80)
    production = MockModelClient(accuracy=0.90)
    runner = BenchmarkRunner(None)
    suite = BenchmarkSuite.make_arithmetic_suite()
    results = runner.compare_adapters(candidate, production, [suite])
    assert len(results) == 1
    assert results[0].passed == False  # 0.80 < 0.95 * 0.90 = 0.855


def test_benchmark_passes_when_equal_or_better():
    candidate = MockModelClient(accuracy=0.92)
    production = MockModelClient(accuracy=0.90)
    runner = BenchmarkRunner(None)
    results = runner.compare_adapters(candidate, production, [BenchmarkSuite.make_arithmetic_suite()])
    assert results[0].passed == True


def test_safety_perfect_refusal_passes():
    client = MockSafetyModelClient(refusal_rate=1.0)
    probe = SafetyProbe(client)
    result = probe.run(SAFETY_PROMPTS)
    assert result.passed == True
    assert result.refusal_rate == 1.0
    assert len(result.failures) == 0


def test_safety_low_refusal_fails():
    client = MockSafetyModelClient(refusal_rate=0.6)
    probe = SafetyProbe(client)
    result = probe.run(SAFETY_PROMPTS)
    assert result.passed == False
    assert len(result.failures) > 0


def test_behavioral_similar_models_pass_ks():
    candidate = MockModelClient(accuracy=0.85, seed=1)
    production = MockModelClient(accuracy=0.85, seed=2)
    prompts = [f"What is {i}+{i}?" for i in range(30)]
    checker = BehavioralDriftChecker(candidate, production, prompts)
    result = checker.check_response_length_distribution()
    assert result.passed == True


def test_improvement_verifier_detects_improvement():
    candidate = MockModelClient(accuracy=0.90)
    production = MockModelClient(accuracy=0.40)
    verifier = ImprovementVerifier(candidate, production, None)
    prompts = [f"failure prompt {i}" for i in range(20)]
    result = verifier.verify(prompts, cluster_id=0)
    assert result.passed == True
    assert result.improvement_pct > 10.0


def test_full_gate_passes_good_adapter():
    candidate = MockModelClient(accuracy=0.95)
    production = MockModelClient(accuracy=0.88)
    safety = MockSafetyModelClient(refusal_rate=1.0)
    prompts = [f"What is {i}+{i}?" for i in range(20)]
    gate = EvaluationGate(candidate, production, None, prompts, prompts[:10])
    decision = gate.evaluate("adapter-v2", cluster_id=0)
    assert decision.tier1_passed == True
    assert decision.tier4_passed == True


def test_full_gate_fails_on_regression():
    candidate = MockModelClient(accuracy=0.70)
    production = MockModelClient(accuracy=0.90)
    prompts = [f"What is {i}+{i}?" for i in range(20)]
    gate = EvaluationGate(candidate, production, None, prompts, prompts[:10])
    decision = gate.evaluate("adapter-bad", cluster_id=0)
    assert decision.tier1_passed == False
    assert decision.overall_passed == False
