"""Tier 1: Capability regression benchmarks."""
import random
import re
from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkResult:
    benchmark_name: str
    candidate_score: float
    production_score: float
    score_delta_pct: float
    passed: bool


@dataclass
class BenchmarkSuite:
    name: str
    examples: List[dict]

    @classmethod
    def make_arithmetic_suite(cls) -> "BenchmarkSuite":
        return cls(
            name="arithmetic",
            examples=[
                {"question": f"What is {a}+{b}?", "correct_answer": str(a + b)}
                for a, b in [
                    (2, 3), (10, 15), (7, 8), (100, 200), (15, 25),
                    (3, 7), (12, 18), (50, 50), (99, 1), (33, 67),
                ]
            ],
        )

    @classmethod
    def make_instruction_following_suite(cls) -> "BenchmarkSuite":
        return cls(
            name="instruction_following",
            examples=[
                {"question": "List exactly 3 fruits", "correct_answer": "3_items"},
                {"question": "Write a word that starts with 'Z'", "correct_answer": "Z_prefix"},
                {"question": "Answer in exactly one word: What color is the sky?", "correct_answer": "Blue"},
            ],
        )


class MockModelClient:
    def __init__(self, accuracy: float = 0.85, model_id: str = "mock-adapter-v1", seed: int = 42):
        self.accuracy = accuracy
        self.model_id = model_id
        self.call_count = 0
        self._rng = random.Random(seed)

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        match = re.search(r"(\d+)\+(\d+)", prompt)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            correct = str(a + b)
            if self._rng.random() < self.accuracy:
                return f"The answer is {correct}."
            else:
                return f"The answer is {a + b + 1}."
        return f"Response to: {prompt[:30]}"


class BenchmarkRunner:
    def __init__(self, model_client):
        self.model_client = model_client

    def run_suite(self, suite: BenchmarkSuite) -> float:
        correct = 0
        for example in suite.examples:
            response = self.model_client.generate(example["question"])
            if self._check_answer(response, example["correct_answer"]):
                correct += 1
        return correct / len(suite.examples)

    def compare_adapters(
        self,
        candidate_client,
        production_client,
        suites: List[BenchmarkSuite],
    ) -> List[BenchmarkResult]:
        results: List[BenchmarkResult] = []
        for suite in suites:
            candidate_runner = BenchmarkRunner(candidate_client)
            production_runner = BenchmarkRunner(production_client)
            candidate_score = candidate_runner.run_suite(suite)
            production_score = production_runner.run_suite(suite)

            if production_score > 0.0:
                delta_pct = (candidate_score - production_score) / production_score * 100.0
            else:
                delta_pct = 0.0 if candidate_score == 0.0 else 100.0

            passed = candidate_score >= 0.95 * production_score

            results.append(
                BenchmarkResult(
                    benchmark_name=suite.name,
                    candidate_score=candidate_score,
                    production_score=production_score,
                    score_delta_pct=delta_pct,
                    passed=passed,
                )
            )
        return results

    def _check_answer(self, response: str, expected: str) -> bool:
        if expected == "3_items":
            items = [
                line.strip()
                for line in response.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return len(items) == 3

        if expected == "Z_prefix":
            words = response.split()
            return any(w.upper().startswith("Z") for w in words)

        return expected.lower() in response.lower()
