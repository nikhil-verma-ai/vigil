"""
Unit tests for DPO training components and cost/provisioner utilities.

Covers:
  - MockDPOJob produces adapter artifact and correct reward_accuracy shape
  - MockDPOJob reward_accuracy monotonically improves over training
  - load_dpo_dataset_from_jsonl: valid + invalid inputs
  - DPOConfig defaults and custom values
  - CostTracker: record, total, breakdown, budget guard
  - MockGPUProvisioner: provision / terminate lifecycle

These tests are CPU-only and have zero external dependencies.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow import without installation
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import datasets  # noqa: E402  (must come after path setup)

from services.trainer.dpo import (  # noqa: E402
    DPOConfig,
    DPOResult,
    MockDPOJob,
    load_dpo_dataset_from_jsonl,
)
from services.trainer.cost_tracker import CostTracker  # noqa: E402
from services.trainer.gpu_provisioner import MockGPUProvisioner  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_preference_dataset(n: int = 20) -> datasets.Dataset:
    """Create a minimal {prompt, chosen, rejected} Dataset with n records."""
    pairs = [
        {"prompt": f"Q{i}", "chosen": f"correct answer {i}", "rejected": f"wrong answer {i}"}
        for i in range(n)
    ]
    return datasets.Dataset.from_list(pairs)


def _train_eval_split(ds: datasets.Dataset, seed: int = 42):
    split = ds.train_test_split(test_size=0.2, seed=seed)
    return split["train"], split["test"]


# ---------------------------------------------------------------------------
# MockDPOJob tests
# ---------------------------------------------------------------------------

class TestMockDPOJob:
    def test_produces_adapter_config(self):
        """
        MockDPOJob must create dpo_adapter/adapter_config.json on success.

        Invariant: adapter_path == output_dir/dpo_adapter and contains
                   adapter_config.json with beta and loss_type keys.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DPOConfig(beta=0.1, logging_steps=5)
            job = MockDPOJob(config=config, output_dir=tmpdir)
            train, eval_ds = _train_eval_split(_make_preference_dataset(20))

            result = job.run("mock://sft-adapter", "mock://ref-adapter", train, eval_ds)

            adapter_cfg = os.path.join(tmpdir, "dpo_adapter", "adapter_config.json")
            assert os.path.exists(adapter_cfg), "adapter_config.json must exist"

            with open(adapter_cfg) as fh:
                cfg = json.load(fh)
            assert cfg["beta"] == 0.1
            assert cfg["loss_type"] == "sigmoid"
            assert cfg["mock"] is True

    def test_result_type_and_reward_accuracy_above_chance(self):
        """
        DPOResult.reward_accuracy must be > 0.5 (better than random) after training.

        Invariant: reward_accuracy in (0.5, 1.0] after a full mock run.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DPOConfig(beta=0.1, logging_steps=5)
            job = MockDPOJob(config=config, output_dir=tmpdir)
            train, eval_ds = _train_eval_split(_make_preference_dataset(20))

            result = job.run("mock://sft-adapter", "mock://ref-adapter", train, eval_ds)

            assert isinstance(result, DPOResult)
            assert result.reward_accuracy > 0.5, (
                f"Reward accuracy {result.reward_accuracy:.4f} must exceed random baseline 0.5"
            )

    def test_adapter_path_matches_result(self):
        """result.adapter_path must point to an existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job = MockDPOJob(DPOConfig(), tmpdir)
            train, eval_ds = _train_eval_split(_make_preference_dataset(20))
            result = job.run("mock://sft", "mock://ref", train, eval_ds)

            assert os.path.isdir(result.adapter_path)

    def test_result_losses_are_positive(self):
        """final_train_loss and final_eval_loss must be > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job = MockDPOJob(DPOConfig(), tmpdir)
            train, eval_ds = _train_eval_split(_make_preference_dataset(20))
            result = job.run("mock://sft", "mock://ref", train, eval_ds)

            assert result.final_train_loss > 0
            assert result.final_eval_loss > 0

    def test_result_training_steps_bounded_by_dataset(self):
        """
        training_steps must be <= dataset size (MockDPOJob caps at min(50, len)).
        """
        n = 15
        with tempfile.TemporaryDirectory() as tmpdir:
            job = MockDPOJob(DPOConfig(logging_steps=5), tmpdir)
            train, eval_ds = _train_eval_split(_make_preference_dataset(n))
            result = job.run("mock://sft", "mock://ref", train, eval_ds)

            assert result.training_steps <= n

    def test_result_duration_positive(self):
        """duration_seconds must be > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job = MockDPOJob(DPOConfig(), tmpdir)
            train, eval_ds = _train_eval_split(_make_preference_dataset(20))
            result = job.run("mock://sft", "mock://ref", train, eval_ds)
            assert result.duration_seconds > 0


class TestMockDPOJobRewardAccuracyImproves:
    def test_reward_accuracy_improves_over_training(self):
        """
        reward_accuracy must end higher than it started.

        Invariant: accuracy_log[-1] > accuracy_log[0] for a dataset >= 10 pairs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            accuracy_log: list[float] = []
            config = DPOConfig(logging_steps=5)
            job = MockDPOJob(
                config,
                tmpdir,
                progress_callback=lambda step, loss, acc: accuracy_log.append(acc),
            )
            train, eval_ds = _train_eval_split(_make_preference_dataset(50))
            result = job.run("mock://sft", "mock://ref", train, eval_ds)

            assert len(accuracy_log) >= 2, "Must have at least 2 log entries to compare"
            assert accuracy_log[-1] > accuracy_log[0], (
                f"Reward accuracy did not improve: {accuracy_log[0]:.4f} → {accuracy_log[-1]:.4f}"
            )

    def test_progress_callback_receives_all_three_args(self):
        """
        progress_callback(step, loss, reward_accuracy) must receive all three args
        with correct types.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_entries: list[tuple] = []
            job = MockDPOJob(
                DPOConfig(logging_steps=5),
                tmpdir,
                progress_callback=lambda s, l, a: log_entries.append((s, l, a)),
            )
            train, eval_ds = _train_eval_split(_make_preference_dataset(30))
            job.run("mock://sft", "mock://ref", train, eval_ds)

            assert len(log_entries) > 0
            step, loss, acc = log_entries[0]
            assert isinstance(step, int)
            assert isinstance(loss, float)
            assert isinstance(acc, float)

    def test_final_reward_accuracy_matches_result(self):
        """
        result.reward_accuracy must match the last progress_callback accuracy value.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            accuracy_log: list[float] = []
            job = MockDPOJob(
                DPOConfig(logging_steps=5),
                tmpdir,
                progress_callback=lambda s, l, a: accuracy_log.append(a),
            )
            train, eval_ds = _train_eval_split(_make_preference_dataset(30))
            result = job.run("mock://sft", "mock://ref", train, eval_ds)

            assert abs(result.reward_accuracy - accuracy_log[-1]) < 1e-9

    def test_reward_accuracy_always_in_unit_interval(self):
        """reward_accuracy must always be in [0.0, 1.0] at every logged step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            accuracy_log: list[float] = []
            job = MockDPOJob(
                DPOConfig(logging_steps=3),
                tmpdir,
                progress_callback=lambda s, l, a: accuracy_log.append(a),
            )
            train, eval_ds = _train_eval_split(_make_preference_dataset(50))
            job.run("mock://sft", "mock://ref", train, eval_ds)

            for acc in accuracy_log:
                assert 0.0 <= acc <= 1.0, f"Reward accuracy {acc} outside [0, 1]"


# ---------------------------------------------------------------------------
# load_dpo_dataset_from_jsonl tests
# ---------------------------------------------------------------------------

class TestLoadDPODataset:
    def test_loads_valid_jsonl(self, tmp_path):
        """Valid JSONL with required keys produces a Dataset with correct columns."""
        path = tmp_path / "pairs.jsonl"
        records = [
            {"prompt": f"Q{i}", "chosen": f"yes {i}", "rejected": f"no {i}"}
            for i in range(10)
        ]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        ds = load_dpo_dataset_from_jsonl(str(path))

        assert len(ds) == 10
        assert set(ds.column_names) >= {"prompt", "chosen", "rejected"}

    def test_skips_blank_lines(self, tmp_path):
        """Blank lines in JSONL must be silently ignored."""
        path = tmp_path / "blank.jsonl"
        records = [{"prompt": f"q{i}", "chosen": f"c{i}", "rejected": f"r{i}"} for i in range(5)]
        path.write_text(
            "\n".join(json.dumps(r) for r in records[:2])
            + "\n\n"
            + "\n".join(json.dumps(r) for r in records[2:])
        )

        ds = load_dpo_dataset_from_jsonl(str(path))
        assert len(ds) == 5

    def test_missing_key_raises_assertion(self, tmp_path):
        """Records missing 'chosen' must raise AssertionError."""
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"prompt": "q", "rejected": "r"}) + "\n")

        with pytest.raises(AssertionError, match="missing required key"):
            load_dpo_dataset_from_jsonl(str(path))

    def test_missing_rejected_raises_assertion(self, tmp_path):
        """Records missing 'rejected' must raise AssertionError."""
        path = tmp_path / "bad2.jsonl"
        path.write_text(json.dumps({"prompt": "q", "chosen": "c"}) + "\n")

        with pytest.raises(AssertionError, match="missing required key"):
            load_dpo_dataset_from_jsonl(str(path))


# ---------------------------------------------------------------------------
# DPOConfig tests
# ---------------------------------------------------------------------------

class TestDPOConfig:
    def test_defaults(self):
        cfg = DPOConfig()
        assert cfg.beta == 0.1
        assert cfg.loss_type == "sigmoid"
        assert cfg.learning_rate == 5e-5
        assert cfg.num_train_epochs == 1
        assert cfg.logging_steps == 10

    def test_custom_values(self):
        cfg = DPOConfig(beta=0.5, loss_type="simpo", learning_rate=1e-5, logging_steps=20)
        assert cfg.beta == 0.5
        assert cfg.loss_type == "simpo"
        assert cfg.learning_rate == 1e-5
        assert cfg.logging_steps == 20


# ---------------------------------------------------------------------------
# CostTracker tests
# ---------------------------------------------------------------------------

class TestCostTracker:
    def test_records_and_totals(self):
        """
        total_for_cycle must equal the sum of all recorded amounts.
        """
        tracker = CostTracker(max_cycle_budget_usd=30.0)
        tracker.record("cycle-001", "sft_gpu", 8.50, {})
        tracker.record("cycle-001", "dpo_gpu", 4.25, {})
        tracker.record("cycle-001", "synthesis_api", 5.00, {})

        total = tracker.total_for_cycle("cycle-001")
        assert abs(total - 17.75) < 0.001

    def test_breakdown(self):
        """
        get_breakdown must return per-component totals.
        """
        tracker = CostTracker(max_cycle_budget_usd=30.0)
        tracker.record("cycle-001", "sft_gpu", 8.50, {})
        tracker.record("cycle-001", "synthesis_api", 5.00, {})

        breakdown = tracker.get_breakdown("cycle-001")
        assert breakdown["sft_gpu"] == 8.50
        assert breakdown["synthesis_api"] == 5.00

    def test_budget_guard_exceeded(self):
        """
        would_exceed_budget returns True when total + additional > max_budget.
        """
        tracker = CostTracker(max_cycle_budget_usd=20.0)
        tracker.record("cycle-001", "sft_gpu", 18.00, {})

        # 18 + 3 = 21 > 20 → should exceed
        assert tracker.would_exceed_budget("cycle-001", 3.00) is True

    def test_budget_guard_within_budget(self):
        """
        would_exceed_budget returns False when total + additional <= max_budget.
        """
        tracker = CostTracker(max_cycle_budget_usd=20.0)
        tracker.record("cycle-001", "sft_gpu", 18.00, {})

        # 18 + 1.5 = 19.5 < 20 → should not exceed
        assert tracker.would_exceed_budget("cycle-001", 1.50) is False

    def test_empty_cycle_total_is_zero(self):
        """A cycle with no records must have total == 0.0."""
        tracker = CostTracker(max_cycle_budget_usd=30.0)
        assert tracker.total_for_cycle("nonexistent") == 0.0

    def test_multiple_cycles_isolated(self):
        """
        Records from different cycle_ids must not bleed into each other.
        """
        tracker = CostTracker(max_cycle_budget_usd=30.0)
        tracker.record("cycle-A", "sft_gpu", 10.0, {})
        tracker.record("cycle-B", "sft_gpu", 5.0, {})

        assert abs(tracker.total_for_cycle("cycle-A") - 10.0) < 0.001
        assert abs(tracker.total_for_cycle("cycle-B") - 5.0) < 0.001

    def test_negative_amount_raises(self):
        """record() must raise ValueError on negative amounts."""
        tracker = CostTracker(max_cycle_budget_usd=30.0)
        with pytest.raises(ValueError):
            tracker.record("cycle-001", "sft_gpu", -1.0, {})

    def test_breakdown_empty_for_unknown_cycle(self):
        """get_breakdown for an unknown cycle returns an empty dict."""
        tracker = CostTracker(max_cycle_budget_usd=30.0)
        assert tracker.get_breakdown("ghost-cycle") == {}

    def test_same_component_accumulated(self):
        """Multiple records for the same component are summed in get_breakdown."""
        tracker = CostTracker(max_cycle_budget_usd=30.0)
        tracker.record("cycle-001", "sft_gpu", 3.0, {})
        tracker.record("cycle-001", "sft_gpu", 2.0, {})

        bd = tracker.get_breakdown("cycle-001")
        assert abs(bd["sft_gpu"] - 5.0) < 0.001


# ---------------------------------------------------------------------------
# MockGPUProvisioner tests
# ---------------------------------------------------------------------------

class TestMockGPUProvisioner:
    def test_provision_returns_running_instance(self):
        """Provisioned instance must have status='RUNNING' and is_spot=True."""
        provisioner = MockGPUProvisioner()
        instance = provisioner.provision("A100-80GB")

        assert instance.status == "RUNNING"
        assert instance.is_spot is True
        assert instance.hourly_cost_usd > 0
        assert instance.instance_id != ""

    def test_terminate_sets_terminated_status(self):
        """After terminate(), instance.status must be 'TERMINATED'."""
        provisioner = MockGPUProvisioner()
        instance = provisioner.provision("A100-80GB")
        provisioner.terminate(instance)

        assert instance.status == "TERMINATED"
        assert provisioner.terminate_count == 1

    def test_provision_count_increments(self):
        """provision_count must increment for each provision() call."""
        provisioner = MockGPUProvisioner()
        provisioner.provision()
        provisioner.provision()
        assert provisioner.provision_count == 2

    def test_instance_ids_are_unique(self):
        """Each provisioned instance must have a unique instance_id."""
        provisioner = MockGPUProvisioner()
        ids = {provisioner.provision().instance_id for _ in range(5)}
        assert len(ids) == 5

    def test_terminate_removes_from_active_instances(self):
        """After terminate(), instance must no longer be in active_instances."""
        provisioner = MockGPUProvisioner()
        instance = provisioner.provision()
        instance_id = instance.instance_id
        assert instance_id in provisioner.active_instances

        provisioner.terminate(instance)
        assert instance_id not in provisioner.active_instances

    def test_simulate_interruption(self):
        """
        When simulate_interruption_after=1, the first provisioned instance
        must have status='INTERRUPTED'.
        """
        provisioner = MockGPUProvisioner(simulate_interruption_after=1)
        instance = provisioner.provision()
        assert instance.status == "INTERRUPTED"

    def test_gpu_type_stored_on_instance(self):
        """The requested gpu_type must be stored as instance_type."""
        provisioner = MockGPUProvisioner()
        instance = provisioner.provision("H100-80GB")
        assert instance.instance_type == "H100-80GB"

    def test_terminate_count_increments(self):
        """terminate_count must increment exactly once per call."""
        provisioner = MockGPUProvisioner()
        i1 = provisioner.provision()
        i2 = provisioner.provision()
        provisioner.terminate(i1)
        provisioner.terminate(i2)
        assert provisioner.terminate_count == 2
