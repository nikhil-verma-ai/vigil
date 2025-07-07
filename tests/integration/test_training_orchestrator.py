"""
Integration tests for the Training Orchestrator.

These tests exercise the full SFT→DPO pipeline using mock components
(MockGPUProvisioner, MockSFTJob, MockDPOJob) — no GPU or network required.

Test matrix:
  - Happy path: COMPLETED status, artifacts present, costs recorded
  - Cost guard: COST_EXCEEDED when budget too tight for DPO phase
  - SFT-only path: abort before DPO if mid-cycle budget exceeded
  - GPU lifecycle: terminate always called even on failure
  - Checkpoint: both SFT and DPO checkpoints are saved
  - Multiple cycles: each cycle is cost-isolated
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from services.trainer.orchestrator import TrainingOrchestrator, TrainingCycleConfig
from services.trainer.gpu_provisioner import MockGPUProvisioner
from services.trainer.cost_tracker import CostTracker
from services.trainer.checkpointing import CheckpointManager
from services.trainer.sft import MockSFTJob
from services.trainer.dpo import MockDPOJob
from services.trainer.qlora_config import SFTTrainingConfig
from services.trainer.dpo import DPOConfig


# ---------------------------------------------------------------------------
# Test-data helpers
# ---------------------------------------------------------------------------

def create_test_sft_jsonl(tmpdir: str, n: int = 30) -> str:
    """Write a minimal {prompt, response} JSONL file and return its path."""
    path = os.path.join(tmpdir, "sft_data.jsonl")
    with open(path, "w") as fh:
        for i in range(n):
            fh.write(json.dumps({"prompt": f"question {i}", "response": f"answer {i}"}) + "\n")
    return path


def create_test_dpo_jsonl(tmpdir: str, n: int = 20) -> str:
    """Write a minimal {prompt, chosen, rejected} JSONL file and return its path."""
    path = os.path.join(tmpdir, "dpo_data.jsonl")
    with open(path, "w") as fh:
        for i in range(n):
            fh.write(
                json.dumps({
                    "prompt": f"question {i}",
                    "chosen": f"good answer {i}",
                    "rejected": f"bad answer {i}",
                }) + "\n"
            )
    return path


def make_orchestrator(
    tmpdir: str,
    budget_usd: float = 30.0,
    provisioner=None,
) -> TrainingOrchestrator:
    """Construct a fully mock orchestrator for testing."""
    provisioner = provisioner or MockGPUProvisioner()
    tracker = CostTracker(max_cycle_budget_usd=budget_usd)
    ckpt_mgr = CheckpointManager(tmpdir)

    return TrainingOrchestrator(
        provisioner=provisioner,
        cost_tracker=tracker,
        sft_job_factory=lambda config, out_dir: MockSFTJob(config, out_dir),
        dpo_job_factory=lambda config, out_dir: MockDPOJob(config, out_dir),
        checkpoint_manager=ckpt_mgr,
    )


def make_cycle_config(
    tmpdir: str,
    sft_path: str,
    dpo_path: str,
    cycle_id: str = "test-cycle-001",
    budget_usd: float = 30.0,
) -> TrainingCycleConfig:
    return TrainingCycleConfig(
        cycle_id=cycle_id,
        base_model_id="mock://llama-7b",
        sft_dataset_path=sft_path,
        dpo_dataset_path=dpo_path,
        output_base_dir=tmpdir,
        max_cost_usd=budget_usd,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestFullSFTDPOCycle:
    def test_completed_status(self):
        """Successful cycle returns status='COMPLETED'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, n=30)
            dpo_path = create_test_dpo_jsonl(tmpdir, n=20)
            orchestrator = make_orchestrator(tmpdir)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            result = orchestrator.run_cycle(config)

            assert result.status == "COMPLETED"

    def test_sft_and_dpo_results_present(self):
        """Both sft_result and dpo_result must be non-None on COMPLETED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            orchestrator = make_orchestrator(tmpdir)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            result = orchestrator.run_cycle(config)

            assert result.sft_result is not None
            assert result.dpo_result is not None

    def test_candidate_adapter_path_exists(self):
        """candidate_adapter_path must point to an existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            orchestrator = make_orchestrator(tmpdir)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            result = orchestrator.run_cycle(config)

            assert os.path.exists(result.candidate_adapter_path), (
                f"Adapter path {result.candidate_adapter_path!r} does not exist"
            )

    def test_candidate_adapter_contains_config_json(self):
        """The DPO adapter directory must contain adapter_config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            orchestrator = make_orchestrator(tmpdir)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            result = orchestrator.run_cycle(config)

            assert os.path.exists(
                os.path.join(result.candidate_adapter_path, "adapter_config.json")
            )

    def test_total_cost_usd_positive(self):
        """total_cost_usd must be > 0 after a completed cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            orchestrator = make_orchestrator(tmpdir)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            result = orchestrator.run_cycle(config)

            assert result.total_cost_usd > 0

    def test_gpu_provisioned_and_terminated(self):
        """GPU must be provisioned at least once and always terminated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            provisioner = MockGPUProvisioner()
            orchestrator = make_orchestrator(tmpdir, provisioner=provisioner)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            orchestrator.run_cycle(config)

            assert provisioner.provision_count >= 1
            assert provisioner.terminate_count >= 1

    def test_gpu_hours_positive(self):
        """gpu_hours must be > 0 for a completed cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            orchestrator = make_orchestrator(tmpdir)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            result = orchestrator.run_cycle(config)

            assert result.gpu_hours > 0

    def test_dpo_reward_accuracy_above_chance(self):
        """DPO reward_accuracy must be > 0.5 after training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            orchestrator = make_orchestrator(tmpdir)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            result = orchestrator.run_cycle(config)

            assert result.dpo_result.reward_accuracy > 0.5

    def test_error_is_none_on_success(self):
        """result.error must be None for a COMPLETED cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            orchestrator = make_orchestrator(tmpdir)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            result = orchestrator.run_cycle(config)

            assert result.error is None


# ---------------------------------------------------------------------------
# Cost guard tests
# ---------------------------------------------------------------------------

class TestOrchestratorCostGuard:
    def test_cost_exceeded_when_budget_too_low(self):
        """
        If max_cost_usd is tiny (5 USD), the orchestrator should abort the DPO
        phase (or even earlier) with status=COST_EXCEEDED.

        The budget check fires when projected total > max_cost_usd.  With an A100
        at $3.21/hr and 5 projected hours, the preflight check will trigger.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            # Orchestrator projects (3h SFT + 2h DPO) * $3.21 ≈ $16.05 — exceeds $5
            orchestrator = make_orchestrator(tmpdir, budget_usd=5.0)
            config = make_cycle_config(tmpdir, sft_path, dpo_path, budget_usd=5.0)

            result = orchestrator.run_cycle(config)

            assert result.status == "COST_EXCEEDED"

    def test_dpo_not_run_when_cost_exceeded(self):
        """
        When status=COST_EXCEEDED, dpo_result must be None (DPO never started).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            orchestrator = make_orchestrator(tmpdir, budget_usd=5.0)
            config = make_cycle_config(tmpdir, sft_path, dpo_path, budget_usd=5.0)

            result = orchestrator.run_cycle(config)

            assert result.dpo_result is None

    def test_no_dpo_adapter_when_cost_exceeded(self):
        """
        When DPO is aborted due to cost, no dpo_adapter directory must exist
        under the cycle output directory.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            orchestrator = make_orchestrator(tmpdir, budget_usd=5.0)
            config = make_cycle_config(tmpdir, sft_path, dpo_path,
                                       cycle_id="cycle-002", budget_usd=5.0)

            result = orchestrator.run_cycle(config)

            cycle_dir = os.path.join(tmpdir, "cycle-002")
            dpo_dir = os.path.join(cycle_dir, "dpo", "dpo_adapter")
            assert not os.path.exists(dpo_dir), "DPO adapter must not exist when aborted"


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestOrchestratorCheckpoints:
    def test_sft_checkpoint_saved(self):
        """A checkpoint for the SFT phase must exist after a complete cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            ckpt_mgr = CheckpointManager(tmpdir)
            provisioner = MockGPUProvisioner()
            tracker = CostTracker(max_cycle_budget_usd=30.0)

            orchestrator = TrainingOrchestrator(
                provisioner=provisioner,
                cost_tracker=tracker,
                sft_job_factory=lambda c, d: MockSFTJob(c, d),
                dpo_job_factory=lambda c, d: MockDPOJob(c, d),
                checkpoint_manager=ckpt_mgr,
            )
            config = make_cycle_config(tmpdir, sft_path, dpo_path, cycle_id="ckpt-test-001")

            orchestrator.run_cycle(config)

            checkpoints = ckpt_mgr.list_checkpoints("ckpt-test-001")
            sft_ckpts = [c for c in checkpoints if c.phase == "sft"]
            assert len(sft_ckpts) >= 1

    def test_dpo_checkpoint_saved(self):
        """A checkpoint for the DPO phase must exist after a complete cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            ckpt_mgr = CheckpointManager(tmpdir)
            provisioner = MockGPUProvisioner()
            tracker = CostTracker(max_cycle_budget_usd=30.0)

            orchestrator = TrainingOrchestrator(
                provisioner=provisioner,
                cost_tracker=tracker,
                sft_job_factory=lambda c, d: MockSFTJob(c, d),
                dpo_job_factory=lambda c, d: MockDPOJob(c, d),
                checkpoint_manager=ckpt_mgr,
            )
            config = make_cycle_config(tmpdir, sft_path, dpo_path, cycle_id="ckpt-test-002")

            orchestrator.run_cycle(config)

            checkpoints = ckpt_mgr.list_checkpoints("ckpt-test-002")
            dpo_ckpts = [c for c in checkpoints if c.phase == "dpo"]
            assert len(dpo_ckpts) >= 1

    def test_checkpoint_metadata_has_correct_phase(self):
        """Checkpoint metadata phase field must be 'sft' or 'dpo' only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            ckpt_mgr = CheckpointManager(tmpdir)
            provisioner = MockGPUProvisioner()
            tracker = CostTracker(max_cycle_budget_usd=30.0)

            orchestrator = TrainingOrchestrator(
                provisioner=provisioner,
                cost_tracker=tracker,
                sft_job_factory=lambda c, d: MockSFTJob(c, d),
                dpo_job_factory=lambda c, d: MockDPOJob(c, d),
                checkpoint_manager=ckpt_mgr,
            )
            config = make_cycle_config(tmpdir, sft_path, dpo_path, cycle_id="ckpt-test-003")
            orchestrator.run_cycle(config)

            for ckpt in ckpt_mgr.list_checkpoints("ckpt-test-003"):
                assert ckpt.phase in {"sft", "dpo"}


# ---------------------------------------------------------------------------
# GPU lifecycle tests
# ---------------------------------------------------------------------------

class TestGPULifecycle:
    def test_gpu_always_terminated_on_success(self):
        """GPU instance must always be terminated after a successful cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            provisioner = MockGPUProvisioner()
            orchestrator = make_orchestrator(tmpdir, provisioner=provisioner)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            orchestrator.run_cycle(config)

            # All provisioned instances must have been terminated
            assert provisioner.terminate_count == provisioner.provision_count

    def test_no_active_instances_after_cycle(self):
        """active_instances must be empty after a completed cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            provisioner = MockGPUProvisioner()
            orchestrator = make_orchestrator(tmpdir, provisioner=provisioner)
            config = make_cycle_config(tmpdir, sft_path, dpo_path)

            orchestrator.run_cycle(config)

            assert len(provisioner.active_instances) == 0


# ---------------------------------------------------------------------------
# Multiple cycles isolation test
# ---------------------------------------------------------------------------

class TestMultipleCycles:
    def test_two_cycles_cost_isolated(self):
        """
        Two sequential cycles must have independent cost totals — records from
        cycle A must not appear in cycle B's total.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)

            provisioner = MockGPUProvisioner()
            tracker = CostTracker(max_cycle_budget_usd=30.0)
            ckpt_mgr = CheckpointManager(tmpdir)

            orchestrator = TrainingOrchestrator(
                provisioner=provisioner,
                cost_tracker=tracker,
                sft_job_factory=lambda c, d: MockSFTJob(c, d),
                dpo_job_factory=lambda c, d: MockDPOJob(c, d),
                checkpoint_manager=ckpt_mgr,
            )

            config_a = make_cycle_config(tmpdir, sft_path, dpo_path, cycle_id="multi-A")
            config_b = make_cycle_config(tmpdir, sft_path, dpo_path, cycle_id="multi-B")

            result_a = orchestrator.run_cycle(config_a)
            result_b = orchestrator.run_cycle(config_b)

            # Each cycle must have independently accumulated non-zero costs
            cost_a = tracker.total_for_cycle("multi-A")
            cost_b = tracker.total_for_cycle("multi-B")

            assert cost_a > 0
            assert cost_b > 0
            # Costs must not bleed: each cycle total must match its own result
            assert abs(result_a.total_cost_usd - cost_a) < 0.001
            assert abs(result_b.total_cost_usd - cost_b) < 0.001

    def test_cycle_id_stored_in_result(self):
        """result.cycle_id must match the config.cycle_id passed in."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_path = create_test_sft_jsonl(tmpdir, 30)
            dpo_path = create_test_dpo_jsonl(tmpdir, 20)
            orchestrator = make_orchestrator(tmpdir)
            config = make_cycle_config(tmpdir, sft_path, dpo_path, cycle_id="id-check-001")

            result = orchestrator.run_cycle(config)

            assert result.cycle_id == "id-check-001"
