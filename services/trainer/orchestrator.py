"""
Training cycle orchestrator — wires SFT → DPO → artifact publishing.

Responsibilities:
  1. Budget pre-flight check before provisioning
  2. GPU provision / terminate (always in a try/finally to prevent resource leaks)
  3. SFT phase with checkpoint after completion
  4. Mid-cycle budget guard before starting DPO
  5. DPO phase warm-started from SFT adapter
  6. Final checkpoint
  7. Kafka event publication (training.jobs topic)

Status values:
  COMPLETED      — SFT + DPO both finished successfully
  FAILED         — unrecoverable exception during training
  COST_EXCEEDED  — budget guard triggered before or after SFT phase
  CANCELLED      — externally requested cancellation (future use)

Thread safety: not required — one orchestrator per process, sequential phases.
"""
from __future__ import annotations

import datetime
import os
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import structlog

log = structlog.get_logger(__name__)

# Insert project root so shared.schemas is importable without install
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared.schemas.events import (
    TrainingCycleEvent,
    TriggerType,
    TOPIC_TRAINING_JOBS,
)
from services.trainer.sft import SFTResult
from services.trainer.dpo import DPOResult
from services.trainer.checkpointing import CheckpointMetadata
from services.trainer.qlora_config import SFTTrainingConfig
from services.trainer.dataset import load_sft_dataset_from_jsonl, split_dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingCycleConfig:
    """
    All parameters needed to execute a full SFT → DPO training cycle.

    Fields:
      cycle_id            — globally unique identifier for this cycle
      base_model_id       — HuggingFace model ID or local path; "mock://" for tests
      sft_dataset_path    — path to {prompt, response} JSONL for SFT
      dpo_dataset_path    — path to {prompt, chosen, rejected} JSONL for DPO
      output_base_dir     — root directory under which per-cycle artifacts are written
      max_cost_usd        — hard budget ceiling; phases are aborted if exceeded
      trigger_event_id    — optional drift event ID that triggered this cycle
      triggered_by        — "DRIFT" | "SCHEDULE" | "MANUAL"
    """
    cycle_id: str
    base_model_id: str
    sft_dataset_path: str
    dpo_dataset_path: str
    output_base_dir: str
    max_cost_usd: float = 30.0
    trigger_event_id: Optional[str] = None
    triggered_by: str = "DRIFT"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """
    Outcome of a completed (or failed/aborted) training cycle.

    Fields:
      cycle_id               — mirrors TrainingCycleConfig.cycle_id
      status                 — COMPLETED | FAILED | COST_EXCEEDED | CANCELLED
      sft_result             — SFTResult if SFT phase completed, else None
      dpo_result             — DPOResult if DPO phase completed, else None
      candidate_adapter_path — path to the final DPO adapter (or SFT if DPO skipped)
      total_cost_usd         — sum of all recorded costs for this cycle
      gpu_hours              — estimated GPU wall-clock hours
      error                  — exception message/traceback if status == FAILED
    """
    cycle_id: str
    status: str
    sft_result: Optional[SFTResult]
    dpo_result: Optional[DPOResult]
    candidate_adapter_path: str
    total_cost_usd: float
    gpu_hours: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TrainingOrchestrator:
    """
    Orchestrates a full SFT → DPO training cycle.

    Purpose:  sequence all phases, manage GPU lifecycle, enforce cost budgets,
              save checkpoints, and publish Kafka events.
    Inputs:
      provisioner        — GPUProvisioner or MockGPUProvisioner
      cost_tracker       — CostTracker for budget accounting
      sft_job_factory    — callable(SFTTrainingConfig, output_dir) → SFTTrainingJob/MockSFTJob
      dpo_job_factory    — callable(DPOConfig, output_dir) → DPOTrainingJob/MockDPOJob
      checkpoint_manager — CheckpointManager for saving run state
      kafka_producer     — optional confluent_kafka.Producer; skipped if None
    """

    # Estimated GPU hours per phase (used for projected cost check).
    # These are conservative upper bounds; real durations depend on dataset size.
    _PROJECTED_SFT_HOURS: float = 3.0
    _PROJECTED_DPO_HOURS: float = 2.0
    _HOURLY_RATE_USD: float = 3.21   # A100 spot estimate

    def __init__(
        self,
        provisioner,
        cost_tracker,
        sft_job_factory: Callable,
        dpo_job_factory: Callable,
        checkpoint_manager,
        kafka_producer=None,
    ) -> None:
        self.provisioner = provisioner
        self.cost_tracker = cost_tracker
        self.sft_job_factory = sft_job_factory
        self.dpo_job_factory = dpo_job_factory
        self.checkpoint_manager = checkpoint_manager
        self.kafka_producer = kafka_producer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(self, config: TrainingCycleConfig) -> OrchestratorResult:
        """
        Execute a full training cycle: provision → SFT → checkpoint → DPO → publish.

        Purpose:  drive the complete SFT→DPO pipeline with budget enforcement,
                  GPU lifecycle management, checkpointing, and event publishing.
        Inputs:   config — TrainingCycleConfig with all cycle parameters
        Outputs:  OrchestratorResult describing final cycle state
        Complexity: O(training steps * model size)
        Side effects:
          - Provisions and terminates GPU instance
          - Writes artifacts to config.output_base_dir/<cycle_id>/
          - Records costs to self.cost_tracker
          - Saves checkpoints via self.checkpoint_manager
          - Publishes TrainingCycleEvent to Kafka (if producer available)
        """
        cycle_dir = os.path.join(config.output_base_dir, config.cycle_id)
        os.makedirs(cycle_dir, exist_ok=True)

        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        sft_result: Optional[SFTResult] = None
        dpo_result: Optional[DPOResult] = None
        instance = None

        log.info("cycle_start", cycle_id=config.cycle_id, triggered_by=config.triggered_by)

        # ------------------------------------------------------------------
        # Step 1: Pre-flight budget check
        # ------------------------------------------------------------------
        projected_total = (
            (self._PROJECTED_SFT_HOURS + self._PROJECTED_DPO_HOURS)
            * self._HOURLY_RATE_USD
        )
        if self.cost_tracker.would_exceed_budget(config.cycle_id, projected_total):
            log.warning(
                "cycle_cost_exceeded_preflight",
                cycle_id=config.cycle_id,
                projected_usd=projected_total,
                budget_usd=config.max_cost_usd,
            )
            return self._make_result(
                config, "COST_EXCEEDED", sft_result, dpo_result, cycle_dir,
                "Projected cost exceeds budget before provisioning",
            )

        # ------------------------------------------------------------------
        # Steps 2–7: GPU-protected block — always terminate in finally
        # ------------------------------------------------------------------
        try:
            # Step 2: Provision GPU
            instance = self.provisioner.provision("A100-80GB")
            log.info(
                "gpu_provisioned",
                cycle_id=config.cycle_id,
                instance_id=instance.instance_id,
                hourly_usd=instance.hourly_cost_usd,
            )

            # ---- Step 3: SFT phase ----------------------------------------
            sft_result = self._run_sft_phase(config, cycle_dir, instance)
            sft_cost = (sft_result.duration_seconds / 3600.0) * instance.hourly_cost_usd
            self.cost_tracker.record(
                cycle_id=config.cycle_id,
                component="sft_gpu",
                amount_usd=sft_cost,
                metadata={
                    "instance_id": instance.instance_id,
                    "steps": sft_result.training_steps,
                    "duration_s": sft_result.duration_seconds,
                },
            )

            # Step 4: Checkpoint after SFT
            self.checkpoint_manager.save(
                CheckpointMetadata(
                    cycle_id=config.cycle_id,
                    step=sft_result.training_steps,
                    phase="sft",
                    train_loss=sft_result.final_train_loss,
                    eval_loss=sft_result.final_eval_loss,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                ),
                sft_result.adapter_path,
            )

            # Step 5: Mid-cycle budget guard
            remaining_budget = config.max_cost_usd - self.cost_tracker.total_for_cycle(config.cycle_id)
            projected_dpo_cost = self._PROJECTED_DPO_HOURS * instance.hourly_cost_usd
            if self.cost_tracker.would_exceed_budget(config.cycle_id, projected_dpo_cost):
                log.warning(
                    "cycle_cost_exceeded_post_sft",
                    cycle_id=config.cycle_id,
                    current_usd=self.cost_tracker.total_for_cycle(config.cycle_id),
                    projected_dpo_usd=projected_dpo_cost,
                    budget_usd=config.max_cost_usd,
                )
                return self._make_result(
                    config, "COST_EXCEEDED", sft_result, dpo_result, cycle_dir,
                    "Cost exceeded budget after SFT phase; DPO aborted",
                )

            # ---- Step 6: DPO phase (warm-started from SFT adapter) ---------
            dpo_result = self._run_dpo_phase(
                config, cycle_dir, instance, sft_result.adapter_path
            )
            dpo_cost = (dpo_result.duration_seconds / 3600.0) * instance.hourly_cost_usd
            self.cost_tracker.record(
                cycle_id=config.cycle_id,
                component="dpo_gpu",
                amount_usd=dpo_cost,
                metadata={
                    "instance_id": instance.instance_id,
                    "steps": dpo_result.training_steps,
                    "reward_accuracy": dpo_result.reward_accuracy,
                    "duration_s": dpo_result.duration_seconds,
                },
            )

            # Step 7: Final checkpoint
            self.checkpoint_manager.save(
                CheckpointMetadata(
                    cycle_id=config.cycle_id,
                    step=dpo_result.training_steps,
                    phase="dpo",
                    train_loss=dpo_result.final_train_loss,
                    eval_loss=dpo_result.final_eval_loss,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                ),
                dpo_result.adapter_path,
            )

            result = self._make_result(
                config, "COMPLETED", sft_result, dpo_result, cycle_dir
            )

        except Exception as exc:
            tb = traceback.format_exc()
            log.error("cycle_failed", cycle_id=config.cycle_id, error=str(exc), traceback=tb)
            result = self._make_result(
                config, "FAILED", sft_result, dpo_result, cycle_dir,
                error=f"{exc}\n{tb}",
            )

        finally:
            # Always terminate the GPU — never leave instances running
            if instance is not None and instance.status == "RUNNING":
                try:
                    self.provisioner.terminate(instance)
                    log.info("gpu_terminated_cleanup", instance_id=instance.instance_id)
                except Exception as term_exc:
                    log.error(
                        "gpu_terminate_failed",
                        instance_id=instance.instance_id,
                        error=str(term_exc),
                    )

        # Step 8: Publish Kafka event
        self._publish_kafka_event(config, result, started_at)

        log.info(
            "cycle_complete",
            cycle_id=config.cycle_id,
            status=result.status,
            total_cost_usd=result.total_cost_usd,
            gpu_hours=result.gpu_hours,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_sft_phase(
        self, config: TrainingCycleConfig, cycle_dir: str, instance
    ) -> SFTResult:
        """
        Load the SFT dataset, create an SFT job, and run it.

        Purpose:  isolate SFT phase logic for testability and readability.
        Inputs:
          config    — cycle configuration
          cycle_dir — output directory for this cycle
          instance  — provisioned GPUInstance (used for logging only)
        Outputs:  SFTResult
        Side effects: writes SFT adapter to cycle_dir/sft/
        """
        sft_out = os.path.join(cycle_dir, "sft")
        os.makedirs(sft_out, exist_ok=True)

        log.info("sft_phase_start", cycle_id=config.cycle_id)

        sft_dataset = load_sft_dataset_from_jsonl(config.sft_dataset_path)
        train_ds, eval_ds = split_dataset(sft_dataset, eval_fraction=0.1)

        sft_config = SFTTrainingConfig()
        job = self.sft_job_factory(sft_config, sft_out)
        result = job.run(config.base_model_id, train_ds, eval_ds)

        log.info(
            "sft_phase_complete",
            cycle_id=config.cycle_id,
            steps=result.training_steps,
            train_loss=result.final_train_loss,
        )
        return result

    def _run_dpo_phase(
        self,
        config: TrainingCycleConfig,
        cycle_dir: str,
        instance,
        sft_adapter_path: str,
    ) -> DPOResult:
        """
        Load the DPO dataset, create a DPO job, and run it warm-started from SFT.

        Purpose:  isolate DPO phase logic; warm-start from SFT adapter so the
                  policy already understands the task format before preference tuning.
        Inputs:
          config           — cycle configuration
          cycle_dir        — output directory for this cycle
          instance         — provisioned GPUInstance (used for logging only)
          sft_adapter_path — path to completed SFT adapter for warm-start
        Outputs:  DPOResult
        Side effects: writes DPO adapter to cycle_dir/dpo/
        """
        from services.trainer.dpo import DPOConfig, load_dpo_dataset_from_jsonl

        dpo_out = os.path.join(cycle_dir, "dpo")
        os.makedirs(dpo_out, exist_ok=True)

        log.info("dpo_phase_start", cycle_id=config.cycle_id, warm_start=sft_adapter_path)

        dpo_dataset = load_dpo_dataset_from_jsonl(config.dpo_dataset_path)
        split = dpo_dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split["train"], split["test"]

        dpo_config = DPOConfig()
        job = self.dpo_job_factory(dpo_config, dpo_out)

        # Reference model = base model (production adapter); warm start = SFT adapter
        result = job.run(
            model_path=sft_adapter_path,
            reference_path=config.base_model_id,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )

        log.info(
            "dpo_phase_complete",
            cycle_id=config.cycle_id,
            steps=result.training_steps,
            reward_accuracy=result.reward_accuracy,
        )
        return result

    def _make_result(
        self,
        config: TrainingCycleConfig,
        status: str,
        sft_result: Optional[SFTResult],
        dpo_result: Optional[DPOResult],
        cycle_dir: str,
        error: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        Build an OrchestratorResult from accumulated state.

        Purpose:  centralise result construction to keep run_cycle readable.
        Inputs:   all accumulated phase results and status string
        Outputs:  OrchestratorResult
        Complexity: O(records) for cost lookup
        Side effects: none
        """
        total_cost = self.cost_tracker.total_for_cycle(config.cycle_id)
        total_seconds = (
            (sft_result.duration_seconds if sft_result else 0.0)
            + (dpo_result.duration_seconds if dpo_result else 0.0)
        )
        gpu_hours = total_seconds / 3600.0

        # candidate adapter: prefer DPO output, fall back to SFT, then cycle_dir
        if dpo_result is not None:
            candidate_path = dpo_result.adapter_path
        elif sft_result is not None:
            candidate_path = sft_result.adapter_path
        else:
            candidate_path = cycle_dir

        return OrchestratorResult(
            cycle_id=config.cycle_id,
            status=status,
            sft_result=sft_result,
            dpo_result=dpo_result,
            candidate_adapter_path=candidate_path,
            total_cost_usd=total_cost,
            gpu_hours=gpu_hours,
            error=error,
        )

    def _publish_kafka_event(
        self,
        config: TrainingCycleConfig,
        result: OrchestratorResult,
        started_at: str,
    ) -> None:
        """
        Publish a TrainingCycleEvent to the training.jobs Kafka topic.

        Purpose:  notify downstream services (evaluator, deployer) that a
                  training cycle has completed.
        Inputs:
          config     — cycle configuration
          result     — completed OrchestratorResult
          started_at — ISO-8601 timestamp when the cycle began
        Outputs:  None
        Side effects: produces a Kafka message if self.kafka_producer is set;
                      logs a warning and continues if Kafka is unavailable
        """
        if self.kafka_producer is None:
            return

        try:
            trigger_type = TriggerType(config.triggered_by)
        except ValueError:
            trigger_type = TriggerType.MANUAL

        event = TrainingCycleEvent(
            cycle_id=config.cycle_id,
            triggered_by=trigger_type,
            trigger_event_id=config.trigger_event_id,
            started_at=started_at,
            status=result.status,
            base_adapter_id=config.base_model_id,
            candidate_adapter_id=(
                result.candidate_adapter_path if result.status == "COMPLETED" else None
            ),
            cost_usd=result.total_cost_usd,
            gpu_hours=result.gpu_hours,
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        )

        try:
            self.kafka_producer.produce(
                topic=TOPIC_TRAINING_JOBS,
                key=config.cycle_id.encode("utf-8"),
                value=event.to_json().encode("utf-8"),
            )
            self.kafka_producer.poll(0)
            log.info(
                "kafka_event_published",
                topic=TOPIC_TRAINING_JOBS,
                cycle_id=config.cycle_id,
                status=result.status,
            )
        except Exception as exc:
            # Kafka failure must never crash the orchestrator — training already
            # completed; the event can be re-emitted by a reconciliation job.
            log.error(
                "kafka_publish_failed",
                cycle_id=config.cycle_id,
                error=str(exc),
            )
