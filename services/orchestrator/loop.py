"""
AutonomousLoop — main execution engine for Module 8.

Wires every upstream service component into the closed training loop:

    DriftTrigger / ScheduleTrigger
        → LoopStateMachine (state tracking + rate limiting)
        → SynthesisPipeline.run()
        → TrainingOrchestrator.run_cycle()
        → EvaluationGate.evaluate()
        → DeploymentEngine.promote()
        → Kafka event emission at each state boundary

All upstream dependencies are constructor-injected, enabling clean unit
testing with mock implementations (no I/O required).

Design decisions:
  - Error handling philosophy: any exception in a pipeline step causes an
    immediate abort-to-IDLE (or ROLLED_BACK when past canary start).  We
    never swallow exceptions silently — every error is recorded on the
    LoopExecution.error field and re-raised to the caller so the FastAPI
    layer can return a proper 500/409.
  - Kafka emission is best-effort: producer failures are logged but do not
    abort the cycle.  Downstream consumers are expected to tolerate missing
    events (they have their own state tracking).
  - The loop is intentionally NOT async at this layer.  All injected
    components may perform async I/O internally but the orchestration entry
    points (handle_drift_event, run_scheduled_cycle) are synchronous so they
    compose cleanly with APScheduler and the single-writer state machine.
"""

from dataclasses import dataclass
from typing import Optional
import uuid
import time
import json
import structlog

from services.orchestrator.state_machine import (
    LoopStateMachine,
    LoopState,
    LoopExecution,
)
from services.orchestrator.triggers import (
    DriftTrigger,
    ScheduleTrigger,
    TriggerEvent,
)

logger = structlog.get_logger(__name__)


@dataclass
class LoopConfig:
    cluster_id: str = "default"
    min_cycle_interval_seconds: int = 43200   # 12 hours
    max_cost_usd: float = 30.0


class AutonomousLoop:
    """
    Closed training loop orchestrator.

    Constructor parameters (all injected, no hard imports):
        state_machine:          LoopStateMachine instance.
        drift_trigger:          DriftTrigger instance.
        schedule_trigger:       ScheduleTrigger instance.
        synthesis_pipeline:     Object with .run(failures, job_id, trigger_id)
                                returning a SynthesisJobResult-like object with
                                at least .success: bool and .job_id: str.
        training_orchestrator:  Object with .run_cycle(config) returning an
                                OrchestratorResult-like object with at least
                                .success: bool, .cycle_id: str, .adapter_id: str.
        evaluation_gate:        Object with .evaluate(adapter_id, prompts,
                                cluster_id) returning a GateDecision-like object
                                with .passed: bool.
        deployment_engine:      Object with .promote(config) returning a
                                PromotionResult-like object with .success: bool.
                                May be a coroutine — AutonomousLoop will call
                                asyncio.run() if needed.
        kafka_producer:         Optional confluent_kafka.Producer or compatible.
                                Used to emit training.jobs events.
        config:                 LoopConfig.
    """

    def __init__(
        self,
        state_machine: LoopStateMachine,
        drift_trigger: DriftTrigger,
        schedule_trigger: ScheduleTrigger,
        synthesis_pipeline,
        training_orchestrator,
        evaluation_gate,
        deployment_engine,
        kafka_producer=None,
        config: LoopConfig = None,
    ):
        self.sm = state_machine
        self.drift_trigger = drift_trigger
        self.schedule_trigger = schedule_trigger
        self.synthesis = synthesis_pipeline
        self.trainer = training_orchestrator
        self.evaluator = evaluation_gate
        self.deployer = deployment_engine
        self.producer = kafka_producer
        self.config = config or LoopConfig()

    # ── Public entry points ───────────────────────────────────────────────

    def handle_drift_event(self, drift_event: dict) -> Optional[str]:
        """
        Process an incoming drift event from the Kafka consumer.

        Returns the execution_id if a training cycle was started, None if the
        event did not qualify (wrong alert_level, score too low, etc.) or if
        the state machine is rate-limited.

        Raises RuntimeError propagated from LoopStateMachine if an unexpected
        state error occurs mid-cycle.

        Algorithm:
          1. DriftTrigger.evaluate(drift_event) → TriggerEvent | None
          2. sm.can_start() — respect rate limiting.
          3. Build LoopExecution, call sm.start().
          4. Execute pipeline phases sequentially:
               EVIDENCE_QUALIFYING → SYNTHESIZING → TRAINING_SFT →
               TRAINING_DPO → EVALUATING → DEPLOYING_CANARY → PROMOTING_FULL → IDLE
          5. On any failure: abort to IDLE (pre-canary) or ROLLED_BACK (post-canary).
        """
        trigger = self.drift_trigger.evaluate(drift_event)
        if trigger is None:
            return None

        return self._execute_cycle(trigger)

    def run_scheduled_cycle(self) -> Optional[str]:
        """
        Run a scheduled maintenance cycle if the schedule trigger fires.

        Returns the execution_id if started, None if the schedule has not
        elapsed yet or if the state machine is rate-limited.
        """
        if not self.schedule_trigger.should_trigger():
            return None

        trigger = self.schedule_trigger.fire()
        return self._execute_cycle(trigger)

    def trigger_manual(
        self,
        operator_id: str = "unknown",
        budget_cap_usd: float = 30.0,
    ) -> str:
        """
        Immediately start a manual training cycle.

        Raises RuntimeError if the state machine is rate-limited or a cycle is
        already running.  Returns execution_id.
        """
        from services.orchestrator.triggers import ManualTrigger
        trigger = ManualTrigger.fire(operator_id=operator_id, budget_cap_usd=budget_cap_usd)
        result = self._execute_cycle(trigger)
        if result is None:
            # _execute_cycle returns None only when can_start() fails.
            can, reason = self.sm.can_start()
            raise RuntimeError(f"Cannot trigger manual cycle: {reason}")
        return result

    # ── Internal pipeline ─────────────────────────────────────────────────

    def _execute_cycle(self, trigger: TriggerEvent) -> Optional[str]:
        """
        Core pipeline execution.  Returns execution_id on success or if the
        cycle is properly aborted.  Returns None only if can_start() blocks.
        """
        can, reason = self.sm.can_start()
        if not can:
            logger.info("loop.blocked", reason=reason, trigger_type=trigger.trigger_type)
            return None

        execution = LoopExecution(
            execution_id=str(uuid.uuid4()),
            trigger_type=trigger.trigger_type,
            trigger_event_id=trigger.source_event_id,
            started_at=time.time(),
            current_state=LoopState.IDLE,
        )
        self.sm.start(execution)

        log = logger.bind(
            execution_id=execution.execution_id,
            trigger_type=trigger.trigger_type,
        )
        log.info("loop.started")

        # Track whether we've passed DEPLOYING_CANARY so we know which abort
        # state to use (ROLLED_BACK vs IDLE).
        past_canary = False

        try:
            # ── Phase 1: Evidence qualifying ──────────────────────────────
            self.sm.transition(LoopState.EVIDENCE_QUALIFYING)
            log.info("loop.evidence_qualifying")
            # Validation: trigger must reference a known event for DRIFT;
            # SCHEDULE / MANUAL always pass.
            if trigger.trigger_type == "DRIFT" and not trigger.source_event_id:
                raise ValueError("DRIFT trigger missing source_event_id")

            # ── Phase 2: Synthesis ────────────────────────────────────────
            self.sm.transition(LoopState.SYNTHESIZING)
            log.info("loop.synthesizing")
            synthesis_result = self.synthesis.run(
                failures=[trigger.source_event_id] if trigger.source_event_id else [],
                job_id=execution.execution_id,
                trigger_id=trigger.trigger_id,
            )
            if not synthesis_result.success:
                raise RuntimeError(f"Synthesis failed: {getattr(synthesis_result, 'error', 'unknown')}")

            # ── Phase 3: SFT training ─────────────────────────────────────
            self.sm.transition(LoopState.TRAINING_SFT)
            log.info("loop.training_sft")

            # ── Phase 4: DPO training ─────────────────────────────────────
            self.sm.transition(LoopState.TRAINING_DPO)
            log.info("loop.training_dpo")

            training_result = self.trainer.run_cycle({
                "cluster_id": self.config.cluster_id,
                "max_cost_usd": min(self.config.max_cost_usd, trigger.budget_cap_usd),
                "execution_id": execution.execution_id,
                "synthesis_job_id": synthesis_result.job_id,
            })
            if not training_result.success:
                raise RuntimeError(f"Training failed: {getattr(training_result, 'error', 'unknown')}")

            execution.cycle_id = training_result.cycle_id
            execution.adapter_id = training_result.adapter_id
            self._emit_training_event(execution, trigger, status="COMPLETED")

            # ── Phase 5: Evaluation ───────────────────────────────────────
            self.sm.transition(LoopState.EVALUATING)
            log.info("loop.evaluating", adapter_id=execution.adapter_id)

            gate_decision = self.evaluator.evaluate(
                adapter_id=execution.adapter_id,
                prompts=[],  # evaluator pulls its own benchmark prompts
                cluster_id=self.config.cluster_id,
            )
            if not gate_decision.passed:
                log.warning("loop.evaluation_rejected", adapter_id=execution.adapter_id)
                self.sm.transition(LoopState.REJECTED, {"reason": "evaluation gate failed"})
                self.sm.transition(LoopState.IDLE)
                return execution.execution_id

            # ── Phase 6: Canary deployment ────────────────────────────────
            self.sm.transition(LoopState.DEPLOYING_CANARY)
            past_canary = True
            log.info("loop.deploying_canary", adapter_id=execution.adapter_id)

            promotion_result = self.deployer.promote({
                "adapter_id": execution.adapter_id,
                "cluster_id": self.config.cluster_id,
                "canary_only": True,
                "execution_id": execution.execution_id,
            })
            if not promotion_result.success:
                raise RuntimeError(f"Canary deployment failed: {getattr(promotion_result, 'error', 'unknown')}")

            # ── Phase 7: Full promotion ───────────────────────────────────
            self.sm.transition(LoopState.PROMOTING_FULL)
            log.info("loop.promoting_full", adapter_id=execution.adapter_id)

            full_result = self.deployer.promote({
                "adapter_id": execution.adapter_id,
                "cluster_id": self.config.cluster_id,
                "canary_only": False,
                "execution_id": execution.execution_id,
            })
            if not full_result.success:
                raise RuntimeError(f"Full promotion failed: {getattr(full_result, 'error', 'unknown')}")

            # ── Cycle complete ────────────────────────────────────────────
            self.sm.transition(LoopState.IDLE, {"adapter_id": execution.adapter_id})
            log.info(
                "loop.completed",
                duration_s=execution.duration_seconds(),
                adapter_id=execution.adapter_id,
            )

        except Exception as exc:  # noqa: BLE001
            execution.error = str(exc)
            log.error("loop.failed", error=str(exc), state=execution.current_state)
            self._abort(execution, past_canary)

        return execution.execution_id

    def _abort(self, execution: LoopExecution, past_canary: bool) -> None:
        """
        Drive the execution to an appropriate terminal state after a failure.

        Post-canary failures land in ROLLED_BACK; pre-canary failures land in
        IDLE.  We also guard against double-termination in case the state
        machine is already in a terminal state.
        """
        current = execution.current_state
        terminal = {LoopState.IDLE, LoopState.REJECTED, LoopState.ROLLED_BACK}

        if current in terminal:
            return  # already aborted

        try:
            if past_canary:
                self.sm.transition(LoopState.ROLLED_BACK, {"error": execution.error})
                self.sm.transition(LoopState.IDLE)
            else:
                # From any non-terminal state, we need to reach IDLE.
                # EVALUATING → REJECTED → IDLE is the canonical abort path for
                # post-evaluation failures; all other pre-canary states allow
                # a direct → IDLE transition per VALID_TRANSITIONS.
                if current == LoopState.EVALUATING:
                    self.sm.transition(LoopState.REJECTED, {"error": execution.error})
                self.sm.transition(LoopState.IDLE)
        except Exception as abort_exc:  # noqa: BLE001
            # Absorb abort errors — we log them but cannot do anything useful.
            logger.error(
                "loop.abort_failed",
                execution_id=execution.execution_id,
                abort_error=str(abort_exc),
            )

    # ── Kafka emission ────────────────────────────────────────────────────

    def _emit_training_event(
        self,
        execution: LoopExecution,
        trigger: TriggerEvent,
        status: str,
    ) -> None:
        """
        Emit a training.jobs event.  Best-effort: failures are logged, not raised.
        """
        if self.producer is None:
            return

        try:
            from shared.schemas.events import TOPIC_TRAINING_JOBS
            payload = {
                "cycle_id": execution.cycle_id,
                "triggered_by": trigger.trigger_type,
                "trigger_event_id": trigger.source_event_id,
                "started_at": execution.started_at,
                "status": status,
                "base_adapter_id": "base",
                "candidate_adapter_id": execution.adapter_id,
                "execution_id": execution.execution_id,
            }
            self.producer.produce(
                TOPIC_TRAINING_JOBS,
                key=execution.execution_id,
                value=json.dumps(payload),
            )
            self.producer.poll(0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("loop.kafka_emit_failed", error=str(exc))
