"""
Autonomous Loop Orchestrator — Module 8.

Wires together every upstream service into a single closed training loop:

    DriftTrigger / ScheduleTrigger
        → LoopStateMachine (IDLE → EVIDENCE_QUALIFYING → SYNTHESIZING
                            → TRAINING_SFT → TRAINING_DPO → EVALUATING
                            → DEPLOYING_CANARY → PROMOTING_FULL → IDLE)
        → Kafka event emission at each state boundary

Public API:
  - state_machine.LoopStateMachine
  - state_machine.LoopState
  - state_machine.LoopExecution
  - state_machine.LoopStateMachineConfig
  - triggers.DriftTrigger
  - triggers.ScheduleTrigger
  - triggers.TriggerEvent
  - loop.AutonomousLoop
  - loop.LoopConfig
"""

from services.orchestrator.state_machine import (
    LoopStateMachine,
    LoopState,
    LoopExecution,
    LoopStateMachineConfig,
    VALID_TRANSITIONS,
)
from services.orchestrator.triggers import (
    DriftTrigger,
    ScheduleTrigger,
    TriggerEvent,
)
from services.orchestrator.loop import AutonomousLoop, LoopConfig

__all__ = [
    "LoopStateMachine",
    "LoopState",
    "LoopExecution",
    "LoopStateMachineConfig",
    "VALID_TRANSITIONS",
    "DriftTrigger",
    "ScheduleTrigger",
    "TriggerEvent",
    "AutonomousLoop",
    "LoopConfig",
]
