from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import time

class AdapterState(str, Enum):
    CANDIDATE = "CANDIDATE"
    STAGING = "STAGING"
    PROMOTING = "PROMOTING"
    PRODUCTION = "PRODUCTION"
    SUPERSEDED = "SUPERSEDED"
    ROLLED_BACK = "ROLLED_BACK"

VALID_TRANSITIONS = {
    AdapterState.CANDIDATE: [AdapterState.STAGING, AdapterState.ROLLED_BACK],
    AdapterState.STAGING: [AdapterState.PROMOTING, AdapterState.ROLLED_BACK],
    AdapterState.PROMOTING: [AdapterState.PRODUCTION, AdapterState.ROLLED_BACK],
    AdapterState.PRODUCTION: [AdapterState.SUPERSEDED, AdapterState.ROLLED_BACK],
    AdapterState.SUPERSEDED: [],
    AdapterState.ROLLED_BACK: [],
}

@dataclass
class AdapterRecord:
    adapter_id: str
    state: AdapterState
    promoted_at: Optional[str] = None
    rolled_back_at: Optional[str] = None
    rollback_reason: Optional[str] = None
    state_history: List[dict] = field(default_factory=list)

class AdapterStateMachine:
    def __init__(self):
        self._records: dict = {}

    def register(self, adapter_id: str) -> AdapterRecord:
        record = AdapterRecord(adapter_id=adapter_id, state=AdapterState.CANDIDATE)
        self._records[adapter_id] = record
        return record

    def transition(self, adapter_id: str, new_state: AdapterState, metadata: dict = None) -> AdapterRecord:
        record = self._records.get(adapter_id)
        if not record:
            raise KeyError(f"Adapter {adapter_id} not registered")
        if new_state not in VALID_TRANSITIONS[record.state]:
            raise ValueError(f"Invalid transition: {record.state} -> {new_state}. Allowed: {VALID_TRANSITIONS[record.state]}")
        record.state_history.append({"from": record.state.value, "to": new_state.value, "timestamp": time.time(), "metadata": metadata or {}})
        record.state = new_state
        return record

    def get(self, adapter_id: str) -> Optional[AdapterRecord]:
        return self._records.get(adapter_id)

    def get_production(self) -> Optional[AdapterRecord]:
        for r in self._records.values():
            if r.state == AdapterState.PRODUCTION:
                return r
        return None
