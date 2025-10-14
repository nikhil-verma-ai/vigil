from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class RollbackTrigger:
    reason: str
    details: dict

@dataclass
class RollbackResult:
    success: bool
    duration_ms: float
    replicas_reverted: int
    error: Optional[str] = None

class RollbackManager:
    def __init__(self, vllm_client, redis_coordinator, state_machine):
        self.vllm = vllm_client
        self.redis = redis_coordinator
        self.sm = state_machine

    async def rollback(self, candidate_id: str, previous_id: str, cluster_id: str, trigger: RollbackTrigger) -> RollbackResult:
        t0 = time.time()
        try:
            results = await self.vllm.load_adapter(previous_id, f"/adapters/{previous_id}", load_inplace=True)
            self.redis.set_current_production(cluster_id, previous_id)
            record = self.sm.get(candidate_id)
            if record and record.state.value not in ("ROLLED_BACK",):
                try:
                    from services.deployer.state_machine import AdapterState
                    self.sm.transition(candidate_id, AdapterState.ROLLED_BACK, {"reason": trigger.reason})
                except Exception:
                    pass
            successes = sum(1 for r in results if r.success)
            return RollbackResult(True, (time.time()-t0)*1000, successes)
        except Exception as e:
            return RollbackResult(False, (time.time()-t0)*1000, 0, str(e))

    def should_trigger_on_canary_anomaly(self, canary_score: float, production_score: float, threshold_sigma: float = 2.0) -> bool:
        return (canary_score - production_score) > threshold_sigma
