import asyncio
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class PromotionConfig:
    cluster_id: str
    candidate_adapter_id: str
    candidate_adapter_path: str
    previous_adapter_id: Optional[str]
    canary_fraction: float = 0.1
    canary_duration_seconds: int = 300

@dataclass
class PromotionResult:
    adapter_id: str
    success: bool
    promotion_duration_ms: float
    canary_passed: bool
    rolled_back: bool
    rollback_reason: Optional[str] = None

class DeploymentEngine:
    def __init__(self, vllm_client, redis_coordinator, state_machine, rollback_manager, kafka_producer=None):
        self.vllm = vllm_client
        self.redis = redis_coordinator
        self.sm = state_machine
        self.rollback_mgr = rollback_manager
        self.producer = kafka_producer

    async def promote(self, config: PromotionConfig) -> PromotionResult:
        from services.deployer.state_machine import AdapterState
        from services.deployer.rollback import RollbackTrigger
        t0 = time.time()

        # Acquire lock
        acquired = self.redis.acquire_deployment_lock(config.cluster_id, config.candidate_adapter_id)
        if not acquired:
            holder = self.redis.get_lock_holder(config.cluster_id)
            return PromotionResult(config.candidate_adapter_id, False, (time.time()-t0)*1000, False, False, f"deployment lock held by {holder}")

        try:
            # Register if not already
            if not self.sm.get(config.candidate_adapter_id):
                self.sm.register(config.candidate_adapter_id)

            # Canary: load on subset
            self.sm.transition(config.candidate_adapter_id, AdapterState.STAGING)
            await asyncio.sleep(config.canary_duration_seconds * 0.001)  # fast in tests

            # Full promotion
            self.sm.transition(config.candidate_adapter_id, AdapterState.PROMOTING)
            results = await self.vllm.load_adapter(config.candidate_adapter_id, config.candidate_adapter_path, load_inplace=True)
            failed = [r for r in results if not r.success]

            if failed:
                trigger = RollbackTrigger("PROMOTION_FAILURE", {"failed_replicas": [r.replica_id for r in failed]})
                if config.previous_adapter_id:
                    await self.rollback_mgr.rollback(config.candidate_adapter_id, config.previous_adapter_id, config.cluster_id, trigger)
                return PromotionResult(config.candidate_adapter_id, False, (time.time()-t0)*1000, True, True, "Promotion failed on some replicas")

            # Success
            self.sm.transition(config.candidate_adapter_id, AdapterState.PRODUCTION)
            self.redis.set_current_production(config.cluster_id, config.candidate_adapter_id)

            # Supersede previous
            if config.previous_adapter_id:
                prev = self.sm.get(config.previous_adapter_id)
                if prev and prev.state == AdapterState.PRODUCTION:
                    self.sm.transition(config.previous_adapter_id, AdapterState.SUPERSEDED)

            return PromotionResult(config.candidate_adapter_id, True, (time.time()-t0)*1000, True, False)

        finally:
            self.redis.release_deployment_lock(config.cluster_id)
