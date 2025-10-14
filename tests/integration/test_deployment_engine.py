import asyncio, pytest
from services.deployer.state_machine import AdapterStateMachine, AdapterState
from services.deployer.redis_coordinator import RedisCoordinator, MockRedisClient
from services.deployer.vllm_client import MockVLLMClient
from services.deployer.rollback import RollbackManager
from services.deployer.engine import DeploymentEngine, PromotionConfig

def test_full_promotion_happy_path():
    mock_redis = MockRedisClient()
    coord = RedisCoordinator(mock_redis)
    sm = AdapterStateMachine()
    sm.register("adapter-v1")
    sm.transition("adapter-v1", AdapterState.STAGING)
    sm.transition("adapter-v1", AdapterState.PROMOTING)
    sm.transition("adapter-v1", AdapterState.PRODUCTION)
    coord.set_current_production("cluster-1", "adapter-v1")

    vllm = MockVLLMClient(replica_count=4, simulate_latency_ms=5.0)
    rollback_mgr = RollbackManager(vllm, coord, sm)
    engine = DeploymentEngine(vllm, coord, sm, rollback_mgr)

    config = PromotionConfig("cluster-1", "adapter-v2", "/tmp/v2", "adapter-v1", canary_duration_seconds=0)
    result = asyncio.run(engine.promote(config))

    assert result.success == True
    assert result.rolled_back == False
    assert sm.get("adapter-v2").state == AdapterState.PRODUCTION
    assert coord.get_current_production("cluster-1") == "adapter-v2"
    assert coord.get_lock_holder("cluster-1") is None

def test_concurrent_promotion_blocked():
    mock_redis = MockRedisClient()
    coord = RedisCoordinator(mock_redis)
    coord.acquire_deployment_lock("cluster-1", "adapter-v2")
    sm = AdapterStateMachine()
    engine = DeploymentEngine(MockVLLMClient(), coord, sm, None)
    config = PromotionConfig("cluster-1", "adapter-v3", "/tmp/v3", "adapter-v2", canary_duration_seconds=0)
    result = asyncio.run(engine.promote(config))
    assert result.success == False
    assert "lock" in (result.rollback_reason or "").lower()
