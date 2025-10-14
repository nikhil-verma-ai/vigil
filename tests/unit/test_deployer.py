import asyncio, pytest
from services.deployer.state_machine import AdapterStateMachine, AdapterState
from services.deployer.redis_coordinator import RedisCoordinator, MockRedisClient
from services.deployer.vllm_client import MockVLLMClient
from services.deployer.rollback import RollbackManager, RollbackTrigger

def test_state_machine_valid_transition():
    sm = AdapterStateMachine()
    sm.register("a1")
    sm.transition("a1", AdapterState.STAGING)
    assert sm.get("a1").state == AdapterState.STAGING

def test_state_machine_invalid_transition_raises():
    sm = AdapterStateMachine()
    sm.register("a1")
    with pytest.raises(ValueError, match="Invalid transition"):
        sm.transition("a1", AdapterState.PRODUCTION)

def test_state_machine_history_recorded():
    sm = AdapterStateMachine()
    sm.register("a1")
    sm.transition("a1", AdapterState.STAGING)
    sm.transition("a1", AdapterState.PROMOTING)
    h = sm.get("a1").state_history
    assert len(h) == 2
    assert h[0]["from"] == "CANDIDATE" and h[0]["to"] == "STAGING"
    assert h[1]["from"] == "STAGING" and h[1]["to"] == "PROMOTING"

def test_redis_lock_acquisition_exclusive():
    coord = RedisCoordinator(MockRedisClient())
    assert coord.acquire_deployment_lock("c1", "a1") == True
    assert coord.acquire_deployment_lock("c1", "a2") == False
    assert coord.get_lock_holder("c1") == "a1"

def test_redis_lock_release():
    coord = RedisCoordinator(MockRedisClient())
    coord.acquire_deployment_lock("c1", "a1")
    coord.release_deployment_lock("c1")
    assert coord.get_lock_holder("c1") is None
    assert coord.acquire_deployment_lock("c1", "a2") == True

def test_redis_production_tracking():
    coord = RedisCoordinator(MockRedisClient())
    coord.set_current_production("c1", "adapter-v3")
    assert coord.get_current_production("c1") == "adapter-v3"

def test_mock_vllm_loads_all_replicas():
    client = MockVLLMClient(replica_count=4)
    results = asyncio.run(client.load_adapter("adapter-v2", "/tmp/v2"))
    assert len(results) == 4
    assert all(r.success for r in results)
    for i in range(4):
        assert client.get_active_adapter(f"replica-{i}") == "adapter-v2"

def test_mock_vllm_partial_failure():
    client = MockVLLMClient(replica_count=4, fail_replicas=[2])
    results = asyncio.run(client.load_adapter("v2", "/tmp/v2"))
    assert sum(1 for r in results if r.success) == 3
    assert sum(1 for r in results if not r.success) == 1

def test_rollback_trigger_anomaly_threshold():
    rm = RollbackManager(None, None, None)
    assert rm.should_trigger_on_canary_anomaly(5.0, 1.0, 2.0) == True
    assert rm.should_trigger_on_canary_anomaly(1.5, 1.0, 2.0) == False
