import asyncio
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoadAdapterResult:
    replica_id: str
    success: bool
    latency_ms: float
    error: Optional[str] = None

class MockVLLMClient:
    def __init__(self, replica_count: int = 4, simulate_latency_ms: float = 10.0, fail_replicas: List[int] = None):
        self.replica_count = replica_count
        self.simulate_latency_ms = simulate_latency_ms
        self.fail_replicas = fail_replicas or []
        self.loaded_adapters: dict = {}
        self.load_call_count = 0

    async def load_adapter(self, adapter_id: str, adapter_path: str, load_inplace: bool = True) -> List[LoadAdapterResult]:
        self.load_call_count += 1
        tasks = [self._load_one(i, adapter_id) for i in range(self.replica_count)]
        return await asyncio.gather(*tasks)

    async def _load_one(self, i: int, adapter_id: str) -> LoadAdapterResult:
        await asyncio.sleep(self.simulate_latency_ms / 1000.0)
        if i in self.fail_replicas:
            return LoadAdapterResult(f"replica-{i}", False, self.simulate_latency_ms, "Simulated failure")
        self.loaded_adapters[f"replica-{i}"] = adapter_id
        return LoadAdapterResult(f"replica-{i}", True, self.simulate_latency_ms)

    def get_active_adapter(self, replica_id: str) -> Optional[str]:
        return self.loaded_adapters.get(replica_id)
