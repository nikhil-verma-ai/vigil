class MockRedisClient:
    def __init__(self):
        self._store = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self._store:
            return None
        self._store[key] = str(value).encode() if isinstance(value, str) else value
        return True

    def get(self, key):
        return self._store.get(key)

    def delete(self, key):
        self._store.pop(key, None)

LOCK_KEY = "deployment-lock:{cluster_id}"
PROD_KEY = "current-production:{cluster_id}"
LOCK_TTL = 300

class RedisCoordinator:
    def __init__(self, redis_client):
        self.redis = redis_client

    def acquire_deployment_lock(self, cluster_id: str, adapter_id: str) -> bool:
        key = LOCK_KEY.format(cluster_id=cluster_id)
        result = self.redis.set(key, adapter_id, nx=True, ex=LOCK_TTL)
        return result is True

    def release_deployment_lock(self, cluster_id: str):
        self.redis.delete(LOCK_KEY.format(cluster_id=cluster_id))

    def get_lock_holder(self, cluster_id: str):
        v = self.redis.get(LOCK_KEY.format(cluster_id=cluster_id))
        return v.decode() if v else None

    def set_current_production(self, cluster_id: str, adapter_id: str):
        self.redis.set(PROD_KEY.format(cluster_id=cluster_id), adapter_id)

    def get_current_production(self, cluster_id: str):
        v = self.redis.get(PROD_KEY.format(cluster_id=cluster_id))
        return v.decode() if v else None
