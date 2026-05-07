import json
import os
from datetime import datetime, timedelta
from typing import Any

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None


class _MemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[datetime | None, str]] = {}

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if not entry:
            return None
        expire_at, value = entry
        if expire_at and datetime.utcnow() > expire_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        expire_at = datetime.utcnow() + timedelta(seconds=ex) if ex else None
        self._store[key] = (expire_at, value)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def delete_by_prefix(self, prefix: str) -> int:
        keys = [k for k in self._store.keys() if k.startswith(prefix)]
        for key in keys:
            self._store.pop(key, None)
        return len(keys)


class CacheClient:
    def __init__(self) -> None:
        self._memory = _MemoryCache()
        self._redis = None
        redis_url = os.getenv("REDIS_URL")
        if redis and redis_url:
            try:
                client = redis.Redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self._redis = client
            except Exception:
                self._redis = None

    def get_json(self, key: str) -> Any | None:
        raw = self._get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ex: int | None = None) -> None:
        self._set(key, json.dumps(value, ensure_ascii=False), ex=ex)

    def delete(self, key: str) -> None:
        if self._redis is not None:
            self._redis.delete(key)
        else:
            self._memory.delete(key)

    def delete_by_prefix(self, prefix: str) -> int:
        if self._redis is not None:
            deleted = 0
            for key in self._redis.scan_iter(match=f"{prefix}*"):
                deleted += int(self._redis.delete(key) or 0)
            return deleted
        return self._memory.delete_by_prefix(prefix)

    def _get(self, key: str) -> str | None:
        if self._redis is not None:
            return self._redis.get(key)
        return self._memory.get(key)

    def _set(self, key: str, value: str, ex: int | None = None) -> None:
        if self._redis is not None:
            self._redis.set(name=key, value=value, ex=ex)
        else:
            self._memory.set(key, value, ex=ex)


cache_client = CacheClient()
