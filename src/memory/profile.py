from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import redis

from .base import MemoryBackend
from ..storage import read_json, write_json


class ProfileMemory(MemoryBackend):
    """Long-term profile memory with Redis-first behavior and file fallback."""

    name = "profile"

    def __init__(
        self,
        storage_path: str | Path = "data/profile_memory.json",
        redis_url: str | None = None,
    ) -> None:
        self.storage_path = Path(storage_path)
        self._profiles: dict[str, dict[str, Any]] = read_json(self.storage_path, {})
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis = self._connect()

    def _connect(self) -> redis.Redis | None:
        try:
            client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            client.ping()
            return client
        except redis.RedisError:
            return None

    def retrieve(self, query: str, *, user_id: str | None = None) -> dict[str, Any]:
        if not user_id:
            return {}
        merged = dict(self._profiles.get(user_id, {}))
        if self._redis is not None:
            try:
                merged.update(dict(self._redis.hgetall(self._redis_key(user_id))))
            except redis.RedisError:
                pass
        return merged

    def save(self, payload: Any, *, user_id: str | None = None) -> None:
        if not user_id or not isinstance(payload, dict):
            return
        existing = self._profiles.setdefault(user_id, {})
        existing.update(payload)
        write_json(self.storage_path, self._profiles)
        if self._redis is not None:
            try:
                self._redis.hset(self._redis_key(user_id), mapping={key: str(value) for key, value in payload.items()})
            except redis.RedisError:
                pass

    def _redis_key(self, user_id: str) -> str:
        return f"profile:{user_id}"
