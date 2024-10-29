from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from joblib import Memory


class TaskCache:
    _instance: Optional[TaskCache] = None

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.memory = Memory(self.cache_dir, verbose=0)
        self.enabled = True

    @classmethod
    def get_instance(cls) -> Optional[TaskCache]:
        return cls._instance

    @classmethod
    @contextmanager
    def enable(cls, cache_dir: Path):
        previous_instance = cls._instance
        try:
            cls._instance = TaskCache(cache_dir)
            yield cls._instance
        finally:
            cls._instance = previous_instance

    @classmethod
    def cached(cls, func):
        def wrapper(*args, **kwargs):
            cache_instance = cls.get_instance()

            if cache_instance and cache_instance.enabled:
                cached_func = cache_instance.memory.cache(func)
                return cached_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper
