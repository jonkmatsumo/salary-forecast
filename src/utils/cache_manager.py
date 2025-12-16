"""In-memory cache manager with TTL support, size limits, and eviction policies."""

import threading
from typing import Any, Dict, Optional, cast

from cachetools import TTLCache

from src.utils.env_loader import get_env_var
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Centralized cache manager with TTL and size limits.

    Provides separate caches for different data types with configurable
    TTL and size limits. Thread-safe operations.
    """

    def __init__(self) -> None:
        """Initialize CacheManager with default or environment-configured caches."""
        self._lock = threading.RLock()

        disable_cache_val = get_env_var("DISABLE_CACHE", "")
        disable_cache = cast(str, disable_cache_val).lower() in ("1", "true", "yes")
        self._enabled = not disable_cache

        llm_size = int(cast(str, get_env_var("CACHE_LLM_SIZE", "100")))
        llm_ttl = int(cast(str, get_env_var("CACHE_LLM_TTL", "3600")))
        preprocessing_size = int(cast(str, get_env_var("CACHE_PREPROCESSING_SIZE", "200")))
        preprocessing_ttl = int(cast(str, get_env_var("CACHE_PREPROCESSING_TTL", "1800")))
        correlation_size = int(cast(str, get_env_var("CACHE_CORRELATION_SIZE", "50")))
        correlation_ttl = int(cast(str, get_env_var("CACHE_CORRELATION_TTL", "3600")))
        geo_size = int(cast(str, get_env_var("CACHE_GEO_SIZE", "500")))
        geo_ttl = int(cast(str, get_env_var("CACHE_GEO_TTL", "86400")))

        self._caches: Dict[str, TTLCache[str, Any]] = {
            "llm": TTLCache(maxsize=llm_size, ttl=llm_ttl),
            "preprocessing": TTLCache(maxsize=preprocessing_size, ttl=preprocessing_ttl),
            "correlation": TTLCache(maxsize=correlation_size, ttl=correlation_ttl),
            "geo": TTLCache(maxsize=geo_size, ttl=geo_ttl),
        }

    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            cache_type (str): Cache type ("llm", "preprocessing", "correlation", "geo").
            key (str): Cache key.

        Returns:
            Optional[Any]: Cached value or None if not found/expired.
        """
        if not self._enabled:
            return None
        with self._lock:
            cache = self._caches.get(cache_type)
            if cache is None:
                logger.warning(f"Unknown cache type: {cache_type}")
                return None
            return cache.get(key)

    def set(self, cache_type: str, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            cache_type (str): Cache type ("llm", "preprocessing", "correlation", "geo").
            key (str): Cache key.
            value (Any): Value to cache.
        """
        if not self._enabled:
            return
        with self._lock:
            cache = self._caches.get(cache_type)
            if cache is None:
                logger.warning(f"Unknown cache type: {cache_type}")
                return
            cache[key] = value

    def clear(self, cache_type: Optional[str] = None) -> None:
        """Clear cache entries.

        Args:
            cache_type (Optional[str]): Cache type to clear. If None, clears all caches.
        """
        with self._lock:
            if cache_type:
                cache = self._caches.get(cache_type)
                if cache:
                    cache.clear()
            else:
                for cache in self._caches.values():
                    cache.clear()

    def invalidate(self, cache_type: str, key: str) -> None:
        """Invalidate specific cache entry.

        Args:
            cache_type (str): Cache type.
            key (str): Cache key to invalidate.
        """
        with self._lock:
            cache = self._caches.get(cache_type)
            if cache and key in cache:
                del cache[key]

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get cache statistics.

        Returns:
            Dict[str, Dict[str, Any]]: Statistics for each cache type.
        """
        with self._lock:
            stats: Dict[str, Dict[str, Any]] = {}
            for cache_type, cache in self._caches.items():
                stats[cache_type] = {
                    "size": len(cache),
                    "maxsize": cache.maxsize,
                    "ttl": cache.ttl,
                }
            return stats


_global_cache_manager: Optional[CacheManager] = None
_manager_lock = threading.Lock()


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance (singleton).

    Returns:
        CacheManager: Global cache manager instance.
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        with _manager_lock:
            if _global_cache_manager is None:
                _global_cache_manager = CacheManager()
    return _global_cache_manager
