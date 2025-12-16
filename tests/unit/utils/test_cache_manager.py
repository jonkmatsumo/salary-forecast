"""Tests for cache manager."""

import time
from unittest.mock import patch

from src.utils.cache_manager import CacheManager, get_cache_manager


class TestCacheManager:
    """Test suite for CacheManager."""

    def test_get_set_basic(self) -> None:
        """Test basic get/set operations."""
        manager = CacheManager()
        manager.set("llm", "key1", "value1")
        assert manager.get("llm", "key1") == "value1"
        assert manager.get("llm", "nonexistent") is None

    def test_clear_specific_cache(self) -> None:
        """Test clearing a specific cache."""
        manager = CacheManager()
        manager.set("llm", "key1", "value1")
        manager.set("preprocessing", "key2", "value2")
        manager.clear("llm")
        assert manager.get("llm", "key1") is None
        assert manager.get("preprocessing", "key2") == "value2"

    def test_clear_all_caches(self) -> None:
        """Test clearing all caches."""
        manager = CacheManager()
        manager.set("llm", "key1", "value1")
        manager.set("preprocessing", "key2", "value2")
        manager.clear()
        assert manager.get("llm", "key1") is None
        assert manager.get("preprocessing", "key2") is None

    def test_invalidate_specific_key(self) -> None:
        """Test invalidating a specific key."""
        manager = CacheManager()
        manager.set("llm", "key1", "value1")
        manager.set("llm", "key2", "value2")
        manager.invalidate("llm", "key1")
        assert manager.get("llm", "key1") is None
        assert manager.get("llm", "key2") == "value2"

    def test_get_stats(self) -> None:
        """Test getting cache statistics."""
        manager = CacheManager()
        manager.set("llm", "key1", "value1")
        stats = manager.get_stats()
        assert "llm" in stats
        assert stats["llm"]["size"] == 1
        assert stats["llm"]["maxsize"] > 0
        assert stats["llm"]["ttl"] > 0

    def test_unknown_cache_type(self) -> None:
        """Test handling of unknown cache type."""
        manager = CacheManager()
        assert manager.get("unknown", "key") is None
        manager.set("unknown", "key", "value")

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration of cache entries."""
        with patch.dict("os.environ", {"CACHE_LLM_TTL": "1"}):
            manager = CacheManager()
            manager.set("llm", "key1", "value1")
            assert manager.get("llm", "key1") == "value1"
            time.sleep(2)
            assert manager.get("llm", "key1") is None

    def test_size_limit_eviction(self) -> None:
        """Test size limit eviction (LRU)."""
        with patch.dict("os.environ", {"CACHE_LLM_SIZE": "2"}):
            manager = CacheManager()
            manager.set("llm", "key1", "value1")
            manager.set("llm", "key2", "value2")
            manager.set("llm", "key3", "value3")
            stats = manager.get_stats()
            assert stats["llm"]["size"] <= 2

    def test_thread_safety(self) -> None:
        """Test thread safety of cache operations."""
        import threading

        manager = CacheManager()
        results: list = []

        def worker(thread_id: int) -> None:
            for i in range(10):
                key = f"key_{thread_id}_{i}"
                manager.set("llm", key, f"value_{thread_id}_{i}")
                result = manager.get("llm", key)
                results.append(result is not None)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)

    def test_get_cache_manager_singleton(self) -> None:
        """Test that get_cache_manager returns singleton instance."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        assert manager1 is manager2

    def test_all_cache_types(self) -> None:
        """Test all cache types are available."""
        manager = CacheManager()
        cache_types = ["llm", "preprocessing", "correlation", "geo"]
        for cache_type in cache_types:
            manager.set(cache_type, "test_key", "test_value")
            assert manager.get(cache_type, "test_key") == "test_value"
