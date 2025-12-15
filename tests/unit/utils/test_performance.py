"""Unit tests for performance monitoring utilities."""

import time
import unittest

from src.utils.performance import (
    LLMCallTracker,
    PerformanceMetrics,
    clear_metrics,
    estimate_llm_cost,
    export_metrics_to_csv,
    export_metrics_to_json,
    format_cost,
    generate_performance_summary,
    get_all_metrics,
    get_global_llm_tracker,
    get_llm_metrics_summary,
    get_metric_stats,
    set_global_llm_tracker,
    timing_decorator,
)


class TestTimingDecorator(unittest.TestCase):
    """Tests for timing_decorator."""

    def setUp(self):
        clear_metrics()

    def test_timing_decorator_sync(self):
        """Test timing decorator on sync function."""

        @timing_decorator(metric_name="test_function")
        def test_func():
            time.sleep(0.01)
            return "result"

        result = test_func()
        self.assertEqual(result, "result")

        stats = get_metric_stats("test_function")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 1)
        self.assertGreater(stats["total"], 0)

    def test_timing_decorator_async(self):
        """Test timing decorator on async function."""
        import asyncio

        @timing_decorator(metric_name="test_async_function")
        async def test_async_func():
            await asyncio.sleep(0.01)
            return "async_result"

        result = asyncio.run(test_async_func())
        self.assertEqual(result, "async_result")

        stats = get_metric_stats("test_async_function")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 1)

    def test_timing_decorator_with_exception(self):
        """Test timing decorator handles exceptions."""

        @timing_decorator(metric_name="test_error_function")
        def test_func():
            raise ValueError("test error")

        with self.assertRaises(ValueError):
            test_func()

        stats = get_metric_stats("test_error_function_error")
        self.assertIsNotNone(stats)


class TestPerformanceMetrics(unittest.TestCase):
    """Tests for PerformanceMetrics context manager."""

    def setUp(self):
        clear_metrics()

    def test_performance_metrics_context(self):
        """Test PerformanceMetrics context manager."""
        with PerformanceMetrics("test_metric") as pm:
            time.sleep(0.01)

        self.assertGreater(pm.elapsed, 0)

        stats = get_metric_stats("test_metric")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 1)

    def test_performance_metrics_elapsed(self):
        """Test elapsed property."""
        with PerformanceMetrics("test_elapsed") as pm:
            time.sleep(0.01)
            elapsed = pm.elapsed
            self.assertGreater(elapsed, 0)


class TestLLMCallTracker(unittest.TestCase):
    """Tests for LLMCallTracker."""

    def test_llm_tracker_record(self):
        """Test recording LLM calls."""
        tracker = LLMCallTracker(model="gpt-4", provider="openai")
        tracker.record(prompt_tokens=100, completion_tokens=50, latency=0.5)

        self.assertEqual(tracker.call_count, 1)
        self.assertEqual(tracker.total_tokens, 150)
        self.assertGreater(tracker.total_cost, 0)
        self.assertEqual(tracker.avg_latency, 0.5)

    def test_llm_tracker_summary(self):
        """Test LLM tracker summary."""
        tracker = LLMCallTracker(model="gpt-4", provider="openai")
        tracker.record(prompt_tokens=100, completion_tokens=50, latency=0.5)
        tracker.record(prompt_tokens=200, completion_tokens=100, latency=0.8)

        summary = tracker.get_summary()
        self.assertEqual(summary["call_count"], 2)
        self.assertEqual(summary["total_tokens"], 450)
        self.assertGreater(summary["total_cost"], 0)
        self.assertGreater(summary["avg_latency"], 0)

    def test_llm_tracker_context(self):
        """Test LLMCallTracker as context manager."""
        with LLMCallTracker(model="gpt-4", provider="openai") as tracker:
            tracker.record(prompt_tokens=100, completion_tokens=50, latency=0.5)

        self.assertEqual(tracker.call_count, 1)


class TestCostEstimation(unittest.TestCase):
    """Tests for cost estimation functions."""

    def test_estimate_openai_cost(self):
        """Test OpenAI cost estimation."""
        cost = estimate_llm_cost("openai", "gpt-4", 1000000, 500000)
        self.assertGreater(cost, 0)

    def test_estimate_gemini_cost(self):
        """Test Gemini cost estimation."""
        cost = estimate_llm_cost("gemini", "gemini-pro", 1000000, 500000)
        self.assertGreater(cost, 0)

    def test_estimate_unknown_model(self):
        """Test cost estimation with unknown model uses default."""
        cost = estimate_llm_cost("openai", "unknown-model", 1000, 500)
        self.assertGreater(cost, 0)

    def test_format_cost(self):
        """Test cost formatting."""
        self.assertIn("$", format_cost(0.001))
        self.assertIn("$", format_cost(1.0))
        self.assertIn("$", format_cost(100.0))


class TestGlobalTracker(unittest.TestCase):
    """Tests for global LLM tracker."""

    def setUp(self):
        set_global_llm_tracker(None)

    def test_set_get_global_tracker(self):
        """Test setting and getting global tracker."""
        tracker = LLMCallTracker(model="test", provider="openai")
        set_global_llm_tracker(tracker)

        retrieved = get_global_llm_tracker()
        self.assertEqual(retrieved, tracker)

    def test_get_llm_metrics_summary(self):
        """Test getting LLM metrics summary."""
        tracker = LLMCallTracker(model="test", provider="openai")
        tracker.record(prompt_tokens=100, completion_tokens=50, latency=0.5)
        set_global_llm_tracker(tracker)

        summary = get_llm_metrics_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["total_tokens"], 150)


class TestPerformanceReporting(unittest.TestCase):
    """Tests for performance reporting functions."""

    def setUp(self):
        clear_metrics()
        set_global_llm_tracker(None)

    def test_generate_performance_summary(self):
        """Test generating performance summary."""
        with PerformanceMetrics("test_metric"):
            time.sleep(0.01)

        summary = generate_performance_summary()
        self.assertIn("timing_metrics", summary)
        self.assertIn("llm_metrics", summary)
        self.assertIn("preprocessing_metrics", summary)

    def test_export_metrics_to_json(self):
        """Test exporting metrics to JSON."""
        import os
        import tempfile

        with PerformanceMetrics("test_export"):
            time.sleep(0.01)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            export_metrics_to_json(filepath)
            self.assertTrue(os.path.exists(filepath))
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_export_metrics_to_csv(self):
        """Test exporting metrics to CSV."""
        import os
        import tempfile

        with PerformanceMetrics("test_export_csv"):
            time.sleep(0.01)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            filepath = f.name

        try:
            export_metrics_to_csv(filepath)
            self.assertTrue(os.path.exists(filepath))
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestMetricStats(unittest.TestCase):
    """Tests for metric statistics functions."""

    def setUp(self):
        clear_metrics()

    def test_get_metric_stats(self):
        """Test getting metric statistics."""
        with PerformanceMetrics("test_stats"):
            time.sleep(0.01)

        stats = get_metric_stats("test_stats")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 1)
        self.assertIn("mean", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("total", stats)

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        with PerformanceMetrics("metric1"):
            pass
        with PerformanceMetrics("metric2"):
            pass

        all_metrics = get_all_metrics()
        self.assertIn("metric1", all_metrics)
        self.assertIn("metric2", all_metrics)

    def test_clear_metrics(self):
        """Test clearing metrics."""
        with PerformanceMetrics("test_clear"):
            pass

        clear_metrics()
        all_metrics = get_all_metrics()
        self.assertEqual(len(all_metrics), 0)
