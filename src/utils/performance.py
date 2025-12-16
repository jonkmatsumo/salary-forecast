"""Performance monitoring utilities for tracking execution time, LLM API calls, and costs."""

import asyncio
import functools
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

_metrics_store: Dict[str, List[float]] = {}
_metrics_lock = threading.Lock()

_global_llm_tracker: Any = None
_global_llm_lock = threading.Lock()


def timing_decorator(metric_name: Optional[str] = None, log_result: bool = False):
    """Decorator to measure function execution time.

    Args:
        metric_name (Optional[str]): Metric name. If None, uses function name.
        log_result (bool): Log timing result.

    Returns:
        Callable: Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.perf_counter() - start_time
                    _record_metric(name, elapsed)
                    if log_result:
                        logger.debug(f"[PERF] {name} took {elapsed:.4f}s")
                    return result
                except Exception:
                    elapsed = time.perf_counter() - start_time
                    _record_metric(f"{name}_error", elapsed)
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start_time
                    _record_metric(name, elapsed)
                    if log_result:
                        logger.debug(f"[PERF] {name} took {elapsed:.4f}s")
                    return result
                except Exception:
                    elapsed = time.perf_counter() - start_time
                    _record_metric(f"{name}_error", elapsed)
                    raise

            return sync_wrapper

    return decorator


def _record_metric(name: str, value: float) -> None:
    """Record a metric value.

    Args:
        name (str): Metric name.
        value (float): Metric value.
    """
    with _metrics_lock:
        if name not in _metrics_store:
            _metrics_store[name] = []
        _metrics_store[name].append(value)


class PerformanceMetrics:
    """Context manager for tracking performance metrics."""

    def __init__(self, metric_name: str):
        """Initialize performance metrics tracker.

        Args:
            metric_name (str): Metric name.
        """
        self.metric_name = metric_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "PerformanceMetrics":
        """Start timing.

        Returns:
            PerformanceMetrics: Self instance.
        """
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and record metric.

        Args:
            exc_type (Any): Exception type.
            exc_val (Any): Exception value.
            exc_tb (Any): Traceback.
        """
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            elapsed = self.end_time - self.start_time
            _record_metric(self.metric_name, elapsed)

    @property
    def elapsed(self) -> float:
        """Get elapsed time.

        Returns:
            float: Elapsed time in seconds.
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time


class LLMCallTracker:
    """Track LLM API calls with token usage and costs."""

    def __init__(self, model: str, provider: str = "openai", global_tracking: bool = False):
        """Initialize LLM call tracker.

        Args:
            model (str): Model name.
            provider (str): Provider name.
            global_tracking (bool): If True, also record to global tracker.
        """
        self.model = model
        self.provider = provider.lower()
        self.calls: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.lock = threading.Lock()
        self.global_tracking = global_tracking

    def __enter__(self) -> "LLMCallTracker":
        """Start tracking.

        Returns:
            LLMCallTracker: Self instance.
        """
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop tracking.

        Args:
            exc_type (Any): Exception type.
            exc_val (Any): Exception value.
            exc_tb (Any): Traceback.
        """
        pass

    def record(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        latency: Optional[float] = None,
    ) -> None:
        """Record an LLM call.

        Args:
            prompt_tokens (int): Prompt tokens.
            completion_tokens (int): Completion tokens.
            total_tokens (Optional[int]): Total tokens.
            latency (Optional[float]): Call latency in seconds.
        """
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        if latency is None and self.start_time is not None:
            latency = time.perf_counter() - self.start_time

        cost = estimate_llm_cost(self.provider, self.model, prompt_tokens, completion_tokens)

        call_data = {
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency": latency or 0.0,
            "cost": cost,
            "timestamp": time.time(),
        }

        with self.lock:
            self.calls.append(call_data)

        if self.global_tracking:
            global_tracker = get_global_llm_tracker()
            if global_tracker:
                with global_tracker.lock:
                    global_tracker.calls.append(call_data)

    @property
    def total_tokens(self) -> int:
        """Get total tokens across all calls.

        Returns:
            int: Total tokens.
        """
        with self.lock:
            return sum(call["total_tokens"] for call in self.calls)

    @property
    def total_cost(self) -> float:
        """Get total estimated cost.

        Returns:
            float: Total cost in USD.
        """
        with self.lock:
            costs = [float(call["cost"]) for call in self.calls]
            return sum(costs)

    @property
    def avg_latency(self) -> float:
        """Get average latency.

        Returns:
            float: Average latency in seconds.
        """
        from typing import cast

        with self.lock:
            if not self.calls:
                return 0.0
            latencies = [call["latency"] for call in self.calls if call["latency"] > 0]
            result = sum(latencies) / len(latencies) if latencies else 0.0
            return cast(float, result)

    @property
    def call_count(self) -> int:
        """Get number of calls.

        Returns:
            int: Call count.
        """
        with self.lock:
            return len(self.calls)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dict[str, Any]: Summary statistics.
        """
        with self.lock:
            total_tokens = sum(call["total_tokens"] for call in self.calls)
            costs = [float(call["cost"]) for call in self.calls]
            total_cost = sum(costs)
            latencies = [call["latency"] for call in self.calls if call["latency"] > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            return {
                "model": self.model,
                "provider": self.provider,
                "call_count": len(self.calls),
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "avg_latency": avg_latency,
                "prompt_tokens": sum(call["prompt_tokens"] for call in self.calls),
                "completion_tokens": sum(call["completion_tokens"] for call in self.calls),
            }


def get_metric_stats(metric_name: str) -> Optional[Dict[str, float]]:
    """Get statistics for a metric.

    Args:
        metric_name (str): Metric name.

    Returns:
        Optional[Dict[str, float]]: Statistics dict or None if metric doesn't exist.
    """
    with _metrics_lock:
        if metric_name not in _metrics_store or not _metrics_store[metric_name]:
            return None

        values = _metrics_store[metric_name]
        return {
            "count": len(values),
            "total": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }


def get_all_metrics() -> Dict[str, List[float]]:
    """Get all recorded metrics.

    Returns:
        Dict[str, List[float]]: All metrics.
    """
    with _metrics_lock:
        return _metrics_store.copy()


def clear_metrics() -> None:
    """Clear all recorded metrics."""
    with _metrics_lock:
        _metrics_store.clear()


def get_global_llm_tracker() -> Optional["LLMCallTracker"]:
    """Get the global LLM tracker instance.

    Returns:
        Optional[LLMCallTracker]: Global tracker or None.
    """
    with _global_llm_lock:
        tracker = _global_llm_tracker
        if tracker is None:
            return None
        if isinstance(tracker, LLMCallTracker):
            return tracker
        return None


def set_global_llm_tracker(tracker: Optional["LLMCallTracker"]) -> None:
    """Set the global LLM tracker instance.

    Args:
        tracker (Optional[LLMCallTracker]): Tracker to set.
    """
    with _global_llm_lock:
        global _global_llm_tracker
        _global_llm_tracker = tracker


def get_llm_metrics_summary() -> Optional[Dict[str, Any]]:
    """Get summary of all LLM metrics from global tracker.

    Returns:
        Optional[Dict[str, Any]]: Summary or None.
    """
    tracker = get_global_llm_tracker()
    if tracker:
        return tracker.get_summary()
    return None


OPENAI_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4-turbo-preview": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-4-32k": (60.0, 120.0),
    "gpt-3.5-turbo": (0.5, 1.5),
    "gpt-3.5-turbo-16k": (3.0, 4.0),
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.6),
}

GEMINI_PRICING: Dict[str, Tuple[float, float]] = {
    "gemini-pro": (0.5, 1.5),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.3),
    "gemini-ultra": (10.0, 30.0),
}


def estimate_llm_cost(
    provider: str, model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Estimate cost for LLM API call.

    Args:
        provider (str): Provider name.
        model (str): Model name.
        prompt_tokens (int): Prompt tokens.
        completion_tokens (int): Completion tokens.

    Returns:
        float: Estimated cost in USD.
    """
    provider_lower = provider.lower()

    if provider_lower == "openai":
        pricing = OPENAI_PRICING.get(model.lower())
        if not pricing:
            pricing = OPENAI_PRICING.get("gpt-3.5-turbo", (0.5, 1.5))
    elif provider_lower == "gemini":
        pricing = GEMINI_PRICING.get(model.lower())
        if not pricing:
            pricing = GEMINI_PRICING.get("gemini-pro", (0.5, 1.5))
    else:
        return 0.0

    prompt_price_per_1m, completion_price_per_1m = pricing

    prompt_cost = (prompt_tokens / 1_000_000) * prompt_price_per_1m
    completion_cost = (completion_tokens / 1_000_000) * completion_price_per_1m

    return prompt_cost + completion_cost


def format_cost(cost: float) -> str:
    """Format cost as currency string.

    Args:
        cost (float): Cost in USD.

    Returns:
        str: Formatted cost string.
    """
    if cost < 0.01:
        return f"${cost * 1000:.2f} (millicents)"
    elif cost < 1.0:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def generate_performance_summary() -> Dict[str, Any]:
    """Generate a comprehensive performance summary.

    Returns:
        Dict[str, Any]: Performance summary.
    """
    summary: Dict[str, Any] = {
        "timing_metrics": {},
        "llm_metrics": None,
        "preprocessing_metrics": {},
    }

    all_metrics = get_all_metrics()
    for metric_name, values in all_metrics.items():
        stats = get_metric_stats(metric_name)
        if stats:
            summary["timing_metrics"][metric_name] = stats

    llm_summary = get_llm_metrics_summary()
    if llm_summary:
        summary["llm_metrics"] = llm_summary

    preprocessing_metrics = [
        "preprocessing_data_cleaning_time",
        "preprocessing_outlier_removal_time",
        "preprocessing_feature_encoding_time",
        "preprocessing_ranked_encoder_time",
        "preprocessing_proximity_encoder_time",
        "preprocessing_cost_of_living_encoder_time",
        "preprocessing_metro_population_encoder_time",
        "preprocessing_date_normalizer_time",
    ]

    for metric_name in preprocessing_metrics:
        stats = get_metric_stats(metric_name)
        if stats:
            summary["preprocessing_metrics"][metric_name] = stats

    return summary


def export_metrics_to_json(filepath: str) -> None:
    """Export all metrics to JSON file.

    Args:
        filepath (str): Output file path.
    """
    import json

    summary = generate_performance_summary()
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)


def export_metrics_to_csv(filepath: str) -> None:
    """Export timing metrics to CSV file.

    Args:
        filepath (str): Output file path.
    """
    import csv

    all_metrics = get_all_metrics()
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_name", "value"])
        for metric_name, values in all_metrics.items():
            for value in values:
                writer.writerow([metric_name, value])


def print_performance_report() -> None:
    """Print a formatted performance report to console."""
    summary = generate_performance_summary()

    print("\n" + "=" * 80)
    print("PERFORMANCE REPORT")
    print("=" * 80)

    if summary["llm_metrics"]:
        llm = summary["llm_metrics"]
        print("\nLLM Metrics:")
        print(f"  Total Calls: {llm['call_count']}")
        print(f"  Total Tokens: {llm['total_tokens']:,}")
        print(f"  Prompt Tokens: {llm['prompt_tokens']:,}")
        print(f"  Completion Tokens: {llm['completion_tokens']:,}")
        print(f"  Total Cost: {format_cost(llm['total_cost'])}")
        print(f"  Average Latency: {llm['avg_latency']:.3f}s")

    if summary["timing_metrics"]:
        print("\nTiming Metrics:")
        for metric_name, stats in summary["timing_metrics"].items():
            print(f"  {metric_name}:")
            print(f"    Count: {stats['count']}")
            print(f"    Total: {stats['total']:.4f}s")
            print(f"    Mean: {stats['mean']:.4f}s")
            print(f"    Min: {stats['min']:.4f}s")
            print(f"    Max: {stats['max']:.4f}s")

    if summary["preprocessing_metrics"]:
        print("\nPreprocessing Metrics:")
        total_preprocessing = 0.0
        for metric_name, stats in summary["preprocessing_metrics"].items():
            total_preprocessing += stats["total"]
            print(f"  {metric_name}: {stats['total']:.4f}s (mean: {stats['mean']:.4f}s)")
        print(f"  Total Preprocessing Time: {total_preprocessing:.4f}s")

    print("\n" + "=" * 80 + "\n")


def extract_tokens_from_langchain_response(response: Any) -> Tuple[int, int, int]:
    """Extract token usage from LangChain response.

    Args:
        response (Any): LangChain response object.

    Returns:
        Tuple[int, int, int]: (prompt_tokens, completion_tokens, total_tokens).
    """
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if hasattr(response, "response_metadata"):
        metadata = response.response_metadata
        if metadata:
            if "token_usage" in metadata:
                usage = metadata["token_usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            elif "usage" in metadata:
                usage = metadata["usage"]
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

    if hasattr(response, "response_metadata") and not total_tokens:
        metadata = response.response_metadata
        if metadata and "model_name" in metadata:
            model_name = metadata["model_name"]
            if "gpt-4" in model_name.lower():
                prompt_tokens = len(str(response.content)) // 4
                completion_tokens = len(str(response.content)) // 4
                total_tokens = prompt_tokens + completion_tokens

    return (prompt_tokens, completion_tokens, total_tokens)
