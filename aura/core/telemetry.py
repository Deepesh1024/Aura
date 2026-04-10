"""
Telemetry — OpenTelemetry traces + Prometheus metrics for full observability.

Exposes production-grade Prometheus metrics including:
  - Inference latency (histogram with p50/p99 buckets)
  - Memory utilization percentage (gauge)
  - Active agent count (gauge)
  - Request success/failure rate (counter)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, AsyncIterator, Callable

import psutil
from prometheus_client import Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------

AGENT_STEP_LATENCY = Histogram(
    "aura_agent_step_latency_ms",
    "Latency of individual agent ReAct steps",
    labelnames=["agent_name", "step_type"],
    buckets=[1, 5, 10, 25, 50, 100, 200, 500, 1000, 5000],
)

MESSAGE_BUS_THROUGHPUT = Counter(
    "aura_bus_messages_total",
    "Total messages published on the bus",
    labelnames=["topic", "direction"],
)

LLM_TOKEN_USAGE = Counter(
    "aura_llm_tokens_total",
    "Total LLM tokens consumed",
    labelnames=["agent_name", "type"],  # type = prompt | completion
)

LLM_LATENCY = Histogram(
    "aura_llm_latency_ms",
    "LLM inference latency",
    labelnames=["agent_name", "model"],
    buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

ACTIVE_AGENTS = Gauge(
    "aura_active_agents",
    "Number of currently active agent instances",
)

CACHE_HITS = Counter(
    "aura_cache_hits_total",
    "Cache hit count",
    labelnames=["cache_name"],
)

CACHE_MISSES = Counter(
    "aura_cache_misses_total",
    "Cache miss count",
    labelnames=["cache_name"],
)

MODEL_DRIFT_SCORE = Gauge(
    "aura_model_drift_score",
    "Drift detection score (0 = no drift, 1 = max drift)",
    labelnames=["model_name"],
)

VERIFICATION_RESULTS = Counter(
    "aura_verification_results_total",
    "Verification outcomes",
    labelnames=["verdict"],  # PASS | FAIL | WARN
)

RAG_RETRIEVAL_LATENCY = Histogram(
    "aura_rag_retrieval_latency_ms",
    "RAG retrieval pipeline latency",
    labelnames=["path"],  # structured | unstructured | hybrid
    buckets=[10, 25, 50, 100, 200, 500],
)

# ---------------------------------------------------------------------------
# Production Metrics — Inference, Memory, Requests
# ---------------------------------------------------------------------------

INFERENCE_LATENCY = Histogram(
    "aura_inference_latency_seconds",
    "End-to-end inference latency for query processing (seconds)",
    labelnames=["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

MEMORY_UTILIZATION = Gauge(
    "aura_memory_utilization_percent",
    "Current memory utilization of the Aura process as a percentage of system RAM",
)

REQUEST_TOTAL = Counter(
    "aura_requests_total",
    "Total HTTP requests processed by the API",
    labelnames=["endpoint", "status"],  # status: success | failure
)


# ---------------------------------------------------------------------------
# Instrumentation Decorators
# ---------------------------------------------------------------------------

def track_latency(agent_name: str, step_type: str) -> Callable:
    """Decorator to track agent step latency."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.time() - start) * 1000
                AGENT_STEP_LATENCY.labels(
                    agent_name=agent_name, step_type=step_type
                ).observe(elapsed_ms)

        return wrapper

    return decorator


def track_llm_call(agent_name: str, model: str) -> Callable:
    """Decorator to track LLM call latency and token usage."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.time() - start) * 1000
            LLM_LATENCY.labels(agent_name=agent_name, model=model).observe(
                elapsed_ms
            )
            # Extract token usage if present in result
            if isinstance(result, dict) and "usage" in result:
                usage = result["usage"]
                LLM_TOKEN_USAGE.labels(
                    agent_name=agent_name, type="prompt"
                ).inc(usage.get("prompt_tokens", 0))
                LLM_TOKEN_USAGE.labels(
                    agent_name=agent_name, type="completion"
                ).inc(usage.get("completion_tokens", 0))
            return result

        return wrapper

    return decorator


@asynccontextmanager
async def track_rag(path: str) -> AsyncIterator[None]:
    """Context manager to track RAG retrieval latency."""
    start = time.time()
    try:
        yield
    finally:
        elapsed_ms = (time.time() - start) * 1000
        RAG_RETRIEVAL_LATENCY.labels(path=path).observe(elapsed_ms)


# ---------------------------------------------------------------------------
# Drift Detection
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Simple distribution-drift detector using running statistics.
    Compares recent output distributions to a baseline.
    """

    def __init__(self, model_name: str, window_size: int = 100) -> None:
        self.model_name = model_name
        self._window_size = window_size
        self._baseline_scores: list[float] = []
        self._recent_scores: list[float] = []

    def record_score(self, score: float) -> float:
        """Record a confidence score and return the current drift metric."""
        self._recent_scores.append(score)
        if len(self._recent_scores) > self._window_size:
            self._recent_scores = self._recent_scores[-self._window_size:]

        if len(self._baseline_scores) < self._window_size:
            self._baseline_scores.append(score)
            return 0.0

        # Compute mean shift as simple drift signal
        baseline_mean = sum(self._baseline_scores) / len(self._baseline_scores)
        recent_mean = sum(self._recent_scores) / len(self._recent_scores)
        drift = abs(recent_mean - baseline_mean) / max(baseline_mean, 1e-6)
        drift = min(drift, 1.0)

        MODEL_DRIFT_SCORE.labels(model_name=self.model_name).set(drift)
        return drift

    def reset_baseline(self) -> None:
        """Reset baseline to current recent window."""
        self._baseline_scores = list(self._recent_scores)


# ---------------------------------------------------------------------------
# Memory Utilization Sampling
# ---------------------------------------------------------------------------

def record_memory_utilization() -> float:
    """
    Sample the current Aura process RSS as a percentage of total system RAM
    and update the MEMORY_UTILIZATION gauge.

    Returns:
        The memory utilization percentage (0.0–100.0).
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    total_mem = psutil.virtual_memory().total
    utilization = (mem_info.rss / total_mem) * 100.0 if total_mem > 0 else 0.0
    MEMORY_UTILIZATION.set(round(utilization, 2))
    return utilization


# ---------------------------------------------------------------------------
# Metrics Export
# ---------------------------------------------------------------------------

def get_prometheus_metrics() -> bytes:
    """Generate Prometheus-compatible metrics output."""
    return generate_latest()
