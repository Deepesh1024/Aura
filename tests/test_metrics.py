"""
Tests for Aura Prometheus metrics — inference latency, memory utilization,
active agents, and request success/failure counters.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY, generate_latest

from aura.core.telemetry import (
    ACTIVE_AGENTS,
    INFERENCE_LATENCY,
    MEMORY_UTILIZATION,
    REQUEST_TOTAL,
    record_memory_utilization,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics_output() -> str:
    """Return the current Prometheus metrics as a decoded string."""
    return generate_latest(REGISTRY).decode("utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferenceLatency:
    """Verify the inference latency histogram records observations."""

    def test_observe_latency(self) -> None:
        INFERENCE_LATENCY.labels(endpoint="/query").observe(0.123)
        output = _metrics_output()
        assert "aura_inference_latency_seconds_bucket" in output
        assert "aura_inference_latency_seconds_count" in output
        assert "aura_inference_latency_seconds_sum" in output

    def test_histogram_has_endpoint_label(self) -> None:
        INFERENCE_LATENCY.labels(endpoint="/query").observe(0.5)
        output = _metrics_output()
        assert 'endpoint="/query"' in output


class TestMemoryUtilization:
    """Verify memory utilization gauge sampling."""

    def test_record_memory_utilization_returns_percentage(self) -> None:
        pct = record_memory_utilization()
        assert 0.0 <= pct <= 100.0, f"Memory utilization {pct}% out of range"

    def test_gauge_set_after_record(self) -> None:
        record_memory_utilization()
        output = _metrics_output()
        assert "aura_memory_utilization_percent" in output


class TestActiveAgents:
    """Verify active agent count gauge."""

    def test_set_active_agents(self) -> None:
        ACTIVE_AGENTS.set(3)
        output = _metrics_output()
        assert "aura_active_agents" in output
        # Check the gauge value line contains 3.0
        for line in output.splitlines():
            if line.startswith("aura_active_agents "):
                assert "3.0" in line
                break

    def test_decrement_on_shutdown(self) -> None:
        ACTIVE_AGENTS.set(0)
        output = _metrics_output()
        for line in output.splitlines():
            if line.startswith("aura_active_agents "):
                assert "0.0" in line
                break


class TestRequestCounters:
    """Verify request success/failure counters."""

    def test_increment_success(self) -> None:
        REQUEST_TOTAL.labels(endpoint="/query", status="success").inc()
        output = _metrics_output()
        assert 'aura_requests_total{endpoint="/query",status="success"}' in output

    def test_increment_failure(self) -> None:
        REQUEST_TOTAL.labels(endpoint="/query", status="failure").inc()
        output = _metrics_output()
        assert 'aura_requests_total{endpoint="/query",status="failure"}' in output

    def test_counters_only_increase(self) -> None:
        before = REQUEST_TOTAL.labels(endpoint="/query", status="success")._value.get()
        REQUEST_TOTAL.labels(endpoint="/query", status="success").inc()
        after = REQUEST_TOTAL.labels(endpoint="/query", status="success")._value.get()
        assert after == before + 1.0
