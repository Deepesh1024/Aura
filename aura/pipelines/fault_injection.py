"""
Production Fault Simulation — Network partitioning, SQL timeouts, data skew,
and chaos engineering for the MPP layer.

Replaces naive "simulation" with a production-mocking strategy that proves
understanding of real-world distributed data system failures.

Why this matters:
  Apple's MPP stack handles petabytes across distributed Snowflake/SingleStore
  clusters. An interviewer will ask: "What happens when the database doesn't
  respond?" This module answers that question with working code.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fault Types
# ---------------------------------------------------------------------------

class FaultType(Enum):
    NETWORK_PARTITION = auto()   # Complete loss of connectivity
    SQL_TIMEOUT = auto()         # Query exceeds execution deadline
    DATA_SKEW = auto()           # Hot partition causes imbalanced load
    CONNECTION_POOL_EXHAUSTION = auto()  # All connections in use
    STALE_CACHE = auto()         # Cache returns outdated data
    PARTIAL_RESULT = auto()      # Query returns incomplete data
    QUERY_OPTIMIZER_REGRESSION = auto()  # Bad query plan from optimizer


@dataclass(slots=True)
class FaultConfig:
    """Configuration for a specific fault injection."""
    fault_type: FaultType
    probability: float = 0.0       # 0.0 = never, 1.0 = always
    duration_ms: int = 0           # How long the fault persists
    affected_regions: list[str] = field(default_factory=list)  # Empty = all
    affected_tables: list[str] = field(default_factory=list)
    error_message: str = ""
    enabled: bool = True


@dataclass
class FaultEvent:
    """Record of a fault that was injected."""
    fault_type: FaultType
    triggered_at: float = field(default_factory=time.time)
    target_sql: str = ""
    region: str = ""
    resolution: str = ""  # How the system handled it
    latency_added_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_type": self.fault_type.name,
            "triggered_at": self.triggered_at,
            "target_sql": self.target_sql[:100],
            "region": self.region,
            "resolution": self.resolution,
            "latency_added_ms": self.latency_added_ms,
        }


# ---------------------------------------------------------------------------
# Fault Injector
# ---------------------------------------------------------------------------

class FaultInjector:
    """
    Chaos engineering layer for the MPP simulator.

    Wraps SQL execution with configurable fault injection to prove
    the system handles real-world distributed data failures.
    """

    def __init__(self) -> None:
        self._faults: dict[FaultType, FaultConfig] = {}
        self._event_log: list[FaultEvent] = []
        self._circuit_breaker_open: dict[str, float] = {}  # region → open_until

    def configure(self, config: FaultConfig) -> None:
        """Register a fault configuration."""
        self._faults[config.fault_type] = config
        logger.info(
            "Fault configured: %s (p=%.2f, regions=%s)",
            config.fault_type.name,
            config.probability,
            config.affected_regions or ["ALL"],
        )

    def configure_preset(self, preset: str) -> None:
        """Load a preset fault profile."""
        presets = {
            "chaos": [
                FaultConfig(FaultType.NETWORK_PARTITION, probability=0.05, duration_ms=2000),
                FaultConfig(FaultType.SQL_TIMEOUT, probability=0.10, duration_ms=5000,
                            error_message="Query execution exceeded 30s deadline"),
                FaultConfig(FaultType.DATA_SKEW, probability=0.15,
                            affected_regions=["APAC"]),
                FaultConfig(FaultType.CONNECTION_POOL_EXHAUSTION, probability=0.03,
                            error_message="No available connections (200/200 in use)"),
                FaultConfig(FaultType.PARTIAL_RESULT, probability=0.05),
            ],
            "network_stress": [
                FaultConfig(FaultType.NETWORK_PARTITION, probability=0.20, duration_ms=5000,
                            affected_regions=["APAC", "MEA"]),
                FaultConfig(FaultType.SQL_TIMEOUT, probability=0.25, duration_ms=10000),
            ],
            "production_realistic": [
                FaultConfig(FaultType.SQL_TIMEOUT, probability=0.02, duration_ms=3000),
                FaultConfig(FaultType.DATA_SKEW, probability=0.08,
                            affected_regions=["APAC"]),
                FaultConfig(FaultType.STALE_CACHE, probability=0.05),
                FaultConfig(FaultType.QUERY_OPTIMIZER_REGRESSION, probability=0.01),
            ],
        }

        for config in presets.get(preset, presets["production_realistic"]):
            self.configure(config)

    async def maybe_inject(
        self,
        sql: str,
        region: str | None = None,
        table: str | None = None,
    ) -> FaultEvent | None:
        """
        Probabilistically inject a fault before SQL execution.
        Returns a FaultEvent if a fault was triggered, None otherwise.
        """
        for fault_type, config in self._faults.items():
            if not config.enabled:
                continue

            # Check region filter
            if config.affected_regions and region:
                if region not in config.affected_regions:
                    continue

            # Check table filter
            if config.affected_tables and table:
                if table not in config.affected_tables:
                    continue

            # Probabilistic trigger
            if random.random() > config.probability:
                continue

            # Fault triggered — execute the fault
            event = await self._execute_fault(fault_type, config, sql, region or "")
            self._event_log.append(event)
            return event

        return None

    async def _execute_fault(
        self,
        fault_type: FaultType,
        config: FaultConfig,
        sql: str,
        region: str,
    ) -> FaultEvent:
        """Execute a specific fault."""
        event = FaultEvent(
            fault_type=fault_type,
            target_sql=sql,
            region=region,
        )

        if fault_type == FaultType.NETWORK_PARTITION:
            # Simulate network partition — circuit breaker opens
            await asyncio.sleep(config.duration_ms / 1000.0)
            self._circuit_breaker_open[region] = time.time() + (config.duration_ms / 1000.0)
            event.latency_added_ms = config.duration_ms
            event.resolution = "circuit_breaker_opened"
            raise ConnectionError(
                f"Network partition: cannot reach {region} "
                f"(circuit breaker open for {config.duration_ms}ms)"
            )

        elif fault_type == FaultType.SQL_TIMEOUT:
            await asyncio.sleep(min(config.duration_ms / 1000.0, 5.0))
            event.latency_added_ms = config.duration_ms
            event.resolution = "timeout_exceeded"
            raise TimeoutError(
                config.error_message or
                f"SQL execution timeout after {config.duration_ms}ms"
            )

        elif fault_type == FaultType.DATA_SKEW:
            # Don't raise — inject latency proportional to skew
            skew_penalty = random.uniform(100, 500)
            await asyncio.sleep(skew_penalty / 1000.0)
            event.latency_added_ms = skew_penalty
            event.resolution = f"data_skew_penalty_{skew_penalty:.0f}ms"
            logger.warning(
                "Data skew detected in %s: +%.0fms penalty",
                region, skew_penalty,
            )

        elif fault_type == FaultType.CONNECTION_POOL_EXHAUSTION:
            event.resolution = "pool_exhausted"
            raise ConnectionError(
                config.error_message or "Connection pool exhausted"
            )

        elif fault_type == FaultType.STALE_CACHE:
            event.resolution = "stale_cache_served"
            event.latency_added_ms = 0.1  # Cache is fast but wrong
            logger.warning("Stale cache served for query: %s", sql[:60])

        elif fault_type == FaultType.PARTIAL_RESULT:
            event.resolution = "partial_result_returned"
            logger.warning("Partial result returned for query: %s", sql[:60])

        elif fault_type == FaultType.QUERY_OPTIMIZER_REGRESSION:
            # Bad query plan — sequential scan instead of index
            penalty = random.uniform(500, 3000)
            await asyncio.sleep(penalty / 1000.0)
            event.latency_added_ms = penalty
            event.resolution = f"optimizer_regression_sequential_scan_{penalty:.0f}ms"
            logger.warning("Query optimizer regression: sequential scan for %s", sql[:60])

        return event

    def is_circuit_open(self, region: str) -> bool:
        """Check if the circuit breaker is open for a region."""
        deadline = self._circuit_breaker_open.get(region)
        if deadline and time.time() < deadline:
            return True
        # Auto-close expired breakers
        self._circuit_breaker_open.pop(region, None)
        return False

    def get_event_log(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._event_log]

    def get_stats(self) -> dict[str, Any]:
        fault_counts: dict[str, int] = {}
        for event in self._event_log:
            name = event.fault_type.name
            fault_counts[name] = fault_counts.get(name, 0) + 1

        return {
            "total_faults": len(self._event_log),
            "by_type": fault_counts,
            "circuit_breakers_open": list(self._circuit_breaker_open.keys()),
            "active_configs": [
                {"type": ft.name, "probability": fc.probability}
                for ft, fc in self._faults.items() if fc.enabled
            ],
        }

    def reset(self) -> None:
        self._event_log.clear()
        self._circuit_breaker_open.clear()


# ---------------------------------------------------------------------------
# Resilient Executor — wraps SQL execution with retry + fallback
# ---------------------------------------------------------------------------

class ResilientSQLExecutor:
    """
    Production-grade SQL executor with retry logic, circuit breakers,
    and fallback strategies for distributed MPP environments.

    This is what you show in the interview when they ask:
    "What happens when Snowflake doesn't respond?"
    """

    def __init__(
        self,
        mpp_simulator: Any,
        fault_injector: FaultInjector | None = None,
        max_retries: int = 3,
        timeout_ms: int = 30_000,
        retry_backoff_ms: int = 500,
    ) -> None:
        self.mpp = mpp_simulator
        self.faults = fault_injector
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms
        self.retry_backoff_ms = retry_backoff_ms
        self._execution_log: list[dict[str, Any]] = []

    async def execute(
        self,
        sql: str,
        region: str | None = None,
        fallback_sql: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute SQL with production-grade resilience:
          1. Check circuit breaker
          2. Inject faults (if configured)
          3. Execute with timeout
          4. Retry on transient failures
          5. Fall back to simpler query on persistent failure
        """
        # Check circuit breaker
        if self.faults and region and self.faults.is_circuit_open(region):
            return {
                "error": f"Circuit breaker OPEN for {region}",
                "fallback": True,
                "sql": sql,
                "resolution": "circuit_breaker",
            }

        last_error = None
        for attempt in range(self.max_retries):
            start = time.time()
            try:
                # Fault injection (if configured)
                if self.faults:
                    fault_event = await self.faults.maybe_inject(sql, region)
                    if fault_event and fault_event.resolution == "stale_cache_served":
                        # Stale cache — proceed but flag the result
                        result = await self._execute_with_timeout(sql)
                        result["warning"] = "stale_cache"
                        return result

                # Execute the actual query
                result = await self._execute_with_timeout(sql)
                elapsed_ms = (time.time() - start) * 1000

                self._execution_log.append({
                    "sql": sql,
                    "attempt": attempt + 1,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "status": "success",
                    "region": region,
                })

                return result

            except TimeoutError as e:
                last_error = e
                logger.warning(
                    "SQL timeout (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, sql[:60],
                )
                # Exponential backoff
                await asyncio.sleep(
                    (self.retry_backoff_ms * (2 ** attempt)) / 1000.0
                )

            except ConnectionError as e:
                last_error = e
                logger.error("Connection error: %s", e)
                # Don't retry connection errors — circuit breaker handles it
                break

            except Exception as e:
                last_error = e
                logger.error("SQL execution error (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_backoff_ms / 1000.0)

        # All retries exhausted — try fallback
        if fallback_sql:
            logger.info("Attempting fallback query: %s", fallback_sql[:60])
            try:
                result = await self._execute_with_timeout(fallback_sql)
                result["fallback"] = True
                result["original_sql"] = sql
                result["original_error"] = str(last_error)
                return result
            except Exception as fb_error:
                logger.error("Fallback also failed: %s", fb_error)

        # Complete failure
        self._execution_log.append({
            "sql": sql,
            "attempt": self.max_retries,
            "status": "failed",
            "error": str(last_error),
            "region": region,
        })

        return {
            "error": str(last_error),
            "sql": sql,
            "attempts": self.max_retries,
            "resolution": "all_retries_exhausted",
        }

    async def _execute_with_timeout(self, sql: str) -> dict[str, Any]:
        """Execute SQL with a hard timeout."""
        try:
            return await asyncio.wait_for(
                self.mpp.execute(sql),
                timeout=self.timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"SQL timeout after {self.timeout_ms}ms")

    def get_execution_log(self) -> list[dict[str, Any]]:
        return list(self._execution_log)

    def get_reliability_stats(self) -> dict[str, Any]:
        """Compute reliability metrics from execution history."""
        if not self._execution_log:
            return {"total": 0}

        total = len(self._execution_log)
        success = sum(1 for e in self._execution_log if e["status"] == "success")
        failures = total - success
        latencies = [
            e["elapsed_ms"] for e in self._execution_log
            if e["status"] == "success" and "elapsed_ms" in e
        ]

        return {
            "total_executions": total,
            "successes": success,
            "failures": failures,
            "success_rate": f"{(success / total) * 100:.1f}%",
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p99_latency_ms": (
                round(sorted(latencies)[int(len(latencies) * 0.99)], 2)
                if len(latencies) > 10 else 0
            ),
            "retry_rate": f"{sum(1 for e in self._execution_log if e.get('attempt', 1) > 1) / total * 100:.1f}%",
        }
