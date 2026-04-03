"""Tests for the critical systems hardening modules:
  - MemoryBudgetManager (context-window saturation prevention)
  - FaultInjector + ResilientSQLExecutor (production fault simulation)
  - GuardrailFramework (safe exploration verification)
"""

from __future__ import annotations

import pytest

from aura.core.budget import MemoryBudgetManager, MemoryTier, estimate_tokens
from aura.pipelines.fault_injection import (
    FaultConfig,
    FaultInjector,
    FaultType,
    ResilientSQLExecutor,
)
from aura.pipelines.mpp_simulator import MPPSimulator
from verifier.guardrails import GuardrailFramework, SafetyZone


# =========================================================================
# Memory Budget Tests
# =========================================================================

class TestMemoryBudget:
    def test_token_estimation(self):
        assert estimate_tokens("hello") >= 1
        assert estimate_tokens("a" * 400) == 100  # 400 chars / 4

    def test_add_context_within_budget(self):
        mgr = MemoryBudgetManager(total_budget_tokens=1000)
        result = mgr.add_context("agent_a", "Short context", tier=MemoryTier.CONVERSATION)
        report = mgr.get_budget_report()
        assert report["used_tokens"] < 1000
        assert report["per_agent"]["agent_a"] > 0

    def test_shared_context_deduplication(self):
        mgr = MemoryBudgetManager(total_budget_tokens=500)
        mgr.add_context("shared", "SLA: p99 < 200ms", tier=MemoryTier.GROUNDING, shared=True)
        mgr.add_context("shared", "SLA: p99 < 200ms", tier=MemoryTier.GROUNDING, shared=True)
        report = mgr.get_budget_report()
        # Second add should be deduplicated
        assert report["shared_tokens"] == estimate_tokens("SLA: p99 < 200ms")

    def test_eviction_on_budget_exceeded(self):
        mgr = MemoryBudgetManager(total_budget_tokens=100)

        # Add enough content to exceed budget
        for i in range(20):
            mgr.add_context(
                "agent_a",
                f"Turn {i}: " + "x" * 100,
                tier=MemoryTier.CONVERSATION,
            )

        report = mgr.get_budget_report()
        assert report["used_tokens"] <= report["total_budget"]
        assert report["evictions"] > 0

    def test_grounding_never_evicted(self):
        mgr = MemoryBudgetManager(total_budget_tokens=200)

        # Add grounding data
        mgr.add_context("shared", "CRITICAL: SLA is 200ms", tier=MemoryTier.GROUNDING, shared=True)

        # Flood with conversation to trigger eviction
        for i in range(30):
            mgr.add_context("agent_a", f"Conversation turn {i} " + "x" * 80)

        # Grounding should still be retrievable
        context = mgr.get_context_for_agent("agent_a")
        assert "CRITICAL: SLA is 200ms" in context

    def test_cross_agent_context(self):
        mgr = MemoryBudgetManager(total_budget_tokens=2000)

        mgr.add_context("shared", "Shared fact", tier=MemoryTier.GROUNDING, shared=True)
        mgr.add_context("planner", "Planner observation", tier=MemoryTier.CONVERSATION)
        mgr.add_context("data_architect", "SQL result", tier=MemoryTier.TOOL_OUTPUT)

        context = mgr.get_cross_agent_context(["planner", "data_architect"])
        assert "[SHARED]" in context
        assert "[PLANNER]" in context
        assert "[DATA_ARCHITECT]" in context

    def test_budget_report_tiers(self):
        mgr = MemoryBudgetManager(total_budget_tokens=5000)
        mgr.add_context("a", "Grounding", tier=MemoryTier.GROUNDING)
        mgr.add_context("a", "Tool output", tier=MemoryTier.TOOL_OUTPUT)
        mgr.add_context("a", "Conversation", tier=MemoryTier.CONVERSATION)

        report = mgr.get_budget_report()
        assert report["tier_breakdown"]["grounding"] > 0
        assert report["tier_breakdown"]["tool_output"] > 0
        assert report["tier_breakdown"]["conversation"] > 0

    def test_100_turn_never_exceeds_budget(self):
        """The critical test: 100-turn multi-agent task should NEVER exceed budget."""
        mgr = MemoryBudgetManager(total_budget_tokens=2048)

        for turn in range(100):
            agent = ["planner", "data_architect", "verifier"][turn % 3]
            mgr.add_context(
                agent,
                f"Turn {turn}: Agent {agent} analyzed region metrics " + "data " * 20,
                tier=MemoryTier.CONVERSATION,
            )
            if turn % 5 == 0:
                mgr.add_context(
                    agent,
                    f"SQL: SELECT * FROM metrics → 50 rows " + "result " * 15,
                    tier=MemoryTier.TOOL_OUTPUT,
                )

            report = mgr.get_budget_report()
            assert report["used_tokens"] <= report["total_budget"], \
                f"Budget exceeded at turn {turn}: {report['used_tokens']} > {report['total_budget']}"


# =========================================================================
# Fault Injection Tests
# =========================================================================

class TestFaultInjection:
    @pytest.fixture
    async def mpp(self):
        sim = MPPSimulator(db_path=":memory:", synthetic_rows=100)
        await sim.initialize()
        yield sim
        await sim.close()

    @pytest.mark.asyncio
    async def test_no_faults_baseline(self, mpp):
        injector = FaultInjector()  # No faults configured
        executor = ResilientSQLExecutor(mpp, fault_injector=injector)

        result = await executor.execute("SELECT COUNT(*) FROM retail_metrics")
        assert "error" not in result
        assert result["row_count"] >= 0

    @pytest.mark.asyncio
    async def test_sql_timeout_with_retry(self, mpp):
        injector = FaultInjector()
        injector.configure(FaultConfig(
            FaultType.SQL_TIMEOUT,
            probability=1.0,  # Always timeout
            duration_ms=100,
        ))
        executor = ResilientSQLExecutor(
            mpp, fault_injector=injector,
            max_retries=2, retry_backoff_ms=50,
        )

        result = await executor.execute(
            "SELECT 1",
            fallback_sql="SELECT 1",
        )
        # Should have retried and eventually fallen through
        assert "error" in result or "fallback" in result

    @pytest.mark.asyncio
    async def test_data_skew_adds_latency(self, mpp):
        injector = FaultInjector()
        injector.configure(FaultConfig(
            FaultType.DATA_SKEW,
            probability=1.0,  # Always trigger
            affected_regions=["APAC"],
        ))
        executor = ResilientSQLExecutor(mpp, fault_injector=injector)

        result = await executor.execute(
            "SELECT COUNT(*) FROM retail_metrics",
            region="APAC",
        )
        # Data skew doesn't fail — just adds latency
        assert "error" not in result

        # Check event log
        events = injector.get_event_log()
        assert len(events) >= 1
        assert events[0]["fault_type"] == "DATA_SKEW"

    @pytest.mark.asyncio
    async def test_region_filter(self, mpp):
        injector = FaultInjector()
        injector.configure(FaultConfig(
            FaultType.SQL_TIMEOUT,
            probability=1.0,
            duration_ms=100,
            affected_regions=["APAC"],  # Only APAC
        ))
        executor = ResilientSQLExecutor(mpp, fault_injector=injector, max_retries=1)

        # NA should NOT be affected
        result_na = await executor.execute("SELECT 1", region="NA")
        assert "error" not in result_na

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, mpp):
        injector = FaultInjector()
        injector.configure(FaultConfig(
            FaultType.NETWORK_PARTITION,
            probability=1.0,
            duration_ms=100,
            affected_regions=["APAC"],
        ))
        executor = ResilientSQLExecutor(mpp, fault_injector=injector, max_retries=1)

        # First call triggers partition → circuit breaker opens
        result1 = await executor.execute("SELECT 1", region="APAC")
        assert "error" in result1

        # Second call should hit circuit breaker immediately
        result2 = await executor.execute("SELECT 1", region="APAC")
        assert "circuit_breaker" in str(result2)

    @pytest.mark.asyncio
    async def test_preset_profiles(self, mpp):
        injector = FaultInjector()
        injector.configure_preset("production_realistic")

        stats = injector.get_stats()
        assert len(stats["active_configs"]) >= 3

    @pytest.mark.asyncio
    async def test_reliability_stats(self, mpp):
        injector = FaultInjector()
        executor = ResilientSQLExecutor(mpp, fault_injector=injector)

        for _ in range(10):
            await executor.execute("SELECT 1")

        stats = executor.get_reliability_stats()
        assert stats["total_executions"] == 10
        assert stats["successes"] == 10
        assert stats["success_rate"] == "100.0%"


# =========================================================================
# Guardrail Tests
# =========================================================================

class TestGuardrails:
    def test_green_zone(self):
        gf = GuardrailFramework()
        result = gf.evaluate({
            "output": "APAC latency averages 185ms",
            "latency_ms": 100,
            "sql": "SELECT * FROM metrics",
            "sandbox": True,
            "confidence": 0.95,
        })
        assert result.zone == SafetyZone.GREEN
        assert result.exploration_allowed is True

    def test_yellow_zone_absolute_language(self):
        gf = GuardrailFramework()
        result = gf.evaluate({
            "output": "The system always performs perfectly",
            "latency_ms": 100,
            "confidence": 0.85,
        })
        assert result.zone == SafetyZone.YELLOW
        assert result.requires_human is True
        assert result.exploration_allowed is True  # YELLOW allows exploration

    def test_red_zone_pii(self):
        gf = GuardrailFramework()
        result = gf.evaluate({
            "output": "Customer SSN is 123-45-6789",
            "latency_ms": 50,
        })
        assert result.zone == SafetyZone.RED
        assert result.exploration_allowed is False
        assert len(result.blocked) > 0

    def test_red_zone_destructive_sql(self):
        gf = GuardrailFramework()
        result = gf.evaluate({
            "output": "Done",
            "sql": "DROP TABLE retail_metrics",
            "latency_ms": 50,
        }, agent="data_architect")
        assert result.zone == SafetyZone.RED

    def test_yellow_zone_low_confidence(self):
        gf = GuardrailFramework()
        result = gf.evaluate({
            "output": "The data suggests moderate latency",
            "latency_ms": 50,
            "confidence": 0.5,
        })
        assert result.zone == SafetyZone.YELLOW
        assert any("confidence" in s.lower() or "review" in s.lower()
                    for s in result.suggestions)

    def test_exploration_evaluation(self):
        gf = GuardrailFramework()
        result = gf.evaluate_exploration({
            "output": "Trying a novel optimization approach",
            "latency_ms": 300,  # Exceeds soft SLA but under hard limit
            "confidence": 0.65,
        })
        assert result.zone == SafetyZone.YELLOW
        assert result.exploration_allowed is True
        assert any("YELLOW ZONE" in s for s in result.suggestions)

    def test_red_zone_blocks_exploration(self):
        gf = GuardrailFramework()
        result = gf.evaluate_exploration({
            "output": "SSN: 123-45-6789",
            "latency_ms": 50,
        })
        assert result.zone == SafetyZone.RED
        assert result.exploration_allowed is False

    def test_stats(self):
        gf = GuardrailFramework()
        gf.evaluate({"output": "Safe output", "latency_ms": 50})
        gf.evaluate({"output": "Always works", "latency_ms": 50})

        stats = gf.get_stats()
        assert stats["evaluations"] == 2
        assert "GREEN" in stats["zone_distribution"] or "YELLOW" in stats["zone_distribution"]

    def test_custom_guardrail_registration(self):
        from verifier.guardrails import Guardrail

        gf = GuardrailFramework()
        gf.register(Guardrail(
            name="custom_check",
            check_fn=lambda d: d.get("score", 0) > 50,
            severity="soft",
            remediation="Increase score above 50",
        ))

        result = gf.evaluate({"output": "Low score", "score": 30, "latency_ms": 50})
        assert any("custom_check" in w for w in result.warnings)
