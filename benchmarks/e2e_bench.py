"""
E2E Latency Benchmark — Measures REAL end-to-end latency from user query
to autonomous action, not just message bus speed.

This is the benchmark that actually matters.
Bus latency (0.55ms) is a vanity metric if total E2E exceeds 2000ms.

What's measured:
  1. Full orchestration: Query → Planner → Data Architect → Verifier → Action
  2. Individual component breakdown (where time actually goes)
  3. Fault tolerance under simulated production failures
  4. Memory budget utilization during long-running tasks
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

from aura.core.bus import AsyncAgentBus
from aura.core.budget import MemoryBudgetManager, MemoryTier
from aura.core.memory import MemoryStore
from aura.agents.data_architect import DataArchitectAgent
from aura.agents.planner import PlannerAgent
from aura.agents.verifier import VerifierAgent
from aura.pipelines.executor import ActionExecutor
from aura.pipelines.fault_injection import FaultInjector, ResilientSQLExecutor
from aura.pipelines.mpp_engine import MPPEngine
from aura.pipelines.rag import RAGPipeline
from verifier.engine import VerificationEngine
from verifier.grounding import GroundingManager
from verifier.guardrails import GuardrailFramework, SafetyZone


async def setup_system() -> dict[str, Any]:
    """Initialize the full Aura system for benchmarking."""
    bus = AsyncAgentBus(distributed=False)
    await bus.start()

    memory = MemoryStore()
    await memory.connect()

    mpp = MPPEngine(db_path=":memory:", dataset_path="/tmp/aura_tpcds_bench")
    await mpp.initialize()

    # Fault injection — production-realistic profile
    faults = FaultInjector()
    faults.configure_preset("production_realistic")

    # Resilient executor wrapping the MPP
    resilient = ResilientSQLExecutor(mpp, fault_injector=faults, max_retries=3)

    rag = RAGPipeline(mpp_simulator=mpp, top_k=5)
    executor = ActionExecutor(mpp_simulator=mpp, sandbox_mode=True)
    verification_engine = VerificationEngine()
    grounding_manager = GroundingManager()
    guardrails = GuardrailFramework()
    budget = MemoryBudgetManager(total_budget_tokens=8192)

    data_arch = DataArchitectAgent(bus=bus, memory=memory, mpp_simulator=mpp)
    await data_arch.on_start()

    verifier = VerifierAgent(
        bus=bus, memory=memory,
        verification_engine=verification_engine,
        grounding_manager=grounding_manager,
    )
    await verifier.on_start()

    planner = PlannerAgent(bus=bus, memory=memory, max_react_iterations=5)
    await planner.on_start()

    return {
        "bus": bus, "memory": memory, "mpp": mpp, "faults": faults,
        "resilient": resilient, "rag": rag, "executor": executor,
        "guardrails": guardrails, "budget": budget,
        "data_architect": data_arch, "verifier": verifier,
        "planner": planner,
    }


async def teardown_system(system: dict[str, Any]) -> None:
    await system["data_architect"].on_stop()
    await system["verifier"].on_stop()
    await system["planner"].on_stop()
    await system["bus"].shutdown()
    await system["mpp"].close()


# ---------------------------------------------------------------------------
# Benchmark 1: Full E2E Orchestration
# ---------------------------------------------------------------------------

async def bench_e2e_orchestration(
    system: dict[str, Any], iterations: int = 20
) -> dict[str, Any]:
    """
    Benchmark the REAL E2E latency:
    User query → Planner decomposition → Data Architect SQL → Verifier check → Action
    """
    planner = system["planner"]
    executor = system["executor"]
    guardrails = system["guardrails"]

    queries = [
        "Analyze Retail latency in APAC and suggest infrastructure patches",
        "What is the average CPU usage across all regions?",
        "Find all high-severity incidents and suggest remediation",
        "Compare retail throughput between NA and APAC",
        "Identify degraded infrastructure nodes and plan maintenance",
    ]

    e2e_latencies: list[float] = []
    component_times: dict[str, list[float]] = {
        "planner_ms": [], "action_ms": [], "guardrail_ms": [],
    }
    zones: dict[str, int] = {"GREEN": 0, "YELLOW": 0, "RED": 0}

    for i in range(iterations):
        query = queries[i % len(queries)]
        total_start = time.time()

        # Phase 1: Planner orchestration (includes Data Architect + Verifier calls)
        plan_start = time.time()
        trace = await planner.run(query)
        plan_ms = (time.time() - plan_start) * 1000
        component_times["planner_ms"].append(plan_ms)

        # Phase 2: Guardrail check
        gr_start = time.time()
        gr_result = guardrails.evaluate(
            {"output": str(trace.final_output), "latency_ms": plan_ms},
            agent="planner",
        )
        gr_ms = (time.time() - gr_start) * 1000
        component_times["guardrail_ms"].append(gr_ms)
        zones[gr_result.zone.name] = zones.get(gr_result.zone.name, 0) + 1

        # Phase 3: Autonomous action (if GREEN/YELLOW zone)
        action_ms = 0.0
        if gr_result.exploration_allowed:
            act_start = time.time()
            await executor.monitor_and_act()
            action_ms = (time.time() - act_start) * 1000
        component_times["action_ms"].append(action_ms)

        total_ms = (time.time() - total_start) * 1000
        e2e_latencies.append(total_ms)

    sorted_lat = sorted(e2e_latencies)
    n = len(sorted_lat)

    return {
        "iterations": iterations,
        "e2e_p50_ms": round(sorted_lat[n // 2], 2),
        "e2e_p95_ms": round(sorted_lat[int(n * 0.95)], 2),
        "e2e_p99_ms": round(sorted_lat[int(n * 0.99)], 2),
        "e2e_max_ms": round(max(sorted_lat), 2),
        "e2e_mean_ms": round(statistics.mean(sorted_lat), 2),
        "component_breakdown": {
            k: round(statistics.mean(v), 2) for k, v in component_times.items()
        },
        "guardrail_zones": zones,
        "sla_met_2000ms": sum(1 for l in e2e_latencies if l <= 2000),
        "sla_met_200ms": sum(1 for l in e2e_latencies if l <= 200),
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Component Breakdown
# ---------------------------------------------------------------------------

async def bench_component_breakdown(
    system: dict[str, Any], iterations: int = 50
) -> dict[str, Any]:
    """
    Isolate individual component latencies to identify bottlenecks.
    This answers: "Where does the time actually go?"
    """
    mpp = system["mpp"]
    rag = system["rag"]
    data_arch = system["data_architect"]
    guardrails = system["guardrails"]

    results: dict[str, list[float]] = {
        "sql_execution": [], "rag_hybrid": [],
        "agent_react": [], "guardrail_eval": [],
    }

    for _ in range(iterations):
        # SQL execution
        t = time.time()
        await mpp.execute(
            "SELECT region, AVG(latency_ms) FROM retail_metrics "
            "WHERE region = 'APAC' GROUP BY region"
        )
        results["sql_execution"].append((time.time() - t) * 1000)

        # RAG hybrid retrieval
        t = time.time()
        await rag.retrieve("APAC retail latency analysis", mode="hybrid")
        results["rag_hybrid"].append((time.time() - t) * 1000)

        # Single agent ReAct cycle
        t = time.time()
        await data_arch.run("Find APAC latency")
        results["agent_react"].append((time.time() - t) * 1000)

        # Guardrail evaluation
        t = time.time()
        guardrails.evaluate(
            {"output": "APAC latency is 185ms", "latency_ms": 50, "confidence": 0.9}
        )
        results["guardrail_eval"].append((time.time() - t) * 1000)

    return {
        component: {
            "mean_ms": round(statistics.mean(lats), 4),
            "p50_ms": round(sorted(lats)[len(lats) // 2], 4),
            "p99_ms": round(sorted(lats)[int(len(lats) * 0.99)], 4),
        }
        for component, lats in results.items()
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Memory Budget Under Load
# ---------------------------------------------------------------------------

async def bench_memory_saturation(
    system: dict[str, Any], num_turns: int = 100
) -> dict[str, Any]:
    """
    Simulate a long-running multi-agent task and verify that the
    memory budget manager prevents context-window saturation.
    """
    budget = system["budget"]
    budget.reset()

    # Simulate grounding data (always stays)
    budget.add_context(
        "shared", "SLA: p99 latency < 200ms. APAC nodes: 312.",
        tier=MemoryTier.GROUNDING, shared=True,
    )

    peak_tokens = 0
    eviction_count = 0

    for turn in range(num_turns):
        agent = ["planner", "data_architect", "verifier"][turn % 3]

        # Simulate conversation and tool outputs
        budget.add_context(
            agent,
            f"Turn {turn}: Agent {agent} processed query about APAC "
            f"infrastructure metrics and found latency={150 + turn}ms",
            tier=MemoryTier.CONVERSATION,
        )

        if turn % 3 == 0:
            budget.add_context(
                agent,
                f"SQL Result: SELECT region, AVG(latency_ms) FROM retail_metrics "
                f"WHERE region = 'APAC' → avg_lat={185 + (turn % 20)}ms",
                tier=MemoryTier.TOOL_OUTPUT,
            )

        report = budget.get_budget_report()
        current_tokens = report["used_tokens"]
        peak_tokens = max(peak_tokens, current_tokens)
        eviction_count = report["evictions"]

    final_report = budget.get_budget_report()
    return {
        "total_turns": num_turns,
        "final_tokens": final_report["used_tokens"],
        "peak_tokens": peak_tokens,
        "budget_limit": final_report["total_budget"],
        "never_exceeded": final_report["used_tokens"] <= final_report["total_budget"],
        "utilization": final_report["utilization"],
        "evictions": eviction_count,
        "tier_breakdown": final_report["tier_breakdown"],
        "per_agent": final_report["per_agent"],
    }


# ---------------------------------------------------------------------------
# Benchmark 4: Fault Tolerance
# ---------------------------------------------------------------------------

async def bench_fault_tolerance(
    system: dict[str, Any], iterations: int = 50
) -> dict[str, Any]:
    """
    Benchmark system behavior under production-realistic fault injection.
    """
    resilient = system["resilient"]
    faults = system["faults"]
    faults.reset()

    queries = [
        ("SELECT AVG(latency_ms) FROM retail_metrics WHERE region = 'APAC'",
         "SELECT AVG(latency_ms) FROM retail_metrics"),
        ("SELECT COUNT(*) FROM infrastructure_nodes WHERE status = 'degraded'",
         "SELECT COUNT(*) FROM infrastructure_nodes"),
    ]

    successes = 0
    fallbacks = 0
    failures = 0

    for i in range(iterations):
        sql, fallback = queries[i % len(queries)]
        result = await resilient.execute(sql, region="APAC", fallback_sql=fallback)

        if "error" in result:
            failures += 1
        elif result.get("fallback"):
            fallbacks += 1
        else:
            successes += 1

    stats = resilient.get_reliability_stats()
    fault_stats = faults.get_stats()

    return {
        "iterations": iterations,
        "successes": successes,
        "fallbacks": fallbacks,
        "failures": failures,
        "success_rate": f"{(successes + fallbacks) / iterations * 100:.1f}%",
        "fault_stats": fault_stats,
        "reliability": stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 70)
    print("  AURA — E2E Latency & Systems Benchmark Suite")
    print("  (This is what actually matters, not bus speed)")
    print("=" * 70)

    system = await setup_system()

    # 1. E2E Orchestration
    print("\n[1] FULL E2E ORCHESTRATION (20 queries)")
    print("    Query → Planner → Data Architect → Verifier → Action")
    print("-" * 50)
    results = await bench_e2e_orchestration(system, 20)
    for k, v in results.items():
        if k == "component_breakdown":
            print(f"  {'component_breakdown':>30s}:")
            for ck, cv in v.items():
                print(f"    {ck:>28s}: {cv} ms")
        elif k == "guardrail_zones":
            print(f"  {'guardrail_zones':>30s}: {v}")
        else:
            print(f"  {k:>30s}: {v}")

    # 2. Component Breakdown
    print("\n[2] COMPONENT BREAKDOWN (50 iterations)")
    print("    Where does the time actually go?")
    print("-" * 50)
    breakdown = await bench_component_breakdown(system, 50)
    for component, metrics in breakdown.items():
        print(f"  {component:>20s}:  mean={metrics['mean_ms']:.3f}ms  "
              f"p50={metrics['p50_ms']:.3f}ms  p99={metrics['p99_ms']:.3f}ms")

    # 3. Memory Saturation
    print("\n[3] MEMORY BUDGET UNDER LOAD (100-turn multi-agent task)")
    print("    Does context-window saturation prevention work?")
    print("-" * 50)
    mem_results = await bench_memory_saturation(system, 100)
    for k, v in mem_results.items():
        if isinstance(v, dict):
            print(f"  {k:>25s}:")
            for mk, mv in v.items():
                print(f"    {mk:>23s}: {mv}")
        else:
            print(f"  {k:>25s}: {v}")

    # 4. Fault Tolerance
    print("\n[4] FAULT TOLERANCE (50 queries under chaos)")
    print("    Production-realistic failures: timeouts, skew, partitions")
    print("-" * 50)
    fault_results = await bench_fault_tolerance(system, 50)
    for k, v in fault_results.items():
        if isinstance(v, dict):
            print(f"  {k:>25s}:")
            for fk, fv in v.items():
                print(f"    {fk:>23s}: {fv}")
        else:
            print(f"  {k:>25s}: {v}")

    await teardown_system(system)

    print("\n" + "=" * 70)
    print("  E2E Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
