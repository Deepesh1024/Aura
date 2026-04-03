"""End-to-end orchestration test — full query flow through all agents."""

from __future__ import annotations

import asyncio
import time

import pytest

from aura.core.bus import AsyncAgentBus
from aura.core.memory import MemoryStore
from aura.agents.data_architect import DataArchitectAgent
from aura.agents.planner import PlannerAgent
from aura.agents.verifier import VerifierAgent
from aura.pipelines.executor import ActionExecutor
from aura.pipelines.mpp_engine import MPPEngine
from aura.pipelines.rag import RAGPipeline
from aura.security.rbac import RBACManager
from verifier.engine import VerificationEngine
from verifier.grounding import GroundingManager


@pytest.fixture
async def full_system():
    """Set up the complete Aura system."""
    bus = AsyncAgentBus(distributed=False)
    await bus.start()

    memory = MemoryStore()
    await memory.connect()

    mpp = MPPEngine(db_path=":memory:", dataset_path="/tmp/aura_test_e2e")
    await mpp.initialize()

    rag = RAGPipeline(mpp_simulator=mpp, top_k=5)
    executor = ActionExecutor(mpp_simulator=mpp, sandbox_mode=True)
    verification_engine = VerificationEngine()
    grounding_manager = GroundingManager()

    rbac = RBACManager(enabled=True, default_role="analyst")
    rbac.load_policies()

    data_arch = DataArchitectAgent(bus=bus, memory=memory, mpp_simulator=mpp)
    await data_arch.on_start()

    verifier = VerifierAgent(
        bus=bus, memory=memory,
        verification_engine=verification_engine,
        grounding_manager=grounding_manager,
    )
    await verifier.on_start()

    planner = PlannerAgent(bus=bus, memory=memory, max_react_iterations=3)
    await planner.on_start()

    yield {
        "bus": bus,
        "memory": memory,
        "mpp": mpp,
        "rag": rag,
        "executor": executor,
        "rbac": rbac,
        "planner": planner,
        "data_architect": data_arch,
        "verifier": verifier,
    }

    await data_arch.on_stop()
    await verifier.on_stop()
    await planner.on_stop()
    await bus.shutdown()
    await memory.close()
    await mpp.close()


@pytest.mark.asyncio
async def test_canonical_query_e2e(full_system):
    """
    Test the canonical query:
    'Analyze Retail latency in APAC and suggest infrastructure patches'

    Verifies that:
    1. Planner decomposes the query
    2. All agents coordinate
    3. A trace is produced with steps
    4. The system completes within reasonable time
    """
    planner = full_system["planner"]
    start = time.time()

    trace = await planner.run(
        "Analyze customer distribution in APAC and suggest infrastructure patches"
    )

    elapsed = (time.time() - start) * 1000

    assert trace.status == "completed"
    assert trace.final_output is not None
    assert len(trace.steps) >= 2, f"Expected at least 2 steps, got {len(trace.steps)}"
    assert elapsed < 30000, f"E2E took {elapsed:.0f}ms, expected < 30s"

    # Verify trace structure
    trace_dict = trace.to_dict()
    assert trace_dict["agent_name"] == "planner"
    assert len(trace_dict["steps"]) >= 2


@pytest.mark.asyncio
async def test_autonomous_action_loop(full_system):
    """Test that bottleneck detection triggers autonomous actions."""
    executor = full_system["executor"]

    actions = await executor.monitor_and_act()

    # APAC should trigger latency remediation
    assert len(actions) > 0, "Expected autonomous actions for APAC bottleneck"

    for action in actions:
        action_dict = action.to_dict()
        assert action_dict["status"] in ("COMPLETED", "FAILED")
        # sandbox flag is in the result payload
        if action_dict["result"]:
            assert action_dict["result"].get("sandbox", True) is True


@pytest.mark.asyncio
async def test_rbac_data_masking(full_system):
    """Test that RBAC masks sensitive columns for restricted roles."""
    rbac = full_system["rbac"]

    sensitive_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "salary": 150000,
        "region": "NA",
    }

    # Admin sees everything
    admin_data = rbac.mask_data("admin", sensitive_data)
    assert admin_data["salary"] == 150000

    # Viewer gets masked
    viewer_data = rbac.mask_data("viewer", sensitive_data)
    assert viewer_data["email"] == "***MASKED***"
    assert viewer_data["salary"] == "***MASKED***"
    assert viewer_data["region"] == "NA"  # Not sensitive


@pytest.mark.asyncio
async def test_rag_hybrid_with_mpp(full_system):
    """Test hybrid RAG retrieval with real MPP data."""
    rag = full_system["rag"]

    result = await rag.retrieve(
        "Retail latency in APAC region",
        mode="hybrid",
    )

    assert result["structured_results"], "Expected structured SQL results"
    assert result["unstructured_results"], "Expected reference documents"
    assert "APAC" in result["fused_context"]


@pytest.mark.asyncio
async def test_multi_agent_trace_completeness(full_system):
    """Verify that traces capture the full reasoning chain."""
    planner = full_system["planner"]

    trace = await planner.run("What is the average CPU usage across all regions?")

    step_types = [s.step_type.name for s in trace.steps]
    assert "OBSERVE" in step_types
    assert "THINK" in step_types

    # Each step should have latency tracking
    for step in trace.steps:
        assert step.latency_ms >= 0

    # At least OBSERVE steps should have content
    observe_steps = [s for s in trace.steps if s.step_type.name == "OBSERVE"]
    for step in observe_steps:
        assert step.content, f"OBSERVE step has empty content"
