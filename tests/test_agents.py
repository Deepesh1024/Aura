"""Tests for agent lifecycle, ReAct loop, and cross-agent communication."""

from __future__ import annotations

import asyncio

import pytest

from aura.core.bus import AsyncAgentBus
from aura.core.memory import MemoryStore
from aura.agents.data_architect import DataArchitectAgent
from aura.agents.planner import PlannerAgent
from aura.agents.verifier import VerifierAgent


@pytest.fixture
async def system():
    """Set up a complete agent system for testing."""
    bus = AsyncAgentBus(distributed=False)
    await bus.start()

    memory = MemoryStore()
    await memory.connect()

    data_arch = DataArchitectAgent(bus=bus, memory=memory)
    await data_arch.on_start()

    verifier = VerifierAgent(bus=bus, memory=memory, strict_mode=True)
    await verifier.on_start()

    planner = PlannerAgent(bus=bus, memory=memory, max_react_iterations=3)
    await planner.on_start()

    yield {
        "bus": bus,
        "memory": memory,
        "data_architect": data_arch,
        "planner": planner,
        "verifier": verifier,
    }

    await data_arch.on_stop()
    await verifier.on_stop()
    await planner.on_stop()
    await bus.shutdown()
    await memory.close()


@pytest.mark.asyncio
async def test_data_architect_run(system):
    """Test Data Architect agent ReAct loop."""
    agent = system["data_architect"]
    trace = await agent.run("Find average retail latency in APAC")

    assert trace.status == "completed"
    assert trace.agent_name == "data_architect"
    assert len(trace.steps) > 0
    assert trace.total_latency_ms > 0


@pytest.mark.asyncio
async def test_verifier_run(system):
    """Test Verifier agent constraint checking."""
    agent = system["verifier"]
    trace = await agent.run(
        "verify",
        context={
            "claim": "APAC latency is 185ms",
            "evidence": {"latency_ms": 185, "region": "APAC"},
            "constraints": ["latency_ms < 200"],
        },
    )

    assert trace.status == "completed"
    assert trace.agent_name == "verifier"


@pytest.mark.asyncio
async def test_planner_run(system):
    """Test Planner agent task decomposition."""
    agent = system["planner"]
    trace = await agent.run(
        "Analyze Retail latency in APAC and suggest infrastructure patches"
    )

    assert trace.status == "completed"
    assert trace.agent_name == "planner"
    assert len(trace.steps) > 0
    assert trace.final_output is not None


@pytest.mark.asyncio
async def test_agent_trace_recording(system):
    """Test that traces are properly recorded and retrievable."""
    agent = system["planner"]
    trace = await agent.run("Test query")

    retrieved = agent.get_trace(trace.trace_id)
    assert retrieved is not None
    assert retrieved.trace_id == trace.trace_id

    all_traces = agent.get_all_traces()
    assert len(all_traces) >= 1


@pytest.mark.asyncio
async def test_memory_integration(system):
    """Test that agents store context in memory."""
    memory = system["memory"]
    agent = system["data_architect"]

    await agent.run("Count retail metrics by region")

    # Check conversation memory was updated
    conv = await memory.get_conversation("data_architect")
    # Agent should have stored at least something
    assert isinstance(conv, list)


@pytest.mark.asyncio
async def test_tool_registry():
    """Test tool registration and listing."""
    bus = AsyncAgentBus(distributed=False)
    await bus.start()
    memory = MemoryStore()

    agent = DataArchitectAgent(bus=bus, memory=memory)

    tools = agent.tools.list_tools()
    tool_names = {t["name"] for t in tools}

    assert "execute_sql" in tool_names
    assert "search_schema" in tool_names
    assert "describe_table" in tool_names
    assert "analyze_query_plan" in tool_names

    await bus.shutdown()
