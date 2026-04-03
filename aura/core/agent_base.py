"""
Base Agent with ReAct (Reason + Act) loop, tool registry, and message bus integration.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine

from aura.core.bus import AsyncAgentBus, Envelope, Priority

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

ToolFunction = Callable[..., Coroutine[Any, Any, Any]]


@dataclass(slots=True)
class ToolSpec:
    """Specification for a tool available to an agent."""
    name: str
    description: str
    function: ToolFunction
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_permission: str | None = None


class ToolRegistry:
    """Dynamic tool registry with permission-aware lookup."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        logger.debug("Registered tool: %s", spec.name)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self, role: str | None = None) -> list[dict[str, Any]]:
        """List available tools, optionally filtered by role permissions."""
        result = []
        for spec in self._tools.values():
            if spec.requires_permission and role:
                # Simplified check — real impl delegates to RBAC module
                pass
            result.append({
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            })
        return result

    async def execute(self, name: str, **kwargs: Any) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise ValueError(f"Unknown tool: {name}")
        return await spec.function(**kwargs)


# ---------------------------------------------------------------------------
# ReAct Step Types
# ---------------------------------------------------------------------------

class StepType(Enum):
    OBSERVE = auto()
    THINK = auto()
    ACT = auto()
    REFLECT = auto()


@dataclass(slots=True)
class ReActStep:
    """A single step in the ReAct loop."""
    step_type: StepType
    content: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentTrace:
    """Full execution trace for observability and debugging."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_name: str = ""
    query: str = ""
    steps: list[ReActStep] = field(default_factory=list)
    final_output: Any = None
    total_latency_ms: float = 0.0
    status: str = "pending"  # pending | running | completed | failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "query": self.query,
            "steps": [
                {
                    "type": s.step_type.name,
                    "content": s.content,
                    "tool_name": s.tool_name,
                    "tool_input": s.tool_input,
                    "tool_output": str(s.tool_output) if s.tool_output else None,
                    "latency_ms": s.latency_ms,
                }
                for s in self.steps
            ],
            "final_output": self.final_output,
            "total_latency_ms": self.total_latency_ms,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Abstract base agent implementing the ReAct (Reason + Act) loop.

    Lifecycle:
        on_start() → [observe → think → act → reflect]* → on_stop()

    Subclasses implement:
        - observe(): gather context
        - think(): reason about next action
        - act(): execute tools
        - reflect(): evaluate results
        - should_stop(): termination condition
    """

    def __init__(
        self,
        name: str,
        bus: AsyncAgentBus,
        max_iterations: int = 10,
    ) -> None:
        self.name = name
        self.bus = bus
        self.max_iterations = max_iterations
        self.tools = ToolRegistry()
        self._running = False
        self._traces: dict[str, AgentTrace] = {}

    # ---- lifecycle hooks ----

    async def on_start(self) -> None:
        """Called once when the agent starts. Override for setup."""
        self._running = True
        # Subscribe to messages directed at this agent
        self.bus.subscribe(self.name, self._handle_message)
        logger.info("Agent [%s] started", self.name)

    async def on_stop(self) -> None:
        """Called once when the agent stops. Override for cleanup."""
        self._running = False
        logger.info("Agent [%s] stopped", self.name)

    async def on_error(self, error: Exception, trace: AgentTrace) -> None:
        """Called when an error occurs during the ReAct loop."""
        logger.exception("Agent [%s] error: %s", self.name, error)
        trace.status = "failed"

    # ---- ReAct loop steps (override in subclasses) ----

    @abstractmethod
    async def observe(self, query: str, context: dict[str, Any]) -> str:
        """Gather observations and context relevant to the query."""

    @abstractmethod
    async def think(
        self, query: str, observations: str, history: list[ReActStep]
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        """
        Reason about what to do next.
        Returns: (thought, tool_name_or_none, tool_input_or_none)
        """

    @abstractmethod
    async def act(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Execute a tool and return the result."""

    @abstractmethod
    async def reflect(
        self, query: str, history: list[ReActStep], tool_output: Any
    ) -> tuple[str, bool]:
        """
        Evaluate the result.
        Returns: (reflection, should_stop)
        """

    def should_stop(self, iteration: int, history: list[ReActStep]) -> bool:
        """Default stop condition: max iterations reached."""
        return iteration >= self.max_iterations

    # ---- main execution ----

    async def run(self, query: str, context: dict[str, Any] | None = None) -> AgentTrace:
        """
        Execute the full ReAct loop for a given query.
        Returns a complete AgentTrace for observability.
        """
        context = context or {}
        trace = AgentTrace(agent_name=self.name, query=query, status="running")
        self._traces[trace.trace_id] = trace
        start_time = time.time()

        try:
            for iteration in range(self.max_iterations):
                # 1. Observe
                step_start = time.time()
                observations = await self.observe(query, context)
                observe_step = ReActStep(
                    step_type=StepType.OBSERVE,
                    content=observations,
                    latency_ms=(time.time() - step_start) * 1000,
                )
                trace.steps.append(observe_step)

                # 2. Think
                step_start = time.time()
                thought, tool_name, tool_input = await self.think(
                    query, observations, trace.steps
                )
                think_step = ReActStep(
                    step_type=StepType.THINK,
                    content=thought,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    latency_ms=(time.time() - step_start) * 1000,
                )
                trace.steps.append(think_step)

                # 3. Act (if tool selected)
                tool_output = None
                if tool_name and tool_input is not None:
                    step_start = time.time()
                    tool_output = await self.act(tool_name, tool_input)
                    act_step = ReActStep(
                        step_type=StepType.ACT,
                        content=f"Executed {tool_name}",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_output=tool_output,
                        latency_ms=(time.time() - step_start) * 1000,
                    )
                    trace.steps.append(act_step)

                # 4. Reflect
                step_start = time.time()
                reflection, done = await self.reflect(
                    query, trace.steps, tool_output
                )
                reflect_step = ReActStep(
                    step_type=StepType.REFLECT,
                    content=reflection,
                    latency_ms=(time.time() - step_start) * 1000,
                )
                trace.steps.append(reflect_step)

                if done or self.should_stop(iteration + 1, trace.steps):
                    trace.final_output = reflection
                    break

            trace.status = "completed"

        except Exception as e:
            await self.on_error(e, trace)
            trace.final_output = str(e)

        trace.total_latency_ms = (time.time() - start_time) * 1000
        return trace

    # ---- message bus integration ----

    async def _handle_message(self, envelope: Envelope) -> None:
        """Handle incoming messages from the bus."""
        query = envelope.payload.get("query", "")
        context = envelope.payload.get("context", {})
        context["_correlation_id"] = envelope.correlation_id
        context["_source"] = envelope.source

        trace = await self.run(query, context)

        # Send reply if this was a request
        reply_topic = f"_reply.{envelope.correlation_id}"
        reply = Envelope(
            source=self.name,
            target=envelope.source,
            payload={"trace": trace.to_dict(), "output": trace.final_output},
            correlation_id=envelope.correlation_id,
            parent_id=envelope.correlation_id,
            priority=envelope.priority,
        )
        await self.bus.publish(reply_topic, reply)

    async def send_to(
        self,
        target: str,
        payload: dict[str, Any],
        priority: Priority = Priority.NORMAL,
        timeout: float = 30.0,
    ) -> Envelope:
        """Send a request to another agent and wait for response."""
        envelope = Envelope(
            source=self.name,
            target=target,
            payload=payload,
            priority=priority,
        )
        return await self.bus.request(target, envelope, timeout=timeout)

    def get_trace(self, trace_id: str) -> AgentTrace | None:
        return self._traces.get(trace_id)

    def get_all_traces(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._traces.values()]
