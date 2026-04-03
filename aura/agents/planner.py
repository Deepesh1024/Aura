"""
Reasoning & Planning Agent — ReAct loop with task decomposition and tool orchestration.

Responsible for:
  1. Decomposing complex business queries into executable sub-goals
  2. Coordinating other agents via the message bus
  3. Maintaining long-term memory for multi-turn context
  4. Selecting and sequencing tools dynamically
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from aura.core.agent_base import BaseAgent, ReActStep, ToolSpec
from aura.core.bus import AsyncAgentBus, Envelope, Priority
from aura.core.memory import MemoryStore
from aura.core.telemetry import AGENT_STEP_LATENCY, LLM_LATENCY

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    The central orchestrator agent.  Uses a full ReAct loop to:
      - Decompose complex queries into sub-tasks
      - Delegate to Data Architect and Verifier agents
      - Trigger autonomous actions when bottlenecks detected
      - Maintain conversation context across turns
    """

    def __init__(
        self,
        bus: AsyncAgentBus,
        memory: MemoryStore,
        llm_client: Any = None,
        max_react_iterations: int = 10,
        available_tools: list[str] | None = None,
    ) -> None:
        super().__init__(name="planner", bus=bus, max_iterations=max_react_iterations)
        self.memory = memory
        self.llm = llm_client
        self._available_tools = available_tools or [
            "execute_sql", "search_schema", "describe_table",
            "analyze_query_plan", "verify_output", "trigger_action",
        ]
        self._sub_task_traces: list[dict[str, Any]] = []
        self._register_tools()

    def _register_tools(self) -> None:
        self.tools.register(ToolSpec(
            name="delegate_to_data_architect",
            description="Send a data query to the Data Architect agent",
            function=self._delegate_data_architect,
            parameters={"query": "str", "context": "dict"},
        ))
        self.tools.register(ToolSpec(
            name="delegate_to_verifier",
            description="Send an output to the Neuro-Symbolic Verifier for validation",
            function=self._delegate_verifier,
            parameters={"claim": "str", "evidence": "dict", "constraints": "list"},
        ))
        self.tools.register(ToolSpec(
            name="trigger_action",
            description="Trigger an autonomous remediation action",
            function=self._trigger_action,
            parameters={"action_type": "str", "parameters": "dict"},
        ))
        self.tools.register(ToolSpec(
            name="decompose_task",
            description="Break a complex query into sequenced sub-tasks",
            function=self._decompose_task,
            parameters={"query": "str"},
        ))
        self.tools.register(ToolSpec(
            name="synthesize_results",
            description="Combine results from multiple sub-tasks into a coherent answer",
            function=self._synthesize_results,
            parameters={"results": "list", "original_query": "str"},
        ))

    # ---- ReAct Loop ----

    async def observe(self, query: str, context: dict[str, Any]) -> str:
        """Build comprehensive observation from memory and context."""
        # Retrieve conversation history
        conv = await self.memory.get_conversation(self.name, limit=10)
        conv_text = "\n".join(
            f"  [{e.metadata.get('role', 'system')}] {e.content}" for e in conv
        )

        # Retrieve relevant episodes
        episodes = await self.memory.get_episodes(
            agent_name=self.name, limit=5
        )
        episode_text = "\n".join(
            f"  [{e.metadata.get('event_type', '?')}] {e.content}" for e in episodes
        )

        # Check for any pending sub-task results
        pending = [t for t in self._sub_task_traces if t.get("status") == "pending"]

        observation = (
            f"QUERY: {query}\n\n"
            f"CONVERSATION HISTORY:\n{conv_text or '  (none)'}\n\n"
            f"RECENT EPISODES:\n{episode_text or '  (none)'}\n\n"
            f"CONTEXT: {json.dumps(context, default=str)}\n\n"
            f"PENDING SUB-TASKS: {len(pending)}\n"
            f"AVAILABLE TOOLS: {[t['name'] for t in self.tools.list_tools()]}\n"
        )
        return observation

    async def think(
        self, query: str, observations: str, history: list[ReActStep]
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        """Plan the next action using the LLM."""
        history_summary = self._summarize_history(history)

        prompt = (
            "You are the Planning Agent in the Aura system. Your role is to:\n"
            "1. Decompose complex business queries into executable sub-tasks\n"
            "2. Delegate data operations to the Data Architect agent\n"
            "3. Validate outputs through the Verifier agent\n"
            "4. Trigger autonomous actions when needed\n\n"
            f"CURRENT OBSERVATIONS:\n{observations}\n\n"
            f"EXECUTION HISTORY:\n{history_summary}\n\n"
            "Available actions:\n"
            "  - decompose_task: Break query into sub-tasks\n"
            "  - delegate_to_data_architect: Query data\n"
            "  - delegate_to_verifier: Validate outputs\n"
            "  - trigger_action: Execute remediation\n"
            "  - synthesize_results: Combine sub-task results\n"
            "  - FINISH: Return final answer\n\n"
            "Respond EXACTLY:\n"
            "THOUGHT: <reasoning>\n"
            "ACTION: <tool_name or FINISH>\n"
            "INPUT: <JSON input>\n"
        )

        response = await self._call_llm(prompt)
        return self._parse_react_response(response)

    async def act(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Execute the planned action."""
        start = time.time()
        result = await self.tools.execute(tool_name, **tool_input)
        elapsed_ms = (time.time() - start) * 1000

        AGENT_STEP_LATENCY.labels(
            agent_name=self.name, step_type="act"
        ).observe(elapsed_ms)

        # Log to episodic memory
        await self.memory.add_episode(
            self.name, "tool_execution",
            f"Executed {tool_name} → {str(result)[:200]}",
            tool=tool_name,
            latency_ms=elapsed_ms,
        )

        return result

    async def reflect(
        self, query: str, history: list[ReActStep], tool_output: Any
    ) -> tuple[str, bool]:
        """Evaluate progress and decide whether to continue."""
        prompt = (
            "Evaluate the current progress toward answering the query.\n\n"
            f"QUERY: {query}\n"
            f"LATEST OUTPUT: {str(tool_output)[:500]}\n"
            f"STEPS TAKEN: {len(history)}\n\n"
            "Respond:\n"
            "DONE: <final summary> — if the query is fully answered\n"
            "CONTINUE: <reason> — if more work is needed\n"
        )

        response = await self._call_llm(prompt)

        if response.strip().upper().startswith("DONE"):
            # Store final result in conversation memory
            await self.memory.add_conversation(
                self.name,
                f"Q: {query}\nA: {response}",
                role="assistant",
            )
            return response, True

        return response, False

    # ---- Tool Implementations ----

    async def _delegate_data_architect(
        self, query: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a query to the Data Architect agent via the message bus."""
        try:
            reply = await self.send_to(
                "data_architect",
                payload={"query": query, "context": context or {}},
                priority=Priority.HIGH,
                timeout=30.0,
            )
            return reply.payload
        except TimeoutError:
            logger.error("Data Architect timed out for query: %s", query)
            return {"error": "Data Architect agent timed out", "query": query}

    async def _delegate_verifier(
        self,
        claim: str,
        evidence: dict[str, Any] | None = None,
        constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send an output to the Verifier agent for validation."""
        try:
            reply = await self.send_to(
                "verifier",
                payload={
                    "query": "verify",
                    "context": {
                        "claim": claim,
                        "evidence": evidence or {},
                        "constraints": constraints or [],
                    },
                },
                priority=Priority.HIGH,
                timeout=15.0,
            )
            return reply.payload
        except TimeoutError:
            return {"verdict": "WARN", "reason": "Verifier timed out"}

    async def _trigger_action(
        self, action_type: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Trigger an autonomous remediation action."""
        try:
            reply = await self.send_to(
                "executor",
                payload={
                    "query": "execute_action",
                    "context": {
                        "action_type": action_type,
                        "parameters": parameters or {},
                    },
                },
                priority=Priority.CRITICAL,
                timeout=60.0,
            )
            return reply.payload
        except TimeoutError:
            return {
                "status": "failed",
                "reason": "Executor timed out",
                "action_type": action_type,
            }

    async def _decompose_task(self, query: str) -> list[dict[str, Any]]:
        """Break a complex query into ordered sub-tasks."""
        prompt = (
            "Decompose this business query into executable sub-tasks.\n"
            "Each sub-task should be assigned to an agent.\n\n"
            f"QUERY: {query}\n\n"
            "Return a JSON array of objects with:\n"
            "  - task: description\n"
            "  - agent: data_architect | planner | verifier\n"
            "  - depends_on: list of task indices this depends on\n"
        )

        response = await self._call_llm(prompt)

        try:
            # Try to extract JSON from the response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                tasks = json.loads(response[start:end])
            else:
                tasks = self._default_decomposition(query)
        except (json.JSONDecodeError, ValueError):
            tasks = self._default_decomposition(query)

        self._sub_task_traces = [
            {"task": t, "status": "pending"} for t in tasks
        ]
        return tasks

    async def _synthesize_results(
        self, results: list[Any], original_query: str
    ) -> str:
        """Combine sub-task results into a coherent final answer."""
        prompt = (
            "Synthesize the following results into a coherent answer.\n\n"
            f"ORIGINAL QUERY: {original_query}\n"
            f"RESULTS:\n{json.dumps(results, default=str, indent=2)}\n\n"
            "Provide a clear, actionable summary.\n"
        )
        return await self._call_llm(prompt)

    # ---- Utilities ----

    def _default_decomposition(self, query: str) -> list[dict[str, Any]]:
        """Fallback task decomposition when LLM parsing fails."""
        return [
            {
                "task": f"Analyze data for: {query}",
                "agent": "data_architect",
                "depends_on": [],
            },
            {
                "task": f"Verify findings for: {query}",
                "agent": "verifier",
                "depends_on": [0],
            },
            {
                "task": f"Synthesize and recommend actions for: {query}",
                "agent": "planner",
                "depends_on": [0, 1],
            },
        ]

    def _summarize_history(self, history: list[ReActStep]) -> str:
        """Create a concise summary of execution history."""
        if not history:
            return "(no history yet)"
        lines = []
        for i, step in enumerate(history[-10:]):
            tool_info = f" → {step.tool_name}" if step.tool_name else ""
            lines.append(
                f"  Step {i+1} [{step.step_type.name}]{tool_info}: "
                f"{step.content[:100]}"
            )
        return "\n".join(lines)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with mock fallback."""
        if self.llm is None:
            return self._mock_llm_response(prompt)
        try:
            start = time.time()
            response = await self.llm.acomplete(prompt)
            elapsed_ms = (time.time() - start) * 1000
            LLM_LATENCY.labels(
                agent_name=self.name, model="mock"
            ).observe(elapsed_ms)
            return response
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return f"ERROR: {e}"

    def _mock_llm_response(self, prompt: str) -> str:
        """Deterministic mock responses for testing."""
        if "decompose" in prompt.lower():
            return json.dumps([
                {
                    "task": "Query retail latency metrics for APAC region",
                    "agent": "data_architect",
                    "depends_on": [],
                },
                {
                    "task": "Identify infrastructure bottlenecks from metrics",
                    "agent": "data_architect",
                    "depends_on": [0],
                },
                {
                    "task": "Verify analysis against business constraints",
                    "agent": "verifier",
                    "depends_on": [0, 1],
                },
                {
                    "task": "Generate infrastructure patch recommendations",
                    "agent": "planner",
                    "depends_on": [0, 1, 2],
                },
            ])

        if "synthesize" in prompt.lower():
            return (
                "## Analysis: Retail Latency in APAC\n\n"
                "**Findings:**\n"
                "- Average latency in APAC: 185ms (above 150ms SLA)\n"
                "- Primary bottleneck: Database query serialization on node ap-east-2\n"
                "- Secondary: CDN cache miss rate at 23% (target: <10%)\n\n"
                "**Recommended Patches:**\n"
                "1. Enable read replicas for ap-east-2 (est. -40ms latency)\n"
                "2. Increase CDN edge cache TTL from 300s to 900s\n"
                "3. Deploy query connection pooling (max_connections: 50→200)\n"
            )

        if "evaluate" in prompt.lower() or "sufficient" in prompt.lower():
            return "DONE: Analysis complete with actionable recommendations."

        return (
            "THOUGHT: I should decompose this complex query into sub-tasks.\n"
            "ACTION: decompose_task\n"
            f'INPUT: {{"query": "Analyze Retail latency in APAC"}}\n'
        )

    def _parse_react_response(
        self, response: str
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        """Parse structured ReAct response from LLM."""
        thought = ""
        action = None
        tool_input = None

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("THOUGHT:"):
                thought = line[8:].strip()
            elif line.startswith("ACTION:"):
                action = line[7:].strip()
                if action.upper() == "FINISH":
                    action = None
            elif line.startswith("INPUT:"):
                raw = line[6:].strip()
                try:
                    tool_input = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    tool_input = {"raw": raw}

        return thought, action, tool_input
