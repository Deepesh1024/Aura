"""
Data Architect Agent — SQL generation, schema mapping, and vector search.

Interfaces with the MPP simulator and vector database to:
  1. Generate high-efficiency SQL from natural language
  2. Map schemas and resolve column references
  3. Perform semantic search over table/column metadata
"""

from __future__ import annotations

import logging
from typing import Any

from aura.core.agent_base import BaseAgent, ReActStep, ToolSpec
from aura.core.bus import AsyncAgentBus
from aura.core.memory import MemoryStore
from aura.core.telemetry import AGENT_STEP_LATENCY
from verifier.pii_redactor import PIIRedactor

logger = logging.getLogger(__name__)


class DataArchitectAgent(BaseAgent):
    """
    Specialized agent for data infrastructure operations.

    Tools:
      - execute_sql: Execute SQL against the MPP simulator
      - search_schema: Semantic search over table/column metadata
      - describe_table: Get full schema for a table
      - analyze_query_plan: EXPLAIN a SQL query
    """

    def __init__(
        self,
        bus: AsyncAgentBus,
        memory: MemoryStore,
        mpp_simulator: Any = None,
        llm_client: Any = None,
        max_sql_retries: int = 3,
        schema_search_top_k: int = 5,
    ) -> None:
        super().__init__(name="data_architect", bus=bus, max_iterations=5)
        self.memory = memory
        self.mpp = mpp_simulator
        self.llm = llm_client
        self.max_sql_retries = max_sql_retries
        self.schema_search_top_k = schema_search_top_k
        self._register_tools()

    def _register_tools(self) -> None:
        self.tools.register(ToolSpec(
            name="execute_sql",
            description="Execute a SQL query against the MPP data warehouse",
            function=self._execute_sql,
            parameters={"sql": "str"},
        ))
        self.tools.register(ToolSpec(
            name="search_schema",
            description="Semantic search over table and column metadata",
            function=self._search_schema,
            parameters={"query": "str", "top_k": "int"},
        ))
        self.tools.register(ToolSpec(
            name="describe_table",
            description="Get the full schema definition of a table",
            function=self._describe_table,
            parameters={"table_name": "str"},
        ))
        self.tools.register(ToolSpec(
            name="analyze_query_plan",
            description="Get the EXPLAIN plan for a SQL query",
            function=self._analyze_query_plan,
            parameters={"sql": "str"},
        ))

    # ---- ReAct Loop Implementation ----

    async def observe(self, query: str, context: dict[str, Any]) -> str:
        """Gather schema context relevant to the query."""
        # Get recent conversation context
        conv = await self.memory.get_conversation(self.name, limit=5)
        conv_text = "\n".join(e.content for e in conv) if conv else "No prior context."

        # Search for relevant schema information
        schema_results = await self._search_schema(query=query, top_k=self.schema_search_top_k)

        observation = (
            f"Query: {query}\n"
            f"Prior context: {conv_text}\n"
            f"Relevant schema: {schema_results}\n"
            f"Available tools: {[t['name'] for t in self.tools.list_tools()]}"
        )
        return observation

    async def think(
        self, query: str, observations: str, history: list[ReActStep]
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        """Decide whether to generate SQL, search schema, or return results."""
        history_text = "\n".join(
            f"[{s.step_type.name}] {s.content}" for s in history[-6:]
        )

        prompt = (
            "You are a Data Architect Agent. Analyze the query and observations.\n"
            "Based on the context, decide the next action.\n\n"
            f"QUERY: {query}\n"
            f"OBSERVATIONS: {observations}\n"
            f"HISTORY:\n{history_text}\n\n"
            "Respond in this exact format:\n"
            "THOUGHT: <your reasoning>\n"
            "ACTION: <tool_name or FINISH>\n"
            "INPUT: <JSON input for the tool, or final answer if FINISH>\n"
        )

        response = await self._call_llm(prompt)
        return self._parse_react_response(response)

    async def act(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Execute the selected tool."""
        AGENT_STEP_LATENCY.labels(
            agent_name=self.name, step_type="act"
        ).observe(0)  # Updated by decorator in real calls

        return await self.tools.execute(tool_name, **tool_input)

    async def reflect(
        self, query: str, history: list[ReActStep], tool_output: Any
    ) -> tuple[str, bool]:
        """Evaluate whether the SQL results answer the query."""
        if tool_output is None:
            return "No tool was executed. Need to determine next step.", False

        # Check if we have enough data to answer
        prompt = (
            "You are evaluating whether the tool output answers the original query.\n\n"
            f"QUERY: {query}\n"
            f"TOOL OUTPUT: {tool_output}\n\n"
            "Is this sufficient to answer the query?\n"
            "Respond: DONE: <summary> or CONTINUE: <reason to continue>\n"
        )
        response = await self._call_llm(prompt)

        if response.strip().upper().startswith("DONE"):
            await self.memory.add_conversation(
                self.name, f"Answered: {query} → {response}"
            )
            return response, True
        return response, False

    # ---- Tool Implementations ----

    async def _execute_sql(self, sql: str) -> dict[str, Any]:
        """Execute SQL against the MPP simulator/engine with AST-based PII redaction."""
        if self.mpp is None:
            return {"error": "MPP engine not initialized", "sql": sql}

        # 1. Autonomously Redact PII (The Deterministic Privacy Guard)
        redactor = PIIRedactor()
        try:
            safe_sql, redacted_cols = redactor.redact_sql(sql)
            if redacted_cols:
                logger.info("PII Redactor intercepted queries: masked %s", redacted_cols)
        except Exception as e:
            return {"error": f"Privacy check failed: {e}", "sql": sql}

        for attempt in range(self.max_sql_retries):
            try:
                # Execute the safe, rewritten SQL
                result = await self.mpp.execute(safe_sql)
                await self.memory.add_episode(
                    self.name, "sql_execution",
                    f"SQL: {safe_sql} → {result.get('row_count', len(result.get('rows', [])))} rows",
                    attempt=attempt,
                )
                if redacted_cols:
                    result["privacy_notice"] = f"Columns masked: {redacted_cols}"
                return result
            except Exception as e:
                if attempt == self.max_sql_retries - 1:
                    return {"error": str(e), "sql": sql, "attempts": attempt + 1}
                logger.warning("SQL retry %d: %s", attempt + 1, e)

        return {"error": "Max retries exceeded", "sql": sql}

    async def _search_schema(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Semantic search over column/table metadata."""
        if self.mpp is None:
            # Return mock schema for dev
            return [
                {"table": "retail_metrics", "columns": [
                    "region", "latency_ms", "throughput_qps",
                    "timestamp", "service_name",
                ]},
                {"table": "infrastructure_nodes", "columns": [
                    "node_id", "region", "status", "cpu_usage",
                    "memory_usage", "last_patch_date",
                ]},
                {"table": "service_incidents", "columns": [
                    "incident_id", "service_name", "region",
                    "severity", "resolution_time_hours",
                ]},
            ][:top_k]

        return await self.mpp.search_schema(query, top_k=top_k)

    async def _describe_table(self, table_name: str) -> dict[str, Any]:
        """Get full schema for a specific table."""
        if self.mpp is None:
            return {
                "table": table_name,
                "columns": [],
                "partitions": 8,
                "row_count": 100000,
            }
        return await self.mpp.describe_table(table_name)

    async def _analyze_query_plan(self, sql: str) -> dict[str, Any]:
        """Get EXPLAIN output for a SQL query."""
        if self.mpp is None:
            return {
                "sql": sql,
                "plan": "SEQUENTIAL SCAN → FILTER → AGGREGATE",
                "estimated_cost": 1250.0,
                "estimated_rows": 5000,
            }
        return await self.mpp.explain(sql)

    # ---- Utility ----

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM, with mock support for local dev."""
        if self.llm is None:
            # Mock response for development
            return self._mock_llm_response(prompt)

        try:
            response = await self.llm.acomplete(prompt)
            return response
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return f"ERROR: {e}"

    def _mock_llm_response(self, prompt: str) -> str:
        """Generate deterministic mock responses for testing."""
        if "THOUGHT:" in prompt or "next action" in prompt.lower():
            return (
                "THOUGHT: The query asks about retail latency. "
                "I should search for relevant tables first.\n"
                "ACTION: search_schema\n"
                'INPUT: {"query": "retail latency metrics", "top_k": 3}\n'
            )
        if "sufficient" in prompt.lower() or "evaluate" in prompt.lower():
            return "DONE: Found relevant performance data in retail_metrics table."
        return "THOUGHT: Processing query.\nACTION: FINISH\nINPUT: Analysis complete."

    def _parse_react_response(
        self, response: str
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        """Parse THOUGHT/ACTION/INPUT from LLM response."""
        import json as json_mod

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
                    tool_input = json_mod.loads(raw)
                except (json_mod.JSONDecodeError, ValueError):
                    tool_input = {"raw": raw}

        return thought, action, tool_input
