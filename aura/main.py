"""
Aura — FastAPI Application Entry Point.

Endpoints:
  POST /query         — Submit a business query to the orchestrator
  GET  /traces/{id}   — Retrieve decision-making traces
  GET  /health        — System health and agent status
  GET  /metrics       — Prometheus-compatible metrics
  WS   /ws/stream     — Real-time agent reasoning stream
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from aura.core.bus import AsyncAgentBus, Envelope, Priority
from aura.core.config import get_settings
from aura.core.memory import MemoryStore
from aura.core.telemetry import get_prometheus_metrics
from aura.agents.data_architect import DataArchitectAgent
from aura.agents.planner import PlannerAgent
from aura.agents.verifier import VerifierAgent
from aura.pipelines.executor import ActionExecutor
from aura.pipelines.mpp_engine import MPPEngine
from aura.pipelines.rag import RAGPipeline
from aura.security.rbac import RBACManager
from verifier.engine import VerificationEngine
from verifier.grounding import GroundingManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (initialized in lifespan)
# ---------------------------------------------------------------------------

state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize all system components on startup."""
    settings = get_settings()

    # Core infrastructure
    bus = AsyncAgentBus(
        distributed=settings.ray.enabled,
        redis_url=settings.redis.url,
        stream_key=settings.redis.message_bus_stream,
    )
    await bus.start()

    memory = MemoryStore(
        redis_url=settings.redis.url if not settings.llm.mock_mode else None,
        prefix=settings.redis.memory_prefix,
    )
    await memory.connect()

    # MPP Simulator
    mpp = MPPSimulator(
        db_path=settings.mpp_simulator.db_path,
        synthetic_rows=settings.mpp_simulator.synthetic_rows,
        latency_injection_ms=settings.mpp_simulator.latency_injection_ms,
        partitions=settings.mpp_simulator.partitions,
    )
    await mpp.initialize()

    # Verification engine
    verification_engine = VerificationEngine()
    grounding_manager = GroundingManager()

    # RAG pipeline
    rag = RAGPipeline(mpp_simulator=mpp, top_k=5)

    # Action executor
    executor = ActionExecutor(mpp_simulator=mpp, sandbox_mode=True)

    # RBAC
    rbac = RBACManager(
        enabled=settings.rbac.enabled,
        default_role=settings.rbac.default_role,
    )
    rbac.load_policies()

    # Agents
    data_architect = DataArchitectAgent(
        bus=bus,
        memory=memory,
        mpp_simulator=mpp,
        max_sql_retries=settings.agents.data_architect.max_sql_retries,
        schema_search_top_k=settings.agents.data_architect.schema_search_top_k,
    )
    await data_architect.on_start()

    verifier = VerifierAgent(
        bus=bus,
        memory=memory,
        verification_engine=verification_engine,
        grounding_manager=grounding_manager,
        strict_mode=settings.agents.verifier.strict_mode,
    )
    await verifier.on_start()

    planner = PlannerAgent(
        bus=bus,
        memory=memory,
        max_react_iterations=settings.agents.planner.max_react_iterations,
        available_tools=settings.agents.planner.tools,
    )
    await planner.on_start()

    # Store in state
    state.update({
        "bus": bus,
        "memory": memory,
        "mpp": mpp,
        "rag": rag,
        "executor": executor,
        "rbac": rbac,
        "planner": planner,
        "data_architect": data_architect,
        "verifier": verifier,
        "verification_engine": verification_engine,
        "grounding_manager": grounding_manager,
        "traces": {},
        "start_time": time.time(),
    })

    logger.info("Aura system initialized — all agents online")
    yield

    # Shutdown
    await data_architect.on_stop()
    await verifier.on_stop()
    await planner.on_stop()
    await bus.shutdown()
    await memory.close()
    await mpp.close()
    logger.info("Aura system shut down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Aura — Autonomous Multi-Agent MPP Orchestrator",
    version="0.1.0",
    description="Production-grade multi-agent system for enterprise data environments",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="Business query in natural language")
    role: str = Field(default="analyst", description="User role for RBAC")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    mode: str = Field(default="full", description="Execution mode: full | data_only | verify_only")


class QueryResponse(BaseModel):
    trace_id: str
    status: str
    output: Any
    agents_used: list[str]
    total_latency_ms: float
    actions_taken: list[dict[str, Any]]
    verification: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    agents: dict[str, str]
    bus_metrics: dict[str, Any]
    mpp_tables: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def submit_query(req: QueryRequest) -> QueryResponse:
    """Submit a business query to the Aura orchestrator."""
    rbac: RBACManager = state["rbac"]
    planner: PlannerAgent = state["planner"]
    executor: ActionExecutor = state["executor"]

    # RBAC check
    if not rbac.check_permission(req.role, "agents", "invoke:planner"):
        raise HTTPException(403, f"Role '{req.role}' cannot invoke planner")

    start = time.time()

    # Run the planner agent
    context = {**req.context, "role": req.role, "mode": req.mode}
    trace = await planner.run(req.query, context)

    # Check for autonomous actions
    actions_taken = []
    if req.mode == "full":
        action_records = await executor.monitor_and_act()
        actions_taken = [r.to_dict() for r in action_records]

    total_ms = (time.time() - start) * 1000

    # Store trace
    state["traces"][trace.trace_id] = trace.to_dict()

    return QueryResponse(
        trace_id=trace.trace_id,
        status=trace.status,
        output=trace.final_output,
        agents_used=[trace.agent_name],
        total_latency_ms=round(total_ms, 2),
        actions_taken=actions_taken,
    )


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """Retrieve a decision-making trace by ID."""
    # Check local planner traces
    planner: PlannerAgent = state["planner"]
    trace = planner.get_trace(trace_id)
    if trace:
        return trace.to_dict()

    # Check stored traces
    stored = state["traces"].get(trace_id)
    if stored:
        return stored

    raise HTTPException(404, f"Trace {trace_id} not found")


@app.get("/traces")
async def list_traces(limit: int = 20) -> list[dict[str, Any]]:
    """List recent traces."""
    planner: PlannerAgent = state["planner"]
    all_traces = planner.get_all_traces()
    return all_traces[-limit:]


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health and agent status."""
    bus: AsyncAgentBus = state["bus"]
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=round(time.time() - state["start_time"], 1),
        agents={
            "planner": "active",
            "data_architect": "active",
            "verifier": "active",
        },
        bus_metrics=bus.get_metrics(),
        mpp_tables=3,
    )


@app.get("/metrics")
async def prometheus_metrics() -> PlainTextResponse:
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(
        get_prometheus_metrics().decode(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/actions")
async def list_actions() -> list[dict[str, Any]]:
    """List all executed autonomous actions."""
    executor: ActionExecutor = state["executor"]
    return executor.get_action_log()


@app.post("/actions/{action_id}/rollback")
async def rollback_action(action_id: str) -> dict[str, Any]:
    """Rollback a previously executed action."""
    executor: ActionExecutor = state["executor"]
    return await executor.rollback(action_id)


@app.get("/rag/retrieve")
async def rag_retrieve(
    query: str,
    mode: str = "hybrid",
    role: str = "analyst",
) -> dict[str, Any]:
    """Direct RAG retrieval endpoint."""
    rag: RAGPipeline = state["rag"]
    return await rag.retrieve(query, mode=mode, role=role)


@app.get("/schema/search")
async def search_schema(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search table schemas."""
    mpp: MPPSimulator = state["mpp"]
    return await mpp.search_schema(query, top_k=top_k)


@app.get("/rbac/roles")
async def list_roles() -> list[dict[str, Any]]:
    """List available RBAC roles."""
    rbac: RBACManager = state["rbac"]
    return rbac.list_roles()


# ---------------------------------------------------------------------------
# WebSocket — Real-time agent stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def agent_stream(websocket: WebSocket) -> None:
    """Stream real-time agent reasoning steps."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            role = data.get("role", "analyst")

            planner: PlannerAgent = state["planner"]
            trace = await planner.run(query, {"role": role})

            # Stream each step
            for step in trace.steps:
                await websocket.send_json({
                    "type": step.step_type.name,
                    "content": step.content,
                    "tool": step.tool_name,
                    "latency_ms": step.latency_ms,
                })

            # Final result
            await websocket.send_json({
                "type": "COMPLETE",
                "trace_id": trace.trace_id,
                "output": trace.final_output,
                "total_latency_ms": trace.total_latency_ms,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
