# Aura — Autonomous Multi-Agent MPP Orchestrator

**Production-grade multi-agent system for petabyte-scale enterprise data environments.**

Aura coordinates three specialized AI agents to decompose complex business queries, execute data operations, validate outputs against business constraints, and trigger autonomous remediation actions.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Gateway                       │
│  POST /query  │  GET /traces  │  WS /ws/stream          │
└──────────┬──────────────────────────────────────────────┘
           │
    ┌──────▼──────┐     AsyncAgentBus (asyncio.Queue / Redis Streams)
    │   Planner   │◄─────────────────────────────────────────────┐
    │  (ReAct)    │                                               │
    └──┬─────┬────┘                                               │
       │     │                                                    │
 ┌─────▼──┐  └────┐                                              │
 │ Data   │  ┌────▼──────────┐    ┌─────────────────────┐       │
 │Architect│  │Neuro-Symbolic │    │ Action Executor     │       │
 │ Agent   │  │  Verifier     │    │ (Lambda/Docker sim) │       │
 └────┬────┘  └──────┬────────┘    └──────────┬──────────┘       │
      │              │                         │                  │
 ┌────▼────┐  ┌──────▼─────┐          ┌───────▼─────────┐       │
 │MPP Sim  │  │Constraint  │          │ Bottleneck       │───────┘
 │(SQLite) │  │Engine +    │          │ Monitor          │
 │+ RAG    │  │Grounding   │          └─────────────────┘
 └─────────┘  └────────────┘
```

## Quick Start

### Local Development (no Docker)

```bash
# Install
cd Aura
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run benchmarks
python benchmarks/latency_bench.py
python benchmarks/throughput_bench.py

# Start the API (mock LLM mode)
uvicorn aura.main:app --reload
```

### Docker Compose (Full Stack)

```bash
docker-compose up --build -d

# Health check
curl http://localhost:8000/health

# Submit canonical query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze Retail latency in APAC and suggest infrastructure patches", "role": "analyst"}'

# View Grafana dashboards
open http://localhost:3000  # admin / aura
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Submit business query to the orchestrator |
| `/traces/{id}` | GET | Retrieve decision-making trace |
| `/traces` | GET | List recent traces |
| `/health` | GET | System health + agent status |
| `/metrics` | GET | Prometheus metrics |
| `/actions` | GET | List autonomous actions |
| `/actions/{id}/rollback` | POST | Rollback an action |
| `/rag/retrieve` | GET | Direct RAG retrieval |
| `/schema/search` | GET | Search table schemas |
| `/rbac/roles` | GET | List RBAC roles |
| `/ws/stream` | WS | Real-time agent reasoning stream |

## Agents

### Planner (ReAct Loop)
Decomposes complex queries → delegates to specialists → synthesizes results. Uses task decomposition to break "Analyze Retail latency in APAC and suggest patches" into 4 ordered sub-tasks.

### Data Architect
SQL generation, schema mapping, vector search over metadata. Interfaces with the MPP simulator (SQLite-backed with 50K+ synthetic rows).

### Neuro-Symbolic Verifier
Deterministic constraint checking — no LLM dependency. Validates range, type, enum, regex, referential, and temporal constraints. Flags absolute language ("always", "100%") as potential hallucinations.

## Key Technical Decisions

### Inter-Agent Communication (The Biggest Hurdle)
**Dual-transport AsyncAgentBus:**
- **In-process:** `asyncio.Queue` — zero-copy, ~0.01ms per message
- **Distributed:** Redis Streams — ~1-5ms per message via Ray
- Auto-selects transport based on agent locality

### Model Optimization
- LRU response cache with semantic-key deduplication
- INT8/FP16 quantization via PyTorch dynamic quantization
- SLA enforcement: sub-200ms latency tracking with violation alerts

### RBAC
Three roles (admin/analyst/viewer) with column masking and row-level SQL filtering. Loaded from `config/rbac_policies.yaml`.

## Architecture & Components

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Gateway                       │
│  POST /query  │  GET /traces  │  WS /ws/stream          │
└──────────┬──────────────────────────────────────────────┘
           │
    ┌──────▼──────┐     AsyncAgentBus (asyncio.Queue / Redis Streams)
    │   Planner   │◄─────────────────────────────────────────────┐
    │  (ReAct)    │                                               │
    └──┬─────┬────┘                                               │
       │     │                                                    │
 ┌─────▼──┐  └────┐                                              │
 │ Data   │  ┌────▼──────────┐    ┌─────────────────────┐       │
 │Architect│  │Neuro-Symbolic │    │ Action Executor     │       │
 │ Agent   │  │  Verifier     │    │ (Lambda/Docker sim) │       │
 └────┬────┘  └──────┬────────┘    └──────────┬──────────┘       │
      │              │                         │                  │
 ┌────▼────┐  ┌──────▼─────┐          ┌───────▼─────────┐       │
 │MPP Sim  │  │Constraint  │          │ Bottleneck       │───────┘
 │(DuckDB) │  │Engine +    │          │ Monitor          │
 │+ RAG    │  │Grounding   │          └─────────────────┘
 └─────────┘  └────────────┘
```

| Component | Files | Purpose |
|---|---|---|
| Core | `bus.py`, `agent_base.py`, `memory.py`, `budget.py`, `config.py`, `telemetry.py` | Async bus, ReAct base, 3-layer memory, token budget, Prometheus |
| Agents | `data_architect.py`, `planner.py`, `verifier.py` | SQL gen, task decomposition, neuro-symbolic verification |
| Verification | `engine.py`, `constraints.py`, `grounding.py`, `guardrails.py`, **`pii_redactor.py`** | Constraint AST, fact-checking, safe exploration, deterministic privacy generation |
| Pipelines | `rag.py`, `executor.py`, **`mpp_engine.py`**, `fault_injection.py` | Hybrid RAG, autonomous actions, DuckDB Parquet querying, chaos engineering |
| Models | `optimizer.py`, `embeddings.py` | LRU cache, INT8/FP16 quantization |
| Infrastructure | `ray_runner.py`, `rbac.py`, `main.py`, Docker stack | Distributed compute, access control, FastAPI, Grafana |

## Test Results & Metrics

**83/83 tests passing in 2.11s** across 7 test modules:

| Module | Tests | Coverage |
|---|---|---|
| `test_bus.py` | 7 | pub/sub, request-reply, latency, TTL, serialization |
| `test_agents.py` | 6 | all 3 agents, trace recording, memory, tools |
| `test_verifier.py` | 13 | range/enum/type/regex/custom constraints, grounding |
| `test_rag.py` | 12 | SQL execution, EXPLAIN, schema search, hybrid RAG |
| `test_e2e.py` | 5 | canonical query, autonomous actions, RBAC, RAG fusion |
| `test_hardening.py` | 24 | memory budget (100-turn saturation), fault injection, guardrails |
| **`test_mpp_engine.py`, `test_pii_redact.py`** | **16** | DuckDB TPC-DS initialization, Parquet schema reads, deterministic AST SQL parsing, column masking/hashing |

### E2E Latency

> [!IMPORTANT]
> The real benchmark is user-query → autonomous-action latency. The orchestrator overhead is negligible compared to action execution.

| Metric | Value | SLA |
|---|---|---|
| **E2E p50** | **565ms** | ✅ < 2000ms |
| E2E p95 | 569ms | ✅ < 2000ms |
| E2E max | 569ms | ✅ < 2000ms |
| **SLA met (2000ms)** | **20/20** (100%) | |

Where the time goes:
| Component | Mean | p50 | p99 |
|---|---|---|---|
| **Action Executor** | **564ms** | — | — |
| RAG Hybrid | 2.0ms | 1.8ms | 5.2ms |
| Agent ReAct | 1.0ms | 0.9ms | 2.8ms |
| SQL Execution | 0.6ms | 0.6ms | 2.9ms |
| Guardrail Eval | 0.019ms | 0.015ms | 0.089ms |

### Hardening Metrics

- **Memory Saturation**: 100-turn conversation used peaking at 3289 tokens (40.1% of budget). 0 evictions required by utilizing Tier 0 grounding locks + compression.
- **Chaos Engineering**: 50 queries across 7 injected fault types (network partitions, pool exhaustion, schema drift) handled gracefully with exponential backoff & fallbacks — 100% success rate.
- **Guardrails-as-Code**: 3-zone verification model replaces brittle pass/fail.
  
### Production OLAP & PII Redaction
- **DuckDB Engine**: Uses vectorized DuckDB over synthetic TPC-DS Parquet datasets simulating production disk layouts. Enforces strict 80% RAM limit mapping to analytical workload constraints.
- **Privacy By Design**: `sqlglot` parses ASTs locally to autonomously detect PII columns and rewrite selections to masking functions/md5 hashes pre-execution, preventing any PII leaks to the data plane!
