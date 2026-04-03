"""Pydantic settings management — loads from YAML + environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_seconds: int = 30
    mock_mode: bool = True


class VectorDBSettings(BaseModel):
    provider: str = "milvus"
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "aura_embeddings"
    embedding_dim: int = 384
    metric_type: str = "COSINE"


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"
    max_connections: int = 20
    message_bus_stream: str = "aura:bus"
    memory_prefix: str = "aura:mem:"


class MPPSimulatorSettings(BaseModel):
    engine: str = "sqlite"
    db_path: str = ":memory:"
    synthetic_rows: int = 100_000
    latency_injection_ms: int = 0
    partitions: int = 8


class RaySettings(BaseModel):
    enabled: bool = False
    address: str = "auto"
    num_cpus: int = 4
    dashboard_port: int = 8265


class InferenceSettings(BaseModel):
    max_latency_ms: int = 200
    cache_enabled: bool = True
    cache_max_size: int = 10_000
    quantization: str = "none"


class TelemetrySettings(BaseModel):
    enabled: bool = True
    prometheus_port: int = 9090
    trace_export: str = "console"


class RBACSettings(BaseModel):
    enabled: bool = True
    default_role: str = "viewer"


class PlannerAgentSettings(BaseModel):
    max_react_iterations: int = 10
    tools: list[str] = Field(default_factory=lambda: [
        "execute_sql", "search_schema", "describe_table",
        "analyze_query_plan", "verify_output", "trigger_action",
    ])


class DataArchitectAgentSettings(BaseModel):
    max_sql_retries: int = 3
    schema_search_top_k: int = 5


class VerifierAgentSettings(BaseModel):
    strict_mode: bool = True
    max_constraint_depth: int = 5


class AgentSettings(BaseModel):
    planner: PlannerAgentSettings = Field(default_factory=PlannerAgentSettings)
    data_architect: DataArchitectAgentSettings = Field(default_factory=DataArchitectAgentSettings)
    verifier: VerifierAgentSettings = Field(default_factory=VerifierAgentSettings)


class AppSettings(BaseModel):
    name: str = "Aura"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"


class AuraSettings(BaseModel):
    """Root settings model for the entire Aura system."""

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    mpp_simulator: MPPSimulatorSettings = Field(default_factory=MPPSimulatorSettings)
    ray: RaySettings = Field(default_factory=RaySettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    rbac: RBACSettings = Field(default_factory=RBACSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict on failure."""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _apply_env_overrides(data: dict[str, Any], prefix: str = "AURA") -> dict[str, Any]:
    """
    Overlay environment variables onto the config dict.
    E.g., AURA_LLM_MOCK_MODE=true → data["llm"]["mock_mode"] = True.
    """
    for key, value in os.environ.items():
        if not key.startswith(f"{prefix}_"):
            continue
        parts = key[len(prefix) + 1 :].lower().split("_")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                break
        else:
            final_key = parts[-1]
            # Type coercion
            if value.lower() in ("true", "false"):
                target[final_key] = value.lower() == "true"
            elif value.isdigit():
                target[final_key] = int(value)
            else:
                try:
                    target[final_key] = float(value)
                except ValueError:
                    target[final_key] = value
    return data


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> AuraSettings:
    """
    Load settings from YAML, overlay environment variables, and return
    a validated Pydantic model.  Cached after first call.
    """
    if config_path is None:
        config_path = os.environ.get(
            "AURA_CONFIG_PATH",
            str(Path(__file__).resolve().parents[2] / "config" / "settings.yaml"),
        )
    raw = _load_yaml(Path(config_path))
    raw = _apply_env_overrides(raw)
    return AuraSettings(**raw)
