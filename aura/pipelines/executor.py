"""
Autonomous Action Executor — detects bottlenecks and triggers
remediation actions via Docker sandbox or simulated Lambda functions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


@dataclass
class ActionRecord:
    """Audit record for an executed action."""
    action_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    action_type: str = ""
    status: ActionStatus = ActionStatus.PENDING
    parameters: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    triggered_by: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    rollback_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "status": self.status.name,
            "parameters": self.parameters,
            "result": self.result,
            "triggered_by": self.triggered_by,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_ms": (self.completed_at - self.started_at) * 1000 if self.completed_at else 0,
            "rollback_available": self.rollback_available,
        }


class ActionExecutor:
    """
    Autonomous action executor that monitors for bottlenecks and
    triggers remediation scripts in sandboxed environments.

    Supported action types:
      - scale_read_replicas: Simulates enabling read replicas
      - adjust_cache_ttl: Simulates CDN/cache TTL adjustment
      - increase_connection_pool: Simulates connection pool scaling
      - apply_patch: Simulates rolling infrastructure patch
      - run_diagnostic: Executes diagnostic script
    """

    def __init__(
        self,
        mpp_simulator: Any = None,
        sandbox_mode: bool = True,
    ) -> None:
        self.mpp = mpp_simulator
        self.sandbox_mode = sandbox_mode
        self._action_log: list[ActionRecord] = []
        self._action_handlers: dict[str, Any] = {
            "scale_read_replicas": self._action_scale_replicas,
            "adjust_cache_ttl": self._action_adjust_cache,
            "increase_connection_pool": self._action_scale_pool,
            "apply_patch": self._action_apply_patch,
            "run_diagnostic": self._action_run_diagnostic,
            "restart_service": self._action_restart_service,
        }

    async def execute_action(
        self,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        triggered_by: str = "system",
    ) -> ActionRecord:
        """Execute a remediation action."""
        parameters = parameters or {}
        record = ActionRecord(
            action_type=action_type,
            parameters=parameters,
            triggered_by=triggered_by,
            started_at=time.time(),
        )

        handler = self._action_handlers.get(action_type)
        if handler is None:
            record.status = ActionStatus.FAILED
            record.result = {"error": f"Unknown action: {action_type}"}
            record.completed_at = time.time()
            self._action_log.append(record)
            return record

        record.status = ActionStatus.RUNNING
        try:
            result = await handler(parameters)
            record.status = ActionStatus.COMPLETED
            record.result = result
            record.rollback_available = result.get("rollback_available", False)
        except Exception as e:
            record.status = ActionStatus.FAILED
            record.result = {"error": str(e)}
            logger.exception("Action %s failed: %s", action_type, e)

        record.completed_at = time.time()
        self._action_log.append(record)

        logger.info(
            "Action %s [%s] completed in %.1fms — %s",
            record.action_id,
            action_type,
            (record.completed_at - record.started_at) * 1000,
            record.status.name,
        )

        return record

    async def monitor_and_act(self) -> list[ActionRecord]:
        """
        Autonomous monitoring loop: detect bottlenecks and trigger actions.
        Returns list of actions taken.
        """
        if self.mpp is None:
            return []

        bottlenecks = await self.mpp.detect_bottlenecks()
        actions_taken = []

        for bottleneck in bottlenecks:
            btype = bottleneck["type"]
            region = bottleneck.get("region", "unknown")

            if btype == "high_latency":
                # Trigger latency remediation
                actions = self._plan_latency_remediation(bottleneck)
                for action_type, params in actions:
                    record = await self.execute_action(
                        action_type, params, triggered_by="bottleneck_monitor"
                    )
                    actions_taken.append(record)

            elif btype == "degraded_nodes":
                record = await self.execute_action(
                    "run_diagnostic",
                    {"region": region, "check_type": "node_health"},
                    triggered_by="bottleneck_monitor",
                )
                actions_taken.append(record)

        return actions_taken

    def _plan_latency_remediation(
        self, bottleneck: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Generate a remediation plan for latency bottlenecks."""
        region = bottleneck.get("region", "unknown")
        avg_latency = bottleneck.get("avg_latency_ms", 0)
        cache_miss = bottleneck.get("cache_miss_rate", 0)

        actions = []

        # High cache miss → adjust TTL
        if cache_miss > 0.15:
            actions.append((
                "adjust_cache_ttl",
                {
                    "region": region,
                    "current_ttl_seconds": 300,
                    "new_ttl_seconds": 900,
                    "reason": f"Cache miss rate {cache_miss:.2%} exceeds 15% threshold",
                },
            ))

        # High latency → scale read replicas
        if avg_latency > 150:
            actions.append((
                "scale_read_replicas",
                {
                    "region": region,
                    "current_replicas": 2,
                    "target_replicas": 4,
                    "reason": f"Avg latency {avg_latency:.0f}ms exceeds 150ms threshold",
                },
            ))

        # Very high latency → also scale connection pool
        if avg_latency > 200:
            actions.append((
                "increase_connection_pool",
                {
                    "region": region,
                    "current_max": 50,
                    "target_max": 200,
                    "reason": f"Critical latency {avg_latency:.0f}ms > 200ms SLA",
                },
            ))

        return actions

    # ---- Action Handlers ----

    async def _action_scale_replicas(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate scaling read replicas."""
        await asyncio.sleep(0.1)  # Simulate execution time
        return {
            "action": "scale_read_replicas",
            "region": params.get("region"),
            "previous_replicas": params.get("current_replicas", 2),
            "new_replicas": params.get("target_replicas", 4),
            "estimated_latency_reduction_ms": 40,
            "rollback_available": True,
            "sandbox": self.sandbox_mode,
        }

    async def _action_adjust_cache(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate CDN cache TTL adjustment."""
        await asyncio.sleep(0.05)
        return {
            "action": "adjust_cache_ttl",
            "region": params.get("region"),
            "previous_ttl": params.get("current_ttl_seconds", 300),
            "new_ttl": params.get("new_ttl_seconds", 900),
            "estimated_cache_hit_improvement": 0.15,
            "rollback_available": True,
            "sandbox": self.sandbox_mode,
        }

    async def _action_scale_pool(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate connection pool scaling."""
        await asyncio.sleep(0.05)
        return {
            "action": "increase_connection_pool",
            "region": params.get("region"),
            "previous_max": params.get("current_max", 50),
            "new_max": params.get("target_max", 200),
            "rollback_available": True,
            "sandbox": self.sandbox_mode,
        }

    async def _action_apply_patch(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate rolling infrastructure patch."""
        await asyncio.sleep(0.2)

        if self.sandbox_mode:
            return {
                "action": "apply_patch",
                "region": params.get("region"),
                "nodes_patched": params.get("node_count", 10),
                "patch_version": params.get("version", "2024.03.15"),
                "canary_passed": True,
                "rollback_available": True,
                "sandbox": True,
            }

        # In non-sandbox mode, would execute Docker script
        return {"action": "apply_patch", "sandbox": False, "blocked": True}

    async def _action_run_diagnostic(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Run diagnostic check (sandboxed)."""
        await asyncio.sleep(0.1)

        check_type = params.get("check_type", "general")
        region = params.get("region", "unknown")

        return {
            "action": "run_diagnostic",
            "region": region,
            "check_type": check_type,
            "findings": [
                f"DNS resolution in {region}: 12ms (normal)",
                f"DB connection pool utilization: 85% (warning)",
                f"Memory pressure on 3 nodes: moderate",
            ],
            "recommendations": [
                "Increase connection pool max_connections",
                "Schedule memory-heavy batch jobs off-peak",
            ],
            "sandbox": self.sandbox_mode,
        }

    async def _action_restart_service(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate service restart."""
        await asyncio.sleep(0.15)
        return {
            "action": "restart_service",
            "service": params.get("service_name", "unknown"),
            "region": params.get("region", "unknown"),
            "downtime_ms": 250,
            "rollback_available": False,
            "sandbox": self.sandbox_mode,
        }

    # ---- Rollback ----

    async def rollback(self, action_id: str) -> dict[str, Any]:
        """Rollback a previously executed action."""
        record = next(
            (r for r in self._action_log if r.action_id == action_id), None
        )
        if record is None:
            return {"error": f"Action {action_id} not found"}
        if not record.rollback_available:
            return {"error": f"Action {action_id} does not support rollback"}
        if record.status == ActionStatus.ROLLED_BACK:
            return {"error": f"Action {action_id} already rolled back"}

        record.status = ActionStatus.ROLLED_BACK
        return {
            "action_id": action_id,
            "status": "rolled_back",
            "original_action": record.action_type,
        }

    # ---- Query ----

    def get_action_log(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._action_log]

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        record = next(
            (r for r in self._action_log if r.action_id == action_id), None
        )
        return record.to_dict() if record else None
