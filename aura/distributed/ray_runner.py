"""
Ray Distributed Execution — Actor wrappers for distributed agent deployment.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RayRunner:
    """
    Manages distributed execution of agents via Ray.

    Features:
      - Actor wrappers for each agent type
      - Auto-scaling based on queue depth
      - Integration with Redis Streams transport
    """

    def __init__(
        self,
        enabled: bool = False,
        address: str = "auto",
        num_cpus: int = 4,
    ) -> None:
        self.enabled = enabled
        self.address = address
        self.num_cpus = num_cpus
        self._ray: Any = None
        self._actors: dict[str, Any] = {}
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize Ray runtime."""
        if not self.enabled:
            logger.info("Ray distributed mode disabled")
            return False

        try:
            import ray

            if not ray.is_initialized():
                ray.init(
                    address=self.address,
                    num_cpus=self.num_cpus,
                    dashboard_port=8265,
                    logging_level=logging.WARNING,
                )
            self._ray = ray
            self._initialized = True
            logger.info(
                "Ray initialized: %s (CPUs: %d)",
                ray.cluster_resources(),
                self.num_cpus,
            )
            return True

        except ImportError:
            logger.warning("Ray not installed, falling back to local execution")
            return False
        except Exception as e:
            logger.error("Ray init failed: %s", e)
            return False

    def create_agent_actor(
        self,
        agent_class: type,
        agent_name: str,
        **kwargs: Any,
    ) -> Any:
        """
        Create a Ray actor for an agent.
        Returns the actor handle or None if Ray is not available.
        """
        if not self._initialized or self._ray is None:
            return None

        try:
            RemoteAgent = self._ray.remote(agent_class)
            actor = RemoteAgent.remote(**kwargs)
            self._actors[agent_name] = actor
            logger.info("Created Ray actor for agent: %s", agent_name)
            return actor
        except Exception as e:
            logger.error("Failed to create Ray actor %s: %s", agent_name, e)
            return None

    async def remote_run(
        self, agent_name: str, query: str, context: dict[str, Any] | None = None
    ) -> Any:
        """Execute a query on a remote agent actor."""
        actor = self._actors.get(agent_name)
        if actor is None:
            raise ValueError(f"No actor found for agent: {agent_name}")

        try:
            result_ref = actor.run.remote(query, context)
            result = await asyncio.wrap_future(result_ref.future())
            return result
        except Exception as e:
            logger.error("Remote execution failed for %s: %s", agent_name, e)
            raise

    def get_cluster_info(self) -> dict[str, Any]:
        """Get Ray cluster information."""
        if not self._initialized or self._ray is None:
            return {"enabled": False, "status": "not_initialized"}

        try:
            return {
                "enabled": True,
                "status": "running",
                "resources": dict(self._ray.cluster_resources()),
                "available": dict(self._ray.available_resources()),
                "actors": list(self._actors.keys()),
                "nodes": len(self._ray.nodes()),
            }
        except Exception as e:
            return {"enabled": True, "status": "error", "error": str(e)}

    async def scale_actor(
        self, agent_name: str, replicas: int
    ) -> dict[str, Any]:
        """Scale an agent actor (placeholder for auto-scaling logic)."""
        if not self._initialized:
            return {"error": "Ray not initialized"}

        return {
            "agent": agent_name,
            "target_replicas": replicas,
            "status": "scaling_requested",
        }

    async def shutdown(self) -> None:
        """Shut down Ray."""
        if self._initialized and self._ray is not None:
            self._ray.shutdown()
            self._initialized = False
            logger.info("Ray shut down")
