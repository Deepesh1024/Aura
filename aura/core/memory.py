"""
Long-term memory store — Redis-backed with three memory subsystems:
  1. Sliding window (conversation context)
  2. Semantic memory (vector similarity over past interactions)
  3. Episodic memory (structured event log)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryEntry:
    """A single memory record."""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_name: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    memory_type: str = "episodic"  # episodic | semantic | conversation

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "agent_name": self.agent_name,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type,
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(**data)


class MemoryStore:
    """
    Unified memory store supporting conversation context, episodic logs,
    and semantic search.

    Uses Redis when available, falls back to in-memory dict for local dev.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        prefix: str = "aura:mem:",
        window_size: int = 20,
    ) -> None:
        self._redis_url = redis_url
        self._prefix = prefix
        self._window_size = window_size
        self._redis: Any | None = None

        # In-memory fallback stores
        self._conversation: dict[str, list[MemoryEntry]] = {}
        self._episodic: list[MemoryEntry] = []
        self._semantic: list[MemoryEntry] = []

    async def connect(self) -> None:
        """Connect to Redis if URL provided."""
        if self._redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self._redis_url, decode_responses=True
                )
                await self._redis.ping()
                logger.info("Memory store connected to Redis")
            except Exception as e:
                logger.warning("Redis unavailable, using in-memory store: %s", e)
                self._redis = None

    # ---- Conversation Memory (Sliding Window) ----

    async def add_conversation(self, agent_name: str, content: str, **meta: Any) -> str:
        """Add a message to the sliding conversation window."""
        entry = MemoryEntry(
            agent_name=agent_name,
            content=content,
            metadata=meta,
            memory_type="conversation",
        )

        if self._redis:
            key = f"{self._prefix}conv:{agent_name}"
            await self._redis.lpush(key, entry.serialize())
            await self._redis.ltrim(key, 0, self._window_size - 1)
        else:
            ctx = self._conversation.setdefault(agent_name, [])
            ctx.insert(0, entry)
            self._conversation[agent_name] = ctx[: self._window_size]

        return entry.entry_id

    async def get_conversation(
        self, agent_name: str, limit: int | None = None
    ) -> list[MemoryEntry]:
        """Retrieve recent conversation entries."""
        limit = limit or self._window_size

        if self._redis:
            key = f"{self._prefix}conv:{agent_name}"
            raw = await self._redis.lrange(key, 0, limit - 1)
            return [MemoryEntry.from_dict(json.loads(r)) for r in raw]

        entries = self._conversation.get(agent_name, [])
        return entries[:limit]

    # ---- Episodic Memory (Structured Event Log) ----

    async def add_episode(
        self, agent_name: str, event_type: str, content: str, **meta: Any
    ) -> str:
        """Log a structured event to episodic memory."""
        entry = MemoryEntry(
            agent_name=agent_name,
            content=content,
            metadata={"event_type": event_type, **meta},
            memory_type="episodic",
        )

        if self._redis:
            key = f"{self._prefix}episodes"
            await self._redis.zadd(key, {entry.serialize(): entry.timestamp})
            # Keep last 10K episodes
            await self._redis.zremrangebyrank(key, 0, -10001)
        else:
            self._episodic.append(entry)
            if len(self._episodic) > 10000:
                self._episodic = self._episodic[-10000:]

        return entry.entry_id

    async def get_episodes(
        self,
        agent_name: str | None = None,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """Retrieve episodic memories with optional filtering."""
        if self._redis:
            key = f"{self._prefix}episodes"
            raw = await self._redis.zrevrange(key, 0, limit * 2)
            entries = [MemoryEntry.from_dict(json.loads(r)) for r in raw]
        else:
            entries = list(reversed(self._episodic))

        # Filter
        if agent_name:
            entries = [e for e in entries if e.agent_name == agent_name]
        if event_type:
            entries = [
                e for e in entries if e.metadata.get("event_type") == event_type
            ]

        return entries[:limit]

    # ---- Semantic Memory (Vector Similarity) ----

    async def add_semantic(
        self, agent_name: str, content: str, embedding: list[float] | None = None, **meta: Any
    ) -> str:
        """
        Store a semantic memory entry.
        In production, embeddings are indexed in Milvus.
        Here we store the raw entry for retrieval.
        """
        entry = MemoryEntry(
            agent_name=agent_name,
            content=content,
            metadata={"embedding_dim": len(embedding) if embedding else 0, **meta},
            memory_type="semantic",
        )

        if self._redis:
            key = f"{self._prefix}semantic:{agent_name}"
            await self._redis.lpush(key, entry.serialize())
            await self._redis.ltrim(key, 0, 999)
        else:
            self._semantic.append(entry)

        return entry.entry_id

    async def search_semantic(
        self,
        query_embedding: list[float] | None = None,
        agent_name: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """
        Search semantic memory by similarity.
        Simplified: returns recent entries. Production impl uses Milvus ANN.
        """
        if self._redis and agent_name:
            key = f"{self._prefix}semantic:{agent_name}"
            raw = await self._redis.lrange(key, 0, top_k - 1)
            return [MemoryEntry.from_dict(json.loads(r)) for r in raw]

        entries = self._semantic
        if agent_name:
            entries = [e for e in entries if e.agent_name == agent_name]
        return entries[-top_k:]

    # ---- Utilities ----

    async def clear(self, agent_name: str | None = None) -> None:
        """Clear all memories, optionally for a specific agent."""
        if agent_name:
            self._conversation.pop(agent_name, None)
            self._episodic = [
                e for e in self._episodic if e.agent_name != agent_name
            ]
            self._semantic = [
                e for e in self._semantic if e.agent_name != agent_name
            ]
        else:
            self._conversation.clear()
            self._episodic.clear()
            self._semantic.clear()

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
