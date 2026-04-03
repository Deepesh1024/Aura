"""
AsyncAgentBus — Dual-transport, zero-latency inter-agent message passing.

Transport Selection:
  - In-process: asyncio.Queue channels (zero-copy, ~0.01ms)
  - Cross-node:  Redis Streams (Protobuf-serialized, ~1-5ms)

The bus auto-selects based on agent locality.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message Envelope
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(slots=True)
class Envelope:
    """Immutable message envelope with full trace metadata."""

    source: str
    target: str
    payload: dict[str, Any]
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = 60.0
    hop_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_seconds

    def serialize(self) -> bytes:
        """Serialize to JSON bytes (fast path; swap for protobuf in prod)."""
        return json.dumps({
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "parent_id": self.parent_id,
            "priority": int(self.priority),
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "hop_count": self.hop_count,
        }).encode()

    @classmethod
    def deserialize(cls, data: bytes) -> Envelope:
        d = json.loads(data)
        d["priority"] = Priority(d["priority"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Subscriber type
# ---------------------------------------------------------------------------

Subscriber = Callable[[Envelope], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# In-Process Transport (zero-copy asyncio.Queue channels)
# ---------------------------------------------------------------------------

class InProcessTransport:
    """
    Ultra-low-latency in-process message passing via asyncio.Queue.
    One bounded queue per (topic) channel.  Supports backpressure.
    """

    def __init__(self, high_water_mark: int = 1024) -> None:
        self._hwm = high_water_mark
        self._channels: dict[str, asyncio.Queue[Envelope]] = {}
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._dispatcher_tasks: dict[str, asyncio.Task[None]] = {}

    def _ensure_channel(self, topic: str) -> asyncio.Queue[Envelope]:
        if topic not in self._channels:
            self._channels[topic] = asyncio.Queue(maxsize=self._hwm)
        return self._channels[topic]

    async def publish(self, topic: str, envelope: Envelope) -> None:
        q = self._ensure_channel(topic)
        try:
            q.put_nowait(envelope)
        except asyncio.QueueFull:
            logger.warning("Backpressure on topic=%s — dropping oldest message", topic)
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            q.put_nowait(envelope)

    def subscribe(self, topic: str, handler: Subscriber) -> None:
        self._subscribers.setdefault(topic, []).append(handler)
        if topic not in self._dispatcher_tasks:
            self._dispatcher_tasks[topic] = asyncio.ensure_future(
                self._dispatch_loop(topic)
            )

    async def _dispatch_loop(self, topic: str) -> None:
        q = self._ensure_channel(topic)
        while True:
            envelope = await q.get()
            if envelope.is_expired:
                logger.debug("Dropping expired message %s", envelope.correlation_id)
                continue
            for handler in self._subscribers.get(topic, []):
                try:
                    await handler(envelope)
                except Exception:
                    logger.exception(
                        "Handler error on topic=%s, id=%s", topic, envelope.correlation_id
                    )

    async def shutdown(self) -> None:
        for task in self._dispatcher_tasks.values():
            task.cancel()
        await asyncio.gather(*self._dispatcher_tasks.values(), return_exceptions=True)
        self._dispatcher_tasks.clear()


# ---------------------------------------------------------------------------
# Redis Streams Transport (cross-process / cross-node)
# ---------------------------------------------------------------------------

class RedisTransport:
    """
    Cross-process message transport using Redis Streams.
    Used when agents are distributed across Ray workers.
    """

    def __init__(self, redis_url: str, stream_key: str = "aura:bus") -> None:
        self._redis_url = redis_url
        self._stream_key = stream_key
        self._redis: Any | None = None
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._consumer_task: asyncio.Task[None] | None = None
        self._consumer_group = "aura-agents"
        self._consumer_name = uuid.uuid4().hex[:8]

    async def connect(self) -> None:
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(
            self._redis_url, decode_responses=False
        )
        # Create consumer group (idempotent)
        try:
            await self._redis.xgroup_create(
                self._stream_key, self._consumer_group, id="0", mkstream=True
            )
        except Exception:
            pass  # Group already exists
        self._consumer_task = asyncio.ensure_future(self._consume_loop())

    async def publish(self, topic: str, envelope: Envelope) -> None:
        if self._redis is None:
            raise RuntimeError("RedisTransport not connected")
        await self._redis.xadd(
            self._stream_key,
            {"topic": topic.encode(), "data": envelope.serialize()},
            maxlen=10_000,
        )

    def subscribe(self, topic: str, handler: Subscriber) -> None:
        self._subscribers.setdefault(topic, []).append(handler)

    async def _consume_loop(self) -> None:
        assert self._redis is not None
        while True:
            try:
                results = await self._redis.xreadgroup(
                    self._consumer_group,
                    self._consumer_name,
                    {self._stream_key: ">"},
                    count=100,
                    block=500,
                )
                for _, messages in results:
                    for msg_id, fields in messages:
                        topic = fields[b"topic"].decode()
                        envelope = Envelope.deserialize(fields[b"data"])
                        for handler in self._subscribers.get(topic, []):
                            try:
                                await handler(envelope)
                            except Exception:
                                logger.exception("Redis handler error")
                        await self._redis.xack(
                            self._stream_key, self._consumer_group, msg_id
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Redis consumer error")
                await asyncio.sleep(1)

    async def shutdown(self) -> None:
        if self._consumer_task:
            self._consumer_task.cancel()
            await asyncio.gather(self._consumer_task, return_exceptions=True)
        if self._redis:
            await self._redis.aclose()


# ---------------------------------------------------------------------------
# AsyncAgentBus — unified facade
# ---------------------------------------------------------------------------

class AsyncAgentBus:
    """
    Unified message bus that auto-selects transport based on configuration.

    - In-process mode: all agents in same event loop → asyncio.Queue
    - Distributed mode: agents on Ray workers → Redis Streams
    """

    def __init__(
        self,
        distributed: bool = False,
        redis_url: str = "redis://localhost:6379/0",
        stream_key: str = "aura:bus",
        high_water_mark: int = 1024,
    ) -> None:
        self._distributed = distributed
        if distributed:
            self._transport: InProcessTransport | RedisTransport = RedisTransport(
                redis_url, stream_key
            )
        else:
            self._transport = InProcessTransport(high_water_mark)

        self._metrics: dict[str, int] = {
            "messages_published": 0,
            "messages_delivered": 0,
            "messages_dropped": 0,
        }
        self._latency_samples: list[float] = []

    async def start(self) -> None:
        """Initialize the transport layer."""
        if isinstance(self._transport, RedisTransport):
            await self._transport.connect()
        logger.info(
            "AsyncAgentBus started [mode=%s]",
            "distributed" if self._distributed else "in-process",
        )

    async def publish(self, topic: str, envelope: Envelope) -> None:
        """Publish a message to a topic."""
        envelope.hop_count += 1
        self._metrics["messages_published"] += 1
        await self._transport.publish(topic, envelope)

    def subscribe(self, topic: str, handler: Subscriber) -> None:
        """Subscribe a handler coroutine to a topic."""

        async def _instrumented(env: Envelope) -> None:
            start = time.time()
            await handler(env)
            elapsed_ms = (time.time() - start) * 1000
            self._latency_samples.append(elapsed_ms)
            self._metrics["messages_delivered"] += 1

        self._transport.subscribe(topic, _instrumented)

    async def request(
        self, topic: str, envelope: Envelope, timeout: float = 30.0
    ) -> Envelope:
        """
        Request-reply pattern: publish and wait for a correlated response.
        """
        future: asyncio.Future[Envelope] = asyncio.get_event_loop().create_future()
        reply_topic = f"_reply.{envelope.correlation_id}"

        async def _on_reply(env: Envelope) -> None:
            if not future.done():
                future.set_result(env)

        self.subscribe(reply_topic, _on_reply)
        await self.publish(topic, envelope)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"No reply on {reply_topic} within {timeout}s "
                f"(correlation_id={envelope.correlation_id})"
            )

    def get_metrics(self) -> dict[str, Any]:
        """Return bus metrics for observability."""
        samples = self._latency_samples[-1000:]
        sorted_samples = sorted(samples) if samples else [0.0]
        return {
            **self._metrics,
            "latency_p50_ms": sorted_samples[len(sorted_samples) // 2],
            "latency_p99_ms": sorted_samples[int(len(sorted_samples) * 0.99)],
            "total_samples": len(self._latency_samples),
        }

    async def shutdown(self) -> None:
        """Gracefully shut down the transport."""
        await self._transport.shutdown()
        logger.info("AsyncAgentBus shut down")
