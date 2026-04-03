"""Tests for AsyncAgentBus — message passing latency and correctness."""

from __future__ import annotations

import asyncio
import time

import pytest

from aura.core.bus import AsyncAgentBus, Envelope, Priority


@pytest.fixture
async def bus():
    b = AsyncAgentBus(distributed=False, high_water_mark=128)
    await b.start()
    yield b
    await b.shutdown()


@pytest.mark.asyncio
async def test_publish_subscribe(bus: AsyncAgentBus):
    """Test basic pub/sub message delivery."""
    received: list[Envelope] = []

    async def handler(env: Envelope):
        received.append(env)

    bus.subscribe("test_topic", handler)

    env = Envelope(
        source="agent_a",
        target="agent_b",
        payload={"data": "hello"},
    )
    await bus.publish("test_topic", env)
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].payload["data"] == "hello"
    assert received[0].source == "agent_a"


@pytest.mark.asyncio
async def test_request_reply(bus: AsyncAgentBus):
    """Test request-reply pattern."""
    async def echo_handler(env: Envelope):
        reply = Envelope(
            source="responder",
            target=env.source,
            payload={"echo": env.payload["message"]},
            correlation_id=env.correlation_id,
        )
        reply_topic = f"_reply.{env.correlation_id}"
        await bus.publish(reply_topic, reply)

    bus.subscribe("echo_service", echo_handler)

    request = Envelope(
        source="requester",
        target="echo_service",
        payload={"message": "ping"},
    )
    reply = await bus.request("echo_service", request, timeout=5.0)

    assert reply.payload["echo"] == "ping"


@pytest.mark.asyncio
async def test_message_latency(bus: AsyncAgentBus):
    """Test that in-process message delivery is sub-1ms."""
    latencies: list[float] = []

    async def handler(env: Envelope):
        latency = (time.time() - env.timestamp) * 1000
        latencies.append(latency)

    bus.subscribe("latency_test", handler)

    for _ in range(100):
        env = Envelope(
            source="sender",
            target="receiver",
            payload={"ts": time.time()},
        )
        await bus.publish("latency_test", env)

    await asyncio.sleep(0.2)

    assert len(latencies) == 100
    avg = sum(latencies) / len(latencies)
    p99 = sorted(latencies)[98]

    assert avg < 10.0, f"Average latency {avg:.2f}ms exceeds threshold"
    assert p99 < 50.0, f"P99 latency {p99:.2f}ms exceeds threshold"


@pytest.mark.asyncio
async def test_expired_messages_dropped(bus: AsyncAgentBus):
    """Test that expired messages are not delivered."""
    received: list[Envelope] = []

    async def handler(env: Envelope):
        received.append(env)

    bus.subscribe("expire_test", handler)

    env = Envelope(
        source="a",
        target="b",
        payload={"data": 1},
        ttl_seconds=0.0,  # Already expired
        timestamp=time.time() - 1,
    )
    await bus.publish("expire_test", env)
    await asyncio.sleep(0.1)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_priority_levels(bus: AsyncAgentBus):
    """Test message priority assignment."""
    env = Envelope(
        source="a",
        target="b",
        payload={},
        priority=Priority.CRITICAL,
    )
    assert env.priority == Priority.CRITICAL
    assert int(env.priority) == 3


@pytest.mark.asyncio
async def test_bus_metrics(bus: AsyncAgentBus):
    """Test bus metrics collection."""
    async def noop(env: Envelope):
        pass

    bus.subscribe("metrics_test", noop)

    for _ in range(10):
        await bus.publish(
            "metrics_test",
            Envelope(source="a", target="b", payload={}),
        )

    await asyncio.sleep(0.1)
    metrics = bus.get_metrics()

    assert metrics["messages_published"] == 10
    assert metrics["messages_delivered"] == 10


@pytest.mark.asyncio
async def test_envelope_serialization():
    """Test Envelope serialization/deserialization."""
    original = Envelope(
        source="agent_x",
        target="agent_y",
        payload={"key": "value", "num": 42},
        priority=Priority.HIGH,
    )
    data = original.serialize()
    restored = Envelope.deserialize(data)

    assert restored.source == original.source
    assert restored.target == original.target
    assert restored.payload == original.payload
    assert restored.priority == original.priority
    assert restored.correlation_id == original.correlation_id
