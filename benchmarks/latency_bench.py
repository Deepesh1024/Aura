"""
Latency Benchmark — Measures inter-agent communication and inference latency.

Run: python benchmarks/latency_bench.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

from aura.core.bus import AsyncAgentBus, Envelope, Priority


async def bench_inprocess_latency(num_messages: int = 1000) -> dict[str, float]:
    """Benchmark in-process message bus latency."""
    bus = AsyncAgentBus(distributed=False, high_water_mark=2048)
    await bus.start()

    latencies: list[float] = []

    async def handler(env: Envelope) -> None:
        lat = (time.time() - env.timestamp) * 1000
        latencies.append(lat)

    bus.subscribe("bench_topic", handler)

    # Warmup
    for _ in range(100):
        await bus.publish(
            "bench_topic",
            Envelope(source="bench", target="bench", payload={}),
        )
    await asyncio.sleep(0.5)
    latencies.clear()

    # Benchmark
    start = time.time()
    for _ in range(num_messages):
        await bus.publish(
            "bench_topic",
            Envelope(source="bench", target="bench", payload={"i": 0}),
        )

    await asyncio.sleep(1.0)
    total_time = time.time() - start

    await bus.shutdown()

    sorted_lat = sorted(latencies)
    return {
        "messages": len(latencies),
        "total_time_s": round(total_time, 3),
        "throughput_msg_per_s": round(len(latencies) / total_time),
        "p50_ms": round(sorted_lat[len(sorted_lat) // 2], 4) if sorted_lat else 0,
        "p95_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 4) if sorted_lat else 0,
        "p99_ms": round(sorted_lat[int(len(sorted_lat) * 0.99)], 4) if sorted_lat else 0,
        "max_ms": round(max(sorted_lat), 4) if sorted_lat else 0,
        "mean_ms": round(statistics.mean(sorted_lat), 4) if sorted_lat else 0,
        "stdev_ms": round(statistics.stdev(sorted_lat), 4) if len(sorted_lat) > 1 else 0,
    }


async def bench_request_reply(num_requests: int = 100) -> dict[str, float]:
    """Benchmark request-reply pattern latency."""
    bus = AsyncAgentBus(distributed=False)
    await bus.start()

    async def echo(env: Envelope) -> None:
        reply = Envelope(
            source="echo",
            target=env.source,
            payload=env.payload,
            correlation_id=env.correlation_id,
        )
        await bus.publish(f"_reply.{env.correlation_id}", reply)

    bus.subscribe("echo_service", echo)

    latencies = []

    for _ in range(num_requests):
        req = Envelope(
            source="client",
            target="echo_service",
            payload={"ts": time.time()},
        )
        start = time.time()
        await bus.request("echo_service", req, timeout=5.0)
        latencies.append((time.time() - start) * 1000)

    await bus.shutdown()

    sorted_lat = sorted(latencies)
    return {
        "requests": num_requests,
        "p50_ms": round(sorted_lat[len(sorted_lat) // 2], 4),
        "p95_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 4),
        "p99_ms": round(sorted_lat[int(len(sorted_lat) * 0.99)], 4),
        "mean_ms": round(statistics.mean(sorted_lat), 4),
    }


async def main() -> None:
    print("=" * 60)
    print("  AURA — Latency Benchmark Suite")
    print("=" * 60)

    print("\n[1] In-Process Message Bus (1000 messages)")
    print("-" * 40)
    results = await bench_inprocess_latency(1000)
    for k, v in results.items():
        print(f"  {k:>25s}: {v}")

    print("\n[2] Request-Reply Pattern (100 requests)")
    print("-" * 40)
    results = await bench_request_reply(100)
    for k, v in results.items():
        print(f"  {k:>25s}: {v}")

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
