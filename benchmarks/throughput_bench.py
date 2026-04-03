"""
Throughput Benchmark — Concurrent query processing capacity.

Run: python benchmarks/throughput_bench.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

from aura.core.bus import AsyncAgentBus
from aura.core.memory import MemoryStore
from aura.agents.data_architect import DataArchitectAgent
from aura.agents.planner import PlannerAgent
from aura.pipelines.mpp_simulator import MPPSimulator


async def bench_concurrent_queries(
    concurrency: int = 10, queries_per_worker: int = 5
) -> dict[str, float]:
    """Benchmark concurrent query processing."""
    bus = AsyncAgentBus(distributed=False)
    await bus.start()
    memory = MemoryStore()
    await memory.connect()

    mpp = MPPSimulator(db_path=":memory:", synthetic_rows=5000, partitions=4)
    await mpp.initialize()

    agent = DataArchitectAgent(bus=bus, memory=memory, mpp_simulator=mpp)
    await agent.on_start()

    queries = [
        "Find average latency in APAC",
        "Count incidents by severity",
        "Show node status distribution",
        "Analyze retail throughput trends",
        "Compare cache miss rates by region",
    ]

    latencies: list[float] = []
    errors = 0

    async def worker(worker_id: int) -> None:
        nonlocal errors
        for i in range(queries_per_worker):
            query = queries[i % len(queries)]
            start = time.time()
            try:
                trace = await agent.run(query)
                lat = (time.time() - start) * 1000
                latencies.append(lat)
                if trace.status != "completed":
                    errors += 1
            except Exception:
                errors += 1

    total_start = time.time()
    workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    await asyncio.gather(*workers)
    total_time = time.time() - total_start

    total_queries = concurrency * queries_per_worker
    sorted_lat = sorted(latencies)

    await agent.on_stop()
    await bus.shutdown()
    await mpp.close()

    return {
        "concurrency": concurrency,
        "total_queries": total_queries,
        "completed": len(latencies),
        "errors": errors,
        "total_time_s": round(total_time, 3),
        "throughput_qps": round(len(latencies) / total_time, 2),
        "p50_ms": round(sorted_lat[len(sorted_lat) // 2], 2) if sorted_lat else 0,
        "p95_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 2) if sorted_lat else 0,
        "p99_ms": round(sorted_lat[int(len(sorted_lat) * 0.99)], 2) if sorted_lat else 0,
        "mean_ms": round(statistics.mean(sorted_lat), 2) if sorted_lat else 0,
    }


async def bench_mpp_sql_throughput(num_queries: int = 500) -> dict[str, float]:
    """Benchmark raw SQL execution throughput."""
    mpp = MPPSimulator(db_path=":memory:", synthetic_rows=10000, partitions=8)
    await mpp.initialize()

    queries = [
        "SELECT region, AVG(latency_ms) FROM retail_metrics GROUP BY region",
        "SELECT COUNT(*) FROM infrastructure_nodes WHERE status = 'active'",
        "SELECT severity, COUNT(*) FROM service_incidents GROUP BY severity",
        "SELECT region, MAX(cpu_usage) FROM infrastructure_nodes GROUP BY region",
        "SELECT service_name, AVG(throughput_qps) FROM retail_metrics GROUP BY service_name",
    ]

    latencies = []
    for i in range(num_queries):
        sql = queries[i % len(queries)]
        start = time.time()
        await mpp.execute(sql)
        latencies.append((time.time() - start) * 1000)

    await mpp.close()

    sorted_lat = sorted(latencies)
    return {
        "queries": num_queries,
        "p50_ms": round(sorted_lat[len(sorted_lat) // 2], 4),
        "p95_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 4),
        "p99_ms": round(sorted_lat[int(len(sorted_lat) * 0.99)], 4),
        "mean_ms": round(statistics.mean(sorted_lat), 4),
        "throughput_qps": round(num_queries / (sum(sorted_lat) / 1000), 2),
    }


async def main() -> None:
    print("=" * 60)
    print("  AURA — Throughput Benchmark Suite")
    print("=" * 60)

    print("\n[1] Concurrent Agent Queries (10 workers × 5 queries)")
    print("-" * 40)
    results = await bench_concurrent_queries(10, 5)
    for k, v in results.items():
        print(f"  {k:>25s}: {v}")

    print("\n[2] Raw MPP SQL Throughput (500 queries)")
    print("-" * 40)
    results = await bench_mpp_sql_throughput(500)
    for k, v in results.items():
        print(f"  {k:>25s}: {v}")

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
