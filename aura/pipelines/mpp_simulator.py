"""
MPP Simulator — In-process Snowflake/MPP simulation using SQLite.

Features:
  - Synthetic data generation (retail metrics, infra nodes, incidents)
  - Partitioned tables with simulated distributed query plans
  - Configurable latency injection for bottleneck simulation
  - EXPLAIN output for query plan analysis
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# Synthetic data templates
REGIONS = ["APAC", "NA", "EU", "LATAM", "MEA"]
SERVICES = ["retail", "payments", "auth", "search", "recommendations", "inventory"]
STATUSES = ["active", "inactive", "degraded", "maintenance"]
SEVERITIES = ["critical", "high", "medium", "low", "info"]


class MPPSimulator:
    """
    Simulates a Snowflake/MPP data warehouse using SQLite.

    Provides realistic enterprise data for the Data Architect agent
    to query, analyze, and optimize.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        synthetic_rows: int = 100_000,
        latency_injection_ms: int = 0,
        partitions: int = 8,
    ) -> None:
        self.db_path = db_path
        self.synthetic_rows = synthetic_rows
        self.latency_injection_ms = latency_injection_ms
        self.partitions = partitions
        self._db: aiosqlite.Connection | None = None
        self._query_log: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        """Create tables and populate with synthetic data."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")

        await self._create_tables()
        await self._populate_data()
        logger.info(
            "MPP Simulator initialized: %d rows across %d partitions",
            self.synthetic_rows,
            self.partitions,
        )

    async def _create_tables(self) -> None:
        """Create the simulated MPP schema."""
        assert self._db is not None

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS retail_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT NOT NULL,
                service_name TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                throughput_qps REAL NOT NULL,
                error_rate REAL NOT NULL,
                cache_miss_rate REAL NOT NULL,
                timestamp TEXT NOT NULL,
                partition_id INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS infrastructure_nodes (
                node_id TEXT PRIMARY KEY,
                region TEXT NOT NULL,
                status TEXT NOT NULL,
                cpu_usage REAL NOT NULL,
                memory_usage REAL NOT NULL,
                disk_usage REAL NOT NULL,
                last_patch_date TEXT NOT NULL,
                uptime_hours INTEGER NOT NULL,
                partition_id INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS service_incidents (
                incident_id TEXT PRIMARY KEY,
                service_name TEXT NOT NULL,
                region TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                resolution_time_hours REAL,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                partition_id INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_retail_region
                ON retail_metrics(region);
            CREATE INDEX IF NOT EXISTS idx_retail_service
                ON retail_metrics(service_name);
            CREATE INDEX IF NOT EXISTS idx_nodes_region
                ON infrastructure_nodes(region);
            CREATE INDEX IF NOT EXISTS idx_incidents_region
                ON service_incidents(region);
        """)

    async def _populate_data(self) -> None:
        """Generate and insert synthetic enterprise data."""
        assert self._db is not None

        # Retail metrics
        metrics = []
        for i in range(min(self.synthetic_rows, 50000)):
            region = random.choice(REGIONS)
            service = random.choice(SERVICES)
            # APAC has higher latency (simulated bottleneck)
            base_latency = 180 if region == "APAC" else random.uniform(80, 130)
            metrics.append((
                region,
                service,
                base_latency + random.gauss(0, 20),
                random.uniform(5000, 50000),
                random.uniform(0.001, 0.01),
                random.uniform(0.05, 0.30) if region == "APAC" else random.uniform(0.05, 0.12),
                f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:00:00Z",
                i % self.partitions,
            ))

        await self._db.executemany(
            "INSERT INTO retail_metrics "
            "(region, service_name, latency_ms, throughput_qps, error_rate, cache_miss_rate, timestamp, partition_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            metrics,
        )

        # Infrastructure nodes
        nodes = []
        for i in range(min(self.synthetic_rows // 50, 2000)):
            region = random.choice(REGIONS)
            status = random.choices(STATUSES, weights=[85, 3, 8, 4])[0]
            nodes.append((
                f"node-{region.lower()}-{i:04d}",
                region,
                status,
                random.uniform(10, 95),
                random.uniform(20, 90),
                random.uniform(10, 80),
                f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                random.randint(1, 8760),
                i % self.partitions,
            ))

        await self._db.executemany(
            "INSERT INTO infrastructure_nodes "
            "(node_id, region, status, cpu_usage, memory_usage, disk_usage, last_patch_date, uptime_hours, partition_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            nodes,
        )

        # Service incidents
        incidents = []
        for i in range(min(self.synthetic_rows // 100, 1000)):
            region = random.choice(REGIONS)
            incidents.append((
                f"INC-{i:06d}",
                random.choice(SERVICES),
                region,
                random.choice(SEVERITIES),
                f"Performance degradation in {random.choice(SERVICES)} service",
                random.uniform(0.5, 48.0) if random.random() > 0.1 else None,
                f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:00:00Z",
                None if random.random() > 0.85 else f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:00:00Z",
                i % self.partitions,
            ))

        await self._db.executemany(
            "INSERT INTO service_incidents "
            "(incident_id, service_name, region, severity, title, resolution_time_hours, created_at, resolved_at, partition_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            incidents,
        )

        await self._db.commit()

    async def execute(self, sql: str) -> dict[str, Any]:
        """Execute a SQL query with optional latency injection."""
        assert self._db is not None

        # Inject latency if configured
        if self.latency_injection_ms > 0:
            await asyncio.sleep(self.latency_injection_ms / 1000.0)

        start = time.time()
        try:
            cursor = await self._db.execute(sql)
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            elapsed_ms = (time.time() - start) * 1000

            result = {
                "columns": columns,
                "rows": [dict(zip(columns, row)) for row in rows],
                "row_count": len(rows),
                "elapsed_ms": elapsed_ms,
                "sql": sql,
            }

            self._query_log.append({
                "sql": sql,
                "elapsed_ms": elapsed_ms,
                "row_count": len(rows),
                "timestamp": time.time(),
            })

            return result

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error("SQL execution error: %s — %s", sql, e)
            raise RuntimeError(f"SQL error: {e}") from e

    async def explain(self, sql: str) -> dict[str, Any]:
        """Get the query plan for a SQL statement."""
        assert self._db is not None

        try:
            cursor = await self._db.execute(f"EXPLAIN QUERY PLAN {sql}")
            plan_rows = await cursor.fetchall()
            plan_text = "\n".join(
                f"{'  ' * row[1]}{row[3]}" for row in plan_rows
            )
            return {
                "sql": sql,
                "plan": plan_text,
                "estimated_cost": len(plan_rows) * 100.0,
                "estimated_rows": self.synthetic_rows // 10,
                "uses_index": "USING INDEX" in plan_text.upper() or "SEARCH" in plan_text.upper(),
            }
        except Exception as e:
            return {"sql": sql, "plan": f"EXPLAIN error: {e}", "estimated_cost": -1}

    async def search_schema(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search table schemas by keyword relevance."""
        assert self._db is not None

        tables = []
        cursor = await self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = [row[0] for row in await cursor.fetchall()]

        for table_name in table_names:
            cursor = await self._db.execute(f"PRAGMA table_info({table_name})")
            columns = await cursor.fetchall()
            col_names = [col[1] for col in columns]
            col_types = [col[2] for col in columns]

            # Simple relevance scoring
            query_lower = query.lower()
            score = sum(
                1 for c in col_names if any(
                    word in c.lower() for word in query_lower.split()
                )
            )
            score += 2 if any(
                word in table_name.lower() for word in query_lower.split()
            ) else 0

            tables.append({
                "table": table_name,
                "columns": col_names,
                "types": col_types,
                "relevance_score": score,
            })

        tables.sort(key=lambda t: t["relevance_score"], reverse=True)
        return tables[:top_k]

    async def describe_table(self, table_name: str) -> dict[str, Any]:
        """Get full schema for a table."""
        assert self._db is not None

        cursor = await self._db.execute(f"PRAGMA table_info({table_name})")
        columns = await cursor.fetchall()

        cursor = await self._db.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = (await cursor.fetchone())[0]

        return {
            "table": table_name,
            "columns": [
                {"name": col[1], "type": col[2], "nullable": not col[3], "pk": bool(col[5])}
                for col in columns
            ],
            "row_count": count,
            "partitions": self.partitions,
        }

    async def detect_bottlenecks(self) -> list[dict[str, Any]]:
        """Scan for performance bottlenecks in the simulated data."""
        assert self._db is not None

        bottlenecks = []

        # Check for high latency regions
        cursor = await self._db.execute("""
            SELECT region, AVG(latency_ms) as avg_lat, MAX(latency_ms) as max_lat,
                   AVG(cache_miss_rate) as avg_cache_miss
            FROM retail_metrics
            GROUP BY region
            HAVING AVG(latency_ms) > 150
        """)
        for row in await cursor.fetchall():
            bottlenecks.append({
                "type": "high_latency",
                "region": row[0],
                "avg_latency_ms": round(row[1], 2),
                "max_latency_ms": round(row[2], 2),
                "cache_miss_rate": round(row[3], 4),
                "severity": "critical" if row[1] > 200 else "high",
            })

        # Check for degraded nodes
        cursor = await self._db.execute("""
            SELECT region, COUNT(*) as degraded_count
            FROM infrastructure_nodes
            WHERE status = 'degraded'
            GROUP BY region
            HAVING COUNT(*) > 2
        """)
        for row in await cursor.fetchall():
            bottlenecks.append({
                "type": "degraded_nodes",
                "region": row[0],
                "count": row[1],
                "severity": "high",
            })

        return bottlenecks

    def get_query_log(self) -> list[dict[str, Any]]:
        """Return the query execution log."""
        return list(self._query_log)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
