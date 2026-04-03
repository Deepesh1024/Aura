"""
DuckDB OLAP Engine — Real MPP performance over analytical partitioned files.

Replaces the SQLite simulator with a production-ready DuckDB engine.
- Directly queries partitioned Parquet files (simulating a TPC-DS scale dataset).
- Enforces a tight memory limit (e.g. 80% RAM or a strict 1GB for stress testing)
  to ensure the Data Architect learns to optimize for disk I/O and predicate pushdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


class MPPEngine:
    """
    OLAP DuckDB Engine simulating an Apple-tier petabyte data environment.
    Queries Parquet files directly, enforcing predicate pushdown optimization.
    """

    def __init__(self, db_path: str = ":memory:", dataset_path: str = "/tmp/aura_tpcds"):
        self.db_path = db_path
        self.dataset_path = dataset_path
        self.conn: duckdb.DuckDBPyConnection | None = None

    async def initialize(self) -> None:
        """Initialize the DuckDB engine and bootstrap synthetic Parquet partitions."""
        logger.info("Initializing DuckDB MPP Engine at %s", self.db_path)
        
        # We use an thread-safe background execution via asyncio loop
        loop = asyncio.get_running_loop()
        self.conn = await loop.run_in_executor(None, duckdb.connect, self.db_path)

        # 1. Enforce Memory Limit
        # Limit memory to simulate restricted resources, forcing disk I/O on large queries
        # Instead of dynamically calculating 80% system RAM which can cause out-of-core
        # issues on CI/CD runners, we set a strict 1GB limit to definitively stress test.
        # Apple's JD emphasizes optimizing query plans — constraints force optimization.
        await loop.run_in_executor(None, self.conn.execute, "PRAGMA memory_limit='1GB'")
        await loop.run_in_executor(None, self.conn.execute, "PRAGMA threads=4")
        
        # 2. Bootstrap Partitioned Parquet Data (Simulated TPC-DS)
        await self._bootstrap_parquet_data()
        
        # 3. Create views over the parquet files to expose them to the Data Architect
        # DuckDB can create views from parquet paths using hive partitioning
        customer_path = os.path.join(self.dataset_path, "customer_data", "*", "*.parquet")
        store_sales_path = os.path.join(self.dataset_path, "store_sales", "*", "*.parquet")
        
        # Add a view for customer
        await loop.run_in_executor(
            None, 
            self.conn.execute, 
            f"CREATE OR REPLACE VIEW customers AS SELECT * FROM read_parquet('{customer_path}', hive_partitioning=true)"
        )
        
        # Add a view for store_sales
        await loop.run_in_executor(
            None, 
            self.conn.execute, 
            f"CREATE OR REPLACE VIEW store_sales AS SELECT * FROM read_parquet('{store_sales_path}', hive_partitioning=true)"
        )

    async def _bootstrap_parquet_data(self) -> None:
        """Generate partitioned Parquet files simulating a slice of TPC-DS."""
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            
        loop = asyncio.get_running_loop()

        # Check if partitioned data already exists
        customer_dir = os.path.join(self.dataset_path, "customer_data")
        if not os.path.exists(customer_dir):
            os.makedirs(customer_dir, exist_ok=True)
            logger.info("Bootstrapping TPC-DS Customer Parquet Data...")
            
            # Generate synthetic customer data with sensitive columns for the PII redactor to catch
            gen_customer_sql = f"""
            COPY (
                SELECT 
                    i AS c_customer_sk,
                    'cust_' || i AS c_customer_id,
                    'pii_email_' || i || '@example.com' AS email,
                    '1-800-' || i AS phone,
                    'user_' || i AS username,
                    (i % 5) + 1 AS region_id,
                    CASE WHEN i % 2 = 0 THEN 'APAC' ELSE 'NA' END AS region
                FROM range(1, 10001) t(i)
            ) TO '{customer_dir}' (FORMAT PARQUET, PARTITION_BY (region), OVERWRITE_OR_IGNORE 1);
            """
            await loop.run_in_executor(None, self.conn.execute, gen_customer_sql)

        sales_dir = os.path.join(self.dataset_path, "store_sales")
        if not os.path.exists(sales_dir):
            os.makedirs(sales_dir, exist_ok=True)
            logger.info("Bootstrapping TPC-DS Store Sales Parquet Data...")
            
            # Generate synthetic sales data partitioned by year
            gen_sales_sql = f"""
            COPY (
                SELECT 
                    i AS ss_item_sk,
                    (i % 10000) + 1 AS ss_customer_sk,
                    (random() * 100)::DECIMAL(10,2) AS ss_sales_price,
                    (random() * 10)::INT AS ss_quantity,
                    2020 + (i % 4) AS ss_sold_year
                FROM range(1, 50001) t(i)
            ) TO '{sales_dir}' (FORMAT PARQUET, PARTITION_BY (ss_sold_year), OVERWRITE_OR_IGNORE 1);
            """
            await loop.run_in_executor(None, self.conn.execute, gen_sales_sql)

    async def execute(self, sql: str) -> dict[str, Any]:
        """Execute a query against DuckDB and return results natively."""
        if not self.conn:
            raise RuntimeError("MPPEngine not initialized")
            
        loop = asyncio.get_running_loop()
        start = time.time()
        
        try:
            # DuckDB fetchdf() converts to pandas, fetchall() gives tuples
            cursor = await loop.run_in_executor(None, self.conn.execute, sql)
            # Use fetchdf to get dictionary structure easily via to_dict('records')
            # Since pandas is heavy we fetch tuples and build dict manually
            columns = [desc[0] for desc in cursor.description]
            records = cursor.fetchall()
            
            results = [dict(zip(columns, row)) for row in records]
            elapsed_ms = (time.time() - start) * 1000
            
            return {
                "row_count": len(results),
                "rows": results,
                "columns": [desc[0] for desc in self.conn.description] if self.conn.description else [],
                "elapsed_ms": elapsed_ms,
                "engine": "duckdb"
            }
        except Exception as e:
            logger.error("DuckDB execution error: %s", e)
            raise ValueError(f"MPPEngine Error: {e}")

    async def explain(self, sql: str) -> str:
        """Return the physical execution plan emphasizing predicate pushdown."""
        if not self.conn:
            return "Engine not initialized."
            
        loop = asyncio.get_running_loop()
        try:
            # Using EXPLAIN ANALYZE provides actual execution stats if needed, 
            # but standard EXPLAIN is safer for AST inspection by Data Architect.
            explain_sql = f"EXPLAIN {sql}"
            cursor = await loop.run_in_executor(None, self.conn.execute, explain_sql)
            plan = cursor.fetchall()
            
            # format the query plan
            plan_str = "\n".join(row[1] for row in plan)
            return plan_str
        except Exception as e:
            logger.error("DuckDB explain error: %s", e)
            return str(e)
            
    async def get_schema(self) -> dict[str, Any]:
        """Retrieve the database schema."""
        if not self.conn:
             return {}
             
        loop = asyncio.get_running_loop()
        query = """
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema='main'
        """
        cursor = await loop.run_in_executor(None, self.conn.execute, query)
        records = cursor.fetchall()
        
        schema: dict[str, list[dict[str, str]]] = {}
        for row in records:
            tbl, col, dtype = row
            if tbl not in schema:
                schema[tbl] = []
            schema[tbl].append({"name": col, "type": dtype})
            
        return schema

    async def search_schema(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Mock semantic search over column/table metadata for DuckDB."""
        schema = await self.get_schema()
        results = []
        for table, cols in schema.items():
            results.append({
                "table": table,
                "columns": [c["name"] for c in cols],
                "description": f"Table {table} containing {len(cols)} columns."
            })
        return results[:top_k]

    async def describe_table(self, table_name: str) -> dict[str, Any]:
        """Get full schema for a specific table."""
        schema = await self.get_schema()
        cols = schema.get(table_name, [])
        return {
            "table": table_name,
            "columns": cols,
            "partitions": 8,  # Mocked partition count
            "row_count": 10000,
        }

    async def detect_bottlenecks(self) -> list[dict[str, Any]]:
        """Mock bottleneck detection for compatibility."""
        return [
            {
                "query": "SELECT * FROM store_sales",
                "latency_ms": 1500,
                "type": "high_latency",
                "region": "APAC",
                "avg_latency_ms": 250,
                "cache_miss_rate": 0.20,
                "reason": "Missing partition filter",
                "suggested_action": "Add partition key region_id"
            }
        ]

    def get_query_log(self) -> list[dict[str, Any]]:
        """Mock query log for compatibility."""
        return [{"sql": "SELECT 1", "elapsed_ms": 5, "timestamp": time.time()}]

    async def close(self) -> None:
        """Close the DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
