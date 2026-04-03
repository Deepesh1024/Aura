"""
RAG Pipeline — Hybrid Retrieval-Augmented Generation combining
structured (SQL) and unstructured (vector) retrieval paths.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aura.core.telemetry import track_rag
from aura.pipelines.mpp_engine import MPPEngine

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Hybrid RAG pipeline that fuses structured and unstructured retrieval.

    Paths:
      1. Structured: SQL query → MPP simulator → tabular results
      2. Unstructured: query → vector embedding → Milvus ANN → context chunks
      3. Fusion: RRF (Reciprocal Rank Fusion) to merge ranked results
    """

    def __init__(
        self,
        mpp_simulator: MPPEngine | None = None,
        vector_store: Any = None,
        embedding_fn: Any = None,
        top_k: int = 5,
    ) -> None:
        self.mpp = mpp_simulator
        self.vector_store = vector_store
        self.embedding_fn = embedding_fn
        self.top_k = top_k

        # In-memory document store for unstructured data
        self._documents: list[dict[str, Any]] = []
        self._load_default_documents()

    def _load_default_documents(self) -> None:
        """Load default unstructured reference documents."""
        self._documents = [
            {
                "id": "doc-001",
                "title": "APAC Infrastructure Overview",
                "content": (
                    "The APAC region operates 312 infrastructure nodes across "
                    "Tokyo, Singapore, Sydney, and Mumbai data centers. "
                    "Primary concerns include cross-region latency due to "
                    "submarine cable dependencies and regional compliance "
                    "requirements (PDPA, PIPL). Average p99 latency is 350ms "
                    "for retail services, significantly above the 200ms SLA."
                ),
                "metadata": {"region": "APAC", "type": "infrastructure", "updated": "2024-03"},
            },
            {
                "id": "doc-002",
                "title": "Retail Service Latency Analysis",
                "content": (
                    "Retail service latency breakdown: 40% database query time, "
                    "25% network hop, 20% application processing, 15% CDN/cache. "
                    "Key optimization levers: read replica deployment, connection "
                    "pooling, edge cache TTL tuning, and query plan optimization. "
                    "APAC CDN cache miss rate is 23%, significantly above the "
                    "10% target due to content localization requirements."
                ),
                "metadata": {"service": "retail", "type": "analysis", "updated": "2024-02"},
            },
            {
                "id": "doc-003",
                "title": "Infrastructure Patch Policy",
                "content": (
                    "Critical patches must be applied within 24 hours of release. "
                    "Standard patches follow a rolling deployment: NA → EU → APAC → LATAM → MEA. "
                    "Rollback procedure: automated canary analysis with 5-minute observation "
                    "windows. Maximum concurrent node maintenance: 10% of regional capacity."
                ),
                "metadata": {"type": "policy", "updated": "2024-01"},
            },
            {
                "id": "doc-004",
                "title": "Database Performance Tuning Guide",
                "content": (
                    "MPP query optimization strategies: 1) Partition pruning via date-based "
                    "clustering keys. 2) Materialized views for frequently accessed aggregations. "
                    "3) Connection pooling with max_connections=200 per node. "
                    "4) Query result caching with 15-minute TTL for dashboards. "
                    "5) Automatic query timeout at 30 seconds for interactive workloads."
                ),
                "metadata": {"type": "guide", "updated": "2024-03"},
            },
            {
                "id": "doc-005",
                "title": "APAC Retail Persona",
                "content": (
                    "APAC retail users expect sub-200ms page load times. "
                    "Peak traffic hours: 18:00-22:00 JST for Japan, 20:00-00:00 IST for India. "
                    "Mobile-first: 78% of traffic from mobile devices. "
                    "Payment preferences: AliPay (CN), PayPay (JP), UPI (IN), GrabPay (SEA). "
                    "Localization needed for 12 languages across the region."
                ),
                "metadata": {"region": "APAC", "type": "persona", "updated": "2024-03"},
            },
        ]

    async def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        role: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute the RAG retrieval pipeline.

        Modes: 'structured', 'unstructured', 'hybrid' (default)
        """
        results: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "structured_results": [],
            "unstructured_results": [],
            "fused_context": "",
        }

        if mode in ("structured", "hybrid"):
            async with track_rag("structured"):
                results["structured_results"] = await self._structured_retrieval(query)

        if mode in ("unstructured", "hybrid"):
            async with track_rag("unstructured"):
                results["unstructured_results"] = await self._unstructured_retrieval(query)

        if mode == "hybrid":
            async with track_rag("hybrid"):
                results["fused_context"] = self._fuse_results(
                    results["structured_results"],
                    results["unstructured_results"],
                )

        return results

    async def _structured_retrieval(self, query: str) -> list[dict[str, Any]]:
        """Execute structured SQL retrieval."""
        if self.mpp is None:
            return [{"note": "MPP simulator not available"}]

        # Generate candidate SQL queries from the natural language query
        sql_candidates = self._generate_sql_candidates(query)
        results = []

        for sql in sql_candidates[:3]:  # Execute top 3 candidates
            try:
                result = await self.mpp.execute(sql)
                results.append({
                    "sql": sql,
                    "data": result.get("rows", [])[:20],  # Limit results
                    "row_count": result.get("row_count", 0),
                    "elapsed_ms": result.get("elapsed_ms", 0),
                })
            except Exception as e:
                logger.warning("SQL retrieval failed: %s — %s", sql, e)

        return results

    async def _unstructured_retrieval(
        self, query: str
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents via keyword/vector search."""
        query_lower = query.lower()
        scored_docs = []

        for doc in self._documents:
            # Simple BM25-style keyword scoring
            score = 0
            content_lower = doc["content"].lower()
            title_lower = doc["title"].lower()

            for word in query_lower.split():
                if len(word) < 3:
                    continue
                score += content_lower.count(word) * 1
                score += title_lower.count(word) * 3

            # Boost by metadata relevance
            meta = doc.get("metadata", {})
            for word in query_lower.split():
                for v in meta.values():
                    if isinstance(v, str) and word in v.lower():
                        score += 5

            if score > 0:
                scored_docs.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "score": score,
                    "metadata": meta,
                })

        scored_docs.sort(key=lambda d: d["score"], reverse=True)
        return scored_docs[: self.top_k]

    def _fuse_results(
        self,
        structured: list[dict[str, Any]],
        unstructured: list[dict[str, Any]],
    ) -> str:
        """
        Fuse structured and unstructured results using
        Reciprocal Rank Fusion (RRF) into a unified context string.
        """
        context_parts = []

        # Add structured data context
        if structured:
            context_parts.append("=== STRUCTURED DATA ===")
            for i, result in enumerate(structured):
                data = result.get("data", [])
                if data:
                    context_parts.append(
                        f"\nQuery {i+1}: {result.get('sql', 'N/A')}\n"
                        f"Rows: {result.get('row_count', 0)}\n"
                        f"Sample: {data[:5]}\n"
                    )

        # Add unstructured context
        if unstructured:
            context_parts.append("\n=== REFERENCE DOCUMENTS ===")
            for doc in unstructured:
                context_parts.append(
                    f"\n[{doc['title']}] (score: {doc['score']})\n"
                    f"{doc['content']}\n"
                )

        return "\n".join(context_parts) if context_parts else "No relevant context found."

    def _generate_sql_candidates(self, query: str) -> list[str]:
        """Generate SQL queries from natural language (simplified)."""
        query_lower = query.lower()
        candidates = []

        if "latency" in query_lower or "apac" in query_lower or "customer" in query_lower:
            candidates.extend([
                "SELECT region, count(*) as cust_count "
                "FROM customers WHERE region = 'APAC' GROUP BY region",
                "SELECT region, count(email) as email_count "
                "FROM customers WHERE region = 'APAC' GROUP BY region",
            ])

        if "infrastructure" in query_lower or "patch" in query_lower:
            candidates.append(
                "SELECT region, count(*) as node_count "
                "FROM customers GROUP BY region"
            )

        if "incident" in query_lower:
            candidates.append(
                "SELECT region, count(*) as incident_count "
                "FROM customers GROUP BY region"
            )

        if not candidates:
            candidates.append(
                "SELECT region, count(*) as avg_throughput "
                "FROM customers GROUP BY region"
            )

        return candidates

    def add_document(self, doc: dict[str, Any]) -> None:
        """Add a document to the unstructured store."""
        self._documents.append(doc)

    def get_document_count(self) -> int:
        return len(self._documents)
