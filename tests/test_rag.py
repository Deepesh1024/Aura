"""Tests for the RAG pipeline and MPP simulator."""

from __future__ import annotations

import pytest

from aura.pipelines.mpp_engine import MPPEngine
from aura.pipelines.rag import RAGPipeline


@pytest.fixture
async def mpp():
    mpp = MPPEngine(db_path=":memory:", dataset_path="/tmp/aura_test_rag")
    await mpp.initialize()
    yield mpp
    await mpp.close()


@pytest.fixture
async def rag(mpp):
    return RAGPipeline(mpp_simulator=mpp, top_k=3)


class TestMPPSimulator:
    @pytest.mark.asyncio
    async def test_execute_select(self, mpp: MPPEngine):
        result = await mpp.execute(
            "SELECT region, COUNT(*) as cnt FROM customers GROUP BY region"
        )
        assert result["row_count"] > 0
        assert "rows" in result

    @pytest.mark.asyncio
    async def test_execute_with_filter(self, mpp: MPPEngine):
        result = await mpp.execute(
            "SELECT count(*) as avg_lat FROM customers WHERE region = 'APAC'"
        )
        assert result["row_count"] == 1
        assert result["rows"][0]["avg_lat"] > 0

    @pytest.mark.asyncio
    async def test_explain_plan(self, mpp: MPPEngine):
        result = await mpp.explain(
            "SELECT * FROM customers WHERE region = 'APAC'"
        )
        assert "read_parquet" in result.lower()

    @pytest.mark.asyncio
    async def test_search_schema(self, mpp: MPPEngine):
        results = await mpp.search_schema("customer dataset apac")
        assert len(results) > 0
        assert "customers" in [r["table"] for r in results]

    @pytest.mark.asyncio
    async def test_describe_table(self, mpp: MPPEngine):
        result = await mpp.describe_table("customers")
        assert result["table"] == "customers"
        assert result["row_count"] > 0
        assert len(result["columns"]) > 0

    @pytest.mark.asyncio
    async def test_detect_bottlenecks(self, mpp: MPPEngine):
        bottlenecks = await mpp.detect_bottlenecks()
        assert isinstance(bottlenecks, list)
        assert len(bottlenecks) > 0

    @pytest.mark.asyncio
    async def test_query_log(self, mpp: MPPEngine):
        await mpp.execute("SELECT 1")
        log = mpp.get_query_log()
        assert len(log) == 1
        assert "elapsed_ms" in log[0]


class TestRAGPipeline:
    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, rag: RAGPipeline):
        result = await rag.retrieve(
            "Retail latency in APAC", mode="hybrid"
        )
        assert "structured_results" in result
        assert "unstructured_results" in result
        assert "fused_context" in result
        assert len(result["fused_context"]) > 0

    @pytest.mark.asyncio
    async def test_structured_retrieval(self, rag: RAGPipeline):
        result = await rag.retrieve(
            "latency in APAC", mode="structured"
        )
        assert len(result["structured_results"]) > 0

    @pytest.mark.asyncio
    async def test_unstructured_retrieval(self, rag: RAGPipeline):
        result = await rag.retrieve(
            "APAC infrastructure overview", mode="unstructured"
        )
        assert len(result["unstructured_results"]) > 0

    def test_document_count(self, rag: RAGPipeline):
        assert rag.get_document_count() >= 5

    @pytest.mark.asyncio
    async def test_add_document(self, rag: RAGPipeline):
        initial = rag.get_document_count()
        rag.add_document({
            "id": "test-doc",
            "title": "Test Document",
            "content": "Test content about retail performance.",
            "metadata": {"type": "test"},
        })
        assert rag.get_document_count() == initial + 1
