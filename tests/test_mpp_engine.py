import pytest
import os
from aura.pipelines.mpp_engine import MPPEngine


@pytest.mark.asyncio
class TestMPPEngine:
    async def test_initialize_and_query(self, tmpdir):
        # Use tmpdir for dataset
        dataset_path = str(tmpdir.mkdir("dataset"))
        engine = MPPEngine(db_path=":memory:", dataset_path=dataset_path)
        
        await engine.initialize()
        
        # Test basic query on generated customer view
        res = await engine.execute("SELECT region, count(*) as c FROM customers GROUP BY region")
        assert "rows" in res
        assert len(res["rows"]) == 2  # APAC and NA
        assert res["row_count"] == 2
        
        await engine.close()

    async def test_duckdb_schema_search(self, tmpdir):
        dataset_path = str(tmpdir.mkdir("dataset2"))
        engine = MPPEngine(db_path=":memory:", dataset_path=dataset_path)
        await engine.initialize()
        
        schema = await engine.get_schema()
        assert "customers" in schema
        
        # describe table
        desc = await engine.describe_table("customers")
        assert desc["table"] == "customers"
        assert len(desc["columns"]) > 0
        
        await engine.close()

    async def test_explain_plan(self, tmpdir):
        dataset_path = str(tmpdir.mkdir("dataset3"))
        engine = MPPEngine(db_path=":memory:", dataset_path=dataset_path)
        await engine.initialize()
        
        plan = await engine.explain("SELECT * FROM customers WHERE region = 'APAC'")
        assert "read_parquet" in plan.lower()
        assert "apac" in plan.lower()
        
        await engine.close()
