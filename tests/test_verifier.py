"""Tests for the Neuro-Symbolic Verification Engine."""

from __future__ import annotations

import pytest

from verifier.engine import (
    Constraint,
    ConstraintSet,
    ConstraintType,
    Verdict,
    VerificationEngine,
)
from verifier.grounding import GroundingManager
from verifier.constraints import (
    build_enterprise_constraints,
    build_security_constraints,
    from_yaml,
)


class TestConstraint:
    def test_range_pass(self):
        c = Constraint(
            name="latency_ok",
            constraint_type=ConstraintType.RANGE,
            field="latency_ms",
            max_value=200.0,
        )
        result = c.evaluate({"latency_ms": 150})
        assert result.verdict == Verdict.PASS

    def test_range_fail(self):
        c = Constraint(
            name="latency_high",
            constraint_type=ConstraintType.RANGE,
            field="latency_ms",
            max_value=200.0,
        )
        result = c.evaluate({"latency_ms": 350})
        assert result.verdict == Verdict.FAIL

    def test_enum_pass(self):
        c = Constraint(
            name="valid_region",
            constraint_type=ConstraintType.ENUM,
            field="region",
            allowed_values=["APAC", "NA", "EU"],
        )
        result = c.evaluate({"region": "APAC"})
        assert result.verdict == Verdict.PASS

    def test_enum_fail(self):
        c = Constraint(
            name="valid_region",
            constraint_type=ConstraintType.ENUM,
            field="region",
            allowed_values=["APAC", "NA", "EU"],
        )
        result = c.evaluate({"region": "MARS"})
        assert result.verdict == Verdict.FAIL

    def test_type_check(self):
        c = Constraint(
            name="type_str",
            constraint_type=ConstraintType.TYPE,
            field="name",
            value="str",
        )
        assert c.evaluate({"name": "hello"}).verdict == Verdict.PASS
        assert c.evaluate({"name": 42}).verdict == Verdict.FAIL

    def test_regex_pass(self):
        c = Constraint(
            name="no_ssn",
            constraint_type=ConstraintType.REGEX,
            field="text",
            pattern=r"^(?!.*\d{3}-\d{2}-\d{4}).*$",
        )
        result = c.evaluate({"text": "Hello world"})
        assert result.verdict == Verdict.PASS

    def test_regex_fail(self):
        c = Constraint(
            name="no_ssn",
            constraint_type=ConstraintType.REGEX,
            field="text",
            pattern=r"^(?!.*\d{3}-\d{2}-\d{4}).*$",
        )
        result = c.evaluate({"text": "SSN is 123-45-6789"})
        assert result.verdict == Verdict.FAIL

    def test_missing_field_warns(self):
        c = Constraint(
            name="check",
            constraint_type=ConstraintType.RANGE,
            field="missing_field",
            max_value=100,
        )
        result = c.evaluate({"other_field": 50})
        assert result.verdict == Verdict.WARN

    def test_custom_predicate(self):
        c = Constraint(
            name="custom",
            constraint_type=ConstraintType.CUSTOM,
            custom_fn=lambda data: data.get("x", 0) > 10,
        )
        assert c.evaluate({"x": 20}).verdict == Verdict.PASS
        assert c.evaluate({"x": 5}).verdict == Verdict.FAIL


class TestVerificationEngine:
    @pytest.mark.asyncio
    async def test_evaluate_pass(self):
        engine = VerificationEngine()
        result = await engine.evaluate(
            {"latency_ms": 150, "region": "APAC", "cpu_usage": 45,
             "throughput_qps": 1000},
        )
        # WARN is acceptable when some optional fields are missing
        assert result["verdict"] in ("PASS", "WARN")
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_fail_latency(self):
        engine = VerificationEngine()
        result = await engine.evaluate(
            {"latency_ms": 500, "region": "APAC", "cpu_usage": 45},
        )
        assert result["verdict"] == "FAIL"
        assert result["failed"] >= 1

    @pytest.mark.asyncio
    async def test_evaluate_fail_region(self):
        engine = VerificationEngine()
        result = await engine.evaluate(
            {"latency_ms": 100, "region": "MARS", "cpu_usage": 45},
        )
        assert result["verdict"] == "FAIL"

    @pytest.mark.asyncio
    async def test_inline_constraints(self):
        engine = VerificationEngine()
        result = await engine.evaluate(
            {"latency_ms": 100, "region": "NA", "cpu_usage": 10, "score": 95,
             "throughput_qps": 1000},
            constraints=["score > 50"],
        )
        # WARN is acceptable when some optional fields are missing
        assert result["verdict"] in ("PASS", "WARN")
        assert result["failed"] == 0


class TestConstraintSets:
    def test_enterprise_constraints(self):
        cs = build_enterprise_constraints()
        assert len(cs) > 5

        results = cs.evaluate_all({
            "latency_ms": 150,
            "throughput_qps": 500,
            "error_rate": 0.01,
            "cpu_usage": 60,
            "memory_usage": 70,
            "region": "NA",
            "severity": "high",
            "status": "active",
            "cache_miss_rate": 0.08,
        })
        verdicts = [r.verdict for r in results]
        assert Verdict.FAIL not in verdicts

    def test_security_constraints(self):
        cs = build_security_constraints()
        results = cs.evaluate_all({"_text": "Normal text without PII"})
        assert all(r.verdict == Verdict.PASS for r in results)

    def test_yaml_constraints(self):
        rules = [
            {"name": "test_range", "type": "range", "field": "x", "max": 100},
            {"name": "test_enum", "type": "enum", "field": "color", "allowed_values": ["red", "blue"]},
        ]
        cs = from_yaml(rules)
        assert len(cs) == 2

        results = cs.evaluate_all({"x": 50, "color": "red"})
        assert all(r.verdict == Verdict.PASS for r in results)


class TestGroundingManager:
    @pytest.mark.asyncio
    async def test_verify_clean_claim(self):
        gm = GroundingManager()
        result = await gm.verify("APAC latency averages 185ms", "retail_metrics")
        assert result["verdict"] == "PASS"

    @pytest.mark.asyncio
    async def test_verify_absolute_language(self):
        gm = GroundingManager()
        result = await gm.verify(
            "The system always performs perfectly with 100% uptime",
            "infrastructure",
        )
        assert result["verdict"] == "WARN"

    def test_list_sources(self):
        gm = GroundingManager()
        sources = gm.list_sources()
        assert "infrastructure" in sources
        assert "retail_metrics" in sources

    def test_register_custom_source(self):
        gm = GroundingManager()
        gm.register_source("custom", {"key": "value"})
        assert "custom" in gm.list_sources()
        assert gm.get_source("custom") == {"key": "value"}
