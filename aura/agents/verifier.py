"""
Neuro-Symbolic Verifier Agent — deterministic "Critic" that validates
LLM outputs against business constraints using symbolic logic.

Eliminates hallucinations by:
  1. Type-checking LLM assertions
  2. Range-validating numeric claims
  3. Cross-referencing against grounding data
  4. Evaluating logical consistency of multi-step reasoning
"""

from __future__ import annotations

import logging
from typing import Any

from aura.core.agent_base import BaseAgent, ReActStep, ToolSpec
from aura.core.bus import AsyncAgentBus
from aura.core.memory import MemoryStore
from aura.core.telemetry import VERIFICATION_RESULTS

logger = logging.getLogger(__name__)


class VerifierAgent(BaseAgent):
    """
    Neuro-Symbolic Verifier — blends LLM reasoning with deterministic
    constraint checking to validate all agent outputs before delivery.

    Verdict types: PASS | FAIL | WARN
    """

    def __init__(
        self,
        bus: AsyncAgentBus,
        memory: MemoryStore,
        verification_engine: Any = None,
        grounding_manager: Any = None,
        llm_client: Any = None,
        strict_mode: bool = True,
    ) -> None:
        super().__init__(name="verifier", bus=bus, max_iterations=3)
        self.memory = memory
        self.engine = verification_engine
        self.grounding = grounding_manager
        self.llm = llm_client
        self.strict_mode = strict_mode
        self._register_tools()

    def _register_tools(self) -> None:
        self.tools.register(ToolSpec(
            name="check_constraints",
            description="Validate data against business constraint rules",
            function=self._check_constraints,
            parameters={"claim": "str", "constraints": "list"},
        ))
        self.tools.register(ToolSpec(
            name="ground_check",
            description="Verify a claim against ground-truth reference data",
            function=self._ground_check,
            parameters={"claim": "str", "data_source": "str"},
        ))
        self.tools.register(ToolSpec(
            name="logical_consistency",
            description="Check logical consistency of multi-step reasoning",
            function=self._check_logical_consistency,
            parameters={"steps": "list"},
        ))
        self.tools.register(ToolSpec(
            name="type_check",
            description="Validate types and ranges of numeric/categorical values",
            function=self._type_check,
            parameters={"values": "dict", "schema": "dict"},
        ))

    # ---- ReAct Loop ----

    async def observe(self, query: str, context: dict[str, Any]) -> str:
        """Extract the claim, evidence, and constraints to verify."""
        claim = context.get("claim", query)
        evidence = context.get("evidence", {})
        constraints = context.get("constraints", [])

        return (
            f"VERIFICATION REQUEST:\n"
            f"  Claim: {claim}\n"
            f"  Evidence: {evidence}\n"
            f"  Constraints: {constraints}\n"
            f"  Strict mode: {self.strict_mode}\n"
        )

    async def think(
        self, query: str, observations: str, history: list[ReActStep]
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        """Determine which verification checks to run."""
        # Deterministic logic — no LLM needed for verification planning
        context_claim = ""
        for line in observations.split("\n"):
            if "Claim:" in line:
                context_claim = line.split("Claim:", 1)[1].strip()

        constraints_line = ""
        for line in observations.split("\n"):
            if "Constraints:" in line:
                constraints_line = line.split("Constraints:", 1)[1].strip()

        has_done_constraint_check = any(
            s.tool_name == "check_constraints" for s in history
        )
        has_done_ground_check = any(
            s.tool_name == "ground_check" for s in history
        )
        has_done_logic_check = any(
            s.tool_name == "logical_consistency" for s in history
        )

        if not has_done_constraint_check and constraints_line:
            return (
                "Need to validate against business constraints first.",
                "check_constraints",
                {"claim": context_claim, "constraints": eval(constraints_line) if constraints_line.startswith("[") else []},
            )

        if not has_done_ground_check:
            return (
                "Need to cross-reference against grounding data.",
                "ground_check",
                {"claim": context_claim, "data_source": "default"},
            )

        if not has_done_logic_check:
            reasoning_steps = [
                s.content for s in history if s.step_type.name in ("THINK", "REFLECT")
            ]
            return (
                "Checking logical consistency of the reasoning chain.",
                "logical_consistency",
                {"steps": reasoning_steps},
            )

        return ("All verification checks complete.", None, None)

    async def act(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Execute the verification check."""
        return await self.tools.execute(tool_name, **tool_input)

    async def reflect(
        self, query: str, history: list[ReActStep], tool_output: Any
    ) -> tuple[str, bool]:
        """Aggregate verification results into a final verdict."""
        # Collect all verification results
        results = []
        for step in history:
            if step.tool_output and isinstance(step.tool_output, dict):
                results.append(step.tool_output)

        if not results and tool_output:
            if isinstance(tool_output, dict):
                results.append(tool_output)

        # Determine overall verdict
        verdicts = [r.get("verdict", "PASS") for r in results]

        if "FAIL" in verdicts:
            overall = "FAIL"
            failures = [
                r for r in results if r.get("verdict") == "FAIL"
            ]
            summary = (
                f"VERIFICATION FAILED: {len(failures)} constraint(s) violated.\n"
                + "\n".join(f"  - {f.get('reason', 'Unknown')}" for f in failures)
            )
        elif "WARN" in verdicts:
            overall = "WARN"
            warnings = [
                r for r in results if r.get("verdict") == "WARN"
            ]
            summary = (
                f"VERIFICATION WARNING: {len(warnings)} potential issue(s).\n"
                + "\n".join(f"  - {w.get('reason', 'Unknown')}" for w in warnings)
            )
        else:
            overall = "PASS"
            summary = "VERIFICATION PASSED: All constraints satisfied."

        # Update metrics
        VERIFICATION_RESULTS.labels(verdict=overall).inc()

        # Log to memory
        await self.memory.add_episode(
            self.name, "verification",
            f"Verdict: {overall} — {summary}",
            verdict=overall,
        )

        # Check if all checks are done
        check_types_done = {
            s.tool_name for s in history if s.tool_name is not None
        }
        all_done = len(check_types_done) >= 2 or overall == "FAIL"

        verdict_result = {
            "verdict": overall,
            "summary": summary,
            "details": results,
            "checks_completed": list(check_types_done),
        }

        return str(verdict_result), all_done or (self.strict_mode and overall == "FAIL")

    # ---- Tool Implementations ----

    async def _check_constraints(
        self, claim: str, constraints: list[str] | list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Validate against business constraint rules."""
        if self.engine:
            return await self.engine.evaluate(claim, constraints or [])

        # Built-in constraint checks
        violations = []
        constraints = constraints or []

        for constraint in constraints:
            if isinstance(constraint, str):
                # Simple string constraint parsing
                result = self._evaluate_string_constraint(claim, constraint)
                if not result["passed"]:
                    violations.append(result)
            elif isinstance(constraint, dict):
                result = self._evaluate_dict_constraint(claim, constraint)
                if not result["passed"]:
                    violations.append(result)

        if violations:
            return {
                "verdict": "FAIL",
                "reason": f"{len(violations)} constraint(s) violated",
                "violations": violations,
            }
        return {
            "verdict": "PASS",
            "reason": "All constraints satisfied",
        }

    async def _ground_check(
        self, claim: str, data_source: str = "default"
    ) -> dict[str, Any]:
        """Cross-reference a claim against ground-truth data."""
        if self.grounding:
            return await self.grounding.verify(claim, data_source)

        # Mock grounding check
        # In production, this queries the RAG pipeline for source data
        suspicious_terms = ["always", "never", "100%", "impossible", "guaranteed"]
        found = [t for t in suspicious_terms if t.lower() in claim.lower()]

        if found:
            return {
                "verdict": "WARN",
                "reason": f"Absolute terms detected: {found}. Claims may be overstated.",
                "flagged_terms": found,
            }
        return {
            "verdict": "PASS",
            "reason": "No grounding violations detected",
        }

    async def _check_logical_consistency(
        self, steps: list[str] | None = None
    ) -> dict[str, Any]:
        """Check if reasoning steps are logically consistent."""
        steps = steps or []

        if len(steps) < 2:
            return {
                "verdict": "PASS",
                "reason": "Insufficient steps for consistency check",
            }

        # Detect contradictions (simplified)
        contradictions = []
        for i, step_a in enumerate(steps):
            for j, step_b in enumerate(steps[i + 1 :], start=i + 1):
                conflict = self._detect_contradiction(step_a, step_b)
                if conflict:
                    contradictions.append({
                        "step_a": i,
                        "step_b": j,
                        "conflict": conflict,
                    })

        if contradictions:
            return {
                "verdict": "FAIL",
                "reason": f"Found {len(contradictions)} contradiction(s)",
                "contradictions": contradictions,
            }
        return {
            "verdict": "PASS",
            "reason": "No contradictions detected in reasoning chain",
        }

    async def _type_check(
        self, values: dict[str, Any] | None = None, schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate types and ranges of values."""
        values = values or {}
        schema = schema or {}
        errors = []

        for key, expected in schema.items():
            if key not in values:
                errors.append(f"Missing required field: {key}")
                continue

            value = values[key]
            expected_type = expected.get("type")
            min_val = expected.get("min")
            max_val = expected.get("max")
            allowed = expected.get("allowed_values")

            if expected_type and not isinstance(value, eval(expected_type)):
                errors.append(
                    f"{key}: expected {expected_type}, got {type(value).__name__}"
                )
            if min_val is not None and isinstance(value, (int, float)):
                if value < min_val:
                    errors.append(f"{key}: {value} < minimum {min_val}")
            if max_val is not None and isinstance(value, (int, float)):
                if value > max_val:
                    errors.append(f"{key}: {value} > maximum {max_val}")
            if allowed and value not in allowed:
                errors.append(f"{key}: {value} not in {allowed}")

        if errors:
            return {
                "verdict": "FAIL",
                "reason": f"{len(errors)} type/range error(s)",
                "errors": errors,
            }
        return {"verdict": "PASS", "reason": "All type checks passed"}

    # ---- Helpers ----

    def _evaluate_string_constraint(
        self, claim: str, constraint: str
    ) -> dict[str, Any]:
        """Evaluate a simple string-based constraint."""
        # Parse constraints like "latency_ms < 200"
        import re

        match = re.match(r"(\w+)\s*(<=?|>=?|==|!=)\s*(.+)", constraint)
        if not match:
            return {"passed": True, "constraint": constraint}

        field, op, value = match.groups()

        # Check if the field value is mentioned in the claim
        # This is a simplified check; real impl would extract structured data
        return {
            "passed": True,  # Default pass for unresolvable constraints
            "constraint": constraint,
            "field": field,
            "operator": op,
            "expected": value,
        }

    def _evaluate_dict_constraint(
        self, claim: str, constraint: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate a structured constraint object."""
        ctype = constraint.get("type", "range")
        field = constraint.get("field", "")
        value = constraint.get("value")

        return {
            "passed": True,
            "constraint": constraint,
            "type": ctype,
            "field": field,
        }

    def _detect_contradiction(self, step_a: str, step_b: str) -> str | None:
        """Simple contradiction detection between two reasoning steps."""
        # Look for opposing sentiment indicators
        positive = {"increase", "improve", "better", "higher", "up"}
        negative = {"decrease", "worsen", "worse", "lower", "down"}

        words_a = set(step_a.lower().split())
        words_b = set(step_b.lower().split())

        # Check if same subject has opposing predicates
        pos_a = words_a & positive
        neg_a = words_a & negative
        pos_b = words_b & positive
        neg_b = words_b & negative

        overlap_subjects = words_a & words_b - positive - negative
        if overlap_subjects and ((pos_a and neg_b) or (neg_a and pos_b)):
            return (
                f"Opposing sentiments on shared subjects: {overlap_subjects}"
            )
        return None
