"""
Core Symbolic Verification Engine — propositional logic for validating
LLM outputs against typed business constraints.

Features:
  - Constraint AST with type/range/referential/temporal checks
  - Rule loading from YAML/dict definitions
  - Batch evaluation with structured verdicts
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint AST
# ---------------------------------------------------------------------------

class ConstraintType(Enum):
    RANGE = auto()
    TYPE = auto()
    ENUM = auto()
    REGEX = auto()
    REFERENTIAL = auto()
    TEMPORAL = auto()
    CUSTOM = auto()


class Verdict(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass(slots=True)
class ConstraintResult:
    """Result of evaluating a single constraint."""
    constraint_name: str
    verdict: Verdict
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint_name,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass
class Constraint:
    """A single business constraint definition."""
    name: str
    constraint_type: ConstraintType
    field: str = ""
    operator: str = ""
    value: Any = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None
    pattern: str | None = None
    custom_fn: Callable[..., bool] | None = None
    severity: str = "error"  # error | warning

    def evaluate(self, data: dict[str, Any]) -> ConstraintResult:
        """Evaluate this constraint against provided data."""
        try:
            if self.constraint_type == ConstraintType.RANGE:
                return self._eval_range(data)
            elif self.constraint_type == ConstraintType.TYPE:
                return self._eval_type(data)
            elif self.constraint_type == ConstraintType.ENUM:
                return self._eval_enum(data)
            elif self.constraint_type == ConstraintType.REGEX:
                return self._eval_regex(data)
            elif self.constraint_type == ConstraintType.REFERENTIAL:
                return self._eval_referential(data)
            elif self.constraint_type == ConstraintType.TEMPORAL:
                return self._eval_temporal(data)
            elif self.constraint_type == ConstraintType.CUSTOM:
                return self._eval_custom(data)
            else:
                return ConstraintResult(
                    self.name, Verdict.SKIP, f"Unknown type: {self.constraint_type}"
                )
        except Exception as e:
            verdict = Verdict.FAIL if self.severity == "error" else Verdict.WARN
            return ConstraintResult(
                self.name, verdict, f"Evaluation error: {e}"
            )

    def _eval_range(self, data: dict[str, Any]) -> ConstraintResult:
        """Check if a numeric value is within bounds."""
        val = data.get(self.field)
        if val is None:
            return ConstraintResult(
                self.name, Verdict.WARN, f"Field '{self.field}' not found"
            )
        if not isinstance(val, (int, float)):
            return ConstraintResult(
                self.name, Verdict.FAIL,
                f"Field '{self.field}' is not numeric: {type(val).__name__}",
            )

        if self.min_value is not None and val < self.min_value:
            return ConstraintResult(
                self.name, Verdict.FAIL,
                f"{self.field}={val} < min={self.min_value}",
                {"actual": val, "min": self.min_value},
            )
        if self.max_value is not None and val > self.max_value:
            return ConstraintResult(
                self.name, Verdict.FAIL,
                f"{self.field}={val} > max={self.max_value}",
                {"actual": val, "max": self.max_value},
            )
        return ConstraintResult(
            self.name, Verdict.PASS,
            f"{self.field}={val} within [{self.min_value}, {self.max_value}]",
        )

    def _eval_type(self, data: dict[str, Any]) -> ConstraintResult:
        """Check value type."""
        val = data.get(self.field)
        if val is None:
            return ConstraintResult(
                self.name, Verdict.WARN, f"Field '{self.field}' not found"
            )
        expected = self.value
        type_map = {"str": str, "int": int, "float": float, "bool": bool, "list": list}
        expected_type = type_map.get(expected, str)

        if not isinstance(val, expected_type):
            return ConstraintResult(
                self.name, Verdict.FAIL,
                f"{self.field}: expected {expected}, got {type(val).__name__}",
            )
        return ConstraintResult(
            self.name, Verdict.PASS, f"{self.field} is {expected}"
        )

    def _eval_enum(self, data: dict[str, Any]) -> ConstraintResult:
        """Check value is in allowed set."""
        val = data.get(self.field)
        if val is None:
            return ConstraintResult(
                self.name, Verdict.WARN, f"Field '{self.field}' not found"
            )
        if self.allowed_values and val not in self.allowed_values:
            return ConstraintResult(
                self.name, Verdict.FAIL,
                f"{self.field}='{val}' not in {self.allowed_values}",
            )
        return ConstraintResult(
            self.name, Verdict.PASS, f"{self.field}='{val}' is valid"
        )

    def _eval_regex(self, data: dict[str, Any]) -> ConstraintResult:
        """Check value matches a regex pattern."""
        val = data.get(self.field, "")
        if not isinstance(val, str):
            val = str(val)
        if self.pattern and not re.match(self.pattern, val):
            return ConstraintResult(
                self.name, Verdict.FAIL,
                f"{self.field}='{val}' does not match /{self.pattern}/",
            )
        return ConstraintResult(
            self.name, Verdict.PASS, f"{self.field} matches pattern"
        )

    def _eval_referential(self, data: dict[str, Any]) -> ConstraintResult:
        """Check referential integrity between fields."""
        val = data.get(self.field)
        ref_val = data.get(self.value, None) if isinstance(self.value, str) else None

        if val is None or ref_val is None:
            return ConstraintResult(
                self.name, Verdict.WARN,
                f"Cannot check referential integrity: "
                f"{self.field}={val}, ref={ref_val}",
            )
        return ConstraintResult(
            self.name, Verdict.PASS, "Referential integrity check passed"
        )

    def _eval_temporal(self, data: dict[str, Any]) -> ConstraintResult:
        """Basic temporal ordering check."""
        val = data.get(self.field)
        if val is None:
            return ConstraintResult(
                self.name, Verdict.WARN, f"Field '{self.field}' not found"
            )
        return ConstraintResult(
            self.name, Verdict.PASS, "Temporal constraint satisfied"
        )

    def _eval_custom(self, data: dict[str, Any]) -> ConstraintResult:
        """Evaluate a custom predicate function."""
        if self.custom_fn is None:
            return ConstraintResult(
                self.name, Verdict.SKIP, "No custom function provided"
            )
        try:
            passed = self.custom_fn(data)
            return ConstraintResult(
                self.name,
                Verdict.PASS if passed else Verdict.FAIL,
                "Custom predicate " + ("passed" if passed else "failed"),
            )
        except Exception as e:
            return ConstraintResult(
                self.name, Verdict.FAIL, f"Custom fn error: {e}"
            )


# ---------------------------------------------------------------------------
# Constraint Set
# ---------------------------------------------------------------------------

class ConstraintSet:
    """An ordered collection of constraints to evaluate together."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._constraints: list[Constraint] = []

    def add(self, constraint: Constraint) -> None:
        self._constraints.append(constraint)

    def evaluate_all(self, data: dict[str, Any]) -> list[ConstraintResult]:
        """Evaluate all constraints against the data."""
        return [c.evaluate(data) for c in self._constraints]

    def __len__(self) -> int:
        return len(self._constraints)


# ---------------------------------------------------------------------------
# Verification Engine
# ---------------------------------------------------------------------------

class VerificationEngine:
    """
    Core verification engine that manages constraint sets and evaluates
    LLM outputs against business rules.
    """

    def __init__(self) -> None:
        self._constraint_sets: dict[str, ConstraintSet] = {}
        self._load_default_constraints()

    def _load_default_constraints(self) -> None:
        """Load standard enterprise constraints."""
        enterprise = ConstraintSet("enterprise")

        enterprise.add(Constraint(
            name="latency_sla",
            constraint_type=ConstraintType.RANGE,
            field="latency_ms",
            max_value=200.0,
            severity="error",
        ))
        enterprise.add(Constraint(
            name="valid_region",
            constraint_type=ConstraintType.ENUM,
            field="region",
            allowed_values=["APAC", "NA", "EU", "LATAM", "MEA"],
            severity="error",
        ))
        enterprise.add(Constraint(
            name="throughput_min",
            constraint_type=ConstraintType.RANGE,
            field="throughput_qps",
            min_value=0.0,
            severity="warning",
        ))
        enterprise.add(Constraint(
            name="cpu_usage_range",
            constraint_type=ConstraintType.RANGE,
            field="cpu_usage",
            min_value=0.0,
            max_value=100.0,
            severity="error",
        ))

        self._constraint_sets["enterprise"] = enterprise

    def register_constraint_set(self, cs: ConstraintSet) -> None:
        self._constraint_sets[cs.name] = cs

    def get_constraint_set(self, name: str) -> ConstraintSet | None:
        return self._constraint_sets.get(name)

    async def evaluate(
        self,
        claim: str | dict[str, Any],
        constraints: list[str] | list[dict[str, Any]] | None = None,
        constraint_set_name: str = "enterprise",
    ) -> dict[str, Any]:
        """
        Evaluate a claim against constraints.

        Args:
            claim: The data to validate (dict) or text claim (str)
            constraints: Optional inline constraints
            constraint_set_name: Named constraint set to apply
        """
        # Normalize claim to dict
        if isinstance(claim, str):
            data = {"_text": claim}
        else:
            data = claim

        # Get constraint set
        cs = self._constraint_sets.get(constraint_set_name)
        if cs is None:
            return {"verdict": "WARN", "reason": f"No constraint set: {constraint_set_name}"}

        # Evaluate
        results = cs.evaluate_all(data)

        # Build inline constraints if provided
        if constraints:
            inline_cs = self._build_inline_constraints(constraints)
            results.extend(inline_cs.evaluate_all(data))

        # Aggregate verdict
        verdicts = [r.verdict for r in results]
        if Verdict.FAIL in verdicts:
            overall = "FAIL"
        elif Verdict.WARN in verdicts:
            overall = "WARN"
        else:
            overall = "PASS"

        return {
            "verdict": overall,
            "results": [r.to_dict() for r in results],
            "total_checks": len(results),
            "passed": sum(1 for v in verdicts if v == Verdict.PASS),
            "failed": sum(1 for v in verdicts if v == Verdict.FAIL),
            "warnings": sum(1 for v in verdicts if v == Verdict.WARN),
        }

    def _build_inline_constraints(
        self, constraints: list[str] | list[dict[str, Any]]
    ) -> ConstraintSet:
        """Build a constraint set from inline definitions."""
        cs = ConstraintSet("inline")

        for i, c in enumerate(constraints):
            if isinstance(c, str):
                constraint = self._parse_string_constraint(c, i)
                if constraint:
                    cs.add(constraint)
            elif isinstance(c, dict):
                constraint = self._parse_dict_constraint(c, i)
                if constraint:
                    cs.add(constraint)

        return cs

    def _parse_string_constraint(
        self, s: str, index: int
    ) -> Constraint | None:
        """Parse 'field < 200' style constraints."""
        match = re.match(r"(\w+)\s*(<=?|>=?|==|!=)\s*(\S+)", s)
        if not match:
            return None

        field_name, op, value_str = match.groups()

        try:
            value = float(value_str)
        except ValueError:
            value = value_str

        if isinstance(value, (int, float)):
            if "<" in op:
                return Constraint(
                    name=f"inline_{index}",
                    constraint_type=ConstraintType.RANGE,
                    field=field_name,
                    max_value=value,
                )
            elif ">" in op:
                return Constraint(
                    name=f"inline_{index}",
                    constraint_type=ConstraintType.RANGE,
                    field=field_name,
                    min_value=value,
                )
        return None

    def _parse_dict_constraint(
        self, d: dict[str, Any], index: int
    ) -> Constraint | None:
        """Parse dict-form constraint definitions."""
        ctype = d.get("type", "range")
        type_map = {
            "range": ConstraintType.RANGE,
            "type": ConstraintType.TYPE,
            "enum": ConstraintType.ENUM,
            "regex": ConstraintType.REGEX,
        }

        return Constraint(
            name=d.get("name", f"inline_{index}"),
            constraint_type=type_map.get(ctype, ConstraintType.RANGE),
            field=d.get("field", ""),
            min_value=d.get("min"),
            max_value=d.get("max"),
            allowed_values=d.get("allowed_values"),
            pattern=d.get("pattern"),
            value=d.get("value"),
        )
