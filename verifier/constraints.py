"""
Business Constraint Definitions — DSL for defining validation rules.

Examples:
  - latency_ms < 200
  - region IN ['APAC', 'NA', 'EU']
  - Custom predicate functions
"""

from __future__ import annotations

from typing import Any

from verifier.engine import Constraint, ConstraintSet, ConstraintType


def build_enterprise_constraints() -> ConstraintSet:
    """Build the standard enterprise constraint set."""
    cs = ConstraintSet("enterprise")

    # ---- Performance SLAs ----
    cs.add(Constraint(
        name="latency_sla",
        constraint_type=ConstraintType.RANGE,
        field="latency_ms",
        max_value=200.0,
        severity="error",
    ))
    cs.add(Constraint(
        name="throughput_min",
        constraint_type=ConstraintType.RANGE,
        field="throughput_qps",
        min_value=100.0,
        severity="warning",
    ))
    cs.add(Constraint(
        name="error_rate_max",
        constraint_type=ConstraintType.RANGE,
        field="error_rate",
        max_value=0.05,
        severity="error",
    ))

    # ---- Infrastructure ----
    cs.add(Constraint(
        name="cpu_usage",
        constraint_type=ConstraintType.RANGE,
        field="cpu_usage",
        min_value=0.0,
        max_value=100.0,
        severity="error",
    ))
    cs.add(Constraint(
        name="memory_usage",
        constraint_type=ConstraintType.RANGE,
        field="memory_usage",
        min_value=0.0,
        max_value=100.0,
        severity="error",
    ))

    # ---- Data Quality ----
    cs.add(Constraint(
        name="valid_region",
        constraint_type=ConstraintType.ENUM,
        field="region",
        allowed_values=["APAC", "NA", "EU", "LATAM", "MEA"],
        severity="error",
    ))
    cs.add(Constraint(
        name="valid_severity",
        constraint_type=ConstraintType.ENUM,
        field="severity",
        allowed_values=["critical", "high", "medium", "low", "info"],
        severity="error",
    ))
    cs.add(Constraint(
        name="valid_status",
        constraint_type=ConstraintType.ENUM,
        field="status",
        allowed_values=["active", "inactive", "degraded", "maintenance"],
        severity="error",
    ))

    # ---- Custom Business Logic ----
    cs.add(Constraint(
        name="cache_miss_threshold",
        constraint_type=ConstraintType.RANGE,
        field="cache_miss_rate",
        max_value=0.15,
        severity="warning",
    ))

    return cs


def build_security_constraints() -> ConstraintSet:
    """Constraints for security-sensitive operations."""
    cs = ConstraintSet("security")

    cs.add(Constraint(
        name="no_pii_in_output",
        constraint_type=ConstraintType.REGEX,
        field="_text",
        pattern=r"^(?!.*\b\d{3}-\d{2}-\d{4}\b).*$",  # No SSN patterns
        severity="error",
    ))
    cs.add(Constraint(
        name="no_credit_cards",
        constraint_type=ConstraintType.REGEX,
        field="_text",
        pattern=r"^(?!.*\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b).*$",
        severity="error",
    ))

    return cs


def from_yaml(rules: list[dict[str, Any]], name: str = "custom") -> ConstraintSet:
    """Build constraints from a list of YAML-style rule definitions."""
    cs = ConstraintSet(name)
    type_map = {
        "range": ConstraintType.RANGE,
        "type": ConstraintType.TYPE,
        "enum": ConstraintType.ENUM,
        "regex": ConstraintType.REGEX,
        "referential": ConstraintType.REFERENTIAL,
        "temporal": ConstraintType.TEMPORAL,
    }

    for rule in rules:
        cs.add(Constraint(
            name=rule.get("name", "unnamed"),
            constraint_type=type_map.get(
                rule.get("type", "range"), ConstraintType.RANGE
            ),
            field=rule.get("field", ""),
            min_value=rule.get("min"),
            max_value=rule.get("max"),
            allowed_values=rule.get("allowed_values"),
            pattern=rule.get("pattern"),
            value=rule.get("value"),
            severity=rule.get("severity", "error"),
        ))

    return cs
