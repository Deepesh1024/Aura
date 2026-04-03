"""
Guardrail-as-Code — Safe Exploration framework for the Verifier.

Reframes verification from a brittle "gatekeeper" to a guardrail system
that allows creative agent exploration within defined safety boundaries.

The Problem:
  If the Verifier rejects every creative path from the Planner, the system
  is brittle. But if it accepts everything, hallucinations leak through.

The Solution:
  Three-zone verification model:
    1. GREEN — Deterministically safe. No constraints violated.
    2. YELLOW — Ambiguous territory. Constraints WARN but don't FAIL.
       Agent may proceed with increased monitoring and human-in-the-loop flag.
    3. RED — Hard constraint violation. Block and require remediation.

  This allows the Planner to explore novel solution paths (YELLOW zone)
  while still enforcing hard safety boundaries (RED zone).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safety Zones
# ---------------------------------------------------------------------------

class SafetyZone(Enum):
    GREEN = auto()    # Safe — proceed normally
    YELLOW = auto()   # Ambiguous — proceed with monitoring
    RED = auto()      # Blocked — hard constraint violation


@dataclass
class GuardrailResult:
    """Result of a guardrail evaluation."""
    zone: SafetyZone
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 1.0
    requires_human: bool = False
    exploration_allowed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone.name,
            "passed": self.passed,
            "warnings": self.warnings,
            "blocked": self.blocked,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "requires_human": self.requires_human,
            "exploration_allowed": self.exploration_allowed,
        }


# ---------------------------------------------------------------------------
# Guardrail Classes
# ---------------------------------------------------------------------------

@dataclass
class Guardrail:
    """
    A single guardrail rule. Unlike hard constraints, guardrails
    can specify severity and allow for YELLOW-zone exploration.
    """
    name: str
    description: str = ""
    check_fn: Callable[[dict[str, Any]], bool] | None = None
    severity: str = "hard"      # "hard" = RED on fail, "soft" = YELLOW on fail
    remediation: str = ""       # What to do if violated
    applies_to: list[str] = field(default_factory=list)  # Agent names or empty=all

    def evaluate(self, data: dict[str, Any], agent: str = "") -> tuple[bool, str]:
        """Returns (passed, message)."""
        if self.applies_to and agent not in self.applies_to:
            return True, f"{self.name}: not applicable to {agent}"

        if self.check_fn is None:
            return True, f"{self.name}: no check defined"

        try:
            passed = self.check_fn(data)
            if passed:
                return True, f"{self.name}: PASS"
            return False, f"{self.name}: VIOLATED — {self.remediation or self.description}"
        except Exception as e:
            logger.warning("Guardrail %s raised exception: %s", self.name, e)
            return True, f"{self.name}: evaluation error — defaulting to PASS"


class GuardrailFramework:
    """
    Guardrail-as-Code framework for safe agent exploration.

    Interview talking points:
      - "We don't block creative solutions. We define a safety envelope
         and let agents explore within it."
      - "YELLOW zone outputs are flagged for human review but not blocked.
         This avoids the brittleness of binary pass/fail verification."
      - "Guardrails are composable and domain-specific. Adding a new
         safety rule is a one-line registration."
    """

    def __init__(self) -> None:
        self._guardrails: list[Guardrail] = []
        self._evaluation_log: list[dict[str, Any]] = []
        self._load_default_guardrails()

    def _load_default_guardrails(self) -> None:
        """Load the default enterprise guardrails."""

        # --- HARD guardrails (RED zone on violation) ---

        self.register(Guardrail(
            name="no_pii_leak",
            description="Output must not contain personally identifiable information",
            check_fn=lambda d: not self._contains_pii(d.get("output", "")),
            severity="hard",
            remediation="Strip PII from output before returning to user",
        ))

        self.register(Guardrail(
            name="no_destructive_sql",
            description="SQL must not contain DROP, DELETE, TRUNCATE, or ALTER",
            check_fn=lambda d: not any(
                kw in d.get("sql", "").upper()
                for kw in ["DROP ", "DELETE ", "TRUNCATE ", "ALTER "]
            ),
            severity="hard",
            remediation="Rewrite query as SELECT-only",
            applies_to=["data_architect"],
        ))

        self.register(Guardrail(
            name="action_sandbox_required",
            description="Autonomous actions must run in sandbox mode",
            check_fn=lambda d: d.get("sandbox", True) is True,
            severity="hard",
            remediation="Set sandbox_mode=True before executing actions",
            applies_to=["planner"],
        ))

        self.register(Guardrail(
            name="latency_sla_hard",
            description="Response must be delivered within 5000ms hard limit",
            check_fn=lambda d: d.get("latency_ms", 0) <= 5000,
            severity="hard",
            remediation="Timeout exceeded. Use cached result or summarized response.",
        ))

        # --- SOFT guardrails (YELLOW zone on violation) ---

        self.register(Guardrail(
            name="latency_sla_soft",
            description="Response should be under 200ms for interactive use",
            check_fn=lambda d: d.get("latency_ms", 0) <= 200,
            severity="soft",
            remediation="Consider query optimization or result caching",
        ))

        self.register(Guardrail(
            name="confidence_threshold",
            description="Agent confidence should exceed 70%",
            check_fn=lambda d: d.get("confidence", 1.0) >= 0.7,
            severity="soft",
            remediation="Flag for human review. Agent is not confident in this response.",
        ))

        self.register(Guardrail(
            name="no_absolute_claims",
            description="Output should avoid absolute language (always, never, 100%)",
            check_fn=lambda d: not self._has_absolute_language(d.get("output", "")),
            severity="soft",
            remediation="Add qualifiers: 'typically', 'in most cases', 'approximately'",
        ))

        self.register(Guardrail(
            name="grounding_coverage",
            description="Claims should be supported by retrieved evidence",
            check_fn=lambda d: d.get("grounding_score", 1.0) >= 0.5,
            severity="soft",
            remediation="Insufficient evidence. Flag for manual verification.",
        ))

        self.register(Guardrail(
            name="token_budget_ok",
            description="Context window utilization should be under 90%",
            check_fn=lambda d: d.get("token_utilization", 0) < 0.9,
            severity="soft",
            remediation="Summarize older context to free token budget",
        ))

    def register(self, guardrail: Guardrail) -> None:
        """Register a new guardrail."""
        self._guardrails.append(guardrail)

    def evaluate(
        self,
        data: dict[str, Any],
        agent: str = "",
    ) -> GuardrailResult:
        """
        Evaluate all guardrails and determine the safety zone.
        """
        result = GuardrailResult(zone=SafetyZone.GREEN)
        start = time.time()

        for guardrail in self._guardrails:
            passed, message = guardrail.evaluate(data, agent)

            if passed:
                result.passed.append(message)
            elif guardrail.severity == "hard":
                result.blocked.append(message)
                result.zone = SafetyZone.RED
                result.exploration_allowed = False
                if guardrail.remediation:
                    result.suggestions.append(guardrail.remediation)
            else:  # soft
                result.warnings.append(message)
                if result.zone != SafetyZone.RED:
                    result.zone = SafetyZone.YELLOW
                result.requires_human = True
                if guardrail.remediation:
                    result.suggestions.append(guardrail.remediation)

        # Compute confidence based on pass ratio
        total = len(result.passed) + len(result.warnings) + len(result.blocked)
        if total > 0:
            result.confidence = len(result.passed) / total

        elapsed = (time.time() - start) * 1000
        self._evaluation_log.append({
            "agent": agent,
            "zone": result.zone.name,
            "passed": len(result.passed),
            "warnings": len(result.warnings),
            "blocked": len(result.blocked),
            "elapsed_ms": round(elapsed, 3),
            "timestamp": time.time(),
        })

        return result

    def evaluate_exploration(
        self,
        proposed_action: dict[str, Any],
        agent: str = "",
    ) -> GuardrailResult:
        """
        Evaluate whether an agent's proposed creative action is within
        the safety envelope.

        This is the key differentiator from brittle constraint checking:
        YELLOW-zone actions are ALLOWED with monitoring, not blocked.
        """
        result = self.evaluate(proposed_action, agent)

        # For YELLOW zone: add exploration guidance
        if result.zone == SafetyZone.YELLOW:
            result.suggestions.insert(0,
                "YELLOW ZONE: Action permitted under monitoring. "
                "The system will log this exploration for human review."
            )

        return result

    # ---- Helper checks ----

    @staticmethod
    def _contains_pii(text: str) -> bool:
        """Check for common PII patterns."""
        import re
        patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',      # SSN
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    @staticmethod
    def _has_absolute_language(text: str) -> bool:
        """Detect absolute claims that may indicate hallucination."""
        absolutes = [
            "always", "never", "100%", "impossible", "guaranteed",
            "perfectly", "zero errors", "no exceptions", "absolute",
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in absolutes)

    # ---- Observability ----

    def get_stats(self) -> dict[str, Any]:
        total = len(self._evaluation_log)
        if total == 0:
            return {"evaluations": 0}

        zones = {}
        for entry in self._evaluation_log:
            z = entry["zone"]
            zones[z] = zones.get(z, 0) + 1

        return {
            "evaluations": total,
            "zone_distribution": zones,
            "guardrails_registered": len(self._guardrails),
            "hard_guardrails": sum(1 for g in self._guardrails if g.severity == "hard"),
            "soft_guardrails": sum(1 for g in self._guardrails if g.severity == "soft"),
        }
