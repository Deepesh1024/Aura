"""
Memory Budget Manager — Prevents context-window saturation when multiple
agents participate in long-running tasks.

The Problem:
  When Planner → Data Architect → Verifier all contribute to a multi-turn task,
  naive memory concatenation blows the context window (4K-128K tokens depending
  on model). Each agent's conversation history, episodic logs, and retrieved
  context all compete for the same finite token budget.

The Solution:
  Token-aware memory budgeting with priority-based eviction:
    1. Hard token budget per agent per task (configurable)
    2. Sliding summarization — old context → compressed summaries
    3. Cross-agent deduplication — shared facts stored once
    4. Priority tiers: grounding data > recent tool outputs > conversation > summaries
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token Estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Fast token estimation (avoids loading tiktoken for every call).
    Rule of thumb: 1 token ≈ 4 characters for English text.
    Accurate to ~10% for GPT-family models.
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Memory Tier — priority-ordered segments
# ---------------------------------------------------------------------------

class MemoryTier:
    """Priority levels for context allocation. Lower number = higher priority."""
    GROUNDING = 0       # Facts that MUST be in context (constraints, schemas)
    TOOL_OUTPUT = 1     # Recent tool results (SQL data, search results)
    CONVERSATION = 2    # Recent conversation turns
    EPISODIC = 3        # Past event log entries
    SUMMARY = 4         # Compressed summaries of evicted content


@dataclass(slots=True)
class ContextBlock:
    """A block of context with its priority and token cost."""
    content: str
    tier: int
    source: str          # agent name or "shared"
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    is_summary: bool = False

    def __post_init__(self) -> None:
        if self.tokens == 0:
            self.tokens = estimate_tokens(self.content)


# ---------------------------------------------------------------------------
# Memory Budget Manager
# ---------------------------------------------------------------------------

class MemoryBudgetManager:
    """
    Manages context-window token budgets across multiple agents in a task.

    Architecture:
    ┌─────────────────────────────────────────────┐
    │              Total Token Budget              │
    │  (e.g., 8192 tokens for context window)      │
    ├─────────┬─────────┬─────────┬───────────────┤
    │Grounding│  Tool   │  Conv   │   Summaries   │
    │  (20%)  │ Output  │  (30%)  │    (flex)      │
    │         │  (35%)  │         │               │
    └─────────┴─────────┴─────────┴───────────────┘

    When budget is exceeded:
      1. Evict lowest-priority blocks first (summaries → episodic → old conv)
      2. Summarize evicted blocks into compressed summaries
      3. Deduplicate cross-agent shared facts
    """

    def __init__(
        self,
        total_budget_tokens: int = 8192,
        tier_budgets: dict[int, float] | None = None,
        summary_ratio: float = 0.25,
    ) -> None:
        self.total_budget = total_budget_tokens
        self.summary_ratio = summary_ratio  # Compression target for summaries

        # Default tier budget allocation (percentage of total)
        self._tier_budgets = tier_budgets or {
            MemoryTier.GROUNDING: 0.20,
            MemoryTier.TOOL_OUTPUT: 0.35,
            MemoryTier.CONVERSATION: 0.30,
            MemoryTier.EPISODIC: 0.10,
            MemoryTier.SUMMARY: 0.05,
        }

        # Active context blocks indexed by agent
        self._blocks: dict[str, list[ContextBlock]] = {}
        # Shared context (grounding data, constraints) — stored once
        self._shared: list[ContextBlock] = []
        # Eviction log for auditing
        self._eviction_log: list[dict[str, Any]] = []

    # ---- Core API ----

    def add_context(
        self,
        agent_name: str,
        content: str,
        tier: int = MemoryTier.CONVERSATION,
        shared: bool = False,
    ) -> bool:
        """
        Add a context block. Returns True if added without eviction.
        Triggers eviction if budget exceeded.
        """
        block = ContextBlock(content=content, tier=tier, source=agent_name)

        if shared:
            # Check for duplicate shared content
            if not self._is_duplicate_shared(content):
                self._shared.append(block)
        else:
            self._blocks.setdefault(agent_name, []).append(block)

        # Check budget and evict if needed
        needed_eviction = self._enforce_budget()
        return not needed_eviction

    def get_context_for_agent(
        self,
        agent_name: str,
        max_tokens: int | None = None,
    ) -> str:
        """
        Assemble the context window for a specific agent.
        Returns content ordered by priority (highest first).
        """
        max_tokens = max_tokens or self.total_budget
        blocks: list[ContextBlock] = []

        # 1. Shared grounding (always included first)
        blocks.extend(self._shared)

        # 2. Agent-specific blocks
        agent_blocks = self._blocks.get(agent_name, [])
        blocks.extend(agent_blocks)

        # Sort by tier (priority), then by recency
        blocks.sort(key=lambda b: (b.tier, -b.timestamp))

        # Assemble within budget
        assembled: list[str] = []
        used_tokens = 0

        for block in blocks:
            if used_tokens + block.tokens > max_tokens:
                break
            assembled.append(block.content)
            used_tokens += block.tokens

        return "\n\n".join(assembled)

    def get_cross_agent_context(
        self,
        agents: list[str],
        max_tokens: int | None = None,
    ) -> str:
        """
        Assemble context for multi-agent coordination.
        Deduplicates shared facts and allocates budget per-agent.
        """
        max_tokens = max_tokens or self.total_budget
        per_agent_budget = (max_tokens - self._shared_tokens()) // max(len(agents), 1)

        parts: list[str] = []
        used = 0

        # Shared context first
        for block in self._shared:
            if used + block.tokens <= max_tokens:
                parts.append(f"[SHARED] {block.content}")
                used += block.tokens

        # Per-agent context with budget cap
        for agent in agents:
            agent_blocks = sorted(
                self._blocks.get(agent, []),
                key=lambda b: (b.tier, -b.timestamp),
            )
            agent_used = 0
            for block in agent_blocks:
                if agent_used + block.tokens > per_agent_budget:
                    break
                if used + block.tokens > max_tokens:
                    break
                parts.append(f"[{agent.upper()}] {block.content}")
                used += block.tokens
                agent_used += block.tokens

        return "\n\n".join(parts)

    # ---- Budget Enforcement ----

    def _enforce_budget(self) -> bool:
        """
        Enforce the total token budget.
        Evicts lowest-priority, oldest blocks first.
        Returns True if eviction was needed.
        """
        total = self._total_tokens()
        if total <= self.total_budget:
            return False

        overflow = total - self.total_budget
        evicted = 0

        # Gather all blocks, sort by eviction priority (lowest priority, oldest first)
        all_blocks: list[tuple[str, int, ContextBlock]] = []
        for agent, blocks in self._blocks.items():
            for i, block in enumerate(blocks):
                all_blocks.append((agent, i, block))

        # Sort: highest tier number first (lowest priority), then oldest
        all_blocks.sort(key=lambda x: (-x[2].tier, x[2].timestamp))

        indices_to_remove: list[tuple[str, int]] = []

        for agent, idx, block in all_blocks:
            if evicted >= overflow:
                break

            # Don't evict grounding data
            if block.tier == MemoryTier.GROUNDING:
                continue

            # Summarize before evicting
            summary = self._compress_block(block)
            if summary:
                self._blocks.setdefault(agent, []).append(summary)

            indices_to_remove.append((agent, idx))
            evicted += block.tokens

            self._eviction_log.append({
                "agent": agent,
                "tier": block.tier,
                "tokens_freed": block.tokens,
                "content_preview": block.content[:80],
                "timestamp": time.time(),
            })

        # Remove evicted blocks (reverse order to preserve indices)
        for agent, idx in sorted(indices_to_remove, key=lambda x: -x[1]):
            blocks = self._blocks.get(agent, [])
            if idx < len(blocks):
                blocks.pop(idx)

        logger.info(
            "Memory eviction: freed %d tokens (%d blocks), budget=%d/%d",
            evicted,
            len(indices_to_remove),
            self._total_tokens(),
            self.total_budget,
        )
        return True

    def _compress_block(self, block: ContextBlock) -> ContextBlock | None:
        """
        Compress a block into a summary.
        Target: reduce to summary_ratio of original tokens.
        """
        if block.is_summary or block.tokens < 50:
            return None  # Don't re-summarize or compress tiny blocks

        # Extractive summarization: take first and last portions
        target_chars = int(len(block.content) * self.summary_ratio)
        if target_chars < 20:
            return None

        half = target_chars // 2
        compressed = (
            block.content[:half].rstrip()
            + " [...] "
            + block.content[-half:].lstrip()
        )

        return ContextBlock(
            content=f"[SUMMARY] {compressed}",
            tier=MemoryTier.SUMMARY,
            source=block.source,
            is_summary=True,
        )

    # ---- Deduplication ----

    def _is_duplicate_shared(self, content: str) -> bool:
        """Check if content is already in shared context."""
        content_hash = hash(content.strip().lower()[:200])
        for existing in self._shared:
            if hash(existing.content.strip().lower()[:200]) == content_hash:
                return True
        return False

    # ---- Token Accounting ----

    def _total_tokens(self) -> int:
        total = sum(b.tokens for b in self._shared)
        for blocks in self._blocks.values():
            total += sum(b.tokens for b in blocks)
        return total

    def _shared_tokens(self) -> int:
        return sum(b.tokens for b in self._shared)

    def get_budget_report(self) -> dict[str, Any]:
        """Detailed token budget report for observability."""
        tier_usage: dict[int, int] = {}
        agent_usage: dict[str, int] = {}

        for block in self._shared:
            tier_usage[block.tier] = tier_usage.get(block.tier, 0) + block.tokens

        for agent, blocks in self._blocks.items():
            agent_total = 0
            for block in blocks:
                tier_usage[block.tier] = tier_usage.get(block.tier, 0) + block.tokens
                agent_total += block.tokens
            agent_usage[agent] = agent_total

        total = self._total_tokens()
        utilization = total / self.total_budget if self.total_budget > 0 else 0

        return {
            "total_budget": self.total_budget,
            "used_tokens": total,
            "remaining_tokens": max(0, self.total_budget - total),
            "utilization": f"{utilization:.1%}",
            "tier_breakdown": {
                "grounding": tier_usage.get(MemoryTier.GROUNDING, 0),
                "tool_output": tier_usage.get(MemoryTier.TOOL_OUTPUT, 0),
                "conversation": tier_usage.get(MemoryTier.CONVERSATION, 0),
                "episodic": tier_usage.get(MemoryTier.EPISODIC, 0),
                "summaries": tier_usage.get(MemoryTier.SUMMARY, 0),
            },
            "per_agent": agent_usage,
            "shared_tokens": self._shared_tokens(),
            "evictions": len(self._eviction_log),
            "recent_evictions": self._eviction_log[-3:],
        }

    def clear_agent(self, agent_name: str) -> None:
        """Clear all context for a specific agent."""
        self._blocks.pop(agent_name, None)

    def reset(self) -> None:
        """Clear all context."""
        self._blocks.clear()
        self._shared.clear()
        self._eviction_log.clear()
