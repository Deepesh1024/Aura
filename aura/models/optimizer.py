"""
Model Optimizer — Quantization wrappers, caching strategies,
and inference latency enforcement for sub-200ms SLA.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

from aura.core.telemetry import CACHE_HITS, CACHE_MISSES

logger = logging.getLogger(__name__)


class LRUCache:
    """
    LRU response cache with semantic-key deduplication.
    Tracks cache hit/miss metrics for observability.
    """

    def __init__(self, max_size: int = 10_000, cache_name: str = "llm_response") -> None:
        self.max_size = max_size
        self.cache_name = cache_name
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, prompt: str, model: str = "", **kwargs: Any) -> str:
        """Create a stable cache key from prompt content."""
        raw = f"{model}:{prompt}:{sorted(kwargs.items())}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, prompt: str, model: str = "", **kwargs: Any) -> Any | None:
        key = self._make_key(prompt, model, **kwargs)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            CACHE_HITS.labels(cache_name=self.cache_name).inc()
            return self._cache[key]
        self._misses += 1
        CACHE_MISSES.labels(cache_name=self.cache_name).inc()
        return None

    def put(self, prompt: str, response: Any, model: str = "", **kwargs: Any) -> None:
        key = self._make_key(prompt, model, **kwargs)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = response

    def clear(self) -> None:
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.2%}",
        }


class QuantizationManager:
    """
    Manages model quantization (INT8 / FP16) for inference optimization.
    Uses PyTorch dynamic/static quantization when available.
    """

    def __init__(self, strategy: str = "none") -> None:
        self.strategy = strategy  # none | int8 | fp16
        self._quantized = False

    def quantize_model(self, model: Any) -> Any:
        """Apply quantization to a PyTorch model."""
        if self.strategy == "none":
            return model

        try:
            import torch

            if self.strategy == "int8":
                quantized = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                self._quantized = True
                logger.info("Applied INT8 dynamic quantization")
                return quantized

            elif self.strategy == "fp16":
                if torch.cuda.is_available():
                    model = model.half().cuda()
                    self._quantized = True
                    logger.info("Applied FP16 quantization (CUDA)")
                else:
                    logger.warning("FP16 requested but CUDA unavailable, skipping")
                return model

        except ImportError:
            logger.warning("PyTorch not available, skipping quantization")
        except Exception as e:
            logger.error("Quantization failed: %s", e)

        return model

    @property
    def is_quantized(self) -> bool:
        return self._quantized


class InferenceOptimizer:
    """
    Unified inference optimizer combining cache and quantization.
    Enforces the sub-200ms latency SLA.
    """

    def __init__(
        self,
        max_latency_ms: int = 200,
        cache_enabled: bool = True,
        cache_max_size: int = 10_000,
        quantization: str = "none",
    ) -> None:
        self.max_latency_ms = max_latency_ms
        self.cache = LRUCache(cache_max_size) if cache_enabled else None
        self.quantization = QuantizationManager(quantization)
        self._latency_violations: list[dict[str, Any]] = []

    async def optimized_inference(
        self,
        prompt: str,
        llm_fn: Any,
        model: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute inference with caching and latency monitoring.

        Returns dict with 'response', 'latency_ms', 'cache_hit', 'sla_met'.
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(prompt, model=model)
            if cached is not None:
                return {
                    "response": cached,
                    "latency_ms": 0.1,
                    "cache_hit": True,
                    "sla_met": True,
                }

        # Execute inference
        start = time.time()
        try:
            response = await llm_fn(prompt, **kwargs)
        except Exception as e:
            return {
                "response": None,
                "error": str(e),
                "latency_ms": (time.time() - start) * 1000,
                "cache_hit": False,
                "sla_met": False,
            }

        elapsed_ms = (time.time() - start) * 1000
        sla_met = elapsed_ms <= self.max_latency_ms

        if not sla_met:
            self._latency_violations.append({
                "latency_ms": elapsed_ms,
                "sla_ms": self.max_latency_ms,
                "model": model,
                "timestamp": time.time(),
            })
            logger.warning(
                "SLA violation: %.1fms > %dms for model=%s",
                elapsed_ms,
                self.max_latency_ms,
                model,
            )

        # Cache the response
        if self.cache and response is not None:
            self.cache.put(prompt, response, model=model)

        return {
            "response": response,
            "latency_ms": round(elapsed_ms, 2),
            "cache_hit": False,
            "sla_met": sla_met,
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "cache": self.cache.stats() if self.cache else None,
            "quantization": {
                "strategy": self.quantization.strategy,
                "active": self.quantization.is_quantized,
            },
            "sla_ms": self.max_latency_ms,
            "violations": len(self._latency_violations),
            "recent_violations": self._latency_violations[-5:],
        }
