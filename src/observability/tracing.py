"""
Observability layer: LangFuse tracing + Prometheus metrics.

Every agent request gets a LangFuse trace with spans for:
- Query routing
- Each tool call (with latency)
- LLM call (tokens in/out, latency)
- Cache lookups (hit/miss)
- Schema monitor checks

Prometheus counters/histograms are scraped by the Grafana dashboard.
This is what made diagnosing Failures 2, 3, 4, and 5 possible —
without traces, all you see is "the agent is slow" or "it returned wrong data".
"""

from __future__ import annotations

import contextlib
import time
from contextlib import contextmanager
from typing import Any, Iterator

from langfuse import Langfuse
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from src.config import settings


# ─── Prometheus Metrics ───────────────────────────────────────────────────────

class PrometheusMetrics:
    """All Prometheus metrics for the procurement agent."""

    def __init__(self):
        self.llm_latency = Histogram(
            "procurement_agent_llm_latency_ms",
            "LLM call latency in milliseconds",
            buckets=[100, 250, 500, 1000, 2000, 5000, 10000, 15000],
        )
        self.tool_latency = Histogram(
            "procurement_agent_tool_latency_ms",
            "Tool call latency in milliseconds",
            ["tool_name"],
            buckets=[10, 50, 100, 250, 500, 1000, 2000, 5000],
        )
        self.cache_hits = Counter(
            "procurement_agent_cache_hits_total",
            "Semantic cache hits",
        )
        self.cache_misses = Counter(
            "procurement_agent_cache_misses_total",
            "Semantic cache misses",
        )
        self.cache_lookup_latency = Histogram(
            "procurement_agent_cache_lookup_ms",
            "Semantic cache lookup latency",
            buckets=[0.5, 1, 2, 5, 10, 20, 50],
        )
        self.token_refresh_total = Counter(
            "procurement_agent_oauth_token_refresh_total",
            "OAuth2 token refresh attempts",
            ["success"],
        )
        self.agent_requests = Counter(
            "procurement_agent_requests_total",
            "Total agent requests",
            ["intent", "status"],
        )
        self.active_approval_workflows = Gauge(
            "procurement_agent_active_approvals",
            "Approval workflows currently in progress",
        )
        self.schema_drift_detected = Counter(
            "procurement_agent_schema_drift_total",
            "Schema drift detection events",
            ["severity"],
        )

    def record_llm_latency(self, latency_ms: float) -> None:
        self.llm_latency.observe(latency_ms)

    def record_tool_latency(self, tool_name: str, latency_ms: float) -> None:
        self.tool_latency.labels(tool_name=tool_name).observe(latency_ms)

    def record_cache_lookup(self, hit: bool, latency_ms: float) -> None:
        if hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()
        self.cache_lookup_latency.observe(latency_ms)

    def record_token_refresh(self, success: bool) -> None:
        self.token_refresh_total.labels(success=str(success)).inc()

    def record_request(self, intent: str, status: str = "success") -> None:
        self.agent_requests.labels(intent=intent, status=status).inc()

    def record_schema_drift(self, severity: str) -> None:
        self.schema_drift_detected.labels(severity=severity).inc()

    def start_metrics_server(self) -> None:
        start_http_server(settings.PROMETHEUS_PORT)


# ─── LangFuse Tracer ──────────────────────────────────────────────────────────

class Tracer:
    """
    Thin wrapper around LangFuse for distributed tracing.
    Falls back gracefully if LangFuse is unavailable (e.g., in unit tests).
    """

    def __init__(self):
        self._lf: Langfuse | None = None
        self._enabled = bool(settings.LANGFUSE_SECRET_KEY)

    def _get_lf(self) -> Langfuse | None:
        if not self._enabled:
            return None
        if self._lf is None:
            self._lf = Langfuse(
                secret_key=settings.LANGFUSE_SECRET_KEY,
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                host=settings.LANGFUSE_HOST,
            )
        return self._lf

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[_Span]:
        """Context manager that creates a LangFuse span + records Prometheus latency."""
        lf = self._get_lf()
        span = _Span(name=name, lf=lf, attributes=attributes or {})
        t0 = time.perf_counter()
        try:
            yield span
        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_attribute("status", "error")
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            span.set_attribute("duration_ms", latency_ms)
            span._end()
            # Record to Prometheus if it's a tool span
            if name.startswith("tool."):
                metrics.record_tool_latency(name[5:], latency_ms)

    def trace(self, name: str, **kwargs) -> "_Trace":
        """Start a top-level LangFuse trace (one per agent request)."""
        lf = self._get_lf()
        return _Trace(name=name, lf=lf, **kwargs)


class _Span:
    """Thin span wrapper — attributes stored locally, flushed on end."""

    def __init__(self, name: str, lf: Langfuse | None, attributes: dict):
        self.name = name
        self._lf = lf
        self._attrs = dict(attributes)
        self._lf_span = None
        if lf:
            with contextlib.suppress(Exception):
                self._lf_span = lf.span(name=name, metadata=attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        self._attrs[key] = value
        if self._lf_span:
            with contextlib.suppress(Exception):
                self._lf_span.update(metadata={key: value})

    def _end(self) -> None:
        if self._lf_span:
            with contextlib.suppress(Exception):
                self._lf_span.end(metadata=self._attrs)


class _Trace:
    def __init__(self, name: str, lf: Langfuse | None, **kwargs):
        self._lf = lf
        self._trace = None
        if lf:
            with contextlib.suppress(Exception):
                self._trace = lf.trace(name=name, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._trace and self._lf:
            with contextlib.suppress(Exception):
                self._lf.flush()


# ─── Singletons ───────────────────────────────────────────────────────────────

tracer = Tracer()
metrics = PrometheusMetrics()
