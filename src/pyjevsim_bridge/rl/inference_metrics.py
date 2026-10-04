"""Opt-in, segment-local observation of existing inference work only.

This collector never caches verification, changes an artifact, or writes files.
Inclusive stage durations overlap; only ``self_ns`` values are additive. File
rows are another view of the same read/hash spans, not additional elapsed time.
Collectors support sequential work on one process/thread (including a coroutine
run synchronously by its owner's event loop), not concurrent tasks or threads.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar, Token
from functools import wraps
from pathlib import Path
from types import TracebackType
from typing import ParamSpec, Self, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")
_CURRENT: ContextVar[InferenceMetrics | None] = ContextVar("ppo_inference_metrics", default=None)


class _Span:
    def __init__(self, meter: InferenceMetrics, label: str) -> None:
        self.meter, self.label = meter, label
        self.started_ns = self.elapsed_ns = self.children_ns = 0

    def __enter__(self) -> Self:
        self.meter._assert_owner()
        self.meter._stack.append(self)
        self.started_ns = time.perf_counter_ns()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        finished = time.perf_counter_ns()
        self.meter._assert_owner()
        if not self.meter._stack or self.meter._stack[-1] is not self:
            self.meter._accounting_error = True
            raise RuntimeError("inference metric spans must close in sequential nesting order")
        self.meter._stack.pop()
        self.elapsed_ns = finished - self.started_ns
        exclusive = self.elapsed_ns - self.children_ns
        if exclusive < 0:
            self.meter._accounting_error = True
            raise RuntimeError("overlapping inference work cannot be reported as exclusive time")
        if self.meter._stack:
            self.meter._stack[-1].children_ns += self.elapsed_ns
        else:
            self.meter._root_ns += self.elapsed_ns
        row = self.meter._stages.setdefault(
            self.label, {"calls": 0, "failed_calls": 0, "inclusive_ns": 0, "self_ns": 0},
        )
        row["calls"] += 1
        row["failed_calls"] += int(exc_type is not None)
        row["inclusive_ns"] += self.elapsed_ns
        row["self_ns"] += exclusive


class InferenceMetrics:
    """One explicitly selected prefix/suffix collector; disabled is not zero cost."""

    def __init__(self, *, phase: str, enabled: bool = True) -> None:
        if not isinstance(phase, str) or not phase:
            raise ValueError("metric phase must be nonempty text")
        if type(enabled) is not bool:
            raise ValueError("metric enabled must be bool")
        self.phase, self.enabled = phase, enabled
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._token: Token[InferenceMetrics | None] | None = None
        self._entered = self._closed = self._body_failed = self._accounting_error = False
        self._stack: list[_Span] = []
        self._stages: dict[str, dict[str, int]] = {}
        self._files: dict[str, dict[str, int]] = {}
        self._root_ns = 0

    def _assert_owner(self) -> None:
        if (os.getpid(), threading.get_ident()) != (self._pid, self._thread):
            raise RuntimeError("inference metrics belong to one process/thread")
        if not self._entered or self._closed:
            raise RuntimeError("inference metrics context is not active")

    def __enter__(self) -> Self:
        if self._entered or _CURRENT.get() is not None:
            raise RuntimeError("inference collectors cannot be reused or nested")
        if (os.getpid(), threading.get_ident()) != (self._pid, self._thread):
            raise RuntimeError("inference metrics belong to one process/thread")
        self._entered = True
        self._token = _CURRENT.set(self)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._assert_owner()
        self._body_failed = exc_type is not None
        self._closed = True
        if self._token is not None:
            _CURRENT.reset(self._token)
            self._token = None
        if self._stack:
            self._accounting_error = True
            raise RuntimeError("inference collector closed with unfinished spans")

    def span(self, label: str) -> AbstractContextManager[_Span | None]:
        self._assert_owner()
        if not isinstance(label, str) or not label:
            raise ValueError("metric span label must be nonempty text")
        return _Span(self, label) if self.enabled else nullcontext()

    def snapshot(self) -> dict[str, object]:
        if not self._closed:
            raise RuntimeError("collect a metrics snapshot after segment completion")
        stages = {key: dict(row) for key, row in sorted(self._stages.items())}
        own = {key: row["self_ns"] for key, row in stages.items()}
        total_label = "policy_total" if "policy_total" in stages else "actions_total"
        # These are exclusive categories. A scope-boundary parent includes its
        # integrity children; its own overhead is separate from those children.
        categories = (
            "integrity", "input_validation", "feature", "forward",
            "distribution_sampling", "decode", "scope_bind", "scope_final_verify",
        )
        timing = {f"{key}_ns": own.get(key, 0) for key in categories}
        timing["integrity_ns"] += own.get("file_read", 0) + own.get("hash_compute", 0)
        timing["policy_total_ns"] = stages.get(total_label, {}).get("inclusive_ns", 0)
        if "scope_bind" in stages or "scope_final_verify" in stages:
            # Boundary verification lies outside each per-action policy_total
            # span. Sum only top-level spans, including failed boundaries, so
            # nested integrity work is counted exactly once.
            timing["policy_total_ns"] = self._root_ns
            total_label = "segment_scopes_and_actions"
        named = {*categories, "file_read", "hash_compute"}
        timing["unattributed_ns"] = sum(value for key, value in own.items() if key not in named)
        return {
            "schema_version": "ppo-inference-metrics-v1", "phase": self.phase,
            "enabled": self.enabled, "profile": "detailed" if self.enabled else "disabled",
            "complete": not self._accounting_error, "body_failed": self._body_failed,
            "stages": stages, "file_hashes": {
                key: dict(row) for key, row in sorted(self._files.items())
            },
            "timing_ns": timing if self.enabled else {},
            "integrity_detail_ns": {
                "read_ns": own.get("file_read", 0),
                "hash_compute_ns": own.get("hash_compute", 0),
                "nonfile_validation_ns": own.get("integrity", 0),
            } if self.enabled else {},
            "policy_total_source": total_label if (
                total_label in stages or total_label == "segment_scopes_and_actions"
            ) else None,
            "definition": "stages.inclusive_ns overlaps; stages.self_ns and timing categories "
                          "are exclusive; file_hashes repeats read/hash stage information, "
                          "bytes are application reads, not physical disk I/O; "
                          "scoped policy_total includes all segment root spans; "
                          "scope timing categories are exclusive overhead, while "
                          "stages.scope_bind/scope_final_verify.inclusive_ns include verification",
        }


def metric_span(label: str) -> AbstractContextManager[_Span | None]:
    """Observe work only while an explicitly installed segment is active."""
    meter = _CURRENT.get()
    return nullcontext() if meter is None else meter.span(label)


def metric_function(label: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Observe an existing synchronous function without changing call order."""
    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def observed(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if _CURRENT.get() is None:
                return function(*args, **kwargs)
            with metric_span(label):
                return function(*args, **kwargs)
        return observed
    return decorate


def file_sha256(path: Path, *, role: str) -> str:
    """Read once, hash those same bytes once, and optionally observe both."""
    meter = _CURRENT.get()
    if meter is None or not meter.enabled:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not isinstance(role, str) or not role:
        raise ValueError("file hash role must be nonempty text")
    row = meter._files.setdefault(role, {
        "read_calls": 0, "read_failures": 0, "read_bytes": 0, "read_ns": 0,
        "hash_calls": 0, "hash_failures": 0, "hash_compute_ns": 0,
    })
    read_span = _Span(meter, "file_read")
    row["read_calls"] += 1
    try:
        with read_span:
            body = path.read_bytes()
    except BaseException:
        row["read_failures"] += 1
        raise
    finally:
        row["read_ns"] += read_span.elapsed_ns
    row["read_bytes"] += len(body)
    hash_span = _Span(meter, "hash_compute")
    row["hash_calls"] += 1
    try:
        with hash_span:
            return hashlib.sha256(body).hexdigest()
    except BaseException:
        row["hash_failures"] += 1
        raise
    finally:
        row["hash_compute_ns"] += hash_span.elapsed_ns


def inference_metrics_source_sha256() -> str:
    """Identity of the helper that observes, and performs, integrity file reads."""
    return file_sha256(Path(__file__), role="inference_metrics_source")


LOADED_INFERENCE_METRICS_SOURCE_SHA256 = inference_metrics_source_sha256()


__all__ = [
    "InferenceMetrics", "file_sha256", "metric_function", "metric_span",
    "inference_metrics_source_sha256", "LOADED_INFERENCE_METRICS_SOURCE_SHA256",
]
