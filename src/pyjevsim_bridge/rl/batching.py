"""Deterministic adaptive transition batching for federation rollouts."""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pyjevsim_bridge.rl.federation import (
    MAX_PAYLOAD_BYTES,
    EnvelopeValidationError,
    canonical_json,
    validate_envelope,
)


class BatchFlushReason(enum.StrEnum):
    """The first policy boundary that closed a physical transition batch."""

    MAX_RECORDS = "max-records"
    MAX_BYTES = "max-bytes"
    MAX_LOGICAL_TIME_SPAN = "max-logical-time-span"
    MAX_WAIT = "max-wait"
    IDENTITY_CHANGE = "identity-change"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class AdaptiveBatchPolicy:
    """Closed limits for one physical transition interaction."""

    max_records: int = 4
    max_payload_bytes: int = MAX_PAYLOAD_BYTES
    max_logical_time_span: float = 3.0
    max_wait_ns: int = 1_000_000

    def __post_init__(self) -> None:
        for field in ("max_records", "max_payload_bytes", "max_wait_ns"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.max_payload_bytes > MAX_PAYLOAD_BYTES:
            raise ValueError(f"max_payload_bytes must not exceed {MAX_PAYLOAD_BYTES}")
        span = self.max_logical_time_span
        if isinstance(span, bool) or not isinstance(span, (int, float)):
            raise ValueError("max_logical_time_span must be numeric")
        if not math.isfinite(float(span)) or float(span) < 0:
            raise ValueError("max_logical_time_span must be finite and non-negative")


@dataclass(frozen=True)
class PlannedTransitionBatch:
    """One validated physical interaction selected by the batching policy."""

    records: tuple[dict[str, Any], ...]
    flush_reason: BatchFlushReason
    payload_bytes: int
    logical_time_span: float
    waited_ns: int


@dataclass(frozen=True)
class BatchingTelemetry:
    """Immutable aggregate evidence for a batcher lifetime."""

    records: int
    interactions: int
    payload_bytes: int
    maximum_batch_records: int
    flush_counts: tuple[tuple[str, int], ...]


_IDENTITY_FIELDS = (
    "run_id",
    "generation",
    "worker_id",
    "episode_id",
    "policy_version",
)


class AdaptiveTransitionBatcher:
    """Incrementally group transitions without hiding a semantic boundary."""

    def __init__(self, policy: AdaptiveBatchPolicy, *, generation: int) -> None:
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        self.policy = policy
        self.generation = generation
        self._pending: list[dict[str, Any]] = []
        self._first_arrival_ns: int | None = None
        self._last_arrival_ns: int | None = None
        self._last_identity: tuple[object, ...] | None = None
        self._last_step_id: int | None = None
        self._last_logical_time: float | None = None
        self._records = 0
        self._interactions = 0
        self._payload_bytes = 0
        self._maximum_batch_records = 0
        self._flush_counts: dict[BatchFlushReason, int] = {}

    @staticmethod
    def _identity(envelope: Mapping[str, Any]) -> tuple[object, ...]:
        return tuple(envelope[field] for field in _IDENTITY_FIELDS)

    @staticmethod
    def _payload_size(records: Sequence[Mapping[str, Any]]) -> int:
        return len(canonical_json(list(records)))

    @staticmethod
    def _logical_time_span(records: Sequence[Mapping[str, Any]]) -> float:
        times = [float(item["logical_time"]) for item in records]
        return max(times) - min(times)

    def _normalize_arrival(self, now_ns: int) -> None:
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("now_ns must be a non-negative integer")
        if self._last_arrival_ns is not None and now_ns < self._last_arrival_ns:
            raise ValueError("now_ns must be monotonic")
        self._last_arrival_ns = now_ns

    def _emit(self, reason: BatchFlushReason, *, now_ns: int) -> PlannedTransitionBatch:
        if not self._pending or self._first_arrival_ns is None:
            raise RuntimeError("cannot emit an empty transition batch")
        records = tuple(self._pending)
        payload_bytes = self._payload_size(records)
        result = PlannedTransitionBatch(
            records=records,
            flush_reason=reason,
            payload_bytes=payload_bytes,
            logical_time_span=self._logical_time_span(records),
            waited_ns=now_ns - self._first_arrival_ns,
        )
        self._records += len(records)
        self._interactions += 1
        self._payload_bytes += payload_bytes
        self._maximum_batch_records = max(self._maximum_batch_records, len(records))
        self._flush_counts[reason] = self._flush_counts.get(reason, 0) + 1
        self._pending.clear()
        self._first_arrival_ns = None
        return result

    def _ensure_single_record_fits(self, envelope: Mapping[str, Any]) -> None:
        if self._payload_size([envelope]) > self.policy.max_payload_bytes:
            raise EnvelopeValidationError("single transition exceeds adaptive max_payload_bytes")

    def add(self, value: Any, *, now_ns: int) -> tuple[PlannedTransitionBatch, ...]:
        """Admit one record and return every batch closed by this admission."""

        self._normalize_arrival(now_ns)
        envelope = validate_envelope(value, generation=self.generation)
        self._ensure_single_record_fits(envelope)
        current_identity = self._identity(envelope)
        emitted: list[PlannedTransitionBatch] = []
        if self._pending and self._first_arrival_ns is not None:
            if now_ns - self._first_arrival_ns >= self.policy.max_wait_ns:
                emitted.append(self._emit(BatchFlushReason.MAX_WAIT, now_ns=now_ns))
            elif current_identity != self._identity(self._pending[0]):
                emitted.append(self._emit(BatchFlushReason.IDENTITY_CHANGE, now_ns=now_ns))
        if current_identity == self._last_identity:
            if self._last_step_id is not None and int(envelope["step_id"]) <= self._last_step_id:
                raise EnvelopeValidationError("adaptive batch step IDs must be strictly increasing")
            if (
                self._last_logical_time is not None
                and float(envelope["logical_time"]) < self._last_logical_time
            ):
                raise EnvelopeValidationError("adaptive batch logical times must be monotonic")
        else:
            self._last_identity = current_identity
            self._last_step_id = None
            self._last_logical_time = None
        if self._pending:
            candidate = [*self._pending, envelope]
            if self._payload_size(candidate) > self.policy.max_payload_bytes:
                emitted.append(self._emit(BatchFlushReason.MAX_BYTES, now_ns=now_ns))
            elif self._logical_time_span(candidate) > self.policy.max_logical_time_span:
                emitted.append(
                    self._emit(
                        BatchFlushReason.MAX_LOGICAL_TIME_SPAN,
                        now_ns=now_ns,
                    )
                )
        if not self._pending:
            self._first_arrival_ns = now_ns
        self._pending.append(envelope)
        self._last_step_id = int(envelope["step_id"])
        self._last_logical_time = float(envelope["logical_time"])
        if len(self._pending) == self.policy.max_records:
            emitted.append(self._emit(BatchFlushReason.MAX_RECORDS, now_ns=now_ns))
        return tuple(emitted)

    def flush_due(self, *, now_ns: int) -> tuple[PlannedTransitionBatch, ...]:
        """Flush a pending batch only when its wait budget has expired."""

        self._normalize_arrival(now_ns)
        if (
            self._pending
            and self._first_arrival_ns is not None
            and now_ns - self._first_arrival_ns >= self.policy.max_wait_ns
        ):
            return (self._emit(BatchFlushReason.MAX_WAIT, now_ns=now_ns),)
        return ()

    def flush(self, *, now_ns: int) -> tuple[PlannedTransitionBatch, ...]:
        """Explicitly close the pending batch at a safe caller boundary."""

        self._normalize_arrival(now_ns)
        if not self._pending:
            return ()
        return (self._emit(BatchFlushReason.EXPLICIT, now_ns=now_ns),)

    @property
    def telemetry(self) -> BatchingTelemetry:
        return BatchingTelemetry(
            records=self._records,
            interactions=self._interactions,
            payload_bytes=self._payload_bytes,
            maximum_batch_records=self._maximum_batch_records,
            flush_counts=tuple(
                (reason.value, self._flush_counts[reason])
                for reason in BatchFlushReason
                if reason in self._flush_counts
            ),
        )
