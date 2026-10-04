"""Stable invocation results; receipts authorize reports, never identify bodies."""

import hashlib
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompletedCallbackRecovery:
    """A stream reconciliation marker, not an application callback."""

    invocation_identity: bytes


class GenericCallbackState:
    def __init__(self, capacity: int = 8192):
        self._lock = threading.Lock()
        self._capacity = capacity
        self._seen: set[bytes] = set()
        self._pending: dict[bytes, tuple[bool, str]] = {}
        self._completed: deque[bytes] = deque()
        self._tickets: dict[bytes, bytes] = {}
        self._outcomes: dict[bytes, tuple[bool, str]] = {}
        self._running: set[bytes] = set()
        self._waiting: set[bytes] = set()

    def claim(self, receipt: bytes, identity: bytes | None = None) -> bool:
        with self._lock:
            identity = identity or self._tickets.get(receipt) or hashlib.sha256(receipt).digest()
            if len(identity) != 32:
                raise ValueError("callback invocation identity must contain 32 bytes")
            if receipt in self._tickets and self._tickets[receipt] != identity:
                raise ValueError("callback receipt identity changed")
            if identity in self._seen:
                return False
            if len(self._running | self._waiting) >= self._capacity:
                self._tickets.pop(receipt, None)
                raise RuntimeError("pending callback invocation capacity exceeded")
            self._seen.add(identity)
            self._running.add(identity)
            self._tickets[receipt] = identity
            return True

    def entry_failed(self, receipt: bytes) -> None:
        with self._lock:
            identity = self._tickets.get(receipt)
            if identity in self._running:
                self._running.discard(identity)
                self._seen.discard(identity)
                self._tickets.pop(receipt, None)

    def complete(
        self, receipt: bytes, success: bool, exception: str, identity: bytes | None = None
    ) -> None:
        with self._lock:
            identity = identity or self._tickets.get(receipt) or hashlib.sha256(receipt).digest()
            if len(identity) != 32:
                raise ValueError("callback invocation identity must contain 32 bytes")
            outcome = (success, exception)
            if identity in self._outcomes and self._outcomes[identity] != outcome:
                raise RuntimeError("conflicting completed callback outcome")
            if identity not in self._seen and len(self._running | self._waiting) >= self._capacity:
                raise RuntimeError("pending callback invocation capacity exceeded")
            self._tickets[receipt] = identity
            self._seen.add(identity)
            self._running.discard(identity)
            self._outcomes[identity] = outcome
            self._waiting.add(identity)
            self._pending[receipt] = (success, exception)

    def pending(self) -> list[tuple[bytes, bool, str]]:
        with self._lock:
            return [(receipt, *outcome) for receipt, outcome in self._pending.items()]

    def acknowledged(self, receipt: bytes) -> None:
        with self._lock:
            if self._pending.pop(receipt, None) is None:
                return
            self._finish(self._tickets[receipt])

    def _finish(self, identity: bytes) -> None:
        self._waiting.discard(identity)
        for receipt in list(self._pending):
            if self._tickets.get(receipt) == identity:
                self._pending.pop(receipt)
        if identity not in self._completed:
            self._completed.append(identity)
        while len(self._completed) > self._capacity:
            old = self._completed.popleft()
            self._seen.discard(old)
            self._outcomes.pop(old, None)
            self._tickets = {
                ticket: value for ticket, value in self._tickets.items() if value != old
            }

    def checkpoint_outcomes(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._running:
                raise RuntimeError("cannot checkpoint unresolved callback invocation outcomes")
            return [
                {"identity": identity.hex(), "success": self._outcomes[identity][0],
                 "exception": self._outcomes[identity][1]}
                for identity in sorted(self._waiting)
            ]

    def restore_outcomes(self, rows: list[dict[str, Any]]) -> None:
        from ._runtime_checkpoint import validate_outcomes

        validate_outcomes(rows)
        if len(rows) > self._capacity:
            raise ValueError("pending callback invocation capacity exceeded")
        with self._lock:
            if self._seen:
                raise RuntimeError("cannot overwrite callback invocation state")
            for row in rows:
                identity = bytes.fromhex(row["identity"])
                self._seen.add(identity)
                self._waiting.add(identity)
                self._outcomes[identity] = (row["success"], row["exception"])

    def recover(self, event: Any) -> tuple[bytes, bool, str] | None:
        identity = getattr(event, "_callback_invocation_identity", b"")
        receipt = getattr(event, "_callback_receipt", b"")
        completed = getattr(event, "_callback_completed", False)
        if len(identity) != 32 or (completed and receipt) or (not completed and len(receipt) != 32):
            raise ValueError("invalid callback recovery carrier")
        with self._lock:
            if completed:
                if identity in self._waiting:
                    self._finish(identity)
                return None
            if identity not in self._outcomes:
                raise RuntimeError("callback recovery requires a matching completed outcome")
            if identity not in self._waiting:
                return None
            for ticket in list(self._pending):
                if self._tickets.get(ticket) == identity and ticket != receipt:
                    self._pending.pop(ticket)
                    self._tickets.pop(ticket, None)
            self._tickets[receipt] = identity
            self._pending[receipt] = self._outcomes[identity]
            return (receipt, *self._outcomes[identity])

    def has_outcome(self, receipt: bytes) -> bool:
        with self._lock:
            return self._tickets.get(receipt) in self._outcomes


def validate_event_scope(owner: Any, event: Any) -> None:
    fed = getattr(owner, "_federate", None) or owner
    transport = getattr(fed, "_transport", None)
    serial = getattr(event, "_restore_serial", None)
    if serial is not None and serial != getattr(transport, "_restore_serial", 0):
        raise RuntimeError("stale callback from an earlier restore")


def callback_state(owner: Any) -> GenericCallbackState:
    fed = getattr(owner, "_federate", None) or owner
    transport = getattr(fed, "_transport", None)
    handle = getattr(fed, "handle", None)
    if transport is not None and handle is not None:
        states: dict[int, GenericCallbackState] | None = getattr(
            transport, "_generic_callbacks_by_federate", None
        )
        if states is None:
            states = transport._generic_callbacks_by_federate = {}
        return states.setdefault(handle, GenericCallbackState())
    state = getattr(owner, "_generic_callback_invocations", None)
    if state is None:
        state = GenericCallbackState()
        owner._generic_callback_invocations = state
    return state
