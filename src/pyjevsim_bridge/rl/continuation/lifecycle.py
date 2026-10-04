"""Candidate-owned cleanup, with no PID lookup or process termination authority."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReleaseResult:
    success: bool
    released: tuple[str, ...]
    failed_resources: tuple[str, ...]
    errors: tuple[str, ...]


class CleanupLedger:
    """Release only explicitly registered resources, newest first.

    Failed callbacks remain owned and may be tried by an explicit later close.
    A failure is never discarded merely because another resource closed. This
    object does not discover or kill processes and does not imply a resource cap.
    """

    def __init__(self) -> None:
        self._owned: list[tuple[str, Callable[[], object]]] = []
        self._released: list[str] = []
        self._active = False
        self._lock = threading.RLock()

    def add(self, resource_id: str, release: Callable[[], object]) -> None:
        if not isinstance(resource_id, str) or not resource_id or not callable(release):
            raise ValueError("cleanup requires a resource identity and callable")
        with self._lock:
            if self._active or resource_id in self._released or any(
                name == resource_id for name, _ in self._owned
            ):
                raise ValueError("duplicate resource or registration during cleanup")
            self._owned.append((resource_id, release))

    @property
    def pending(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(name for name, _ in self._owned)

    def close(self) -> ReleaseResult:
        with self._lock:
            if self._active:
                return ReleaseResult(False, tuple(self._released), self.pending,
                                     ("cleanup re-entry is not completion",))
            self._active = True
            failed: list[tuple[str, Callable[[], object]]] = []
            errors: list[str] = []
            try:
                for resource_id, release in reversed(self._owned):
                    try:
                        receipt = release()
                        if receipt is not None and getattr(receipt, "success", None) is not True:
                            raise RuntimeError("resource did not confirm successful cleanup")
                    except BaseException as exc:
                        failed.append((resource_id, release))
                        errors.append(f"{resource_id}: {type(exc).__name__}: {exc}"[:1024])
                    else:
                        self._released.append(resource_id)
                self._owned = list(reversed(failed))
                return ReleaseResult(not failed, tuple(self._released),
                                     tuple(name for name, _ in failed), tuple(errors))
            finally:
                self._active = False
