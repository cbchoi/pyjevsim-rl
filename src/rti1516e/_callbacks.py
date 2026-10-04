"""Serialized callback delivery, independent of the asyncio transport loop."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .errors import CallNotAllowedFromWithinCallback, UnsupportedCallbackModel
from .events import (
    FederationNotRestored,
    FederationNotSaved,
    FederationRestoreBegun,
    FederationRestored,
    FederationSaved,
    InitiateFederateRestore,
    InitiateFederateSave,
    RequestFederationRestoreFailed,
    RequestFederationRestoreSucceeded,
)


class CallbackModel(StrEnum):
    HLA_EVOKED = "HLA_EVOKED"
    HLA_IMMEDIATE = "HLA_IMMEDIATE"


HLA_EVOKED = CallbackModel.HLA_EVOKED
HLA_IMMEDIATE = CallbackModel.HLA_IMMEDIATE


@dataclass(frozen=True)
class ServiceCallback:
    method: str
    args: tuple[Any, ...]


@dataclass
class _Callback:
    event: Any
    epoch: int
    done: bool = False
    recognized: bool = False
    error: BaseException | None = None
    error_observed: bool = False
    checkpoint_deferred: bool = False


class CallbackDispatcher:
    def __init__(self, invoke: Callable[[Any], bool], model: CallbackModel | str) -> None:
        try:
            self.model = CallbackModel(model)
        except (ValueError, TypeError) as exc:
            raise UnsupportedCallbackModel(str(model)) from exc
        self._invoke = invoke
        self._condition = threading.Condition()
        self._evoke_lock = threading.Lock()
        self._delivery_lock = threading.RLock()
        self._local = threading.local()
        self._queue: deque[_Callback] = deque()
        self._errors: deque[_Callback] = deque()
        self._worker: threading.Thread | None = None
        self._enabled = True
        self._closed = False
        self._fired = 0
        self._observed = 0
        self._epoch = 0
        self._admission_retry_blocked = False
        self._inflight = 0

    def snapshot_pending(self) -> list[Any]:
        with self._condition:
            if self._inflight:
                raise RuntimeError("cannot checkpoint an executing callback")
            return [callback.event for callback in self._queue]

    def retain_current_admission(self) -> None:
        """Retain a pre-body failure; retry only on the next explicit delivery call."""
        if getattr(self._local, "active", False):
            self._local.retry_admission = True

    def defer_current_checkpoint_admission(self) -> bool:
        """Keep a sync confirmation pending while checkpoint controls pass it."""
        if not getattr(self._local, "active", False):
            return False
        self._local.retry_checkpoint = True
        return True

    @staticmethod
    def _checkpoint_control(event: Any) -> bool:
        return (not getattr(event, "_callback_recovered", False)
                and not getattr(event, "_callback_outcome_only", False)
                and isinstance(event, (
                    InitiateFederateSave, FederationSaved, FederationNotSaved,
                    RequestFederationRestoreSucceeded, RequestFederationRestoreFailed,
                    FederationRestoreBegun, InitiateFederateRestore,
                    FederationRestored, FederationNotRestored,
                )))

    def _next_callback_index_locked(self) -> int | None:
        if not self._queue:
            return None
        if not self._queue[0].checkpoint_deferred:
            return 0
        # Ordinary callbacks stay FIFO. Only checkpoint protocol callbacks may
        # pass a deferred confirmation, otherwise its terminal cannot arrive.
        return next((index for index, callback in enumerate(self._queue)
                     if self._checkpoint_control(callback.event)), None)

    def _checkpoint_blocked_locked(self) -> bool:
        return bool(self._queue) and self._next_callback_index_locked() is None

    def _retry_checkpoint_locked(self) -> None:
        for callback in self._queue:
            callback.checkpoint_deferred = False

    def guard_reentrant(self) -> None:
        if getattr(self._local, "active", False):
            raise CallNotAllowedFromWithinCallback("service is not allowed within a callback")

    def submit(self, event: Any, *, wait: bool) -> bool:
        return self.submit_many([event], wait=wait)

    def submit_many(self, events: list[Any], *, wait: bool) -> bool:
        """Publish a restore terminal and local FIFO before a worker can enter."""
        if not events:
            return False
        with self._condition:
            if self._closed:
                return False
            callbacks = [_Callback(event, self._epoch) for event in events]
            self._queue.extend(callbacks)
            callback = callbacks[-1]
            self._start_worker_locked()
            self._condition.notify_all()
            if self.model == HLA_EVOKED or not wait or getattr(self._local, "active", False):
                return True
            while (not callback.done and self._enabled and not self._closed
                   and not self._admission_retry_blocked
                   and not self._checkpoint_blocked_locked()):
                self._condition.wait()
            for value in callbacks:
                if value.error is not None:
                    value.error_observed = True
                    raise value.error
            return callback.recognized if callback.done else True

    def _start_worker_locked(self) -> None:
        if (
            self.model == HLA_IMMEDIATE
            and self._enabled
            and not self._closed
            and not self._admission_retry_blocked
            and self._next_callback_index_locked() is not None
            and self._worker is None
        ):
            self._worker = threading.Thread(
                target=self._work, name="rti-ambassador-callbacks", daemon=True
            )
            self._worker.start()

    def _work(self) -> None:
        while True:
            with self._condition:
                index = self._next_callback_index_locked()
                if (self._closed or not self._enabled or index is None
                        or self._admission_retry_blocked):
                    self._worker = None
                    self._condition.notify_all()
                    return
                callback = self._queue[index]
                del self._queue[index]
                self._inflight += 1
            try:
                self._deliver(callback, asynchronous=True)
            finally:
                with self._condition:
                    self._inflight -= 1

    def _deliver(self, callback: _Callback, *, asynchronous: bool) -> bool:
        with self._delivery_lock:
            with self._condition:
                if self._closed or callback.epoch != self._epoch:
                    callback.done = True
                    self._condition.notify_all()
                    return False
                if not self._enabled:
                    self._queue.appendleft(callback)
                    return False
            return self._invoke_reserved(callback, asynchronous=asynchronous)

    def _invoke_reserved(self, callback: _Callback, *, asynchronous: bool) -> bool:
        self._local.active = True
        self._local.retry_admission = False
        self._local.retry_checkpoint = False
        try:
            callback.recognized = self._invoke(callback.event)
        except BaseException as exc:
            callback.error = exc
            callback.recognized = not self._local.retry_admission
        finally:
            self._local.active = False
            with self._condition:
                if (self._local.retry_admission and not self._closed
                        and callback.epoch == self._epoch):
                    self._queue.appendleft(_Callback(callback.event, callback.epoch))
                    self._admission_retry_blocked = asynchronous
                if (self._local.retry_checkpoint and not self._closed
                        and callback.epoch == self._epoch):
                    self._queue.appendleft(_Callback(
                        callback.event, callback.epoch, checkpoint_deferred=True
                    ))
                if (callback.recognized and callback.epoch == self._epoch
                        and self._checkpoint_control(callback.event)):
                    self._retry_checkpoint_locked()
                callback.done = True
                if callback.recognized:
                    self._fired += 1
                if callback.error is not None and asynchronous:
                    self._errors.append(callback)
                self._condition.notify_all()
        if callback.error is not None and not asynchronous:
            callback.error_observed = True
            raise callback.error
        return callback.recognized

    def _raise_pending_error_locked(self) -> None:
        while self._errors:
            callback = self._errors.popleft()
            if not callback.error_observed and callback.error is not None:
                callback.error_observed = True
                raise callback.error

    @staticmethod
    def _window(minimum: float, maximum: float | None) -> tuple[float, float]:
        minimum = float(minimum)
        maximum = minimum if maximum is None else float(maximum)
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum < 0
            or maximum < minimum
        ):
            raise ValueError("callback wait bounds must be finite, nonnegative, and ordered")
        return minimum, maximum

    def evoke(self, minimum: float, maximum: float | None, *, multiple: bool) -> bool:
        self.guard_reentrant()
        minimum, maximum = self._window(minimum, maximum)
        with self._evoke_lock:
            if self.model == HLA_IMMEDIATE:
                return self._observe_immediate(minimum, maximum)
            start = time.monotonic()
            delivered = False
            with self._condition:
                self._retry_checkpoint_locked()
            while True:
                with self._condition:
                    if self._closed or not self._enabled:
                        return delivered
                    elapsed = time.monotonic() - start
                    index = self._next_callback_index_locked()
                    if index is not None and (not delivered or elapsed < maximum):
                        callback = self._queue[index]
                        del self._queue[index]
                        self._inflight += 1
                    else:
                        wait_until = minimum if delivered else maximum
                        if elapsed >= wait_until:
                            return delivered
                        self._condition.wait(wait_until - elapsed)
                        continue
                try:
                    delivered = self._deliver(callback, asynchronous=False) or delivered
                finally:
                    with self._condition:
                        self._inflight -= 1
                if delivered and not multiple:
                    return True

    def _observe_immediate(self, minimum: float, maximum: float) -> bool:
        start = time.monotonic()
        with self._condition:
            self._raise_pending_error_locked()
            self._admission_retry_blocked = False
            self._retry_checkpoint_locked()
            self._start_worker_locked()
            while True:
                self._raise_pending_error_locked()
                elapsed = time.monotonic() - start
                if elapsed >= minimum and self._fired != self._observed:
                    self._observed = self._fired
                    return True
                if self._closed or elapsed >= maximum:
                    return False
                self._condition.wait(maximum - elapsed if elapsed >= minimum else minimum - elapsed)

    def enable(self, *, wait: bool) -> None:
        with self._condition:
            self._enabled = True
            self._admission_retry_blocked = False
            self._retry_checkpoint_locked()
            last = self._queue[-1] if self._queue else None
            self._start_worker_locked()
            self._condition.notify_all()
            if self.model == HLA_EVOKED or not wait or getattr(self._local, "active", False):
                return
            while (last is not None and not last.done and self._enabled and not self._closed
                   and not self._admission_retry_blocked
                   and not self._checkpoint_blocked_locked()):
                self._condition.wait()
            self._raise_pending_error_locked()

    def disable(self) -> None:
        with self._delivery_lock, self._condition:
            self._enabled = False
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._epoch += 1
            self._admission_retry_blocked = False
            for callback in self._queue:
                callback.done = True
            self._queue.clear()
            self._condition.notify_all()

    def close(self) -> None:
        self.guard_reentrant()
        with self._delivery_lock, self._condition:
            self._closed = True
            self.clear()
            worker = self._worker
        if worker is not None:
            worker.join()
