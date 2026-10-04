"""Persistent spawn-process runtime foundation for local rollout workers."""

from __future__ import annotations

import asyncio
import math
import multiprocessing
import pickle
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from multiprocessing.connection import wait as connection_wait
from typing import Any, Literal

from ._local_process_worker import process_worker_main
from .local_wire import WIRE_VERSION, LocalProcessWireError, decode_frame, encode_frame

# Large synchronous pipe writes can block before reply polling begins. Keep the
# optimized fan-out for small envelopes; larger batches retain the independent
# sender path. This is a dispatch policy, not a hard OS send-timeout guarantee.
_BATCH_SEND_BYTES = 16 * 1024


@dataclass(frozen=True, slots=True)
class ProcessBackendOptions:
    """Bounded startup, frame, and shutdown policy for spawned workers."""

    startup_timeout: float = 10.0
    operation_timeout: float = 30.0
    shutdown_timeout: float = 2.0
    max_frame_bytes: int = 8 * 1024 * 1024
    # Batched dispatch is an explicit experimental option: actual Windows
    # comparisons did not establish a benefit over the compatibility path.
    dispatch_mode: Literal["individual", "batched"] = "individual"

    def __post_init__(self) -> None:
        for name in ("startup_timeout", "operation_timeout", "shutdown_timeout"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if (
            isinstance(self.max_frame_bytes, bool)
            or not isinstance(self.max_frame_bytes, int)
            or self.max_frame_bytes <= 0
        ):
            raise ValueError("max_frame_bytes must be positive")
        if self.dispatch_mode not in ("individual", "batched"):
            raise ValueError("dispatch_mode must be 'individual' or 'batched'")


@dataclass(frozen=True, slots=True)
class ProcessCleanupReceipt:
    worker_id: str
    incarnation: int
    pid: int | None
    action: str
    exitcode: int | None
    error: str | None = None


class LocalProcessWorkerError(RuntimeError):
    """Typed worker, PID, phase, and remote-failure attribution."""

    def __init__(
        self,
        worker_id: str,
        phase: str,
        message: str,
        *,
        pid: int | None = None,
        exitcode: int | None = None,
        request_id: int | None = None,
        remote_type: str | None = None,
        remote_module: str | None = None,
        remote_traceback: str | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.phase = phase
        self.pid = pid
        self.exitcode = exitcode
        self.request_id = request_id
        self.remote_type = remote_type
        self.remote_module = remote_module
        self.remote_traceback = remote_traceback
        super().__init__(
            f"process worker {worker_id} {phase} failed"
            f" (pid={pid}, exitcode={exitcode}, request_id={request_id}): {message}"
        )


@dataclass(slots=True)
class _Worker:
    worker_id: str
    incarnation: int
    process: Any
    connection: Any
    pid: int
    request_id: int = 0
    closed: bool = False
    tainted: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class ProcessWorkerSupervisor:
    """Own one persistent spawn process per importable environment factory."""

    def __init__(
        self,
        factories: Mapping[str, Callable[[], object]],
        *,
        options: ProcessBackendOptions | None = None,
    ) -> None:
        if not isinstance(factories, Mapping) or not factories:
            raise ValueError("factories must be a non-empty mapping")
        self.options = options or ProcessBackendOptions()
        worker_ids = tuple(sorted(factories))
        if any(not isinstance(item, str) or not item for item in worker_ids):
            raise ValueError("worker IDs must be non-empty strings")
        serialized: dict[str, bytes] = {}
        for worker_id in worker_ids:
            factory = factories[worker_id]
            if not callable(factory):
                raise LocalProcessWorkerError(
                    worker_id,
                    "factory-preflight",
                    "factory provider must be callable",
                    remote_type="TypeError",
                )
            try:
                value = pickle.dumps(factory, protocol=pickle.HIGHEST_PROTOCOL)
            except BaseException as exc:
                raise LocalProcessWorkerError(
                    worker_id, "factory-preflight", str(exc), remote_type=type(exc).__name__
                ) from exc
            if len(value) > self.options.max_frame_bytes:
                raise LocalProcessWorkerError(
                    worker_id,
                    "factory-preflight",
                    f"serialized factory size {len(value)} exceeds limit "
                    f"{self.options.max_frame_bytes}",
                )
            serialized[worker_id] = value

        self._factory_payloads = serialized
        self._workers: dict[str, _Worker] = {}
        self._cleanup_receipts: dict[str, ProcessCleanupReceipt] = {}
        self._cleanup_history: list[ProcessCleanupReceipt] = []
        self._context = multiprocessing.get_context("spawn")
        try:
            for worker_id in worker_ids:
                self._spawn_worker_process(worker_id, 0)
            # All children are running before we wait for any HELLO/READY
            # pair.  Python import and factory materialization therefore
            # overlap instead of growing linearly with the worker count.
            for worker_id in worker_ids:
                provisional = self._workers[worker_id]
                self._workers[worker_id] = self._start_worker(
                    worker_id,
                    provisional.incarnation,
                    provisional.process,
                    provisional.connection,
                )
        except BaseException:
            self._cleanup_sync()
            raise

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._workers))

    @property
    def pids(self) -> Mapping[str, int]:
        return {worker_id: worker.pid for worker_id, worker in self._workers.items()}

    @property
    def incarnations(self) -> Mapping[str, int]:
        return {worker_id: worker.incarnation for worker_id, worker in self._workers.items()}

    @property
    def cleanup_receipts(self) -> Mapping[str, ProcessCleanupReceipt]:
        """Return the latest terminal process receipt for each worker."""

        return dict(self._cleanup_receipts)

    @property
    def cleanup_history(self) -> tuple[ProcessCleanupReceipt, ...]:
        """Return append-only terminal receipts for every owned incarnation."""

        return tuple(self._cleanup_history)

    def is_alive(self, worker_id: str) -> bool:
        return bool(self._worker(worker_id).process.is_alive())

    def _spawn_worker(self, worker_id: str, incarnation: int) -> None:
        self._spawn_worker_process(worker_id, incarnation)
        provisional = self._workers[worker_id]
        try:
            worker = self._start_worker(
                worker_id,
                incarnation,
                provisional.process,
                provisional.connection,
            )
        except BaseException:
            self._cleanup_worker_sync(worker_id)
            raise
        self._workers[worker_id] = worker

    def _spawn_worker_process(self, worker_id: str, incarnation: int) -> None:
        """Start one provisional child without waiting for materialization."""

        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=process_worker_main,
            args=(
                child,
                worker_id,
                incarnation,
                self._factory_payloads[worker_id],
                self.options.max_frame_bytes,
            ),
            name=f"pyjevsim-rl-{worker_id}",
        )
        try:
            process.start()
        except BaseException as exc:
            parent.close()
            child.close()
            raise LocalProcessWorkerError(
                worker_id,
                "spawn",
                str(exc) or type(exc).__name__,
                remote_type=type(exc).__name__,
            ) from exc
        child.close()
        provisional_pid = process.pid
        if provisional_pid is None:
            parent.close()
            process.terminate()
            process.join(self.options.shutdown_timeout)
            raise LocalProcessWorkerError(
                worker_id, "spawn", "spawned worker did not receive a PID"
            )
        self._workers[worker_id] = _Worker(worker_id, incarnation, process, parent, provisional_pid)

    def _start_worker(
        self, worker_id: str, incarnation: int, process: Any, connection: Any
    ) -> _Worker:
        hello = self._receive_startup(worker_id, incarnation, process, connection, "hello")
        ready = self._receive_startup(worker_id, incarnation, process, connection, "ready")
        hello_identity = (hello["worker_id"], hello["incarnation"], hello["pid"])
        ready_identity = (ready["worker_id"], ready["incarnation"], ready["pid"])
        if hello_identity != ready_identity:
            raise LocalProcessWorkerError(
                worker_id,
                "startup",
                f"HELLO/READY identity mismatch: {hello_identity!r} != {ready_identity!r}",
                pid=process.pid,
            )
        if ready["status"] == "error":
            self._raise_remote(worker_id, process, ready, None)
        return _Worker(worker_id, incarnation, process, connection, int(hello["pid"]))

    def _receive_startup(
        self,
        worker_id: str,
        incarnation: int,
        process: Any,
        connection: Any,
        expected_kind: str,
    ) -> dict[str, Any]:
        if not connection.poll(self.options.startup_timeout):
            raise LocalProcessWorkerError(
                worker_id,
                "startup",
                f"timed out waiting for {expected_kind}",
                pid=process.pid,
                exitcode=process.exitcode,
            )
        try:
            frame: dict[str, Any] = decode_frame(
                connection.recv_bytes(maxlength=self.options.max_frame_bytes),
                max_bytes=self.options.max_frame_bytes,
            )
        except (EOFError, OSError, LocalProcessWireError) as exc:
            raise LocalProcessWorkerError(
                worker_id,
                "startup",
                str(exc),
                pid=process.pid,
                exitcode=process.exitcode,
            ) from exc
        expected = (expected_kind, worker_id, incarnation, process.pid)
        actual = (
            frame["kind"],
            frame["worker_id"],
            frame["incarnation"],
            frame["pid"],
        )
        if actual != expected:
            raise LocalProcessWorkerError(
                worker_id,
                "startup",
                f"startup identity differs: expected={expected!r}, actual={actual!r}",
                pid=process.pid,
            )
        return frame

    async def reset(self, worker_id: str, *, seed: int | None = None) -> object:
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None")
        return await asyncio.to_thread(self._rpc, worker_id, "reset", {"seed": seed})

    async def step(self, worker_id: str, action: object) -> object:
        action_payload = self.preflight_action(worker_id, action)
        return await self.step_preflighted(worker_id, action_payload)

    def preflight_action(self, worker_id: str, action: object) -> bytes:
        """Serialize one action without calling or mutating its worker."""

        worker = self._worker(worker_id)
        try:
            action_payload = pickle.dumps(action, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException as exc:
            raise LocalProcessWorkerError(
                worker_id, "action-preflight", str(exc), remote_type=type(exc).__name__
            ) from exc
        if len(action_payload) > self.options.max_frame_bytes:
            raise LocalProcessWorkerError(
                worker_id,
                "action-preflight",
                f"serialized action size {len(action_payload)} exceeds limit "
                f"{self.options.max_frame_bytes}",
            )
        request = {
            "version": WIRE_VERSION,
            "kind": "request",
            "request_id": worker.request_id + 1,
            "worker_id": worker_id,
            "incarnation": worker.incarnation,
            "operation": "step",
            "payload": {"action": action_payload},
        }
        try:
            encode_frame(request, max_bytes=self.options.max_frame_bytes)
        except LocalProcessWireError as exc:
            raise LocalProcessWorkerError(
                worker_id,
                "action-preflight",
                str(exc),
                pid=worker.pid,
                request_id=worker.request_id + 1,
            ) from exc
        return action_payload

    async def step_preflighted(self, worker_id: str, action_payload: bytes) -> object:
        """Send an action already validated by :meth:`preflight_action`."""

        return await asyncio.to_thread(
            self._rpc,
            worker_id,
            "step",
            {"action": action_payload},
            "action-preflight",
        )

    async def reset_batch(self, seeds: Mapping[str, int | None]) -> list[object | BaseException]:
        """Reset the complete cohort; the owning pool shields cancellation."""

        copied = self._batch_values(seeds)
        if any(
            seed is not None and (type(seed) is not int or seed < 0) for seed in copied.values()
        ):
            raise ValueError("seeds must be non-negative integers or None")
        return await self._dispatch_batch(
            "reset", {worker_id: {"seed": seed} for worker_id, seed in copied.items()}
        )

    async def step_batch_preflighted(
        self, action_payloads: Mapping[str, bytes]
    ) -> list[object | BaseException]:
        """Dispatch already serialized actions after whole-cohort envelope admission."""

        copied = self._batch_values(action_payloads)
        if any(not isinstance(payload, bytes) for payload in copied.values()):
            raise TypeError("preflighted actions must be bytes")
        return await self._dispatch_batch(
            "step", {worker_id: {"action": payload} for worker_id, payload in copied.items()}
        )

    def _batch_values(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(values, Mapping):
            raise TypeError("batch arguments must be a mapping keyed by every worker ID")
        copied = dict(values)
        if set(copied) != set(self.worker_ids):
            raise ValueError("batch arguments must contain exactly every worker ID")
        return {worker_id: copied[worker_id] for worker_id in self.worker_ids}

    async def _dispatch_batch(
        self, operation: str, payloads: Mapping[str, dict[str, object]]
    ) -> list[object | BaseException]:
        results = await asyncio.to_thread(self._rpc_batch, operation, payloads)
        if results is not None:
            return results
        # No request ID was consumed and nothing was sent by _rpc_batch. Frame
        # preparation is repeated under each lock on this compatibility path.
        return list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        self._rpc,
                        worker_id,
                        operation,
                        payloads[worker_id],
                        "action-preflight" if operation == "step" else "request-preflight",
                    )
                    for worker_id in self.worker_ids
                ),
                return_exceptions=True,
            )
        )

    def _rpc_batch(
        self, operation: str, payloads: Mapping[str, dict[str, object]]
    ) -> list[object | BaseException] | None:
        """Prepare-all, send-all, readiness-drain; None selects large-frame fallback.

        One request per worker is outstanding while all canonical locks are held.
        Deadlines begin immediately before send_bytes, not when that peer is
        visited during collection. Synchronous OS send/recv calls themselves are
        not forcibly interruptible; the individual path has the same limitation.
        """

        workers = [self._worker(worker_id) for worker_id in self.worker_ids]
        prepared: dict[str, tuple[int, bytes]] = {}
        results: dict[str, object | BaseException] = {}
        phase = "action-preflight" if operation == "step" else "request-preflight"
        with ExitStack() as stack:
            for worker in workers:
                stack.enter_context(worker.lock)
            for worker in workers:
                request_id = worker.request_id + 1
                try:
                    if worker.closed or worker.tainted or not worker.process.is_alive():
                        raise LocalProcessWorkerError(
                            worker.worker_id,
                            operation,
                            "worker is closed, tainted or not alive",
                            pid=worker.pid,
                            exitcode=worker.process.exitcode,
                            request_id=request_id,
                        )
                    request = {
                        "version": WIRE_VERSION,
                        "kind": "request",
                        "request_id": request_id,
                        "worker_id": worker.worker_id,
                        "incarnation": worker.incarnation,
                        "operation": operation,
                        "payload": payloads[worker.worker_id],
                    }
                    prepared[worker.worker_id] = (
                        request_id,
                        encode_frame(request, max_bytes=self.options.max_frame_bytes),
                    )
                except BaseException as exc:
                    results[worker.worker_id] = (
                        exc
                        if isinstance(exc, LocalProcessWorkerError)
                        else LocalProcessWorkerError(
                            worker.worker_id,
                            phase,
                            str(exc),
                            pid=worker.pid,
                            request_id=request_id,
                            remote_type=type(exc).__name__,
                        )
                    )
            if results:
                # Failed preflight must not turn valid peers into apparently
                # successful None returns, or advance any member of the cohort.
                return [
                    results.get(worker.worker_id)
                    or LocalProcessWorkerError(
                        worker.worker_id,
                        phase,
                        "batch not sent because peer preflight failed",
                        pid=worker.pid,
                        request_id=worker.request_id + 1,
                    )
                    for worker in workers
                ]
            if any(len(frame) > _BATCH_SEND_BYTES for _, frame in prepared.values()):
                return None

            pending: dict[str, tuple[_Worker, int, float]] = {}
            for worker in workers:
                request_id, encoded = prepared[worker.worker_id]
                worker.request_id = request_id
                deadline = time.monotonic() + self.options.operation_timeout
                try:
                    worker.connection.send_bytes(encoded)
                except BaseException as exc:
                    results[worker.worker_id] = self._batch_transport_error(
                        worker, operation, request_id, exc
                    )
                else:
                    pending[worker.worker_id] = (worker, request_id, deadline)

            while pending:
                remaining = max(0.0, min(item[2] for item in pending.values()) - time.monotonic())
                try:
                    ready = connection_wait(
                        [item[0].connection for item in pending.values()], timeout=remaining
                    )
                except BaseException as exc:
                    for worker_id, (worker, request_id, _) in pending.items():
                        results[worker_id] = self._batch_transport_error(
                            worker, operation, request_id, exc
                        )
                    pending.clear()
                    break
                observed_at = time.monotonic()
                for worker_id, (worker, request_id, deadline) in tuple(pending.items()):
                    if worker.connection in ready and observed_at <= deadline:
                        try:
                            reply = decode_frame(
                                worker.connection.recv_bytes(
                                    maxlength=self.options.max_frame_bytes
                                ),
                                max_bytes=self.options.max_frame_bytes,
                            )
                        except BaseException as exc:
                            results[worker_id] = self._batch_transport_error(
                                worker, operation, request_id, exc
                            )
                        else:
                            try:
                                results[worker_id] = self._reply_result(worker, request_id, reply)
                            except BaseException as exc:
                                results[worker_id] = exc
                        del pending[worker_id]
                    elif observed_at >= deadline:
                        results[worker_id] = self._batch_transport_error(
                            worker, operation, request_id, TimeoutError("operation timed out")
                        )
                        del pending[worker_id]
            # A broken pipe/EOF can become visible just before the OS publishes
            # the crashed child's exit status. Refresh attribution only AFTER
            # every peer has been drained or timed out, using one shared grace
            # budget rather than a separate delay per failed worker.
            exit_deadline = time.monotonic() + 0.05
            for worker in workers:
                result = results[worker.worker_id]
                if (
                    isinstance(result, LocalProcessWorkerError)
                    and result.exitcode is None
                    and result.request_id is not None
                    and isinstance(result.__cause__, (EOFError, OSError))
                ):
                    try:
                        worker.process.join(timeout=max(0.0, exit_deadline - time.monotonic()))
                    except BaseException as exc:
                        result.add_note(
                            f"post-drain exit status refresh failed: {type(exc).__name__}"
                        )
                    else:
                        results[worker.worker_id] = self._batch_transport_error(
                            worker, operation, result.request_id, result.__cause__
                        )
            return [results[worker.worker_id] for worker in workers]

    @staticmethod
    def _batch_transport_error(
        worker: _Worker, operation: str, request_id: int, error: BaseException
    ) -> LocalProcessWorkerError:
        # Do not join a failed peer here: even a short join would hold up draining
        # other already-ready replies and bias their individual deadlines.
        worker.tainted = True
        result = LocalProcessWorkerError(
            worker.worker_id,
            operation,
            str(error) or type(error).__name__,
            pid=worker.pid,
            exitcode=worker.process.exitcode,
            request_id=request_id,
        )
        result.__cause__ = error
        return result

    async def replace_worker(self, worker_id: str) -> None:
        """Create a fresh environment incarnation, preserving a live PID."""

        worker = self._worker(worker_id)
        next_incarnation = worker.incarnation + 1
        if not worker.closed and not worker.tainted and worker.process.is_alive():
            try:
                await asyncio.to_thread(self._rpc, worker_id, "replace", {})
            except BaseException:
                await asyncio.to_thread(self._cleanup_worker_sync, worker_id)
                # Consume the failed replacement incarnation only after the
                # terminal receipt records the actually owned old runtime.
                worker.incarnation = next_incarnation
                raise
            worker.incarnation = next_incarnation
            return
        await asyncio.to_thread(self._cleanup_worker_sync, worker_id)
        await asyncio.to_thread(self._spawn_worker, worker_id, next_incarnation)

    async def close_worker(self, worker_id: str) -> None:
        worker = self._worker(worker_id)
        if worker.closed:
            return
        if worker.tainted:
            await asyncio.to_thread(self._cleanup_worker_sync, worker_id)
            return
        if not worker.process.is_alive():
            await asyncio.to_thread(worker.process.join, self.options.shutdown_timeout)
            worker.connection.close()
            worker.closed = True
            self._record_cleanup(
                ProcessCleanupReceipt(
                    worker_id,
                    worker.incarnation,
                    worker.pid,
                    "already-exited",
                    worker.process.exitcode,
                )
            )
            return
        try:
            await asyncio.to_thread(self._rpc, worker_id, "close", {})
        except BaseException:
            await asyncio.to_thread(self._cleanup_worker_sync, worker_id)
            raise
        await asyncio.to_thread(worker.process.join, self.options.shutdown_timeout)
        if worker.process.is_alive():
            error = LocalProcessWorkerError(
                worker_id, "close", "worker did not exit after close reply", pid=worker.pid
            )
            await asyncio.to_thread(self._cleanup_worker_sync, worker_id)
            raise error
        worker.connection.close()
        worker.closed = True
        self._record_cleanup(
            ProcessCleanupReceipt(
                worker_id,
                worker.incarnation,
                worker.pid,
                "closed",
                worker.process.exitcode,
            )
        )

    async def close(self) -> None:
        results = await asyncio.gather(
            *(self.close_worker(worker_id) for worker_id in self.worker_ids),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        self._cleanup_sync()
        if errors:
            raise errors[0]

    def _rpc(
        self,
        worker_id: str,
        operation: str,
        payload: dict[str, object],
        preflight_phase: str | None = None,
    ) -> object:
        worker = self._worker(worker_id)
        with worker.lock:
            if worker.closed:
                raise LocalProcessWorkerError(
                    worker_id, operation, "worker is closed", pid=worker.pid
                )
            worker.request_id += 1
            request_id = worker.request_id
            request = {
                "version": WIRE_VERSION,
                "kind": "request",
                "request_id": request_id,
                "worker_id": worker_id,
                "incarnation": worker.incarnation,
                "operation": operation,
                "payload": payload,
            }
            try:
                encoded_request = encode_frame(request, max_bytes=self.options.max_frame_bytes)
            except LocalProcessWireError as exc:
                raise LocalProcessWorkerError(
                    worker_id,
                    preflight_phase or "request-preflight",
                    str(exc),
                    pid=worker.pid,
                    request_id=request_id,
                ) from exc
            try:
                worker.connection.send_bytes(encoded_request)
                if not worker.connection.poll(self.options.operation_timeout):
                    raise TimeoutError("operation timed out")
                reply = decode_frame(
                    worker.connection.recv_bytes(maxlength=self.options.max_frame_bytes),
                    max_bytes=self.options.max_frame_bytes,
                )
            except (EOFError, OSError, TimeoutError, LocalProcessWireError) as exc:
                worker.tainted = True
                worker.process.join(timeout=0.05)
                raise LocalProcessWorkerError(
                    worker_id,
                    operation,
                    str(exc),
                    pid=worker.pid,
                    exitcode=worker.process.exitcode,
                    request_id=request_id,
                ) from exc
            return self._reply_result(worker, request_id, reply)

    def _reply_result(self, worker: _Worker, request_id: int, reply: Mapping[str, Any]) -> object:
        """Shared reply fencing and decoding for both dispatch modes."""

        worker_id = worker.worker_id
        expected = ("reply", request_id, worker_id, worker.incarnation, worker.pid)
        actual = (
            reply.get("kind"),
            reply.get("request_id"),
            reply.get("worker_id"),
            reply.get("incarnation"),
            reply.get("pid"),
        )
        if actual != expected:
            worker.tainted = True
            raise LocalProcessWorkerError(
                worker_id,
                "protocol",
                f"reply identity differs: expected={expected!r}, actual={actual!r}",
                pid=worker.pid,
                request_id=request_id,
            )
        if reply["status"] == "error":
            self._raise_remote(worker_id, worker.process, reply, request_id)
        result_payload = reply["payload"]
        if not isinstance(result_payload, bytes):
            worker.tainted = True
            raise LocalProcessWorkerError(worker_id, "protocol", "reply payload is not bytes")
        try:
            return pickle.loads(result_payload)  # noqa: S301 - trusted local child only
        except BaseException as exc:
            worker.tainted = True
            raise LocalProcessWorkerError(
                worker_id,
                "result-deserialization",
                str(exc),
                pid=worker.pid,
                request_id=request_id,
            ) from exc

    def _raise_remote(
        self,
        worker_id: str,
        process: Any,
        frame: Mapping[str, Any],
        request_id: int | None,
    ) -> None:
        error = frame["error"]
        if not isinstance(error, Mapping):
            raise LocalProcessWorkerError(worker_id, "protocol", "remote error is invalid")
        raise LocalProcessWorkerError(
            worker_id,
            str(error["phase"]),
            str(error["message"]),
            pid=int(frame["pid"]),
            exitcode=process.exitcode,
            request_id=request_id,
            remote_type=str(error["type"]),
            remote_module=str(error["module"]),
            remote_traceback=str(error["traceback"]),
        )

    def _worker(self, worker_id: str) -> _Worker:
        try:
            return self._workers[worker_id]
        except KeyError as exc:
            raise KeyError(f"unknown process worker {worker_id!r}") from exc

    def _cleanup_sync(self) -> None:
        for worker_id in reversed(tuple(self._workers)):
            self._cleanup_worker_sync(worker_id)

    def _cleanup_worker_sync(self, worker_id: str) -> None:
        worker = self._worker(worker_id)
        if worker.closed:
            return
        action = "already-exited"
        error: str | None = None
        try:
            if worker.process.is_alive():
                action = "terminated"
                worker.process.terminate()
                worker.process.join(self.options.shutdown_timeout)
            if worker.process.is_alive():
                action = "killed"
                worker.process.kill()
                worker.process.join(self.options.shutdown_timeout)
            if worker.process.is_alive():
                error = "worker remains alive after kill"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        worker.connection.close()
        worker.closed = not worker.process.is_alive()
        self._record_cleanup(
            ProcessCleanupReceipt(
                worker_id,
                worker.incarnation,
                worker.pid,
                action,
                worker.process.exitcode,
                error,
            )
        )

    def _record_cleanup(self, receipt: ProcessCleanupReceipt) -> None:
        key = (receipt.worker_id, receipt.incarnation, receipt.pid)
        for retained in self._cleanup_history:
            retained_key = (retained.worker_id, retained.incarnation, retained.pid)
            if retained_key == key:
                if retained != receipt:
                    raise RuntimeError("process cleanup receipt changed for a retired incarnation")
                return
        self._cleanup_history.append(receipt)
        self._cleanup_receipts[receipt.worker_id] = receipt
