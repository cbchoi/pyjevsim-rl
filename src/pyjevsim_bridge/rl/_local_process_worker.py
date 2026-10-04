"""Spawn-safe persistent worker entry point for the local process backend."""

from __future__ import annotations

import asyncio
import inspect
import os
import pickle
import traceback
from collections.abc import Awaitable, Callable
from multiprocessing.connection import Connection

from .local_wire import WIRE_VERSION, LocalProcessWireError, decode_frame, encode_frame


def _error(phase: str, exc: BaseException) -> dict[str, str]:
    return {
        "phase": phase,
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": str(exc) or type(exc).__name__,
        "traceback": "".join(traceback.format_exception(exc)).strip() or type(exc).__name__,
    }


def _invoke(
    loop: asyncio.AbstractEventLoop,
    method: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):

        async def await_result(value: Awaitable[object]) -> object:
            return await value

        return loop.run_until_complete(await_result(result))
    return result


def _send(connection: Connection, frame: dict[str, object], max_frame_bytes: int) -> None:
    connection.send_bytes(encode_frame(frame, max_bytes=max_frame_bytes))


def _send_error_reply(
    connection: Connection,
    *,
    request_id: int,
    worker_id: str,
    incarnation: int,
    pid: int,
    phase: str,
    exc: BaseException,
    max_frame_bytes: int,
) -> None:
    full_error = _error(phase, exc)
    compact_error = {
        "phase": phase,
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": "reply exceeds frame limit",
        "traceback": type(exc).__name__,
    }
    last_error: LocalProcessWireError | None = None
    for error in (full_error, compact_error):
        reply: dict[str, object] = {
            "version": WIRE_VERSION,
            "kind": "reply",
            "request_id": request_id,
            "worker_id": worker_id,
            "incarnation": incarnation,
            "pid": pid,
            "status": "error",
            "payload": None,
            "error": error,
        }
        try:
            encoded = encode_frame(reply, max_bytes=max_frame_bytes)
        except LocalProcessWireError as wire_error:
            last_error = wire_error
            continue
        connection.send_bytes(encoded)
        return
    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to encode worker error reply")


def process_worker_main(
    connection: Connection,
    worker_id: str,
    incarnation: int,
    factory_payload: bytes,
    max_frame_bytes: int,
) -> None:
    """Materialize one environment and serve reset/step/close until terminal."""

    pid = os.getpid()
    loop = asyncio.new_event_loop()
    try:
        _send(
            connection,
            {
                "version": WIRE_VERSION,
                "kind": "hello",
                "worker_id": worker_id,
                "incarnation": incarnation,
                "pid": pid,
            },
            max_frame_bytes,
        )
        try:
            factory = pickle.loads(factory_payload)  # noqa: S301 - parent-preflighted
            environment = factory()
            if not all(hasattr(environment, name) for name in ("reset", "step", "close")):
                raise TypeError("factory must return reset/step/close methods")
        except BaseException as exc:
            _send(
                connection,
                {
                    "version": WIRE_VERSION,
                    "kind": "ready",
                    "worker_id": worker_id,
                    "incarnation": incarnation,
                    "pid": pid,
                    "status": "error",
                    "error": _error("factory-materialization", exc),
                },
                max_frame_bytes,
            )
            return
        _send(
            connection,
            {
                "version": WIRE_VERSION,
                "kind": "ready",
                "worker_id": worker_id,
                "incarnation": incarnation,
                "pid": pid,
                "status": "ready",
                "error": None,
            },
            max_frame_bytes,
        )

        while True:
            request = decode_frame(
                connection.recv_bytes(maxlength=max_frame_bytes),
                max_bytes=max_frame_bytes,
            )
            request_id = int(request["request_id"])
            operation = str(request["operation"])
            payload = request["payload"]
            replaced = False
            try:
                if request["worker_id"] != worker_id or request["incarnation"] != incarnation:
                    raise ValueError("request worker/incarnation differs from worker runtime")
                if not isinstance(payload, dict):
                    raise TypeError("request payload must be a dict")
                if operation == "reset":
                    raw = _invoke(loop, environment.reset, seed=payload["seed"])
                elif operation == "step":
                    action = pickle.loads(payload["action"])  # noqa: S301 - parent-preflighted
                    raw = _invoke(loop, environment.step, action)
                elif operation == "replace":
                    retired_environment = environment
                    _invoke(loop, retired_environment.close)
                    replacement = factory()
                    if not all(hasattr(replacement, name) for name in ("reset", "step", "close")):
                        raise TypeError("factory must return reset/step/close methods")
                    if replacement is retired_environment:
                        raise ValueError("factory reused the retired environment instance")
                    environment = replacement
                    raw = None
                    replaced = True
                else:
                    raw = _invoke(loop, environment.close)
            except BaseException as exc:
                _send_error_reply(
                    connection,
                    request_id=request_id,
                    worker_id=worker_id,
                    incarnation=incarnation,
                    pid=pid,
                    phase=operation,
                    exc=exc,
                    max_frame_bytes=max_frame_bytes,
                )
            else:
                try:
                    result_payload = pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL)
                    reply: dict[str, object] = {
                        "version": WIRE_VERSION,
                        "kind": "reply",
                        "request_id": request_id,
                        "worker_id": worker_id,
                        "incarnation": incarnation,
                        "pid": pid,
                        "status": "ok",
                        "payload": result_payload,
                        "error": None,
                    }
                    encoded_reply = encode_frame(reply, max_bytes=max_frame_bytes)
                except BaseException as exc:
                    _send_error_reply(
                        connection,
                        request_id=request_id,
                        worker_id=worker_id,
                        incarnation=incarnation,
                        pid=pid,
                        phase="result-serialization",
                        exc=exc,
                        max_frame_bytes=max_frame_bytes,
                    )
                else:
                    connection.send_bytes(encoded_reply)
                    if replaced:
                        incarnation += 1
            if operation == "close":
                return
    finally:
        loop.close()
        connection.close()
