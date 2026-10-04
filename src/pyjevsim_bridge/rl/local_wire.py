"""Closed, versioned wire contract for local spawned rollout workers."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any

WIRE_VERSION = 1

_HELLO_FIELDS = frozenset({"version", "kind", "worker_id", "incarnation", "pid"})
_READY_FIELDS = _HELLO_FIELDS | {"status", "error"}
_REQUEST_FIELDS = frozenset(
    {"version", "kind", "request_id", "worker_id", "incarnation", "operation", "payload"}
)
_REPLY_FIELDS = frozenset(
    {
        "version",
        "kind",
        "request_id",
        "worker_id",
        "incarnation",
        "pid",
        "status",
        "payload",
        "error",
    }
)
_ERROR_FIELDS = frozenset({"phase", "type", "module", "message", "traceback"})


class LocalProcessWireError(ValueError):
    """A spawned worker frame violated the closed local wire contract."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LocalProcessWireError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise LocalProcessWireError(f"{name} keys must be strings")
    return value


def _exact(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    fields = set(value)
    if fields != expected:
        raise LocalProcessWireError(
            f"{name} fields differ: missing={sorted(expected - fields)}, "
            f"extra={sorted(fields - expected)}"
        )


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LocalProcessWireError(f"{name} must be a non-empty string")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LocalProcessWireError(f"{name} must be a non-negative integer")
    return value


def validate_error(value: object) -> dict[str, str]:
    error = _mapping(value, "error")
    _exact(error, _ERROR_FIELDS, "error")
    return {field: _non_empty_string(error[field], f"error.{field}") for field in _ERROR_FIELDS}


def validate_frame(value: object) -> dict[str, Any]:
    frame = _mapping(value, "frame")
    kind = frame.get("kind")
    if kind == "hello":
        _exact(frame, _HELLO_FIELDS, "hello")
    elif kind == "ready":
        _exact(frame, _READY_FIELDS, "ready")
    elif kind == "request":
        _exact(frame, _REQUEST_FIELDS, "request")
    elif kind == "reply":
        _exact(frame, _REPLY_FIELDS, "reply")
    else:
        raise LocalProcessWireError(f"unsupported frame kind {kind!r}")

    if frame["version"] != WIRE_VERSION:
        raise LocalProcessWireError(f"unsupported wire version {frame['version']!r}")
    _non_empty_string(frame["worker_id"], "worker_id")
    _non_negative_int(frame["incarnation"], "incarnation")

    if kind in {"hello", "ready", "reply"} and (
        isinstance(frame["pid"], bool) or not isinstance(frame["pid"], int) or frame["pid"] <= 0
    ):
        raise LocalProcessWireError("pid must be a positive integer")
    if kind in {"request", "reply"}:
        _non_negative_int(frame["request_id"], "request_id")
    if kind == "request":
        if frame["operation"] not in {"reset", "step", "replace", "close"}:
            raise LocalProcessWireError("request operation must be reset, step, replace, or close")
        payload = _mapping(frame["payload"], "request.payload")
        expected = {
            "reset": frozenset({"seed"}),
            "step": frozenset({"action"}),
            "replace": frozenset(),
            "close": frozenset(),
        }[frame["operation"]]
        _exact(payload, expected, "request.payload")
        if frame["operation"] == "reset":
            seed = payload["seed"]
            if seed is not None:
                _non_negative_int(seed, "request.payload.seed")
        if frame["operation"] == "step" and not isinstance(payload["action"], bytes):
            raise LocalProcessWireError("request.payload.action must be bytes")
    elif kind == "ready":
        if frame["status"] not in {"ready", "error"}:
            raise LocalProcessWireError("ready status must be ready or error")
        if frame["status"] == "ready" and frame["error"] is not None:
            raise LocalProcessWireError("ready frame may not contain an error")
        if frame["status"] == "error":
            validate_error(frame["error"])
    elif kind == "reply":
        if frame["status"] not in {"ok", "error"}:
            raise LocalProcessWireError("reply status must be ok or error")
        if frame["status"] == "ok":
            if frame["error"] is not None or not isinstance(frame["payload"], bytes):
                raise LocalProcessWireError("successful reply requires bytes payload and no error")
        else:
            if frame["payload"] is not None:
                raise LocalProcessWireError("error reply payload must be null")
            validate_error(frame["error"])
    return dict(frame)


def encode_frame(value: object, *, max_bytes: int) -> bytes:
    """Validate and serialize a frame under an explicit size bound."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    frame = validate_frame(value)
    payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > max_bytes:
        raise LocalProcessWireError(f"wire frame size {len(payload)} exceeds limit {max_bytes}")
    return payload


def decode_frame(payload: bytes, *, max_bytes: int) -> dict[str, Any]:
    """Deserialize and validate a frame from a trusted local worker."""

    if not isinstance(payload, bytes):
        raise LocalProcessWireError("wire payload must be bytes")
    if len(payload) > max_bytes:
        raise LocalProcessWireError(f"wire frame size {len(payload)} exceeds limit {max_bytes}")
    try:
        value = pickle.loads(payload)  # noqa: S301 - trusted local child only
    except Exception as exc:
        raise LocalProcessWireError(f"wire payload cannot be decoded: {exc}") from exc
    return validate_frame(value)
