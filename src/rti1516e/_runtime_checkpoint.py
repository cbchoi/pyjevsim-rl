"""Portable private runtime state; pending callbacks are recovered by the server."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict, cast

from google.protobuf.json_format import MessageToDict, ParseDict

from ._logical_time import INTEGER64_TIME, require_supported_logical_time, validate_time

MAX_CHECKPOINT_BYTES = 1 << 20
MAX_ENTRIES = 8192
FORMAT = "gorti-python-sdk-runtime"
FIELDS = {
    "format",
    "version",
    "federation_generation",
    "federate_handle",
    "callback_epoch",
    "logical_time_implementation_name",
    "logical_time",
    "object_knowledge",
    "next_retraction_handle",
    "used_retraction_handles",
    "pending_callbacks",
    "callback_recovery",
}
IDENTITY_FIELDS = {
    "format",
    "version",
    "federation_generation",
    "federate_handle",
    "logical_time_implementation_name",
}
ENVELOPE_FIELDS = {
    "magic",
    "version",
    "language",
    "scalar",
    "generation",
    "self",
    "state",
    "checksum",
}


class _TimeValue(TypedDict, total=False):
    integer64_value: str
    float64_value: int | float


class _CallbackRecovery(Protocol):
    invocation_identity: bytes
    outcome_only: bool
    completed: bool


def _compact(raw: str) -> bytes:
    result = []
    quoted = escaped = False
    for char in raw:
        if quoted or char not in " \t\r\n":
            result.append(char)
        if escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
    return "".join(result).encode("utf-8")


def _raw_state(payload: bytes) -> bytes:
    raw = payload.decode("utf-8")
    decoder = json.JSONDecoder()
    position = raw.index("{") + 1
    while True:
        while raw[position] in " \t\r\n,":
            position += 1
        key, position = decoder.raw_decode(raw, position)
        while raw[position] in " \t\r\n:":
            position += 1
        start = position
        _, position = decoder.raw_decode(raw, position)
        if key == "state":
            return _compact(raw[start:position])


def _checksum(envelope: Mapping[str, object], state_bytes: bytes) -> str:
    header = (
        f"GRS1\n1\npython\n{envelope['scalar']}\n{envelope['generation']}\n{envelope['self']}\n"
    ).encode("ascii")
    return hashlib.sha256(header + state_bytes).hexdigest()


def uint64(value: object) -> int:
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise ValueError("checkpoint uint64 is invalid")
    return value


def entries(value: object) -> list[object]:
    if type(value) is not list or len(value) > MAX_ENTRIES:
        raise ValueError("checkpoint collection is invalid or exceeds capacity")
    return value


def time_dict(value: object, selected: str) -> _TimeValue | None:
    if value is None:
        return None
    logical_time = validate_time(value, selected)
    if selected == INTEGER64_TIME:
        return {"integer64_value": str(logical_time)}
    return {"float64_value": logical_time}


def reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate checkpoint field")
        result[key] = value
    return result


def validate_outcomes(rows: object) -> None:
    seen: set[str] = set()
    for row in entries(rows):
        if type(row) is not dict or set(row) != {"identity", "success", "exception"}:
            raise ValueError("checkpoint pending outcome fields differ")
        identity = row["identity"]
        if (
            type(identity) is not str
            or len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)
            or identity in seen
        ):
            raise ValueError("checkpoint pending outcome identity is invalid")
        if type(row["success"]) is not bool or type(row["exception"]) is not str:
            raise ValueError("checkpoint pending outcome types differ")
        if row["success"] and row["exception"]:
            raise ValueError("successful checkpoint outcome cannot carry an exception")
        seen.add(identity)


def validate_recovery_carrier(
    wire: Any, *, require_receipt: bool = False
) -> _CallbackRecovery | None:
    # Generated protobuf carriers expose schema-dependent recovery fields.
    recovery = cast(
        _CallbackRecovery | None,
        wire.callback_recovery
        if "callback_recovery" in wire.DESCRIPTOR.fields_by_name
        and wire.HasField("callback_recovery")
        else None,
    )
    if recovery is None:
        if require_receipt and len(wire.callback_receipt) != 32:
            raise ValueError("checkpoint callback receipt must contain 32 bytes")
        return None
    if len(recovery.invocation_identity) != 32:
        raise ValueError("invalid callback recovery identity")
    if recovery.completed:
        if (
            not recovery.outcome_only
            or wire.callback_receipt
            or wire.WhichOneof("event") is not None
        ):
            raise ValueError("invalid completed callback recovery marker")
    elif len(wire.callback_receipt) != 32 or wire.WhichOneof("event") is None:
        raise ValueError("invalid callback recovery carrier")
    return recovery


def require_recovery_schema() -> None:
    from rti.v1 import stream_pb2

    field = stream_pb2.FederateEvent.DESCRIPTOR.fields_by_name.get("callback_recovery")
    if field is None or not {"invocation_identity", "outcome_only", "completed"}.issubset(
        field.message_type.fields_by_name
    ):
        raise ValueError("server-v2 callback recovery requires updated wire bindings")


def decode(payload: bytes, selected: str) -> dict[str, Any]:
    from rti.v1 import stream_pb2

    if not isinstance(payload, bytes) or len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds 1 MiB")
    try:
        # JSON remains dynamic until the exact field and value checks below.
        envelope: Any = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        if type(envelope) is not dict or set(envelope) != ENVELOPE_FIELDS:
            raise ValueError("checkpoint envelope fields differ")
        if (
            envelope["magic"] != "GRS1"
            or type(envelope["version"]) is not int
            or envelope["version"] != 1
            or envelope["language"] != "python"
        ):
            raise ValueError("unsupported checkpoint envelope")
        for field in ("generation", "self"):
            uint64(envelope[field])
        if not envelope["generation"] or not envelope["self"]:
            raise ValueError("checkpoint envelope identity is invalid")
        if envelope["scalar"] != selected:
            raise ValueError("checkpoint logical time family differs")
        checksum = envelope["checksum"]
        if type(checksum) is not str or not hmac.compare_digest(
            checksum, _checksum(envelope, _raw_state(payload))
        ):
            raise ValueError("checkpoint checksum differs")
        state: dict[str, Any] = envelope["state"]
        fields = (
            FIELDS | {"pending_outcomes"}
            if type(state) is dict and state.get("callback_recovery") == "server-v2"
            else FIELDS
        )
        if (
            type(state) is dict
            and state.get("callback_recovery") == "server-v2"
            and "pending_local_notifications" in state
        ):
            fields = fields | {"pending_local_notifications"}
        if (
            type(state) is dict
            and state.get("callback_recovery") == "server-v2"
            and "pending_local_notifications" in state
        ):
            fields = fields | {"pending_local_notifications"}
        if type(state) is not dict or set(state) != fields - IDENTITY_FIELDS:
            raise ValueError("checkpoint private fields differ")
        state.update(
            format=FORMAT,
            version=1,
            federation_generation=envelope["generation"],
            federate_handle=envelope["self"],
            logical_time_implementation_name=envelope["scalar"],
        )
        if type(state) is not dict or set(state) != fields:
            raise ValueError("checkpoint fields differ")
        if state["format"] != FORMAT or type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("unsupported checkpoint format or version")
        if state["callback_recovery"] not in ("server", "server-v2"):
            raise ValueError("checkpoint requires authoritative server callback recovery")
        if state["callback_recovery"] == "server-v2":
            require_recovery_schema()
            validate_outcomes(state["pending_outcomes"])
            if "pending_local_notifications" in state:
                from ._local_notifications import from_rows

                state["decoded_local_notifications"] = from_rows(
                    state["pending_local_notifications"], envelope["generation"]
                )
            if "pending_local_notifications" in state:
                from ._local_notifications import from_rows

                state["decoded_local_notifications"] = from_rows(
                    state["pending_local_notifications"]
                )
        family = require_supported_logical_time(state["logical_time_implementation_name"])
        if family != selected:
            raise ValueError("checkpoint logical time family differs")
        for name in ("federation_generation", "federate_handle", "callback_epoch"):
            uint64(state[name])
        if not state["federation_generation"] or not state["federate_handle"]:
            raise ValueError("checkpoint identity is invalid")
        logical_time = state["logical_time"]
        if logical_time is not None:
            key = "integer64_value" if selected == INTEGER64_TIME else "float64_value"
            if type(logical_time) is not dict or set(logical_time) != {key}:
                raise ValueError("checkpoint logical time representation differs")
            value = logical_time[key]
            if selected == INTEGER64_TIME:
                if type(value) is not str or not value.isascii() or not value.isdecimal():
                    raise ValueError("checkpoint Integer64 time is invalid")
                value = int(value)
                if str(value) != logical_time[key]:
                    raise ValueError("checkpoint Integer64 time is not canonical")
            state["decoded_time"] = validate_time(value, selected)
        else:
            state["decoded_time"] = None
        knowledge = state["object_knowledge"]
        if type(knowledge) is not dict or set(knowledge) != {"latest", "retired"}:
            raise ValueError("checkpoint object knowledge fields differ")
        for name in ("latest", "retired"):
            seen = set()
            for row in entries(knowledge[name]):
                if type(row) is not list or len(row) != 2:
                    raise ValueError("checkpoint object knowledge row differs")
                handle, epoch = map(uint64, row)
                if not handle or handle in seen:
                    raise ValueError("checkpoint object knowledge identity is invalid")
                seen.add(handle)
        hint = state["next_retraction_handle"]
        if hint is not None:
            uint64(hint)
        used = entries(state["used_retraction_handles"])
        if len(set(map(uint64, used))) != len(used) or 0 in used:
            raise ValueError("checkpoint retraction handles are invalid")
        wires = []
        for callback in entries(state["pending_callbacks"]):
            wire = ParseDict(cast(dict[str, Any], callback), stream_pb2.FederateEvent())
            validate_recovery_carrier(
                wire, require_receipt=state["callback_recovery"] == "server-v2"
            )
            completed = (
                "callback_recovery" in wire.DESCRIPTOR.fields_by_name
                and wire.HasField("callback_recovery")
                and wire.callback_recovery.completed
            )
            if wire.WhichOneof("event") is None and not completed:
                raise ValueError("checkpoint callback has no event")
            wires.append(wire)
        state["decoded_callbacks"] = wires
        return state
    except (TypeError, KeyError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid runtime checkpoint") from exc
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("invalid runtime checkpoint") from exc


def encode(state: Mapping[str, object]) -> bytes:
    private = {key: value for key, value in state.items() if key not in IDENTITY_FIELDS}
    state_bytes = json.dumps(
        private, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    envelope: dict[str, object] = {
        "magic": "GRS1",
        "version": 1,
        "language": "python",
        "scalar": state["logical_time_implementation_name"],
        "generation": state["federation_generation"],
        "self": state["federate_handle"],
        "state": private,
    }
    envelope["checksum"] = _checksum(envelope, state_bytes)
    payload = json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds 1 MiB")
    decode(payload, cast(str, state["logical_time_implementation_name"]))
    return payload


def callback_dict(event: object) -> dict[str, Any]:
    wire = getattr(event, "_wire_carrier", None)
    if wire is None:
        raise ValueError("pending callback has no portable wire carrier")
    completed = (
        "callback_recovery" in wire.DESCRIPTOR.fields_by_name
        and wire.HasField("callback_recovery")
        and wire.callback_recovery.completed
    )
    if not wire.callback_receipt and not completed:
        raise ValueError("pending callback has no authoritative recovery receipt")
    return MessageToDict(wire, preserving_proto_field_name=True)
