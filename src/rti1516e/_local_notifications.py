"""Portable completed-RPC notifications, with no server authority or work."""

from __future__ import annotations

import secrets
import threading
from collections import deque
from collections.abc import Collection, Sequence
from dataclasses import replace
from enum import IntEnum
from typing import NotRequired, Protocol, TypedDict, TypeVar, cast

from ._callbacks import ServiceCallback
from ._services import FederationExecutionInformation
from .savepoint import (
    FederateRestoreStatus,
    FederateSaveStatus,
    RestoreState,
    RestoreStatus,
    RestoreStatusResponse,
    SaveState,
    SaveStatus,
    SaveStatusResponse,
)

KINDS = {
    "reportFederationExecutions",
    "confirmAttributeTransportationTypeChange",
    "reportAttributeTransportationType",
    "confirmInteractionTransportationTypeChange",
    "reportInteractionTransportationType",
    "federationSaveStatusResponse",
    "federationRestoreStatusResponse",
}
METADATA = {
    "_local_notification_id",
    "_local_notification_order",
    "_local_notification_scope",
    "_restore_serial",
}


LocalNotification = ServiceCallback | SaveStatusResponse | RestoreStatusResponse
_EventT = TypeVar("_EventT")


class LocalNotificationRow(TypedDict):
    local_id: str
    order: int
    scope: str
    kind: str
    payload: dict[str, object]


class _ExecutionRow(TypedDict):
    name: str
    logical_time_implementation_name: str
    mode: NotRequired[int]
    federates_joined: NotRequired[int]
    federation_generation: NotRequired[int]


class _ExecutionPayload(TypedDict):
    executions: list[_ExecutionRow]


class _SaveStatusRow(TypedDict):
    federate_handle: int
    status: int


class _RestoreStatusRow(TypedDict):
    pre_restore_handle: int
    post_restore_handle: int
    status: int


class _SavePayload(TypedDict):
    statuses: list[_SaveStatusRow]
    state: NotRequired[int]
    active_label: NotRequired[str]


class _RestorePayload(TypedDict):
    statuses: list[_RestoreStatusRow]
    state: NotRequired[int]
    active_label: NotRequired[str]


class _NotificationMetadata(Protocol):
    _local_notification_id: str
    _local_notification_order: int
    _local_notification_scope: str


class _NotificationOwner(Protocol):
    _local_notification_state: LocalNotificationState


class _NotificationTransport(Protocol):
    _local_notification_states: dict[object, LocalNotificationState]


class LocalNotificationState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._completed: deque[str] = deque()
        self._inflight = 0

    def begin(self, event: object) -> bool:
        to_row(event)
        identity = cast(_NotificationMetadata, event)._local_notification_id
        with self._lock:
            if identity in self._seen:
                return False
            self._seen.add(identity)
            self._inflight += 1
        return True

    def finish(self, event: object) -> None:
        from ._runtime_checkpoint import MAX_ENTRIES

        with self._lock:
            self._inflight -= 1
            self._completed.append(cast(_NotificationMetadata, event)._local_notification_id)
            if len(self._completed) > MAX_ENTRIES:
                self._seen.remove(self._completed.popleft())

    def checkpoint(self) -> None:
        with self._lock:
            if self._inflight:
                raise RuntimeError("cannot checkpoint an executing local notification")


def notification_state(owner: object) -> LocalNotificationState:
    fed: object = getattr(owner, "_federate", None) or owner
    transport = cast(_NotificationTransport | None, getattr(fed, "_transport", None))
    handle: object = getattr(fed, "handle", None)
    if transport is not None and handle is not None:
        states = cast(
            dict[object, LocalNotificationState] | None,
            getattr(transport, "_local_notification_states", None),
        )
        if states is None:
            states = transport._local_notification_states = {}
        if handle not in states:
            states[handle] = LocalNotificationState()
        return states[handle]
    state = cast(LocalNotificationState | None, getattr(owner, "_local_notification_state", None))
    if state is None:
        state = cast(_NotificationOwner, owner)._local_notification_state = LocalNotificationState()
    return state


def kind_of(event: object) -> str | None:
    if isinstance(event, ServiceCallback) and event.method in KINDS:
        return event.method
    if isinstance(event, SaveStatusResponse):
        return "federationSaveStatusResponse"
    if isinstance(event, RestoreStatusResponse):
        return "federationRestoreStatusResponse"
    return None


def scope_of(kind: str | None) -> str:
    return "connection" if kind == "reportFederationExecutions" else "member"


def prepare(event: _EventT, order: int, serial: int | None = None) -> _EventT:
    if kind_of(event) is None or getattr(event, "_wire_carrier", None) is not None:
        return event
    if hasattr(event, "_local_notification_id"):
        return event
    event = cast(_EventT, replace(cast(LocalNotification, event)))
    object.__setattr__(event, "_local_notification_id", secrets.token_hex(32))
    object.__setattr__(event, "_local_notification_order", order)
    object.__setattr__(event, "_local_notification_scope", scope_of(kind_of(event)))
    if serial is not None:
        object.__setattr__(event, "_restore_serial", serial)
    return event


def _fields(
    value: object, required: Collection[str], optional: Collection[str] = ()
) -> dict[str, object]:
    if type(value) is not dict or not set(required) <= set(value) <= set(required) | set(optional):
        raise ValueError("local notification fields differ")
    return cast(dict[str, object], value)


def _uint(value: object, *, zero: bool = False, bits: int = 64) -> int:
    if type(value) is not int or not (0 if zero else 1) <= value < 1 << bits:
        raise ValueError("invalid local notification integer")
    return value


def _string(value: object, *, nonempty: bool = False) -> None:
    if type(value) is not str or (nonempty and not value):
        raise ValueError("invalid local notification string")


def _enum(value: object, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid local notification enum")


def _array(value: object) -> list[object]:
    from ._runtime_checkpoint import entries

    return entries(value)


def validate_payload(kind: str, payload: object) -> None:
    if kind == "reportFederationExecutions":
        payload = _fields(payload, {"executions"})
        names: set[str] = set()
        for item in _array(payload["executions"]):
            row = _fields(
                item,
                {"name", "logical_time_implementation_name"},
                {"mode", "federates_joined", "federation_generation"},
            )
            _string(row["name"], nonempty=True)
            _string(row["logical_time_implementation_name"], nonempty=True)
            name = cast(str, row["name"])
            if name in names:
                raise ValueError("duplicate local notification execution name")
            names.add(name)
            if "mode" in row:
                _enum(row["mode"], 0, 2)
            if "federates_joined" in row:
                _uint(row["federates_joined"], zero=True, bits=32)
            if "federation_generation" in row:
                _uint(row["federation_generation"], zero=True)
    elif kind in ("federationSaveStatusResponse", "federationRestoreStatusResponse"):
        restore = kind == "federationRestoreStatusResponse"
        payload = _fields(payload, {"statuses"}, {"state", "active_label"})
        if "state" in payload:
            _enum(payload["state"], 1, 5 if restore else 4)
        if "active_label" in payload:
            _string(payload["active_label"])
        seen: set[int] = set()
        for item in _array(payload["statuses"]):
            fields = (
                {"pre_restore_handle", "post_restore_handle", "status"}
                if restore
                else {"federate_handle", "status"}
            )
            row = _fields(item, fields)
            participant = _uint(row["pre_restore_handle"] if restore else row["federate_handle"])
            if participant in seen:
                raise ValueError("duplicate local notification participant")
            seen.add(participant)
            if restore:
                historical = _uint(row["post_restore_handle"], zero=True)
                if historical and not (historical & 0xFFFFFFFF):
                    raise ValueError("invalid historical local notification handle")
            _enum(row["status"], 1, 6 if restore else 4)
    else:
        fields = {
            "confirmAttributeTransportationTypeChange": {
                "object_handle",
                "attribute_handles",
                "transport_type",
            },
            "reportAttributeTransportationType": {
                "object_handle",
                "attribute_handle",
                "transport_type",
            },
            "confirmInteractionTransportationTypeChange": {
                "interaction_class_handle",
                "transport_type",
            },
            "reportInteractionTransportationType": {
                "federate_handle",
                "interaction_class_handle",
                "transport_type",
            },
        }[kind]
        payload = _fields(payload, fields)
        for field in fields - {"attribute_handles", "transport_type"}:
            _uint(payload[field])
        _enum(payload["transport_type"], 1, 2)
        if "attribute_handles" in fields:
            handles = [_uint(value) for value in _array(payload["attribute_handles"])]
            if len(set(handles)) != len(handles):
                raise ValueError("duplicate local notification attribute")


def validate_rows(rows: object, generation: int | None = None) -> None:
    seen: set[str] = set()
    previous = 0
    for item in _array(rows):
        row = _fields(item, {"local_id", "order", "scope", "kind", "payload"})
        identity = row["local_id"]
        if (
            type(identity) is not str
            or len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)
            or identity in seen
        ):
            raise ValueError("invalid or duplicate local notification identity")
        seen.add(identity)
        order = _uint(row["order"])
        if order <= previous:
            raise ValueError("local notification order must increase")
        previous = order
        kind = row["kind"]
        if type(kind) is not str or kind not in KINDS:
            raise ValueError("unsupported Python local notification kind")
        if row["scope"] != scope_of(kind):
            raise ValueError("local notification scope differs")
        validate_payload(kind, row["payload"])
        if generation is not None:
            payload = cast(dict[str, object], row["payload"])
            refs: list[int] = []
            if kind == "reportInteractionTransportationType":
                refs = [cast(int, payload["federate_handle"])]
            elif kind == "federationSaveStatusResponse":
                refs = [
                    value["federate_handle"] for value in cast(_SavePayload, payload)["statuses"]
                ]
            elif kind == "federationRestoreStatusResponse":
                refs = [
                    value["pre_restore_handle"]
                    for value in cast(_RestorePayload, payload)["statuses"]
                ]
            if any(value >> 32 != generation or not (value & 0xFFFFFFFF) for value in refs):
                raise ValueError("local notification federate belongs to another generation")


def _native_handle(value: object) -> object:
    from .handles import _StrongHandle

    return int(value) if isinstance(value, _StrongHandle) else value


def _native_enum(value: object, enum: type[IntEnum]) -> int:
    if type(value) is enum:
        return int(value)
    if type(value) is not int:
        raise ValueError("invalid local notification enum")
    return value


def _sequence(value: object) -> Sequence[object]:
    if type(value) not in (list, tuple):
        raise ValueError("local notification requires a materialized sequence")
    sequence = cast(Sequence[object], value)
    _array(list(sequence))
    return sequence


def to_row(event: object) -> LocalNotificationRow:
    kind = kind_of(event)
    native = (
        {"method", "args"}
        if isinstance(event, ServiceCallback)
        else {"state", "active_label", "federate_statuses"}
    )
    if kind is None or set(vars(event)) - native - METADATA:
        raise ValueError("local notification carries unsupported data or authority")
    try:
        payload: dict[str, object]
        if isinstance(event, ServiceCallback) and type(event.args) is not tuple:
            raise ValueError("local notification requires materialized arguments")
        if kind == "reportFederationExecutions":
            (executions,) = cast(ServiceCallback, event).args
            _sequence(executions)
            if any(type(value) is not FederationExecutionInformation for value in executions):
                raise ValueError("local notification requires typed execution values")
            payload = {
                "executions": [
                    {
                        "name": value.name,
                        "mode": value.mode,
                        "federates_joined": value.federates_joined,
                        "federation_generation": value.federation_generation,
                        "logical_time_implementation_name": value.logical_time_implementation_name,
                    }
                    for value in executions
                ]
            }
        elif kind == "federationSaveStatusResponse":
            save_event = cast(SaveStatusResponse, event)
            _sequence(save_event.federate_statuses)
            if any(type(value) is not FederateSaveStatus for value in save_event.federate_statuses):
                raise ValueError("local notification requires typed save status")
            payload = {
                "state": _native_enum(save_event.state, SaveState),
                "active_label": save_event.active_label,
                "statuses": [
                    {
                        "federate_handle": _native_handle(value.federate_handle),
                        "status": _native_enum(value.status, SaveStatus),
                    }
                    for value in save_event.federate_statuses
                ],
            }
        elif kind == "federationRestoreStatusResponse":
            restore_event = cast(RestoreStatusResponse, event)
            _sequence(restore_event.federate_statuses)
            if any(
                type(value) is not FederateRestoreStatus
                for value in restore_event.federate_statuses
            ):
                raise ValueError("local notification requires typed restore status")
            payload = {
                "state": _native_enum(restore_event.state, RestoreState),
                "active_label": restore_event.active_label,
                "statuses": [
                    {
                        "pre_restore_handle": _native_handle(value.pre_restore_handle),
                        "post_restore_handle": _native_handle(value.post_restore_handle),
                        "status": _native_enum(value.status, RestoreStatus),
                    }
                    for value in restore_event.federate_statuses
                ],
            }
        else:
            fields = {
                "confirmAttributeTransportationTypeChange": (
                    "object_handle",
                    "attribute_handles",
                    "transport_type",
                ),
                "reportAttributeTransportationType": (
                    "object_handle",
                    "attribute_handle",
                    "transport_type",
                ),
                "confirmInteractionTransportationTypeChange": (
                    "interaction_class_handle",
                    "transport_type",
                ),
                "reportInteractionTransportationType": (
                    "federate_handle",
                    "interaction_class_handle",
                    "transport_type",
                ),
            }[kind]
            args = cast(ServiceCallback, event).args
            if len(args) != len(fields):
                raise ValueError("local notification argument count differs")
            payload = dict(zip(fields, args, strict=False))
            if "attribute_handles" in payload:
                payload["attribute_handles"] = [
                    _native_handle(value) for value in _sequence(payload["attribute_handles"])
                ]
            for field in payload.keys() - {"attribute_handles", "transport_type"}:
                payload[field] = _native_handle(payload[field])
        metadata = cast(_NotificationMetadata, event)
        row: LocalNotificationRow = {
            "local_id": metadata._local_notification_id,
            "order": metadata._local_notification_order,
            "scope": metadata._local_notification_scope,
            "kind": kind,
            "payload": payload,
        }
        validate_rows([row])
        return row
    except (AttributeError, TypeError) as exc:
        raise ValueError("invalid local notification value") from exc


def from_rows(rows: object, generation: int | None = None) -> list[LocalNotification]:
    validate_rows(rows, generation)
    result: list[LocalNotification] = []
    for row in cast(list[LocalNotificationRow], rows):
        kind, payload = row["kind"], row["payload"]
        event: LocalNotification
        if kind == "reportFederationExecutions":
            event = ServiceCallback(
                kind,
                (
                    tuple(
                        FederationExecutionInformation(
                            value["name"],
                            value.get("mode", 0),
                            value.get("federates_joined", 0),
                            value.get("federation_generation", 0),
                            value["logical_time_implementation_name"],
                        )
                        for value in cast(_ExecutionPayload, payload)["executions"]
                    ),
                ),
            )
        elif kind == "federationSaveStatusResponse":
            save_payload = cast(_SavePayload, payload)
            event = SaveStatusResponse(
                SaveState(save_payload.get("state", 1)),
                save_payload.get("active_label", ""),
                tuple(
                    FederateSaveStatus(value["federate_handle"], SaveStatus(value["status"]))
                    for value in save_payload["statuses"]
                ),
            )
        elif kind == "federationRestoreStatusResponse":
            restore_payload = cast(_RestorePayload, payload)
            event = RestoreStatusResponse(
                RestoreState(restore_payload.get("state", 1)),
                restore_payload.get("active_label", ""),
                tuple(
                    FederateRestoreStatus(
                        value["pre_restore_handle"],
                        value["post_restore_handle"],
                        RestoreStatus(value["status"]),
                    )
                    for value in restore_payload["statuses"]
                ),
            )
        else:
            fields = {
                "confirmAttributeTransportationTypeChange": (
                    "object_handle",
                    "attribute_handles",
                    "transport_type",
                ),
                "reportAttributeTransportationType": (
                    "object_handle",
                    "attribute_handle",
                    "transport_type",
                ),
                "confirmInteractionTransportationTypeChange": (
                    "interaction_class_handle",
                    "transport_type",
                ),
                "reportInteractionTransportationType": (
                    "federate_handle",
                    "interaction_class_handle",
                    "transport_type",
                ),
            }[kind]
            event = ServiceCallback(kind, tuple(payload[field] for field in fields))
        object.__setattr__(event, "_local_notification_id", row["local_id"])
        object.__setattr__(event, "_local_notification_order", row["order"])
        object.__setattr__(event, "_local_notification_scope", row["scope"])
        result.append(event)
    return result
