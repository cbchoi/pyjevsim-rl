"""Transport registry — pluggable in-process doubles + real gRPC for the RTI client.

The Layer 1 SDK (rti1516e.connection.RtiConnection) has three backing transports:

  - ``memory://<name>``   — in-process pure-Python ``InProcessTransport``
                            (production-suitable; historically called
                            ``FakeRtiServer`` and re-exported under that
                            name for back-compat).
  - ``grpc://host:port``  — real gRPC channel to an rtid binary, wrapped
                            in :class:`GrpcTransport` and built on demand
                            by ``RtiConnection.__aenter__``.
  - ``grpcs://host:port`` — TLS-secured variant of ``grpc://``. Server-
                            side TLS is provided by rtid's
                            ``--tls-cert/--tls-key`` flag pair (M6 W1B);
                            the client supplies a CA bundle via
                            ``RtiConnection.connect(url, ca_cert=...)``
                            or relies on the system trust store when
                            ``ca_cert`` is omitted.

Both transports satisfy the same duck-typed surface that ``RtiConnection``
+ ``Federate`` reach for:

  - ``record(method, **kwargs) -> Any`` (or awaitable; the SDK awaits the
    result if it is awaitable, so a sync fake and an async gRPC client can
    coexist in the same call sites).
  - ``events_for(handle) -> asyncio.Queue``
  - ``allocate_handle() -> int``

The ``GrpcTransport`` maps SDK operations to generated gRPC stubs:

  - Federation, declaration, object, interaction, time, and event-stream
    operations are dispatched to their corresponding RTI services.
  - Class names are translated to numeric handles via the FOM (parsed
    on-demand from the FederationSpec.fom_modules paths). Both this Python
    SDK and the Go-side ``fomHandle.LookupInteractionClass`` derive
    handles from a sorted-by-name index, so the two sides agree
    deterministically.
  - Time-management RPCs such as NextMessageRequest,
    EnableTimeRegulation, and EnableTimeConstrained use rtid's
    TimeService. The bridge resolves pending advances when a
    TimeAdvanceGrant arrives on ``StreamService.Events``.

This module is internal to rti1516e and is not part of the public API.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast, overload

if TYPE_CHECKING:  # pragma: no cover - import guard for type checking only
    import grpc

    from ._ownership_callbacks import OwnershipCallbacks

# M39 (HA-1): FederateEvent oneof variants that reached _translate_event
# without a branch. Warned once per variant per process (see the default
# branch at the bottom of _translate_event).
_UNTRANSLATED_EVENT_WARNED: set[str] = set()


# Imported by lifecycle tests; retain the existing exception identity and name.
class _CallbackStreamClosed(RuntimeError):  # noqa: N818
    """A clean stream EOF, unexpected unless resignation has succeeded."""


class _QueueStorage(Protocol):
    @property
    def _queue(self) -> Iterable[Any]: ...


# Module-level registry keyed by URL. The value is intentionally typed as
# ``Any`` because the fake server lives in test code (pysdk/tests/spec/m4/
# _fakes/) and the SDK must not import test packages. The runtime contract
# is: the registered object exposes ``record(method, **kwargs)``,
# ``events_for(handle) -> asyncio.Queue``, and ``allocate_handle() -> int``.
_TRANSPORT_REGISTRY: dict[str, Any] = {}


def register_fake(url: str, server: Any) -> None:
    """Bind ``server`` as the transport returned by ``lookup(url)``.

    If a transport is already registered under ``url``, it is replaced
    (last-writer-wins). This is fine for tests, which construct a fresh
    fake per test function.
    """
    _TRANSPORT_REGISTRY[url] = server


def unregister(url: str) -> None:
    """Remove the registered transport for ``url``. No-op if absent."""
    _TRANSPORT_REGISTRY.pop(url, None)


def lookup(url: str) -> Any | None:
    """Return the registered transport for ``url``, or None if unregistered."""
    return _TRANSPORT_REGISTRY.get(url)


def clear() -> None:
    """Remove every registered transport. Useful for test isolation."""
    _TRANSPORT_REGISTRY.clear()


# --- Resign-action mapping (M36) ---------------------------------------------

# IEEE 1516.1-2010 §4.10 resign-action designators → ``rti.v1.ResignAction``
# wire enum names (M24 W2). Three of the proto names are shortened forms of
# the IEEE designators, so a plain ``"RESIGN_ACTION_" + name`` concat is NOT
# correct — keep this table exhaustive and explicit.
_RESIGN_ACTION_WIRE_NAMES: dict[str, str] = {
    "UNCONDITIONALLY_DIVEST_ATTRIBUTES": "RESIGN_ACTION_UNCONDITIONALLY_DIVEST_ATTRIBUTES",
    "DELETE_OBJECTS": "RESIGN_ACTION_DELETE_OBJECTS",
    "CANCEL_PENDING_OWNERSHIP_ACQUISITIONS": "RESIGN_ACTION_CANCEL_PENDING_OWNERSHIP",
    "DELETE_OBJECTS_THEN_DIVEST": "RESIGN_ACTION_DELETE_THEN_DIVEST",
    "CANCEL_THEN_DELETE_THEN_DIVEST": "RESIGN_ACTION_CANCEL_THEN_DELETE",
    "NO_ACTION": "RESIGN_ACTION_NO_ACTION",
}

#: Public view of the accepted IEEE §4.10 designator strings (Layer 2
#: validates against this before dispatching the resign).
RESIGN_ACTION_NAMES = frozenset(_RESIGN_ACTION_WIRE_NAMES)

_ValueT = TypeVar("_ValueT")


def resign_action_to_proto(action: int | str | None) -> int:
    """Map a resign action to the ``rti.v1.ResignAction`` enum value.

    - ``None``  → ``RESIGN_ACTION_UNCONDITIONALLY_DIVEST_ATTRIBUTES``
      (the pre-M24 default).
    - ``int``   → passed through unchanged (caller already holds the
      proto enum value; M24 W2 contract).
    - ``str``   → IEEE 1516.1-2010 §4.10 designator, translated via
      :data:`_RESIGN_ACTION_WIRE_NAMES`. Unknown names raise
      ``ValueError`` (§4.10 InvalidResignAction).
    """
    from rti.v1 import common_pb2

    if action is None:
        return int(common_pb2.ResignAction.RESIGN_ACTION_UNCONDITIONALLY_DIVEST_ATTRIBUTES)
    if isinstance(action, int):
        return action
    wire_name = _RESIGN_ACTION_WIRE_NAMES.get(action)
    if wire_name is None:
        valid = ", ".join(sorted(_RESIGN_ACTION_WIRE_NAMES))
        raise ValueError(f"invalid resign action {action!r}; expected one of: {valid}")
    return int(common_pb2.ResignAction.Value(wire_name))


# --- Real-gRPC transport ----------------------------------------------------


class GrpcTransport:
    """Real-gRPC client wired against the generated stubs in ``_generated/``.

    Constructed by ``RtiConnection.__aenter__`` when the URL scheme is
    ``grpc``. Maintains:

      - A per-class-name → numeric handle table built lazily from the
        FederationSpec.fom_modules on the first ``create_federation``
        call (so name→handle resolution matches the rtid side).
      - Per-federate event queues populated by a background asyncio task
        that drains the StreamService.Events server stream.
      - A monotonic handle allocator scoped to this transport instance,
        used for object handles when the wire returns one (federate
        handles are allocated by the server and pulled from
        JoinFederationResponse).
    """

    _restore_failure: BaseException | None

    def __init__(self, channel: grpc.aio.Channel, url: str) -> None:
        # Lazy import keeps grpc out of the import surface for spec tests
        # that only use memory:// transports. Generated stubs are
        # namespaced as ``rti.v1`` and live under
        # rti1516e/_generated/rti/v1/; ensure_generated_path puts that
        # directory on sys.path so the import resolves.
        _ensure_generated_path()
        from rti.v1 import (
            declaration_pb2_grpc,
            federation_pb2_grpc,
            object_pb2_grpc,
            stream_pb2_grpc,
            time_pb2_grpc,
        )

        self.channel = channel
        self.url = url
        self.federation = federation_pb2_grpc.FederationServiceStub(channel)
        self.declaration = declaration_pb2_grpc.DeclarationServiceStub(channel)
        self.objects = object_pb2_grpc.ObjectServiceStub(channel)
        self.streams = stream_pb2_grpc.StreamServiceStub(channel)
        # Time-management operations use the generated TimeService stub.
        self.time = time_pb2_grpc.TimeServiceStub(channel)
        # Federation name → name-resolver state. The first
        # ``create_federation`` for a given federation name parses the
        # FOM and caches name→handle maps; later RPCs reuse them.
        self._federation_name: str | None = None
        self._generation_by_federation: dict[str, int] = {}
        self._stale_federation_generations: set[str] = set()
        self._logical_time_by_federation: dict[str, str] = {}
        self._interaction_handles: dict[str, int] = {}
        self._object_class_handles: dict[str, int] = {}
        self._inverse_interaction_handles: dict[int, str] = {}
        # M12 W2: lazily-populated cache for attribute name → handle
        # lookups; keyed by class name. Populated on first
        # publish_object_class / subscribe_object_class call by
        # walking the cached FOM (see _populate_handle_tables for the
        # FOM cache).
        self._attribute_handle_cache: dict[str, dict[str, int]] = {}
        # The parsed FOM is cached so attribute lookups don't need to
        # re-parse. Set on every successful _populate_handle_tables.
        self._fom_cache: Any | None = None
        self._mim_module: Any | None = None
        self._requested_mims: dict[str, Any] = {}
        # Per-federate event queues + the background draining tasks.
        self._event_queues: dict[int, asyncio.Queue[Any]] = {}
        self._event_sinks: dict[int, Callable[[Any], object]] = {}
        self._stream_tasks: dict[int, asyncio.Task[None]] = {}
        self._session_tokens: dict[int, bytes] = {}
        self._callback_entry_supported: dict[int, bool] = {}
        self._pending_join_cleanups: dict[tuple[str, int], bytes] = {}
        self._identity_bindings: dict[int, Callable[[int], None]] = {}
        self._callback_epochs: dict[int, int] = {}
        self._ownership_callbacks_by_federate: dict[int, Any] = {}
        self._retraction_hints: dict[int, int | None] = {}
        self._logical_times_by_federate: dict[int, int | float | None] = {}
        self._used_retractions: dict[int, set[int]] = {}
        self._staged_sdk_restores: dict[int, tuple[str, Any]] = {}
        self._sdk_restore_ready: dict[int, bool] = {}
        self._checkpoint_sources: dict[int, Callable[[], list[Any]]] = {}
        self._callback_reports_pending: dict[int, set[bytes]] = {}
        self._generic_callbacks_by_federate: dict[int, Any] = {}
        self._callback_ticket_identities: dict[int, dict[bytes, bytes]] = {}
        self._restored_local_notifications: dict[int, list[Any]] = {}
        # Object handle allocator (used only as a local fallback; the
        # registry response carries the canonical handle when present).
        self._next_handle = 1

    # --- Surface consumed by RtiConnection / Federate -----------------------

    def callback_invocation_entry_supported(self, federate_handle: int) -> bool:
        supported: dict[int, bool] = getattr(self, "_callback_entry_supported", {})
        return supported.get(federate_handle, False)

    def logical_time_implementation_name(self, federation_name: str | None = None) -> str:
        """Last confirmed selection; absent legacy metadata means Float64."""
        from ._logical_time import FLOAT64_TIME

        selections: dict[str, str] = getattr(self, "_logical_time_by_federation", {})
        return selections.get(
            federation_name or getattr(self, "_federation_name", None) or "", FLOAT64_TIME
        )

    def _write_time(self, message: Any, field: str, value: Any) -> None:
        from ._logical_time import write_time

        write_time(message, field, value, self.logical_time_implementation_name())

    @overload
    def _read_time(
        self, message: Any, field: str, *, optional: Literal[False] = False
    ) -> int | float: ...

    @overload
    def _read_time(self, message: Any, field: str, *, optional: bool) -> int | float | None: ...

    def _read_time(self, message: Any, field: str, *, optional: bool = False) -> int | float | None:
        from ._logical_time import read_time

        return read_time(message, field, self.logical_time_implementation_name(), optional=optional)

    def allocate_handle(self) -> int:
        """Mint a fresh local handle. Used as a fallback if the server
        response did not include one (it always should for register_object,
        but the SDK contract permits this fallback)."""
        h = self._next_handle
        self._next_handle += 1
        return h

    def events_for(self, federate_handle: int) -> asyncio.Queue[Any]:
        """Return the asyncio.Queue draining events for ``federate_handle``.

        Setdefault so the SDK can call this before the stream task has
        produced anything; the background task pushes into the same queue.
        """
        return self._event_queues.setdefault(federate_handle, asyncio.Queue())

    def set_event_sink(self, federate_handle: int, sink: Callable[[Any], object] | None) -> None:
        """Install a same-loop event sink, or restore queue delivery."""
        if sink is None:
            self._event_sinks.pop(federate_handle, None)
        else:
            self._event_sinks[federate_handle] = sink
            # Joining starts the stream before the ambassador installs its
            # direct sink. Transfer that backlog before any later callback.
            queue = self._event_queues.get(federate_handle)
            while queue is not None and not queue.empty():
                event = queue.get_nowait()
                serial = getattr(self, "_restore_serial", 0)
                if getattr(event, "_restore_serial", serial) != serial:
                    continue
                sink(event)

    def bind_federate_identity(self, handle: int, update: Callable[[int], None]) -> None:
        self._identity_bindings[handle] = update

    def restored_retraction_hint(self, handle: int) -> int | None:
        return getattr(self, "_retraction_hints", {}).get(handle)

    def sdk_checkpoint(self, handle: int) -> bytes:
        from ._generic_callbacks import GenericCallbackState
        from ._local_notifications import kind_of, to_row
        from ._runtime_checkpoint import FORMAT, callback_dict, encode, time_dict

        generic = self._generic_callbacks_by_federate.setdefault(handle, GenericCallbackState())
        outcomes = generic.checkpoint_outcomes()
        local_state = getattr(self, "_local_notification_states", {}).get(handle)
        if local_state is not None:
            local_state.checkpoint()
        if any(
            not generic.has_outcome(receipt)
            for receipt in getattr(self, "_callback_reports_pending", {}).get(handle, ())
        ):
            raise RuntimeError("cannot checkpoint unresolved callback invocation outcomes")
        state = getattr(self, "_ownership_callbacks_by_federate", {}).get(handle)
        knowledge: dict[str, list[list[int]]] = {"latest": [], "retired": []}
        if state is not None:
            with state._lock:
                if state._pending or state.active_task is not None:
                    raise RuntimeError("cannot checkpoint unresolved ownership callbacks")
                knowledge = {
                    "latest": [list(row) for row in sorted(state._latest.items())],
                    "retired": [list(row) for row in sorted(state._retired.items())],
                }
        queue = self._event_queues.get(handle)
        # Snapshot without consuming callbacks; asyncio's stubs omit the backing deque.
        pending = list(cast(_QueueStorage, queue)._queue) if queue is not None else []
        source = getattr(self, "_checkpoint_sources", {}).get(handle)
        if source is not None:
            pending = source() + pending
        family = self.logical_time_implementation_name()
        local = [
            to_row(event)
            for event in pending
            if kind_of(event) is not None and getattr(event, "_wire_carrier", None) is None
        ]
        runtime = {
            "format": FORMAT,
            "version": 1,
            "federation_generation": self._generation_by_federation[
                cast(str, self._federation_name)
            ],
            "federate_handle": handle,
            "callback_epoch": self._callback_epochs.get(handle, 0),
            "logical_time_implementation_name": family,
            "logical_time": time_dict(
                getattr(self, "_logical_times_by_federate", {}).get(handle), family
            ),
            "object_knowledge": knowledge,
            "next_retraction_handle": self.restored_retraction_hint(handle),
            "used_retraction_handles": sorted(
                getattr(self, "_used_retractions", {}).get(handle, set())
            ),
            "pending_callbacks": [
                callback_dict(event)
                for event in pending
                if kind_of(event) is None or getattr(event, "_wire_carrier", None) is not None
            ],
            "callback_recovery": "server-v2",
            "pending_outcomes": outcomes,
        }
        if local:
            runtime["pending_local_notifications"] = local
        return encode(runtime)

    def _decode_carrier(self, handle: int, wire: Any) -> Any:
        import hashlib

        from rti.v1 import stream_pb2

        from ._generic_callbacks import CompletedCallbackRecovery
        from ._runtime_checkpoint import validate_recovery_carrier
        from .events import TimeAdvanceGrant, TimeConstrainedEnabled, TimeRegulationEnabled

        recovery = validate_recovery_carrier(wire)
        receipt = bytes(wire.callback_receipt)
        identity = (
            bytes(recovery.invocation_identity)
            if recovery is not None
            else hashlib.sha256(receipt).digest()
        )
        outcome_only = recovery is not None and recovery.outcome_only
        completed = recovery is not None and recovery.completed
        event = CompletedCallbackRecovery(identity) if completed else self._translate_event(wire)
        if event is None:
            if outcome_only:
                raise ValueError("callback recovery carrier has no supported body")
            return None
        carrier = stream_pb2.FederateEvent()
        carrier.CopyFrom(wire)
        object.__setattr__(event, "_wire_carrier", carrier)
        if wire.callback_receipt:
            object.__setattr__(event, "_callback_receipt", bytes(wire.callback_receipt))
            if recovery is not None:
                self._callback_ticket_identities.setdefault(handle, {})[receipt] = identity
        object.__setattr__(event, "_callback_invocation_identity", identity)
        object.__setattr__(event, "_callback_outcome_only", bool(outcome_only))
        object.__setattr__(event, "_callback_completed", bool(completed))
        object.__setattr__(event, "_callback_recovered", recovery is not None)
        if wire.HasField("ownership_receipt"):
            object.__setattr__(event, "_ownership_receipt", carrier.ownership_receipt)
        object.__setattr__(event, "_object_knowledge_epoch", int(wire.object_knowledge_epoch))
        if not outcome_only and isinstance(
            event, (TimeAdvanceGrant, TimeConstrainedEnabled, TimeRegulationEnabled)
        ):
            times = getattr(self, "_logical_times_by_federate", None)
            if times is None:
                times = self._logical_times_by_federate = {}
            times[handle] = event.time
        return event

    def guard_service(self) -> None:
        if getattr(self, "_restore_failure", None) is not None:
            raise RuntimeError(
                "restore transition failed; reconnect is required"
            ) from self._restore_failure
        if getattr(self, "_restore_applying", False):
            raise RuntimeError("restore transition is applying")

    def guard_restore_complete(self, handle: int) -> None:
        self.guard_service()
        if getattr(self, "_sdk_restore_ready", {}).get(handle) is not True:
            raise RuntimeError("no validated SDK restore stage is ready")

    async def _apply_restore_terminal(self, old_handle: int, event: Any) -> int:
        from .events import FederationNotRestored, FederationRestored, InitiateFederateRestore

        if getattr(event, "_callback_recovered", False) or getattr(
            event, "_callback_outcome_only", False
        ):
            return old_handle
        staged = getattr(self, "_staged_sdk_restores", None)
        if staged is None:
            staged = self._staged_sdk_restores = {}
        ready = getattr(self, "_sdk_restore_ready", None)
        if ready is None:
            ready = self._sdk_restore_ready = {}
        if isinstance(event, InitiateFederateRestore):
            from ._runtime_checkpoint import decode

            ready[old_handle] = False
            staged.pop(old_handle, None)
            runtime = (
                decode(event.client_state, self.logical_time_implementation_name())
                if event.client_state
                else None
            )
            if runtime is not None:
                if (
                    runtime["federation_generation"]
                    != self._generation_by_federation[cast(str, self._federation_name)]
                ):
                    raise ValueError("restored SDK state belongs to another execution")
                for wire in runtime["decoded_callbacks"]:
                    completed = (
                        "callback_recovery" in wire.DESCRIPTOR.fields_by_name
                        and wire.HasField("callback_recovery")
                        and wire.callback_recovery.completed
                    )
                    if not completed and self._translate_event(wire) is None:
                        raise ValueError("restored SDK callback is unsupported")
            staged[old_handle] = (event.label, runtime)
            ready[old_handle] = True
            return old_handle
        if isinstance(event, FederationNotRestored):
            ready.pop(old_handle, None)
            staged.pop(old_handle, None)
            return old_handle

        if not isinstance(event, FederationRestored):
            return old_handle
        ready.pop(old_handle, None)
        if event.federate_handle is None and event.callback_epoch is None:
            if staged.get(old_handle, (None, None))[1] is not None:
                raise ValueError("nonempty SDK restore requires terminal identity")
            staged.pop(old_handle, None)
            return old_handle
        entry = staged.get(old_handle)
        runtime = entry[1] if entry is not None else None
        if entry is not None and entry[0] != event.label:
            raise ValueError("restored SDK checkpoint label differs")
        if runtime is not None and runtime["federate_handle"] != event.federate_handle:
            raise ValueError("restored SDK checkpoint identity or epoch differs")
        self._restore_applying = True
        self._restore_serial = getattr(self, "_restore_serial", 0) + 1
        handle, epoch = int(event.federate_handle or 0), event.callback_epoch
        if handle <= 0 or epoch is None or epoch <= self._callback_epochs.get(old_handle, 0):
            error = RuntimeError("invalid restore callback identity or epoch")
            self._restore_failure = error
            raise error
        queue = self._event_queues.get(old_handle)
        if queue is not None:
            while not queue.empty():
                queue.get_nowait()
        try:
            await self.refresh_fom(handle)
        except BaseException as exc:
            self._restore_failure = exc
            raise
        queue = self._event_queues.pop(old_handle, None)
        if queue is not None:
            self._event_queues[handle] = queue

        def rebind(mapping: dict[int, _ValueT]) -> None:
            value = mapping.pop(old_handle, None)
            if value is not None:
                mapping[handle] = value

        event_sinks = self._event_sinks
        stream_tasks = self._stream_tasks
        identity_bindings = self._identity_bindings
        session_tokens: dict[int, bytes] = getattr(self, "_session_tokens", {})
        entry_supported: dict[int, bool] = getattr(self, "_callback_entry_supported", {})
        checkpoint_sources: dict[int, Callable[[], list[Any]]] = getattr(
            self, "_checkpoint_sources", {}
        )
        rebind(event_sinks)
        rebind(stream_tasks)
        rebind(identity_bindings)
        rebind(session_tokens)
        rebind(entry_supported)
        rebind(checkpoint_sources)
        self._callback_epochs.pop(old_handle, None)
        self._callback_epochs[handle] = int(epoch)
        states = getattr(self, "_ownership_callbacks_by_federate", {})
        state = states.pop(old_handle, None)
        if state is not None:
            states[handle] = state
            state.reset_knowledge()
            if state.enabled:
                state.restore_scope(
                    self._generation_by_federation[cast(str, self._federation_name)], int(epoch)
                )
            if runtime is not None:
                with state._lock:
                    state._latest = dict(runtime["object_knowledge"]["latest"])
                    state._retired = dict(runtime["object_knowledge"]["retired"])
        restored_time: int | float | None = runtime["decoded_time"] if runtime else None
        restored_retractions: set[int] = (
            set(runtime["used_retraction_handles"]) if runtime else set()
        )
        times: dict[int, int | float | None] | None = getattr(
            self, "_logical_times_by_federate", None
        )
        if times is None:
            times = self._logical_times_by_federate = {}
        times.pop(old_handle, None)
        times[handle] = restored_time
        used_retractions: dict[int, set[int]] | None = getattr(self, "_used_retractions", None)
        if used_retractions is None:
            used_retractions = self._used_retractions = {}
        used_retractions.pop(old_handle, None)
        used_retractions[handle] = restored_retractions
        hints = getattr(self, "_retraction_hints", None)
        if hints is None:
            hints = self._retraction_hints = {}
        hints.pop(old_handle, None)
        hints[handle] = event.next_retraction_handle
        getattr(self, "_callback_reports_pending", {}).pop(old_handle, None)
        from ._generic_callbacks import GenericCallbackState

        generic = GenericCallbackState()
        generic.restore_outcomes(runtime.get("pending_outcomes", []) if runtime else [])
        generics = getattr(self, "_generic_callbacks_by_federate", None)
        if generics is None:
            generics = self._generic_callbacks_by_federate = {}
        generics.pop(old_handle, None)
        generics[handle] = generic
        getattr(self, "_local_notification_states", {}).pop(old_handle, None)
        locals_by_handle = getattr(self, "_restored_local_notifications", None)
        if locals_by_handle is None:
            locals_by_handle = self._restored_local_notifications = {}
        locals_by_handle.pop(old_handle, None)
        locals_by_handle[handle] = runtime.get("decoded_local_notifications", []) if runtime else []
        getattr(self, "_callback_ticket_identities", {}).pop(old_handle, None)
        staged.pop(old_handle, None)
        update = self._identity_bindings.get(handle)
        if update is not None:
            update(handle)
        self._restore_applying = False
        return handle

    def _deliver_event(self, federate_handle: int, event: Any) -> None:
        from .events import FederationRestored

        if hasattr(event, "_local_notification_id"):
            object.__setattr__(event, "_restore_serial", getattr(self, "_restore_serial", 0))
        sink = self._event_sinks.get(federate_handle)
        if sink is None:
            value = event
            if federate_handle in getattr(self, "_identity_bindings", {}) and not isinstance(
                event, BaseException
            ):
                object.__setattr__(event, "_restore_serial", getattr(self, "_restore_serial", 0))
            self.events_for(federate_handle).put_nowait(value)
        else:
            sink(event)
        if (
            isinstance(event, FederationRestored)
            and event.federate_handle is not None
            and not getattr(event, "_callback_recovered", False)
        ):
            for local in getattr(self, "_restored_local_notifications", {}).pop(
                federate_handle, []
            ):
                self._deliver_event(federate_handle, local)

    async def close(self) -> None:
        """Tear down: cancel every stream task, then close the channel."""
        # Known failed-join memberships must be retired before their channel is
        # closed. A failed retry leaves both the capability and transport usable.
        pending = getattr(self, "_pending_join_cleanups", {})
        for (federation_name, handle), token in list(pending.items()):
            await self._resign_federation(
                handle,
                federation_name=federation_name,
                session_token=token,
            )
            del pending[(federation_name, handle)]
        failures: list[BaseException] = []
        for task in list(self._stream_tasks.values()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError as exc:
                current = asyncio.current_task()
                if not task.cancelled() or (current is not None and current.cancelling()):
                    failures.append(exc)
            except BaseException as exc:
                failures.append(exc)
        self._stream_tasks.clear()
        self._event_sinks.clear()
        getattr(self, "_identity_bindings", {}).clear()
        getattr(self, "_callback_epochs", {}).clear()
        getattr(self, "_ownership_callbacks_by_federate", {}).clear()
        getattr(self, "_generic_callbacks_by_federate", {}).clear()
        getattr(self, "_callback_ticket_identities", {}).clear()
        getattr(self, "_callback_reports_pending", {}).clear()
        getattr(self, "_checkpoint_sources", {}).clear()
        getattr(self, "_restored_local_notifications", {}).clear()
        getattr(self, "_local_notification_states", {}).clear()
        getattr(self, "_sdk_restore_ready", {}).clear()
        getattr(self, "_staged_sdk_restores", {}).clear()
        getattr(self, "_retraction_hints", {}).clear()
        # RtiConnection retains this transport until close returns.  Do not
        # turn a channel failure (including cancellation) into false success;
        # the owner must be able to retry or report the incomplete boundary.
        try:
            await self.channel.close()
        except BaseException as exc:
            failures.append(exc)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "gRPC transport stream/channel cleanup failed",
                failures,
            )

    async def record(self, method: str, **kwargs: Any) -> Any:  # noqa: PLR0911, PLR0912
        """Dispatch a single SDK call to the matching gRPC RPC.

        The SDK's higher layers call ``transport.record(method, **kwargs)``
        and may ``await`` the result. The fake returns synchronously; this
        method is async because every gRPC call is awaitable. Both forms
        coexist because ``Federate.publish_object_class`` (and friends)
        treat the result as opaque.
        """
        if method == "create_federation":
            return await self._create_federation(
                kwargs["spec"],
                exist_ok=kwargs.get("exist_ok", True),
            )
        if method == "destroy_federation":
            return await self._destroy_federation(kwargs["federation_name"])
        if method == "join_federation":
            return await self._join_federation(
                kwargs["spec"],
                kwargs["federate_name"],
                kwargs.get("federate_type", ""),
            )
        if method == "resign_federation":
            return await self._resign_federation(
                kwargs["federate_handle"],
                kwargs.get("action"),
            )
        if method == "publish_interaction_class":
            return await self._publish_interaction(kwargs["federate_handle"], kwargs["class_name"])
        if method == "subscribe_interaction_class":
            return await self._subscribe_interaction(
                kwargs["federate_handle"],
                kwargs["class_name"],
                active=kwargs.get("active", True),
            )
        if method == "publish_object_class":
            return await self._publish_object_class(
                kwargs["federate_handle"],
                kwargs["class_name"],
                kwargs["attributes"],
            )
        if method == "subscribe_object_class":
            return await self._subscribe_object_class(
                kwargs["federate_handle"],
                kwargs["class_name"],
                kwargs["attributes"],
                active=kwargs.get("active", True),
                update_rate_designator=kwargs.get("update_rate_designator", ""),
            )
        if method == "set_advisory_switch":
            return await self._set_advisory_switch(
                kwargs["federate_handle"], kwargs["switch_kind"], kwargs["enabled"]
            )
        if method == "register_object_instance":
            return await self._register_object_instance(
                kwargs["federate_handle"],
                kwargs["class_name"],
                kwargs.get("instance_name"),
            )
        if method == "update_attributes":
            return await self._update_attributes(
                kwargs["federate_handle"],
                kwargs["object_handle"],
                kwargs["values"],
                kwargs.get("timestamp"),
                tag=kwargs.get("tag", b""),
                retraction_handle=kwargs.get("retraction_handle", 0),
            )
        if method == "send_interaction":
            return await self._send_interaction(
                kwargs["federate_handle"],
                kwargs["class_name"],
                kwargs.get("parameters") or {},
                kwargs.get("timestamp"),
                tag=kwargs.get("tag", b""),
                retraction_handle=kwargs.get("retraction_handle", 0),
            )
        if method == "enable_time_regulation":
            return await self._enable_time_regulation(
                kwargs["federate_handle"],
                kwargs["lookahead"],
            )
        if method == "enable_time_constrained":
            return await self._enable_time_constrained(
                kwargs["federate_handle"],
            )
        if method == "next_message_request":
            return await self._next_message_request(
                kwargs["federate_handle"],
                kwargs["time"],
            )
        if method == "disable_time_regulation":
            return await self._disable_time_regulation(kwargs["federate_handle"])
        if method == "disable_time_constrained":
            return await self._disable_time_constrained(kwargs["federate_handle"])
        if method == "modify_lookahead":
            return await self._modify_lookahead(
                kwargs["federate_handle"],
                kwargs["lookahead"],
            )
        if method == "next_message_request_available":
            return await self._next_message_request_available(
                kwargs["federate_handle"],
                kwargs["time"],
            )
        if method == "time_advance_request":
            return await self._time_advance_request(
                kwargs["federate_handle"],
                kwargs["time"],
            )
        if method == "time_advance_request_available":
            return await self._time_advance_request_available(
                kwargs["federate_handle"],
                kwargs["time"],
            )
        if method == "flush_queue_request":
            return await self._flush_queue_request(
                kwargs["federate_handle"],
                kwargs["time"],
            )
        if method == "query_logical_time":
            return await self._query_logical_time(kwargs["federate_handle"])
        if method == "query_lookahead":
            return await self._query_lookahead(kwargs["federate_handle"])
        if method == "query_lbts":
            return await self._query_lbts()
        if method == "enable_asynchronous_delivery":
            return await self._enable_asynchronous_delivery(kwargs["federate_handle"])
        if method == "disable_asynchronous_delivery":
            return await self._disable_asynchronous_delivery(kwargs["federate_handle"])
        if method == "delete_object_instance":
            return await self._delete_object_instance(
                kwargs["federate_handle"],
                kwargs["object_handle"],
                kwargs.get("tag") or b"",
                kwargs.get("timestamp"),
                retraction_handle=kwargs.get("retraction_handle", 0),
            )
        if method == "local_delete_object_instance":
            return await self._local_delete_object_instance(
                kwargs["federate_handle"],
                kwargs["object_handle"],
            )
        if method == "request_attribute_value_update":
            return await self._request_attribute_value_update(
                kwargs["federate_handle"],
                kwargs["object_handle"],
                list(kwargs.get("attribute_handles") or []),
                kwargs.get("tag") or b"",
            )
        if method == "request_class_attribute_value_update":
            return await self._request_class_attribute_value_update(
                kwargs["federate_handle"],
                kwargs["object_class_handle"],
                list(kwargs.get("attribute_handles") or []),
                kwargs.get("tag") or b"",
            )
        if method == "change_attribute_transportation_type":
            return await self._change_attribute_transportation_type(
                kwargs["federate_handle"],
                kwargs["object_handle"],
                list(kwargs.get("attribute_handles") or []),
                int(kwargs["transport_type"]),
            )
        if method == "change_interaction_transportation_type":
            return await self._change_interaction_transportation_type(
                kwargs["federate_handle"],
                kwargs["interaction_class_handle"],
                int(kwargs["transport_type"]),
            )
        if method == "retract":
            from rti.v1 import object_pb2

            from ._grpc_errors import translate_rpc_error

            try:
                await self.objects.Retract(
                    object_pb2.RetractRequest(
                        wire_version=1,
                        federation_name=self._federation_name or "",
                        federate_handle=kwargs["federate_handle"],
                        message_retraction_handle=int(kwargs["retraction_handle"]),
                    )
                )
            except Exception as exc:
                translate_rpc_error(exc)
            return None
        if method in ("change_attribute_order_type", "change_interaction_order_type"):
            from rti.v1 import object_pb2

            from ._grpc_errors import translate_rpc_error

            common = {
                "wire_version": 1,
                "federation_name": self._federation_name or "",
                "federate_handle": kwargs["federate_handle"],
                "order_type": int(kwargs["order_type"]),
            }
            try:
                if method == "change_attribute_order_type":
                    await self.objects.ChangeAttributeOrderType(
                        object_pb2.ChangeAttributeOrderRequest(
                            **common,
                            object_handle=int(kwargs["object_handle"]),
                            attribute_handles=[int(h) for h in kwargs["attribute_handles"]],
                        )
                    )
                else:
                    await self.objects.ChangeInteractionOrderType(
                        object_pb2.ChangeInteractionOrderRequest(
                            **common,
                            interaction_class_handle=int(kwargs["interaction_class_handle"]),
                        )
                    )
            except Exception as exc:
                translate_rpc_error(exc)
            return None
        # Unknown method — surface a clear error rather than a silent
        # drop; better the test fails loudly than passes by omission.
        raise NotImplementedError(
            f"GrpcTransport.record: method {method!r} not implemented for cut-1"
        )

    # --- Per-RPC dispatch helpers ------------------------------------------

    async def _resolve_federation_generation(self, federation_name: str) -> int:
        cached = self._generation_by_federation.get(federation_name)
        stale: set[str] = getattr(self, "_stale_federation_generations", set())
        if cached is not None and federation_name not in stale:
            return cached
        from rti.v1 import common_pb2, federation_pb2

        response = await self.federation.ListFederations(
            federation_pb2.ListFederationsRequest(
                wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            )
        )
        for summary in response.federations:
            if summary.name == federation_name:
                generation = int(summary.federation_generation)
                if cached is not None and generation != cached:
                    getattr(self, "_logical_time_by_federation", {}).pop(federation_name, None)
                    getattr(self, "_requested_mims", {}).pop(federation_name, None)
                self._generation_by_federation[federation_name] = generation
                stale.discard(federation_name)
                return generation
        from .errors import FederationExecutionDoesNotExist

        raise FederationExecutionDoesNotExist(f"federation {federation_name!r} does not exist")

    async def _create_federation(self, spec: Any, *, exist_ok: bool = True) -> None:
        """CreateFederation. ``exist_ok=True`` (the rolled create-on-join
        path) swallows FederationAlreadyExists so the second federate to
        create the same federation succeeds silently; ``exist_ok=False``
        (§4.5 createFederationExecution, M39 HA-2) surfaces it as the
        typed ``FederationExecutionAlreadyExists``. Every other failure
        is translated either way.
        """
        from rti.v1 import common_pb2, federation_pb2

        from ._grpc_errors import translate_rpc_error
        from ._logical_time import logical_time_name, require_supported_logical_time
        from .errors import FederationCreationResponseError
        from .fom.modules import load_mim_module, module_paths

        requested_time = getattr(spec, "logical_time_implementation_name", "")
        expected_time = require_supported_logical_time(requested_time)
        paths = module_paths(spec.fom_modules, "fom_modules")
        selected_mim = load_mim_module(getattr(spec, "mim_module", None))
        # Build FOMModule list + cache the name→handle maps (Python-side
        # FOM parser; same sort order as Go-side). The file read happens
        # in a sync helper so this async coroutine doesn't trip the
        # ASYNC240 lint (sync filesystem in async context); FOM modules
        # are tiny and load-once so the blocking is not material.
        fom_modules = [
            common_pb2.FOMModule(path=str(path), xml=_read_fom_bytes(path)) for path in paths
        ]

        req = federation_pb2.CreateFederationRequest(
            wire_version=(
                common_pb2.WireVersion.WIRE_VERSION_V2
                if selected_mim is not None
                else common_pb2.WireVersion.WIRE_VERSION_V1
            ),
            federation_name=spec.name,
            fom_modules=fom_modules,
            mode=_mode_to_proto(spec.mode),
            stall_timeout_seconds=float(spec.stall_timeout_seconds),
            seed=spec.seed,
            logical_time_implementation_name=requested_time,
        )
        if selected_mim is not None:
            req.mim_module.CopyFrom(
                common_pb2.FOMModule(path=selected_mim.designator, xml=selected_mim.xml)
            )
        try:
            response = await self.federation.CreateFederation(req)
        except Exception as exc:  # noqa: BLE001 — translate_rpc_error reraises
            # FederationAlreadyExists is benign on the rolled path — the
            # second federate to create the same federation should
            # succeed silently.
            if exist_ok and _is_already_exists(exc):
                await self._resolve_federation_generation(spec.name)
                self._federation_name = spec.name
                self._populate_handle_tables(paths, mim_module=selected_mim)
                return
            translate_rpc_error(exc)
        generation = int(response.federation_generation)
        self._generation_by_federation[req.federation_name] = generation
        getattr(self, "_stale_federation_generations", set()).discard(req.federation_name)
        selected_time = logical_time_name(getattr(response, "logical_time_implementation_name", ""))
        if selected_time != expected_time:
            raise FederationCreationResponseError(
                req.federation_name,
                generation,
                int(getattr(response, "effective_seed", 0)),
                selected_time,
            )
        self._federation_name = spec.name
        self._populate_handle_tables(paths, mim_module=selected_mim)
        if not hasattr(self, "_logical_time_by_federation"):
            self._logical_time_by_federation = {}
        self._logical_time_by_federation[spec.name] = selected_time
        if not hasattr(self, "_requested_mims"):
            self._requested_mims = {}
        self._requested_mims[spec.name] = selected_mim
        return

    async def _destroy_federation(self, federation_name: str) -> None:
        """§4.6 destroyFederationExecution (M39 HA-2). Typed failures:
        FederatesCurrentlyJoined while members remain,
        FederationExecutionDoesNotExist for an unknown name."""
        from rti.v1 import common_pb2, federation_pb2

        from ._grpc_errors import translate_rpc_error

        generation = await self._resolve_federation_generation(federation_name)
        req = federation_pb2.DestroyFederationRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=federation_name,
            expected_federation_generation=generation,
        )
        try:
            await self.federation.DestroyFederation(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
        self._generation_by_federation.pop(federation_name, None)
        getattr(self, "_stale_federation_generations", set()).discard(federation_name)
        getattr(self, "_logical_time_by_federation", {}).pop(federation_name, None)
        getattr(self, "_requested_mims", {}).pop(federation_name, None)

    async def _join_federation(self, spec: Any, federate_name: str, federate_type: str = "") -> int:
        from rti.v1 import common_pb2, federation_pb2

        from ._grpc_errors import translate_rpc_error
        from ._logical_time import require_supported_logical_time
        from .fom.modules import load_mim_module, module_paths

        require_supported_logical_time(getattr(spec, "logical_time_implementation_name", ""))
        generation = await self._resolve_federation_generation(spec.name)
        selected_mim = load_mim_module(getattr(spec, "mim_module", None))
        if selected_mim is None:
            selected_mim = getattr(self, "_requested_mims", {}).get(spec.name)
        self._federation_name = spec.name
        # Make sure the handle tables are populated even if the caller
        # joined an already-existing federation (no create call here).
        if not self._interaction_handles and spec.fom_modules:
            self._populate_handle_tables(spec.fom_modules, mim_module=selected_mim)

        # M13 thread B (docs/srs.md §10.4): forward the optional
        # federate_type. Old SDK callers that don't pass it land here
        # with an empty string — the rtid treats absent as "no type
        # declared", preserving cut-1 wire-version compatibility.
        additional = [
            common_pb2.FOMModule(path=str(path), xml=_read_fom_bytes(path))
            for path in module_paths(
                getattr(spec, "additional_fom_modules", []), "additional_fom_modules"
            )
        ]
        req = federation_pb2.JoinFederationRequest(
            wire_version=(
                common_pb2.WireVersion.WIRE_VERSION_V2
                if selected_mim is not None
                else common_pb2.WireVersion.WIRE_VERSION_V1
            ),
            federation_name=spec.name,
            federate_name=federate_name,
            federate_type=federate_type,
            expected_federation_generation=generation,
            additional_fom_modules=additional,
        )
        if selected_mim is not None:
            req.expected_mim_module.CopyFrom(
                common_pb2.FOMModule(path=selected_mim.designator, xml=selected_mim.xml)
            )
        try:
            host = os.environ.get("GORTI_FEDERATE_HOST", "")
            if host:
                resp = await self.federation.JoinFederation(
                    req, metadata=(("gorti-federate-host-bin", host.encode("utf-8")),)
                )
            else:
                resp = await self.federation.JoinFederation(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise  # unreachable; translate_rpc_error always raises
        federate_handle = int(resp.federate_handle)
        if not hasattr(self, "_session_tokens"):
            self._session_tokens = {}
        self._session_tokens[federate_handle] = bytes(getattr(resp, "session_token", b""))
        try:
            selected_time = require_supported_logical_time(
                getattr(resp, "logical_time_implementation_name", "")
            )
            expected_time = getattr(self, "_logical_time_by_federation", {}).get(spec.name)
            requested_time = getattr(spec, "logical_time_implementation_name", "")
            if expected_time is None and requested_time:
                expected_time = require_supported_logical_time(requested_time)
            if expected_time is not None and selected_time != expected_time:
                raise RuntimeError(
                    "server join response changed the confirmed logical time implementation"
                )
            if getattr(resp, "HasField", lambda _: False)("fom_view"):
                if selected_mim is not None and (
                    not resp.fom_view.HasField("mim_module")
                    or resp.fom_view.mim_module.path != selected_mim.designator
                    or resp.fom_view.mim_module.xml != selected_mim.xml
                ):
                    raise RuntimeError("server did not confirm the selected MIM")
                self._install_fom_view(resp.fom_view, replace=True)
            elif selected_mim is not None:
                raise RuntimeError("server did not confirm the selected MIM")
            elif additional:
                raise RuntimeError("server did not confirm additional FOM bindings")
        except Exception as join_error:
            identity = (req.federation_name, federate_handle)
            self._pending_join_cleanups[identity] = self.session_token(federate_handle)
            try:
                await self._resign_federation(
                    federate_handle,
                    federation_name=identity[0],
                    session_token=self._pending_join_cleanups[identity],
                )
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "join validation and authenticated cleanup failed",
                    [join_error, cleanup_error],
                ) from join_error
            del self._pending_join_cleanups[identity]
            raise
        if not hasattr(self, "_logical_time_by_federation"):
            self._logical_time_by_federation = {}
        self._logical_time_by_federation[spec.name] = selected_time
        # Open the per-federate event stream as soon as we know the
        # handle; the background task pushes events into the queue
        # ``events_for(federate_handle)`` returns.
        self._callback_entry_supported[federate_handle] = bool(
            getattr(resp, "callback_invocation_entry_supported", False)
        )
        self._start_event_stream(federate_handle)
        return federate_handle

    async def _resign_federation(
        self,
        federate_handle: int,
        action: int | str | None = None,
        *,
        federation_name: str | None = None,
        session_token: bytes | None = None,
    ) -> None:
        from rti.v1 import common_pb2, federation_pb2

        federation_name = self._federation_name if federation_name is None else federation_name
        if federation_name is None:
            return
        # M24 W2 — caller may pass an explicit ResignAction value.
        # M36 — Layer 2 passes the IEEE §4.10 designator string; map it
        # here so the higher layers stay proto-free.
        # Default = UNCONDITIONALLY_DIVEST_ATTRIBUTES (matches pre-M24).
        action = resign_action_to_proto(action)
        req = federation_pb2.ResignFederationRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=federation_name,
            federate_handle=federate_handle,
            action=action,
        )
        # The server may close the stream before the successful RPC response
        # arrives. Defer only that terminal signal, keeping callbacks usable
        # throughout a rejected (or cancelled) resignation.
        token = self.session_token(federate_handle) if session_token is None else session_token
        owns_session = self.session_token(federate_handle) == token
        task = self._stream_tasks.get(federate_handle) if owns_session else None
        generation = getattr(self, "_generation_by_federation", {}).get(federation_name)
        sinks = self._event_sinks if task is not None else None
        sink = sinks.get(federate_handle) if sinks is not None else None
        stream_ends: list[_CallbackStreamClosed] = []
        resigned = False
        retired_current = False

        def owns_current_membership() -> bool:
            return (
                owns_session
                and self.session_token(federate_handle) == token
                and self._stream_tasks.get(federate_handle) is task
            )

        def owns_retired_slot() -> bool:
            return (
                retired_current
                and not self.session_token(federate_handle)
                and federate_handle not in self._stream_tasks
            )

        def during_resign(event: Any) -> None:
            target = sink
            if not (owns_current_membership() or owns_retired_slot()):
                target = sinks.get(federate_handle) if sinks is not None else None
                if target is during_resign:
                    target = None
            elif isinstance(event, _CallbackStreamClosed):
                stream_ends.append(event)
                return
            if target is not None:
                target(event)
            else:
                if federate_handle in getattr(self, "_identity_bindings", {}) and not isinstance(
                    event, BaseException
                ):
                    object.__setattr__(
                        event, "_restore_serial", getattr(self, "_restore_serial", 0)
                    )
                self.events_for(federate_handle).put_nowait(event)

        if sinks is not None:
            sinks[federate_handle] = during_resign
        try:
            try:
                if token:
                    await self.federation.ResignFederation(
                        req, metadata=self.session_metadata(token)
                    )
                else:
                    await self.federation.ResignFederation(req)
            except Exception as exc:  # noqa: BLE001
                from ._grpc_errors import translate_rpc_error

                translate_rpc_error(exc)
            resigned = True
            # Once membership ends, another connection may recreate the same name.
            if not hasattr(self, "_stale_federation_generations"):
                self._stale_federation_generations = set()
            if getattr(self, "_generation_by_federation", {}).get(federation_name) == generation:
                self._stale_federation_generations.add(federation_name)
            tokens = getattr(self, "_session_tokens", {})
            retired_current = owns_current_membership()
            if retired_current:
                tokens.pop(federate_handle, None)
                getattr(self, "_ownership_callbacks_by_federate", {}).pop(federate_handle, None)
                getattr(self, "_checkpoint_sources", {}).pop(federate_handle, None)
                getattr(self, "_callback_reports_pending", {}).pop(federate_handle, None)
                getattr(self, "_generic_callbacks_by_federate", {}).pop(federate_handle, None)
                getattr(self, "_callback_ticket_identities", {}).pop(federate_handle, None)
                getattr(self, "_restored_local_notifications", {}).pop(federate_handle, None)
                getattr(self, "_local_notification_states", {}).pop(federate_handle, None)
                getattr(self, "_sdk_restore_ready", {}).pop(federate_handle, None)
                getattr(self, "_staged_sdk_restores", {}).pop(federate_handle, None)
                if task is not None:
                    self._stream_tasks.pop(federate_handle, None)
            if task is not None:
                # Accepted resignation closes the server outbox. Drain its EOF
                # rather than cancelling a live gRPC call against a closing loop.
                try:
                    await task
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if not task.cancelled() or (current is not None and current.cancelling()):
                        raise
        finally:
            same_membership = owns_current_membership()
            retired_slot = owns_retired_slot()
            if sinks is not None and sinks.get(federate_handle) is during_resign:
                if sink is None or not (same_membership or retired_slot):
                    sinks.pop(federate_handle, None)
                else:
                    sinks[federate_handle] = sink
            if not resigned and same_membership:
                for event in stream_ends:
                    self._deliver_event(federate_handle, event)
        return

    async def _publish_interaction(self, federate_handle: int, class_arg: int | str) -> None:
        """M27 Phase D: ``class_arg`` accepts ``int`` (FOM handle) or
        ``str`` (FOM name). Subscriber federates that joined an
        already-created federation may have an empty local FOM cache;
        passing the handle directly (resolved via SupportService) is
        the safe path."""
        from rti.v1 import common_pb2, declaration_pb2

        from ._grpc_errors import translate_rpc_error

        cls = self._resolve_interaction_class_handle(class_arg)
        req = declaration_pb2.PubInterRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            interaction_class_handle=cls,
        )
        try:
            await self.declaration.PublishInteractionClass(req)
        except Exception as exc:  # noqa: BLE001 — translate_rpc_error reraises
            translate_rpc_error(exc)
        return

    async def _subscribe_interaction(
        self, federate_handle: int, class_arg: int | str, *, active: bool = True
    ) -> None:
        """M27 Phase D: see _publish_interaction."""
        from rti.v1 import common_pb2, declaration_pb2

        from ._grpc_errors import translate_rpc_error

        cls = self._resolve_interaction_class_handle(class_arg)
        req = declaration_pb2.SubInterRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            interaction_class_handle=cls,
            active=active,
        )
        try:
            await self.declaration.SubscribeInteractionClass(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
        return

    async def _send_interaction(
        self,
        federate_handle: int,
        class_arg: int | str,
        parameters: dict[int | str, Any],
        timestamp: float | None,
        *,
        tag: bytes = b"",
        retraction_handle: int = 0,
    ) -> None:
        """M27 Phase B: ``class_arg`` and parameter dict keys accept
        ``int`` (handle, IEEE 1516 service-style) or ``str`` (FOM name)."""
        from rti.v1 import common_pb2, object_pb2

        cls = self._resolve_interaction_class_handle(class_arg)
        # Resolve parameter keys: int → use directly; str → look up.
        param_map: dict[int, bytes] = {}
        class_name_for_index: str | None = None
        for key, payload in parameters.items():
            if isinstance(key, int):
                param_map[int(key)] = _coerce_payload(payload)
                continue
            if class_name_for_index is None:
                class_name_for_index = self._interaction_class_name_for(class_arg)
            if class_name_for_index is None:
                continue  # unknown class; can't resolve names
            param_index = self._parameter_indices_for(class_name_for_index)
            if key not in param_index:
                # The pyjevsim bridge transports an opaque value as
                # ``_payload`` without knowing the FOM parameter name. Permit
                # that alias only when the class has exactly one declared
                # parameter, so multi-parameter HLA APIs remain unambiguous.
                unique_handles = set(param_index.values())
                if key == "_payload" and len(unique_handles) == 1:
                    param_map[next(iter(unique_handles))] = _coerce_payload(payload)
                continue
            param_map[param_index[key]] = _coerce_payload(payload)
        req = object_pb2.SendInteractionRequest(
            user_supplied_tag=bytes(tag),
            message_retraction_handle=int(retraction_handle),
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            interaction_class_handle=cls,
            parameters=param_map,
        )
        if timestamp is not None:
            self._write_time(req, "logical_time", timestamp)
        try:
            await self.objects.SendInteraction(req)
            self._remember_retraction(federate_handle, timestamp, retraction_handle)
        except Exception as exc:  # noqa: BLE001
            from ._grpc_errors import translate_rpc_error

            translate_rpc_error(exc)
        return

    def _remember_retraction(self, handle: int, timestamp: Any, retraction_handle: int) -> None:
        if timestamp is not None and retraction_handle:
            used = getattr(self, "_used_retractions", None)
            if used is None:
                used = self._used_retractions = {}
            used.setdefault(handle, set()).add(retraction_handle)

    # --- TimeService dispatchers -----------------------------------------------

    async def _enable_time_regulation(
        self,
        federate_handle: int,
        lookahead: float,
    ) -> None:
        """Dispatch TimeService.EnableTimeRegulation (M21)."""
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.EnableRegulationRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        self._write_time(req, "lookahead", lookahead)
        try:
            await self.time.EnableTimeRegulation(req)
        except Exception as exc:  # noqa: BLE001 — translate_rpc_error reraises
            translate_rpc_error(exc)

    async def _enable_time_constrained(self, federate_handle: int) -> None:
        """Dispatch TimeService.EnableTimeConstrained (M21)."""
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.EnableConstrainedRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        try:
            await self.time.EnableTimeConstrained(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _next_message_request(
        self,
        federate_handle: int,
        t: float,
    ) -> None:
        """Dispatch TimeService.NextMessageRequest (M21)."""
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.NERRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        self._write_time(req, "logical_time", t)
        try:
            await self.time.NextMessageRequest(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    # --- Additional TimeService dispatchers ------------------------------------

    async def _disable_time_regulation(self, federate_handle: int) -> None:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.DisableRegulationRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        try:
            await self.time.DisableTimeRegulation(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _disable_time_constrained(self, federate_handle: int) -> None:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.DisableConstrainedRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        try:
            await self.time.DisableTimeConstrained(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _modify_lookahead(
        self,
        federate_handle: int,
        lookahead: float,
    ) -> None:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.ModifyLookaheadRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        self._write_time(req, "lookahead", lookahead)
        try:
            await self.time.ModifyLookahead(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _next_message_request_available(
        self,
        federate_handle: int,
        t: float,
    ) -> None:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.NMRARequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        self._write_time(req, "logical_time", t)
        try:
            await self.time.NextMessageRequestAvailable(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _time_advance_request(
        self,
        federate_handle: int,
        t: float,
    ) -> None:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.TARRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        self._write_time(req, "logical_time", t)
        try:
            await self.time.TimeAdvanceRequest(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _time_advance_request_available(
        self,
        federate_handle: int,
        t: float,
    ) -> None:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.TARARequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        self._write_time(req, "logical_time", t)
        try:
            await self.time.TimeAdvanceRequestAvailable(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _flush_queue_request(
        self,
        federate_handle: int,
        t: float,
    ) -> None:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.FQRRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        self._write_time(req, "logical_time", t)
        try:
            await self.time.FlushQueueRequest(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _query_logical_time(self, federate_handle: int) -> int | float:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.QueryFederateTimeRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        try:
            resp = await self.time.QueryLogicalTime(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise  # unreachable; translate_rpc_error always raises
        value = self._read_time(resp, "logical_time")
        times = getattr(self, "_logical_times_by_federate", None)
        if times is None:
            times = self._logical_times_by_federate = {}
        times[federate_handle] = value
        return value

    async def _query_lookahead(self, federate_handle: int) -> int | float:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.QueryFederateTimeRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        try:
            resp = await self.time.QueryLookahead(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return self._read_time(resp, "lookahead")

    async def _query_lbts(self) -> tuple[int | float, bool]:
        """Return (lbts, finite). Federation-scoped (no federate handle)."""
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.QueryLBTSRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
        )
        try:
            resp = await self.time.QueryLBTS(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return (self._read_time(resp, "lbts") if resp.finite else 0, bool(resp.finite))

    # --- Asynchronous-delivery dispatchers -------------------------------------

    async def _enable_asynchronous_delivery(self, federate_handle: int) -> None:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.EnableAsynchronousDeliveryRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        try:
            await self.time.EnableAsynchronousDelivery(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _disable_asynchronous_delivery(self, federate_handle: int) -> None:
        from rti.v1 import common_pb2, time_pb2

        from ._grpc_errors import translate_rpc_error

        req = time_pb2.DisableAsynchronousDeliveryRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
        )
        try:
            await self.time.DisableAsynchronousDelivery(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    # --- ObjectService.DeleteObjectInstance dispatcher ---

    async def _delete_object_instance(
        self,
        federate_handle: int,
        object_handle: int,
        tag: bytes,
        timestamp: float | None,
        *,
        retraction_handle: int = 0,
    ) -> None:
        from rti.v1 import common_pb2, object_pb2

        from ._grpc_errors import translate_rpc_error

        req = object_pb2.DeleteObjectInstanceRequest(
            message_retraction_handle=int(retraction_handle),
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            object_handle=object_handle,
            user_supplied_tag=tag,
        )
        if timestamp is not None:
            self._write_time(req, "logical_time", timestamp)
        try:
            await self.objects.DeleteObjectInstance(req)
            self._remember_retraction(federate_handle, timestamp, retraction_handle)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _local_delete_object_instance(
        self,
        federate_handle: int,
        object_handle: int,
    ) -> None:
        from rti.v1 import common_pb2, object_pb2

        from ._grpc_errors import translate_rpc_error

        req = object_pb2.LocalDeleteObjectInstanceRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            object_handle=object_handle,
        )
        try:
            states: dict[int, OwnershipCallbacks] = getattr(
                self, "_ownership_callbacks_by_federate", {}
            )
            state = states.get(federate_handle)
            if state is not None and state.enabled:
                scope = state.scope
                token = self.session_token(federate_handle)
                serial = getattr(self, "_restore_serial", 0)
                response = await self.objects.LocalDeleteObjectInstanceFenced(
                    req, metadata=self.session_metadata(token)
                )
                if (
                    states.get(federate_handle) is not state
                    or state.scope != scope
                    or scope != (int(response.federation_generation), int(response.callback_epoch))
                    or scope[0] != self._generation_by_federation.get(req.federation_name)
                    or self.session_token(federate_handle) != token
                    or getattr(self, "_restore_serial", 0) != serial
                    or getattr(self, "_restore_applying", False)
                ):
                    raise RuntimeError("local-delete response has stale callback scope")
                state.retire(object_handle, int(response.retired_knowledge_epoch))
            else:
                await self.objects.LocalDeleteObjectInstance(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _request_attribute_value_update(
        self,
        federate_handle: int,
        object_handle: int,
        attribute_handles: list[int],
        tag: bytes,
    ) -> None:
        from rti.v1 import common_pb2, object_pb2

        from ._grpc_errors import translate_rpc_error

        req = object_pb2.RequestAttributeValueUpdateRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            object_handle=object_handle,
            attribute_handles=[int(h) for h in attribute_handles],
            user_supplied_tag=tag,
        )
        try:
            await self.objects.RequestAttributeValueUpdate(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _request_class_attribute_value_update(
        self,
        federate_handle: int,
        object_class_handle: int,
        attribute_handles: list[int],
        tag: bytes,
    ) -> None:
        from rti.v1 import common_pb2, object_pb2

        from ._grpc_errors import translate_rpc_error

        req = object_pb2.RequestClassAttributeValueUpdateRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            object_class_handle=object_class_handle,
            attribute_handles=[int(h) for h in attribute_handles],
            user_supplied_tag=tag,
        )
        try:
            await self.objects.RequestClassAttributeValueUpdate(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _change_attribute_transportation_type(
        self,
        federate_handle: int,
        object_handle: int,
        attribute_handles: list[int],
        transport_type: int,
    ) -> None:
        from rti.v1 import common_pb2, object_pb2

        from ._grpc_errors import translate_rpc_error

        req = object_pb2.ChangeAttributeTransportRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            object_handle=object_handle,
            attribute_handles=[int(h) for h in attribute_handles],
            transport_type=transport_type,
        )
        try:
            await self.objects.ChangeAttributeTransportationType(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _change_interaction_transportation_type(
        self,
        federate_handle: int,
        interaction_class_handle: int,
        transport_type: int,
    ) -> None:
        from rti.v1 import common_pb2, object_pb2

        from ._grpc_errors import translate_rpc_error

        req = object_pb2.ChangeInteractionTransportRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            interaction_class_handle=interaction_class_handle,
            transport_type=transport_type,
        )
        try:
            await self.objects.ChangeInteractionTransportationType(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _publish_object_class(
        self,
        federate_handle: int,
        class_arg: int | str,
        attributes: list[int | str],
    ) -> None:
        """Dispatch DeclarationService.PublishObjectClassAttributes (M12 W2).

        M27 Phase B: ``class_arg`` and each entry of ``attributes`` accept
        either ``int`` (already-resolved handle, IEEE 1516 service-style) or ``str``
        (FOM name, pysdk convenience). Mixed lists are allowed. Unknown
        names resolve to handle 0 and are silently dropped — the Go side
        rejects unknown handles at the wire layer.
        """
        from rti.v1 import common_pb2, declaration_pb2

        cls = self._resolve_object_class_handle(class_arg)
        attr_handles = self._resolve_attribute_handles(class_arg, attributes)
        req = declaration_pb2.PubObjAttrsRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            object_class_handle=cls,
            attribute_handles=attr_handles,
        )
        try:
            await self.declaration.PublishObjectClassAttributes(req)
        except Exception as exc:  # noqa: BLE001
            from ._grpc_errors import translate_rpc_error

            translate_rpc_error(exc)
        return

    async def _subscribe_object_class(
        self,
        federate_handle: int,
        class_arg: int | str,
        attributes: list[int | str],
        *,
        active: bool = True,
        update_rate_designator: str = "",
    ) -> None:
        """Dispatch DeclarationService.SubscribeObjectClassAttributes.

        See _publish_object_class for the M27 Phase B int|str semantics.
        """
        from rti.v1 import common_pb2, declaration_pb2

        cls = self._resolve_object_class_handle(class_arg)
        attr_handles = self._resolve_attribute_handles(class_arg, attributes)
        req = declaration_pb2.SubObjAttrsRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            object_class_handle=cls,
            attribute_handles=attr_handles,
            active=active,
            update_rate_designator=update_rate_designator,
        )
        try:
            await self.declaration.SubscribeObjectClassAttributes(req)
        except Exception as exc:  # noqa: BLE001
            from ._grpc_errors import translate_rpc_error

            translate_rpc_error(exc)
        return

    async def _set_advisory_switch(
        self, federate_handle: int, switch_kind: int, enabled: bool
    ) -> None:
        from rti.v1 import common_pb2, declaration_pb2

        from ._grpc_errors import translate_rpc_error

        request = declaration_pb2.SetAdvisorySwitchRequest(
            wire_version=common_pb2.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            switch_kind=switch_kind,
            enabled=enabled,
        )
        try:
            await self.declaration.SetAdvisorySwitch(request)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def _register_object_instance(
        self,
        federate_handle: int,
        class_arg: int | str,
        instance_name: str | None,
    ) -> int:
        """Dispatch ObjectService.RegisterObjectInstance (M12 W2).

        M27 Phase B: ``class_arg`` accepts ``int`` (IEEE 1516 service-style handle)
        or ``str`` (FOM name). Returns the minted object handle.
        """
        from rti.v1 import common_pb2, object_pb2

        cls = self._resolve_object_class_handle(class_arg)
        req = object_pb2.RegisterObjectRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            object_class_handle=cls,
            object_name=instance_name or "",
        )
        try:
            resp = await self.objects.RegisterObjectInstance(req)
        except Exception as exc:  # noqa: BLE001
            from ._grpc_errors import translate_rpc_error

            translate_rpc_error(exc)
            raise  # unreachable; translate_rpc_error always raises
        return int(resp.object_handle)

    async def _update_attributes(
        self,
        federate_handle: int,
        object_handle: int,
        values: dict[str, Any],
        timestamp: float | None,
        *,
        tag: bytes = b"",
        retraction_handle: int = 0,
    ) -> None:
        """Resolve names in the caller's known class, then send one update."""
        from rti.v1 import common_pb2, object_pb2

        if timestamp is not None:
            from ._logical_time import validate_time

            validate_time(
                timestamp, self.logical_time_implementation_name(), allow_legacy_final=True
            )
        support = None
        known_class = 0
        class_name = None
        if any(not isinstance(key, int) for key in values):
            from .support import SupportClient

            # Knowledge can change after local deletion or restore. Query it
            # instead of caching a class for the object's entire handle lifetime.
            support = SupportClient(
                self.channel,
                federation_name=self._federation_name or "",
                federate_handle=federate_handle,
                session_token=self.session_token(federate_handle),
            )
            known_class = await support.get_known_object_class_handle(int(object_handle))
            class_name = self._object_class_name_for(known_class)

        attr_map: dict[int, bytes] = {}
        for key, payload in values.items():
            if isinstance(key, int):
                handle = int(key)
            else:
                handle = (
                    self._attribute_handle_for(class_name, str(key))
                    if class_name is not None
                    else 0
                )
                if handle == 0:
                    if support is None:
                        raise RuntimeError("attribute name resolution is unavailable")
                    handle = await support.get_attribute_handle(known_class, str(key))
            if handle in attr_map:
                raise ValueError("multiple update keys identify the same attribute")
            attr_map[handle] = _coerce_payload(payload)
        req = object_pb2.UpdateAttributeValuesRequest(
            user_supplied_tag=bytes(tag),
            message_retraction_handle=int(retraction_handle),
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name or "",
            federate_handle=federate_handle,
            object_handle=int(object_handle),
            attributes=attr_map,
        )
        if timestamp is not None:
            self._write_time(req, "logical_time", timestamp)
        try:
            await self.objects.UpdateAttributeValues(req)
            self._remember_retraction(federate_handle, timestamp, retraction_handle)
        except Exception as exc:  # noqa: BLE001
            from ._grpc_errors import translate_rpc_error

            translate_rpc_error(exc)
        return

    def _object_class_handle_for(self, class_name: str) -> int:
        """Return the numeric object-class handle for ``class_name``; 0 on miss."""
        return self._object_class_handles.get(class_name, 0)

    def _object_class_name_for(self, handle: int) -> str | None:
        """Return the FOM name for an object-class handle, or None on miss.
        M27 Phase B — inverse lookup for handle-keyed dispatch paths
        that still need to resolve attribute names. Linear over the
        cached map; the map is small enough (handful of classes) that
        the cost is negligible vs caching an inverse dict."""
        for name, h in self._object_class_handles.items():
            if h == handle:
                return name
        return None

    def _interaction_class_name_for(self, class_arg: int | str) -> str | None:
        """Return the FOM name for an interaction class identifier.

        M27 Phase B helper. Accepts either ``int`` (handle, reverse-
        looked-up via the existing inverse map) or ``str`` (passes
        through if known to the FOM). Returns None if the identifier
        does not resolve.
        """
        if isinstance(class_arg, int):
            return self._inverse_interaction_handles.get(class_arg)
        if class_arg in self._interaction_handles:
            return class_arg
        return None

    def _resolve_object_class_handle(self, class_arg: int | str) -> int:
        """M27 Phase B: int → identity; str → FOM lookup; 0 on miss."""
        if isinstance(class_arg, int):
            return int(class_arg)
        return self._object_class_handle_for(class_arg)

    def _resolve_interaction_class_handle(self, class_arg: int | str) -> int:
        """M27 Phase B: int → identity; str → FOM lookup; 0 on miss."""
        if isinstance(class_arg, int):
            return int(class_arg)
        return self._interaction_handle_for(class_arg)

    def _resolve_attribute_handles(
        self,
        class_arg: int | str,
        attributes: list[int | str],
    ) -> list[int]:
        """M27 Phase B: resolve mixed-type attribute list to handles.

        For ``int`` entries: pass through as already-resolved handles.
        For ``str`` entries: look up via the FOM, using ``class_arg`` to
        scope the lookup. If ``class_arg`` is an ``int`` (handle), the
        class name is inverse-looked-up first; failure to resolve the
        class skips all string-keyed attribute lookups.
        """
        out: list[int] = []
        class_name: str | None = None
        for a in attributes:
            if isinstance(a, int):
                out.append(int(a))
                continue
            if class_name is None:
                class_name = (
                    class_arg
                    if isinstance(class_arg, str)
                    else self._object_class_name_for(class_arg)
                )
            if class_name is None:
                continue
            h = self._attribute_handle_for(class_name, a)
            if h != 0:
                out.append(h)
        return out

    def _attribute_handle_for(self, class_name: str, attr_name: str) -> int:
        """Return the numeric attribute handle for (class, attr); 0 on miss.

        Cached lazily on first lookup. The handle index is the
        attribute's 1-based position within the class's parsed
        attribute list (same convention as the Go-side fomHandle.
        LookupAttribute — see ``rti/cmd/rtid/foms.go``).
        """
        cache = self._attribute_handle_cache.get(class_name)
        if cache is None:
            cache = self._build_attribute_cache(class_name)
            self._attribute_handle_cache[class_name] = cache
        return cache.get(attr_name, 0)

    def _build_attribute_cache(self, class_name: str) -> dict[str, int]:
        """Walk the FOM and build an attribute-name → 1-based-handle map.

        Returns an empty map when the FOM has not been parsed (no
        federation create with modules) or when the class is not
        present in the FOM. The Go side's LookupAttribute returns
        InvalidAttributeHandle in the same scenario; the cache mirror
        keeps name resolution cheap and consistent.
        """
        if self._fom_cache is None:
            return {}
        for oc in self._fom_cache.object_classes:
            if oc.name == class_name:
                return {a.name: i + 1 for i, a in enumerate(oc.attributes)}
        return {}

    # --- Stream draining ----------------------------------------------------

    @staticmethod
    def session_metadata(token: bytes) -> tuple[tuple[str, bytes], ...]:
        return (("gorti-session-token-bin", token),) if token else ()

    def session_token(self, federate_handle: int) -> bytes:
        tokens: dict[int, bytes] = getattr(self, "_session_tokens", {})
        return tokens.get(federate_handle, b"")

    def _start_event_stream(self, federate_handle: int) -> None:
        """Launch the background task that drains StreamService.Events
        for ``federate_handle`` into the local asyncio.Queue."""
        if federate_handle in self._stream_tasks:
            return
        self.events_for(federate_handle)  # ensure queue exists
        loop = asyncio.get_event_loop()
        self._stream_tasks[federate_handle] = loop.create_task(self._drain_events(federate_handle))

    async def _drain_events(self, federate_handle: int) -> None:
        """Background task: forward FederateEvent -> typed event onto the queue."""
        from rti.v1 import common_pb2, stream_pb2

        if self._federation_name is None:
            return
        req = stream_pb2.EventsRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=federate_handle,
        )
        self.events_for(federate_handle)
        token = self.session_token(federate_handle)
        task = asyncio.current_task()
        managed = federate_handle in getattr(self, "_stream_tasks", {})

        def owns_current_stream() -> bool:
            return not managed or (
                self._stream_tasks.get(federate_handle) is task
                and self.session_token(federate_handle) == token
            )

        try:
            if not owns_current_stream():
                return
            stream = (
                self.streams.Events(req, metadata=self.session_metadata(token))
                if token
                else self.streams.Events(req)
            )
            async for fed_event in stream:
                if not owns_current_stream():
                    continue
                translated = self._decode_carrier(federate_handle, fed_event)
                if translated is not None:
                    if fed_event.WhichOneof("event") == "transportation_change_confirmed":
                        object.__setattr__(translated, "_transportation_caller", federate_handle)
                    fields = fed_event.DESCRIPTOR.fields_by_name
                    if "ownership_receipt" in fields and fed_event.HasField("ownership_receipt"):
                        from rti.v1 import callback_pb2

                        receipt_value = callback_pb2.OwnershipCallbackReceipt()
                        receipt_value.CopyFrom(fed_event.ownership_receipt)
                        object.__setattr__(translated, "_ownership_receipt", receipt_value)
                    if "object_knowledge_epoch" in fields:
                        object.__setattr__(
                            translated,
                            "_object_knowledge_epoch",
                            int(fed_event.object_knowledge_epoch),
                        )
                    state = getattr(self, "_ownership_callbacks_by_federate", {}).get(
                        federate_handle
                    )
                    if state is not None and not getattr(
                        translated, "_callback_outcome_only", False
                    ):
                        state.observe(translated)
                    receipt = bytes(getattr(fed_event, "callback_receipt", b""))
                    if receipt:
                        object.__setattr__(translated, "_callback_receipt", receipt)
                    federate_handle = await self._apply_restore_terminal(
                        federate_handle, translated
                    )
                    if owns_current_stream():
                        self._deliver_event(federate_handle, translated)
            raise _CallbackStreamClosed("callback stream closed unexpectedly")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # A retired drainer must not poison a replacement callback sink.
            if owns_current_stream():
                self._deliver_event(federate_handle, exc)

    def _translate_event(self, fed_event: Any) -> Any | None:  # noqa: PLR0911, PLR0912, PLR0915
        """Translate a wire FederateEvent into one of rti1516e.events.*.

        M39 (HA-1): every FederateEvent oneof variant on the M37 wire has
        a branch here. Variants with NO branch (a future wire addition,
        or an unknown field from a newer server) hit the default at the
        bottom, which warns ONCE per variant instead of silently
        dropping the event.
        """
        from rti1516e.events import (
            AttributeOwnershipAcquisitionNotification,
            DiscoverObjectInstance,
            FederationHalted,
            FederationNotSaved,
            FederationSaved,
            FederationSynchronized,
            InitiateFederateSave,
            ReceiveInteraction,
            ReflectAttributeValues,
            RequestAttributeOwnershipAssumption,
            RequestDivestitureConfirmation,
            SynchronizationPointAnnounced,
            TimeAdvanceGrant,
            TimeConstrainedEnabled,
            TimeRegulationEnabled,
        )
        from rti1516e.handles import AttributeHandle, ObjectClassHandle

        which = fed_event.WhichOneof("event")
        if which == "transportation_change_confirmed":
            from .events import TransportationChangeConfirmed

            change = fed_event.transportation_change_confirmed
            return TransportationChangeConfirmed(
                bytes(change.ticket),
                int(change.federation_generation),
                int(change.callback_epoch),
                int(change.object_handle),
                tuple(int(value) for value in change.attribute_handles),
                int(change.interaction_class_handle),
                int(change.transport_type),
            )
        if which == "receive":
            r = fed_event.receive
            class_name = self._inverse_interaction_handles.get(
                int(r.interaction_class_handle), str(r.interaction_class_handle)
            )
            params: dict[str, Any] = {}
            inv = {v: k for k, v in self._parameter_indices_for(class_name).items()}
            for handle, payload in r.parameters.items():
                params[inv.get(int(handle), str(handle))] = bytes(payload)
            ts = self._read_time(r, "logical_time", optional=True)
            return ReceiveInteraction(
                class_name=class_name,
                parameters=params,
                timestamp=ts,
                tag=bytes(r.user_supplied_tag),
                metadata=_object_callback_metadata(r),
            )
        if which == "grant":
            return TimeAdvanceGrant(time=self._read_time(fed_event.grant, "logical_time"))
        if which == "time_regulation_enabled":
            return TimeRegulationEnabled(
                time=self._read_time(fed_event.time_regulation_enabled, "logical_time")
            )
        if which == "time_constrained_enabled":
            return TimeConstrainedEnabled(
                time=self._read_time(fed_event.time_constrained_enabled, "logical_time")
            )
        if which == "discover":
            d = fed_event.discover
            return DiscoverObjectInstance(
                object_handle=int(d.object_handle),
                # DEPRECATED identity carrier (stringified handle);
                # M39 adds the typed object_class alongside (§6.9).
                class_name=str(d.object_class_handle),
                instance_name=str(d.object_name),
                object_class=ObjectClassHandle(int(d.object_class_handle)),
                producing_federate=int(d.producing_federate)
                if d.HasField("producing_federate")
                else None,
            )
        if which == "reflect":
            r = fed_event.reflect
            ts = self._read_time(r, "logical_time", optional=True)
            return ReflectAttributeValues(
                object_handle=int(r.object_handle),
                # DEPRECATED string-keyed map (stringified handles);
                # M39 adds the typed attribute_values alongside (§6.11).
                values={str(k): bytes(v) for k, v in r.attributes.items()},
                timestamp=ts,
                attribute_values={
                    AttributeHandle(int(k)): bytes(v) for k, v in r.attributes.items()
                },
                tag=bytes(r.user_supplied_tag),
                metadata=_object_callback_metadata(r),
            )
        if which == "remove":
            # M23 — RemoveObjectInstance per IEEE 1516.1 §6.16.
            from .events import RemoveObjectInstance

            rm = fed_event.remove
            ts = self._read_time(rm, "logical_time", optional=True)
            return RemoveObjectInstance(
                object_handle=int(rm.object_handle),
                tag=bytes(rm.user_supplied_tag),
                timestamp=ts,
                metadata=_object_callback_metadata(rm),
            )
        if which == "turn_updates_on":
            from .events import TurnUpdatesOnForObjectInstance

            event = fed_event.turn_updates_on
            return TurnUpdatesOnForObjectInstance(
                object_handle=int(event.object_handle),
                attribute_handles=tuple(int(h) for h in event.attribute_handles),
                update_rate_designator=event.update_rate_designator
                if event.HasField("update_rate_designator")
                else None,
            )
        if which == "turn_updates_off":
            from .events import TurnUpdatesOffForObjectInstance

            event = fed_event.turn_updates_off
            return TurnUpdatesOffForObjectInstance(
                object_handle=int(event.object_handle),
                attribute_handles=tuple(int(h) for h in event.attribute_handles),
            )
        if which == "provide_update":
            # M23 — ProvideAttributeValueUpdate per IEEE 1516.1 §6.26.
            from .events import ProvideAttributeValueUpdate

            pv = fed_event.provide_update
            return ProvideAttributeValueUpdate(
                object_handle=int(pv.object_handle),
                attribute_handles=tuple(int(h) for h in pv.attribute_handles),
                tag=bytes(pv.user_supplied_tag),
            )
        # M12 W2 cut-2 service-group callbacks (deferral #1 close).
        if which == "sync_announced":
            a = fed_event.sync_announced
            return SynchronizationPointAnnounced(
                label=str(a.label),
                tag=bytes(a.tag),
                required_federates=tuple(int(h) for h in a.required_federates),
            )
        if which == "sync_synchronized":
            s = fed_event.sync_synchronized
            return FederationSynchronized(
                label=str(s.label),
                # §4.15 failed-to-sync set (M37, additive field 2).
                failed_to_sync=tuple(int(h) for h in s.failed_to_sync),
            )
        # M37 §4.12 — sync registration acks (tags 22/23).
        if which == "sync_registration_succeeded":
            from .events import SynchronizationPointRegistrationSucceeded

            return SynchronizationPointRegistrationSucceeded(
                label=str(fed_event.sync_registration_succeeded.label),
            )
        if which == "sync_registration_failed":
            from .events import (
                SynchronizationPointFailureReason,
                SynchronizationPointRegistrationFailed,
            )

            f = fed_event.sync_registration_failed
            reason_map = {
                1: SynchronizationPointFailureReason.SYNCHRONIZATION_POINT_LABEL_NOT_UNIQUE,
                2: SynchronizationPointFailureReason.SYNCHRONIZATION_SET_MEMBER_NOT_JOINED,
            }
            return SynchronizationPointRegistrationFailed(
                label=str(f.label),
                reason=reason_map.get(int(f.reason)),
            )
        if which == "ownership_assumption":
            o = fed_event.ownership_assumption
            return RequestAttributeOwnershipAssumption(
                object_handle=int(o.object_handle),
                attribute_handles=tuple(int(h) for h in o.attribute_handles),
                divesting_federate=int(o.divesting_federate),
                tag=bytes(o.tag),
            )
        if which == "ownership_acquired":
            o = fed_event.ownership_acquired
            return AttributeOwnershipAcquisitionNotification(
                object_handle=int(o.object_handle),
                attribute_handles=tuple(int(h) for h in o.attribute_handles),
                owning_federate=int(o.owning_federate),
                tag=bytes(o.tag),
            )
        if which == "ownership_divest_confirmed":
            o = fed_event.ownership_divest_confirmed
            return RequestDivestitureConfirmation(
                object_handle=int(o.object_handle),
                attribute_handles=tuple(int(h) for h in o.attribute_handles),
            )
        # M37 §7.11 — the current owner is asked to release (tag 33).
        if which == "ownership_release_requested":
            from .events import RequestAttributeOwnershipRelease

            o = fed_event.ownership_release_requested
            return RequestAttributeOwnershipRelease(
                object_handle=int(o.object_handle),
                attribute_handles=tuple(int(h) for h in o.attribute_handles),
                tag=bytes(o.tag),
            )
        # M37 §7.10 — acquisition-if-available lost (tag 34).
        if which == "ownership_unavailable":
            from .events import AttributeOwnershipUnavailable

            o = fed_event.ownership_unavailable
            return AttributeOwnershipUnavailable(
                object_handle=int(o.object_handle),
                attribute_handles=tuple(int(h) for h in o.attribute_handles),
            )
        if which == "save_initiate":
            s = fed_event.save_initiate
            save_time = self._read_time(s, "save_time", optional=True)
            return InitiateFederateSave(label=str(s.label), save_time=save_time)
        if which == "ownership_acquisition_cancelled":
            from .events import ConfirmAttributeOwnershipAcquisitionCancellation

            value = fed_event.ownership_acquisition_cancelled
            return ConfirmAttributeOwnershipAcquisitionCancellation(
                int(value.object_handle), tuple(int(h) for h in value.attribute_handles)
            )
        if which == "save_completed":
            return FederationSaved(label=str(fed_event.save_completed.label))
        if which == "save_failed":
            return FederationNotSaved(label=str(fed_event.save_failed.label))
        # Restore family. Tags 43-45 predate M37 (M17.25) but were
        # silently dropped by this switch until M39; 46-48 are M37.
        if which == "restore_initiate":
            from .events import InitiateFederateRestore

            r = fed_event.restore_initiate
            return InitiateFederateRestore(
                label=str(r.label),
                federate_handle=int(r.federate_handle),
                federate_name=str(r.federate_name),
                client_state=bytes(r.client_state),
            )
        if which == "restore_completed":
            from .events import FederationRestored

            restored = fed_event.restore_completed
            return FederationRestored(
                label=str(restored.label),
                federate_handle=int(restored.federate_handle)
                if restored.HasField("federate_handle")
                else None,
                callback_epoch=int(restored.callback_epoch)
                if restored.HasField("callback_epoch")
                else None,
                next_retraction_handle=int(restored.next_retraction_handle)
                if restored.HasField("next_retraction_handle")
                else None,
            )
        if which == "restore_failed":
            from .events import FederationNotRestored

            return FederationNotRestored(label=str(fed_event.restore_failed.label))
        if which == "restore_request_succeeded":
            from .events import RequestFederationRestoreSucceeded

            return RequestFederationRestoreSucceeded(
                label=str(fed_event.restore_request_succeeded.label),
            )
        if which == "restore_request_failed":
            from .events import RequestFederationRestoreFailed

            r = fed_event.restore_request_failed
            return RequestFederationRestoreFailed(
                label=str(r.label),
                reason=str(r.reason),
            )
        if which == "restore_begun":
            from .events import FederationRestoreBegun

            return FederationRestoreBegun()
        # M37 §5.10-§5.13 — registration / interaction advisories.
        if which == "start_registration":
            from .events import StartRegistrationForObjectClass

            return StartRegistrationForObjectClass(
                object_class_handle=int(fed_event.start_registration.object_class_handle),
            )
        if which == "stop_registration":
            from .events import StopRegistrationForObjectClass

            return StopRegistrationForObjectClass(
                object_class_handle=int(fed_event.stop_registration.object_class_handle),
            )
        if which == "turn_interactions_on":
            from .events import TurnInteractionsOn

            return TurnInteractionsOn(
                interaction_class_handle=int(
                    fed_event.turn_interactions_on.interaction_class_handle
                ),
            )
        if which == "turn_interactions_off":
            from .events import TurnInteractionsOff

            return TurnInteractionsOff(
                interaction_class_handle=int(
                    fed_event.turn_interactions_off.interaction_class_handle
                ),
            )
        # M37 §6.17/§6.18 — DDM scope advisories.
        if which == "attributes_in_scope":
            from .events import AttributesInScope

            s = fed_event.attributes_in_scope
            return AttributesInScope(
                object_handle=int(s.object_handle),
                attribute_handles=tuple(int(h) for h in s.attribute_handles),
            )
        if which == "attributes_out_of_scope":
            from .events import AttributesOutOfScope

            s = fed_event.attributes_out_of_scope
            return AttributesOutOfScope(
                object_handle=int(s.object_handle),
                attribute_handles=tuple(int(h) for h in s.attribute_handles),
            )
        # M37 §8.22 — retraction of a delivered TSO message.
        if which == "retraction_requested":
            from .events import RequestRetraction

            r = fed_event.retraction_requested
            return RequestRetraction(
                sender_federate=int(r.sender_federate),
                retraction_handle=int(r.message_retraction_handle),
            )
        # M26 Phase F — object instance name reservation events.
        if which == "reservation_succeeded":
            from .events import ObjectInstanceNameReservationSucceeded

            return ObjectInstanceNameReservationSucceeded(
                object_name=str(fed_event.reservation_succeeded.object_name),
            )
        if which == "reservation_failed":
            from .events import ObjectInstanceNameReservationFailed

            return ObjectInstanceNameReservationFailed(
                object_name=str(fed_event.reservation_failed.object_name),
            )
        if which == "reservation_multi_succeeded":
            from .events import MultipleObjectInstanceNameReservationSucceeded

            return MultipleObjectInstanceNameReservationSucceeded(
                object_names=tuple(
                    str(n) for n in fed_event.reservation_multi_succeeded.object_names
                ),
            )
        if which == "reservation_multi_failed":
            from .events import MultipleObjectInstanceNameReservationFailed

            mf = fed_event.reservation_multi_failed
            return MultipleObjectInstanceNameReservationFailed(
                requested_names=tuple(str(n) for n in mf.requested_names),
                colliding_names=tuple(str(n) for n in mf.colliding_names),
            )
        if which == "halted":
            # The proto FederationHalted lacks a stalled-federate field;
            # surface 0 as "no specific federate identified" so the
            # dataclass invariant is preserved on the SDK side.
            return FederationHalted(
                cause=str(fed_event.halted.cause),
                stalled_federate_handle=0,
            )
        # No branch matched. Two ways to get here:
        #   - ``which`` names a variant this switch forgot (a wire
        #     addition without a translation — the pre-M39 silent-drop
        #     bug class), or
        #   - ``which is None``: the event arrived from a NEWER server
        #     whose oneof tag this client's generated stubs don't know
        #     (proto3 open-set unknown-field path).
        # Either way: warn ONCE per variant so the gap is visible, then
        # drop the event (the wire contract says unknown variants are
        # skippable).
        tag = which if which is not None else "<unknown-wire-tag>"
        if tag not in _UNTRANSLATED_EVENT_WARNED:
            _UNTRANSLATED_EVENT_WARNED.add(tag)
            import warnings

            warnings.warn(
                f"rti1516e: FederateEvent variant {tag!r} "
                f"(seq={int(getattr(fed_event, 'seq', 0))}) has no pysdk "
                "translation and was dropped — add a branch in "
                "rti1516e/_transport.py _translate_event (warning fires "
                "once per variant)",
                RuntimeWarning,
                stacklevel=2,
            )
        return None

    # --- FOM-driven name → handle resolution -------------------------------

    def _install_fom_view(self, view: Any, *, replace: bool = False) -> None:
        """Publish validated server bindings without assigning local positions."""
        from .fom.modules import MIMModule
        from .fom.parser import _parse_sources

        if int(view.revision) <= 0:
            raise ValueError("invalid authoritative FOM revision")
        selected_mim = None
        if getattr(view, "HasField", lambda _: False)("mim_module"):
            selected_mim = MIMModule(str(view.mim_module.path), bytes(view.mim_module.xml))
        if (
            not replace
            and cast(int, getattr(self, "_fom_revision", 0))
            and selected_mim != getattr(self, "_mim_module", None)
        ):
            raise ValueError("authoritative FOM refresh changed the selected MIM")
        sources = [(str(module.path), bytes(module.xml)) for module in getattr(view, "modules", ())]
        result = (
            _parse_sources(sources, mim_module=selected_mim) if sources or selected_mim else None
        )

        def tables(
            classes: Any, root: str
        ) -> tuple[dict[str, int], dict[int, str], dict[str, dict[str, int]]]:
            aliases: dict[str, int] = {}
            inverse: dict[int, str] = {}
            members: dict[int, dict[str, int]] = {}
            keys: set[str] = set()
            for item in classes:
                handle, name = int(item.handle), str(item.name)
                if handle <= 0 or not name or handle in inverse or name in keys:
                    raise ValueError("invalid authoritative FOM class binding")
                keys.add(name)
                inverse[handle] = name
                member_map: dict[str, int] = {}
                seen: set[int] = set()
                for member in item.members:
                    value = int(member.handle)
                    if value <= 0 or not member.name or value in seen or member.name in member_map:
                        raise ValueError("invalid authoritative FOM member binding")
                    seen.add(value)
                    member_map[str(member.name)] = value
                members[handle] = member_map
                for alias in (name, name.removeprefix(root + "."), name.rsplit(".", 1)[-1]):
                    aliases[alias] = handle if aliases.get(alias, handle) == handle else 0
            for handle, name in inverse.items():
                short = name.rsplit(".", 1)[-1]
                if aliases.get(short) == handle:
                    inverse[handle] = short
            by_alias = {name: members[handle] for name, handle in aliases.items() if handle}
            return aliases, inverse, by_alias

        objects, inverse_objects, attributes = tables(view.object_classes, "HLAobjectRoot")
        interactions, inverse_interactions, parameters = tables(
            view.interaction_classes, "HLAinteractionRoot"
        )
        bindings = {}
        for kind, classes in (
            ("object", view.object_classes),
            ("interaction", view.interaction_classes),
        ):
            for item in classes:
                bindings[(kind, str(item.name))] = (
                    int(item.handle),
                    {str(member.name): int(member.handle) for member in item.members},
                )
        if not replace:
            for key, (handle, members) in getattr(self, "_authoritative_bindings", {}).items():
                current = bindings.get(key)
                if (
                    current is None
                    or current[0] != handle
                    or any(current[1].get(name) != value for name, value in members.items())
                ):
                    raise ValueError("authoritative FOM refresh changed an existing binding")
        for mapping in attributes.values():
            if "HLAprivilegeToDeleteObject" in mapping:
                mapping.setdefault("HLAprivilegeToDelete", mapping["HLAprivilegeToDeleteObject"])
        self._object_class_handles = objects
        self._inverse_object_class_handles = inverse_objects
        self._interaction_handles = interactions
        self._inverse_interaction_handles = inverse_interactions
        self._attribute_handle_cache = attributes
        self._authoritative_parameters = parameters
        self._fom_cache = result.fom if result is not None else None
        self._mim_module = selected_mim
        self._fom_modules = tuple(sources)
        federation_name = getattr(self, "_federation_name", None)
        if federation_name:
            if not hasattr(self, "_requested_mims"):
                self._requested_mims = {}
            self._requested_mims[federation_name] = selected_mim
        self._fom_revision = int(view.revision)
        self._authoritative_bindings = bindings

    async def refresh_fom(self, federate_handle: int) -> None:
        from rti.v1 import common_pb2, federation_pb2

        serial = getattr(self, "_restore_serial", 0)
        view = await self.federation.GetFOMView(
            federation_pb2.GetFOMViewRequest(
                wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
                federation_name=self._federation_name or "",
                federate_handle=federate_handle,
            )
        )
        if serial == getattr(self, "_restore_serial", 0):
            restoring = getattr(self, "_restore_applying", False)
            if restoring or int(view.revision) >= getattr(self, "_fom_revision", 0):
                self._install_fom_view(view, replace=restoring)

    def _populate_handle_tables(
        self, fom_paths: Sequence[str | Path], *, mim_module: Any = None
    ) -> None:
        """Parse the FOM modules + cache name → handle maps.

        Mirrors the Go-side ``fomHandle.LookupInteractionClass``:
        handles are 1-based indices over the sort-by-name interaction
        class list (with HLAinteractionRoot included via the MIM merge).
        """
        from rti1516e.fom import parse

        if not fom_paths and mim_module is None:
            return
        result = parse([Path(p) for p in fom_paths], mim_module=mim_module)
        if result.diagnostics or result.fom is None:
            # Don't fail here — the rtid will reject the FOM if it's
            # really bad, and the Python-side parser may be stricter
            # about HLA built-ins than the Go side.
            return
        fom = result.fom
        # M12 W2: cache the parsed FOM so attribute name → handle
        # lookups (used by publish_object_class / subscribe_object_class /
        # update_attributes) can walk it without re-parsing.
        self._fom_cache = fom
        self._mim_module = fom.mim_module
        # Replicate Go-side MIM merge for handle parity. The Go side
        # injects HLAinteractionRoot before the user-defined classes;
        # see rti/pkg/fom/mim.Merge. The Python parser does the same
        # logical work but cut-1 just sorts the resolved leaf names —
        # which produces handle 1 = "ConsumerAck", handle 2 =
        # "HLAinteractionRoot", handle 3 = "ProducerOutput" for the
        # bridge FOM. The Go side similarly sorts after MIM merge so
        # both ends agree.
        for idx, ic in enumerate(sorted(fom.interaction_classes, key=lambda c: c.name)):
            handle = idx + 1
            self._interaction_handles[ic.name] = handle
            self._inverse_interaction_handles[handle] = ic.name
        for idx, oc in enumerate(sorted(fom.object_classes, key=lambda c: c.name)):
            self._object_class_handles[oc.name] = idx + 1

    def _interaction_handle_for(self, class_name: str) -> int:
        """Return the numeric handle for ``class_name``; 0 on miss.

        Returning 0 (rather than raising) lets the cross-language test
        progress past optional class names — the Go side will reject
        the RPC and surface a typed error the test can assert on.
        """
        return self._interaction_handles.get(class_name, 0)

    def _parameter_indices_for(self, class_name: str) -> dict[str, int]:
        """Return parameter-name → 1-based index for ``class_name``.

        Cached lazily off the FOM. Classes absent from the local FOM fall back
        to a single synthetic parameter. The send path separately accepts the
        bridge's ``_payload`` alias for a class with one declared parameter.
        """
        authoritative = getattr(self, "_authoritative_parameters", None)
        if authoritative is not None:
            return dict(authoritative.get(class_name, {}))
        fom_cache = getattr(self, "_fom_cache", None)
        if fom_cache is not None:
            for interaction in fom_cache.interaction_classes:
                if interaction.name == class_name:
                    return {
                        parameter.name: index
                        for index, parameter in enumerate(interaction.parameters, start=1)
                    }
        # Preserve the bridge's opaque-payload convention for classes
        # that are intentionally absent from the FOM.
        return {"_payload": 1}


# --- Helpers ---------------------------------------------------------------


def _read_fom_bytes(path: str | Path) -> bytes:
    """Read FOM XML bytes synchronously. Extracted so async callers can
    invoke this without tripping the ASYNC240 lint (filesystem in async).
    FOMs are tiny load-once payloads — the blocking is not material."""
    return Path(path).read_bytes()


_GENERATED_PATH_INSTALLED = False


def _ensure_generated_path() -> None:
    """Add bundled wire stubs as a fallback after explicitly selected paths.
    (namespaced ``rti.v1.*``) resolve. Idempotent + lazy: only runs the
    first time a gRPC code path opens a transport."""
    global _GENERATED_PATH_INSTALLED  # noqa: PLW0603 — module-level cache flag
    if _GENERATED_PATH_INSTALLED:
        return
    import sys

    generated = Path(__file__).resolve().parent / "_generated"
    if generated.is_dir():
        path_str = str(generated)
        if path_str not in sys.path:
            sys.path.append(path_str)
    _GENERATED_PATH_INSTALLED = True


def _mode_to_proto(mode: str) -> int:
    """Translate the FederationSpec.mode string to the proto enum value."""
    from rti.v1 import common_pb2

    if mode == "best-effort":
        return int(common_pb2.Mode.MODE_BEST_EFFORT)
    return int(common_pb2.Mode.MODE_VERBOSE)


def _coerce_payload(value: Any) -> bytes:
    """Coerce an arbitrary parameter payload to bytes for the wire."""
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, int):
        # 4 bytes BE matches HLAinteger32BE used by the bridge FOM.
        return int(value).to_bytes(4, byteorder="big", signed=False)
    return repr(value).encode("utf-8")


def _is_already_exists(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like FederationAlreadyExists.

    grpc.aio raises ``grpc.aio.AioRpcError``; we sniff via duck-typed
    .code() rather than importing grpc unconditionally so spec tests
    that don't touch grpc don't pay the import cost.
    """
    code_fn = getattr(exc, "code", None)
    if not callable(code_fn):
        return False
    try:
        code = code_fn()
    except Exception:  # noqa: BLE001
        return False
    name = getattr(code, "name", "") or str(code)
    return "ALREADY_EXISTS" in name


async def build_grpc_transport(
    url: str,
    *,
    ca_cert: bytes | None = None,
    client_cert: bytes | None = None,
    client_key: bytes | None = None,
    bearer_token: str | None = None,
) -> GrpcTransport:
    """Open a real ``grpc.aio`` channel for ``url`` and wrap it.

    Two URL schemes are supported:

      - ``grpc://host:port``  — plaintext ``grpc.aio.insecure_channel``.
      - ``grpcs://host:port`` — TLS-secured ``grpc.aio.secure_channel``.
        ``ca_cert`` (PEM bytes) populates ``root_certificates``; pass
        ``None`` to rely on the system trust store.

    M14 W3 — additional auth knobs:

      - ``client_cert`` + ``client_key`` (both PEM bytes) → mTLS. The
        rtid must have been started with ``--tls-client-ca`` set.
      - ``bearer_token`` → ``authorization: Bearer <token>`` metadata
        on every RPC. Combinable with TLS / mTLS.
    """
    import grpc

    channel_holder: list[Any] = []
    try:
        return _construct_grpc_transport(
            grpc,
            url,
            channel_holder=channel_holder,
            ca_cert=ca_cert,
            client_cert=client_cert,
            client_key=client_key,
            bearer_token=bearer_token,
        )
    except BaseException as construction_error:
        # _construct_grpc_transport publishes the channel through this small
        # holder before constructing generated stubs, so an import/constructor
        # failure cannot orphan a live grpc.aio channel.
        if not channel_holder:
            raise
        channel = channel_holder[0]
        cleanup = asyncio.create_task(channel.close())
        cancellation: asyncio.CancelledError | None = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
        try:
            cleanup.result()
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "gRPC transport construction and channel cleanup both failed",
                [construction_error, cleanup_error],
            ) from construction_error
        if cancellation is not None:
            raise BaseExceptionGroup(
                "gRPC transport construction failed while cleanup was cancelled",
                [construction_error, cancellation],
            ) from construction_error
        raise


def _construct_grpc_transport(
    grpc_module: Any,
    url: str,
    *,
    channel_holder: list[Any],
    ca_cert: bytes | None,
    client_cert: bytes | None,
    client_key: bytes | None,
    bearer_token: str | None,
) -> GrpcTransport:
    """Synchronously allocate a channel and bind generated service stubs."""

    channel: Any | None = None
    if url.startswith("grpcs://"):
        target = url.removeprefix("grpcs://")
        ssl_creds = grpc_module.ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=client_key,
            certificate_chain=client_cert,
        )
        if bearer_token:
            # M14 W3: composite credentials = TLS + per-call metadata.
            # Mirrors Go SDK's bearerCreds path which requires TLS too.
            call_creds = grpc_module.metadata_call_credentials(_bearer_token_plugin(bearer_token))
            ssl_creds = grpc_module.composite_channel_credentials(ssl_creds, call_creds)
        channel = grpc_module.aio.secure_channel(target, ssl_creds)
    elif url.startswith("grpc://"):
        if bearer_token:
            raise ValueError(
                "build_grpc_transport: bearer_token requires grpcs:// "
                "(matches Go SDK's RequireTransportSecurity contract)"
            )
        target = url.removeprefix("grpc://")
        channel = grpc_module.aio.insecure_channel(target)
    else:
        raise ValueError(
            f"build_grpc_transport: unsupported URL scheme in {url!r} "
            "(expected 'grpc://' or 'grpcs://')"
        )
    channel_holder.append(channel)
    return GrpcTransport(channel, url=url)


def _bearer_token_plugin(token: str) -> Any:
    """Return a grpc.AuthMetadataPlugin that attaches authorization:
    Bearer <token> to every RPC. M14 W3.

    Typed as ``Any`` because grpc.AuthMetadataPlugin is duck-typed at
    the call site; declaring the precise type would force a hard
    dependency on grpc's stubs.
    """

    def plugin(_context: Any, callback: Any) -> None:
        callback((("authorization", f"Bearer {token}"),), None)

    return plugin


def _object_callback_metadata(event: Any) -> Any:
    from .events import ObjectCallbackMetadata

    if not event.HasField("metadata"):
        return None
    value = event.metadata

    def optional(name: str) -> int | None:
        return int(getattr(value, name)) if value.HasField(name) else None

    return ObjectCallbackMetadata(
        sent_order=optional("sent_order"),
        received_order=optional("received_order"),
        transportation_type=optional("transportation_type"),
        producing_federate=optional("producing_federate"),
        message_retraction_handle=optional("message_retraction_handle"),
        received_regions=tuple(value.received_regions.region_handles)
        if value.HasField("received_regions")
        else None,
    )
