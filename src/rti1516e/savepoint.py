"""Layer-1 SDK client for the SavepointService.

Thin async wrapper around the generated ``SavepointServiceStub`` that
exposes IEEE 1516-2010 §4.8-4.15 federation save/restore.

Composition:

    async with rti.join_federation(spec, federate_name="alice") as fed:
        await fed.savepoint.request_federation_save("checkpoint-1")
        await fed.savepoint.federate_save_begun()
        await fed.savepoint.federate_save_complete()
        state = await fed.savepoint.query_save_state("checkpoint-1")
        # state == SaveState.SAVED

The ``fed.savepoint`` accessor (see :class:`Federate.savepoint` in
``connection.py``) lazily constructs one :class:`SavepointClient` per
federate, bound to the same gRPC channel + federation_name +
federate_handle the federate already holds.

The proto ``FederateEvent`` oneof carries ``InitiateFederateSave`` (tag 40),
``FederationSaved`` (tag 41), and ``FederationNotSaved`` (tag 42).
Federates receive these callbacks over StreamService.Events as
:class:`rti1516e.events.InitiateFederateSave`,
:class:`rti1516e.events.FederationSaved`, and
:class:`rti1516e.events.FederationNotSaved`. Restore lifecycle
callbacks also arrive through StreamService.Events. Per-federate progress
is available through the federation status queries; label-specific outcome
queries remain available separately. Completion never implicitly begins a save.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from rti1516e._grpc_errors import translate_rpc_error

if TYPE_CHECKING:  # pragma: no cover - type-check imports only
    import grpc


class SaveState(IntEnum):
    """Mirrors ``rti.v1.SaveState`` for clean Pythonic comparison.

    Values intentionally match the proto enum integer values so this
    enum can be used as a drop-in replacement when the SDK exposes
    state to user code.
    """

    UNSPECIFIED = 0
    IDLE = 1
    INITIATED = 2
    SAVED = 3
    NOT_SAVED = 4


class RestoreState(IntEnum):
    """Mirrors ``rti.v1.RestoreState``."""

    UNSPECIFIED = 0
    IDLE = 1
    LOADING = 2
    INITIATED = 3
    COMPLETED = 4
    FAILED = 5


class SaveStatus(IntEnum):
    """Per-federate progress, distinct from a label's save outcome."""

    UNSPECIFIED = 0
    NO_SAVE_IN_PROGRESS = 1
    FEDERATE_INSTRUCTED_TO_SAVE = 2
    FEDERATE_SAVING = 3
    FEDERATE_WAITING_FOR_FEDERATION_TO_SAVE = 4


class RestoreStatus(IntEnum):
    """Per-federate restore progress."""

    UNSPECIFIED = 0
    NO_RESTORE_IN_PROGRESS = 1
    FEDERATE_RESTORE_REQUEST_PENDING = 2
    FEDERATE_WAITING_FOR_RESTORE_TO_BEGIN = 3
    FEDERATE_PREPARED_TO_RESTORE = 4
    FEDERATE_RESTORING = 5
    FEDERATE_WAITING_FOR_FEDERATION_TO_RESTORE = 6


@dataclass(frozen=True)
class FederateSaveStatus:
    federate_handle: int
    status: SaveStatus


@dataclass(frozen=True)
class FederateRestoreStatus:
    pre_restore_handle: int
    post_restore_handle: int
    status: RestoreStatus


@dataclass(frozen=True)
class SaveStatusResponse:
    state: SaveState
    active_label: str
    federate_statuses: tuple[FederateSaveStatus, ...]


@dataclass(frozen=True)
class RestoreStatusResponse:
    state: RestoreState
    active_label: str
    federate_statuses: tuple[FederateRestoreStatus, ...]


class SavepointClient:
    """Federate-bound client for the SavepointService gRPC surface."""

    def __init__(
        self,
        channel: grpc.aio.Channel,
        *,
        federation_name: str,
        federate_handle: int,
        session_token: bytes = b"",
        logical_time_implementation_name: str = "",
        checkpoint_provider: Callable[[], bytes] | None = None,
        restore_complete_guard: Callable[[], None] | None = None,
    ) -> None:
        from rti.v1 import savepoint_pb2_grpc

        self._stub = savepoint_pb2_grpc.SavepointServiceStub(channel)
        self._federation_name = federation_name
        self._federate_handle = int(federate_handle)
        self._session_token = bytes(session_token)
        self._logical_time_implementation_name = logical_time_implementation_name
        self._checkpoint_provider = checkpoint_provider
        self._restore_complete_guard = restore_complete_guard

    def _query_options(self) -> dict[str, tuple[tuple[str, bytes], ...]]:
        token = getattr(self, "_session_token", b"")
        return {"metadata": (("gorti-session-token-bin", token),)} if token else {}

    # --- Save protocol (§4.8-4.11) ------------------------------------------

    async def request_federation_save(self, label: str, *, save_time: float | None = None) -> None:
        """§4.8 — start a save. Optional ``save_time`` pins it to logical time.

        ``save_time=None`` saves now at the current sync point;
        passing a float forwards a scheduled request. The current server
        explicitly rejects scheduled saves as unsupported.
        """
        from rti.v1 import common_pb2, savepoint_pb2

        req = savepoint_pb2.RequestFederationSaveRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            label=label,
        )
        if save_time is not None:
            from ._logical_time import write_time

            write_time(req, "save_time", save_time, self._logical_time_implementation_name)
        try:
            await self._stub.RequestFederationSave(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def federate_save_begun(self) -> None:
        """Explicitly report that this federate has begun saving."""
        from rti.v1 import common_pb2, savepoint_pb2

        req = savepoint_pb2.FederateSaveResponseRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
        )
        try:
            await self._stub.FederateSaveBegun(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def federate_save_complete(self) -> None:
        """§4.10 — notify the RTI this federate has saved successfully."""
        from rti.v1 import common_pb2, savepoint_pb2

        req = savepoint_pb2.FederateSaveResponseRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
        )
        provider = getattr(self, "_checkpoint_provider", None)
        if provider is not None:
            req.client_state = provider()
        try:
            await self._stub.FederateSaveComplete(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def federate_save_not_complete(self) -> None:
        """§4.10 — notify the RTI this federate failed to save."""
        from rti.v1 import common_pb2, savepoint_pb2

        req = savepoint_pb2.FederateSaveResponseRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
        )
        try:
            await self._stub.FederateSaveNotComplete(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def abort_federation_save(self) -> None:
        """Abort the active federation save as this joined federate."""
        from rti.v1 import common_pb2, savepoint_pb2

        req = savepoint_pb2.AbortFederationSaveRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
        )
        try:
            await self._stub.AbortFederationSave(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def query_save_state(self, label: str) -> SaveState:
        """§4.11 — return the current save state for (federation, label)."""
        return (await self.query_federation_save_status(label)).state

    async def query_federation_save_status(self, label: str = "") -> SaveStatusResponse:
        """Return the current vector and optional legacy label-specific outcome."""
        from rti.v1 import common_pb2, savepoint_pb2

        req = savepoint_pb2.QuerySaveStateRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            label=label,
        )
        try:
            resp = await self._stub.QuerySaveState(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
        return SaveStatusResponse(
            state=SaveState(int(resp.state)),
            active_label=resp.active_label,
            federate_statuses=tuple(
                FederateSaveStatus(entry.federate_handle, SaveStatus(entry.status))
                for entry in resp.federate_statuses
            ),
        )

    # --- Restore protocol (§4.12-4.15) --------------------------------------

    async def request_federation_restore(
        self, label: str, *, saved_federation_generation: int | None = None,
    ) -> None:
        """Restore a label, optionally selecting an exact saved execution."""
        from rti.v1 import common_pb2, savepoint_pb2

        req = savepoint_pb2.RequestFederationRestoreRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            label=label,
        )
        if saved_federation_generation is not None:
            req.saved_federation_generation = saved_federation_generation
        try:
            await self._stub.RequestFederationRestore(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def federate_restore_complete(self) -> None:
        """§4.14 — notify the RTI this federate has restored successfully."""
        from rti.v1 import common_pb2, savepoint_pb2

        guard = getattr(self, "_restore_complete_guard", None)
        if not callable(guard):
            raise RuntimeError("no validated SDK restore stage is ready")
        guard()
        req = savepoint_pb2.FederateRestoreResponseRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
        )
        try:
            await self._stub.FederateRestoreComplete(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def abort_federation_restore(self) -> None:
        """Abort the active federation restore as this joined federate."""
        from rti.v1 import common_pb2, savepoint_pb2

        req = savepoint_pb2.AbortFederationRestoreRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
        )
        try:
            await self._stub.AbortFederationRestore(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def federate_restore_not_complete(self) -> None:
        """Report this participant's restore failure without aborting its peers."""
        from rti.v1 import common_pb2, savepoint_pb2

        req = savepoint_pb2.FederateRestoreResponseRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
        )
        try:
            await self._stub.FederateRestoreNotComplete(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def query_restore_state(self, label: str) -> RestoreState:
        """§4.15 — return the current restore state for (federation, label)."""
        return (await self.query_federation_restore_status(label)).state

    async def query_federation_restore_status(self, label: str = "") -> RestoreStatusResponse:
        """Return current pre/post-restore handles and each participant's status."""
        from rti.v1 import common_pb2, savepoint_pb2

        req = savepoint_pb2.QueryRestoreStateRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            label=label,
        )
        try:
            resp = await self._stub.QueryRestoreState(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
        return RestoreStatusResponse(
            state=RestoreState(int(resp.state)),
            active_label=resp.active_label,
            federate_statuses=tuple(
                FederateRestoreStatus(
                    entry.pre_restore_handle,
                    entry.post_restore_handle,
                    RestoreStatus(entry.status),
                )
                for entry in resp.federate_statuses
            ),
        )
