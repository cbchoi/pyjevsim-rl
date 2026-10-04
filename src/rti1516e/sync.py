"""Layer-1 SDK client for the SyncService (M12 W2).

Thin async wrapper around the generated ``SyncServiceStub`` that
exposes IEEE 1516-2010 §4.6-4.7 synchronization-management primitives
to federates.

Composition:

    async with RtiConnection.connect("grpc://...") as rti:
        async with rti.join_federation(spec, federate_name="alice") as fed:
            await fed.sync.register_synchronization_point("phase1")
            await fed.sync.synchronization_point_achieved("phase1")

The ``fed.sync`` accessor (see :class:`Federate.sync` in
``connection.py``) lazily constructs one :class:`SyncClient` per
federate, bound to the same gRPC channel + federation_name +
federate_handle the federate already holds.

This client is intentionally minimal: it forwards each RPC to the
generated stub, maps gRPC ``StatusCode`` errors onto the SDK's typed
``RtiError`` hierarchy via :func:`rti1516e._grpc_errors.translate_rpc_error`,
and otherwise stays out of the way.

M12 W2 follow-up (deferral #1 closed): the proto ``FederateEvent`` oneof
now carries ``SynchronizationPointAnnounced`` (tag 20) and
``FederationSynchronized`` (tag 21). Federates receive both callbacks
over the standard StreamService.Events stream; the typed event is
yielded as :class:`rti1516e.events.SynchronizationPointAnnounced` /
:class:`rti1516e.events.FederationSynchronized` from
``Federate.events()``. Tests can therefore assert on callback delivery
in addition to (or instead of) Query observation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rti1516e._grpc_errors import translate_rpc_error

if TYPE_CHECKING:  # pragma: no cover - type-check imports only
    import grpc


class SyncClient:
    """Federate-bound client for the SyncService gRPC surface.

    Construct via :meth:`Federate.sync` (in connection.py). The
    constructor takes a live ``grpc.aio.Channel`` plus the federation
    name + federate handle the federate has already joined under;
    every RPC populates ``wire_version=WIRE_VERSION_V1`` and the
    bound identity automatically.
    """

    def __init__(
        self,
        channel: grpc.aio.Channel,
        *,
        federation_name: str,
        federate_handle: int,
    ) -> None:
        # Lazy import — the generated stubs live under
        # rti1516e/_generated/ and are only on sys.path after the
        # GrpcTransport's ``_ensure_generated_path`` ran. Importing
        # here (not at module load) keeps memory:// transports free
        # of a hard grpcio-tools dependency.
        from rti.v1 import sync_pb2_grpc

        self._stub = sync_pb2_grpc.SyncServiceStub(channel)
        self._federation_name = federation_name
        self._federate_handle = int(federate_handle)

    async def register_synchronization_point(
        self,
        label: str,
        *,
        tag: bytes = b"",
        required_federates: list[int] | None = None,
    ) -> None:
        """Register a synchronization point (§4.6).

        ``required_federates=None`` means "all currently joined
        federates" per the manager's MembersResolver semantics; an
        explicit list pins the required set.
        """
        from rti.v1 import common_pb2, sync_pb2

        req = sync_pb2.RegisterSyncPointRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            label=label,
            tag=bytes(tag),
        )
        if required_federates is not None:
            req.required_federates.extend(int(h) for h in required_federates)
        try:
            await self._stub.RegisterFederationSynchronizationPoint(req)
        except Exception as exc:  # noqa: BLE001 — translate_rpc_error reraises
            translate_rpc_error(exc)

    async def synchronization_point_achieved(
        self, label: str, *, successfully: bool = True
    ) -> None:
        """Signal that this federate has achieved ``label`` (§4.14).

        When all required federates have called this, the RTI
        transitions the sync point to Achieved and emits
        ``federationSynchronized``.

        ``successfully=False`` (M37 wire field, M39 HA-2 surface, IEEE
        §4.14) still counts toward the transition but lands this
        federate in ``FederationSynchronized.failed_to_sync`` (§4.15).
        """
        from rti.v1 import common_pb2, sync_pb2

        req = sync_pb2.AchieveSyncPointRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            label=label,
            successfully=bool(successfully),
        )
        try:
            await self._stub.SynchronizationPointAchieved(req)
        except Exception as exc:  # noqa: BLE001 — translate_rpc_error reraises
            translate_rpc_error(exc)
