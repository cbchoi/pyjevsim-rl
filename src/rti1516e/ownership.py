"""Layer-1 SDK client for the OwnershipService (M12 W2).

Thin async wrapper around the generated ``OwnershipServiceStub`` that
exposes the IEEE 1516-2010 §7 negotiated divest+acquire two-phase
protocol plus per-attribute ownership queries.

Composition:

    async with rti.join_federation(spec, federate_name="alice") as fed:
        await fed.ownership.negotiated_divest(obj, [attr], tag=b"...")

The ``fed.ownership`` accessor (see :class:`Federate.ownership` in
``connection.py``) lazily constructs one :class:`OwnershipClient` per
federate, bound to the same gRPC channel + federation_name +
federate_handle the federate already holds.

M12 W2 follow-up (deferral #1 closed): the proto ``FederateEvent`` oneof
now carries ``RequestAttributeOwnershipAssumption`` (tag 30),
``AttributeOwnershipAcquisitionNotification`` (tag 31), and
``RequestDivestitureConfirmation`` (tag 32). The corresponding
:mod:`rti1516e.events` dataclasses are yielded by ``Federate.events()``
as the transfer progresses; tests may assert on callback delivery in
addition to using Query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from rti1516e._grpc_errors import translate_rpc_error
from rti1516e.handles import (
    AttributeHandle,
    ObjectInstanceHandle,
)
from rti1516e.sets import AttributeHandleSet

if TYPE_CHECKING:  # pragma: no cover - type-check imports only
    import grpc

# M28 — IEEE 1516 portability type aliases. Public service entry points widen to
# accept typed handles + typed sets; internal protobuf request fields
# stay bare int via existing ``int(h)`` coercion.
_AttributeRef: TypeAlias = "int | AttributeHandle"
_ObjectInstanceRef: TypeAlias = "int | ObjectInstanceHandle"
_AttributeRefList: TypeAlias = "list[int | AttributeHandle] | AttributeHandleSet"


class OwnershipClient:
    """Federate-bound client for the OwnershipService gRPC surface."""

    def __init__(
        self,
        channel: grpc.aio.Channel,
        *,
        federation_name: str,
        federate_handle: int,
    ) -> None:
        from rti.v1 import ownership_pb2_grpc

        self._stub = ownership_pb2_grpc.OwnershipServiceStub(channel)
        self._federation_name = federation_name
        self._federate_handle = int(federate_handle)

    # --- Divest / acquire (§7.2-7.7) ----------------------------------------

    async def release_denied(self, object_handle: int, attribute_handles: list[int]) -> None:
        from rti.v1 import ownership_pb2

        request = ownership_pb2.ReleaseDeniedRequest(
            wire_version=1, federation_name=self._federation_name,
            federate_handle=self._federate_handle, object_handle=object_handle,
            attribute_handles=list(attribute_handles),
        )
        try:
            await self._stub.AttributeOwnershipReleaseDenied(request)
        except Exception as exc:
            translate_rpc_error(exc)
            raise

    async def unconditional_divest(
        self, object_handle: _ObjectInstanceRef, attribute_handles: _AttributeRefList
    ) -> None:
        """§7.2 — drop ownership without asking anyone to take it."""
        from rti.v1 import common_pb2, ownership_pb2

        req = ownership_pb2.UnconditionalDivestRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handles=[int(h) for h in attribute_handles],
        )
        try:
            await self._stub.UnconditionalAttributeOwnershipDivestiture(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def negotiated_divest(
        self,
        object_handle: _ObjectInstanceRef,
        attribute_handles: _AttributeRefList,
        *,
        tag: bytes = b"",
        two_phase: bool = False,
    ) -> None:
        """§7.3 — offer ownership; transfer fires when an acquirer arrives.

        ``two_phase=True`` (M37 wire flag, M39 HA-2 surface): when an
        acquirer engages, the transfer PARKS on the divester's
        requestDivestitureConfirmation callback and completes only on a
        subsequent :meth:`confirm_divestiture` (§7.6). ``False`` keeps
        the pre-M37 one-phase gorti flow.
        """
        from rti.v1 import common_pb2, ownership_pb2

        req = ownership_pb2.NegotiatedDivestRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handles=[int(h) for h in attribute_handles],
            tag=bytes(tag),
            two_phase=bool(two_phase),
        )
        try:
            await self._stub.NegotiatedAttributeOwnershipDivestiture(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def acquire(
        self,
        object_handle: _ObjectInstanceRef,
        attribute_handles: _AttributeRefList,
        *,
        tag: bytes = b"",
        if_available: bool = False,
    ) -> None:
        """§7.8 — request ownership; matched against pending divestitures.

        ``if_available=True`` (M37 wire flag, M39 HA-2 surface) selects
        the §7.9 attributeOwnershipAcquisitionIfAvailable semantics:
        only currently-unowned attributes transfer, nothing is queued,
        and the unavailable subset comes back via the §7.10
        attributeOwnershipUnavailable callback.
        """
        from rti.v1 import common_pb2, ownership_pb2

        req = ownership_pb2.AcquireRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handles=[int(h) for h in attribute_handles],
            tag=bytes(tag),
            if_available=bool(if_available),
        )
        try:
            await self._stub.AttributeOwnershipAcquisition(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def confirm_divestiture(
        self,
        object_handle: _ObjectInstanceRef,
        attribute_handles: _AttributeRefList,
        *,
        tag: bytes = b"",
    ) -> None:
        """§7.6 confirmDivestiture (M37 RPC, M39 HA-2 surface).

        Completes a parked two-phase negotiated divest (see
        :meth:`negotiated_divest` with ``two_phase=True``): the queued
        acquirer becomes the owner atomically.
        """
        from rti.v1 import common_pb2, ownership_pb2

        req = ownership_pb2.ConfirmDivestitureRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handles=[int(h) for h in attribute_handles],
            tag=tag,
        )
        try:
            await self._stub.ConfirmDivestiture(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def cancel_negotiated_divest(
        self, object_handle: _ObjectInstanceRef, attribute_handles: _AttributeRefList
    ) -> None:
        """§7.5 — withdraw a pending negotiated divestiture."""
        from rti.v1 import common_pb2, ownership_pb2

        req = ownership_pb2.CancelDivestRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handles=[int(h) for h in attribute_handles],
        )
        try:
            await self._stub.CancelNegotiatedAttributeOwnershipDivestiture(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def cancel_acquire(
        self, object_handle: _ObjectInstanceRef, attribute_handles: _AttributeRefList
    ) -> None:
        """§7.6 — withdraw a pending acquisition."""
        from rti.v1 import common_pb2, ownership_pb2

        req = ownership_pb2.CancelAcquireRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handles=[int(h) for h in attribute_handles],
        )
        try:
            await self._stub.CancelAttributeOwnershipAcquisition(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def divest_if_wanted(
        self, object_handle: _ObjectInstanceRef, attribute_handles: _AttributeRefList
    ) -> None:
        """§7.7 — transfer ownership only if a matching acquirer is queued."""
        from rti.v1 import common_pb2, ownership_pb2

        req = ownership_pb2.DivestIfWantedRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handles=[int(h) for h in attribute_handles],
        )
        try:
            await self._stub.AttributeOwnershipDivestitureIfWanted(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def divest_if_wanted_with_result(
        self, object_handle: _ObjectInstanceRef, attribute_handles: _AttributeRefList
    ) -> tuple[int, ...]:
        """Return the attributes transferred by this exact operation."""
        from rti.v1 import common_pb2, ownership_pb2

        req = ownership_pb2.DivestIfWantedRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handles=[int(h) for h in attribute_handles],
        )
        try:
            resp = await self._stub.AttributeOwnershipDivestitureIfWantedWithResult(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
        return tuple(int(h) for h in resp.divested_attribute_handles)

    # --- Queries (§7.8-7.9) -------------------------------------------------

    async def query_attribute_ownership(
        self, object_handle: _ObjectInstanceRef, attribute_handle: _AttributeRef
    ) -> tuple[int, bool]:
        """§7.8 — return ``(owner_handle, owned)``.

        ``owned=False`` indicates the attribute is mid-transfer or
        unowned (post-unconditional-divest with no acquirer yet); in
        that case ``owner_handle`` is 0 and should be ignored.
        """
        from rti.v1 import common_pb2, ownership_pb2

        req = ownership_pb2.QueryOwnershipRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handle=int(attribute_handle),
        )
        try:
            resp = await self._stub.QueryAttributeOwnership(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
        return int(resp.owner_federate_handle), bool(resp.owned)

    async def is_attribute_owned_by_federate(
        self, object_handle: _ObjectInstanceRef, attribute_handle: _AttributeRef
    ) -> bool:
        """§7.9 — return True iff this federate currently owns the attribute."""
        from rti.v1 import common_pb2, ownership_pb2

        req = ownership_pb2.IsOwnedRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handle=int(attribute_handle),
        )
        try:
            resp = await self._stub.IsAttributeOwnedByFederate(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
        return bool(resp.owned)
