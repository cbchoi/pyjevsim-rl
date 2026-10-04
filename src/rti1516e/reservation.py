"""Layer-1 SDK client for object instance name reservation (M26 Phase F).

Thin async wrapper around the ObjectService reservation RPCs introduced
in M26 Phase F. The synchronous wire return is Empty; the actual
success/failure is delivered as an event on the federate's stream
(ObjectInstanceNameReservation{Succeeded,Failed} or the Multiple-name
variants). Federates that want the result back inline can use
``await_reservation`` which awaits the matching event.

Usage:

    async with rti.join_federation(spec, federate_name="alice") as fed:
        await fed.reservation.reserve("vehicle-1")
        # Wait for the result event
        result = await fed.reservation.await_reservation("vehicle-1")
        if isinstance(result, ObjectInstanceNameReservationSucceeded):
            await fed.register_object_instance("Vehicle", instance_name="vehicle-1")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rti1516e._grpc_errors import translate_rpc_error

if TYPE_CHECKING:  # pragma: no cover
    import grpc


class ReservationClient:
    """Federate-bound client for the object instance name reservation RPCs."""

    def __init__(
        self,
        channel: grpc.aio.Channel,
        *,
        federation_name: str,
        federate_handle: int,
    ) -> None:
        from rti.v1 import object_pb2_grpc

        self._stub = object_pb2_grpc.ObjectServiceStub(channel)
        self._federation_name = federation_name
        self._federate_handle = int(federate_handle)

    async def reserve(self, object_name: str) -> None:
        """§6.1 — request a single-name reservation.

        Result delivered asynchronously as
        ObjectInstanceNameReservation{Succeeded,Failed} on the
        federate's event stream.
        """
        from rti.v1 import common_pb2, object_pb2

        req = object_pb2.ReserveObjectInstanceNameRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_name=object_name,
        )
        try:
            await self._stub.ReserveObjectInstanceName(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def release(self, object_name: str) -> None:
        """§6.4 — release a previously-reserved name.

        Synchronous response: raises on the SDK side if the federate
        doesn't hold the reservation.
        """
        from rti.v1 import common_pb2, object_pb2

        req = object_pb2.ReleaseObjectInstanceNameRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_name=object_name,
        )
        try:
            await self._stub.ReleaseObjectInstanceName(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def reserve_multiple(self, object_names: list[str]) -> None:
        """§6.5 — request an atomic batch reservation.

        Result delivered asynchronously as
        MultipleObjectInstanceNameReservation{Succeeded,Failed}.
        """
        from rti.v1 import common_pb2, object_pb2

        req = object_pb2.ReserveMultipleObjectInstanceNamesRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_names=[str(n) for n in object_names],
        )
        try:
            await self._stub.ReserveMultipleObjectInstanceNames(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)

    async def release_multiple(self, object_names: list[str]) -> None:
        """Release the complete set atomically in one server transaction."""
        from rti.v1 import object_pb2

        req = object_pb2.ReleaseMultipleObjectInstanceNamesRequest(
            wire_version=1, federation_name=self._federation_name,
            federate_handle=self._federate_handle, object_names=list(object_names),
        )
        try:
            await self._stub.ReleaseMultipleObjectInstanceNames(req)
        except Exception as exc:
            translate_rpc_error(exc)
