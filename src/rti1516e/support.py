"""Layer-1 SDK client for the SupportService (M25 Phase B).

Thin async wrapper around the generated ``SupportServiceStub`` exposing
IEEE 1516.1-2010 §10.2 handle / name / dimension / order / transport
lookup services.

Composition mirrors the cut-3 service-group clients (sync/ddm/...):

    async with RtiConnection.connect("grpc://...") as rti:
        async with rti.join_federation(spec, federate_name="alice") as fed:
            vehicle_h = await fed.support.get_object_class_handle("Vehicle")
            pos_h = await fed.support.get_attribute_handle(vehicle_h, "Position")

The Layer-2 ``Rti1516eAmbassador`` exposes equivalent sync wrappers
(``getObjectClassHandle`` / ``getObjectClassName`` / ...) per reference_rti
naming for federates ported from commercial RTIs.

All lookups are read-only against the federation's FOM and do not
mutate federate state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rti1516e._grpc_errors import translate_rpc_error

if TYPE_CHECKING:  # pragma: no cover - type-check imports only
    import grpc


# Mirror the Go-side constants in rti/internal/transport/grpc/support.go.
ORDER_TYPE_RECEIVE = 1
ORDER_TYPE_TIMESTAMP = 2
TRANSPORT_RELIABLE = 1
TRANSPORT_BEST_EFFORT = 2


class SupportClient:
    """Federate-bound client for FOM and runtime support queries."""

    async def normalize_federate_handle(self, target_handle: int) -> int:
        from rti.v1 import support_pb2

        req = support_pb2.NormalizeFederateHandleRequest(
            wire_version=1, federation_name=self._federation_name,
            federate_handle=self._federate_handle, target_handle=target_handle,
        )
        try:
            response = await self._stub.NormalizeFederateHandle(req)
        except Exception as exc:
            translate_rpc_error(exc)
            raise
        return int(response.value)

    async def normalize_service_group(self, service_group: int) -> int:
        from rti.v1 import support_pb2

        req = support_pb2.NormalizeServiceGroupRequest(
            wire_version=1, federation_name=self._federation_name,
            federate_handle=self._federate_handle, service_group=service_group,
        )
        try:
            response = await self._stub.NormalizeServiceGroup(req)
        except Exception as exc:
            translate_rpc_error(exc)
            raise
        return int(response.value)

    async def get_update_rate_value(self, update_rate_designator: str) -> float:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetUpdateRateValueRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            update_rate_designator=update_rate_designator,
        )
        try:
            resp = await self._stub.GetUpdateRateValue(req)
        except Exception as exc:
            translate_rpc_error(exc)
            raise
        return float(resp.update_rate)

    async def get_update_rate_value_for_attribute(
        self, object_handle: int, attribute_handle: int
    ) -> float:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetUpdateRateValueForAttributeRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
            attribute_handle=int(attribute_handle),
        )
        try:
            resp = await self._stub.GetUpdateRateValueForAttribute(req)
        except Exception as exc:
            translate_rpc_error(exc)
            raise
        return float(resp.update_rate)

    async def get_available_dimensions_for_class_attribute(
        self, class_handle: int, attribute_handle: int
    ) -> list[int]:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetAvailableDimensionsForClassAttributeRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_class_handle=int(class_handle),
            attribute_handle=int(attribute_handle),
        )
        try:
            resp = await self._stub.GetAvailableDimensionsForClassAttribute(req)
        except Exception as exc:
            translate_rpc_error(exc)
            raise
        return [int(value) for value in resp.dimension_handles]

    async def get_available_dimensions_for_interaction_class(self, class_handle: int) -> list[int]:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetAvailableDimensionsForInteractionClassRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            interaction_class_handle=int(class_handle),
        )
        try:
            resp = await self._stub.GetAvailableDimensionsForInteractionClass(req)
        except Exception as exc:
            translate_rpc_error(exc)
            raise
        return [int(value) for value in resp.dimension_handles]

    async def get_dimension_handle_set(self, region_handle: int) -> list[int]:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetDimensionHandleSetRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            region_handle=int(region_handle),
        )
        try:
            resp = await self._stub.GetDimensionHandleSet(req)
        except Exception as exc:
            translate_rpc_error(exc)
            raise
        return [int(value) for value in resp.dimension_handles]

    def __init__(
        self,
        channel: grpc.aio.Channel,
        *,
        federation_name: str,
        federate_handle: int = 0,
        session_token: bytes = b"",
    ) -> None:
        from rti.v1 import support_pb2_grpc

        self._stub = support_pb2_grpc.SupportServiceStub(channel)
        self._federation_name = federation_name
        self._federate_handle = int(federate_handle)
        self._session_token = bytes(session_token)

    def _query_options(self) -> dict[str, tuple[tuple[str, bytes], ...]]:
        token = getattr(self, "_session_token", b"")
        return {"metadata": (("gorti-session-token-bin", token),)} if token else {}

    async def get_object_class_handle(self, class_name: str) -> int:
        """§10.2 — resolve an object class name to its handle."""
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetObjectClassHandleRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            class_name=class_name,
        )
        try:
            resp = await self._stub.GetObjectClassHandle(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc, operation="GetObjectClassHandle")
            raise
        return int(resp.class_handle)

    async def get_object_class_name(self, class_handle: int) -> str:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetObjectClassNameRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            class_handle=int(class_handle),
        )
        try:
            resp = await self._stub.GetObjectClassName(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return str(resp.class_name)

    async def get_attribute_handle(self, class_handle: int, attribute_name: str) -> int:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetAttributeHandleRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            class_handle=int(class_handle),
            attribute_name=attribute_name,
        )
        try:
            resp = await self._stub.GetAttributeHandle(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return int(resp.attribute_handle)

    async def get_attribute_name(self, class_handle: int, attribute_handle: int) -> str:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetAttributeNameRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            class_handle=int(class_handle),
            attribute_handle=int(attribute_handle),
        )
        try:
            resp = await self._stub.GetAttributeName(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return str(resp.attribute_name)

    async def get_interaction_class_handle(self, class_name: str) -> int:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetInteractionClassHandleRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            class_name=class_name,
        )
        try:
            resp = await self._stub.GetInteractionClassHandle(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return int(resp.class_handle)

    async def get_interaction_class_name(self, class_handle: int) -> str:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetInteractionClassNameRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            class_handle=int(class_handle),
        )
        try:
            resp = await self._stub.GetInteractionClassName(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return str(resp.class_name)

    async def get_parameter_handle(self, class_handle: int, parameter_name: str) -> int:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetParameterHandleRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            class_handle=int(class_handle),
            parameter_name=parameter_name,
        )
        try:
            resp = await self._stub.GetParameterHandle(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return int(resp.parameter_handle)

    async def get_parameter_name(self, class_handle: int, parameter_handle: int) -> str:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetParameterNameRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            class_handle=int(class_handle),
            parameter_handle=int(parameter_handle),
        )
        try:
            resp = await self._stub.GetParameterName(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return str(resp.parameter_name)

    async def get_dimension_handle(self, dimension_name: str) -> int:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetDimensionHandleRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            dimension_name=dimension_name,
        )
        try:
            resp = await self._stub.GetDimensionHandle(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return int(resp.dimension_handle)

    async def get_dimension_name(self, dimension_handle: int) -> str:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetDimensionNameRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            dimension_handle=int(dimension_handle),
        )
        try:
            resp = await self._stub.GetDimensionName(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return str(resp.dimension_name)

    async def get_dimension_upper_bound(self, dimension_handle: int) -> int:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetDimensionUpperBoundRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            dimension_handle=int(dimension_handle),
        )
        try:
            resp = await self._stub.GetDimensionUpperBound(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return int(resp.upper_bound)

    async def get_order_type(self, order_name: str) -> int:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetOrderTypeRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            order_name=order_name,
        )
        try:
            resp = await self._stub.GetOrderType(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return int(resp.order_type)

    async def get_order_name(self, order_type: int) -> str:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetOrderNameRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            order_type=int(order_type),
        )
        try:
            resp = await self._stub.GetOrderName(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return str(resp.order_name)

    async def get_transportation_type(self, transportation_name: str) -> int:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetTransportationTypeRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            transportation_name=transportation_name,
        )
        try:
            resp = await self._stub.GetTransportationType(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return int(resp.transportation_type)

    async def get_transportation_name(self, transportation_type: int) -> str:
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetTransportationNameRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            transportation_type=int(transportation_type),
        )
        try:
            resp = await self._stub.GetTransportationName(req, **self._query_options())
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return str(resp.transportation_name)

    async def get_object_instance_handle(self, object_name: str) -> int:
        """§6.30 — resolve a runtime object instance name to its handle.

        M27 Phase C. Raises if no instance with that name is registered.
        """
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetObjectInstanceHandleRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            object_name=object_name,
            federate_handle=self._federate_handle,
        )
        try:
            resp = await self._stub.GetObjectInstanceHandle(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return int(resp.object_handle)

    async def get_object_instance_name(self, object_handle: int) -> str:
        """§6.31 — resolve a runtime object instance handle to its name."""
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetObjectInstanceNameRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            object_handle=int(object_handle),
            federate_handle=self._federate_handle,
        )
        try:
            resp = await self._stub.GetObjectInstanceName(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
            raise
        return str(resp.object_name)

    async def get_known_object_class_handle(self, object_handle: int) -> int:
        """Return the class of an object known to this federate."""
        from rti.v1 import common_pb2, support_pb2

        req = support_pb2.GetKnownObjectClassHandleRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=self._federate_handle,
            object_handle=int(object_handle),
        )
        try:
            resp = await self._stub.GetKnownObjectClassHandle(req)
        except Exception as exc:
            translate_rpc_error(exc)
            raise
        return int(resp.class_handle)
