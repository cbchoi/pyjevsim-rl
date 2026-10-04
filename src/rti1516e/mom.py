"""Layer-1 SDK client for the MomService (M12 W3).

Thin async wrapper around the generated ``MomServiceStub`` that
exposes the cut-3 introspection surface for the IEEE 1516.1-2010 §10
Management Object Model: federates query the HLAfederation +
HLAfederate object snapshots tracked by the runtime.

Composition::

    async with RtiConnection.connect("grpc://...") as rti:
        async with rti.join_federation(spec, federate_name="alice") as fed:
            fed_attrs = await fed.mom.query_federation_attributes()
            print(fed_attrs.federate_handles)

            self_attrs = await fed.mom.query_federate_attributes(fed.handle)
            print(self_attrs.interactions_sent)

            instances = await fed.mom.enumerate_mom_instances()
            for inst in instances:
                print(inst.class_name, inst.federate_handle, inst.instance_name)

The ``fed.mom`` accessor (see :class:`Federate.mom` in
``connection.py``) lazily constructs one :class:`MomClient` per
federate, bound to the same gRPC channel + federation_name the
federate already holds.

Read-only contract: the proto ``MomService`` exposes only Query /
Enumerate RPCs. MOM mutations come from the runtime (federation
lifecycle hooks + per-federate counter increments off the dispatcher
fan-out), not from federate calls. See ``rti/internal/mom/manager.go``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rti1516e._grpc_errors import translate_rpc_error

if TYPE_CHECKING:  # pragma: no cover - type-check imports only
    import grpc


@dataclass(frozen=True)
class FederationAttributes:
    """Wire-shape snapshot of one HLAfederation MOM object's attributes.

    Mirrors ``rti.v1.QueryFederationAttributesResponse``. Frozen so the
    instance is safe to hold across event-loop iterations.
    """

    federation_name: str
    federate_handles: tuple[int, ...] = ()
    fom_module_names: tuple[str, ...] = ()
    attribute_values: dict[str, bytes] = field(default_factory=dict)
    unavailable_attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FederateAttributes:
    """Wire-shape snapshot of one HLAfederate MOM object's attributes.

    Mirrors ``rti.v1.QueryFederateAttributesResponse``. The ``found``
    flag distinguishes "tracked + populated" from "no record (resigned
    or never joined)" without raising — matches the §10 contract that
    a query against a non-extant instance returns empty attributes.
    """

    found: bool
    federate_handle: int = 0
    federate_name: str = ""
    federate_type: str = ""
    time_regulating: bool = False
    time_constrained: bool = False
    logical_time: int | float = 0.0
    lookahead: int | float = 0.0
    interactions_sent: int = 0
    interactions_received: int = 0
    updates_sent: int = 0
    reflections_received: int = 0
    attribute_values: dict[str, bytes] = field(default_factory=dict)
    unavailable_attributes: dict[str, str] = field(default_factory=dict)
    logical_time_value: int | float | None = None
    lookahead_value: int | float | None = None


@dataclass(frozen=True)
class MomInstance:
    """One MOM object instance: HLAfederation singleton or one HLAfederate.

    Mirrors ``rti.v1.MomInstance``.
    """

    class_name: str
    federate_handle: int
    instance_name: str


# IEEE 1516.1-2010 §10 standard MOM class names. Mirrors the constants
# used by the Go-side mom.Manager / mom service handler.
CLASS_HLA_FEDERATION = "HLAobjectRoot.HLAmanager.HLAfederation"
CLASS_HLA_FEDERATE = "HLAobjectRoot.HLAmanager.HLAfederate"


class MomClient:
    """Federate-bound client for the MomService gRPC surface.

    Construct via :meth:`Federate.mom` (in connection.py). The
    constructor takes a live ``grpc.aio.Channel`` plus the
    federation_name the federate has already joined under; every RPC
    populates ``wire_version=WIRE_VERSION_V1`` automatically.
    """

    def __init__(
        self,
        channel: grpc.aio.Channel,
        *,
        federation_name: str,
        federate_handle: int,
        logical_time_implementation_name: str = "",
    ) -> None:
        # Lazy import — the generated stubs live under
        # rti1516e/_generated/ and are only on sys.path after the
        # GrpcTransport's ``_ensure_generated_path`` ran.
        from rti.v1 import mom_pb2_grpc

        self._stub = mom_pb2_grpc.MomServiceStub(channel)
        self._federation_name = federation_name
        self._federate_handle = int(federate_handle)
        self._logical_time_implementation_name = logical_time_implementation_name

    async def query_federation_attributes(self) -> FederationAttributes:
        """Return the HLAfederation MOM object snapshot (§10.2).

        For an unknown federation the runtime returns zero-valued
        attributes (federation_name echoed from the request, empty
        lists). The MomClient surfaces this as a FederationAttributes
        with empty tuples; callers may distinguish via the empty
        federate_handles list when they want a "does this federation
        exist" probe.
        """
        from rti.v1 import common_pb2, mom_pb2

        req = mom_pb2.QueryFederationAttributesRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
        )
        try:
            resp = await self._stub.QueryFederationAttributes(req)
        except Exception as exc:  # noqa: BLE001 — translate_rpc_error reraises
            translate_rpc_error(exc)
        return FederationAttributes(
            federation_name=str(resp.federation_name),
            federate_handles=tuple(int(h) for h in resp.federate_handles),
            fom_module_names=tuple(str(n) for n in resp.fom_module_names),
            attribute_values=dict(getattr(resp, "attribute_values", {})),
            unavailable_attributes=dict(getattr(resp, "unavailable_attributes", {})),
        )

    async def query_federate_attributes(self, federate_handle: int) -> FederateAttributes:
        """Return one HLAfederate MOM object snapshot (§10.3).

        ``federate_handle`` may be any handle the runtime currently
        tracks (including the caller's own ``fed.handle``). When the
        runtime does not track the requested handle, the response's
        ``found`` flag is False and remaining fields are zero-valued.
        """
        from rti.v1 import common_pb2, mom_pb2

        req = mom_pb2.QueryFederateAttributesRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
            federate_handle=int(federate_handle),
        )
        try:
            resp = await self._stub.QueryFederateAttributes(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
        if not resp.found:
            return FederateAttributes(found=False)
        from ._logical_time import read_time

        logical_time = (
            read_time(resp.logical_time, "value", self._logical_time_implementation_name)
            if resp.HasField("logical_time")
            else None
        )
        lookahead = (
            read_time(resp.lookahead, "value", self._logical_time_implementation_name)
            if resp.HasField("lookahead")
            else None
        )
        return FederateAttributes(
            found=True,
            federate_handle=int(resp.federate_handle),
            federate_name=str(resp.federate_name),
            federate_type=str(resp.federate_type),
            time_regulating=bool(resp.time_regulating),
            time_constrained=bool(resp.time_constrained),
            logical_time=logical_time if logical_time is not None else 0,
            lookahead=lookahead if lookahead is not None else 0,
            logical_time_value=logical_time,
            lookahead_value=lookahead,
            interactions_sent=int(resp.interactions_sent),
            interactions_received=int(resp.interactions_received),
            updates_sent=int(resp.updates_sent),
            reflections_received=int(resp.reflections_received),
            attribute_values=dict(getattr(resp, "attribute_values", {})),
            unavailable_attributes=dict(getattr(resp, "unavailable_attributes", {})),
        )

    async def enumerate_mom_instances(self) -> list[MomInstance]:
        """Return the list of MOM object instances for the federation.

        Always begins with one HLAfederation entry (the singleton) when
        the federation is tracked, followed by one HLAfederate per
        joined federate. Ordering is deterministic — federates appear
        in sorted-handle order — so polling clients see stable rows
        across calls.
        """
        from rti.v1 import common_pb2, mom_pb2

        req = mom_pb2.EnumerateMomInstancesRequest(
            wire_version=common_pb2.WireVersion.WIRE_VERSION_V1,
            federation_name=self._federation_name,
        )
        try:
            resp = await self._stub.EnumerateMomInstances(req)
        except Exception as exc:  # noqa: BLE001
            translate_rpc_error(exc)
        return [
            MomInstance(
                class_name=str(inst.class_name),
                federate_handle=int(inst.federate_handle),
                instance_name=str(inst.instance_name),
            )
            for inst in resp.instances
        ]
