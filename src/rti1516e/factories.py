"""IEEE 1516 service-style factory accessors for typed handles and typed collections.

reference_rti federates obtain empty collections via ambassador-exposed
factories (``rtiAmb.getAttributeHandleSetFactory().create()``). These
factory classes back the W2 ambassador slots and the per-handle
``.decode(bytes)`` factories reference_rti exposes for handle deserialization.

NOTE: handle ``.decode(bytes)`` is stubbed for W1 (raises
``NotImplementedError``); the wire shape is uint64 over gRPC and the
production decode path lives in the transport layer. W2 will land
``.decode`` proper if user code needs it for ambassador-API parity.
"""

from __future__ import annotations

from typing import Any

from .handles import (
    AttributeHandle,
    DimensionHandle,
    FederateHandle,
    InteractionClassHandle,
    ObjectClassHandle,
    ObjectInstanceHandle,
    ParameterHandle,
    RegionHandle,
    _StrongHandle,
)
from .sets import (
    AttributeHandleSet,
    AttributeHandleValueMap,
    DimensionHandleSet,
    FederateHandleSet,
    ParameterHandleSet,
    ParameterHandleValueMap,
    RegionHandleSet,
)


class AttributeHandleSetFactory:
    def create(self) -> AttributeHandleSet:
        return AttributeHandleSet()


class AttributeHandleValueMapFactory:
    def create(self, capacity: int = 0) -> AttributeHandleValueMap:
        return AttributeHandleValueMap()


class ParameterHandleValueMapFactory:
    def create(self, capacity: int = 0) -> ParameterHandleValueMap:
        return ParameterHandleValueMap()


class FederateHandleSetFactory:
    def create(self) -> FederateHandleSet:
        return FederateHandleSet()


class DimensionHandleSetFactory:
    def create(self) -> DimensionHandleSet:
        return DimensionHandleSet()


class RegionHandleSetFactory:
    def create(self) -> RegionHandleSet:
        return RegionHandleSet()


# Per-handle factories — reference_rti exposes one per handle type. ``.decode``
# is deferred to W2; the wire shape is uint64 over gRPC and there is no
# settled standalone byte form yet (see TODO).
class _HandleFactoryBase:
    _handle_cls: type[_StrongHandle] = _StrongHandle

    def decode(self, encoded_bytes: bytes) -> Any:
        # TODO(M28 W2): decide canonical handle byte form (big-endian
        # uint64 is the cppsdk wire shape; reference_rti uses java.nio
        # serialization). Defer until a user path needs it.
        raise NotImplementedError("M28 W1: decode deferred to W2")


class AttributeHandleFactory(_HandleFactoryBase):
    _handle_cls = AttributeHandle


class ObjectClassHandleFactory(_HandleFactoryBase):
    _handle_cls = ObjectClassHandle


class InteractionClassHandleFactory(_HandleFactoryBase):
    _handle_cls = InteractionClassHandle


class ParameterHandleFactory(_HandleFactoryBase):
    _handle_cls = ParameterHandle


class ObjectInstanceHandleFactory(_HandleFactoryBase):
    _handle_cls = ObjectInstanceHandle


class FederateHandleFactory(_HandleFactoryBase):
    _handle_cls = FederateHandle


class DimensionHandleFactory(_HandleFactoryBase):
    _handle_cls = DimensionHandle


class RegionHandleFactory(_HandleFactoryBase):
    _handle_cls = RegionHandle


class ParameterHandleSetFactory:
    def create(self) -> ParameterHandleSet:
        return ParameterHandleSet()
