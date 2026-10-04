"""Fenced transportation activation shared by Layer-1 and standard callbacks."""

from dataclasses import replace
from typing import Any

from .events import TransportationChangeConfirmed


async def activate(federate: Any, event: Any) -> Any:
    if not isinstance(event, TransportationChangeConfirmed):
        return event
    federation = federate._require_federation_name()
    transport = federate._transport
    caller = getattr(event, "_transportation_caller", federate.handle)
    if (len(event.ticket) != 32 or caller != federate.handle
            or event.federation_generation != transport._generation_by_federation.get(federation)
            or event.callback_epoch != transport._callback_epochs.get(caller, 0)
            or bool(event.object_handle) == bool(event.interaction_class_handle)
            or (not event.object_handle and event.attribute_handles)
            or event.transportation_type not in (1, 2)):
        raise RuntimeError("invalid or retired transportation confirmation")
    result = await federate._service_rpc(
        "object", "ActivateTransportationTypeChange", "ActivateTransportationTypeChangeRequest",
        authenticated=True, ticket=event.ticket,
        expected_federation_generation=event.federation_generation,
        expected_callback_epoch=event.callback_epoch,
    )
    attributes = tuple(int(value) for value in result.attribute_handles)
    if (bytes(result.ticket) != event.ticket
            or result.federation_generation != event.federation_generation
            or result.callback_epoch != event.callback_epoch
            or result.object_handle != event.object_handle
            or result.interaction_class_handle != event.interaction_class_handle
            or result.transport_type != event.transportation_type
            or len(set(attributes)) != len(attributes)
            or not set(attributes).issubset(event.attribute_handles)):
        raise RuntimeError("mismatched transportation activation")
    activated = replace(event, attribute_handles=attributes)
    for name in ("_callback_receipt", "_transportation_caller", "_callback_invocation_identity",
                 "_callback_recovered", "_callback_outcome_only", "_callback_completed",
                 "_restore_serial", "_wire_carrier"):
        if hasattr(event, name):
            object.__setattr__(activated, name, getattr(event, name))
    return activated
