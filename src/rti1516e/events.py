"""Typed event dataclasses yielded by Federate.events().

The transport maps each supported wire event to one of these stable
dataclass shapes.

Use a discriminated union via ``isinstance`` (or ``match`` in 3.10+):

    async for event in fed.events():
        match event:
            case ReflectAttributeValues(obj, values, ts): ...
            case ReceiveInteraction(cls, params, ts): ...
            case TimeAdvanceGrant(t): ...
            case DiscoverObjectInstance(obj, cls, name): ...
            case SynchronizationPointAnnounced(label, tag, required): ...
            case FederationSynchronized(label): ...
            case RequestAttributeOwnershipAssumption(obj, attrs, divester, tag): ...
            case AttributeOwnershipAcquisitionNotification(obj, attrs, owning): ...
            case RequestDivestitureConfirmation(obj, attrs): ...
            case InitiateFederateSave(label, save_time): ...
            case FederationSaved(label): ...
            case FederationNotSaved(label): ...
            case FederationHalted(cause, federate_handle): ...
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


@dataclass(frozen=True)
class TransportationChangeConfirmed:
    """Prepared confirmation activated by dispatch before application invocation."""

    ticket: bytes
    federation_generation: int
    callback_epoch: int
    object_handle: int
    attribute_handles: tuple[int, ...]
    interaction_class_handle: int
    transportation_type: int

if TYPE_CHECKING:  # pragma: no cover - type-check imports only
    from rti1516e.handles import AttributeHandle, ObjectClassHandle


class OrderType(enum.IntEnum):
    UNSPECIFIED = 0
    RECEIVE = 1
    TIMESTAMP = 2


@dataclass(frozen=True)
class ObjectCallbackMetadata:
    sent_order: int | None = None
    received_order: int | None = None
    transportation_type: int | None = None
    producing_federate: int | None = None
    message_retraction_handle: int | None = None
    received_regions: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TurnUpdatesOnForObjectInstance:
    object_handle: int
    attribute_handles: tuple[int, ...]
    update_rate_designator: str | None = None


@dataclass(frozen=True)
class TurnUpdatesOffForObjectInstance:
    object_handle: int
    attribute_handles: tuple[int, ...]


@dataclass(frozen=True)
class DiscoverObjectInstance:
    """A new object instance was registered by another federate.

    M39 (HA-2 typed-handle parity): ``object_class`` carries the typed
    :class:`rti1516e.handles.ObjectClassHandle` per IEEE 1516.1-2010
    §6.9 discoverObjectInstance. ``class_name`` is retained for
    back-compat (stringified handle on the gRPC path; FOM name on
    in-process transports that choose to populate it) and is
    DEPRECATED as the class identity — new code should use
    ``object_class``.
    """

    object_handle: int
    class_name: str
    instance_name: str
    # Typed §6.9 class handle. None only for legacy producers that
    # construct the event without it (in-process test doubles).
    object_class: ObjectClassHandle | None = None
    producing_federate: int | None = None


@dataclass(frozen=True)
class ReflectAttributeValues:
    """Attribute values updated on an object we subscribe to.

    M39 (HA-2 typed-handle parity): ``attribute_values`` keys the same
    payloads by typed :class:`rti1516e.handles.AttributeHandle` per
    IEEE 1516.1-2010 §6.11. ``values`` (string-keyed) is retained for
    back-compat and is DEPRECATED for handle identity — string keys on
    the gRPC path are stringified attribute handles.
    """

    object_handle: int
    values: dict[str, Any]
    timestamp: float | None  # None for RO; non-None for TSO
    # Typed §6.11 attribute map. Empty for legacy producers.
    attribute_values: dict[AttributeHandle, bytes] = field(default_factory=dict)
    tag: bytes = b""
    metadata: ObjectCallbackMetadata | None = None


@dataclass(frozen=True)
class ReceiveInteraction:
    """An interaction we subscribe to was sent by another federate."""

    class_name: str
    parameters: dict[str, Any]
    timestamp: float | None
    tag: bytes = b""
    metadata: ObjectCallbackMetadata | None = None


@dataclass(frozen=True)
class TimeAdvanceGrant:
    """The RTI granted our pending NER. Federate is now at ``time``."""

    time: float


@dataclass(frozen=True)
class TimeRegulationEnabled:
    """Time regulation is active at the RTI-selected logical time."""

    time: float


@dataclass(frozen=True)
class TimeConstrainedEnabled:
    """Time constraint is active at the RTI-selected logical time."""

    time: float


@dataclass(frozen=True)
class FederationHalted:
    """The federation has been halted (e.g. by stall timeout).

    No further calls on the Federate will succeed; close the connection.
    Named to match the M3 event type emitted from the Go side.
    """

    cause: str  # "stall" | other future causes
    stalled_federate_handle: int


@dataclass(frozen=True)
class RemoveObjectInstance:
    """The object instance has been deleted by its owner (M23, IEEE 1516.1 §6.16)."""

    object_handle: int
    tag: bytes
    timestamp: float | None  # None for RO; non-None for TSO delete
    metadata: ObjectCallbackMetadata | None = None


@dataclass(frozen=True)
class ProvideAttributeValueUpdate:
    """A peer requested fresh values for the listed attributes (M23, IEEE 1516.1 §6.26).

    Owner is expected to respond with ``Federate.update_attributes`` containing
    fresh values for the listed attribute handles.
    """

    object_handle: int
    attribute_handles: tuple[int, ...]
    tag: bytes


# --- Cut-2 service-group callbacks (M12 W2 deferral #1 close) -------------
# Wire-side proto variants live on rti.v1.FederateEvent at oneof tags
# 20/21 (sync), 30/31/32 (ownership), 40/41/42 (save). The dataclasses
# here are the typed Python mirrors yielded by Federate.events() once
# the GrpcTransport's _translate_event resolves the matching oneof.


@dataclass(frozen=True)
class SynchronizationPointAnnounced:
    """A sync point was registered (§4.6).

    Fires on every joined federate (when the implicit "all members"
    set was used at registration) or every required federate (when an
    explicit set was passed).
    """

    label: str
    tag: bytes
    # Empty when the implicit all-members set was used.
    required_federates: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FederationSynchronized:
    """All required federates achieved a sync point (§4.7).

    Fires on every required federate. Federates that were waiting on
    the achieve callback should now resume.

    ``failed_to_sync`` (IEEE 1516.1-2010 §4.15, M37 wire tag 21 field 2)
    lists the federates that achieved the point with
    ``successfully=False``. Empty when everyone succeeded (and when
    talking to pre-M37 servers).
    """

    label: str
    failed_to_sync: tuple[int, ...] = field(default_factory=tuple)


class SynchronizationPointFailureReason(enum.Enum):
    """IEEE 1516.1-2010 SynchronizationPointFailureReason (§4.12)."""

    SYNCHRONIZATION_POINT_LABEL_NOT_UNIQUE = "SYNCHRONIZATION_POINT_LABEL_NOT_UNIQUE"
    SYNCHRONIZATION_SET_MEMBER_NOT_JOINED = "SYNCHRONIZATION_SET_MEMBER_NOT_JOINED"


@dataclass(frozen=True)
class SynchronizationPointRegistrationSucceeded:
    """§4.12 — this federate's sync-point registration was accepted.

    Fires on the REGISTERING federate only (M37 wire tag 22).
    """

    label: str


@dataclass(frozen=True)
class SynchronizationPointRegistrationFailed:
    """§4.12 — this federate's sync-point registration was rejected.

    Fires on the REGISTERING federate only (M37 wire tag 23).
    ``reason`` is None when the server sent an unspecified/unknown
    failure reason.
    """

    label: str
    reason: SynchronizationPointFailureReason | None


@dataclass(frozen=True)
class RequestAttributeOwnershipAssumption:
    """A negotiated divest is in progress; this federate may take over (§7.3).

    Fires on every subscriber of the attribute when the current owner
    calls negotiated_divest. The receiving federate may call
    ``acquire`` to take ownership.
    """

    object_handle: int
    attribute_handles: tuple[int, ...]
    divesting_federate: int
    tag: bytes


@dataclass(frozen=True)
class AttributeOwnershipAcquisitionNotification:
    """A two-phase ownership transfer completed (§7.4).

    Fires on the new owner. ``owning_federate`` echoes the recipient's
    own handle so federates with multiple streams can correlate.
    """

    object_handle: int
    attribute_handles: tuple[int, ...]
    owning_federate: int
    tag: bytes = b""


@dataclass(frozen=True)
class RequestDivestitureConfirmation:
    """An acquirer took over a pending divest (§7.3 divestee half).

    Fires on the federate that called negotiated_divest once the
    transfer to an acquirer fires.
    """

    object_handle: int
    attribute_handles: tuple[int, ...]


@dataclass(frozen=True)
class InitiateFederateSave:
    """The federation save has started (§4.8).

    Fires on every joined federate. ``save_time`` is the optional
    logical-time pin the requester passed; ``None`` when the requester
    asked for "save now".
    """

    label: str
    save_time: float | None


@dataclass(frozen=True)
class FederationSaved:
    """The federation save completed successfully (§4.9 success half).

    Fires on every federate that participated in the save once every
    federate has reported FederateSaveComplete and the bundle has been
    persisted.
    """

    label: str


@dataclass(frozen=True)
class FederationNotSaved:
    """The federation save aborted (§4.9 failure half).

    Fires on every federate that participated in the save when at
    least one reported FederateSaveNotComplete OR bundle persistence
    failed. The save bundle is NOT written.
    """

    label: str


# --- M26 Phase F — object instance name reservation events --------------


@dataclass(frozen=True)
class ObjectInstanceNameReservationSucceeded:
    """IEEE 1516.1 §6.1 — a single-name reservation was accepted.

    The federate may now call registerObjectInstance with this name.
    """

    object_name: str


@dataclass(frozen=True)
class ObjectInstanceNameReservationFailed:
    """IEEE 1516.1 §6.1 — a single-name reservation was rejected.

    The name is already reserved (by this or another federate) or
    already in use by a registered instance. The federate must pick
    a different name.
    """

    object_name: str


@dataclass(frozen=True)
class MultipleObjectInstanceNameReservationSucceeded:
    """IEEE 1516.1 §6.5 — an atomic batch reservation was accepted.

    Every name in ``object_names`` is now reserved for this federate.
    """

    object_names: tuple[str, ...]


@dataclass(frozen=True)
class MultipleObjectInstanceNameReservationFailed:
    """IEEE 1516.1 §6.5 — an atomic batch reservation was rejected.

    NONE of the requested names were reserved. ``colliding_names``
    lists the specific names that caused the failure.
    """

    requested_names: tuple[str, ...]
    colliding_names: tuple[str, ...]


# --- Wire-tag parity events ------------------------------------------------
# Wire-side proto variants: stream.proto tags 33/34 (ownership),
# 43/44/45/46/47/48 (restore family — 43-45 predate M37 but were
# silently dropped by _translate_event until M39), 60-63 (registration /
# interaction advisories), 64/65 (DDM scope), 66 (retraction).


@dataclass(frozen=True)
class RequestAttributeOwnershipRelease:
    """§7.11 — another federate wants attributes this federate owns.

    Fires on the CURRENT OWNER when a peer calls
    attributeOwnershipAcquisition against owned attributes (the acquire
    stays queued). ``tag`` echoes the acquirer's user-supplied tag.
    """

    object_handle: int
    attribute_handles: tuple[int, ...]
    tag: bytes


@dataclass(frozen=True)
class AttributeOwnershipUnavailable:
    """§7.10 — an acquisition-if-available found the attributes owned.

    Fires on the federate that called
    attributeOwnershipAcquisitionIfAvailable (§7.9) for the subset of
    requested attributes that were NOT available. No pending acquire
    was queued for these.
    """

    object_handle: int
    attribute_handles: tuple[int, ...]


@dataclass(frozen=True)
class ConfirmAttributeOwnershipAcquisitionCancellation:
    object_handle: int
    attribute_handles: tuple[int, ...]


@dataclass(frozen=True)
class InitiateFederateRestore:
    """§4.26 — a federation restore began; load state and report back.

    ``federate_handle`` is the handle this federate had at save time
    (0 = no remap); ``federate_name`` is the name on record (may be
    empty on pre-M37 servers).
    """

    label: str
    federate_handle: int
    federate_name: str
    client_state: bytes = b""


@dataclass(frozen=True)
class FederationRestored:
    """§4.14 — the federation restore completed successfully."""

    label: str
    federate_handle: int | None = None
    callback_epoch: int | None = None
    next_retraction_handle: int | None = None


@dataclass(frozen=True)
class FederationNotRestored:
    """§4.14 — the federation restore aborted."""

    label: str


@dataclass(frozen=True)
class RequestFederationRestoreSucceeded:
    """§4.25 — this federate's requestFederationRestore was accepted.

    Fires on the REQUESTING federate only.
    """

    label: str


@dataclass(frozen=True)
class RequestFederationRestoreFailed:
    """§4.25 — this federate's requestFederationRestore was rejected.

    Fires on the REQUESTING federate only; ``reason`` mirrors the
    server's rejection cause.
    """

    label: str
    reason: str


@dataclass(frozen=True)
class FederationRestoreBegun:
    """§4.26 — the restore left idle; precedes InitiateFederateRestore."""


@dataclass(frozen=True)
class StartRegistrationForObjectClass:
    """§5.10 — the class gained its first subscriber; registration is
    now relevant. Fires on each PUBLISHER of the object class."""

    object_class_handle: int


@dataclass(frozen=True)
class StopRegistrationForObjectClass:
    """§5.11 — the class lost its last subscriber. Fires on publishers."""

    object_class_handle: int


@dataclass(frozen=True)
class TurnInteractionsOn:
    """§5.12 — the interaction class gained its first subscriber.
    Fires on each PUBLISHER of the interaction class."""

    interaction_class_handle: int


@dataclass(frozen=True)
class TurnInteractionsOff:
    """§5.13 — the interaction class lost its last subscriber."""

    interaction_class_handle: int


@dataclass(frozen=True)
class AttributesInScope:
    """§6.17 — DDM region overlap brought the attributes into scope.

    Fires on a SUBSCRIBER; precedes the corresponding Reflect.
    """

    object_handle: int
    attribute_handles: tuple[int, ...]


@dataclass(frozen=True)
class AttributesOutOfScope:
    """§6.18 — the attributes dropped out of region-overlap scope."""

    object_handle: int
    attribute_handles: tuple[int, ...]


@dataclass(frozen=True)
class RequestRetraction:
    """§8.22 — a sender retracted a TSO message this federate may have
    seen. ``(sender_federate, retraction_handle)`` is the
    MessageRetractionHandle pair from the original TSO delivery."""

    sender_federate: int
    retraction_handle: int
