"""Typed exceptions, one per ErrorCode in proto/rti/v1/errors.proto.

Each gRPC error response from the RTI is mapped to the matching exception
class. Spec tests in
pysdk/tests/spec/m4/test_spec_m4_events_stream.py assert the mapping is
1:1 against the proto definition.

Exception names and inheritance are part of the public compatibility
contract; ``RtiError`` may provide shared implementation helpers.
"""

from __future__ import annotations


class RtiError(Exception):
    """Base class for all RTI-emitted typed exceptions.

    Carries the underlying gRPC status code (when available) on the
    ``code`` attribute and the per-error-class numeric ID from
    proto/rti/v1/errors.proto on the ``error_code`` attribute.

    The transport reads ``error_code`` from gRPC trailing metadata.
    """

    error_code: int = 0

    def __init__(self, message: str = "", *, error_code: int | None = None) -> None:
        super().__init__(message)
        if error_code is not None:
            self.error_code = error_code


class UnsupportedCallbackModel(RtiError):
    """The requested local callback model is not supported."""


class CallNotAllowedFromWithinCallback(RtiError):
    """A callback attempted a prohibited recursive lifecycle/evocation call."""


class AlreadyConnected(RtiError):
    """The ambassador already has an open connection."""


class CouldNotCreateLogicalTimeFactory(RtiError, ValueError):
    """The requested or selected time implementation cannot be used.

    ValueError compatibility retains the former local selector-validation API.
    """


class RTIinternalError(RtiError):
    """Internal RTI/protocol failure, not a promise that no effects occurred."""


class FederationCreationResponseError(RTIinternalError):
    """A successful create returned unsupported metadata; execution remains."""

    def __init__(
        self, federation_name: str, federation_generation: int,
        effective_seed: int, logical_time_implementation_name: str,
    ) -> None:
        super().__init__(
            f"server already created federation execution {federation_name!r} "
            f"(generation {federation_generation}), but its successful response reported "
            f"unsupported logical time implementation {logical_time_implementation_name!r}; "
            "no automatic destruction was attempted"
        )
        self.federation_name = federation_name
        self.federation_generation = federation_generation
        self.effective_seed = effective_seed
        self.logical_time_implementation_name = logical_time_implementation_name


# --- Federation lifecycle (errors.proto 1..7) -------------------------------


class FederationNotFound(RtiError):
    """ERR_FED_NOT_FOUND (1)."""

    error_code = 1


class FederationAlreadyExists(RtiError):
    """ERR_FED_ALREADY_EXISTS (2)."""

    error_code = 2


class FederationNotJoined(RtiError):
    """ERR_FED_NOT_JOINED (3)."""

    error_code = 3


class FederationAlreadyJoined(RtiError):
    """ERR_FED_ALREADY_JOINED (4)."""

    error_code = 4


class FederationHasFederatesJoined(RtiError):
    """ERR_FED_HAS_FEDERATES_JOINED (5)."""

    error_code = 5


class FederationHaltedError(RtiError):
    """ERR_FED_HALTED (6) — federation halted (e.g. by stall timeout).

    Named with -Error suffix to disambiguate from the FederationHalted
    *event* dataclass in events.py.
    """

    error_code = 6


class FederationInvalidName(RtiError):
    """ERR_FED_INVALID_NAME (7)."""

    error_code = 7


# --- FOM (errors.proto 100..101) --------------------------------------------


class FomValidationFailed(RtiError):
    """ERR_FOM_VALIDATION_FAILED (100)."""

    error_code = 100


class FomMimRedefinition(RtiError):
    """ERR_FOM_MIM_REDEFINITION (101)."""

    error_code = 101


class ErrorReadingFDD(FomValidationFailed):
    """Invalid supplied FOM; retains the product validation-error base type."""


# --- Object management (errors.proto 200..206) ------------------------------


class ObjectNotFound(RtiError):
    """ERR_OBJ_NOT_FOUND (200)."""

    error_code = 200


class ObjectClassNotPublished(RtiError):
    """ERR_OBJ_CLASS_NOT_PUBLISHED (201)."""

    error_code = 201


class ObjectAttributeNotOwned(RtiError):
    """ERR_OBJ_ATTR_NOT_OWNED (202)."""

    error_code = 202


class ObjectHandleInvalid(RtiError):
    """ERR_OBJ_HANDLE_INVALID (203)."""

    error_code = 203


class InteractionNotPublished(RtiError):
    """ERR_OBJ_INTERACTION_NOT_PUBLISHED (204)."""

    error_code = 204


class ObjectClassNotFound(RtiError):
    """ERR_OBJ_CLASS_NOT_FOUND (205)."""

    error_code = 205


class ObjectAttributeNotFound(RtiError):
    """ERR_OBJ_ATTR_NOT_FOUND (206)."""

    error_code = 206


# --- Time management (errors.proto 300..305) --------------------------------


class TimeNotRegulating(RtiError):
    """ERR_TIME_NOT_REGULATING (300)."""

    error_code = 300


class TimeNotConstrained(RtiError):
    """ERR_TIME_NOT_CONSTRAINED (301)."""

    error_code = 301


class TimeInvalidLookahead(RtiError):
    """ERR_TIME_INVALID_LOOKAHEAD (302)."""

    error_code = 302


class TimeRequestInPast(RtiError):
    """ERR_TIME_REQUEST_IN_PAST (303)."""

    error_code = 303


class TimeAlreadyRegulating(RtiError):
    """ERR_TIME_ALREADY_REGULATING (304)."""

    error_code = 304


class TimeAlreadyConstrained(RtiError):
    """ERR_TIME_ALREADY_CONSTRAINED (305)."""

    error_code = 305


# --- Encoding (errors.proto 400..402) ---------------------------------------


class EncodingInsufficientBytes(RtiError):
    """ERR_ENC_INSUFFICIENT_BYTES (400)."""

    error_code = 400


class EncodingTypeMismatch(RtiError):
    """ERR_ENC_TYPE_MISMATCH (401)."""

    error_code = 401


class EncodingPaddingViolation(RtiError):
    """ERR_ENC_PADDING_VIOLATION (402)."""

    error_code = 402


# --- Wire (errors.proto 500..501) -------------------------------------------


class WireVersionMismatch(RtiError):
    """ERR_WIRE_VERSION_MISMATCH (500)."""

    error_code = 500


class WireMalformedMessage(RtiError):
    """ERR_WIRE_MALFORMED_MESSAGE (501)."""

    error_code = 501


# --- IEEE 1516.1-2010 Annex C exception names (M39 HA-3) ---------------------
#
# The rtid attaches trailing gRPC metadata under the key
# ``rti-spec-exception`` whose value is the IEEE 1516.1 exception class
# name in UpperCamelCase exactly as printed in Annex C (e.g.
# ``ObjectClassNotPublished``, ``AttributeNotOwned``). pysdk's shared
# translator (rti1516e._grpc_errors.translate_rpc_error) raises the
# class of the SAME NAME, resolved programmatically over this module —
# so the classes below are named verbatim after Annex C.
#
# Where gorti already had a home-grown class for the condition, the
# Annex-C name SUBCLASSES it: ``except``-clauses written against either
# name keep catching. Standalone Annex-C conditions subclass RtiError
# directly.


class FederationExecutionAlreadyExists(FederationAlreadyExists):
    """Annex C — §4.5 createFederationExecution duplicate name."""


class FederationExecutionDoesNotExist(FederationNotFound):
    """Annex C — §4.6/§4.9 named federation execution does not exist."""


class FederatesCurrentlyJoined(FederationHasFederatesJoined):
    """Annex C — §4.6 destroyFederationExecution with members joined."""


class FederateNotExecutionMember(FederationNotJoined):
    """Annex C — caller is not a joined member of the federation."""


class FederateAlreadyExecutionMember(FederationAlreadyJoined):
    """Annex C — §4.9 joinFederationExecution while already joined."""


class FederateNameAlreadyInUse(RtiError):
    """Annex C — §4.9 the requested federate name is taken."""


class NameNotFound(RtiError):
    """A support-service name lookup could not resolve the supplied name."""


class InvalidObjectClassHandle(RtiError):
    """A support query supplied an invalid object class handle."""


class InvalidInteractionClassHandle(RtiError):
    """A support query supplied an invalid interaction class handle."""


class ObjectClassNotDefined(ObjectClassNotFound):
    """Annex C — object class handle/name is not in the FOM."""


class ObjectInstanceNotKnown(ObjectNotFound):
    """Annex C — object instance handle is unknown to this federate."""


class AttributeNotDefined(ObjectAttributeNotFound):
    """Annex C — attribute handle/name is not in the FOM class."""


class AttributeNotOwned(ObjectAttributeNotOwned):
    """Annex C — §6.10/§7 caller does not own the attribute instance."""


class AttributeNotPublished(RtiError):
    """Annex C — attribute is not in the caller's publication set."""


class AttributeAlreadyOwned(RtiError):
    """Annex C — §7 acquisition target is already owned by the caller."""


class InteractionClassNotDefined(RtiError):
    """Annex C — interaction class handle/name is not in the FOM."""


class InteractionClassNotPublished(InteractionNotPublished):
    """Annex C — §6.12 sendInteraction without publication."""


class InteractionParameterNotDefined(RtiError):
    """Annex C — parameter handle/name is not in the FOM class."""


class DeletePrivilegeNotHeld(RtiError):
    """Annex C — §6.14 deleteObjectInstance without HLAprivilegeToDelete."""


class ObjectInstanceNameInUse(RtiError):
    """Annex C — §6.2 reserveObjectInstanceName collision."""


class ObjectInstanceNameNotReserved(RtiError):
    """Annex C — §6.8 explicit-name register without a reservation."""


class NameSetWasEmpty(RtiError):
    """An object-name reservation service received an empty name set."""


class SynchronizationPointLabelNotAnnounced(RtiError):
    """Annex C — §4.14 achieve on a label never announced here."""


class InvalidLogicalTime(RtiError):
    """Annex C — §8 a supplied logical time is invalid (e.g. a TSO
    timestamp below currentTime + lookahead, M37 EB-3)."""


class SaveInProgress(RtiError):
    """Annex C — operation illegal while a federation save is active."""


class RestoreInProgress(RtiError):
    """Annex C — operation illegal while a federation restore is active."""


class SaveNotInitiated(RtiError):
    """Annex C — §4.13 save-complete without an initiated save."""


class RestoreNotRequested(RtiError):
    """Annex C — §4.28 restore-complete without a requested restore."""


class FederateOwnsAttributes(RtiError):
    """Annex C — §4.10 resign forbidden while owning attributes."""


class OwnershipAcquisitionPending(RtiError):
    """Resignation is forbidden while ownership acquisitions remain pending."""


class AttributeAlreadyBeingAcquired(RtiError):
    """Annex C — §7.8 acquisition already pending for the attribute."""


class AttributeAlreadyBeingDivested(RtiError):
    """Annex C — §7.3 negotiated divest already pending."""


class AttributeDivestitureWasNotRequested(RtiError):
    """Confirm divestiture without a prior request to confirm (7.6.5(b))."""


class NoAcquisitionPending(RtiError):
    """No pending or willing acquirer remains for confirmation (7.6.5(a))."""


class AttributeAcquisitionWasNotRequested(RtiError):
    """Annex C — §7.13 cancel on an acquisition never requested."""


class InvalidRegion(RtiError):
    """Annex C — §9 region handle is invalid."""


class RegionNotCreatedByThisFederate(RtiError):
    """Annex C — §9 region belongs to a different federate."""


class InvalidRegionContext(RtiError):
    """Annex C — §9 region used with a mismatched class/attribute."""


class InvalidRangeBound(RtiError):
    """Annex C - range bounds are invalid for the dimension."""


class MessageCanNoLongerBeRetracted(RtiError):
    """Annex C — §8.21 retraction past the LBTS horizon."""


class InvalidMessageRetractionHandle(RtiError):
    """Annex C — §8.21 unknown retraction handle."""


# Names below complete coverage of the rtid's specExceptionTable
# (rti/internal/transport/grpc/spec_exception.go) so the
# metadata channel resolves every name the server can send.


class IllegalName(FederationInvalidName):
    """Annex C — a supplied name violates the naming rules."""


class InvalidResignAction(RtiError):
    """Annex C — §4.10 unknown resign-action designator."""


class InvalidTransportationType(RtiError):
    """Annex C — transportation type is not a defined designator."""


class RegionInUseForUpdateOrSubscription(RtiError):
    """Annex C — §9 region is bound to an active use."""


class RegionDoesNotContainSpecifiedDimension(RtiError):
    """Annex C — §9 dimension is not part of the region."""


class SaveNotInProgress(RtiError):
    """Annex C — no federation save is in progress."""


class RestoreNotInProgress(RtiError):
    """Annex C — no federation restore is in progress."""


class FederateHasNotBegunSave(RtiError):
    """Annex C — §4.13 save-complete before initiateFederateSave."""


class CouldNotDecode(RtiError):
    """Annex C — server-side decode of caller-supplied bytes failed."""


# Lookup table used to map numeric error codes from gRPC trailers to
# typed exceptions.
ERROR_CODE_TO_EXCEPTION: dict[int, type[RtiError]] = {
    cls.error_code: cls
    for cls in (
        FederationNotFound,
        FederationAlreadyExists,
        FederationNotJoined,
        FederationAlreadyJoined,
        FederationHasFederatesJoined,
        FederationHaltedError,
        FederationInvalidName,
        FomValidationFailed,
        FomMimRedefinition,
        ObjectNotFound,
        ObjectClassNotPublished,
        ObjectAttributeNotOwned,
        ObjectHandleInvalid,
        InteractionNotPublished,
        ObjectClassNotFound,
        ObjectAttributeNotFound,
        TimeNotRegulating,
        TimeNotConstrained,
        TimeInvalidLookahead,
        TimeRequestInPast,
        TimeAlreadyRegulating,
        TimeAlreadyConstrained,
        EncodingInsufficientBytes,
        EncodingTypeMismatch,
        EncodingPaddingViolation,
        WireVersionMismatch,
        WireMalformedMessage,
    )
}
