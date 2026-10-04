"""gRPC StatusCode → typed ``RtiError`` translation for cut-3 services.

Cut-2 service-group handlers (sync / ownership / DDM / savepoint, see
``rti/internal/transport/grpc/*.go``) return standard gRPC status codes
rather than the ``ERR_*`` numeric codes carried in trailers by the
cut-1 federation/declaration/object services. The cut-3 SDK clients
(``rti1516e.sync`` / ``rti1516e.ownership`` / ``rti1516e.ddm`` /
``rti1516e.savepoint``) translate via this helper.

Mapping policy:

  - ``NOT_FOUND``           → :class:`SyncPointNotFound` /
                              :class:`SaveBundleNotFound` /
                              :class:`RegionNotFound` (status-text-driven
                              disambiguation when needed)
  - ``ALREADY_EXISTS``      → :class:`SyncPointAlreadyExists` /
                              :class:`SaveBundleAlreadyExists`
  - ``FAILED_PRECONDITION`` → :class:`InvalidSyncState` /
                              :class:`InvalidSaveState` /
                              :class:`InvalidOwnershipState`
  - ``PERMISSION_DENIED``   → :class:`OwnershipNotPermitted`
  - ``INVALID_ARGUMENT``    → :class:`RtiError` (generic; the wire-level
                              validators surface this for malformed reqs)

The classifier is deliberately coarse: cut-3 spec tests assert on
``RtiError`` (the base class) plus the gRPC code, so a fine-grained
error hierarchy is not required for green tests. The named subclasses
here exist so future tests + user code can catch them by type without
having to import ``grpc`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from rti1516e.errors import RtiError

if TYPE_CHECKING:  # pragma: no cover - type-check imports only
    pass


# --- Cut-3 typed exceptions ---------------------------------------------------
#
# These are subclasses of RtiError so callers can catch the broad base.
# Each class carries a stable ``error_code`` mirroring the proto ErrorCode
# convention (numeric IDs ≥ 600 are reserved for cut-3 service groups; this
# range is currently unused in errors.proto and is therefore a safe internal
# allocation that does not collide with the M0..M11 cut-1 errors).


class SyncPointNotFound(RtiError):
    """gRPC NotFound from SyncService — sync point label is unknown."""

    error_code = 600


class SyncPointAlreadyExists(RtiError):
    """gRPC AlreadyExists from SyncService — sync point label already registered."""

    error_code = 601


class InvalidSyncState(RtiError):
    """gRPC FailedPrecondition from SyncService — illegal state transition."""

    error_code = 602


class OwnershipNotPermitted(RtiError):
    """gRPC PermissionDenied from OwnershipService — caller is not the owner."""

    error_code = 603


class InvalidOwnershipState(RtiError):
    """gRPC FailedPrecondition from OwnershipService — illegal transfer state."""

    error_code = 604


class RegionNotFound(RtiError):
    """gRPC NotFound from DDMService — region/dimension/space is unknown."""

    error_code = 605


class SaveBundleNotFound(RtiError):
    """gRPC NotFound from SavepointService — no bundle for (federation, label)."""

    error_code = 606


class SaveBundleAlreadyExists(RtiError):
    """gRPC AlreadyExists from SavepointService — bundle already on disk."""

    error_code = 607


class InvalidSaveState(RtiError):
    """gRPC FailedPrecondition from SavepointService — illegal state transition."""

    error_code = 608


# --- Time-service typed exceptions ------------------------------------------
#
# Codes 700-708 are reserved for time-management; service-group errors use
# 600-608 (above), and federation/declaration errors use values below 600.


class TimeRegulationAlreadyEnabled(RtiError):
    """Attempted EnableTimeRegulation while already regulating (HLA §8.2)."""

    error_code = 700


class TimeRegulationNotEnabled(RtiError):
    """Operation requires time-regulating state which the federate lacks."""

    error_code = 701


class TimeConstrainedAlreadyEnabled(RtiError):
    """Attempted EnableTimeConstrained while already constrained."""

    error_code = 702


class TimeConstrainedNotEnabled(RtiError):
    """Operation requires time-constrained state which the federate lacks."""

    error_code = 703


class InvalidLookahead(RtiError):
    """Lookahead must be >= 0 and finite (NaN / -Inf / +Inf rejected)."""

    error_code = 704


class LogicalTimeAlreadyPassed(RtiError):
    """Requested logical time is not greater than currentTime + lookahead.

    Maps to server-side core.ErrTimeRequestInPast — note the manager
    fires when t < currentTime + lookahead (with lookahead floor),
    not strictly t < currentTime.
    """

    error_code = 705


class TimeAdvancingState(RtiError):
    """Federate has an outstanding advance request (NER/NMRA/TAR/TARA/FQR)."""

    error_code = 706


class FederationHaltedError(RtiError):
    """Federation is in the terminal halted state.

    Suffixed "Error" to disambiguate from the FederationHalted event class.
    """

    error_code = 707


class TimeAlreadyAsynchronous(RtiError):
    """Federate has already enabled asynchronous delivery."""

    error_code = 708


class TimeNotAsynchronous(RtiError):
    """Federate has not enabled asynchronous delivery."""

    error_code = 709


class ObjectNotOwned(RtiError):
    """Federate is not the owner of the object instance (M23)."""

    error_code = 710


class AttributeNotPublishedByFederation(RtiError):
    """No federate publishes any of the requested attributes (M23)."""

    error_code = 711


class ObjectAlreadyDeleted(RtiError):
    """Object instance has already been deleted (M23)."""

    error_code = 712


class TransportTypeUnspecified(RtiError):
    """Transport type must be Reliable or BestEffort (M23)."""

    error_code = 713


# --- IEEE 1516.1-2010 Annex C names for the time path (M39 HA-3) --------------
#
# Same policy as rti1516e.errors: the Annex-C name subclasses the
# home-grown class so ``except``-clauses against either keep catching.
# The metadata-first translator resolves ``rti-spec-exception`` values
# against class __name__, so these must match Annex C verbatim.


class TimeRegulationIsNotEnabled(TimeRegulationNotEnabled):
    """Annex C — operation requires time-regulating state."""


class TimeConstrainedIsNotEnabled(TimeConstrainedNotEnabled):
    """Annex C — operation requires time-constrained state."""


class InTimeAdvancingState(TimeAdvancingState):
    """Annex C — an advance request is already outstanding."""


class RequestForTimeRegulationPending(RtiError):
    """Annex C — enableTimeRegulation is still in flight."""

    error_code = 714


class RequestForTimeConstrainedPending(RtiError):
    """Annex C — enableTimeConstrained is still in flight."""

    error_code = 715


class AsynchronousDeliveryAlreadyEnabled(TimeAlreadyAsynchronous):
    """Annex C — §8.14 enableAsynchronousDelivery when already on."""


class AsynchronousDeliveryAlreadyDisabled(TimeNotAsynchronous):
    """Annex C — §8.15 disableAsynchronousDelivery when already off."""


# --- Metadata-first spec-exception mapping (M39 HA-3, CONTRACT with HB) --------
#
# The rtid attaches trailing gRPC metadata under ``rti-spec-exception``
# whose value is the IEEE 1516.1 exception class name in UpperCamelCase
# exactly as in Annex C (e.g. ``ObjectClassNotPublished``,
# ``AttributeNotOwned``, ``InvalidLogicalTime``,
# ``FederationExecutionAlreadyExists``). The shared translator reads
# that key FIRST and raises the pysdk class of the same name; the
# code/detail heuristics below remain as the fallback for servers that
# predate the metadata.

_SPEC_EXCEPTION_METADATA_KEY = "rti-spec-exception"

_SPEC_EXCEPTION_TABLE: dict[str, type[RtiError]] | None = None


def spec_exception_table() -> dict[str, type[RtiError]]:
    """Annex-C exception name -> pysdk exception class.

    Built programmatically (once) from every ``RtiError`` subclass
    defined in :mod:`rti1516e.errors` and this module, keyed by class
    ``__name__`` — adding a new exception class automatically makes the
    matching metadata value resolvable.
    """
    global _SPEC_EXCEPTION_TABLE  # noqa: PLW0603 — build-once module cache
    if _SPEC_EXCEPTION_TABLE is None:
        import sys

        from rti1516e import errors as _errors_module

        table: dict[str, type[RtiError]] = {}
        for module in (_errors_module, sys.modules[__name__]):
            for obj in vars(module).values():
                if isinstance(obj, type) and issubclass(obj, RtiError):
                    table[obj.__name__] = obj
        _SPEC_EXCEPTION_TABLE = table
    return _SPEC_EXCEPTION_TABLE


def _spec_exception_name(exc: BaseException) -> str:
    """Return the ``rti-spec-exception`` trailing-metadata value, or ``""``.

    Duck-typed over ``AioRpcError.trailing_metadata()`` (an ``aio.Metadata``
    of (key, value) pairs) and the sync API's Metadatum sequence; bytes
    values are decoded. Any shape surprise degrades to ``""`` (fallback
    heuristics take over) rather than raising.
    """
    metadata_fn = getattr(exc, "trailing_metadata", None)
    if not callable(metadata_fn):
        return ""
    try:
        metadata = metadata_fn()
    except Exception:  # noqa: BLE001 - defensive
        return ""
    if metadata is None:
        return ""
    try:
        for item in metadata:
            key, value = item[0], item[1]
            if str(key).lower() != _SPEC_EXCEPTION_METADATA_KEY:
                continue
            if isinstance(value, bytes | bytearray):
                return bytes(value).decode("utf-8", errors="replace")
            return str(value)
    except Exception:  # noqa: BLE001 - defensive
        return ""
    return ""


# --- Translation entry point --------------------------------------------------


def translate_rpc_error(exc: BaseException, *, operation: str = "") -> NoReturn:
    """Re-raise ``exc`` as the matching RtiError subclass (or pass through).

    The cut-3 service handlers surface failures as ``grpc.aio.AioRpcError``
    with a populated ``code()``. We sniff via duck typing rather than
    importing grpc unconditionally so importing this module from a
    memory:// transport path costs nothing.

    Behavior:

      - If ``exc`` exposes a ``.code()`` returning a recognized gRPC
        code, raise the matching typed exception (preserving the
        original via ``__cause__``).
      - Otherwise, re-raise the original ``exc`` unchanged.

    The function never returns; the type is :class:`typing.NoReturn`.
    """
    code_name = _grpc_code_name(exc)
    detail = _grpc_detail(exc) or str(exc)

    # M39 HA-3 — metadata FIRST: an M39+ rtid names the IEEE Annex C
    # exception class explicitly in trailing metadata; that beats any
    # heuristic. Unknown names (a future exception this pysdk predates)
    # fall through to the heuristics below.
    if spec_name := _spec_exception_name(exc):
        spec_cls = spec_exception_table().get(spec_name)
        if spec_cls is not None:
            raise spec_cls(detail or spec_name) from exc

    # Older servers do not distinguish by-name support lookups in trailers.
    if (operation == "GetObjectClassHandle" and code_name == "NOT_FOUND"
            and detail.startswith("object class ") and detail.endswith("not found")):
        from .errors import NameNotFound

        raise NameNotFound(detail) from exc

    # Time-management errors share gRPC codes
    # (FailedPrecondition for most state errors, InvalidArgument for
    # bad inputs). The detail string is the manager's sentinel.Error()
    # text — sniff that to disambiguate.
    if time_cls := _time_class_for(detail):
        raise time_cls(detail) from exc

    # M39 HA-3 — federation/declaration/object sentinel texts (pre-
    # metadata servers). Same policy as _time_class_for: substring on
    # the distinguishing fragment of core/errors.go.
    if detail_cls := _detail_class_for(detail):
        # Join recovery trusts these as definite admission rejections. Text
        # alone must not turn an uncertain transport/internal failure into one.
        admission_codes = {
            "FederateNameAlreadyInUse": "ALREADY_EXISTS",
            "FederateAlreadyExecutionMember": "ALREADY_EXISTS",
            "SaveInProgress": "FAILED_PRECONDITION",
            "RestoreInProgress": "FAILED_PRECONDITION",
        }
        expected = admission_codes.get(detail_cls.__name__)
        if expected is None or code_name == expected:
            raise detail_cls(detail) from exc

    cls = _CODE_TO_EXCEPTION.get(code_name)
    if cls is not None:
        raise cls(detail) from exc

    # Unknown code (or non-gRPC exception) — propagate unchanged so the
    # caller sees the most diagnostically useful trace.
    raise exc


def _time_class_for(detail: str) -> type[RtiError] | None:
    """Return the time-mgmt typed exception for ``detail``, or None.

    Detail strings come from server-side core.ErrTime*.Error() text
    (see rti/internal/core/errors.go). Substring match on the
    distinguishing fragment.
    """
    # Order matters — pick the most-specific match first.
    if "request for time-regulation pending" in detail:
        return RequestForTimeRegulationPending
    if "request for time-constrained pending" in detail:
        return RequestForTimeConstrainedPending
    if "already time-regulating" in detail:
        return TimeRegulationAlreadyEnabled
    if "not time-regulating" in detail:
        return TimeRegulationNotEnabled
    if "already time-constrained" in detail:
        return TimeConstrainedAlreadyEnabled
    if "not time-constrained" in detail:
        return TimeConstrainedNotEnabled
    if "lookahead must be" in detail:
        return InvalidLookahead
    if "requested time is not greater" in detail:
        return LogicalTimeAlreadyPassed
    if "outstanding advance request" in detail:
        return TimeAdvancingState
    if "federation halted" in detail:
        return FederationHaltedError
    if "already enabled asynchronous" in detail:
        return TimeAlreadyAsynchronous
    if "not enabled asynchronous" in detail:
        return TimeNotAsynchronous
    if "object not owned" in detail:
        return ObjectNotOwned
    if "no federate publishes" in detail:
        return AttributeNotPublishedByFederation
    if "object instance already deleted" in detail:
        return ObjectAlreadyDeleted
    if "transport type must be" in detail:
        return TransportTypeUnspecified
    return None


def _detail_class_for(detail: str) -> type[RtiError] | None:
    """Return the Annex-C typed exception for a federation / object /
    declaration sentinel text, or None.

    Fallback for pre-M39 servers (no ``rti-spec-exception`` metadata).
    Detail strings come from rti/internal/core/errors.go sentinel
    ``Error()`` texts; substring match on the distinguishing fragment,
    most-specific first.
    """
    from rti1516e import errors as e

    if "message retraction designator is invalid" in detail:
        return e.InvalidMessageRetractionHandle
    if "object deletion has already been delivered" in detail:
        return e.MessageCanNoLongerBeRetracted
    if "federation already exists" in detail:
        return e.FederationExecutionAlreadyExists
    if "federation has federates joined" in detail:
        return e.FederatesCurrentlyJoined
    if "federation not found" in detail:
        return e.FederationExecutionDoesNotExist
    if "federate not joined" in detail:
        return e.FederateNotExecutionMember
    if "federate already joined" in detail:
        return e.FederateAlreadyExecutionMember
    if "object class not published" in detail:
        return e.ObjectClassNotPublished
    if "interaction class not published" in detail:
        return e.InteractionClassNotPublished
    if "attribute not owned" in detail:
        return e.AttributeNotOwned
    if "attribute divestiture confirmation was not requested" in detail:
        return e.AttributeDivestitureWasNotRequested
    if "no acquisition pending or willing acquirer for this attribute" in detail:
        return e.NoAcquisitionPending
    if "object class not found" in detail:
        return e.ObjectClassNotDefined
    if "attribute not found" in detail:
        return e.AttributeNotDefined
    if "object not found" in detail:
        return e.ObjectInstanceNotKnown
    if "invalid logical time" in detail:
        return e.InvalidLogicalTime
    if "object instance name already reserved or in use" in detail:
        return e.ObjectInstanceNameInUse
    if "object instance name has not been reserved" in detail:
        return e.ObjectInstanceNameNotReserved
    return None


# --- Helpers ------------------------------------------------------------------


def _grpc_code_name(exc: BaseException) -> str:
    """Return the gRPC status-code name (e.g. ``NOT_FOUND``) or ``""``."""
    code_fn = getattr(exc, "code", None)
    if not callable(code_fn):
        return ""
    try:
        code = code_fn()
    except Exception:  # noqa: BLE001 - defensive
        return ""
    name = getattr(code, "name", None)
    if isinstance(name, str):
        return name
    return str(code)


def _grpc_detail(exc: BaseException) -> str:
    """Return the gRPC status-detail string or ``""``."""
    detail_fn = getattr(exc, "details", None)
    if not callable(detail_fn):
        return ""
    try:
        detail = detail_fn()
    except Exception:  # noqa: BLE001 - defensive
        return ""
    if isinstance(detail, str):
        return detail
    return ""


# Exactly one mapping per gRPC code we care about. Cut-3 services may
# return the same code from different operations (e.g. NotFound from
# Sync Achieve vs Savepoint Restore); tests catch the base RtiError so
# the coarse mapping below is sufficient. Per-service refinement (sniff
# the detail string) can be added later without breaking the contract.
_CODE_TO_EXCEPTION: dict[str, type[RtiError]] = {
    "NOT_FOUND": SyncPointNotFound,
    "ALREADY_EXISTS": SyncPointAlreadyExists,
    "FAILED_PRECONDITION": InvalidSyncState,
    "PERMISSION_DENIED": OwnershipNotPermitted,
}
