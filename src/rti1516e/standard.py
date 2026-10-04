"""Layer 2 — Rti1516eAmbassador (1516-2010 standard-shaped callback API).

Wraps Layer 1 (RtiConnection + Federate) internally; intended for users
porting from Java/C++ RTIs that use the ambassador callback pattern.

This is a thin adapter rather than a second transport implementation;
Layer 1 owns the gRPC and asyncio surface.

Methods preserve the IEEE 1516.1 names (camelCase) for portability;
Python style would use snake_case but that defeats the porting purpose.

The ambassador presents a synchronous API to the caller. Internally it
runs a private asyncio event loop in a background thread; each public
call schedules its async equivalent on that loop and waits for the
result. Immediate callbacks run on a serialized delivery thread; evoked
callbacks run only on the thread calling an evoke method.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
from collections import deque
from collections.abc import Callable, Collection, Coroutine
from concurrent.futures import Future, InvalidStateError
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from rti1516e import _transport
from rti1516e._callbacks import HLA_IMMEDIATE, CallbackDispatcher, CallbackModel, ServiceCallback
from rti1516e._services import FederationExecutionInformation, list_federation_executions
from rti1516e.connection import FederationSpec, RtiConnection
from rti1516e.errors import (
    AlreadyConnected,
    FederateNotExecutionMember,
    RestoreInProgress,
    SaveInProgress,
)
from rti1516e.events import (
    AttributeOwnershipAcquisitionNotification,
    AttributeOwnershipUnavailable,
    AttributesInScope,
    AttributesOutOfScope,
    DiscoverObjectInstance,
    FederationHalted,
    FederationNotRestored,
    FederationNotSaved,
    FederationRestoreBegun,
    FederationRestored,
    FederationSaved,
    FederationSynchronized,
    InitiateFederateRestore,
    InitiateFederateSave,
    MultipleObjectInstanceNameReservationFailed,
    MultipleObjectInstanceNameReservationSucceeded,
    ObjectCallbackMetadata,
    ObjectInstanceNameReservationFailed,
    ObjectInstanceNameReservationSucceeded,
    ProvideAttributeValueUpdate,
    ReceiveInteraction,
    ReflectAttributeValues,
    RemoveObjectInstance,
    RequestAttributeOwnershipAssumption,
    RequestAttributeOwnershipRelease,
    RequestDivestitureConfirmation,
    RequestFederationRestoreFailed,
    RequestFederationRestoreSucceeded,
    RequestRetraction,
    StartRegistrationForObjectClass,
    StopRegistrationForObjectClass,
    SynchronizationPointAnnounced,
    SynchronizationPointFailureReason,
    SynchronizationPointRegistrationFailed,
    SynchronizationPointRegistrationSucceeded,
    TimeAdvanceGrant,
    TimeConstrainedEnabled,
    TimeRegulationEnabled,
    TurnInteractionsOff,
    TurnInteractionsOn,
    TurnUpdatesOffForObjectInstance,
    TurnUpdatesOnForObjectInstance,
)
from rti1516e.factories import (
    AttributeHandleSetFactory,
    AttributeHandleValueMapFactory,
    DimensionHandleSetFactory,
    FederateHandleSetFactory,
    ParameterHandleValueMapFactory,
    RegionHandleSetFactory,
)
from rti1516e.fom.modules import MIMInput
from rti1516e.handles import (
    AttributeHandle,
    DimensionHandle,
    FederateHandle,
    InteractionClassHandle,
    MessageRetractionHandle,
    ObjectClassHandle,
    ObjectInstanceHandle,
    ParameterHandle,
    RegionHandle,
)
from rti1516e.savepoint import (
    FederateRestoreStatus,
    FederateSaveStatus,
    RestoreStatusResponse,
    SaveStatusResponse,
)
from rti1516e.sets import (
    AttributeHandleSet,
    AttributeHandleValueMap,
    DimensionHandleSet,
    FederateHandleSet,
    ParameterHandleSet,
    ParameterHandleValueMap,
    RegionHandleSet,
)

if TYPE_CHECKING:
    from rti1516e.connection import Federate, _FederateContextManager


# M28 — IEEE 1516 portability type aliases. Typed handles are int subclasses, so
# the bare-int callers from M25-M27 keep working at runtime; the alias
# documents that typed forms are accepted too and lets mypy --strict
# resolve mixed-typed reference_rti federate code unchanged.
ObjectClassRef: TypeAlias = "int | str | ObjectClassHandle"
AttributeRef: TypeAlias = "int | str | AttributeHandle"
InteractionClassRef: TypeAlias = "int | str | InteractionClassHandle"
ParameterRef: TypeAlias = "int | str | ParameterHandle"
ObjectInstanceRef: TypeAlias = "int | ObjectInstanceHandle"
DimensionRef: TypeAlias = "int | str | DimensionHandle"
FederateRef: TypeAlias = "int | FederateHandle"
RegionRef: TypeAlias = "int | RegionHandle"
AttributeRefList: TypeAlias = "list[int | str | AttributeHandle] | AttributeHandleSet"
ParameterRefList: TypeAlias = "list[int | str | ParameterHandle] | ParameterHandleSet"
FederateRefList: TypeAlias = "list[int | FederateHandle] | FederateHandleSet"
DimensionRefList: TypeAlias = "list[int | str | DimensionHandle] | DimensionHandleSet"
RegionRefList: TypeAlias = "list[int | RegionHandle] | RegionHandleSet"
AttributeValueDict: TypeAlias = "dict[int | str | AttributeHandle, Any] | AttributeHandleValueMap"
ParameterValueDict: TypeAlias = "dict[int | str | ParameterHandle, Any] | ParameterHandleValueMap"


class Rti1516eAmbassador:
    """1516-2010-shaped ambassador callback API. Wraps Layer 1.

    This is a base class; users subclass it and override the callback
    methods (e.g. ``discoverObjectInstance``, ``reflectAttributeValues``).

    Lifecycle:

        amb = MyAmbassador()
        amb.connect(amb, "memory://fake-rti")
        amb.createFederationExecution("demo", ["demo.fom.xml"])
        amb.joinFederationExecution("alice", "demo")
        amb.publishObjectClassAttributes("Vehicle", ["pos"])
        ...
        amb.resignFederationExecution()
        amb.disconnect()
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._url: str | None = None
        self._connection: RtiConnection | None = None
        self._connection_cm_open = False
        self._federation_name: str | None = None
        self._fom_modules: list[str] = []
        self._mim_module: MIMInput | None = None
        self._logical_time_implementation_name = ""
        self._federate_cm: _FederateContextManager | None = None
        self._federate: Federate | None = None
        self._event_pump_task: Future[None] | None = None
        self._direct_callback_delivery = False
        self._direct_event_sink_installed = False
        self._async_operations_lock = threading.Lock()
        self._async_operations_changed = threading.Condition(self._async_operations_lock)
        self._async_submission_gate = threading.RLock()
        self._async_operations: list[Future[Any]] = []
        self._async_inflight_count = 0
        self._async_operation_limit = 64
        self._async_ordering_barrier_pending = False
        self._async_closing = False
        self._callback_target: Rti1516eAmbassador = self
        # M26 Phase E — evokeCallback callback-fired counter. Bumped
        # exactly once per dispatched event by _pump_events; read by
        # evokeCallback to compute its bool return.
        self._callback_fired_count: int = 0
        # HLA_IMMEDIATE callbacks can land just before an evoke call samples
        # the counter. Retain the last reported count so that race is observed
        # once by the next evoke call instead of being lost.
        self._callback_observed_count: int = 0
        # M27 Phase C — §10.4 callback enable/disable. When False,
        # _dispatch_event buffers events to self._callback_buffer
        # instead of firing override slots; enableCallbacks drains
        # the buffer through the normal dispatch path.
        self._callbacks_enabled: bool = True
        self._callback_buffer: list[Any] = []
        self._local_notification_order = 0
        self._local_notification_lock = threading.RLock()
        self._callback_dispatcher = CallbackDispatcher(self._dispatch_callback, HLA_IMMEDIATE)
        self._callback_report_failures: deque[Exception] = deque(maxlen=32)
        # M28 — IEEE 1516 service-style factory singletons. Stateless; one instance per
        # ambassador per IEEE 1516 API convention.
        self._attribute_handle_set_factory = AttributeHandleSetFactory()
        self._attribute_handle_value_map_factory = AttributeHandleValueMapFactory()
        self._parameter_handle_value_map_factory = ParameterHandleValueMapFactory()
        self._federate_handle_set_factory = FederateHandleSetFactory()
        self._dimension_handle_set_factory = DimensionHandleSetFactory()
        self._region_handle_set_factory = RegionHandleSetFactory()
        # M39 — per-(class, callback) cache of which optional kwargs an
        # override accepts (see _invoke_compat).
        self._accepted_kwargs_cache: dict[tuple[type, str], frozenset[str] | None] = {}

    # --- Connection / federation lifecycle ---

    def connect(
        self, callback_target: Rti1516eAmbassador, url: str,
        *, callback_model: CallbackModel | str = HLA_IMMEDIATE,
    ) -> None:
        """Open the connection. Wraps RtiConnection.connect.

        ``callback_target`` is the object whose ``discover*``/``reflect*`` etc.
        methods are invoked when events arrive. In the common case it's
        ``self`` (subclass override pattern).
        """
        self._callback_dispatcher.guard_reentrant()
        if self._connection is not None:
            raise AlreadyConnected("ambassador is already connected")
        dispatcher = CallbackDispatcher(self._dispatch_callback, callback_model)
        self._callback_dispatcher.close()
        self._callback_dispatcher = dispatcher
        self._callback_error: BaseException | None = None
        # Spin up a private event loop in a background thread. All Layer 1
        # asyncio work runs there; the caller stays sync.
        self._callback_target = callback_target if callback_target is not None else self
        self._url = url
        with self._async_operations_changed:
            self._async_closing = False
        self._start_loop()
        connection = RtiConnection.connect(url)
        try:
            self._connection = self._run(connection.__aenter__())
        except BaseException:
            dispatcher.close()
            self._stop_loop()
            raise
        self._connection_cm_open = True

    def disconnect(self) -> None:
        """Tear down the connection. Idempotent."""
        self._callback_dispatcher.guard_reentrant()
        self._callback_dispatcher.close()
        pending_error: BaseException | None = None
        with self._async_submission_gate:
            with self._async_operations_changed:
                self._async_closing = True
                self._async_operations_changed.notify_all()
            try:
                self._flush_async_operations_locked()
            except BaseException as exc:  # cleanup must still close the channel
                pending_error = exc
            if self._connection is not None and self._connection_cm_open:
                self._run(self._connection.__aexit__(None, None, None))
                self._connection_cm_open = False
            self._connection = None
            self._stop_loop()
        if pending_error is not None:
            raise pending_error

    def createFederationExecution(  # noqa: N802
        self,
        name: str,
        fom_modules: list[str],
        *,
        logical_time_implementation_name: str = "",
        mim_module: MIMInput | None = None,
    ) -> None:
        """§4.5 — create the federation execution.

        Requires an open connection; disconnected calls do not cache inputs.
        M39 HA-2: the create RPC is issued EAGERLY and a
        duplicate name raises the typed
        :class:`rti1516e.errors.FederationExecutionAlreadyExists`
        (IEEE §4.5) — the standard HLA pattern is::

            try:
                amb.createFederationExecution(name, foms)
            except FederationExecutionAlreadyExists:
                pass  # someone else created it first
            amb.joinFederationExecution(federate, name)

        The spec is also stashed for the upcoming join, and the rolled
        create-on-join path stays idempotent — federates that skip
        ``createFederationExecution`` entirely keep working (Layer 1's
        ``join_federation`` creates with ``exist_ok=True``).

        ``logical_time_implementation_name`` is keyword-only. Empty selects
        the default ``HLAfloat64Time``, the only implementation supported by
        the current transport. Other designators raise
        ``CouldNotCreateLogicalTimeFactory`` before creation or caching.
        ``mim_module`` selects an explicit MIM path or ``MIMModule`` value,
        separate from the FOM modules. The server validates its supported
        structural profile. Explicit selection requires the V2 create contract.
        """
        if self._connection is None or not self._connection_cm_open:
            raise RuntimeError("connect() must be called before createFederationExecution()")
        spec = FederationSpec(name=name, fom_modules=fom_modules, mim_module=mim_module,
                              logical_time_implementation_name=logical_time_implementation_name)
        self._run(self._connection.create_federation(spec))
        self._federation_name = name
        self._fom_modules = list(spec.fom_modules)
        self._mim_module = spec.mim_module
        self._logical_time_implementation_name = spec.logical_time_implementation_name

    def destroyFederationExecution(self, name: str) -> None:  # noqa: N802
        """§4.6 — destroy the federation execution.

        M39 HA-2. Raises the typed
        :class:`rti1516e.errors.FederatesCurrentlyJoined` while members
        remain joined and
        :class:`rti1516e.errors.FederationExecutionDoesNotExist` for an
        unknown name.
        """
        if self._connection is None:
            raise RuntimeError("connect() must be called before destroyFederationExecution()")
        self._run(self._connection.destroy_federation(name))

    def joinFederationExecution(  # noqa: N802
        self,
        federate_name: str,
        federation_name: str,
        additional_fom_modules: list[str] | None = None,
        *,
        federate_type: str = "",
    ) -> None:
        """Join an existing federation, optionally extending its live FOM.

        The legacy positions remain ``(federate_name, federation_name,
        additional_fom_modules=None)``. Supply the type explicitly, for
        example ``joinFederationExecution("alice", "demo", federate_type="Vehicle")``.
        It is forwarded unchanged to Layer 1; omitted type stays empty.
        A three-string C++/Java-style call is rejected rather than treating
        the type as a federation name and the name as a sequence of modules.
        """
        self._callback_dispatcher.guard_reentrant()
        if self._connection is None:
            raise RuntimeError("connect() must be called before joinFederationExecution()")
        if isinstance(additional_fom_modules, (str, bytes)):
            raise TypeError(
                "additional_fom_modules must be a sequence of module paths, not a string; "
                "pass federate_type as a keyword"
            )
        if not isinstance(federate_type, str):
            raise TypeError("federate_type must be a string")
        # Honor a prior createFederationExecution; otherwise default to the
        # passed federation_name with no FOM modules.
        spec = FederationSpec(
            name=federation_name,
            fom_modules=list(self._fom_modules) if self._federation_name == federation_name else [],
            additional_fom_modules=list(additional_fom_modules or []),
            mim_module=self._mim_module if self._federation_name == federation_name else None,
            logical_time_implementation_name=(self._logical_time_implementation_name
                                               if self._federation_name == federation_name else ""),
        )
        cm = self._connection.join_federation(
            spec, federate_name=federate_name, federate_type=federate_type,
        )
        self._federate_cm = cm
        self._federate = self._run(cm.__aenter__())
        self._ownership_receipt_profile = "legacy-delivery-only"
        transport = getattr(self._federate, "_transport", None)
        generation = getattr(transport, "_generation_by_federation", {}).get(federation_name, 0)
        if generation:
            try:
                self._run(self._federate.enable_ownership_callback_receipts())
                self._ownership_receipt_profile = "actual-invocation"
            except Exception as exc:
                from ._grpc_errors import _grpc_code_name
                if _grpc_code_name(exc) != "UNIMPLEMENTED":
                    try:
                        self._run(cm.__aexit__(type(exc), exc, exc.__traceback__))
                    finally:
                        self._federate = None
                        self._federate_cm = None
                    raise
        if self._direct_callback_delivery and self._run(
            self._federate.set_event_sink(self._dispatch_event)
        ):
            self._direct_event_sink_installed = True
        else:
            # Start draining events into the user's callbacks.
            self._event_pump_task = asyncio.run_coroutine_threadsafe(
                self._pump_events(), self._loop_required()
            )
        sources = getattr(self._federate._transport, "_checkpoint_sources", None)
        if sources is not None:
            sources[self._federate.handle] = self._checkpoint_pending_callbacks

    def setDirectCallbackDelivery(self, enabled: bool) -> None:  # noqa: N802
        """Select direct stream-to-callback delivery before joining.

        Direct delivery removes the intermediate asyncio queue and pump task.
        Delivery still obeys the selected callback model and stream order.
        Transports without sink support retain the queue fallback.
        """
        if not isinstance(enabled, bool):
            raise TypeError("direct callback delivery requires a bool")
        if self._federate is not None:
            raise RuntimeError("direct callback delivery must be set before joining")
        self._direct_callback_delivery = enabled

    def resignFederationExecution(  # noqa: N802
        self, action: str = "UNCONDITIONALLY_DIVEST_ATTRIBUTES"
    ) -> None:
        """IEEE 1516.1-2010 §4.10 — resign with an explicit action.

        M36: the action designator is threaded through the federate
        context manager down to the wire (M24 W2 ResignAction enum), so
        e.g. ``CANCEL_THEN_DELETE_THEN_DIVEST`` makes the rtid delete
        the resigning federate's instances (subscribers see REMOVE).
        Unknown designators raise ``ValueError`` before any state is
        torn down (§4.10 InvalidResignAction).
        """
        self._callback_dispatcher.guard_reentrant()
        if action not in _transport.RESIGN_ACTION_NAMES:
            valid = ", ".join(sorted(_transport.RESIGN_ACTION_NAMES))
            raise ValueError(f"invalid resign action {action!r}; expected one of: {valid}")
        pending_error: BaseException | None = None
        with self._async_submission_gate:
            try:
                self._flush_async_operations_locked()
            except BaseException as exc:
                pending_error = exc
            # Rejection leaves membership and its callback delivery usable.
            if self._federate_cm is not None:
                self._federate_cm.resign_action = action
                self._run(self._federate_cm.__aexit__(None, None, None))
                self._federate_cm = None
            if self._event_pump_task is not None:
                self._event_pump_task.cancel()
                self._event_pump_task = None
            if self._direct_event_sink_installed and self._federate is not None:
                self._run(self._federate.set_event_sink(None))
                self._direct_event_sink_installed = False
            self._federate = None
        self._callback_dispatcher.clear()
        if pending_error is not None:
            raise pending_error

    # --- Declaration management ---

    def listFederationExecutions(self) -> tuple[FederationExecutionInformation, ...]:  # noqa: N802
        if self._connection is None:
            raise RuntimeError("connect() must be called before listFederationExecutions()")
        rows = cast("tuple[FederationExecutionInformation, ...]",
                    self._run(list_federation_executions(self._connection)))
        self._dispatch_event(ServiceCallback("reportFederationExecutions", (rows,)))
        return rows

    def reportFederationExecutions(  # noqa: N802
        self, executions: tuple[FederationExecutionInformation, ...],
    ) -> None:
        """Override to receive the available execution-list fields."""

    def unpublishObjectClass(self, class_handle: ObjectClassRef) -> None:  # noqa: N802
        self._run(self._fed().unpublish_object_class(class_handle))

    def unpublishObjectClassAttributes(  # noqa: N802
        self, class_handle: ObjectClassRef, attributes: AttributeRefList,
    ) -> None:
        self._run(self._fed().unpublish_object_class(class_handle, attributes=list(attributes)))

    def unsubscribeObjectClass(self, class_handle: ObjectClassRef) -> None:  # noqa: N802
        self._run(self._fed().unsubscribe_object_class(class_handle))

    def unsubscribeObjectClassAttributes(  # noqa: N802
        self, class_handle: ObjectClassRef, attributes: AttributeRefList,
    ) -> None:
        self._run(self._fed().unsubscribe_object_class(class_handle, attributes=list(attributes)))

    def unpublishInteractionClass(self, class_handle: InteractionClassRef) -> None:  # noqa: N802
        self._run(self._fed().unpublish_interaction_class(class_handle))

    def unsubscribeInteractionClass(self, class_handle: InteractionClassRef) -> None:  # noqa: N802
        self._run(self._fed().unsubscribe_interaction_class(class_handle))

    def setAutomaticResignDirective(self, action: int | str) -> None:  # noqa: N802
        self._run(self._fed().set_automatic_resign_directive(action))

    def getAutomaticResignDirective(self) -> int:  # noqa: N802
        return int(self._run(self._fed().get_automatic_resign_directive()))

    def getFederateHandle(self, name: str) -> FederateHandle:  # noqa: N802
        return FederateHandle(self._run(self._fed().get_federate_handle(name)))

    def getFederateName(self, handle: int) -> str:  # noqa: N802
        return str(self._run(self._fed().get_federate_name(handle)))

    def normalizeFederateHandle(self, handle: int) -> int:  # noqa: N802
        return int(self._run(self._fed().support.normalize_federate_handle(handle)))

    def normalizeServiceGroup(self, group: int) -> int:  # noqa: N802
        return int(self._run(self._fed().support.normalize_service_group(group)))

    def queryGALT(self) -> tuple[float, bool]:  # noqa: N802
        return cast("tuple[float, bool]", self._run(self._fed().query_galt()))

    def queryLITS(self) -> tuple[float, bool]:  # noqa: N802
        return cast("tuple[float, bool]", self._run(self._fed().query_lits()))

    def retract(self, handle: int) -> None:
        self._run_after_async_barrier(lambda: self._fed().retract(handle))

    def changeAttributeOrderType(  # noqa: N802
        self, object_handle: int, attribute_handles: list[int], order_type: int,
    ) -> None:
        self._run(self._fed().change_attribute_order_type(
            object_handle, list(attribute_handles), order_type))

    def changeInteractionOrderType(self, class_handle: int, order_type: int) -> None:  # noqa: N802
        self._run(self._fed().change_interaction_order_type(class_handle, order_type))

    def requestAttributeTransportationTypeChange(  # noqa: N802
        self, object_handle: int, attributes: list[int], transport_type: int,
    ) -> None:
        copied = list(attributes)
        self._run(self._fed().request_attribute_transportation_type_change(
            object_handle, copied, transport_type))
        self._dispatch_event(ServiceCallback(
            "confirmAttributeTransportationTypeChange", (object_handle, copied, transport_type)))

    def queryAttributeTransportationType(  # noqa: N802
        self, object_handle: int, attribute_handle: int,
    ) -> None:
        result = self._run(self._fed().query_attribute_transportation_type(
            object_handle, attribute_handle))
        self._dispatch_event(ServiceCallback(
            "reportAttributeTransportationType", (object_handle, attribute_handle, result)))

    def requestInteractionTransportationTypeChange(  # noqa: N802
        self, class_handle: int, transport_type: int,
    ) -> None:
        self._run(self._fed().request_interaction_transportation_type_change(
            class_handle, transport_type))
        self._dispatch_event(ServiceCallback(
            "confirmInteractionTransportationTypeChange", (class_handle, transport_type)))

    def queryInteractionTransportationType(  # noqa: N802
        self, federate_handle: int, class_handle: int,
    ) -> None:
        result = self._run(self._fed().query_interaction_transportation_type(
            federate_handle, class_handle))
        self._dispatch_event(ServiceCallback(
            "reportInteractionTransportationType", (federate_handle, class_handle, result)))

    def confirmAttributeTransportationTypeChange(  # noqa: N802
        self, object_handle: int, attributes: Collection[int], transport_type: int,
    ) -> None:
        """Override to receive a completed attribute transportation change."""

    def reportAttributeTransportationType(  # noqa: N802
        self, object_handle: int, attribute_handle: int, transport_type: int,
    ) -> None:
        """Override to receive the attribute transportation query result."""

    def confirmInteractionTransportationTypeChange(  # noqa: N802
        self, class_handle: int, transport_type: int,
    ) -> None:
        """Override to receive a completed interaction transportation change."""

    def reportInteractionTransportationType(  # noqa: N802
        self, federate_handle: int, class_handle: int, transport_type: int,
    ) -> None:
        """Override to receive the interaction transportation query result."""

    def attributeOwnershipReleaseDenied(  # noqa: N802
        self, object_handle: int, attribute_handles: list[int],
    ) -> None:
        self._run(self._fed().ownership.release_denied(object_handle, list(attribute_handles)))

    def abortFederationSave(self) -> None:  # noqa: N802
        self._run(self._fed().savepoint.abort_federation_save())

    def abortFederationRestore(self) -> None:  # noqa: N802
        self._run(self._fed().savepoint.abort_federation_restore())

    def getRangeBounds(self, region_handle: int, dimension_handle: int) -> tuple[int, int] | None:  # noqa: N802
        return cast("tuple[int, int] | None", self._run(
            self._fed().ddm.query_bounds(region_handle, dimension_handle)))

    def reserveMultipleObjectInstanceName(self, names: list[str]) -> None:  # noqa: N802
        self.reserveMultipleObjectInstanceNames(names)

    def releaseMultipleObjectInstanceName(self, names: list[str]) -> None:  # noqa: N802
        self._run(self._fed().reservation.release_multiple(list(names)))

    def releaseMultipleObjectInstanceNames(self, names: list[str]) -> None:  # noqa: N802
        self.releaseMultipleObjectInstanceName(names)

    def publishObjectClassAttributes(  # noqa: N802
        self, class_name: ObjectClassRef, attributes: AttributeRefList
    ) -> None:
        """M27 Phase B: ``class_name`` accepts ``int`` (IEEE 1516 service-style FOM
        handle, e.g. from ``getObjectClassHandle``) or ``str`` (FOM name).
        Each entry of ``attributes`` is independently ``int`` or ``str``."""
        self._run(self._fed().publish_object_class(class_name, attributes=list(attributes)))

    def subscribeObjectClassAttributes(  # noqa: N802
        self,
        class_name: ObjectClassRef,
        attributes: AttributeRefList,
        active: bool = True,
        updateRateDesignator: str = "",  # noqa: N803
    ) -> None:
        """See :meth:`publishObjectClassAttributes` for the M27 Phase B
        int|str semantics."""
        self._run(
            self._fed().subscribe_object_class(
                class_name,
                attributes=list(attributes),
                active=active,
                update_rate_designator=updateRateDesignator,
            )
        )

    def publishInteractionClass(self, class_name: InteractionClassRef) -> None:  # noqa: N802
        """M27 Phase D: ``class_name`` accepts ``int`` (FOM handle) or ``str``."""
        self._run(self._fed().publish_interaction_class(class_name))

    def subscribeInteractionClass(  # noqa: N802
        self, class_name: InteractionClassRef, active: bool = True
    ) -> None:
        """M27 Phase D: ``class_name`` accepts ``int`` (FOM handle) or ``str``."""
        self._run(self._fed().subscribe_interaction_class(class_name, active=active))

    def enableObjectClassRelevanceAdvisorySwitch(self) -> None:  # noqa: N802
        self._run(self._fed().set_advisory_switch(1, True))

    def disableObjectClassRelevanceAdvisorySwitch(self) -> None:  # noqa: N802
        self._run(self._fed().set_advisory_switch(1, False))

    def enableAttributeRelevanceAdvisorySwitch(self) -> None:  # noqa: N802
        self._run(self._fed().set_advisory_switch(2, True))

    def disableAttributeRelevanceAdvisorySwitch(self) -> None:  # noqa: N802
        self._run(self._fed().set_advisory_switch(2, False))

    def enableAttributeScopeAdvisorySwitch(self) -> None:  # noqa: N802
        self._run(self._fed().set_advisory_switch(3, True))

    def disableAttributeScopeAdvisorySwitch(self) -> None:  # noqa: N802
        self._run(self._fed().set_advisory_switch(3, False))

    def enableInteractionRelevanceAdvisorySwitch(self) -> None:  # noqa: N802
        self._run(self._fed().set_advisory_switch(4, True))

    def disableInteractionRelevanceAdvisorySwitch(self) -> None:  # noqa: N802
        self._run(self._fed().set_advisory_switch(4, False))

    # --- Object management ---

    def registerObjectInstance(  # noqa: N802
        self, class_name: ObjectClassRef, instance_name: str | None = None
    ) -> ObjectInstanceHandle:
        """Register using an object-class handle or FOM class name.

        An explicit instance name requires a successful
        ``objectInstanceNameReservationSucceeded`` callback from a prior
        ``reserveObjectInstanceName`` call. Registration does not reserve it
        implicitly. Registration and deletion retain that reservation until
        it is explicitly released or the reserving federate resigns.
        Omit ``instance_name`` to request an automatically assigned name.
        """
        result = self._run_after_async_barrier(
            lambda: self._fed().register_object_instance(class_name, instance_name=instance_name)
        )
        return ObjectInstanceHandle(result)

    def updateAttributeValues(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        values: AttributeValueDict,
        timestamp: float | None = None,
    ) -> None:
        """M27 Phase B: ``values`` dict keys accept ``int`` (IEEE 1516 service-style
        attribute handle) or ``str`` (FOM attribute name)."""
        copied_values = dict(values)
        self._run_after_async_barrier(
            lambda: self._fed().update_attributes(object_handle, copied_values, timestamp=timestamp)
        )

    def updateAttributeValuesAsync(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        values: AttributeValueDict,
        timestamp: float | None = None,
    ) -> Future[None]:
        """Submit an attribute update without blocking the calling thread.

        This is a gorti extension, not an IEEE 1516.1 method. Submitted
        operations are bounded by ``setAsyncOperationLimit`` and must be
        observed through the returned Future or ``flushAsyncOperations``.
        """
        copied_values = dict(values)
        return cast(
            "Future[None]",
            self._submit_async_operation(
                lambda: self._fed().update_attributes(
                    object_handle, copied_values, timestamp=timestamp
                )
            ),
        )

    def sendInteraction(  # noqa: N802
        self,
        class_name: InteractionClassRef,
        parameters: ParameterValueDict,
        timestamp: float | None = None,
    ) -> None:
        """M27 Phase B: ``class_name`` and ``parameters`` dict keys
        accept ``int`` (handle) or ``str`` (FOM name)."""
        copied_parameters = dict(parameters)
        self._run_after_async_barrier(
            lambda: self._fed().send_interaction(class_name, copied_parameters, timestamp=timestamp)
        )

    def sendInteractionAsync(  # noqa: N802
        self,
        class_name: InteractionClassRef,
        parameters: ParameterValueDict,
        timestamp: float | None = None,
    ) -> Future[None]:
        """Submit an interaction without blocking the calling thread.

        Operations submitted on one ambassador may execute concurrently. Use
        ``flushAsyncOperations`` before a dependent call when cross-operation
        completion order matters.
        """
        copied_parameters = dict(parameters)
        return cast(
            "Future[None]",
            self._submit_async_operation(
                lambda: self._fed().send_interaction(
                    class_name, copied_parameters, timestamp=timestamp
                )
            ),
        )

    # --- Time management ---

    def enableTimeRegulation(self, lookahead: float) -> None:  # noqa: N802
        self._run(self._fed().enable_time_regulation(lookahead))

    def disableTimeRegulation(self) -> None:  # noqa: N802
        self._run(self._fed().disable_time_regulation())

    def enableTimeConstrained(self) -> None:  # noqa: N802
        self._run(self._fed().enable_time_constrained())

    def disableTimeConstrained(self) -> None:  # noqa: N802
        self._run(self._fed().disable_time_constrained())

    def modifyLookahead(self, lookahead: float) -> None:  # noqa: N802
        self._run(self._fed().modify_lookahead(lookahead))

    def nextMessageRequest(self, time: float) -> None:  # noqa: N802
        self._run_after_async_barrier(lambda: self._fed().next_message_request(time))

    def nextMessageRequestAvailable(self, time: float) -> None:  # noqa: N802
        self._run_after_async_barrier(lambda: self._fed().next_message_request_available(time))

    def timeAdvanceRequest(self, time: float) -> None:  # noqa: N802
        self._run_after_async_barrier(lambda: self._fed().time_advance_request(time))

    def timeAdvanceRequestAsync(self, time: float) -> Future[Any]:  # noqa: N802
        """Submit TAR without blocking the caller on the transport round trip.

        This extension preserves call ordering by draining earlier asynchronous
        OM work before TAR is submitted. Later asynchronous work waits for TAR
        acceptance, and dependent synchronous calls retain their existing
        implicit flush barrier.
        """
        with self._async_submission_gate:
            self._flush_async_operations_locked()
            future = self._submit_async_operation(lambda: self._fed().time_advance_request(time))
            self._async_ordering_barrier_pending = True
            return future

    def setAsyncOperationLimit(self, limit: int) -> None:  # noqa: N802
        """Set the maximum number of unflushed asynchronous operations."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("async operation limit must be a positive integer")
        with self._async_submission_gate, self._async_operations_changed:
            if self._async_inflight_count or self._async_operations:
                raise RuntimeError("async operation limit requires an empty generation")
            self._async_operation_limit = limit

    def flushAsyncOperations(self) -> None:  # noqa: N802
        """Wait for all submitted extension operations and raise their first error.

        Every operation is observed before the first exception is re-raised, so
        a failed update cannot leave later submitted work silently abandoned.
        """
        with self._async_submission_gate:
            self._flush_async_operations_locked()

    def _flush_async_operations_locked(self) -> None:
        with self._async_operations_changed:
            pending = self._async_operations
            self._async_operations = []
        if pending and threading.current_thread() is self._loop_thread:
            with self._async_operations_changed:
                self._async_operations = pending + self._async_operations
            raise RuntimeError("cannot flush async operations from the ambassador loop")
        self._async_ordering_barrier_pending = False
        first_error: BaseException | None = None
        for future in pending:
            try:
                future.result()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def timeAdvanceRequestAvailable(self, time: float) -> None:  # noqa: N802
        self._run_after_async_barrier(lambda: self._fed().time_advance_request_available(time))

    def flushQueueRequest(self, time: float) -> None:  # noqa: N802
        self._run_after_async_barrier(lambda: self._fed().flush_queue_request(time))

    def getLogicalTimeImplementationName(self) -> str:  # noqa: N802
        """SDK inspection accessor for the joined execution's selected time type."""
        return self._fed().logical_time_implementation_name

    def queryLogicalTime(self) -> int | float:  # noqa: N802
        return cast("int | float", self._run(self._fed().query_logical_time()))

    def queryLookahead(self) -> int | float:  # noqa: N802
        return cast("int | float", self._run(self._fed().query_lookahead()))

    def queryLBTS(self) -> tuple[float, bool]:  # noqa: N802
        return cast("tuple[float, bool]", self._run(self._fed().query_lbts()))

    def enableAsynchronousDelivery(self) -> None:  # noqa: N802
        self._run(self._fed().enable_asynchronous_delivery())

    def disableAsynchronousDelivery(self) -> None:  # noqa: N802
        self._run(self._fed().disable_asynchronous_delivery())

    # --- M23 W1: §6 deleteObjectInstance ---

    def deleteObjectInstance(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        tag: bytes = b"",
        timestamp: float | None = None,
    ) -> None:
        self._run_after_async_barrier(
            lambda: self._fed().delete_object_instance(object_handle, tag, timestamp)
        )

    def localDeleteObjectInstance(self, object_handle: ObjectInstanceRef) -> None:  # noqa: N802
        with self._fed()._ownership_callback_state().dispatch_fence:
            self._run_after_async_barrier(
                lambda: self._fed().local_delete_object_instance(object_handle)
            )

    def enableOwnershipCallbackReceipts(self) -> None:  # noqa: N802
        self._run(self._fed().enable_ownership_callback_receipts())

    def retryOwnershipCallbacks(self) -> None:  # noqa: N802
        self._run(self._fed().retry_ownership_callbacks())

    def requestAttributeValueUpdate(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        attribute_handles: AttributeRefList,
        tag: bytes = b"",
    ) -> None:
        self._run(
            self._fed().request_attribute_value_update(
                object_handle,
                list(attribute_handles),
                tag,
            )
        )

    def requestClassAttributeValueUpdate(  # noqa: N802
        self,
        object_class_handle: ObjectClassRef,
        attribute_handles: AttributeRefList,
        tag: bytes = b"",
    ) -> None:
        self._run(
            self._fed().request_class_attribute_value_update(
                object_class_handle,
                list(attribute_handles),
                tag,
            )
        )

    def changeAttributeTransportationType(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        attribute_handles: AttributeRefList,
        transport: int,
    ) -> None:
        self._run(
            self._fed().change_attribute_transportation_type(
                object_handle,
                list(attribute_handles),
                transport,
            )
        )

    def changeInteractionTransportationType(  # noqa: N802
        self,
        interaction_class_handle: InteractionClassRef,
        transport: int,
    ) -> None:
        self._run(
            self._fed().change_interaction_transportation_type(
                interaction_class_handle,
                transport,
            )
        )

    # --- §10.2 Support services (M25 Phase B) ---
    # IEEE 1516 service-style sync wrappers around the async SupportClient. Each
    # delegates to fed.support.<method> via _run(). Federates ported
    # from reference_rti use these directly.

    def getObjectClassHandle(self, class_name: str) -> ObjectClassHandle:  # noqa: N802
        return ObjectClassHandle(self._run(self._fed().support.get_object_class_handle(class_name)))

    def getObjectClassName(self, class_handle: ObjectClassRef) -> str:  # noqa: N802
        return str(self._run(self._fed().support.get_object_class_name(class_handle)))

    def getAttributeHandle(  # noqa: N802
        self, class_handle: ObjectClassRef, attribute_name: str
    ) -> AttributeHandle:
        return AttributeHandle(
            self._run(self._fed().support.get_attribute_handle(class_handle, attribute_name))
        )

    def getAttributeName(  # noqa: N802
        self, class_handle: ObjectClassRef, attribute_handle: AttributeRef
    ) -> str:
        return str(
            self._run(self._fed().support.get_attribute_name(class_handle, attribute_handle))
        )

    def getInteractionClassHandle(self, class_name: str) -> InteractionClassHandle:  # noqa: N802
        return InteractionClassHandle(
            self._run(self._fed().support.get_interaction_class_handle(class_name))
        )

    def getInteractionClassName(self, class_handle: InteractionClassRef) -> str:  # noqa: N802
        return str(self._run(self._fed().support.get_interaction_class_name(class_handle)))

    def getParameterHandle(  # noqa: N802
        self, class_handle: InteractionClassRef, parameter_name: str
    ) -> ParameterHandle:
        return ParameterHandle(
            self._run(self._fed().support.get_parameter_handle(class_handle, parameter_name))
        )

    def getParameterName(  # noqa: N802
        self, class_handle: InteractionClassRef, parameter_handle: ParameterRef
    ) -> str:
        return str(
            self._run(self._fed().support.get_parameter_name(class_handle, parameter_handle))
        )

    def getDimensionHandle(self, dimension_name: str) -> DimensionHandle:  # noqa: N802
        return DimensionHandle(self._run(self._fed().support.get_dimension_handle(dimension_name)))

    def getDimensionName(self, dimension_handle: DimensionRef) -> str:  # noqa: N802
        return str(self._run(self._fed().support.get_dimension_name(dimension_handle)))

    def getDimensionUpperBound(self, dimension_handle: DimensionRef) -> int:  # noqa: N802
        return int(self._run(self._fed().support.get_dimension_upper_bound(dimension_handle)))

    def getUpdateRateValue(self, designator: str) -> float:  # noqa: N802
        return float(self._run(self._fed().support.get_update_rate_value(designator)))

    def getUpdateRateValueForAttribute(self, object_handle: int, attribute_handle: int) -> float:  # noqa: N802
        return float(
            self._run(
                self._fed().support.get_update_rate_value_for_attribute(
                    object_handle, attribute_handle
                )
            )
        )

    def getAvailableDimensionsForClassAttribute(  # noqa: N802
        self, class_handle: int, attribute_handle: int
    ) -> DimensionHandleSet:  # noqa: N802
        return DimensionHandleSet(
            self._run(
                self._fed().support.get_available_dimensions_for_class_attribute(
                    class_handle, attribute_handle
                )
            )
        )

    def getAvailableDimensionsForInteractionClass(self, class_handle: int) -> DimensionHandleSet:  # noqa: N802
        return DimensionHandleSet(
            self._run(
                self._fed().support.get_available_dimensions_for_interaction_class(class_handle)
            )
        )

    def getDimensionHandleSet(self, region_handle: int) -> DimensionHandleSet:  # noqa: N802
        return DimensionHandleSet(
            self._run(self._fed().support.get_dimension_handle_set(region_handle))
        )

    def getOrderType(self, order_name: str) -> int:  # noqa: N802
        return int(self._run(self._fed().support.get_order_type(order_name)))

    def getOrderName(self, order_type: int) -> str:  # noqa: N802
        return str(self._run(self._fed().support.get_order_name(order_type)))

    def getTransportationType(self, transportation_name: str) -> int:  # noqa: N802
        return int(self._run(self._fed().support.get_transportation_type(transportation_name)))

    def getTransportationName(self, transportation_type: int) -> str:  # noqa: N802
        return str(self._run(self._fed().support.get_transportation_name(transportation_type)))

    def getObjectInstanceHandle(self, object_name: str) -> ObjectInstanceHandle:  # noqa: N802
        """§6.30 — resolve a runtime object instance name to its handle.
        M27 Phase C."""
        return ObjectInstanceHandle(
            self._run(self._fed().support.get_object_instance_handle(object_name))
        )

    def getObjectInstanceName(self, object_handle: ObjectInstanceRef) -> str:  # noqa: N802
        """§6.31 — resolve a runtime object instance handle to its name."""
        return str(self._run(self._fed().support.get_object_instance_name(object_handle)))

    def getKnownObjectClassHandle(self, object_handle: ObjectInstanceRef) -> ObjectClassHandle:  # noqa: N802
        return ObjectClassHandle(
            self._run(self._fed().support.get_known_object_class_handle(object_handle))
        )

    # --- §10.6 Handle factory accessors (M28 W2) ---

    def getAttributeHandleSetFactory(self) -> AttributeHandleSetFactory:  # noqa: N802
        return self._attribute_handle_set_factory

    def getAttributeHandleValueMapFactory(self) -> AttributeHandleValueMapFactory:  # noqa: N802
        return self._attribute_handle_value_map_factory

    def getParameterHandleValueMapFactory(self) -> ParameterHandleValueMapFactory:  # noqa: N802
        return self._parameter_handle_value_map_factory

    def getFederateHandleSetFactory(self) -> FederateHandleSetFactory:  # noqa: N802
        return self._federate_handle_set_factory

    def getDimensionHandleSetFactory(self) -> DimensionHandleSetFactory:  # noqa: N802
        return self._dimension_handle_set_factory

    def getRegionHandleSetFactory(self) -> RegionHandleSetFactory:  # noqa: N802
        return self._region_handle_set_factory

    # --- §4.11-4.13 Synchronization points (M25 Phase C) ---

    def registerFederationSynchronizationPoint(  # noqa: N802
        self, label: str, tag: bytes = b"", sync_set: FederateRefList | None = None
    ) -> None:
        self._run(
            self._fed().sync.register_synchronization_point(
                label,
                tag=tag,
                required_federates=list(sync_set) if sync_set is not None else None,
            )
        )

    def synchronizationPointAchieved(  # noqa: N802
        self, label: str, successfully: bool = True
    ) -> None:
        """§4.14 — M39 HA-2: ``successfully=False`` still counts toward
        the sync transition but lands this federate in the §4.15
        ``failed_to_sync`` set on federationSynchronized."""
        self._run(
            self._fed().sync.synchronization_point_achieved(
                label,
                successfully=successfully,
            )
        )

    # --- §7 Ownership Management (M25 Phase C) ---

    def unconditionalAttributeOwnershipDivestiture(  # noqa: N802
        self, object_handle: ObjectInstanceRef, attribute_handles: AttributeRefList
    ) -> None:
        self._run(
            self._fed().ownership.unconditional_divest(object_handle, list(attribute_handles))
        )

    def negotiatedAttributeOwnershipDivestiture(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        attribute_handles: AttributeRefList,
        tag: bytes = b"",
        two_phase: bool = False,
    ) -> None:
        """§7.3 — M39 HA-2: ``two_phase=True`` parks the transfer on
        requestDivestitureConfirmation until :meth:`confirmDivestiture`."""
        self._run(
            self._fed().ownership.negotiated_divest(
                object_handle, list(attribute_handles), tag=tag, two_phase=two_phase
            )
        )

    def confirmDivestiture(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        attribute_handles: AttributeRefList,
        tag: bytes = b"",
    ) -> None:
        """§7.6 — complete a parked two-phase negotiated divest (M39 HA-2)."""
        self._run(
            self._fed().ownership.confirm_divestiture(
                object_handle, list(attribute_handles), tag=tag
            )
        )

    def attributeOwnershipAcquisition(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        attribute_handles: AttributeRefList,
        tag: bytes = b"",
    ) -> None:
        self._run(self._fed().ownership.acquire(object_handle, list(attribute_handles), tag=tag))

    def attributeOwnershipAcquisitionIfAvailable(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        attribute_handles: AttributeRefList,
        tag: bytes = b"",
    ) -> None:
        """§7.9 — grab only currently-unowned attributes; nothing is
        queued (M39 HA-2). The unavailable subset arrives via the §7.10
        attributeOwnershipUnavailable callback."""
        self._run(
            self._fed().ownership.acquire(
                object_handle, list(attribute_handles), tag=tag, if_available=True
            )
        )

    def cancelNegotiatedAttributeOwnershipDivestiture(  # noqa: N802
        self, object_handle: ObjectInstanceRef, attribute_handles: AttributeRefList
    ) -> None:
        self._run(
            self._fed().ownership.cancel_negotiated_divest(object_handle, list(attribute_handles))
        )

    def cancelAttributeOwnershipAcquisition(  # noqa: N802
        self, object_handle: ObjectInstanceRef, attribute_handles: AttributeRefList
    ) -> None:
        self._run(self._fed().ownership.cancel_acquire(object_handle, list(attribute_handles)))

    def attributeOwnershipDivestitureIfWanted(  # noqa: N802
        self, object_handle: ObjectInstanceRef, attribute_handles: AttributeRefList
    ) -> tuple[int, ...]:
        return cast(
            "tuple[int, ...]",
            self._run(
                self._fed().ownership.divest_if_wanted_with_result(
                    object_handle, list(attribute_handles)
                )
            ),
        )

    def queryAttributeOwnership(  # noqa: N802
        self, object_handle: ObjectInstanceRef, attribute_handle: AttributeRef
    ) -> tuple[int, bool]:
        return cast(
            "tuple[int, bool]",
            self._run(
                self._fed().ownership.query_attribute_ownership(object_handle, attribute_handle)
            ),
        )

    def isAttributeOwnedByFederate(  # noqa: N802
        self, object_handle: ObjectInstanceRef, attribute_handle: AttributeRef
    ) -> bool:
        return bool(
            self._run(
                self._fed().ownership.is_attribute_owned_by_federate(
                    object_handle, attribute_handle
                )
            )
        )

    # --- §4.8-4.15 Federation save/restore (M25 Phase C) ---

    def requestFederationSave(  # noqa: N802
        self, label: str, save_time: float | None = None
    ) -> None:
        self._run(self._fed().savepoint.request_federation_save(label, save_time=save_time))

    def federateSaveBegun(self) -> None:  # noqa: N802
        self._run(self._fed().savepoint.federate_save_begun())

    def federateSaveComplete(self) -> None:  # noqa: N802
        self._run(self._fed().savepoint.federate_save_complete())

    def federateSaveNotComplete(self) -> None:  # noqa: N802
        self._run(self._fed().savepoint.federate_save_not_complete())

    def queryFederationSaveStatus(self, label: str = "") -> SaveStatusResponse:  # noqa: N802
        async def query() -> SaveStatusResponse:
            response = await self._fed().savepoint.query_federation_save_status(label)
            self._dispatch_event(response)
            return cast("SaveStatusResponse", response)

        return cast("SaveStatusResponse", self._run(query()))

    def requestFederationRestore(  # noqa: N802
        self, label: str, *, saved_federation_generation: int | None = None,
    ) -> None:
        if saved_federation_generation is None:
            self._run(self._fed().savepoint.request_federation_restore(label))
        else:
            self._run(self._fed().savepoint.request_federation_restore(
                label, saved_federation_generation=saved_federation_generation))

    def federateRestoreComplete(self) -> None:  # noqa: N802
        self._run(self._fed().savepoint.federate_restore_complete())

    def federateRestoreNotComplete(self) -> None:  # noqa: N802
        self._run(self._fed().savepoint.federate_restore_not_complete())

    def queryFederationRestoreStatus(self, label: str = "") -> RestoreStatusResponse:  # noqa: N802
        async def query() -> RestoreStatusResponse:
            response = await self._fed().savepoint.query_federation_restore_status(label)
            self._dispatch_event(response)
            return cast("RestoreStatusResponse", response)

        return cast("RestoreStatusResponse", self._run(query()))

    # --- §9 Data Distribution Management (M25 Phase C) ---

    def createRegion(  # noqa: N802
        self, routing_space_handle: int, dimension_handles: DimensionRefList
    ) -> RegionHandle:
        return RegionHandle(
            self._run(self._fed().ddm.create_region(routing_space_handle, list(dimension_handles)))
        )

    def setRangeBounds(  # noqa: N802
        self,
        region_handle: RegionRef,
        dimension_handle: DimensionRef,
        lower_bound: int,
        upper_bound: int,
    ) -> None:
        self._run(
            self._fed().ddm.set_range_bounds(
                region_handle, dimension_handle, lower=lower_bound, upper=upper_bound
            )
        )

    def commitRegionModifications(self, region_handles: RegionRefList) -> None:  # noqa: N802
        self._run(self._fed().ddm.commit_region_modifications(list(region_handles)))

    def deleteRegion(self, region_handle: RegionRef) -> None:  # noqa: N802
        self._run(self._fed().ddm.delete_region(region_handle))

    def subscribeObjectClassAttributesWithRegions(  # noqa: N802
        self,
        object_class_handle: ObjectClassRef,
        attribute_handles: AttributeRefList,
        region_handles: RegionRefList,
        active: bool = True,
        updateRateDesignator: str = "",  # noqa: N803
    ) -> None:
        self._run(
            self._fed().ddm.subscribe_object_class_attributes_with_regions(
                object_class_handle,
                list(attribute_handles),
                list(region_handles),
                active=active,
                update_rate_designator=updateRateDesignator,
            )
        )

    def subscribeInteractionClassWithRegions(  # noqa: N802
        self,
        interaction_class_handle: InteractionClassRef,
        region_handles: RegionRefList,
        active: bool = True,
    ) -> None:
        self._run(
            self._fed().ddm.subscribe_interaction_class_with_regions(
                interaction_class_handle,
                list(region_handles),
                active=active,
            )
        )

    def registerObjectInstanceWithRegions(  # noqa: N802
        self,
        object_class_handle: ObjectClassRef,
        attributes_and_regions: dict[int | str | AttributeHandle, list[int | RegionHandle]],
        instance_name: str = "",
    ) -> ObjectInstanceHandle:
        from rti1516e.ddm import AttributeRegions

        bindings = [
            AttributeRegions(attribute_handle=int(a), region_handles=[int(r) for r in regs])
            for a, regs in attributes_and_regions.items()
        ]
        return ObjectInstanceHandle(
            self._run(
                self._fed().ddm.register_object_instance_with_regions(
                    object_class_handle, bindings, object_name=instance_name
                )
            )
        )

    def associateRegionsForUpdates(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        attributes_and_regions: dict[int | str | AttributeHandle, list[int | RegionHandle]],
    ) -> None:
        from rti1516e.ddm import AttributeRegions

        bindings = [
            AttributeRegions(attribute_handle=int(a), region_handles=[int(r) for r in regs])
            for a, regs in attributes_and_regions.items()
        ]
        self._run(self._fed().ddm.associate_regions_for_updates(object_handle, bindings))

    def unassociateRegionsForUpdates(  # noqa: N802
        self,
        object_handle: ObjectInstanceRef,
        attributes_and_regions: (
            dict[int | str | AttributeHandle, list[int | RegionHandle]] | None
        ) = None,
    ) -> None:
        from rti1516e.ddm import AttributeRegions

        bindings: list[AttributeRegions] | None
        if attributes_and_regions is None:
            bindings = None
        else:
            bindings = [
                AttributeRegions(attribute_handle=int(a), region_handles=[int(r) for r in regs])
                for a, regs in attributes_and_regions.items()
            ]
        self._run(self._fed().ddm.unassociate_regions_for_updates(object_handle, bindings))

    def unsubscribeObjectClassAttributesWithRegions(  # noqa: N802
        self,
        object_class_handle: ObjectClassRef,
        attribute_handles: AttributeRefList,
        region_handles: RegionRefList,
    ) -> None:
        self._run(
            self._fed().ddm.unsubscribe_object_class_attributes_with_regions(
                object_class_handle, list(attribute_handles), list(region_handles)
            )
        )

    def unsubscribeInteractionClassWithRegions(  # noqa: N802
        self, interaction_class_handle: InteractionClassRef, region_handles: RegionRefList
    ) -> None:
        self._run(
            self._fed().ddm.unsubscribe_interaction_class_with_regions(
                interaction_class_handle, list(region_handles)
            )
        )

    def sendInteractionWithRegions(  # noqa: N802
        self,
        interaction_class_handle: InteractionClassRef,
        parameters: dict[int | str | ParameterHandle, bytes] | ParameterHandleValueMap,
        region_handles: RegionRefList,
        timestamp: float | None = None,
    ) -> None:
        self._run(
            self._fed().ddm.send_interaction_with_regions(
                interaction_class_handle,
                dict(parameters),
                list(region_handles),
                timestamp=timestamp,
            )
        )

    def requestAttributeValueUpdateWithRegions(  # noqa: N802
        self,
        object_class_handle: ObjectClassRef,
        attribute_handles: AttributeRefList,
        region_handles: RegionRefList,
        tag: bytes = b"",
    ) -> None:
        self._run(
            self._fed().ddm.request_attribute_value_update_with_regions(
                object_class_handle, list(attribute_handles), list(region_handles), tag=tag
            )
        )

    # --- §11 Management Object Model (M27 Phase D) ---
    # gorti exposes the MOM tracking surface via a query API rather than
    # via the MIM interaction set. The three methods below delegate to
    # fed.mom.* and are typed as Any to avoid pulling the dataclass
    # definitions into the standard.py top of file (the caller imports
    # FederationAttributes / FederateAttributes / MomInstance from
    # rti1516e.mom if they want to pattern-match).

    def queryFederationAttributes(self) -> Any:  # noqa: N802
        """§11 — return the HLAfederation MOM object snapshot.

        Returns :class:`rti1516e.mom.FederationAttributes`.
        """
        return self._run(self._fed().mom.query_federation_attributes())

    def queryFederateAttributes(self, federate_handle: FederateRef) -> Any:  # noqa: N802
        """§11 — return the HLAfederate MOM object snapshot for one federate.

        Returns :class:`rti1516e.mom.FederateAttributes`. The
        ``.found`` flag distinguishes "tracked + populated" from
        "no record (resigned or never joined)".
        """
        return self._run(self._fed().mom.query_federate_attributes(federate_handle))

    def enumerateMomInstances(self) -> Any:  # noqa: N802
        """§11 — list every active MOM instance in the federation.

        Returns ``list[rti1516e.mom.MomInstance]`` covering the
        HLAfederation singleton and one HLAfederate per joined
        federate.
        """
        return self._run(self._fed().mom.enumerate_mom_instances())

    # --- §6.1-6.5 Object instance name reservation (M26 Phase F) ---

    def reserveObjectInstanceName(self, object_name: str) -> None:  # noqa: N802
        """§6.1 — request a name reservation. Result delivered as
        objectInstanceNameReservationSucceeded / Failed callback."""
        self._run(self._fed().reservation.reserve(object_name))

    def releaseObjectInstanceName(self, object_name: str) -> None:  # noqa: N802
        """§6.4 — release a name reservation held by this federate."""
        self._run(self._fed().reservation.release(object_name))

    def reserveMultipleObjectInstanceNames(  # noqa: N802
        self, object_names: list[str]
    ) -> None:
        """§6.5 — atomic batch reservation. Result delivered as
        multipleObjectInstanceNameReservation{Succeeded,Failed}."""
        self._run(self._fed().reservation.reserve_multiple(object_names))

    # --- Callbacks: subclass overrides these ---

    def discoverObjectInstance(  # noqa: N802
        self,
        object_handle: int,
        class_name: str,
        instance_name: str,
        object_class: ObjectClassHandle | None = None,
    ) -> None:
        """Override to handle DiscoverObjectInstance.

        M39 typed-handle parity (§6.9): ``object_class`` is the typed
        :class:`ObjectClassHandle`. Overrides declared with the legacy
        3-argument signature keep working — the dispatcher only passes
        ``object_class`` to overrides that accept it. ``class_name``
        (stringified handle on the gRPC path) is DEPRECATED as the
        class identity; compare against ``object_class`` instead.
        """

    def reflectAttributeValues(  # noqa: N802
        self,
        object_handle: int,
        values: dict[str, Any],
        timestamp: float | None,
        attribute_values: dict[AttributeHandle, bytes] | None = None,
        tag: bytes = b"",
        metadata: ObjectCallbackMetadata | None = None,
    ) -> None:
        """Override to handle ReflectAttributeValues.

        M39 typed-handle parity (§6.11): ``attribute_values`` keys the
        payloads by typed :class:`AttributeHandle`. Legacy 3-argument
        overrides keep working (the dispatcher only passes it to
        overrides that accept it); the string-keyed ``values`` map is
        DEPRECATED for handle identity.
        """

    def receiveInteraction(  # noqa: N802
        self,
        class_name: str,
        parameters: dict[str, Any],
        timestamp: float | None,
        tag: bytes = b"",
        metadata: ObjectCallbackMetadata | None = None,
    ) -> None:
        """Override to handle ReceiveInteraction."""

    def turnUpdatesOnForObjectInstance(  # noqa: N802
        self,
        object_handle: int,
        attribute_handles: list[int],
        update_rate_designator: str | None = None,
    ) -> None:
        """Override to handle publisher update demand and its optional rate."""

    def turnUpdatesOffForObjectInstance(  # noqa: N802
        self,
        object_handle: int,
        attribute_handles: Collection[int],
    ) -> None:
        """Override when no active subscriber demands these attributes."""

    def timeAdvanceGrant(self, time: float) -> None:  # noqa: N802
        """Override to handle TimeAdvanceGrant."""

    def timeRegulationEnabled(self, time: float) -> None:  # noqa: N802
        """Override when regulation becomes active at the supplied logical time."""

    def timeConstrainedEnabled(self, time: float) -> None:  # noqa: N802
        """Override when constraint becomes active at the supplied logical time."""

    def federationHalted(self, cause: str, stalled_federate_handle: int) -> None:  # noqa: N802
        """Override to handle FederationHalted."""

    # --- Callback evocation ---

    def evokeCallback(  # noqa: N802
        self, approx_min_time: float = 0.0, approx_max_time: float | None = None
    ) -> bool:
        """Deliver at most one evoked callback on this calling thread.

        In immediate mode this observes completed callbacks for compatibility.
        The boolean retains this SDK's did-dispatch convention; full normative
        return-value/timing equivalence is not established by the API headers.
        """
        self._callback_dispatcher.guard_reentrant()
        if getattr(self, "_callback_error", None) is not None:
            raise RuntimeError("callback stream failed") from self._callback_error
        return self._callback_dispatcher.evoke(approx_min_time, approx_max_time, multiple=False)

    def evokeMultipleCallbacks(  # noqa: N802
        self, approx_min_time: float = 0.0, approx_max_time: float | None = None
    ) -> bool:
        """Deliver a bounded batch in evoked mode; preserve FIFO order."""
        self._callback_dispatcher.guard_reentrant()
        if getattr(self, "_callback_error", None) is not None:
            raise RuntimeError("callback stream failed") from self._callback_error
        return self._callback_dispatcher.evoke(approx_min_time, approx_max_time, multiple=True)

    def enableCallbacks(self) -> None:  # noqa: N802
        """Enable delivery; evoked callbacks still require an evoke call."""
        guard = getattr(self._federate, "_guard_service", None)
        if callable(guard):
            guard()
        if getattr(self, "_callback_error", None) is not None:
            raise RuntimeError("callback stream failed") from self._callback_error
        self._callbacks_enabled = True
        self._callback_dispatcher.enable(wait=threading.current_thread() is not self._loop_thread)

    def disableCallbacks(self) -> None:  # noqa: N802
        """Suspend delivery without discarding or counting queued callbacks."""
        self._callbacks_enabled = False
        self._callback_dispatcher.disable()

    # --- M25 Phase D — additional FederateAmbassador callbacks ---
    # Each is a no-op by default; federates ported from reference_rti
    # override the ones they care about. The Layer-1 event types
    # were already wired through pysdk/rti1516e/events.py.

    def removeObjectInstance(  # noqa: N802
        self, object_handle: int, tag: bytes, timestamp: float | None
    ) -> None:
        """§6.16 — an instance was deleted by its owner."""

    def provideAttributeValueUpdate(  # noqa: N802
        self, object_handle: int, attribute_handles: tuple[int, ...], tag: bytes
    ) -> None:
        """§6.26 — peer requested fresh values; owner should respond."""

    def synchronizationPointRegistrationSucceeded(self, label: str) -> None:  # noqa: N802
        """§4.12 — this federate's sync-point registration was accepted.

        M39: fires from the wire's SyncRegistrationSucceeded event
        (stream.proto tag 22, registrant only).
        """

    def synchronizationPointRegistrationFailed(  # noqa: N802
        self, label: str, reason: SynchronizationPointFailureReason | None
    ) -> None:
        """§4.12 — this federate's sync-point registration was rejected.

        M39: fires from the wire's SyncRegistrationFailed event
        (stream.proto tag 23, registrant only).
        """

    def announceSynchronizationPoint(self, label: str, tag: bytes) -> None:  # noqa: N802
        """§4.6 — a sync point was announced to this federate.

        reference_rti's announceSynchronizationPoint matches our internal
        SynchronizationPointAnnounced event.
        """

    def federationSynchronized(  # noqa: N802
        self, label: str, failed_to_sync: tuple[int, ...] = ()
    ) -> None:
        """§4.15 — all required federates have achieved the sync point.

        ``failed_to_sync`` lists federates that achieved with
        ``successfully=False`` (empty when everyone succeeded). Legacy
        1-argument overrides keep working — the dispatcher only passes
        the set to overrides that accept it.
        """

    def requestAttributeOwnershipAssumption(  # noqa: N802
        self,
        object_handle: int,
        attribute_handles: tuple[int, ...],
        divesting_federate: int,
        tag: bytes,
    ) -> None:
        """§7.3 — current owner offered ownership; this federate may acquire."""

    def attributeOwnershipAcquisitionNotification(  # noqa: N802
        self,
        object_handle: int,
        attribute_handles: tuple[int, ...],
        owning_federate: int,
    ) -> None:
        """§7.4 — this federate's acquisition succeeded; it now owns the attrs."""

    def attributeOwnershipAcquisitionNotificationWithTag(  # noqa: N802
        self,
        object_handle: int,
        attribute_handles: tuple[int, ...],
        owning_federate: int,
        tag: bytes,
    ) -> None:
        """Tagged callback; the default preserves legacy callback overrides."""
        self.attributeOwnershipAcquisitionNotification(
            object_handle, attribute_handles, owning_federate
        )

    def requestDivestitureConfirmation(  # noqa: N802
        self, object_handle: int, attribute_handles: tuple[int, ...]
    ) -> None:
        """§7.3 (divester half) — pending divest was matched and transferred."""

    def confirmAttributeOwnershipAcquisitionCancellation(  # noqa: N802
        self, object_handle: int, attribute_handles: tuple[int, ...]
    ) -> None:
        """Cancellation callback; this facade is also its default callback target."""

    def initiateFederateSave(  # noqa: N802
        self, label: str, save_time: float | None
    ) -> None:
        """§4.8 — federation save has started; federate must save state."""

    def federationSaved(self, label: str) -> None:  # noqa: N802
        """§4.9 — federation save completed successfully."""

    def federationNotSaved(self, label: str) -> None:  # noqa: N802
        """§4.9 — federation save was aborted; bundle was NOT written."""

    def federationSaveStatusResponse(  # noqa: N802
        self, statuses: tuple[FederateSaveStatus, ...]
    ) -> None:
        """Receive the per-federate vector returned by a save-status query."""

    def federationRestoreStatusResponse(  # noqa: N802
        self, statuses: tuple[FederateRestoreStatus, ...]
    ) -> None:
        """Receive the pre/post-handle vector returned by a restore-status query."""

    # --- M26 Phase F — object instance name reservation callbacks ---

    def objectInstanceNameReservationSucceeded(self, object_name: str) -> None:  # noqa: N802
        """§6.1 — a previously requested name reservation was accepted."""

    def objectInstanceNameReservationFailed(self, object_name: str) -> None:  # noqa: N802
        """§6.1 — a previously requested name reservation was rejected."""

    def multipleObjectInstanceNameReservationSucceeded(  # noqa: N802
        self, object_names: tuple[str, ...]
    ) -> None:
        """§6.5 — an atomic batch reservation was accepted."""

    def multipleObjectInstanceNameReservationFailed(  # noqa: N802
        self, requested_names: tuple[str, ...], colliding_names: tuple[str, ...]
    ) -> None:
        """§6.5 — an atomic batch reservation was rejected (NONE reserved)."""

    # --- Wire-parity callbacks ---
    # Each is a no-op by default; override the ones you care about.

    def requestAttributeOwnershipRelease(  # noqa: N802
        self, object_handle: int, attribute_handles: tuple[int, ...], tag: bytes
    ) -> None:
        """§7.11 — another federate wants attributes this federate owns."""

    def attributeOwnershipUnavailable(  # noqa: N802
        self, object_handle: int, attribute_handles: tuple[int, ...]
    ) -> None:
        """§7.10 — acquisition-if-available found the attributes owned."""

    def initiateFederateRestore(  # noqa: N802
        self, label: str, federate_handle: int, federate_name: str
    ) -> None:
        """§4.26 — a federation restore began; load state and report back."""

    def federationRestored(self, label: str) -> None:  # noqa: N802
        """§4.14 — the federation restore completed successfully."""

    def federationNotRestored(self, label: str) -> None:  # noqa: N802
        """§4.14 — the federation restore aborted."""

    def requestFederationRestoreSucceeded(self, label: str) -> None:  # noqa: N802
        """§4.25 — this federate's restore request was accepted."""

    def requestFederationRestoreFailed(self, label: str, reason: str) -> None:  # noqa: N802
        """§4.25 — this federate's restore request was rejected."""

    def federationRestoreBegun(self) -> None:  # noqa: N802
        """§4.26 — the restore left idle (precedes initiateFederateRestore)."""

    def startRegistrationForObjectClass(  # noqa: N802
        self, object_class_handle: ObjectClassHandle
    ) -> None:
        """§5.10 — the object class gained its first subscriber."""

    def stopRegistrationForObjectClass(  # noqa: N802
        self, object_class_handle: ObjectClassHandle
    ) -> None:
        """§5.11 — the object class lost its last subscriber."""

    def turnInteractionsOn(  # noqa: N802
        self, interaction_class_handle: InteractionClassHandle
    ) -> None:
        """§5.12 — the interaction class gained its first subscriber."""

    def turnInteractionsOff(  # noqa: N802
        self, interaction_class_handle: InteractionClassHandle
    ) -> None:
        """§5.13 — the interaction class lost its last subscriber."""

    def attributesInScope(  # noqa: N802
        self, object_handle: int, attribute_handles: tuple[int, ...]
    ) -> None:
        """§6.17 — DDM region overlap brought the attributes into scope."""

    def attributesOutOfScope(  # noqa: N802
        self, object_handle: int, attribute_handles: tuple[int, ...]
    ) -> None:
        """§6.18 — the attributes dropped out of region-overlap scope."""

    def requestRetraction(  # noqa: N802
        self,
        retraction_handle: MessageRetractionHandle,
        sender_federate: FederateHandle,
    ) -> None:
        """§8.22 — a sender retracted a TSO message this federate saw."""

    # --- Internals ---

    def _start_loop(self) -> None:
        if self._loop is not None:
            return
        loop_ready = threading.Event()

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            loop_ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        thread = threading.Thread(target=_runner, name="rti-ambassador-loop", daemon=True)
        thread.start()
        self._loop_thread = thread
        loop_ready.wait()

    def _stop_loop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
        self._loop = None
        self._loop_thread = None

    def _loop_required(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("ambassador event loop is not running — call connect() first")
        return self._loop

    def _run(self, coro: Any) -> Any:
        """Schedule ``coro`` on the background loop and block until done."""
        if threading.current_thread() is self._loop_thread:
            if inspect.iscoroutine(coro):
                coro.close()
            raise RuntimeError("a synchronous SDK call cannot block its asyncio I/O loop")
        loop = self._loop_required()
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def _run_after_async_barrier(self, factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
        with self._async_submission_gate:
            self._flush_async_operations_locked()
            return self._run(factory())

    def _submit_async_operation(
        self, factory: Callable[[], Coroutine[Any, Any, Any]]
    ) -> Future[Any]:
        """Submit one extension operation with bounded caller backpressure."""
        with self._async_submission_gate:
            if self._async_ordering_barrier_pending:
                self._flush_async_operations_locked()
            loop = self._loop_required()
            with self._async_operations_changed:
                while self._async_inflight_count >= self._async_operation_limit:
                    if self._async_closing:
                        raise RuntimeError("ambassador is disconnecting")
                    if threading.current_thread() is self._loop_thread:
                        raise RuntimeError("async backpressure cannot block the ambassador loop")
                    self._async_operations_changed.wait()
                if self._async_closing:
                    raise RuntimeError("ambassador is disconnecting")
                coroutine = factory()
                try:
                    internal = asyncio.run_coroutine_threadsafe(coroutine, loop)
                except BaseException:
                    coroutine.close()
                    raise
                proxy: Future[Any] = Future()
                self._async_operations.append(internal)
                self._async_inflight_count += 1
            internal.add_done_callback(
                lambda completed: self._async_operation_finished(completed, proxy)
            )
            return proxy

    def _async_operation_finished(self, internal: Future[Any], proxy: Future[Any]) -> None:
        try:
            result = internal.result()
        except BaseException as exc:
            if not proxy.cancelled():
                with contextlib.suppress(InvalidStateError):
                    proxy.set_exception(exc)
        else:
            if not proxy.cancelled():
                with contextlib.suppress(InvalidStateError):
                    proxy.set_result(result)
        finally:
            with self._async_operations_changed:
                self._async_inflight_count -= 1
                self._async_operations_changed.notify_all()

    def _fed(self) -> Federate:
        if self._federate is None:
            raise FederateNotExecutionMember("not joined — call joinFederationExecution() first")
        return self._federate

    def _invoke_compat(self, method_name: str, *args: Any, **optional: Any) -> None:
        """Call ``target.<method_name>(*args)`` plus the ``optional``
        kwargs the override's signature accepts.

        M39: lets the dispatcher pass NEW callback arguments (typed
        handles on discover/reflect, §4.15 failed_to_sync) without
        breaking subclasses written against the older, shorter
        signatures. Acceptance is resolved per (class, method) once and
        cached.
        """
        target = self._callback_target
        method = getattr(target, method_name)
        accepted_names = self._accepted_kwargs(type(target), method_name, method)
        if accepted_names is None:  # **kwargs override — pass everything
            method(*args, **optional)
            return
        method(*args, **{k: v for k, v in optional.items() if k in accepted_names})

    def _accepted_kwargs(
        self, target_type: type, method_name: str, method: Any
    ) -> frozenset[str] | None:
        """Return the parameter names ``method`` accepts beyond its
        positionals, or None when it takes ``**kwargs``. Cached per
        (target class, method name)."""
        key = (target_type, method_name)
        if key in self._accepted_kwargs_cache:
            return self._accepted_kwargs_cache[key]
        try:
            params = inspect.signature(method).parameters
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                result: frozenset[str] | None = None
            else:
                result = frozenset(params)
        except (TypeError, ValueError):  # builtins / exotic callables
            result = frozenset()
        self._accepted_kwargs_cache[key] = result
        return result

    def _dispatch_event(self, event: Any) -> bool:
        """Receive an event without invoking user callbacks on the I/O loop."""
        from ._local_notifications import kind_of, prepare

        restored_locals = []
        transport = getattr(self._federate, "_transport", None)
        if (isinstance(event, FederationRestored) and event.federate_handle is not None
                and not getattr(event, "_callback_recovered", False)
                and not getattr(event, "_callback_outcome_only", False)):
            self._callback_dispatcher.clear()
            self._callback_buffer.clear()
            self._generic_callback_invocations = None
            restored_locals = getattr(transport, "_restored_local_notifications", {}).pop(
                event.federate_handle, [])
            with self._local_notification_lock:
                self._local_notification_order = max(
                    (local._local_notification_order for local in restored_locals), default=0)
            self._local_notification_state = None
        if kind_of(event) is not None and getattr(event, "_wire_carrier", None) is None:
            with self._local_notification_lock:
                order = self._local_notification_order + 1
                event = prepare(event, order, getattr(transport, "_restore_serial", 0))
                self._local_notification_order = max(order, event._local_notification_order)
        for local in restored_locals:
            object.__setattr__(local, "_restore_serial", getattr(transport, "_restore_serial", 0))
        return self._callback_dispatcher.submit_many(
            [event, *restored_locals], wait=threading.current_thread() is not self._loop_thread
        )

    def _checkpoint_pending_callbacks(self) -> list[Any]:
        from ._generic_callbacks import callback_state

        state = callback_state(self)
        state.checkpoint_outcomes()
        return self._callback_dispatcher.snapshot_pending()

    def _dispatch_callback(self, event: Any) -> bool:
        from ._generic_callbacks import callback_state, validate_event_scope

        validate_event_scope(self, event)
        if hasattr(event, "_local_notification_id"):
            from ._local_notifications import notification_state

            local = notification_state(self)
            if not local.begin(event):
                return False
            try:
                return self._invoke_event(event)
            except BaseException:
                self._callback_fired_count += 1
                raise
            finally:
                local.finish(event)
        if getattr(event, "_callback_outcome_only", False):
            generic = callback_state(self)
            outcome = generic.recover(event)
            if outcome is not None:
                try:
                    self._run(self._fed().report_callback_invocation(*outcome))
                except Exception:
                    self._callback_dispatcher.retain_current_admission()
                    raise
                generic.acknowledged(outcome[0])
            return False
        if self._federate is None or not hasattr(self._federate, "_ownership_callback_state"):
            return self._dispatch_callback_unfenced(event)
        state = self._federate._ownership_callback_state()
        with state.dispatch_fence:
            if not state.begin(event):
                self._run(self._federate.flush_ownership_callback_receipts())
                return False
            return self._dispatch_callback_unfenced(event)

    def _dispatch_callback_unfenced(self, event: Any) -> bool:
        from ._generic_callbacks import callback_state
        from .events import (
            ConfirmAttributeOwnershipAcquisitionCancellation,
            TransportationChangeConfirmed,
        )

        tracked_types = (
            ServiceCallback, TransportationChangeConfirmed, FederationHalted,
            DiscoverObjectInstance,
            ReflectAttributeValues, ReceiveInteraction,
            RemoveObjectInstance, ProvideAttributeValueUpdate, SynchronizationPointAnnounced,
            FederationSynchronized, SynchronizationPointRegistrationSucceeded,
            SynchronizationPointRegistrationFailed, RequestAttributeOwnershipAssumption,
            RequestAttributeOwnershipRelease, RequestDivestitureConfirmation,
            AttributeOwnershipAcquisitionNotification, AttributeOwnershipUnavailable,
            ConfirmAttributeOwnershipAcquisitionCancellation, TimeAdvanceGrant,
            TimeRegulationEnabled, TimeConstrainedEnabled, InitiateFederateSave,
            FederationSaved, FederationNotSaved, InitiateFederateRestore, FederationRestored,
            FederationNotRestored, RequestFederationRestoreSucceeded,
            RequestFederationRestoreFailed, FederationRestoreBegun,
            ObjectInstanceNameReservationSucceeded, ObjectInstanceNameReservationFailed,
            MultipleObjectInstanceNameReservationSucceeded,
            MultipleObjectInstanceNameReservationFailed,
            StartRegistrationForObjectClass, StopRegistrationForObjectClass,
            TurnInteractionsOn, TurnInteractionsOff, TurnUpdatesOnForObjectInstance,
            TurnUpdatesOffForObjectInstance, AttributesInScope, AttributesOutOfScope,
            RequestRetraction, SaveStatusResponse, RestoreStatusResponse,
        )
        receipt = (
            getattr(event, "_callback_receipt", b"") if isinstance(event, tracked_types) else b""
        )
        state = callback_state(self)
        if receipt and self._federate is not None and not state.claim(
                receipt, getattr(event, "_callback_invocation_identity", None)):
            self._flush_generic_callback_reports(state)
            return False
        admitted = event
        try:
            if isinstance(event, TransportationChangeConfirmed):
                admitted = self._run(self._fed()._activate_transportation_confirmation(event))
            if (receipt and self._federate is not None
                    and getattr(self._federate, "callback_invocation_entry_supported", False)):
                self._run(self._federate.report_callback_invocation(
                    receipt, True, "", invocation_entry=True))
        except BaseException as exc:
            if receipt:
                state.entry_failed(receipt)
                if (self._federate is not None
                        and hasattr(self._federate, "_ownership_callback_state")):
                    self._federate._ownership_callback_state().entry_failed(event)
            if (isinstance(event, (SynchronizationPointRegistrationSucceeded,
                                   SynchronizationPointRegistrationFailed))
                    and isinstance(exc, (SaveInProgress, RestoreInProgress))
                    and self._callback_dispatcher.defer_current_checkpoint_admission()):
                return False
            if isinstance(event, TransportationChangeConfirmed) and isinstance(exc, Exception):
                self._callback_dispatcher.retain_current_admission()
            raise
        recognized = False
        failure: BaseException | None = None
        try:
            recognized = self._invoke_event(admitted)
            return recognized  # noqa: RET504 - finally needs the invocation result.
        except BaseException as exc:
            recognized = True
            failure = exc
            self._callback_fired_count += 1
            raise
        finally:
            if (recognized and self._federate is not None
                    and hasattr(self._federate, "flush_ownership_callback_receipts")):
                try:
                    self._run(self._federate.flush_ownership_callback_receipts())
                except Exception as exc:
                    self._callback_report_failures.append(exc)
            if recognized and receipt and self._federate is not None:
                state.complete(receipt, failure is None,
                               "" if failure is None else f"{type(failure).__name__}: {failure}")
                self._flush_generic_callback_reports(state)

    def _flush_generic_callback_reports(self, state: Any) -> None:
        for receipt, success, exception in state.pending():
            try:
                self._run(self._fed().report_callback_invocation(receipt, success, exception))
                state.acknowledged(receipt)
            except Exception as exc:
                self._callback_report_failures.append(exc)

    def _invoke_event(self, event: Any) -> bool:
        from .events import (
            ConfirmAttributeOwnershipAcquisitionCancellation,
            TransportationChangeConfirmed,
        )

        if isinstance(event, BaseException):
            self._callback_error = event
            callback = getattr(self._callback_target, "connectionLost", None)
            if callable(callback):
                callback(str(event))
            self._callback_fired_count += 1
            return True
        target = self._callback_target
        if isinstance(event, TransportationChangeConfirmed):
            kind = int(event.transportation_type)
            if event.object_handle:
                target.confirmAttributeTransportationTypeChange(
                    ObjectInstanceHandle(event.object_handle),
                    {AttributeHandle(value) for value in event.attribute_handles}, kind)
            else:
                target.confirmInteractionTransportationTypeChange(
                    InteractionClassHandle(event.interaction_class_handle), kind)
        elif isinstance(event, ServiceCallback):
            getattr(target, event.method)(*event.args)
        elif isinstance(event, DiscoverObjectInstance):
            # M39 §6.9 — the typed object_class rides as an optional
            # kwarg so legacy 3-argument overrides keep working.
            self._invoke_compat(
                "discoverObjectInstance",
                event.object_handle,
                event.class_name,
                event.instance_name,
                object_class=event.object_class,
            )
        elif isinstance(event, ReflectAttributeValues):
            # M39 §6.11 — typed attribute_values as an optional kwarg.
            self._invoke_compat(
                "reflectAttributeValues",
                event.object_handle,
                event.values,
                event.timestamp,
                attribute_values=event.attribute_values,
                tag=event.tag,
                metadata=event.metadata,
            )
        elif isinstance(event, ReceiveInteraction):
            self._invoke_compat(
                "receiveInteraction",
                event.class_name,
                event.parameters,
                event.timestamp,
                tag=event.tag,
                metadata=event.metadata,
            )
        elif isinstance(event, TimeAdvanceGrant):
            target.timeAdvanceGrant(event.time)
        elif isinstance(event, TimeRegulationEnabled):
            target.timeRegulationEnabled(event.time)
        elif isinstance(event, TimeConstrainedEnabled):
            target.timeConstrainedEnabled(event.time)
        elif isinstance(event, FederationHalted):
            target.federationHalted(event.cause, event.stalled_federate_handle)
        elif isinstance(event, RemoveObjectInstance):
            self._invoke_compat(
                "removeObjectInstance",
                event.object_handle,
                event.tag,
                event.timestamp,
                metadata=event.metadata,
            )
        elif isinstance(event, ProvideAttributeValueUpdate):
            target.provideAttributeValueUpdate(
                event.object_handle, event.attribute_handles, event.tag
            )
        elif isinstance(event, SynchronizationPointAnnounced):
            target.announceSynchronizationPoint(event.label, event.tag)
        elif isinstance(event, FederationSynchronized):
            # M39 §4.15 — failed_to_sync as an optional kwarg.
            self._invoke_compat(
                "federationSynchronized",
                event.label,
                failed_to_sync=event.failed_to_sync,
            )
        elif isinstance(event, SynchronizationPointRegistrationSucceeded):
            target.synchronizationPointRegistrationSucceeded(event.label)
        elif isinstance(event, SynchronizationPointRegistrationFailed):
            target.synchronizationPointRegistrationFailed(event.label, event.reason)
        elif isinstance(event, RequestAttributeOwnershipAssumption):
            target.requestAttributeOwnershipAssumption(
                event.object_handle,
                event.attribute_handles,
                event.divesting_federate,
                event.tag,
            )
        elif isinstance(event, AttributeOwnershipAcquisitionNotification):
            target.attributeOwnershipAcquisitionNotificationWithTag(
                event.object_handle,
                event.attribute_handles,
                event.owning_federate,
                event.tag,
            )
        elif isinstance(event, RequestDivestitureConfirmation):
            target.requestDivestitureConfirmation(event.object_handle, event.attribute_handles)
        elif isinstance(event, ConfirmAttributeOwnershipAcquisitionCancellation):
            target.confirmAttributeOwnershipAcquisitionCancellation(
                event.object_handle, event.attribute_handles
            )
        elif isinstance(event, InitiateFederateSave):
            target.initiateFederateSave(event.label, event.save_time)
        elif isinstance(event, FederationSaved):
            target.federationSaved(event.label)
        elif isinstance(event, FederationNotSaved):
            target.federationNotSaved(event.label)
        elif isinstance(event, SaveStatusResponse):
            target.federationSaveStatusResponse(event.federate_statuses)
        elif isinstance(event, RestoreStatusResponse):
            target.federationRestoreStatusResponse(event.federate_statuses)
        elif isinstance(event, ObjectInstanceNameReservationSucceeded):
            target.objectInstanceNameReservationSucceeded(event.object_name)
        elif isinstance(event, ObjectInstanceNameReservationFailed):
            target.objectInstanceNameReservationFailed(event.object_name)
        elif isinstance(event, MultipleObjectInstanceNameReservationSucceeded):
            target.multipleObjectInstanceNameReservationSucceeded(event.object_names)
        elif isinstance(event, MultipleObjectInstanceNameReservationFailed):
            target.multipleObjectInstanceNameReservationFailed(
                event.requested_names, event.colliding_names
            )
        # Dispatch every supported wire event to its ambassador callback.
        elif isinstance(event, RequestAttributeOwnershipRelease):
            target.requestAttributeOwnershipRelease(
                event.object_handle, event.attribute_handles, event.tag
            )
        elif isinstance(event, AttributeOwnershipUnavailable):
            target.attributeOwnershipUnavailable(event.object_handle, event.attribute_handles)
        elif isinstance(event, InitiateFederateRestore):
            target.initiateFederateRestore(event.label, event.federate_handle, event.federate_name)
        elif isinstance(event, FederationRestored):
            target.federationRestored(event.label)
        elif isinstance(event, FederationNotRestored):
            target.federationNotRestored(event.label)
        elif isinstance(event, RequestFederationRestoreSucceeded):
            target.requestFederationRestoreSucceeded(event.label)
        elif isinstance(event, RequestFederationRestoreFailed):
            target.requestFederationRestoreFailed(event.label, event.reason)
        elif isinstance(event, FederationRestoreBegun):
            target.federationRestoreBegun()
        elif isinstance(event, StartRegistrationForObjectClass):
            target.startRegistrationForObjectClass(ObjectClassHandle(event.object_class_handle))
        elif isinstance(event, StopRegistrationForObjectClass):
            target.stopRegistrationForObjectClass(ObjectClassHandle(event.object_class_handle))
        elif isinstance(event, TurnInteractionsOn):
            target.turnInteractionsOn(InteractionClassHandle(event.interaction_class_handle))
        elif isinstance(event, TurnInteractionsOff):
            target.turnInteractionsOff(InteractionClassHandle(event.interaction_class_handle))
        elif isinstance(event, TurnUpdatesOnForObjectInstance):
            self._invoke_compat(
                "turnUpdatesOnForObjectInstance",
                event.object_handle,
                event.attribute_handles,
                update_rate_designator=event.update_rate_designator,
            )
        elif isinstance(event, TurnUpdatesOffForObjectInstance):
            target.turnUpdatesOffForObjectInstance(event.object_handle, event.attribute_handles)
        elif isinstance(event, AttributesInScope):
            target.attributesInScope(event.object_handle, event.attribute_handles)
        elif isinstance(event, AttributesOutOfScope):
            target.attributesOutOfScope(event.object_handle, event.attribute_handles)
        elif isinstance(event, RequestRetraction):
            target.requestRetraction(
                MessageRetractionHandle(event.retraction_handle),
                FederateHandle(event.sender_federate),
            )
        else:
            return False
        self._callback_fired_count += 1
        return True

    async def _pump_events(self) -> None:
        """Drain Federate.events() and dispatch to the appropriate callback."""
        federate = self._federate
        if federate is None:
            return
        try:
            async for event in federate.events():
                self._dispatch_event(event)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._dispatch_event(exc)
