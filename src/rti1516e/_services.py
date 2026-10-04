"""Additional admitted Layer-1 services over the existing generated wire."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._grpc_errors import translate_rpc_error
from ._logical_time import FLOAT64_TIME, logical_time_name


@dataclass(frozen=True)
class FederationExecutionInformation:
    """The fields available in gorti's current federation-list response."""

    name: str
    mode: int
    federates_joined: int
    federation_generation: int
    logical_time_implementation_name: str = FLOAT64_TIME


async def list_federation_executions(connection: Any) -> tuple[FederationExecutionInformation, ...]:
    """List executions without requiring this connection to be joined."""
    from rti.v1 import federation_pb2, federation_pb2_grpc

    async with connection._admit_work() as transport:
        stub = federation_pb2_grpc.FederationServiceStub(transport.channel)
        try:
            response = await stub.ListFederations(
                federation_pb2.ListFederationsRequest(wire_version=1))
        except Exception as exc:
            translate_rpc_error(exc)
            raise
        return tuple(FederationExecutionInformation(
            row.name, int(row.mode), int(row.federates_joined), int(row.federation_generation),
            logical_time_name(getattr(row, "logical_time_implementation_name", "")),
        ) for row in response.federations)


class FederateServices:
    _connection: Any
    _transport: Any
    _session_token: bytes
    handle: int

    async def _activate_transportation_confirmation(self, event: Any) -> Any:
        from ._transportation_callbacks import activate

        return await activate(self, event)

    def _ownership_callback_state(self) -> Any:
        from ._ownership_callbacks import OwnershipCallbacks

        state = getattr(self, "_ownership_callbacks", None)
        if state is None:
            state = self._ownership_callbacks = OwnershipCallbacks()
        if hasattr(self, "handle"):
            states = getattr(self._transport, "_ownership_callbacks_by_federate", None)
            if states is None:
                states = self._transport._ownership_callbacks_by_federate = {}
            states[self.handle] = state
        return state

    async def _ownership_callback_rpc(self, method: str, receipt: Any = None) -> None:
        from rti.v1 import callback_pb2, callback_pb2_grpc

        transport = self._transport
        fields = {"wire_version": 1, "federation_name": self._require_federation_name(),
                  "federate_handle": self.handle}
        if receipt is None:
            generation = await transport._resolve_federation_generation(
                self._require_federation_name()
            )
            request = callback_pb2.OwnershipCallbackScopeRequest(
                **fields, federation_generation=generation,
                callback_epoch=transport._callback_epochs.get(self.handle, 0))
        else:
            request = callback_pb2.ReportOwnershipCallbackInvocationRequest(
                **fields, receipt=receipt
            )
        if len(self._session_token) != 32:
            raise RuntimeError("ownership receipts require an authenticated join session")
        stub = callback_pb2_grpc.OwnershipCallbackServiceStub(self._require_channel())
        try:
            await getattr(stub, method)(
                request, metadata=transport.session_metadata(self._session_token)
            )
        except Exception as exc:
            translate_rpc_error(exc)
            raise

    async def enable_ownership_callback_receipts(self) -> None:
        """Opt into actual terminal callback invocation receipts before ownership work."""
        await self._ownership_callback_rpc("EnableOwnershipCallbackReceipts")
        state = self._ownership_callback_state()
        state.scope = (self._transport._generation_by_federation[self._require_federation_name()],
                       self._transport._callback_epochs.get(self.handle, 0))
        state.enabled = True

    async def report_ownership_callback_invocation(self, event: Any) -> None:
        """Assert actual invocation when manually consuming the Layer-1 event stream."""
        if getattr(event, "_ownership_receipt", None) is None:
            raise ValueError("event has no ownership callback receipt")
        self._ownership_callback_state().begin(event)
        await self.flush_ownership_callback_receipts()

    async def retry_ownership_callbacks(self) -> None:
        await self.flush_ownership_callback_receipts()
        await self._ownership_callback_rpc("RetryOwnershipCallbacks")

    async def flush_ownership_callback_receipts(self) -> None:
        state = self._ownership_callback_state()
        for receipt in state.pending():
            await self._ownership_callback_rpc("ReportOwnershipCallbackInvocation", receipt)
            state.acknowledged(receipt)

    async def dispatch_event(self, event: Any, callback: Any) -> Any:
        """Invoke a Layer-1 handler, suppress stale advice, and acknowledge actual entry."""
        import inspect

        from ._generic_callbacks import callback_state, validate_event_scope

        validate_event_scope(self, event)
        if hasattr(event, "_local_notification_id"):
            from ._local_notifications import notification_state

            local = notification_state(self)
            if not local.begin(event):
                return None
            try:
                result = callback(event)
                return await result if inspect.isawaitable(result) else result
            finally:
                local.finish(event)
        generic = callback_state(self)
        if getattr(event, "_callback_outcome_only", False):
            outcome = generic.recover(event)
            if outcome is not None:
                await self.report_callback_invocation(*outcome)
                generic.acknowledged(outcome[0])
            return None
        state = self._ownership_callback_state()
        async with state.async_fence:
            if not state.begin(event):
                await self.flush_ownership_callback_receipts()
                return None
            from ._generic_callbacks import callback_state

            generic = callback_state(self)
            receipt = getattr(event, "_callback_receipt", b"")
            if receipt and not generic.claim(
                receipt, getattr(event, "_callback_invocation_identity", None)
            ):
                await self.flush_callback_invocation_reports()
                return None
            try:
                admitted = await self._activate_transportation_confirmation(event)
                if receipt and self.callback_invocation_entry_supported:
                    await self.report_callback_invocation(receipt, True, invocation_entry=True)
            except BaseException:
                if receipt:
                    generic.entry_failed(receipt)
                state.entry_failed(event)
                raise
            state.active_task = asyncio.current_task()
            failure: BaseException | None = None
            try:
                result = callback(admitted)
                return await result if inspect.isawaitable(result) else result
            except BaseException as exc:
                failure = exc
                raise
            finally:
                state.active_task = None
                if receipt:
                    detail = "" if failure is None else f"{type(failure).__name__}: {failure}"
                    generic.complete(receipt, failure is None, detail)
                    try:
                        await self.flush_callback_invocation_reports()
                    except BaseException as report_error:
                        if failure is None:
                            raise
                        if hasattr(failure, "add_note"):
                            failure.add_note(f"callback completion remains pending: {report_error}")
                try:
                    await self.flush_ownership_callback_receipts()
                except BaseException as receipt_error:
                    if failure is None:
                        raise
                    if hasattr(failure, "add_note"):
                        failure.add_note(f"ownership receipt remains pending: {receipt_error}")

    if TYPE_CHECKING:

        @property
        def callback_invocation_entry_supported(self) -> bool: ...

        @property
        def logical_time_implementation_name(self) -> str: ...

        def _require_channel(self) -> Any: ...
        def _require_federation_name(self) -> str: ...
        def _guard_service(self) -> None: ...

    async def _service_rpc(
        self,
        service: str,
        method: str,
        request_name: str,
        *,
        authenticated: bool = False,
        rpc_timeout: float | None = None,
        **fields: Any,
    ) -> Any:
        async with self._connection._admit_work():
            self._guard_service()
            if not (method == "ReportCallbackInvocation" and fields.get("invocation_entry")):
                await self.flush_ownership_callback_receipts()
            messages = import_module(f"rti.v1.{service}_pb2")
            stubs = import_module(f"rti.v1.{service}_pb2_grpc")
            stub = getattr(stubs, f"{service.title()}ServiceStub")(self._require_channel())
            for key, resolver in (
                ("object_class_handle", "_resolve_object_class_handle"),
                ("interaction_class_handle", "_resolve_interaction_class_handle"),
            ):
                if key in fields and isinstance(fields[key], str):
                    if key == "object_class_handle" and "attribute_handles" in fields:
                        fields["attribute_handles"] = self._transport._resolve_attribute_handles(
                            fields[key], fields["attribute_handles"]
                        )
                    fields[key] = getattr(self._transport, resolver)(fields[key])
            request_type = getattr(messages, request_name)
            identity: dict[str, Any] = {"wire_version": 1}
            if "federation_name" in request_type.DESCRIPTOR.fields_by_name:
                identity["federation_name"] = self._require_federation_name()
            if "federate_handle" in request_type.DESCRIPTOR.fields_by_name:
                identity["federate_handle"] = self.handle
            request = request_type(**identity, **fields)
            options: dict[str, Any] = {}
            if rpc_timeout is not None:
                options["timeout"] = rpc_timeout
            if authenticated:
                token = getattr(self, "_session_token", b"")
                if len(token) != 32:
                    raise RuntimeError("service requires an authenticated join session")
                options["metadata"] = self._transport.session_metadata(token)
            try:
                return await getattr(stub, method)(request, **options)
            except Exception as exc:
                translate_rpc_error(exc)
                raise

    async def unpublish_object_class(
        self,
        class_handle: int | str,
        *,
        attributes: list[int | str] | None = None,
    ) -> None:
        """Omit attributes for whole-class removal; a supplied empty set is a no-op."""
        fields: dict[str, Any] = {}
        if attributes is not None:
            fields["attribute_set_present"] = True
        await self._service_rpc(
            "declaration",
            "UnpublishObjectClassAttributes",
            "UnpubObjAttrsRequest",
            object_class_handle=class_handle,
            attribute_handles=list(attributes) if attributes is not None else [],
            **fields,
        )

    async def unsubscribe_object_class(
        self,
        class_handle: int | str,
        *,
        attributes: list[int | str] | None = None,
    ) -> None:
        """Omit attributes for whole-class removal; a supplied empty set is a no-op."""
        fields: dict[str, Any] = {}
        if attributes is not None:
            fields["attribute_set_present"] = True
        await self._service_rpc(
            "declaration",
            "UnsubscribeObjectClassAttributes",
            "UnsubObjAttrsRequest",
            object_class_handle=class_handle,
            attribute_handles=list(attributes) if attributes is not None else [],
            **fields,
        )

    async def unpublish_interaction_class(self, class_handle: int | str) -> None:
        await self._service_rpc(
            "declaration",
            "UnpublishInteractionClass",
            "UnpubInterRequest",
            interaction_class_handle=class_handle,
        )

    async def unsubscribe_interaction_class(self, class_handle: int | str) -> None:
        await self._service_rpc(
            "declaration",
            "UnsubscribeInteractionClass",
            "UnsubInterRequest",
            interaction_class_handle=class_handle,
        )

    async def query_galt(self) -> tuple[int | float, bool]:
        result = await self._service_rpc("time", "QueryGALT", "QueryFederateTimeRequest")
        from ._logical_time import read_time

        return (read_time(result, "galt", self.logical_time_implementation_name)
                if result.finite else 0), bool(result.finite)

    async def query_lits(self) -> tuple[int | float, bool]:
        result = await self._service_rpc("time", "QueryLITS", "QueryFederateTimeRequest")
        from ._logical_time import read_time

        return (read_time(result, "lits", self.logical_time_implementation_name)
                if result.finite else 0), bool(result.finite)

    async def set_automatic_resign_directive(self, action: int | str) -> None:
        from rti1516e._transport import resign_action_to_proto

        await self._service_rpc(
            "federation",
            "SetAutomaticResignDirective",
            "SetAutomaticResignDirectiveRequest",
            authenticated=True,
            action=resign_action_to_proto(action),
        )

    async def get_automatic_resign_directive(self) -> int:
        response = await self._service_rpc(
            "federation",
            "GetAutomaticResignDirective",
            "GetAutomaticResignDirectiveRequest",
            authenticated=True,
        )
        return int(response.action)

    async def get_federate_handle(self, name: str) -> int:
        from .errors import RtiError

        response = await self._service_rpc(
            "federation",
            "ListFederationMembers",
            "ListFederationMembersRequest",
        )
        for member in response.members:
            if member.federate_name == name:
                return int(member.federate_handle)
        raise RtiError(f"federate name {name!r} is not present in the joined federation")

    async def get_federate_name(self, handle: int) -> str:
        from .errors import RtiError

        response = await self._service_rpc(
            "federation",
            "ListFederationMembers",
            "ListFederationMembersRequest",
        )
        for member in response.members:
            if member.federate_handle == handle:
                return str(member.federate_name)
        raise RtiError(f"federate handle {handle} is not present in the joined federation")

    async def request_attribute_transportation_type_change(
        self,
        object_handle: int,
        attribute_handles: list[int],
        transport_type: int,
    ) -> None:
        await self._service_rpc(
            "object",
            "RequestAttributeTransportationTypeChange",
            "ChangeAttributeTransportRequest",
            object_handle=object_handle,
            attribute_handles=list(attribute_handles),
            transport_type=transport_type,
        )

    async def query_attribute_transportation_type(
        self,
        object_handle: int,
        attribute_handle: int,
    ) -> int:
        result = await self._service_rpc(
            "object",
            "QueryAttributeTransportationType",
            "QueryAttributeTransportationTypeRequest",
            object_handle=object_handle,
            attribute_handle=attribute_handle,
        )
        return int(result.transport_type)

    async def request_interaction_transportation_type_change(
        self,
        class_handle: int,
        transport_type: int,
    ) -> None:
        await self._service_rpc(
            "object",
            "RequestInteractionTransportationTypeChange",
            "ChangeInteractionTransportRequest",
            interaction_class_handle=class_handle,
            transport_type=transport_type,
        )

    async def query_interaction_transportation_type(
        self,
        target_federate_handle: int,
        class_handle: int,
    ) -> int:
        result = await self._service_rpc(
            "object",
            "QueryInteractionTransportationType",
            "QueryInteractionTransportationTypeRequest",
            target_federate_handle=target_federate_handle,
            interaction_class_handle=class_handle,
        )
        return int(result.transport_type)

    async def flush_callback_invocation_reports(self) -> None:
        from ._generic_callbacks import callback_state

        state = callback_state(self)
        for receipt, success, exception in state.pending():
            await self.report_callback_invocation(receipt, success, exception)
            state.acknowledged(receipt)

    async def report_callback_invocation(
        self,
        callback_receipt: bytes,
        success: bool,
        exception: str = "",
        *,
        invocation_entry: bool = False,
    ) -> None:
        """Report one actual callback attempt; never retry the user callback."""
        if not isinstance(callback_receipt, bytes) or len(callback_receipt) != 32:
            raise ValueError("callback receipt must contain 32 bytes")
        if not isinstance(success, bool) or not isinstance(exception, str):
            raise TypeError("callback outcome requires bool success and string exception")
        if success and exception:
            raise ValueError("successful callback cannot carry an exception")
        if not isinstance(invocation_entry, bool):
            raise TypeError("invocation_entry requires bool")
        if invocation_entry and (not success or exception):
            raise ValueError("callback entry requires success and no exception")
        if invocation_entry and not self.callback_invocation_entry_supported:
            raise RuntimeError("server did not advertise callback invocation entry support")
        if len(getattr(self, "_session_token", b"")) != 32:
            raise RuntimeError("service requires an authenticated join session")
        import grpc

        from ._generic_callbacks import callback_state

        generic = callback_state(self)
        identity = getattr(self._transport, "_callback_ticket_identities", {}).get(
            self.handle, {}).get(callback_receipt)
        generic.claim(callback_receipt, identity)
        if not invocation_entry:
            generic.complete(callback_receipt, success, exception, identity)
        reports = getattr(self._transport, "_callback_reports_pending", None)
        if reports is None:
            reports = self._transport._callback_reports_pending = {}
        pending = reports.setdefault(self.handle, set())
        was_pending = callback_receipt in pending
        pending.add(callback_receipt)
        for attempt in range(3):
            try:
                await self._service_rpc(
                    "support",
                    "ReportCallbackInvocation",
                    "ReportCallbackInvocationRequest",
                    authenticated=True,
                    rpc_timeout=2.0,
                    callback_receipt=callback_receipt,
                    success=success,
                    exception=exception,
                    invocation_entry=invocation_entry,
                )
                if not invocation_entry:
                    pending.discard(callback_receipt)
                    generic.acknowledged(callback_receipt)
                    getattr(self._transport, "_callback_ticket_identities", {}).get(
                        self.handle, {}).pop(callback_receipt, None)
                return
            except Exception as exc:
                cause = exc.__cause__ or exc
                if (
                    not isinstance(cause, grpc.RpcError)
                    or cause.code()
                    not in (
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                    )
                    or attempt == 2
                ):
                    if (invocation_entry and not was_pending and isinstance(cause, grpc.RpcError)
                            and cause.code() not in (grpc.StatusCode.UNAVAILABLE,
                                                     grpc.StatusCode.DEADLINE_EXCEEDED)):
                        pending.discard(callback_receipt)
                        generic.entry_failed(callback_receipt)
                    raise
                await asyncio.sleep(0.01 * (attempt + 1))
