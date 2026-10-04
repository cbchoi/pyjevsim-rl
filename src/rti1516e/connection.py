"""Layer 1 — ``RtiConnection`` and ``Federate`` asyncio API.

Typical usage:

    async with RtiConnection.connect(url="grpc://localhost:8442") as rti:
        async with rti.join_federation(
            FederationSpec(name="demo", fom_modules=["./demo.fom.xml"]),
            federate_name="alice",
        ) as fed:
            await fed.publish_object_class("Vehicle", attributes=["pos"])
            async for event in fed.events():
                ...

The connection-level transport is gRPC over HTTP/2 to the rtid binary in
production; spec tests inject an in-process ``FakeRtiServer`` via the
``memory://`` URL scheme (see rti1516e._transport). Generated stubs live
in rti1516e._generated/ (gitignored; regenerate with `make py-codegen`).

Public method names and signatures form the stable Layer 1 SDK contract.
Private implementation details and backward-compatible dataclass fields
may evolve independently.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
from types import TracebackType
from typing import Any, Self, TypeAlias, cast

from rti1516e._logical_time import FLOAT64_TIME, require_supported_logical_time
from rti1516e._services import FederateServices
from rti1516e._transport import build_grpc_transport as _build_grpc_transport
from rti1516e._transport import lookup as _lookup_transport
from rti1516e.errors import (
    FederateAlreadyExecutionMember,
    FederateNameAlreadyInUse,
    RestoreInProgress,
    SaveInProgress,
)
from rti1516e.fom.modules import MIMInput
from rti1516e.handles import (
    AttributeHandle,
    DimensionHandle,
    FederateHandle,
    InteractionClassHandle,
    ObjectClassHandle,
    ObjectInstanceHandle,
    ParameterHandle,
    RegionHandle,
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

# M28 — IEEE 1516 portability type aliases. Layer 1 widens its accept types so the
# typed-handle / typed-collection callers from Layer 2 land cleanly under
# mypy --strict. Typed handles are int subclasses → no runtime change.
_ObjectClassRef: TypeAlias = "int | str | ObjectClassHandle"
_AttributeRef: TypeAlias = "int | str | AttributeHandle"
_InteractionClassRef: TypeAlias = "int | str | InteractionClassHandle"
_ParameterRef: TypeAlias = "int | str | ParameterHandle"
_ObjectInstanceRef: TypeAlias = "int | ObjectInstanceHandle"
_DimensionRef: TypeAlias = "int | str | DimensionHandle"
_FederateRef: TypeAlias = "int | FederateHandle"
_RegionRef: TypeAlias = "int | RegionHandle"
_AttributeRefList: TypeAlias = "list[int | str | AttributeHandle] | AttributeHandleSet"
_ParameterRefList: TypeAlias = "list[int | str | ParameterHandle] | ParameterHandleSet"
_FederateRefList: TypeAlias = "list[int | FederateHandle] | FederateHandleSet"
_DimensionRefList: TypeAlias = "list[int | str | DimensionHandle] | DimensionHandleSet"
_RegionRefList: TypeAlias = "list[int | RegionHandle] | RegionHandleSet"
_AttributeValueDict: TypeAlias = "dict[int | str | AttributeHandle, Any] | AttributeHandleValueMap"
_ParameterValueDict: TypeAlias = "dict[int | str | ParameterHandle, Any] | ParameterHandleValueMap"


async def _dispatch(transport: Any, method: str, **kwargs: Any) -> Any:
    """Call ``transport.record(method, **kwargs)`` and await if coroutine.

    The fake (FakeRtiServer) returns synchronously; the real GrpcTransport
    returns a coroutine. Both call sites share the same source by funneling
    through this helper. Wrapping the await in ``inspect.isawaitable`` keeps
    the fake path zero-cost (no extra event-loop tick) while letting the
    gRPC path do its real RPC.
    """
    result = transport.record(method, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _await_task_to_completion(
    task: asyncio.Task[Any],
) -> asyncio.CancelledError | None:
    """Wait for an ownership task even when this waiter is cancelled.

    Cancellation is remembered and returned after ``task`` has reached a
    terminal state.  Repeated ``Task.cancel()`` calls therefore cannot strand
    admission counters or joined-federate ownership halfway through a
    lifecycle transition.
    """

    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            # Preserve the first cancellation request.  Later requests must
            # not replace its message/identity while an ownership operation
            # is being brought to a terminal state.
            if cancellation is None:
                cancellation = exc
        except BaseException:
            # The owned task reached a failed terminal state.  Read it below
            # via task.result() so a remembered caller cancellation can be
            # reported alongside the owned failure.
            break
    try:
        task.result()
    except BaseException as owned_error:
        if cancellation is not None:
            raise BaseExceptionGroup(
                "cancelled waiter and ownership operation both failed",
                [cancellation, owned_error],
            ) from owned_error
        raise
    return cancellation


class FederationJoinCommitUnknownError(RuntimeError):
    """Join may have committed remotely but yielded no usable handle.

    The connection is poisoned after this error.  Guessing a handle and
    issuing resign would risk resigning the wrong federate; the only supported
    recovery is :meth:`RtiConnection.recover_commit_unknown`, which closes the
    transport before a new connection is created.
    """

    def __init__(
        self,
        federation_name: str,
        federate_name: str,
        failure: BaseException,
    ) -> None:
        super().__init__(
            "join_federation completion is commit-unknown for "
            f"{federation_name!r}/{federate_name!r}; close this connection "
            "with recover_commit_unknown() before reconnecting"
        )
        self.federation_name = federation_name
        self.federate_name = federate_name
        self.failure = failure


class RtiConnectionPoisonedError(RuntimeError):
    """The connection cannot safely admit work after commit-unknown."""

    def __init__(self, poison: FederationJoinCommitUnknownError) -> None:
        super().__init__(
            "RtiConnection is poisoned by a commit-unknown join; "
            "call recover_commit_unknown() and create a new connection"
        )
        self.poison = poison


@dataclass(frozen=True)
class FederationSpec:
    """Description of a federation to create or join.

    ``mim_module`` is a file path or ``MIMModule(designator, xml)``. It is
    separate from user FOM modules; None selects the legacy embedded MIM on
    creation. Explicit selection also becomes an atomic join precondition.
    """

    name: str
    fom_modules: list[str] = field(default_factory=list)
    mode: str = "verbose"  # "verbose" | "best-effort" federation mode
    seed: int = 0  # 0 = use server default; non-zero pins for determinism
    stall_timeout_seconds: int = 0  # 0 = server default (60s)
    additional_fom_modules: list[str] = field(default_factory=list)
    mim_module: MIMInput | None = None
    logical_time_implementation_name: str = ""

    def __post_init__(self) -> None:
        from .fom.modules import module_paths, validate_mim_input

        object.__setattr__(self, "fom_modules", module_paths(self.fom_modules, "fom_modules"))
        object.__setattr__(self, "additional_fom_modules", module_paths(
            self.additional_fom_modules, "additional_fom_modules"))
        validate_mim_input(self.mim_module)
        require_supported_logical_time(self.logical_time_implementation_name)


class RtiConnection:
    """Async connection to a single rtid instance."""

    def __init__(
        self,
        url: str,
        *,
        options: dict[str, Any] | None = None,
        ca_cert: bytes | None = None,
        client_cert: bytes | None = None,
        client_key: bytes | None = None,
        bearer_token: str | None = None,
    ) -> None:
        self._url = url
        self._options: dict[str, Any] = dict(options) if options else {}
        # PEM-encoded trusted CA bundle for ``grpcs://`` URLs. ``None``
        # means "use system roots" (handed straight through to
        # ``grpc.ssl_channel_credentials(root_certificates=None)``).
        self._ca_cert = ca_cert
        # M14 W3 — mTLS + bearer token.
        self._client_cert = client_cert
        self._client_key = client_key
        self._bearer_token = bearer_token
        self._transport: Any | None = None
        self._opening = False
        self._open_operation: asyncio.Task[Self] | None = None
        self._closed = False
        self._closing = False
        self._poison: FederationJoinCommitUnknownError | None = None
        self._close_lock = asyncio.Lock()
        self._close_operation: asyncio.Task[None] | None = None
        self._admission_condition = asyncio.Condition()
        self._inflight_operations = 0
        self._event_waiters: set[asyncio.Task[Any]] = set()
        self._active_federate_contexts: set[_FederateContextManager] = set()

    @classmethod
    def connect(
        cls,
        url: str,
        *,
        options: dict[str, Any] | None = None,
        ca_cert: bytes | None = None,
        client_cert: bytes | None = None,
        client_key: bytes | None = None,
        bearer_token: str | None = None,
    ) -> Self:
        """Build a connection wrapper bound to ``url``.

        Supported URL schemes:

          - ``memory://<name>``   — in-process driver registered via
            ``InProcessTransport`` (or the legacy ``FakeRtiServer`` alias).
          - ``grpc://host:port``  — real gRPC over plaintext TCP.
          - ``grpcs://host:port`` — real gRPC over TLS. ``ca_cert``
            (PEM bytes) populates the trust store for verifying the
            rtid server cert; pass ``None`` to rely on system roots
            (typical when the rtid cert chains to a publicly trusted CA).

        ``connect()`` is intentionally synchronous so it can be used as the
        head of an ``async with`` statement::

            async with RtiConnection.connect(
                "grpcs://rtid.example.com:8442",
                ca_cert=Path("ca.pem").read_bytes(),
            ) as rti:
                ...

        The actual transport setup happens inside ``__aenter__``.
        """
        return cls(
            url,
            options=options,
            ca_cert=ca_cert,
            client_cert=client_cert,
            client_key=client_key,
            bearer_token=bearer_token,
        )

    async def __aenter__(self) -> Self:
        """Open the transport.

        Dispatch by URL scheme:

          - ``memory://``  — look up the registered in-process driver.
          - ``grpc://``    — open a plaintext ``grpc.aio.insecure_channel``.
          - ``grpcs://``   — open a TLS ``grpc.aio.secure_channel`` using
            ``self._ca_cert`` as the root CA bundle (or system roots if
            ``ca_cert`` was not supplied).

        Anything else raises ``ValueError`` at connect time.
        """
        async with self._admission_condition:
            self._ensure_accepting_work()
            if self._opening or self._transport is not None:
                raise RuntimeError("RtiConnection is already opening or open")
            self._opening = True
            self._inflight_operations += 1
        operation = asyncio.create_task(self._open_once())
        self._open_operation = operation
        opened: Self | None = None
        opening_error: BaseException | None = None
        cancellation: asyncio.CancelledError | None = None
        try:
            cancellation = await _await_task_to_completion(operation)
            opened = operation.result()
        except BaseException as exc:
            opening_error = exc

        release = asyncio.create_task(self._release_open_lease())
        release_error: BaseException | None = None
        try:
            release_cancellation = await _await_task_to_completion(release)
            if cancellation is None:
                cancellation = release_cancellation
        except BaseException as exc:
            release_error = exc
        finally:
            if self._open_operation is operation:
                self._open_operation = None

        primary_error = opening_error
        if primary_error is not None and release_error is not None:
            primary_error = BaseExceptionGroup(
                "transport open and admission release both failed",
                [primary_error, release_error],
            )
        elif release_error is not None:
            primary_error = release_error

        # A caller-cancelled __aenter__ has no matching __aexit__.  Once the
        # builder published a transport, compensate by closing it to
        # completion before propagating that first cancellation.
        if (primary_error is not None or cancellation is not None) and self._transport is not None:
            cleanup = asyncio.create_task(self.close())
            try:
                await _await_task_to_completion(cleanup)
            except BaseException as cleanup_error:
                failures = [
                    error
                    for error in (primary_error, cancellation, cleanup_error)
                    if error is not None
                ]
                raise BaseExceptionGroup(
                    "transport open failed and compensating close failed",
                    failures,
                ) from cleanup_error

        if primary_error is not None:
            raise primary_error
        if cancellation is not None:
            raise cancellation
        if self._closing:
            raise RuntimeError("RtiConnection closed while opening")
        if opened is None:  # pragma: no cover - guarded by operation.result()
            raise RuntimeError("transport open completed without a connection")
        return opened

    async def _release_open_lease(self) -> None:
        async with self._admission_condition:
            self._opening = False
            self._inflight_operations -= 1
            if self._inflight_operations == 0:
                self._admission_condition.notify_all()

    async def _open_once(self) -> Self:
        scheme, _, _ = self._url.partition("://")
        if scheme == "memory":
            transport = _lookup_transport(self._url)
            if transport is None:
                raise RuntimeError(
                    f"no in-process transport registered for {self._url!r} — "
                    "construct an InProcessTransport (or the legacy "
                    "FakeRtiServer alias) first; it auto-registers under "
                    "memory://fake-rti"
                )
            self._transport = transport
        elif scheme in ("grpc", "grpcs"):
            # Real gRPC transport. Plaintext for ``grpc://``;
            # TLS-secured (server auth, no mTLS) for ``grpcs://``.
            # ``ca_cert`` is forwarded to ``ssl_channel_credentials``;
            # ``grpc://`` ignores it.
            self._transport = await _build_grpc_transport(
                self._url,
                ca_cert=self._ca_cert,
                client_cert=self._client_cert,
                client_key=self._client_key,
                bearer_token=self._bearer_token,
            )
        else:
            raise ValueError(
                f"unsupported URL scheme {scheme!r} (expected 'memory', 'grpc', or 'grpcs')"
            )
        if self._closing:
            raise RuntimeError("RtiConnection closed while opening")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Tear down the connection without losing a retryable failure.

        Ownership is cleared only after the transport confirms closure.  A
        transport that raises remains attached to this connection so callers
        can retry and inspect the original failure.
        """
        async with self._close_lock:
            if self._closed:
                return
            operation = self._close_operation
            if operation is None:
                # Admission closes before the transport operation starts.
                # A failed close remains retryable, but shutdown is still
                # fail-closed for federation services.
                self._closing = True
                operation = asyncio.create_task(self._close_once())
                self._close_operation = operation
        try:
            # One caller being cancelled must not cancel cleanup shared by
            # other concurrent callers. Its CancelledError still propagates.
            await asyncio.shield(operation)
        finally:
            if operation.done():
                async with self._close_lock:
                    if self._close_operation is operation:
                        self._close_operation = None

    async def recover_commit_unknown(self) -> None:
        """Close a poisoned connection without guessing a federate handle.

        Recovery is deliberately terminal for this wrapper.  After it
        returns, construct and enter a new :class:`RtiConnection`.
        """

        if self._poison is None:
            raise RuntimeError("RtiConnection has no commit-unknown join")
        await self.close()

    async def _close_once(self) -> None:
        async with self._admission_condition:
            event_waiters = tuple(self._event_waiters)
        for waiter in event_waiters:
            waiter.cancel()
        async with self._admission_condition:
            await self._admission_condition.wait_for(lambda: self._inflight_operations == 0)
            active_contexts = tuple(self._active_federate_contexts)
        if active_contexts:
            results = await asyncio.gather(
                *(context.__aexit__(None, None, None) for context in active_contexts),
                return_exceptions=True,
            )
            resign_errors = [result for result in results if isinstance(result, BaseException)]
            if len(resign_errors) == 1:
                raise resign_errors[0]
            if resign_errors:
                raise BaseExceptionGroup("RtiConnection federate cleanup failed", resign_errors)
        async with self._admission_condition:
            await self._admission_condition.wait_for(
                lambda: self._inflight_operations == 0 and not self._active_federate_contexts
            )
        transport = self._transport
        # GrpcTransport owns a real gRPC channel + background stream
        # tasks; close them. An in-process transport has no close method and
        # is therefore already complete at this ownership boundary.
        close_fn = None if transport is None else getattr(transport, "close", None)
        if callable(close_fn):
            result = close_fn()
            if inspect.isawaitable(result):
                await result
        self._transport = None
        self._closed = True

    def _ensure_accepting_work(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("RtiConnection is closing or closed")
        if self._poison is not None:
            raise RtiConnectionPoisonedError(self._poison) from self._poison

    async def _release_admission_lease(self) -> None:
        async with self._admission_condition:
            self._inflight_operations -= 1
            if self._inflight_operations == 0:
                self._admission_condition.notify_all()

    async def _finish_lease(
        self,
        release: asyncio.Task[None],
        primary_error: BaseException | None,
        *,
        label: str,
    ) -> None:
        try:
            cancellation = await _await_task_to_completion(release)
        except BaseException as release_error:
            if primary_error is not None:
                raise BaseExceptionGroup(
                    f"{label} operation and lease release both failed",
                    [primary_error, release_error],
                ) from release_error
            raise
        if primary_error is not None:
            raise primary_error
        if cancellation is not None:
            raise cancellation

    @asynccontextmanager
    async def _admit_work(self) -> AsyncIterator[Any]:
        async with self._admission_condition:
            self._ensure_accepting_work()
            transport = self.transport
            self._inflight_operations += 1
        primary_error: BaseException | None = None
        try:
            yield transport
        except BaseException as exc:
            primary_error = exc
        release = asyncio.create_task(self._release_admission_lease())
        await self._finish_lease(release, primary_error, label="admitted")

    @asynccontextmanager
    async def _admit_shutdown_work(self) -> AsyncIterator[Any]:
        """Admit dependency-ordered cleanup after normal admission closes."""

        async with self._admission_condition:
            if self._closed or self._transport is None:
                raise RuntimeError("RtiConnection is closed")
            transport = self._transport
            self._inflight_operations += 1
        primary_error: BaseException | None = None
        try:
            yield transport
        except BaseException as exc:
            primary_error = exc
        release = asyncio.create_task(self._release_admission_lease())
        await self._finish_lease(release, primary_error, label="shutdown")

    async def _register_federate_context(self, context: _FederateContextManager) -> None:
        async with self._admission_condition:
            self._active_federate_contexts.add(context)

    async def _release_federate_context(self, context: _FederateContextManager) -> None:
        async with self._admission_condition:
            self._active_federate_contexts.discard(context)
            self._admission_condition.notify_all()

    async def _next_event(self, queue: Any) -> Any:
        """Wait for one event as cancellable, drainable admitted work."""

        waiter = asyncio.current_task()
        if waiter is None:
            raise RuntimeError("event iteration requires an asyncio task")
        async with self._admission_condition:
            self._ensure_accepting_work()
            self._inflight_operations += 1
            self._event_waiters.add(waiter)
        event: Any = None
        primary_error: BaseException | None = None
        try:
            event = await queue.get()
        except BaseException as exc:
            primary_error = exc
        release = asyncio.create_task(self._release_event_waiter(waiter))
        await self._finish_lease(release, primary_error, label="event wait")
        return event

    async def _release_event_waiter(self, waiter: asyncio.Task[Any]) -> None:
        async with self._admission_condition:
            self._event_waiters.discard(waiter)
            self._inflight_operations -= 1
            if self._inflight_operations == 0:
                self._admission_condition.notify_all()

    def _poison_join_commit_unknown(
        self,
        spec: FederationSpec,
        federate_name: str,
        failure: BaseException,
    ) -> FederationJoinCommitUnknownError:
        poison = FederationJoinCommitUnknownError(
            spec.name,
            federate_name,
            failure,
        )
        self._poison = poison
        return poison

    @property
    def transport(self) -> Any:
        """Internal: the open transport (fake or gRPC). Raises if not entered."""
        if self._transport is None:
            raise RuntimeError(
                "RtiConnection is not open — use `async with RtiConnection.connect(...)`"
            )
        return self._transport

    async def create_federation(
        self, spec: FederationSpec, *, exist_ok: bool = False,
        logical_time_implementation_name: str | None = None,
    ) -> None:
        """§4.5 createFederationExecution (M39 HA-2).

        Unlike the rolled create-on-join path (``join_federation``,
        which stays idempotent), this surfaces a duplicate name as the
        typed :class:`rti1516e.errors.FederationExecutionAlreadyExists`
        unless ``exist_ok=True``.

        The optional time keyword overrides the spec without modifying it.
        Empty selects Float64; unsupported implementations fail before dispatch.
        """
        if logical_time_implementation_name is not None:
            spec = replace(spec, logical_time_implementation_name=logical_time_implementation_name)
        require_supported_logical_time(spec.logical_time_implementation_name)
        async with self._admit_work() as transport:
            await _dispatch(
                transport,
                "create_federation",
                spec=spec,
                exist_ok=exist_ok,
            )

    async def destroy_federation(self, federation_name: str) -> None:
        """§4.6 destroyFederationExecution (M39 HA-2).

        Raises the typed
        :class:`rti1516e.errors.FederatesCurrentlyJoined` while members
        remain joined and
        :class:`rti1516e.errors.FederationExecutionDoesNotExist` for an
        unknown name.
        """
        async with self._admit_work() as transport:
            await _dispatch(
                transport,
                "destroy_federation",
                federation_name=federation_name,
            )

    def join_federation(
        self,
        spec: FederationSpec,
        *,
        federate_name: str,
        federate_type: str = "",
    ) -> _FederateContextManager:
        """Open the per-federate async context manager.

        Use as ``async with rti.join_federation(spec, federate_name="x") as fed``.

        ``federate_type`` is the optional HLAfederateType designator for the
        joining federate (M13 thread B — docs/srs.md §10.4); defaults to
        empty string (cut-1 behavior). When set, the rtid records it on
        the federation roster and surfaces it via the MOM HLAfederate
        attribute set.
        """
        self._ensure_accepting_work()
        return _FederateContextManager(self, spec, federate_name, federate_type)


class _FederateContextManager:
    """Internal: returned by RtiConnection.join_federation.

    On enter: records a ``create_federation`` call (idempotent on the server
    side; canned exceptions like ``FederationAlreadyExists`` propagate from
    the fake) followed by ``join_federation``, then allocates a federate
    handle and constructs the Federate.

    On exit: records ``resign_federation``. The connection itself is owned
    by the outer ``async with RtiConnection.connect(...)`` — we do not close
    it here.
    """

    def __init__(
        self,
        connection: RtiConnection,
        spec: FederationSpec,
        federate_name: str,
        federate_type: str = "",
    ) -> None:
        self._connection = connection
        self._spec = spec
        self._federate_name = federate_name
        self._federate_type = federate_type
        self._federate: Federate | None = None
        self._enter_lock = asyncio.Lock()
        self._enter_state = "new"
        self._exit_lock = asyncio.Lock()
        self._exit_operation: asyncio.Task[None] | None = None
        #: M36 — IEEE 1516.1-2010 §4.10 resign-action designator forwarded
        #: to the transport on ``__aexit__``. ``None`` keeps the transport
        #: default (UNCONDITIONALLY_DIVEST_ATTRIBUTES). Layer 2
        #: (``standard.Rti1516eAmbassador.resignFederationExecution``) sets
        #: this before exiting the context manager.
        self.resign_action: str | int | None = None

    async def __aenter__(self) -> Federate:
        async with self._enter_lock:
            if self._enter_state != "new":
                raise RuntimeError(
                    "federate context manager is single-use; "
                    f"current state is {self._enter_state!r}"
                )
            self._enter_state = "entering"
            try:
                return await self._enter_once()
            except BaseException as enter_error:
                # Cancellation may arrive after registration but while the
                # outer admission lease is being released. At that point
                # ownership is real and must be compensated even though
                # _enter_once did not return a Federate to its caller.
                if self._federate is not None:
                    resign = asyncio.create_task(self._resign_once())
                    try:
                        await _await_task_to_completion(resign)
                    except BaseException as cleanup_error:
                        self._enter_state = "failed"
                        raise BaseExceptionGroup(
                            "federate enter and compensating resign both failed",
                            [enter_error, cleanup_error],
                        ) from enter_error
                self._enter_state = "failed"
                raise

    async def _enter_once(self) -> Federate:
        require_supported_logical_time(self._spec.logical_time_implementation_name)
        async with self._connection._admit_work() as transport:
            # The create+join pair is one admitted lifecycle boundary. Close
            # cannot split it after create and before join.
            await _dispatch(
                transport,
                "create_federation",
                spec=self._spec,
                exist_ok=True,
            )
            join_operation = asyncio.create_task(
                _dispatch(
                    transport,
                    "join_federation",
                    spec=self._spec,
                    federate_name=self._federate_name,
                    federate_type=self._federate_type,
                )
            )
            try:
                cancellation = await _await_task_to_completion(join_operation)
                join_response = join_operation.result()
            except (FederateNameAlreadyInUse, FederateAlreadyExecutionMember,
                    SaveInProgress, RestoreInProgress):
                # These admission rejections do not create a new membership.
                # Other failures may follow a committed join or a lost response.
                raise
            except BaseException as join_error:
                raise self._connection._poison_join_commit_unknown(
                    self._spec,
                    self._federate_name,
                    join_error,
                ) from join_error
            # The real gRPC transport returns the federate handle from
            # JoinFederationResponse; the fake returns None and the SDK falls
            # back to allocate_handle() (matching the legacy contract).
            if isinstance(join_response, int):
                handle = int(join_response)
            else:
                handle = int(transport.allocate_handle())
            federate = Federate(
                connection=self._connection,
                transport=transport,
                handle=handle,
                name=self._federate_name,
            )
            self._federate = federate
            registration = asyncio.create_task(self._connection._register_federate_context(self))
            try:
                registration_cancellation = await _await_task_to_completion(registration)
                if cancellation is None:
                    cancellation = registration_cancellation
                if cancellation is not None:
                    raise cancellation
            except BaseException as enter_error:
                # Join has committed server-side ownership.  A cancellation
                # or registration fault must therefore compensate with a
                # completed resign before __aenter__ can fail to its caller.
                resign = asyncio.create_task(self._resign_once())
                try:
                    await _await_task_to_completion(resign)
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "joined federate registration cleanup failed",
                        [enter_error, cleanup_error],
                    ) from enter_error
                raise
        self._enter_state = "active"
        return federate

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        # Do not let an early/concurrent exit observe the transient pre-handle
        # state and return while __aenter__ later publishes ownership.
        async with self._enter_lock:
            pass
        async with self._exit_lock:
            if self._federate is None:
                return
            operation = self._exit_operation
            if operation is None:
                operation = asyncio.create_task(self._resign_once())
                self._exit_operation = operation
        try:
            await asyncio.shield(operation)
        finally:
            if operation.done():
                async with self._exit_lock:
                    if self._exit_operation is operation:
                        self._exit_operation = None

    async def _resign_once(self) -> None:
        federate = self._federate
        if federate is None:
            return
        # A resign failure is an incomplete lifecycle boundary. Keep the
        # federate reachable and surface the error so the caller can retry
        # while the dependent connection is still open.
        async with self._connection._admit_shutdown_work() as transport:
            await _dispatch(
                transport,
                "resign_federation",
                federate_handle=federate.handle,
                federate_name=federate.name,
                action=self.resign_action,
            )
        release = asyncio.create_task(self._connection._release_federate_context(self))
        cancellation = await _await_task_to_completion(release)
        self._federate = None
        self._enter_state = "exited"
        if cancellation is not None:
            raise cancellation


def _admitted_service(method: Callable[..., Any]) -> Callable[..., Any]:
    """Hold a connection admission lease for one federate service call."""

    @wraps(method)
    async def admitted(self: Federate, *args: Any, **kwargs: Any) -> Any:
        async with self._connection._admit_work():
            self._guard_service()
            await self.flush_ownership_callback_receipts()
            return await method(self, *args, **kwargs)

    return admitted


class Federate(FederateServices):
    """A joined federate. Created via ``rti.join_federation(...)``.

    Public attributes ``name`` and ``handle`` are stable. The service-group
    accessors (:meth:`sync`, :meth:`ownership`, :meth:`ddm`,
    :meth:`savepoint`) lazily construct dedicated client wrappers around
    the same underlying gRPC channel. :meth:`mom` exposes the read-only
    Management Object Model introspection surface.
    """

    name: str
    handle: int

    def __init__(
        self,
        *,
        connection: RtiConnection,
        transport: Any,
        handle: int,
        name: str,
    ) -> None:
        self._connection = connection
        self._transport = transport
        self.handle = handle
        self.name = name
        entry_capability = getattr(transport, "callback_invocation_entry_supported", None)
        self._callback_entry_supported = (
            bool(entry_capability(handle)) if callable(entry_capability) else False
        )
        time_reader = getattr(transport, "logical_time_implementation_name", None)
        self._logical_time_implementation_name = (
            time_reader() if callable(time_reader) else FLOAT64_TIME
        )
        token_reader = getattr(transport, "session_token", None)
        self._session_token = token_reader(handle) if callable(token_reader) else b""
        # Lazily-instantiated cut-3 service-group clients. Each is
        # built on first attribute access (the transport may not have
        # the channel attribute when running over the in-process
        # FakeRtiServer; the accessors handle that case explicitly).
        self._sync_client: Any | None = None
        self._ownership_client: Any | None = None
        self._ddm_client: Any | None = None
        self._savepoint_client: Any | None = None
        self._mom_client: Any | None = None
        self._support_client: Any | None = None
        self._reservation_client: Any | None = None

        binder = getattr(transport, "bind_federate_identity", None)
        if callable(binder):
            binder(handle, self._restore_identity)
        self._ownership_callback_state()

    def _restore_identity(self, handle: int) -> None:
        self.handle = handle
        capability = getattr(self._transport, "callback_invocation_entry_supported", None)
        if callable(capability):
            self._callback_entry_supported = bool(capability(handle))
        token = getattr(self._transport, "session_token", None)
        if callable(token):
            self._session_token = token(handle)
        for name in (
            "_sync_client",
            "_ownership_client",
            "_ddm_client",
            "_savepoint_client",
            "_mom_client",
            "_support_client",
            "_reservation_client",
        ):
            client = getattr(self, name)
            if client is not None:
                client._federate_handle = handle
                if hasattr(client, "_session_token"):
                    client._session_token = self._session_token

    @property
    def logical_time_implementation_name(self) -> str:
        """Joined execution's time type; absent legacy metadata means Float64."""
        return self._logical_time_implementation_name

    @property
    def callback_invocation_entry_supported(self) -> bool:
        return self._callback_entry_supported

    @property
    def next_retraction_handle(self) -> int | None:
        """Restored server allocation hint, not an allocation or reservation.

        None or zero does not establish an available identifier. Caller-supplied
        unique identifiers need not be monotonic; the server rejects reuse.
        """
        hint = getattr(self._transport, "restored_retraction_hint", None)
        return hint(self.handle) if callable(hint) else None

    def _guard_service(self) -> None:
        guard = getattr(self._transport, "guard_service", None)
        if callable(guard):
            guard()

    def _guard_restore_complete(self) -> None:
        guard = getattr(self._transport, "guard_restore_complete", None)
        if not callable(guard):
            raise RuntimeError("no validated SDK restore stage is ready")
        guard(self.handle)

    def _guard_client(self, client: Any) -> Any:
        cache: dict[int, _RestoreGuardedClient] | None = getattr(self, "_guarded_clients", None)
        if cache is None:
            cache = {}
            self._guarded_clients = cache
        key = id(client)
        if key not in cache:
            cache[key] = _RestoreGuardedClient(
                client, self._guard_service, self.flush_ownership_callback_receipts
            )
        return cache[key]

    @_admitted_service
    async def refresh_fom(self) -> None:
        """Refresh authoritative bindings after another federate extends the FOM."""
        await self._transport.refresh_fom(self.handle)

    # --- Declaration management ---

    @_admitted_service
    async def publish_object_class(
        self, class_name: _ObjectClassRef, *, attributes: _AttributeRefList
    ) -> None:
        """Declare publication of an object class + its attributes.

        M27 Phase B: ``class_name`` and ``attributes`` accept either
        ``int`` (already-resolved FOM handle, IEEE 1516 service-style) or ``str``
        (FOM name, pysdk convenience). The parameter is still named
        ``class_name`` for source-compat with pre-M27 callers.
        """
        await _dispatch(
            self._transport,
            "publish_object_class",
            federate_handle=self.handle,
            class_name=class_name,
            attributes=list(attributes),
        )

    @_admitted_service
    async def subscribe_object_class(
        self,
        class_name: _ObjectClassRef,
        *,
        attributes: _AttributeRefList,
        active: bool = True,
        update_rate_designator: str = "",
    ) -> None:
        """Declare subscription to an object class + its attributes.

        See :meth:`publish_object_class` for the M27 Phase B union types.
        """
        from .declaration import validate_subscription_options

        validate_subscription_options(active, update_rate_designator)
        await _dispatch(
            self._transport,
            "subscribe_object_class",
            federate_handle=self.handle,
            class_name=class_name,
            attributes=list(attributes),
            active=active,
            update_rate_designator=update_rate_designator,
        )

    @_admitted_service
    async def publish_interaction_class(self, class_name: _InteractionClassRef) -> None:
        """Declare publication of an interaction class.

        M27 Phase D: ``class_name`` accepts ``int`` (FOM handle) or
        ``str`` (FOM name).
        """
        await _dispatch(
            self._transport,
            "publish_interaction_class",
            federate_handle=self.handle,
            class_name=class_name,
        )

    @_admitted_service
    async def subscribe_interaction_class(
        self, class_name: _InteractionClassRef, *, active: bool = True
    ) -> None:
        """Declare subscription to an interaction class.

        M27 Phase D: ``class_name`` accepts ``int`` (FOM handle) or
        ``str`` (FOM name). Subscriber federates that joined an
        already-created federation should prefer the int form
        (handle resolved via :meth:`support.get_interaction_class_handle`).
        """
        from .declaration import validate_subscription_options

        validate_subscription_options(active, "")
        await _dispatch(
            self._transport,
            "subscribe_interaction_class",
            federate_handle=self.handle,
            class_name=class_name,
            active=active,
        )

    @_admitted_service
    async def set_advisory_switch(self, switch_kind: int, enabled: bool) -> None:
        from .declaration import AdvisorySwitchKind

        if isinstance(switch_kind, bool):
            raise ValueError("invalid advisory switch kind")
        kind = AdvisorySwitchKind(switch_kind)
        if type(enabled) is not bool:
            raise TypeError("enabled must be a bool")
        await _dispatch(
            self._transport,
            "set_advisory_switch",
            federate_handle=self.handle,
            switch_kind=int(kind),
            enabled=enabled,
        )

    # --- Object management ---

    @_admitted_service
    async def register_object_instance(
        self, class_name: _ObjectClassRef, *, instance_name: str | None = None
    ) -> int:
        """Register an instance and return its handle.

        M27 Phase B: ``class_name`` accepts ``int`` (FOM handle) or
        ``str`` (FOM name). reference_rti federates that pre-resolved the
        class handle via getObjectClassHandle should pass the int.
        """
        response = await _dispatch(
            self._transport,
            "register_object_instance",
            federate_handle=self.handle,
            class_name=class_name,
            instance_name=instance_name,
        )
        if isinstance(response, int):
            return response
        return int(self._transport.allocate_handle())

    @_admitted_service
    async def update_attributes(
        self,
        object_handle: _ObjectInstanceRef,
        values: _AttributeValueDict,
        *,
        timestamp: float | None = None,
        tag: bytes = b"",
        retraction_handle: int = 0,
    ) -> None:
        """Update one or more attribute values on an object instance.

        M27 Phase B: dict keys accept ``int`` (already-resolved
        attribute handle, IEEE 1516 service-style) or ``str`` (FOM name).
        """
        await _dispatch(
            self._transport,
            "update_attributes",
            tag=bytes(tag),
            retraction_handle=int(retraction_handle),
            federate_handle=self.handle,
            object_handle=object_handle,
            values=dict(values),
            timestamp=timestamp,
        )

    # --- Interaction management ---

    @_admitted_service
    async def change_attribute_order_type(
        self, object_handle: int, attribute_handles: list[int], order_type: int
    ) -> None:
        await _dispatch(
            self._transport,
            "change_attribute_order_type",
            federate_handle=self.handle,
            object_handle=int(object_handle),
            attribute_handles=list(attribute_handles),
            order_type=int(order_type),
        )

    @_admitted_service
    async def change_interaction_order_type(
        self, interaction_class_handle: int, order_type: int
    ) -> None:
        await _dispatch(
            self._transport,
            "change_interaction_order_type",
            federate_handle=self.handle,
            interaction_class_handle=int(interaction_class_handle),
            order_type=int(order_type),
        )

    @_admitted_service
    async def retract(self, retraction_handle: int) -> None:
        await _dispatch(
            self._transport,
            "retract",
            federate_handle=self.handle,
            retraction_handle=int(retraction_handle),
        )

    @_admitted_service
    async def send_interaction(
        self,
        class_name: _InteractionClassRef,
        parameters: _ParameterValueDict,
        *,
        timestamp: float | None = None,
        tag: bytes = b"",
        retraction_handle: int = 0,
    ) -> None:
        """Send an interaction with the given parameters.

        M27 Phase B: ``class_name`` and parameter dict keys accept
        ``int`` (handle) or ``str`` (FOM name).
        """
        await _dispatch(
            self._transport,
            "send_interaction",
            tag=bytes(tag),
            retraction_handle=int(retraction_handle),
            federate_handle=self.handle,
            class_name=class_name,
            parameters=dict(parameters),
            timestamp=timestamp,
        )

    # --- Time management ---

    @_admitted_service
    async def enable_time_regulation(self, lookahead: float) -> None:
        """Request regulation; TimeRegulationEnabled reports actual activation."""
        await _dispatch(
            self._transport,
            "enable_time_regulation",
            federate_handle=self.handle,
            lookahead=lookahead,
        )

    @_admitted_service
    async def enable_time_constrained(self) -> None:
        """Request constraint; TimeConstrainedEnabled reports actual activation."""
        await _dispatch(
            self._transport,
            "enable_time_constrained",
            federate_handle=self.handle,
        )

    @_admitted_service
    async def next_message_request(self, time: float) -> None:
        """Request advance to ``time``. Grant arrives via events()."""
        await _dispatch(
            self._transport,
            "next_message_request",
            federate_handle=self.handle,
            time=time,
        )

    # --- Additional TimeService operations ---

    @_admitted_service
    async def disable_time_regulation(self) -> None:
        """Stop being time-regulating. ``ErrTimeRegulationNotEnabled`` if not."""
        await _dispatch(
            self._transport,
            "disable_time_regulation",
            federate_handle=self.handle,
        )

    @_admitted_service
    async def disable_time_constrained(self) -> None:
        """Stop being time-constrained. ``ErrTimeConstrainedNotEnabled`` if not."""
        await _dispatch(
            self._transport,
            "disable_time_constrained",
            federate_handle=self.handle,
        )

    @_admitted_service
    async def modify_lookahead(self, lookahead: float) -> None:
        """Mutate lookahead without re-enabling regulation."""
        await _dispatch(
            self._transport,
            "modify_lookahead",
            federate_handle=self.handle,
            lookahead=lookahead,
        )

    @_admitted_service
    async def next_message_request_available(self, time: float) -> None:
        """NMRA(t). Grant time may equal LBTS (vs NER's strict-less)."""
        await _dispatch(
            self._transport,
            "next_message_request_available",
            federate_handle=self.handle,
            time=time,
        )

    @_admitted_service
    async def time_advance_request(self, time: float) -> None:
        """TAR(t). Grant fires at min(t, LBTS); pending always clears on grant."""
        await _dispatch(
            self._transport,
            "time_advance_request",
            federate_handle=self.handle,
            time=time,
        )

    @_admitted_service
    async def time_advance_request_available(self, time: float) -> None:
        """TARA(t). Grant time may equal LBTS."""
        await _dispatch(
            self._transport,
            "time_advance_request_available",
            federate_handle=self.handle,
            time=time,
        )

    @_admitted_service
    async def flush_queue_request(self, time: float) -> None:
        """FQR(t). Force-deliver all messages with timestamp ≤ t."""
        await _dispatch(
            self._transport,
            "flush_queue_request",
            federate_handle=self.handle,
            time=time,
        )

    @_admitted_service
    async def query_logical_time(self) -> int | float:
        """Return the federate's current logical time."""
        return cast("int | float", await _dispatch(
            self._transport,
            "query_logical_time",
            federate_handle=self.handle,
        ))

    @_admitted_service
    async def query_lookahead(self) -> int | float:
        """Return the federate's current lookahead. Errors if not regulating."""
        return cast("int | float", await _dispatch(
            self._transport,
            "query_lookahead",
            federate_handle=self.handle,
        ))

    @_admitted_service
    async def query_lbts(self) -> tuple[float, bool]:
        """Return ``(lbts, finite)``. ``finite=False`` ⇒ no regulators."""
        result = await _dispatch(
            self._transport,
            "query_lbts",
        )
        return cast("tuple[float, bool]", result)

    @_admitted_service
    async def enable_asynchronous_delivery(self) -> None:
        """Enable asynchronous delivery. ``TimeAlreadyAsynchronous`` if already on.

        This HLA setting concerns receive-order delivery, not Python asyncio.
        It never releases future timestamp-order messages to a constrained
        federate; those messages remain subject to time advancement.
        """
        await _dispatch(
            self._transport,
            "enable_asynchronous_delivery",
            federate_handle=self.handle,
        )

    @_admitted_service
    async def disable_asynchronous_delivery(self) -> None:
        """Disable asynchronous delivery. ``TimeNotAsynchronous`` if already off."""
        await _dispatch(
            self._transport,
            "disable_asynchronous_delivery",
            federate_handle=self.handle,
        )

    # --- M23 W1: §6 delete_object_instance ---

    @_admitted_service
    async def delete_object_instance(
        self,
        object_handle: _ObjectInstanceRef,
        tag: bytes = b"",
        timestamp: float | None = None,
        *,
        retraction_handle: int = 0,
    ) -> None:
        """Delete an object instance owned by this federate.

        Per IEEE 1516.1-2010 §6.16. Subscribers receive a
        ``RemoveObjectInstance`` event on their events() stream.
        """
        await _dispatch(
            self._transport,
            "delete_object_instance",
            retraction_handle=int(retraction_handle),
            federate_handle=self.handle,
            object_handle=object_handle,
            tag=tag,
            timestamp=timestamp,
        )

    @_admitted_service
    async def local_delete_object_instance(self, object_handle: _ObjectInstanceRef) -> None:
        """Federate-local cleanup; no peer notification (§6.18, M23)."""
        state = self._ownership_callback_state()
        if getattr(state, "active_task", None) is asyncio.current_task():
            await _dispatch(self._transport, "local_delete_object_instance",
                            federate_handle=self.handle, object_handle=object_handle)
        else:
            async with state.async_fence:
                await _dispatch(self._transport, "local_delete_object_instance",
                                federate_handle=self.handle, object_handle=object_handle)

    @_admitted_service
    async def request_attribute_value_update(
        self,
        object_handle: _ObjectInstanceRef,
        attribute_handles: _AttributeRefList,
        tag: bytes = b"",
    ) -> None:
        """Pull-style resync: ask the owner to emit fresh values (§6.24, M23).

        The owner receives a ``ProvideAttributeValueUpdate`` event on
        its events() stream and is expected to respond with
        ``update_attributes``.
        """
        await _dispatch(
            self._transport,
            "request_attribute_value_update",
            federate_handle=self.handle,
            object_handle=object_handle,
            attribute_handles=attribute_handles,
            tag=tag,
        )

    @_admitted_service
    async def request_class_attribute_value_update(
        self,
        object_class_handle: _ObjectClassRef,
        attribute_handles: _AttributeRefList,
        tag: bytes = b"",
    ) -> None:
        """Class-scoped pull (§6.25, M23). Every owner of any instance of
        the class receives a ProvideAttributeValueUpdate event."""
        await _dispatch(
            self._transport,
            "request_class_attribute_value_update",
            federate_handle=self.handle,
            object_class_handle=object_class_handle,
            attribute_handles=attribute_handles,
            tag=tag,
        )

    @_admitted_service
    async def change_attribute_transportation_type(
        self,
        object_handle: _ObjectInstanceRef,
        attribute_handles: _AttributeRefList,
        transport: int,
    ) -> None:
        """Per-instance per-attribute transport override (§6.20, M23).

        ``transport`` is one of the ``TRANSPORTATION_TYPE_*`` enum values
        from ``rti.v1.common_pb2.TransportationType``. M23 ships record-
        only — the wire path doesn't yet route per-message transport.
        """
        await _dispatch(
            self._transport,
            "change_attribute_transportation_type",
            federate_handle=self.handle,
            object_handle=object_handle,
            attribute_handles=attribute_handles,
            transport_type=transport,
        )

    @_admitted_service
    async def change_interaction_transportation_type(
        self,
        interaction_class_handle: _InteractionClassRef,
        transport: int,
    ) -> None:
        """Per-publisher per-class transport override (§6.22, M23)."""
        await _dispatch(
            self._transport,
            "change_interaction_transportation_type",
            federate_handle=self.handle,
            interaction_class_handle=interaction_class_handle,
            transport_type=transport,
        )

    # --- Cut-3 service-group accessors (M12 W2) ---
    #
    # Each property lazily constructs a dedicated thin client wrapper
    # around the same gRPC channel the federate's transport already
    # holds. The clients live in sibling modules (``rti1516e.sync``,
    # ``.ownership``, ``.ddm``, ``.savepoint``) and follow a uniform
    # constructor signature: ``(channel, federation_name, federate_handle)``.
    #
    # Memory:// transports do not own a real gRPC channel, so the
    # accessors raise a clear RuntimeError there. The intended use is
    # cross-process tests + production federates over real gRPC.

    @property
    def sync(self) -> Any:
        """Cut-3 SyncService client for §4.6-4.7 sync-point primitives."""
        if self._sync_client is None:
            from rti1516e.sync import SyncClient

            self._sync_client = SyncClient(
                self._require_channel(),
                federation_name=self._require_federation_name(),
                federate_handle=self.handle,
            )
        return self._guard_client(self._sync_client)

    @property
    def ownership(self) -> Any:
        """Cut-3 OwnershipService client for §7 negotiated transfer + queries."""
        if self._ownership_client is None:
            from rti1516e.ownership import OwnershipClient

            self._ownership_client = OwnershipClient(
                self._require_channel(),
                federation_name=self._require_federation_name(),
                federate_handle=self.handle,
            )
        return self._guard_client(self._ownership_client)

    @property
    def ddm(self) -> Any:
        """Cut-3 DDMService client for §6 region-scoped pub/sub + filtering."""
        if self._ddm_client is None:
            from rti1516e.ddm import DDMClient

            self._ddm_client = DDMClient(
                self._require_channel(),
                federation_name=self._require_federation_name(),
                federate_handle=self.handle,
                session_token=self._session_token,
                logical_time_implementation_name=self.logical_time_implementation_name,
            )
        return self._guard_client(self._ddm_client)

    @property
    def savepoint(self) -> Any:
        """Cut-3 SavepointService client for §4.8-4.15 federation save/restore."""
        if self._savepoint_client is None:
            from rti1516e.savepoint import SavepointClient

            self._savepoint_client = SavepointClient(
                self._require_channel(),
                federation_name=self._require_federation_name(),
                federate_handle=self.handle,
                session_token=self._session_token,
                logical_time_implementation_name=self.logical_time_implementation_name,
                checkpoint_provider=lambda: self._transport.sdk_checkpoint(self.handle),
                restore_complete_guard=self._guard_restore_complete,
            )
        return self._guard_client(self._savepoint_client)

    @property
    def mom(self) -> Any:
        """Cut-3 MomService client for §10 MOM introspection.

        Read-only surface for HLAfederation / HLAfederate object
        snapshots and the per-federate counter set. Federates poll
        these accessors at whatever cadence suits their use case;
        the runtime is goroutine-safe and the snapshot RPCs are
        O(federates).
        """
        if self._mom_client is None:
            from rti1516e.mom import MomClient

            self._mom_client = MomClient(
                self._require_channel(),
                federation_name=self._require_federation_name(),
                federate_handle=self.handle,
                logical_time_implementation_name=self.logical_time_implementation_name,
            )
        return self._guard_client(self._mom_client)

    @property
    def support(self) -> Any:
        """§10.2 SupportService client — handle / name / dimension / order /
        transport lookups against the federation's FOM. M25 Phase B.

        Read-only; safe to call concurrently from any task. Lookups are
        FOM-driven so the returned handles match what the federation
        wire RPCs accept.
        """
        if self._support_client is None:
            from rti1516e.support import SupportClient

            self._support_client = SupportClient(
                self._require_channel(),
                federation_name=self._require_federation_name(),
                federate_handle=self.handle,
                session_token=self._session_token,
            )
        return self._guard_client(self._support_client)

    @property
    def reservation(self) -> Any:
        """§6.1-6.5 ReservationClient — object instance name reservation.

        Result events are delivered on the federate's normal events()
        stream as ObjectInstanceNameReservation{Succeeded,Failed} or
        the Multiple-name variants.
        """
        if self._reservation_client is None:
            from rti1516e.reservation import ReservationClient

            self._reservation_client = ReservationClient(
                self._require_channel(),
                federation_name=self._require_federation_name(),
                federate_handle=self.handle,
            )
        return self._guard_client(self._reservation_client)

    def _require_channel(self) -> Any:
        """Return the underlying ``grpc.aio.Channel`` or raise RuntimeError.

        Cut-3 service-group clients dial real gRPC stubs directly; the
        in-process memory:// transport (FakeRtiServer) has no channel.
        Raise a clear error rather than failing inside the stub
        constructor, which would surface as ``AttributeError``.
        """
        try:
            channel = self._transport.channel
        except AttributeError:
            channel = None
        if channel is None:
            raise RuntimeError(
                "Federate.{sync,ownership,ddm,savepoint,mom} require a "
                "real gRPC transport (grpc:// or grpcs://); the in-process "
                "memory:// transport does not expose a channel"
            )
        return channel

    def _require_federation_name(self) -> str:
        """Return the active federation name or raise RuntimeError.

        Set by ``GrpcTransport`` on every successful create_federation /
        join_federation. Should always be populated by the time the
        federate is observable via ``join_federation()`` __aenter__,
        but the defensive check guards against tests that skip the
        join (none exist today; documented for future-proofing).
        """
        try:
            name = self._transport._federation_name
        except AttributeError:
            name = None
        if not isinstance(name, str) or not name:
            raise RuntimeError(
                "Federate cut-3 clients require a joined federation name; "
                "the transport reports none — was join_federation called?"
            )
        return name

    # --- Event stream ---

    def events(self) -> AsyncIterator[Any]:
        """Yield events emitted by the RTI to this federate.

        Each event is one of the dataclasses in rti1516e.events. The stream
        is open-ended; callers exit via ``break`` or by closing the
        federate context. Backed by ``transport.events_for(handle)`` which
        returns an ``asyncio.Queue`` populated by the test fixture (or, in
        production, drained from the server-streaming RPC).
        """
        return self._iter_events()

    @_admitted_service
    async def set_event_sink(self, sink: Callable[[Any], object] | None) -> bool:
        """Select direct same-loop delivery when the transport supports it."""
        try:
            setter = self._transport.set_event_sink
        except AttributeError:
            setter = None
        if not callable(setter):
            return False
        setter(self.handle, sink)
        return True

    async def _iter_events(self) -> AsyncIterator[Any]:
        queue = self._transport.events_for(self.handle)
        while True:
            event = await self._connection._next_event(queue)
            if isinstance(event, BaseException):
                raise event
            serial = getattr(self._transport, "_restore_serial", 0)
            if getattr(event, "_restore_serial", serial) != serial:
                continue
            yield event


class _RestoreGuardedClient:
    def __init__(
        self, client: Any, guard: Callable[[], None], flush: Callable[..., Any] | None = None
    ) -> None:
        self._client = client
        self._guard = guard
        self._flush = flush

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._client, name)
        if not inspect.iscoroutinefunction(value):
            return value

        @wraps(value)
        async def guarded(*args: Any, **kwargs: Any) -> Any:
            # A rejected local restore must still be reportable to the server.
            if name == "federate_restore_not_complete":
                return await value(*args, **kwargs)
            self._guard()
            if self._flush is not None:
                await self._flush()
            return await value(*args, **kwargs)

        return guarded
