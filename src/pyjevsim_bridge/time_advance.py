"""``HLAFederate`` DEVS time-advance bridge using HLA NER.

Time-advance cycle:

  - Bridge calls coupled_model.time_advance() → DEVS ``ta``.
  - Issues nextMessageRequest(now + ta) on the federate.
  - On TimeAdvanceGrant(t):
      - if t == now + ta (no external event arrived earlier) →
            run output_handler → drain ports → send_interaction
            → internal_transition.
      - if t < now + ta (external event arrived first) →
            external_transition(port, payload). No internal cycle.
  - Loop.

Public method names and the ``run()`` signature form the bridge contract.

Object-class extension (M12 follow-up)
--------------------------------------
The bridge transparently grows an OBJECT-CLASS path when the coupled
model implements any of the methods on
``ObjectClassFederateProtocol`` (see ``_protocol.py``). Detection is
duck-typed via ``hasattr`` so existing interaction-only models keep
working without changes.

  - On startup (lazy, in ``_ensure_federate``):
      ``publish_object_class`` / ``subscribe_object_class`` per the
      model's declarations, then ``register_object_instance`` for
      each entry in ``register_instances``. Handles are stored in
      ``self._instance_handles`` (local_name → handle).
  - In ``_run_internal_cycle`` (in addition to ``output_handler`` →
      ``send_interaction``): if the model has
      ``attribute_update_handler``, drain it and call
      ``update_attributes`` per (instance, attr-dict) pair.
  - In ``_drain_pending_external`` (in addition to
      ``ReceiveInteraction`` → ``external_transition``): dispatch
      ``DiscoverObjectInstance`` to ``discover_handler`` and
      ``ReflectAttributeValues`` to ``reflect_handler``.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from typing import Any, Protocol, cast

from pyjevsim_bridge._protocol import CoupledModelProtocol
from pyjevsim_bridge.port_mapping import PortMapping
from rti1516e.connection import Federate, FederationSpec, RtiConnection
from rti1516e.events import (
    DiscoverObjectInstance,
    FederationHalted,
    ReceiveInteraction,
    ReflectAttributeValues,
    TimeAdvanceGrant,
)


class _DiscoverHandler(Protocol):
    def __call__(
        self, object_handle: int, class_name: str, instance_name: str
    ) -> None: ...


class _ReflectHandler(Protocol):
    def __call__(self, object_handle: int, values: dict[str, Any]) -> None: ...


class HLAFederate:
    """Wrap a pyjevsim CoupledModel as an HLA federate.

    Construct with a connection (either an open RtiConnection or a URL
    that the bridge will connect to itself), the coupled model, the
    federation spec, the federate name, and the port mapping.
    """

    def __init__(
        self,
        coupled_model: CoupledModelProtocol,
        federation: FederationSpec,
        federate_name: str,
        port_mapping: PortMapping,
        *,
        connection: RtiConnection | None = None,
        url: str | None = None,
        time_managed: bool = True,
        lookahead: float = 0.0,
        grant_timeout: float | None = None,
    ) -> None:
        self.coupled_model = coupled_model
        self.federation = federation
        self.federate_name = federate_name
        self.port_mapping = port_mapping
        self._connection = connection
        self._url = url
        if not math.isfinite(lookahead) or lookahead < 0:
            raise ValueError("HLAFederate lookahead must be finite and non-negative")
        if grant_timeout is not None and (
            not math.isfinite(grant_timeout) or grant_timeout <= 0
        ):
            raise ValueError("HLAFederate grant_timeout must be finite and positive")
        self._time_managed = bool(time_managed)
        self._lookahead = float(lookahead)
        self._grant_timeout = grant_timeout
        self._time_management_initialized = False
        self._advance_failed = False
        # Logical federate clock — DEVS ``now``. Advances on every
        # successful TimeAdvanceGrant.
        self._now: float = 0.0
        # Queue of externally-arrived interactions awaiting delivery to
        # the coupled model on the next step. Each entry is the FOM class
        # name + the raw payload; ``step_once`` translates the class to
        # an input port via ``port_mapping.in_port_for_fom_class``.
        self._pending_external: deque[tuple[str, Any]] = deque()
        # Queue of object-class events (DiscoverObjectInstance /
        # ReflectAttributeValues) awaiting delivery to the model's
        # ``discover_handler`` / ``reflect_handler``. Drained on the
        # same cycle as ``_pending_external`` so external-first
        # semantics apply uniformly.
        self._pending_object_events: deque[
            DiscoverObjectInstance | ReflectAttributeValues
        ] = deque()
        # Lazy-initialized SDK objects when step_once is called outside
        # of run(). Held across calls so successive step_once invocations
        # reuse the same federate.
        self._owned_connection: RtiConnection | None = None
        self._owned_federate_cm: Any | None = None
        self._federate: Federate | None = None
        self._close_lock = asyncio.Lock()
        self._close_operation: asyncio.Task[None] | None = None
        # Object-class state. Populated lazily on first
        # ``_ensure_federate`` call when the coupled model exposes
        # ``ObjectClassFederateProtocol`` methods.
        #   _instance_handles : local_name -> object handle (publisher
        #     side; populated from ``register_instances``).
        #   _subscribed_object_classes : set[str] of FOM class names
        #     the model declared via ``object_class_subscriptions``.
        #     Used to filter incoming Discover events.
        #   _object_decls_initialized : guard so publish/subscribe RPCs
        #     fire exactly once even if step_once is called many times.
        self._instance_handles: dict[str, int] = {}
        self._subscribed_object_classes: set[str] = set()
        self._object_decls_initialized: bool = False

    async def run(self, *, ticks: int) -> None:
        """Run the bridge loop for ``ticks`` time-advance cycles.

        Opens the connection (if a URL was supplied) and joins the
        federation, then drives ``step_once`` ``ticks`` times. Resigns
        and closes deterministically on exit, even if a step raises.
        """
        if self._advance_failed:
            raise RuntimeError(
                "HLAFederate: previous advance or cleanup failed; "
                "aclose must complete before run"
            )
        original: BaseException | None = None
        try:
            await self._ensure_federate()
            for _ in range(ticks):
                await self.step_once()
        except BaseException as exc:
            original = exc

        try:
            await self.aclose()
        except BaseException as cleanup_error:
            if original is None:
                raise
            raise BaseExceptionGroup(
                "HLAFederate run failed and cleanup reported errors",
                [original, cleanup_error],
            ) from original
        if original is not None:
            raise original

    async def step_once(self) -> None:
        """Run one bridge cycle. Returns when the cycle's TimeAdvanceGrant
        is fully processed. Used by tests to drive the bridge in
        single-step mode without the run() loop's stop condition.

        Cycle semantics
        ---------------
        - If external events are pending (queued by ``deliver_external``
          or by the previous cycle's grant), drain them via
          ``external_transition`` and return without an internal cycle,
          preserving DEVS external-first semantics.
        - Otherwise: read ``ta``, issue NER for ``_now + ta``, drain
          events until the matching TimeAdvanceGrant. Externals arriving
          in flight are buffered; if the grant time is strictly less
          than the requested time we treat the cycle as external and
          skip the internal phase. Otherwise we run output_handler →
          send_interaction → internal_transition.
        """
        if self._advance_failed:
            raise RuntimeError(
                "HLAFederate: previous advance failed; close and rejoin before retry"
            )
        try:
            fed = await self._ensure_federate()
        except BaseException:
            self._advance_failed = True
            raise

        # External-first: if anything is queued before this cycle, drain
        # it without asking the RTI for a grant. Production traffic for
        # an external event would arrive via fed.events() as a
        # ReceiveInteraction; deliver_external is the test seam for
        # injecting that without a real RTI roundtrip. Object-class
        # events queued via ``deliver_object_event`` participate in the
        # same external-first short-circuit.
        if self._pending_external or self._pending_object_events:
            try:
                self._drain_pending_external()
            except BaseException:
                self._advance_failed = True
                raise
            return

        try:
            ta = float(self.coupled_model.time_advance())
            if not math.isfinite(ta) or ta < 0:
                raise ValueError(
                    f"HLAFederate: model time advance must be finite and non-negative, got {ta!r}"
                )
            t_request = self._now + ta
            if not math.isfinite(t_request) or t_request < self._now:
                raise ValueError(
                    f"HLAFederate: invalid time request {t_request!r} from current {self._now!r}"
                )
        except BaseException:
            self._advance_failed = True
            raise
        try:
            await fed.next_message_request(t_request)

            if self._grant_timeout is None:
                grant = await self._await_grant(fed, t_request)
            else:
                grant = await asyncio.wait_for(
                    self._await_grant(fed, t_request),
                    timeout=self._grant_timeout,
                )
        except TimeoutError as exc:
            self._advance_failed = True
            raise TimeoutError(
                "HLAFederate: timed out waiting for TimeAdvanceGrant "
                f"at requested logical time {t_request}"
            ) from exc
        except BaseException:
            self._advance_failed = True
            raise

        try:
            grant_time = float(grant.time)
            if (
                not math.isfinite(grant_time)
                or grant_time < self._now
                or grant_time > t_request
            ):
                raise ValueError(
                    "HLAFederate: invalid TimeAdvanceGrant "
                    f"{grant_time!r}; current={self._now!r}, requested={t_request!r}"
                )
        except BaseException:
            self._advance_failed = True
            raise

        # The RTI time is committed when the grant is observed. Model-side
        # post-processing may still fail, but retrying from the old time would
        # violate the federate clock and can duplicate the NER.
        self._now = grant_time

        try:
            # If we collected externals while waiting for the grant, deliver
            # them and skip the internal cycle. Otherwise run the internal one.
            if (
                self._pending_external
                or self._pending_object_events
                or grant_time < t_request
            ):
                self._drain_pending_external()
            else:
                await self._run_internal_cycle(fed)
        except BaseException:
            self._advance_failed = True
            raise

    async def deliver_external(self, fom_class_name: str, payload: Any) -> None:
        """Test/integration hook: simulate an externally-arrived
        interaction.

        Production traffic delivers this via ``fed.events()`` as a
        ``ReceiveInteraction``; ``step_once`` collects those into the
        same internal queue this method writes to. The two paths
        converge at ``_drain_pending_external``.
        """
        self._pending_external.append((fom_class_name, payload))

    async def deliver_object_event(
        self,
        event: DiscoverObjectInstance | ReflectAttributeValues,
    ) -> None:
        """Test/integration hook: simulate an externally-arrived
        object-class event (Discover or Reflect).

        Sibling of ``deliver_external``. Production traffic delivers
        these via ``fed.events()`` and ``_await_grant`` queues them
        on the same per-instance pending queue this method writes
        to; both paths converge at ``_drain_pending_external``.
        """
        self._pending_object_events.append(event)

    # --- internal helpers ---------------------------------------------------

    async def _ensure_federate(self) -> Federate:
        """Return an open Federate, lazily joining via the configured
        connection or URL on first call.

        Lazy init lets spec tests call ``step_once`` directly without
        going through ``run()``; the resulting connection + federate
        are owned by the bridge instance and torn down by ``aclose``.
        """
        if self._federate is not None:
            await self._init_time_management(self._federate)
            await self._init_object_class_declarations(self._federate)
            return self._federate

        if self._connection is not None:
            cm = self._connection.join_federation(
                self.federation, federate_name=self.federate_name
            )
        else:
            if self._url is None:
                raise RuntimeError(
                    "HLAFederate: neither connection nor url was supplied"
                )
            self._owned_connection = await RtiConnection.connect(
                self._url
            ).__aenter__()
            cm = self._owned_connection.join_federation(
                self.federation, federate_name=self.federate_name
            )

        self._owned_federate_cm = cm
        federate = await cm.__aenter__()
        self._federate = federate
        await self._init_time_management(federate)
        # Object-class declarations + instance registrations happen
        # exactly once on the first time we acquire a federate. The
        # guard short-circuits subsequent ``_ensure_federate`` calls
        # (no-op when the lazy init already ran).
        await self._init_object_class_declarations(federate)
        return federate

    async def _init_time_management(self, fed: Federate) -> None:
        """Enable the modes required before issuing the first NER.

        The real gorti Time Manager rejects an advance from a federate that is
        neither regulating nor constrained. Earlier bridge tests used an
        in-process fake that auto-granted NER without enforcing this
        precondition, so the missing handshake was invisible until a real
        ``rtid`` was used.

        ``time_managed=False`` is an explicit compatibility escape hatch for a
        custom transport that implements its own clock discipline.
        """
        if self._time_management_initialized or not self._time_managed:
            return
        await fed.enable_time_regulation(self._lookahead)
        await fed.enable_time_constrained()
        self._time_management_initialized = True

    async def _init_object_class_declarations(self, fed: Federate) -> None:
        """Wire object-class publish/subscribe + instance registration.

        Called exactly once per ``HLAFederate`` instance, lazily when
        the underlying federate is first opened. Walks
        ``ObjectClassFederateProtocol`` methods on ``coupled_model``
        defensively (each is optional) and issues the corresponding
        SDK calls.

        Backward-compat: a model that exposes none of the methods
        leaves ``_object_decls_initialized=True`` after this call and
        every later cycle stays purely interaction-driven.
        """
        if self._object_decls_initialized:
            return
        self._object_decls_initialized = True
        model = self.coupled_model

        # Declaration — publication. {class_name: [attr_name, ...]}
        publications = self._safe_call_dict(model, "object_class_publications")
        for class_name in sorted(publications):
            attrs = list(publications[class_name])
            await fed.publish_object_class(class_name, attributes=attrs)

        # Declaration — subscription. Same shape; remember class names so
        # incoming DiscoverObjectInstance events can be filtered.
        subscriptions = self._safe_call_dict(model, "object_class_subscriptions")
        for class_name in sorted(subscriptions):
            attrs = list(subscriptions[class_name])
            await fed.subscribe_object_class(class_name, attributes=attrs)
            self._subscribed_object_classes.add(class_name)

        # Instance registration. {instance_local_name: class_name}.
        # Stash the resulting object handle keyed by local name so the
        # model can refer to its instances by friendly name in the
        # attribute_update_handler return.
        registrations = self._safe_call_dict(model, "register_instances")
        for local_name in sorted(registrations):
            class_name = registrations[local_name]
            handle = await fed.register_object_instance(
                class_name, instance_name=local_name
            )
            self._instance_handles[local_name] = handle

    @staticmethod
    def _safe_call_dict(model: Any, attr: str) -> dict[str, Any]:
        """Call ``model.attr()`` if present and the return is a dict;
        otherwise return ``{}``.

        Duck-typed dispatch — keeps the interaction-only path working
        when the model implements none of the object-class methods.
        Returns an empty dict for any non-dict / falsy return so the
        caller can iterate uniformly.
        """
        fn = getattr(model, attr, None)
        if not callable(fn):
            return {}
        result = fn()
        if not isinstance(result, dict):
            return {}
        return result

    async def register_instance(
        self,
        class_name: str,
        instance_local_name: str,
    ) -> int:
        """Runtime hook for registering an instance after startup.

        Models that need to register instances dynamically (mid-run,
        in response to some external trigger) can call this from
        their ``external_transition`` or via a separate orchestration
        pass. Stores the handle in the same local-name → handle map
        used by ``attribute_update_handler`` dispatch.

        Returns the new object handle (so the caller can use it
        directly for raw ``Federate.update_attributes`` calls if
        needed).
        """
        fed = await self._ensure_federate()
        handle = await fed.register_object_instance(
            class_name, instance_name=instance_local_name
        )
        self._instance_handles[instance_local_name] = handle
        return int(handle)

    async def _await_grant(
        self, fed: Federate, t_request: float
    ) -> TimeAdvanceGrant:
        """Drain ``fed.events()`` until a TimeAdvanceGrant arrives.

        ReceiveInteractions encountered along the way are translated
        to (port_name, payload) entries appended to the pending
        external queue, where ``step_once`` will hand them to
        ``external_transition`` in arrival order.

        ``t_request`` is unused by the cut-1 dispatch (the FakeRtiServer
        auto-grants at exactly t_request) but is part of the signature
        so a future implementation can detect "grant earlier than
        requested" without re-deriving it.
        """
        del t_request  # reserved for future scheduler-aware behavior
        async for event in fed.events():
            if isinstance(event, TimeAdvanceGrant):
                return event
            if isinstance(event, FederationHalted):
                raise RuntimeError(
                    "HLAFederate: federation halted while waiting for grant "
                    f"(cause={event.cause!r}, "
                    f"stalled_federate_handle={event.stalled_federate_handle})"
                )
            if isinstance(event, ReceiveInteraction):
                self._pending_external.append((event.class_name, event.parameters))
                continue
            if isinstance(event, (DiscoverObjectInstance, ReflectAttributeValues)):
                # Object-class events are routed via the parallel
                # _pending_object_events queue and drained alongside
                # ReceiveInteraction events in _drain_pending_external.
                self._pending_object_events.append(event)
                continue
            # Other event kinds, such as FederationHalted, have no
            # coupled-model delivery hook and are ignored here.
        # The async-for can only exit via return inside the loop; the
        # explicit raise here is for type-checker coverage of the "loop
        # exited without yielding" branch.
        raise RuntimeError("HLAFederate: events() exhausted before grant")

    def _drain_pending_external(self) -> None:
        """Hand every queued external event to the matching handler.

        Three event flavors flow through here:

          * ReceiveInteraction (queued in ``_pending_external``) —
            dispatched to ``coupled_model.external_transition(port,
            payload)`` after resolving the input port from the FOM
            class via ``port_mapping``. Events whose FOM class is not
            present in the port mapping are silently dropped — the
            bridge can subscribe to interactions the coupled model
            hasn't bound a port to.

          * DiscoverObjectInstance (queued in
            ``_pending_object_events``) — dispatched to
            ``coupled_model.discover_handler`` if present and the
            event's class is in the model's
            ``object_class_subscriptions`` declarations. Unsubscribed
            classes are dropped on the same rationale.

          * ReflectAttributeValues (queued in
            ``_pending_object_events``) — dispatched to
            ``coupled_model.reflect_handler`` if present. The handler
            sees the raw attribute dict; decoding is the model's
            responsibility (the bridge's
            interaction path treats payloads as opaque too).

        Order: interactions are drained first (preserving the cut-1
        contract), then object-class events. Within each queue,
        FIFO arrival order is preserved.
        """
        while self._pending_external:
            fom_class, payload = self._pending_external.popleft()
            port = self.port_mapping.in_port_for_fom_class(fom_class)
            if port is None:
                continue
            self.coupled_model.external_transition(port, payload)

        # Object-class events. Drained AFTER interactions so the
        # interaction-only contract (which is the dominant case)
        # observes no ordering change.
        while self._pending_object_events:
            event = self._pending_object_events.popleft()
            self._dispatch_object_event(event)

    def _dispatch_object_event(
        self,
        event: DiscoverObjectInstance | ReflectAttributeValues,
    ) -> None:
        """Route one object-class event to the coupled model.

        Uses ``hasattr`` to check for the optional handler methods.
        Models that opt into the subscription path but don't
        implement a handler get the event dropped silently — same
        as the interaction-class "unmapped class" case.
        """
        model = self.coupled_model
        if isinstance(event, DiscoverObjectInstance):
            # Only dispatch discovers for classes the model declared
            # a subscription for. This mirrors the interaction-side
            # filter (unmapped FOM class → drop) and prevents leakage
            # when a single transport hosts multiple federates that
            # happen to share an event queue (the InProcessTransport
            # is per-handle so this is defensive in practice).
            if event.class_name not in self._subscribed_object_classes:
                return
            handler = getattr(model, "discover_handler", None)
            if callable(handler):
                cast("_DiscoverHandler", handler)(
                    event.object_handle,
                    event.class_name,
                    event.instance_name,
                )
            return
        if isinstance(event, ReflectAttributeValues):
            handler = getattr(model, "reflect_handler", None)
            if callable(handler):
                cast("_ReflectHandler", handler)(
                    event.object_handle, dict(event.values)
                )
            return

    async def _run_internal_cycle(self, fed: Federate) -> None:
        """Output → send_interaction (+ optional update_attributes) →
        internal_transition.

        Outputs whose port is not bound to a FOM class in the mapping
        are skipped — same rationale as ``_drain_pending_external``.
        Payloads are passed through opaquely under a ``_payload`` key;
        this bridge does not perform FOM-driven payload encoding.

        Object-class extension: if the model exposes
        ``attribute_update_handler``, drain it after the interaction
        path (deterministic order: interactions first, then attribute
        updates) and call ``Federate.update_attributes`` for each
        ``{instance_local_name: {attr: payload}}`` pair. Instances
        not in the bridge's local-name → handle map are silently
        skipped (matches the unmapped-port convention).
        """
        outputs = self.coupled_model.output_handler()
        # Sort port names before iterating so the wire-visible
        # send_interaction order is deterministic regardless of how
        # the user model assembled its output dict. Mirrors Go's D-2
        # discipline (sort before iterate over maps). M5-audit issue #2.
        for port_name in sorted(outputs):
            payload = outputs[port_name]
            class_name = self.port_mapping.fom_class_for_out_port(port_name)
            if class_name is None:
                continue
            await fed.send_interaction(
                class_name,
                parameters={"_payload": payload},
            )

        # Object-class outgoing path. Optional — the call is no-op for
        # models that don't expose ``attribute_update_handler``.
        await self._run_attribute_update_cycle(fed)

        self.coupled_model.internal_transition()

    async def _run_attribute_update_cycle(self, fed: Federate) -> None:
        """Drain ``attribute_update_handler`` and dispatch
        ``update_attributes`` calls.

        Sibling to the interaction-output path in ``_run_internal_cycle``.
        Iterates instances and attribute names in sorted order for
        wire-determinism (mirrors the
        ``sort-before-iterate-over-maps`` discipline). Unknown
        instance names are silently dropped — same convention as
        unmapped output ports.
        """
        updates = self._safe_call_dict(self.coupled_model, "attribute_update_handler")
        if not updates:
            return
        for local_name in sorted(updates):
            attrs = updates[local_name]
            if not isinstance(attrs, dict) or not attrs:
                continue
            handle = self._instance_handles.get(local_name)
            if handle is None:
                continue
            # Sort attribute names too so update_attributes payloads are
            # iterable in deterministic order downstream.
            ordered_attrs = {k: attrs[k] for k in sorted(attrs)}
            await fed.update_attributes(handle, ordered_attrs)

    async def aclose(self) -> None:
        """Tear down lazily-acquired connection/federate. Safe to call
        multiple times; no-op when nothing is owned.

        Tests that exercise step_once directly do not need to call this
        — the FakeRtiServer-backed transport is purely in-process and
        leaks no resources. The method is provided for symmetry with
        the ``run()`` lifecycle and caller-managed bridge runners.
        """
        async with self._close_lock:
            if (
                self._owned_federate_cm is None
                and self._owned_connection is None
                and self._close_operation is None
            ):
                self._federate = None
                self._clear_session_state()
                return
            operation = self._close_operation
            if operation is None:
                operation = asyncio.create_task(self._aclose_once())
                self._close_operation = operation
        try:
            # Cleanup is shared, but cancellation belongs to each caller.
            # Shielding prevents one waiter from cancelling the operation and
            # lets CancelledError retain its identity on that waiter.
            await asyncio.shield(operation)
        finally:
            if operation.done():
                async with self._close_lock:
                    if self._close_operation is operation:
                        self._close_operation = None

    async def _aclose_once(self) -> None:
        cm = self._owned_federate_cm
        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
            except asyncio.CancelledError:
                self._advance_failed = True
                raise
            except BaseException as exc:
                # The connection is a dependency of resign. Do not close it
                # after a failed federate exit: doing so would make a real
                # retry indistinguishable from a false success.
                self._advance_failed = True
                raise BaseExceptionGroup(
                    "HLAFederate federate cleanup failed", [exc]
                ) from exc
            else:
                self._owned_federate_cm = None
                self._federate = None
        else:
            self._federate = None
        owned = self._owned_connection
        if owned is not None:
            try:
                await owned.__aexit__(None, None, None)
            except asyncio.CancelledError:
                self._advance_failed = True
                raise
            except BaseException as exc:
                self._advance_failed = True
                raise BaseExceptionGroup(
                    "HLAFederate connection cleanup failed", [exc]
                ) from exc
            else:
                self._owned_connection = None
        self._clear_session_state()

    def _clear_session_state(self) -> None:
        """Discard state scoped to one joined federation session."""
        self._time_management_initialized = False
        self._object_decls_initialized = False
        self._advance_failed = False
        self._instance_handles.clear()
        self._subscribed_object_classes.clear()
        self._pending_external.clear()
        self._pending_object_events.clear()
        self._now = 0.0
