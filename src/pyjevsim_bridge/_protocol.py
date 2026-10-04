"""typing.Protocol shims for the bridge's coupled-model contract.

This Protocol describes what the BRIDGE needs from a coupled-model-shaped
object — it uses DEVS-canonical names (`time_advance`, `output_handler`,
`internal_transition`, `external_transition`) for clarity.

Spec tests under pysdk/tests/spec/m4/ consume this Protocol via the
StubCoupledModel test double (avoids a hard pyjevsim runtime dep).

Real pyjevsim (1.3.x and 2.0.x — API-compatible at every surface the
bridge touches) exports `StructuralModel` (coupled) + `BehaviorModel`
(atomic) with DIFFERENT method names than this Protocol:

  pyjevsim                this Protocol
  --------                -------------
  output()                output_handler()
  int_trans()             internal_transition()
  ext_trans(input_ports)  external_transition(port, payload)
  (computed via SysExecutor) → time_advance() returns ta

Real pyjevsim integrations use an adapter that maps the
``SysExecutor`` and ``StructuralModel`` surface to this protocol. The
protocol also permits lightweight coupled-model implementations without
a hard pyjevsim runtime dependency.

The smoke test ``test_spec_m4_pyjevsim_smoke.py`` confirms the
required real-pyjevsim symbols (StructuralModel, BehaviorModel) are
importable; that's the only direct dependency on the upstream API.

Object-class extension (M12 follow-up)
--------------------------------------
The base ``CoupledModelProtocol`` covers the INTERACTION half of HLA —
``output_handler`` returns ``{port: payload}`` which the bridge maps to
``send_interaction`` via ``PortMapping``. The OBJECT-CLASS half
(``register_object_instance`` / ``update_attributes`` /
``ReflectAttributeValues``) is exposed via a SIBLING protocol —
``ObjectClassFederateProtocol`` below — that models opt INTO. The
bridge detects which methods are present (duck-typed via ``hasattr``)
and dispatches accordingly:

  - if the model exposes object-class methods, the bridge calls
    ``publish_object_class`` / ``subscribe_object_class`` /
    ``register_object_instance`` on startup, drains
    ``attribute_update_handler`` each internal cycle (similarly to
    ``output_handler`` but routed via ``update_attributes``), and
    delivers ``DiscoverObjectInstance`` / ``ReflectAttributeValues``
    events through ``discover_handler`` / ``reflect_handler`` hooks.
  - models that DON'T implement these continue on the interaction-
    only path (back-compat is the dominant case).

The two protocols are independent — a model may implement either or
both. Methods are duck-typed (``hasattr``-checked at runtime), not
type-checked; that keeps the existing 5 example federates working
without changes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CoupledModelProtocol(Protocol):
    """Subset of pyjevsim.CoupledModel the bridge requires.

    Real pyjevsim exposes more — the bridge only reaches for these.
    """

    def time_advance(self) -> float:
        """DEVS ``ta`` — when the next internal event fires (seconds)."""
        ...

    def output_handler(self) -> dict[str, Any]:
        """Returns a dict of {output_port_name: payload} drained on
        internal-transition cycle. The bridge maps each port to its
        corresponding FOM interaction class via PortMapping."""
        ...

    def internal_transition(self) -> None:
        """Advance internal state after output_handler."""
        ...

    def external_transition(self, port: str, payload: Any) -> None:
        """Apply an externally-arrived event."""
        ...


@runtime_checkable
class ObjectClassFederateProtocol(Protocol):
    """Optional sibling protocol for federate models that participate in
    HLA OBJECT-CLASS pub/sub (the other half of HLA, distinct from
    interactions).

    A coupled model that implements ANY of these methods opts INTO the
    object-class path. The bridge detects each method via ``hasattr``
    (so partial implementations are legal — e.g. a pure subscriber
    that needs only ``object_class_subscriptions`` +
    ``discover_handler`` + ``reflect_handler``, and a pure publisher
    that needs only ``object_class_publications`` + ``register_instances``
    + ``attribute_update_handler``).

    The interaction-only path (``CoupledModelProtocol`` above) stays the
    dominant case; the bridge keeps wiring it whether or not these
    methods are present.

    Lifecycle / dispatch
    --------------------

    1. **Declaration** (once at startup, on first ``step_once`` /
       ``run`` call). For every entry in ``object_class_publications``,
       the bridge calls ``Federate.publish_object_class``. For every
       entry in ``object_class_subscriptions``, it calls
       ``Federate.subscribe_object_class``.

    2. **Instance registration** (once at startup). For every entry in
       ``register_instances``, the bridge calls
       ``Federate.register_object_instance`` and stores the handle in
       a local-name → handle map so the model can refer to its
       instances by friendly name in
       ``attribute_update_handler``.

    3. **Per-cycle** (in ``_run_internal_cycle``). The bridge calls
       ``attribute_update_handler`` and translates each
       ``{instance_local_name: {attr: payload}}`` entry to
       ``Federate.update_attributes(handle, {attr: payload})``.

    4. **External delivery** (in ``_drain_pending_external``). When
       the bridge sees a ``DiscoverObjectInstance`` event for a class
       this model subscribes to, it calls ``discover_handler``. When
       it sees a ``ReflectAttributeValues`` event, it calls
       ``reflect_handler``. The interaction-class path
       (``external_transition`` via ``port_mapping``) keeps running in
       parallel for ``ReceiveInteraction`` events.

    All methods are optional — ``hasattr`` guards each call site. A
    model that implements only ``object_class_subscriptions`` +
    ``reflect_handler`` is a pure subscriber; one that implements only
    ``object_class_publications`` + ``register_instances`` +
    ``attribute_update_handler`` is a pure publisher.
    """

    def object_class_publications(self) -> dict[str, list[str]]:
        """Return ``{class_name: [attr_name, ...]}`` for the object
        classes this federate publishes. Called once at startup; the
        bridge issues ``publish_object_class`` for each entry."""
        ...

    def object_class_subscriptions(self) -> dict[str, list[str]]:
        """Return ``{class_name: [attr_name, ...]}`` for the object
        classes this federate subscribes to. Called once at startup;
        the bridge issues ``subscribe_object_class`` for each entry
        and remembers the class names so it can route discover/reflect
        events back to the model."""
        ...

    def register_instances(self) -> dict[str, str]:
        """Return ``{instance_local_name: class_name}`` for instances
        this model wants to register at startup. The bridge calls
        ``register_object_instance(class_name, instance_name=local_name)``
        for each entry and stores the resulting object handle in a
        local-name → handle map. Subsequent
        ``attribute_update_handler`` returns refer to instances by
        their ``instance_local_name`` and the bridge dispatches to the
        right handle.

        Models that register instances dynamically (mid-run) can
        return an empty dict here and use the bridge's
        ``register_instance`` runtime hook instead (see
        ``HLAFederate.register_instance``)."""
        ...

    def attribute_update_handler(self) -> dict[str, dict[str, Any]]:
        """Return ``{instance_local_name: {attr_name: payload}}`` for
        outgoing attribute updates this cycle. Called by the bridge
        every internal cycle (after ``output_handler`` for the
        interaction path, before ``internal_transition``).

        Empty dict means "no updates this cycle". Instances not in
        the bridge's local-name → handle map are silently skipped
        (matches the bridge's "unmapped output port" convention)."""
        ...

    def discover_handler(
        self,
        instance_handle: int,
        class_name: str,
        instance_name: str,
    ) -> None:
        """Called when the bridge sees a ``DiscoverObjectInstance``
        event for a class this model subscribes to. Receives the
        object handle, FOM class name, and instance name.

        The bridge stores the handle internally so subsequent
        reflect events can be routed; the handler is purely a
        notification hook for the model to update its own state."""
        ...

    def reflect_handler(
        self,
        instance_handle: int,
        attrs: dict[str, Any],
    ) -> None:
        """Called when the bridge sees a ``ReflectAttributeValues``
        event for an instance the model has been told about via
        ``discover_handler`` (or, lazily, an instance whose handle
        appeared in a reflect without a prior discover — the bridge
        forwards either way)."""
        ...
