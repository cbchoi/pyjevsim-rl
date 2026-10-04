"""InProcessTransport — pure-Python in-process implementation of the RTI
client surface.

This module is the production-suitable extraction of the test double that
historically lived in ``pysdk/tests/spec/m4/_fakes/fake_rti_server.py``.
The class is the same thing wearing a name that signals intent: this is
not "a fake for tests", it is "a real in-process driver" that production
applications can wire when they want a single-process RTI without
spawning the rtid binary or paying gRPC overhead.

Use cases:

  - **Examples** (e.g. ``examples/pyjevsim/runner.py``) — drive a small
    federation in one process for demos, tutorials, or determinism
    harnesses without a live ``rtid``.
  - **Embedding** — applications that want HLA semantics inside a single
    Python process without shipping a Go binary.
  - **Tests** — the historical FakeRtiServer location now re-exports
    this class for back-compat (see ``pysdk/tests/spec/m4/_fakes/
    fake_rti_server.py``).

Surface contract (the duck-typed interface ``RtiConnection`` reaches for):

  - ``record(method, **kwargs) -> Any`` — synchronous; the SDK awaits
    awaitables via ``inspect.isawaitable``, so a sync record is fine.
  - ``events_for(handle) -> asyncio.Queue`` — per-federate event queue.
  - ``allocate_handle() -> int`` — monotonic local handle allocator.

The class auto-registers under ``memory://<name>`` (default
``memory://fake-rti`` for back-compat with the existing M4 test suite).
Callers may pass ``url=`` to choose a different memory URL when running
multiple isolated drivers in one process (e.g. one per test fixture).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


class _EventQueue(asyncio.Queue[Any]):
    """Keep cross-thread producers on the event reader's asyncio loop."""

    def __init__(self) -> None:
        super().__init__()
        self._reader_loop: asyncio.AbstractEventLoop | None = None

    async def get(self) -> Any:
        self._reader_loop = asyncio.get_running_loop()
        return await super().get()

    def put_nowait(self, item: Any) -> None:
        loop = self._reader_loop
        if loop is not None and loop.is_running():
            try:
                producer_loop = asyncio.get_running_loop()
            except RuntimeError:
                producer_loop = None
            if producer_loop is not loop:
                loop.call_soon_threadsafe(super().put_nowait, item)
                return
        super().put_nowait(item)


@dataclass
class RecordedCall:
    """One captured call from the SDK to the in-process driver.

    Identical shape to the historical ``FakeRtiServer.RecordedCall`` —
    examples and tests that already inspect ``.method`` / ``.args`` keep
    working unchanged.
    """

    method: str
    args: dict[str, Any] = field(default_factory=dict)


class InProcessTransport:
    """Pure-Python in-process driver for the RTI client surface.

    Production-suitable extraction of the historical FakeRtiServer.
    Records every call (so determinism harnesses can fingerprint them),
    exposes per-federate event queues (so event producers such as the
    producer/consumer fan-out in ``examples/pyjevsim/runner.py`` can push
    canned events back into a federate's stream), and supports
    optional canned responses keyed by method name.

    Usage::

        srv = InProcessTransport()  # auto-registers under memory://fake-rti
        async with RtiConnection.connect("memory://fake-rti") as rti:
            ...

    Pass ``url=`` to register under a different memory URL when running
    multiple drivers concurrently::

        srv = InProcessTransport(url="memory://demo-1")
    """

    def __init__(self, *, url: str = "memory://fake-rti") -> None:
        self.calls: list[RecordedCall] = []
        # Per-federate-handle event queues. Producers push events and the
        # SDK drains them via its events() async iterator.
        self.event_queues: dict[int, asyncio.Queue[Any]] = {}
        # Optional canned responses keyed by method name. If a method has
        # an entry, the recorded call returns it (or raises if it's an
        # Exception). Lets harnesses inject failure modes.
        self.canned_responses: dict[str, Any] = {}
        # Federation state (name -> {"federates": [..]}).
        self.federations: dict[str, dict[str, Any]] = {}
        # Monotonic handle allocator.
        self._next_handle = 1
        self._url = url
        # Auto-register under the canonical in-process URL so callers
        # that do `await RtiConnection.connect("memory://fake-rti")`
        # find this driver without extra wiring. Last-writer-wins;
        # callers construct one driver per scope, so no contention in
        # practice.
        from rti1516e._transport import register_fake

        register_fake(url, self)

    # --- Recording surface called by the SDK transport layer ----------------

    def record(self, method: str, **kwargs: Any) -> Any:
        """Record a call and return its canned response (or None).

        If ``canned_responses[method]`` is an Exception, raise it.
        Otherwise return the value (or None if no canned response is set).

        Convenience: when ``method == "next_message_request"`` and no
        canned response is set, synthesize a ``TimeAdvanceGrant(time)``
        event into the federate's queue. Production rtid would dispatch
        through TimeService; the in-process driver has no scheduler, so
        the simplest "good enough" semantics for bridge tests is to grant
        immediately at the requested time. Callers that want to suppress
        the auto-grant can register an explicit canned response for
        ``next_message_request`` (any non-Exception value disables the
        synthesis).
        """
        self.calls.append(RecordedCall(method=method, args=dict(kwargs)))
        if method in self.canned_responses:
            response = self.canned_responses[method]
            if isinstance(response, Exception):
                raise response
            return response
        if method == "next_message_request":
            # Lazy import to avoid circular dependency at module import time.
            from rti1516e.events import TimeAdvanceGrant

            grant = TimeAdvanceGrant(time=float(kwargs.get("time", 0.0)))
            target = kwargs.get("federate_handle")
            if target is not None:
                self.events_for(int(target)).put_nowait(grant)
            else:
                # Defensive: broadcast if no federate handle supplied.
                for queue in self.event_queues.values():
                    queue.put_nowait(grant)
        return None

    # --- Event injection surface --------------------------------------------

    def push_event(self, federate_handle: int, event: Any) -> None:
        """Push an event into the federate's event stream.

        The SDK calls ``events_for(handle)`` and yields from the resulting
        queue. Pushing here makes the event visible to the next
        ``async for`` iteration in the federate.
        """
        self.events_for(federate_handle).put_nowait(event)

    def events_for(self, federate_handle: int) -> asyncio.Queue[Any]:
        """Return the per-federate event queue (allocates on first use)."""
        return self.event_queues.setdefault(federate_handle, _EventQueue())

    def allocate_handle(self) -> int:
        """Mint a fresh monotonic handle. Used for register_object_instance,
        join_federation, etc."""
        h = self._next_handle
        self._next_handle += 1
        return h

    # --- Convenience inspection --------------------------------------------

    def calls_for(self, method: str) -> list[RecordedCall]:
        """Return all recorded calls matching ``method``."""
        return [c for c in self.calls if c.method == method]

    def reset(self) -> None:
        """Clear all recorded state. Useful between scenarios."""
        self.calls.clear()
        self.event_queues.clear()
        self.canned_responses.clear()
        self.federations.clear()
        self._next_handle = 1
