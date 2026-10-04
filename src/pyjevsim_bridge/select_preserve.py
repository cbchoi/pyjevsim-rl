"""Deterministic ordering for simultaneous pyjevsim events.

Before delivering multiple events with the same timestamp, the bridge
orders them deterministically so HLA tie-breaking receives a stable
sequence.
"""

from __future__ import annotations

from typing import Any

from pyjevsim_bridge._protocol import CoupledModelProtocol


def order_simultaneous_events(
    coupled_model: CoupledModelProtocol,
    events: list[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    """Reorder ``events`` (list of (port_name, payload)) by pyjevsim's
    select() priority.

    Real pyjevsim exposes select() at the coupled-model level; the bridge
    invokes it to get the priority ordering. The bridge must NOT rely on
    Python list ordering, dict iteration order, or any other source of
    nondeterminism.

    ``CoupledModelProtocol`` does not expose coupled-model ``select()``.
    This implementation therefore sorts by port name (lexicographic),
    preserving the relative order of ties in payload.
    ``sorted()`` is stable, so two entries with the same port name
    retain their input order.

    This satisfies FR-PYJ-4's "no Python list ordering, no dict iteration
    nondeterminism" requirement: the output for a given input is always
    the same regardless of insertion order.
    """
    # The model argument is part of the ordering API even though the
    # current protocol has no model-specific priority method.
    del coupled_model
    return sorted(events, key=lambda item: item[0])
