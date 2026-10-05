"""Value-field factoring of the existing inventory case, not a new domain.

This separate profile retains the original adapter's construction, topology,
composition checks, clock ownership, rebind and reward obligations. Pending and
event lists are exclusive stock-owned values; shared config and clocks are not.
The original research adapter and its archived experiments remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from pyjevsim_bridge.rl.continuation import state_fields
from pyjevsim_bridge.rl.continuation.contracts import fail
from pyjevsim_bridge.rl.continuation.generic_boundary import DeclaredFixedDeltaBoundaryProvider
from pyjevsim_bridge.rl.continuation.generic_bundle import make_declared_bundle
from pyjevsim_bridge.rl.continuation.registry import SourceBinding
from pyjevsim_bridge.rl.continuation.state_fields import DeclaredValueFields, OwnedValueField

from . import inventory as m
from . import inventory_maintenance as extension
from .inventory_adapter import InventoryAdapter, SCHEMA


def _nonnegative_integer(value):
    m.integer(value, "owned inventory count")


def _finite_nonnegative(value):
    m.number(value, "owned inventory penalty")


def _list(value):
    if type(value) is not list:
        fail("owned inventory sequence must be a list")


_SOURCE_FIELDS = DeclaredValueFields((OwnedValueField("cursor", _nonnegative_integer),))
_STOCK_FIELDS = {
    version: DeclaredValueFields(tuple(
        OwnedValueField(name, _list if name in ("pending", "events") else
                        _finite_nonnegative if name == "cumulative_penalty" else _nonnegative_integer)
        for name in m.stock_type(version).DOMAIN_FIELDS))
    for version in (1, 2)
}
_EXTRA_BINDINGS = tuple(
    SourceBinding(logical, str(Path(path)), hashlib.sha256(Path(path).read_bytes()).hexdigest())
    for logical, path in (("continuation.state_fields", state_fields.__file__),
                          ("research.declared_inventory_adapter", __file__)))


class DeclaredInventoryAdapter(InventoryAdapter):
    def __init__(self, version):
        super().__init__(version)
        self.source_bindings += _EXTRA_BINDINGS

    def export_state(self, graph, refs):
        return {"construction": {"schema_id": SCHEMA, "config": copy.deepcopy(graph.config), "seed": graph.seed},
                "values": {"dc": {}, "demand-source": _SOURCE_FIELDS.capture(graph.source),
                           "inventory-stock": _STOCK_FIELDS[self.version].capture(graph.stock)}}

    def restore_into(self, graph, state):
        # Cross-object/composition validation remains in the original adapter
        # and coordinator. This helper handles only local value replacement.
        source = state["values"]["demand-source"]
        stock = state["values"]["inventory-stock"]
        _SOURCE_FIELDS.validate(source)
        _STOCK_FIELDS[self.version].validate(stock)
        _SOURCE_FIELDS.restore_into(graph.source, source)
        _STOCK_FIELDS[self.version].restore_into(graph.stock, stock)


def make_bundle(version):
    profile_id = f"declared-inventory-C1-v{version}"
    adapter = DeclaredInventoryAdapter(version)
    boundary = DeclaredFixedDeltaBoundaryProvider(profile_id=profile_id,
        schema_id=f"inventory-boundary-v{version}", observation_validator=m.validate_observation,
        reward_validator=m.validate_reward_v1 if version == 1 else extension.validate_reward_v2,
        initial_reward_state=m.initial_reward(version), source_bindings=adapter.source_bindings)
    return make_declared_bundle(profile_id=profile_id, model=adapter, boundary=boundary,
        model_provider_id=f"declared-inventory-adapter-v{version}",
        projection_id=f"inventory-observation-v{version}",
        capabilities=("static-flat", "fixed-delta", "deterministic-demand", "pending-replenishment",
                      "inherited-confluence", "restored-time-action", "declared-exclusive-value-fields"))
