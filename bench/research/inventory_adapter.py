"""Inventory declarations over the existing continuation core, unchanged."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from weakref import WeakKeyDictionary

from pyjevsim_bridge.rl.continuation.contracts import (
    ContinuationError, StateObligation, Violation, canonical_bytes, exact_fields, fail, thaw_value)
from pyjevsim_bridge.rl.continuation.generic_boundary import (
    DeclaredFixedDeltaBoundaryProvider, validate_policy_context, validate_sampling_context)
from pyjevsim_bridge.rl.continuation.generic_bundle import make_declared_bundle
from pyjevsim_bridge.rl.continuation.registry import SourceBinding

from . import inventory as m
from . import inventory_maintenance as extension

SCHEMA = "inventory-construction-v1"
_COMMON = {"_SystemObject__object_id", "model_type", "_name", "external_input_ports", "external_output_ports"}
_BEHAVIOR = _COMMON | {"_states", "_cur_state", "global_time", "_cancel_reschedule_f",
    "external_transition_map_tuple", "external_transition_map_state", "internal_transition_map_tuple", "internal_transition_map_state"}
_SHAPES = {m.InventoryGraph: _COMMON | {"model_map", "port_map", "config", "seed", "source", "stock"},
           m.DemandSource: _BEHAVIOR | {"config", "clock", "cursor"},
           m.InventoryStock: _BEHAVIOR | {"config", "clock", *m.InventoryStock.DOMAIN_FIELDS},
           extension.PenaltyStock: _BEHAVIOR | {"config", "clock", *extension.PenaltyStock.DOMAIN_FIELDS}}
_METHODS = {(cls, name): (getattr(cls, name), getattr(getattr(cls, name), "__code__", None))
            for cls in _SHAPES for name in dir(cls) if callable(getattr(cls, name))
            and (not name.startswith("__") or name == "__init__")}
_IMPORT_HASHES = {Path(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                  for path in (__file__, m.__file__, extension.__file__)}


def topology():
    def node(kind, inputs, outputs, schema="inventory-model-v1"):
        return {"type_id": kind, "schema_id": schema, "inputs": inputs, "outputs": outputs}
    return {"nodes": {
        "inventory": node("pyjevsim.structural", ["action"], [], "inventory-graph-v1"),
        "demand-source": node("inventory.demand", [], ["demand"]),
        "inventory-stock": node("inventory.stock", ["action", "demand"], []),
        "dc": node("pyjevsim.default-message-catcher", ["uncaught"], [], "pyjevsim-default-catcher-v1")},
        "couplings": [{"source_node": "inventory", "source_port": "action",
                       "target_node": "inventory-stock", "target_port": "action"},
                      {"source_node": "demand-source", "source_port": "demand",
                       "target_node": "inventory-stock", "target_port": "demand"}],
        "shared_resources": {}, "aliases": []}


def projection(state, now):
    return {"version": state["construction"]["config"]["version"], "logical_time": float(now),
            **copy.deepcopy(state["values"]["inventory-stock"]),
            "demand_cursor": state["values"]["demand-source"]["cursor"]}


class InventoryAdapter:
    def __init__(self, version):
        if version not in (1, 2):
            raise ValueError("unsupported inventory version")
        self.version = self._version = version
        self.source_bindings = tuple(SourceBinding(f"research.{path.stem}", str(path), sha)
                                     for path, sha in _IMPORT_HASHES.items())
        self._clock_owners = WeakKeyDictionary()

    def verify_identity(self):
        if self.version != self._version:
            fail("inventory declaration changed", "CC_INCOMPATIBLE_IDENTITY")
        for (cls, name), (method, code) in _METHODS.items():
            if getattr(cls, name) is not method or getattr(method, "__code__", None) is not code:
                fail("inventory implementation changed", "CC_INCOMPATIBLE_IDENTITY")

    def descriptor(self):
        return {"schema_id": SCHEMA, "type_id": f"inventory-v{self.version}"}

    def topology(self):
        return topology()

    def obligations(self):
        rows = [("01", "domain-values", "stock plus received minus fulfilled and future order conservation"),
                ("02", "demand-cursor", "processed demand prefix matches committed clock"),
                ("03", "pending-orders", "absolute replenishment deadlines match native scheduler"),
                ("04", "configuration-clock", "candidate config aliases and clocks are rebound"),
                ("05", "reward-cache", "incremental reward baseline equals committed accumulators")]
        return tuple(StateObligation(f"OB-I-{key}", f"inventory.{name}", "model",
            ("InventoryGraph.observe",), ("InventoryStock.ext_trans/int_trans",),
            restore_rule="explicit field restoration without transition or reset", invariant=invariant,
            counterexample="tests.test_research_transfer", test_ids=("RES005", "RES006"),
            evidence_status="reviewed") for key, name, invariant in rows)

    def _construction(self, raw):
        exact_fields(raw, {"schema_id", "config", "seed"}, "inventory construction")
        if raw["schema_id"] != SCHEMA or raw["config"]["version"] != self.version:
            fail("inventory construction/version differs", "CC_INCOMPATIBLE_IDENTITY")
        if canonical_bytes(m.validate_config(raw["config"])) != canonical_bytes(raw["config"]):
            fail("noncanonical inventory configuration")
        m.integer(raw["seed"], "seed")
        return raw

    def construction_from_request(self, request):
        return self._construction({"schema_id": SCHEMA,
            "config": m.validate_config(thaw_value(request.model_config)), "seed": request.seed})

    def allocate_shell(self, descriptor, services, cleanup):
        row = self._construction(copy.deepcopy(descriptor))
        graph = m.InventoryGraph(row["config"], row["seed"], services.clock, m.stock_type(self.version))
        self._clock_owners[graph] = (services, services.clock.__func__)
        return graph

    def reference_objects(self, graph):
        return {"inventory": graph, "demand-source": graph.source, "inventory-stock": graph.stock}

    def export_state(self, graph, refs):
        return {"construction": {"schema_id": SCHEMA, "config": copy.deepcopy(graph.config), "seed": graph.seed},
                "values": {"dc": {}, "demand-source": {"cursor": graph.source.cursor},
                           "inventory-stock": graph.stock.domain_state()}}

    def inspect(self, graph):
        try:
            self.verify_identity()
            if type(graph) is not m.InventoryGraph or type(graph.stock) is not m.stock_type(self.version):
                fail("inventory graph/stock type differs")
            for obj in (graph, *graph.leaves()):
                if set(vars(obj)) != _SHAPES[type(obj)]:
                    fail("inventory undeclared field", "CC_UNSUPPORTED_PROFILE")
            owner = self._clock_owners.get(graph)
            for leaf in graph.leaves():
                if leaf.config is not graph.config or owner != (getattr(leaf.clock, "__self__", None),
                                                               getattr(leaf.clock, "__func__", None)):
                    fail("inventory config/clock ownership differs")
            refs, expected = self.reference_objects(graph), {}
            for row in topology()["couplings"]:
                expected.setdefault((refs[row["source_node"]], row["source_port"]), []).append(
                    (refs[row["target_node"]], row["target_port"]))
            if graph.port_map != expected or graph.model_map != {leaf.get_name(): leaf for leaf in graph.leaves()}:
                fail("inventory model topology differs")
            for name, obj in refs.items():
                node = topology()["nodes"][name]
                if obj.get_name() != name or obj.external_input_ports != node["inputs"] or obj.external_output_ports != node["outputs"]:
                    fail("inventory declared ports differ")
            self.validate_payload(self.export_state(graph, None), topology(), None)
        except (ContinuationError, ValueError, TypeError, KeyError, AttributeError) as exc:
            return (Violation(getattr(exc, "code", "CC_INVALID_PAYLOAD"), "model.inspect", "inventory", message=str(exc)),)
        return ()

    def validate_payload(self, state, supplied_topology, profile):
        exact_fields(state, {"construction", "values"}, "inventory state")
        self._construction(state["construction"])
        if supplied_topology != topology():
            fail("inventory topology differs")
        rows = exact_fields(state["values"], {"dc", "demand-source", "inventory-stock"}, "inventory values")
        exact_fields(rows["dc"], set(), "catcher")
        source = exact_fields(rows["demand-source"], {"cursor"}, "demand source")
        cursor = m.integer(source["cursor"], "demand cursor")
        cfg = state["construction"]["config"]
        if cursor > len(cfg["demands"]):
            fail("demand cursor exceeds tape")
        stock = exact_fields(rows["inventory-stock"], set(m.stock_type(self.version).DOMAIN_FIELDS), "inventory stock")
        for key in ("stock", "fulfilled", "lost", "received", "ordered"):
            m.integer(stock[key], key)
        if type(stock["pending"]) is not list or type(stock["events"]) is not list:
            fail("inventory histories must be lists")
        for row in stock["pending"]:
            exact_fields(row, {"due", "quantity"}, "pending order")
            m.number(row["due"], "order due", True)
            m.integer(row["quantity"], "order quantity", 1)
        for row in stock["events"]:
            if row.get("kind") not in ("action", "replenish", "demand"):
                fail("inventory event type differs")
            keys = {"time", "kind", "quantity"} | ({"fulfilled", "lost"} if row["kind"] == "demand" else set())
            exact_fields(row, keys, "inventory event")
            m.number(row["time"], "event time")
            m.integer(row["quantity"], "event quantity")
            if row["kind"] == "demand":
                m.integer(row["fulfilled"], "filled demand")
                m.integer(row["lost"], "lost demand")
        if self.version == 2 and stock["cumulative_penalty"] != stock["lost"] * cfg["penalty_per_unit"]:
            fail("inventory penalty accumulator differs")

    def validate_composition(self, engine_state, model_state, boundary_state, logical_context):
        self.validate_payload(model_state, topology(), None)
        cfg, rows = model_state["construction"]["config"], model_state["values"]
        now, stock = engine_state["executor"]["global_time"], rows["inventory-stock"]
        cursor = rows["demand-source"]["cursor"]
        if cursor != sum(row["at"] <= now for row in cfg["demands"]):
            fail("demand cursor and committed clock differ")
        inventory, filled, lost, received, ordered, previous_time = cfg["initial_stock"], 0, 0, 0, 0, 0.
        actions, replenished, demands = [], [], []
        for row in stock["events"]:
            if not previous_time <= row["time"] <= now:
                fail("inventory event ordering/clock differs")
            previous_time = row["time"]
            if row["kind"] == "action":
                ordered += row["quantity"]
                if row["quantity"]:
                    actions.append({"due": row["time"] + cfg["lead_time"], "quantity": row["quantity"]})
            elif row["kind"] == "replenish":
                inventory += row["quantity"]
                received += row["quantity"]
                replenished.append({"due": row["time"], "quantity": row["quantity"]})
            else:
                expected = min(inventory, row["quantity"])
                if row["fulfilled"] != expected or row["lost"] != row["quantity"] - expected:
                    fail("inventory event ledger fulfillment differs")
                inventory -= expected
                filled += expected
                lost += row["lost"]
                demands.append({"at": row["time"], "quantity": row["quantity"]})
        if demands != cfg["demands"][:cursor] or stock["pending"] != [row for row in actions if row["due"] > now]:
            fail("inventory demand/pending order conservation differs")
        if replenished != [row for row in actions if row["due"] <= now]:
            fail("inventory receipt conservation differs")
        if (inventory, filled, lost, received, ordered) != tuple(stock[key] for key in ("stock", "fulfilled", "lost", "received", "ordered")):
            fail("inventory aggregate and event ledger differ")
        due = {"dc": "+inf", "demand-source": cfg["demands"][cursor]["at"] if cursor < len(cfg["demands"]) else "+inf",
               "inventory-stock": min((row["due"] for row in stock["pending"]), default="+inf")}
        if any(engine_state["executor"]["models"][name]["request_time"] != value for name, value in due.items()):
            fail("inventory domain deadline and native schedule differ")
        observation = projection(model_state, now)
        if observation != boundary_state["environment"]["observation"]:
            fail("inventory observation cache differs")
        expected_reward = {"last_fulfilled": filled}
        if self.version == 2:
            expected_reward["last_penalty"] = stock["cumulative_penalty"]
        if boundary_state["environment"]["step_id"] == 0:
            expected_reward = m.initial_reward(self.version)
        if boundary_state["reward_state"] != expected_reward:
            fail("inventory reward baseline differs")
        if boundary_state["environment"]["seed"] != model_state["construction"]["seed"]:
            fail("inventory seed differs")
        validate_policy_context(logical_context["policy_context"])
        validate_sampling_context(logical_context["sampling_context"], boundary_state["environment"]["run_id"])

    def restore_into(self, graph, state):
        graph.source.cursor = state["values"]["demand-source"]["cursor"]
        for key, value in state["values"]["inventory-stock"].items():
            current = getattr(graph.stock, key)
            if type(current) is list:
                current[:] = copy.deepcopy(value)
            else:
                setattr(graph.stock, key, value)

    def rebind(self, graph, refs, services):
        if any(refs.get(name) is not value for name, value in self.reference_objects(graph).items()):
            fail("inventory restored reference mismatch")
        for leaf in graph.leaves():
            leaf.clock = services.clock
        self._clock_owners[graph] = (services, services.clock.__func__)

    def make_binding(self, graph, services, boundary_state):
        return services.make_binding(**m.callbacks(graph, services.clock, services.inject, boundary_state.value))

    def validate_restored(self, graph, engine_view):
        violations = self.inspect(graph)
        if violations:
            fail(violations[0].message, "CC_CONFORMANCE_FAILED")
        now = engine_view["executor"]["global_time"]
        if graph.observe(now) != projection(self.export_state(graph, None), now):
            fail("inventory restored observation differs")


def make_bundle(version):
    profile_id, adapter = f"inventory-C1-v{version}", InventoryAdapter(version)
    boundary = DeclaredFixedDeltaBoundaryProvider(profile_id=profile_id,
        schema_id=f"inventory-boundary-v{version}", observation_validator=m.validate_observation,
        reward_validator=m.validate_reward_v1 if version == 1 else extension.validate_reward_v2,
        initial_reward_state=m.initial_reward(version), source_bindings=adapter.source_bindings)
    return make_declared_bundle(profile_id=profile_id, model=adapter, boundary=boundary,
        model_provider_id=f"inventory-adapter-v{version}", projection_id=f"inventory-observation-v{version}",
        capabilities=("static-flat", "fixed-delta", "deterministic-demand", "pending-replenishment",
                      "inherited-confluence", "restored-time-action"))
