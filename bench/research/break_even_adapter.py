"""Explicit inventory-risk adapter using unchanged generic continuation APIs."""
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
from . import break_even_domain as m

SCHEMA = "inventory-risk-construction-v1"
PROFILE = "inventory-risk-C1-v1"
_COMMON = {"_SystemObject__object_id", "model_type", "_name", "external_input_ports", "external_output_ports"}
_BEHAVIOR = _COMMON | {"_states", "_cur_state", "global_time", "_cancel_reschedule_f",
    "external_transition_map_tuple", "external_transition_map_state", "internal_transition_map_tuple", "internal_transition_map_state"}
_SHAPES = {m.RiskGraph: _COMMON | {"model_map", "port_map", "config", "seed", "source", "stock"},
           m.DemandSource: _BEHAVIOR | {"config", "clock", "cursor"},
           m.RiskStock: _BEHAVIOR | {"config", "clock", *m.RiskStock.DOMAIN_FIELDS}}
_METHODS = {(cls, name): (getattr(cls, name), getattr(getattr(cls, name), "__code__", None))
            for cls in _SHAPES for name in dir(cls) if callable(getattr(cls, name))
            and (not name.startswith("__") or name == "__init__")}
_IMPORT_HASHES = {Path(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (__file__, m.__file__)}


def topology():
    def node(kind, inputs, outputs, schema="inventory-risk-model-v1"):
        return {"type_id": kind, "schema_id": schema, "inputs": inputs, "outputs": outputs}
    return {"nodes": {
        "inventory-risk": node("pyjevsim.structural", ["action"], [], "inventory-risk-graph-v1"),
        "demand-source": node("inventory-risk.demand", [], ["demand"]),
        "inventory-stock": node("inventory-risk.stock", ["action", "demand"], []),
        "dc": node("pyjevsim.default-message-catcher", ["uncaught"], [], "pyjevsim-default-catcher-v1")},
        "couplings": [{"source_node": "inventory-risk", "source_port": "action",
                       "target_node": "inventory-stock", "target_port": "action"},
                      {"source_node": "demand-source", "source_port": "demand",
                       "target_node": "inventory-stock", "target_port": "demand"}],
        "shared_resources": {}, "aliases": []}


class RiskAdapter:
    def __init__(self):
        self.source_bindings = tuple(SourceBinding(f"research.{path.stem}", str(path), digest)
                                     for path, digest in _IMPORT_HASHES.items())
        self._clock_owners = WeakKeyDictionary()

    def verify_identity(self):
        for (cls, name), (method, code) in _METHODS.items():
            if getattr(cls, name) is not method or getattr(method, "__code__", None) is not code:
                fail("risk implementation changed", "CC_INCOMPATIBLE_IDENTITY")

    def descriptor(self):
        return {"schema_id": SCHEMA, "type_id": m.SCHEMA}

    def topology(self):
        return topology()

    def obligations(self):
        rows = [("01", "products", "full product table and stock/demand conservation"),
                ("02", "cursor", "processed tape prefix agrees with clock and regimes"),
                ("03", "pending", "single absolute receipt deadline agrees with native calendar"),
                ("04", "ownership", "configuration aliases and clock references are rebound"),
                ("05", "reward", "risk and sales reward baselines agree with committed state")]
        return tuple(StateObligation(f"OB-BE-{key}", f"inventory-risk.{name}", "model",
            ("RiskGraph.observe",), ("RiskStock.ext_trans/int_trans",),
            restore_rule="restore declared state without model transitions", invariant=invariant,
            counterexample="tests.test_break_even_domain", test_ids=("BE002", "BE004", "BE005"),
            evidence_status="reviewed") for key, name, invariant in rows)

    def _construction(self, raw):
        exact_fields(raw, {"schema_id", "config", "seed"}, "risk construction")
        if raw["schema_id"] != SCHEMA:
            fail("risk construction identity differs", "CC_INCOMPATIBLE_IDENTITY")
        if canonical_bytes(m.validate_config(raw["config"])) != canonical_bytes(raw["config"]):
            fail("noncanonical risk config")
        m.integer(raw["seed"], "seed")
        if raw["seed"] != raw["config"]["input_seed"]:
            fail("model seed and configured input seed differ")
        return raw

    def construction_from_request(self, request):
        return self._construction({"schema_id": SCHEMA,
            "config": m.validate_config(thaw_value(request.model_config)), "seed": request.seed})

    def allocate_shell(self, descriptor, services, cleanup):
        row = self._construction(copy.deepcopy(descriptor))
        graph = m.RiskGraph(row["config"], row["seed"], services.clock)
        self._clock_owners[graph] = (services, services.clock.__func__)
        return graph

    def reference_objects(self, graph):
        return {"inventory-risk": graph, "demand-source": graph.source, "inventory-stock": graph.stock}

    def export_state(self, graph, refs):
        return {"construction": {"schema_id": SCHEMA, "config": copy.deepcopy(graph.config), "seed": graph.seed},
                "values": {"dc": {}, "demand-source": {"cursor": graph.source.cursor},
                           "inventory-stock": graph.stock.domain_state()}}

    def inspect(self, graph):
        try:
            self.verify_identity()
            if type(graph) is not m.RiskGraph or type(graph.stock) is not m.RiskStock:
                fail("risk graph/stock type differs")
            for obj in (graph, *graph.leaves()):
                if type(obj) not in _SHAPES or set(vars(obj)) != _SHAPES[type(obj)]:
                    fail("risk undeclared fields", "CC_UNSUPPORTED_PROFILE")
            owner = self._clock_owners.get(graph)
            for leaf in graph.leaves():
                if leaf.config is not graph.config or owner != (getattr(leaf.clock, "__self__", None),
                                                               getattr(leaf.clock, "__func__", None)):
                    fail("risk config/clock ownership differs")
            refs, expected = self.reference_objects(graph), {}
            for row in topology()["couplings"]:
                expected.setdefault((refs[row["source_node"]], row["source_port"]), []).append(
                    (refs[row["target_node"]], row["target_port"]))
            if graph.port_map != expected or graph.model_map != {leaf.get_name(): leaf for leaf in graph.leaves()}:
                fail("risk topology differs")
            for name, obj in refs.items():
                node = topology()["nodes"][name]
                if obj.get_name() != name or obj.external_input_ports != node["inputs"] or obj.external_output_ports != node["outputs"]:
                    fail("risk ports differ")
            self.validate_payload(self.export_state(graph, None), topology(), None)
        except (ContinuationError, ValueError, TypeError, KeyError, AttributeError) as exc:
            return (Violation(getattr(exc, "code", "CC_INVALID_PAYLOAD"), "model.inspect", "inventory-risk", message=str(exc)),)
        return ()

    def validate_payload(self, state, supplied_topology, profile):
        exact_fields(state, {"construction", "values"}, "risk state")
        self._construction(state["construction"])
        if supplied_topology != topology():
            fail("risk topology differs")
        rows = exact_fields(state["values"], {"dc", "demand-source", "inventory-stock"}, "risk values")
        exact_fields(rows["dc"], set(), "catcher")
        exact_fields(rows["demand-source"], {"cursor"}, "source")
        m.validate_domain_state(rows["inventory-stock"], state["construction"]["config"], rows["demand-source"]["cursor"])

    def validate_composition(self, engine_state, model_state, boundary_state, logical_context):
        self.validate_payload(model_state, topology(), None)
        cfg, rows = model_state["construction"]["config"], model_state["values"]
        now, stock = engine_state["executor"]["global_time"], rows["inventory-stock"]
        cursor = rows["demand-source"]["cursor"]
        m.validate_domain_state(stock, cfg, cursor, now)
        due = {"dc": "+inf", "demand-source": cfg["demands"][cursor]["at"] if cursor < len(cfg["demands"]) else "+inf",
               "inventory-stock": stock["pending"]["due"] if stock["pending"] else "+inf"}
        if any(engine_state["executor"]["models"][name]["request_time"] != value for name, value in due.items()):
            fail("risk calendar differs")
        if m.scalar_observation(stock, now) != boundary_state["environment"]["observation"]:
            fail("risk observation cache differs")
        reward = {"last_fulfilled": stock["fulfilled"], "last_cumulative_risk": stock["cumulative_risk"]}
        if boundary_state["environment"]["step_id"] == 0:
            reward = m.initial_reward()
        if boundary_state["reward_state"] != reward:
            fail("risk reward baseline differs")
        if boundary_state["environment"]["seed"] != model_state["construction"]["seed"]:
            fail("risk seed differs")
        validate_policy_context(logical_context["policy_context"])
        validate_sampling_context(logical_context["sampling_context"], boundary_state["environment"]["run_id"])

    def restore_into(self, graph, state):
        graph.source.cursor = state["values"]["demand-source"]["cursor"]
        for key, value in state["values"]["inventory-stock"].items():
            setattr(graph.stock, key, copy.deepcopy(value))

    def rebind(self, graph, refs, services):
        if any(refs.get(name) is not value for name, value in self.reference_objects(graph).items()):
            fail("risk restored references differ")
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
        if graph.observe(now) != m.scalar_observation(self.export_state(graph, None)["values"]["inventory-stock"], now):
            fail("risk restored observation differs")


def make_bundle():
    adapter = RiskAdapter()
    boundary = DeclaredFixedDeltaBoundaryProvider(profile_id=PROFILE,
        schema_id="inventory-risk-boundary-v1", observation_validator=m.validate_observation,
        reward_validator=m.validate_reward, initial_reward_state=m.initial_reward(),
        source_bindings=adapter.source_bindings)
    return make_declared_bundle(profile_id=PROFILE, model=adapter, boundary=boundary,
        model_provider_id="inventory-risk-adapter-v1", projection_id="inventory-risk-observation-v1",
        capabilities=("static-flat", "fixed-delta", "deterministic-demand", "pending-replenishment",
                      "inherited-confluence", "restored-time-action", "risk-scenarios"))
