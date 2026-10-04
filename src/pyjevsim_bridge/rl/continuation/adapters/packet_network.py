"""Held-out packet domain adapter; no writes to engine/wrapper state."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from weakref import WeakKeyDictionary

from ...models import packet_network as m
from ..contracts import (ContinuationError, StateObligation, Violation, canonical_bytes,
                         decode_json, exact_fields, fail, thaw_value)
from ..generic_boundary import validate_policy_context, validate_sampling_context
from ..registry import SourceBinding

_IMPORT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CONSTRUCTION_SCHEMA = "packet-network-construction-v1"
NAMES = ("packet-source", "packet-router", "link-a", "link-b", "packet-sink", "dc")
_COMMON = {"_SystemObject__object_id", "model_type", "_name", "external_input_ports", "external_output_ports"}
_BEHAVIOR = _COMMON | {"_states", "_cur_state", "global_time", "_cancel_reschedule_f",
    "external_transition_map_tuple", "external_transition_map_state",
    "internal_transition_map_tuple", "internal_transition_map_state"}
_SHAPES = {
    m.PacketNetworkGraph: _COMMON | {"model_map", "port_map", "config", "seed", "source", "router", "link_a", "link_b", "sink"},
    m.PacketSource: _BEHAVIOR | {"config", "clock", "cursor"},
    m.PacketRouter: _BEHAVIOR | {"config", "clock", "route", "pending", "routed"},
    m.PacketLink: _BEHAVIOR | {"config", "clock", "active", "waiting", "inbox"},
    m.PacketSink: _BEHAVIOR | {"clock", "results", "pending"},
}
_METHODS = {cls: {name: (getattr(cls, name), getattr(getattr(cls, name), "__code__", None))
    for base in cls.__mro__ for name, member in vars(base).items()
    if callable(member) and (not name.startswith("__") or name == "__init__")} for cls in _SHAPES}
_HELPERS = {name: (getattr(m, name), getattr(m, name).__code__) for name in
    ("closed", "number", "integer", "validate_config", "validate_result", "validate_observation", "validate_reward_state", "emit")}


def detach(value):
    return decode_json(canonical_bytes(value))


def construction(raw):
    exact_fields(raw, {"schema_id", "config", "seed"}, "packet construction")
    if raw["schema_id"] != CONSTRUCTION_SCHEMA:
        fail("packet construction schema differs")
    try:
        if canonical_bytes(m.validate_config(raw["config"])) != canonical_bytes(raw["config"]):
            fail("noncanonical packet config")
        m.integer(raw["seed"], "seed")
    except (ValueError, TypeError, OverflowError) as exc:
        fail(str(exc))
    return raw


def topology():
    def node(kind, inputs, outputs, schema="packet-network-model-v1"):
        return {"type_id": kind, "schema_id": schema, "inputs": inputs, "outputs": outputs}
    routes = [("packet-network", "action", "packet-router", "action"),
        ("packet-source", "packet", "packet-router", "packet"),
        ("packet-router", "a", "link-a", "packet"), ("packet-router", "b", "link-b", "packet"),
        ("link-a", "result", "packet-sink", "result"), ("link-b", "result", "packet-sink", "result"),
        ("packet-sink", "result", "packet-network", "result")]
    return {"nodes": {
        "packet-network": node("pyjevsim.structural", ["action"], ["result"], "packet-network-graph-v1"),
        "packet-source": node("packet.source", [], ["packet"]),
        "packet-router": node("packet.router", ["packet", "action"], ["a", "b"]),
        "link-a": node("packet.link", ["packet"], ["result"]),
        "link-b": node("packet.link", ["packet"], ["result"]),
        "packet-sink": node("packet.sink", ["result"], ["result"]),
        "dc": node("pyjevsim.default-message-catcher", ["uncaught"], [], "pyjevsim-default-catcher-v1")},
        "couplings": [{"source_node": a, "source_port": b, "target_node": c, "target_port": d} for a, b, c, d in routes],
        "shared_resources": {}, "aliases": []}


def projection(state, now):
    rows, cfg = state["values"], state["construction"]["config"]
    results = rows["packet-sink"]["results"]
    delivered = sum(row["status"] == "delivered" for row in results)
    links = {}
    for side in ("a", "b"):
        link = rows[f"link-{side}"]
        links[f"link_{side}"] = {"active": None if link["active"] is None else link["active"]["packet"],
            "remaining": 0.0 if link["active"] is None else link["active"]["end_at"] - now,
            "waiting": list(link["waiting"])}
    return {"schema_version": m.OBS_SCHEMA, "logical_time": float(now), "released": rows["packet-source"]["cursor"],
        "delivered": delivered, "dropped": len(results) - delivered, "route": rows["packet-router"]["route"],
        **links, "results": results, "source_exhausted": rows["packet-source"]["cursor"] == len(cfg["packets"])}


class PacketNetworkModelAdapter:
    def __init__(self):
        self.source_bindings = (
            SourceBinding("continuation.adapters.packet_network", str(Path(__file__)), _IMPORT_SHA256),
            SourceBinding("models.packet_network", str(Path(m.__file__)), m._IMPORT_SHA256))
        self._clock_owners = WeakKeyDictionary()

    def verify_identity(self):
        for cls, methods in _METHODS.items():
            if any(getattr(cls, name) is not method or getattr(method, "__code__", None) is not code
                   for name, (method, code) in methods.items()):
                fail("packet implementation changed", "CC_INCOMPATIBLE_IDENTITY")
        if any(getattr(m, name) is not function or function.__code__ is not code
               for name, (function, code) in _HELPERS.items()):
            fail("packet helper changed", "CC_INCOMPATIBLE_IDENTITY")

    def descriptor(self):
        return {"schema_id": CONSTRUCTION_SCHEMA, "type_id": "packet-network-v1"}

    def topology(self):
        return topology()

    def obligations(self):
        rows = [("01", "source_cursor", "PacketSource.output", "PacketSource.int_trans", "next ordered burst is unchanged"),
            ("02", "routing", "PacketRouter.output", "PacketRouter.ext_trans/int_trans", "current route affects only future dispatch"),
            ("03", "link_fifo", "PacketLink._plan", "PacketLink.ext_trans/int_trans", "finite ordered FIFO and input bag"),
            ("04", "link_residual_deadline", "PacketLink._plan/_schedule", "PacketLink.int_trans", "absolute finish/deadline and deadline tie"),
            ("05", "sink_results", "PacketNetworkGraph.observe", "PacketSink.ext_trans", "packet conservation and drop/delivery reward inputs"),
            ("06", "constants_and_clock", "all packet callbacks", "allocate_shell/rebind", "exact static topology and candidate-owned clock")]
        return tuple(StateObligation(f"OB-N-{key}", f"model.packet.{path}", "model", (read,), (write,),
            restore_rule="explicit domain values and candidate clock rebind; no transition or reset",
            invariant=rule, counterexample="test_rl_continuation_packet",
            test_ids=("TC-CC-006", "TC-CC-011", "TC-CC-012", "TC-CC-019"), evidence_status="reviewed")
            for key, path, read, write, rule in rows)

    def construction_from_request(self, request):
        return construction(detach({"schema_id": CONSTRUCTION_SCHEMA,
            "config": m.validate_config(thaw_value(request.model_config)), "seed": request.seed}))

    def allocate_shell(self, descriptor, services, cleanup):
        row = construction(detach(descriptor))
        graph = m.PacketNetworkGraph(row["config"], row["seed"], services.clock)
        self._clock_owners[graph] = (id(services), services.clock.__func__)
        return graph

    def reference_objects(self, graph):
        return {"packet-network": graph, **{leaf.get_name(): leaf for leaf in graph.leaves()}}

    def inspect(self, graph):
        try:
            self.verify_identity()
            if type(graph) is not m.PacketNetworkGraph:
                fail("unknown packet graph type", "CC_UNSUPPORTED_PROFILE")
            objects = (graph, graph.source, graph.router, graph.link_a, graph.link_b, graph.sink)
            for obj in objects:
                if type(obj) not in _SHAPES or set(vars(obj)) != _SHAPES[type(obj)]:
                    fail("unknown packet object/field", "CC_UNSUPPORTED_PROFILE")
            if any(leaf.config is not graph.config for leaf in graph.leaves()[:-1]):
                fail("packet configuration alias differs", "CC_UNSUPPORTED_PROFILE")
            owner = self._clock_owners.get(graph)
            for leaf in graph.leaves():
                if owner != (id(getattr(leaf.clock, "__self__", None)), getattr(leaf.clock, "__func__", None)):
                    fail("packet clock is not candidate owned", "CC_UNSUPPORTED_PROFILE")
            refs, expected_routes = self.reference_objects(graph), {}
            for route in topology()["couplings"]:
                key = refs[route["source_node"]], route["source_port"]
                expected_routes.setdefault(key, []).append((refs[route["target_node"]], route["target_port"]))
            if graph.port_map != expected_routes or graph.model_map != {x.get_name(): x for x in graph.leaves()}:
                fail("packet topology changed", "CC_UNSUPPORTED_PROFILE")
            for name, obj in refs.items():
                node = topology()["nodes"][name]
                if obj.get_name() != name or obj.external_input_ports != node["inputs"] or obj.external_output_ports != node["outputs"]:
                    fail("packet node ports/name changed", "CC_UNSUPPORTED_PROFILE")
            self.validate_payload(self.export_state(graph, None), topology(), None)
        except (ContinuationError, ValueError, TypeError, AttributeError, KeyError) as exc:
            return (Violation(getattr(exc, "code", "CC_INVALID_PAYLOAD"), "model.inspect", "packet-network", "OB-N-06", str(exc)),)
        return ()

    def export_state(self, graph, refs):
        return detach({"construction": {"schema_id": CONSTRUCTION_SCHEMA, "config": graph.config, "seed": graph.seed},
            "values": {"dc": {}, "packet-source": {"cursor": graph.source.cursor},
                "packet-router": {"route": graph.router.route, "pending": graph.router.pending, "routed": graph.router.routed},
                **{name: {"active": obj.active, "waiting": obj.waiting, "inbox": obj.inbox}
                   for name, obj in (("link-a", graph.link_a), ("link-b", graph.link_b))},
                "packet-sink": {"results": graph.sink.results, "pending": graph.sink.pending}}})

    def validate_payload(self, state, supplied_topology, profile):
        exact_fields(state, {"construction", "values"}, "packet model state")
        construction(state["construction"])
        if supplied_topology != topology():
            fail("packet topology differs", "CC_UNSUPPORTED_PROFILE")
        rows = exact_fields(state["values"], set(NAMES), "packet values")
        exact_fields(rows["dc"], set(), "catcher")
        cfg = state["construction"]["config"]

        def packet(value):
            if m.integer(value, "packet", 1) > len(cfg["packets"]):
                fail("unknown packet")

        def sequence(value):
            if type(value) is not list or len(set(value)) != len(value):
                fail("packet ID sequence must be a unique list")
            for pid in value:
                packet(pid)

        try:
            source = exact_fields(rows["packet-source"], {"cursor"}, "source")
            if m.integer(source["cursor"], "cursor") > len(cfg["packets"]):
                fail("source cursor beyond inputs")
            router = exact_fields(rows["packet-router"], {"route", "pending", "routed"}, "router")
            if router["route"] not in ("a", "b"):
                fail("router route differs")
            sequence(router["pending"])
            if router["pending"] != sorted(router["pending"]):
                fail("router received bag order differs")
            if type(router["routed"]) is not list:
                fail("route ledger must be list")
            for row in router["routed"]:
                exact_fields(row, {"packet", "route", "time"}, "route row")
                packet(row["packet"])
                m.number(row["time"], "route time")
                if row["route"] not in ("a", "b"):
                    fail("route ledger destination differs")
            for side in ("a", "b"):
                row = exact_fields(rows[f"link-{side}"], {"active", "waiting", "inbox"}, "link")
                sequence(row["waiting"])
                sequence(row["inbox"])
                if (len(row["waiting"]) > cfg[f"capacity_{side}"]
                    or row["waiting"] != sorted(row["waiting"])
                    or row["inbox"] != sorted(row["inbox"])):
                    fail("link capacity/bag order differs")
                active = row["active"]
                if active is not None:
                    exact_fields(active, {"packet", "start_at", "finish_at", "end_at", "outcome"}, "active packet")
                    packet(active["packet"])
                    if row["waiting"] and active["packet"] >= row["waiting"][0]:
                        fail("active packet and FIFO semantic order differ")
                    for key in ("start_at", "finish_at", "end_at"):
                        m.number(active[key], key)
                    spec = cfg["packets"][active["packet"] - 1]
                    if (active["start_at"] < spec["at"] or active["start_at"] >= spec["deadline"]
                        or active["finish_at"] != active["start_at"] + spec["size"] / cfg[f"rate_{side}"]
                        or active["end_at"] != min(active["finish_at"], spec["deadline"])
                        or active["outcome"] != ("delivered" if active["finish_at"] <= spec["deadline"] else "ttl")):
                        fail("packet transmission/deadline state differs")
            sink = exact_fields(rows["packet-sink"], {"results", "pending"}, "sink")
            for name in ("results", "pending"):
                if type(sink[name]) is not list:
                    fail("sink result list required")
                for row in sink[name]:
                    m.validate_result(row)
                    packet(row["packet"])
        except (ValueError, TypeError, OverflowError) as exc:
            fail(str(exc))

    def validate_composition(self, engine_state, model_state, boundary_state, logical_context):
        self.validate_payload(model_state, topology(), None)
        cfg, rows = model_state["construction"]["config"], model_state["values"]
        now = engine_state["executor"]["global_time"]
        source, router, sink = rows["packet-source"], rows["packet-router"], rows["packet-sink"]
        if router["pending"] or sink["pending"] or any(rows[name]["inbox"] for name in ("link-a", "link-b")):
            fail("packet zero-time work is not committed")
        if source["cursor"] != sum(spec["at"] <= now for spec in cfg["packets"]):
            fail("packet source cursor and logical time differ")
        routed = router["routed"]
        if sorted(row["packet"] for row in routed) != list(range(1, source["cursor"] + 1)):
            fail("route ledger packet conservation differs")
        routes = {row["packet"]: row["route"] for row in routed}
        if any(row["time"] != cfg["packets"][row["packet"] - 1]["at"] for row in routed):
            fail("packet routing time differs from input event")
        inventory = []
        due = {"packet-source": cfg["packets"][source["cursor"]]["at"] if source["cursor"] < len(cfg["packets"]) else None,
               "packet-router": None, "packet-sink": None, "dc": None}
        for side in ("a", "b"):
            link = rows[f"link-{side}"]
            active = link["active"]
            owned = link["waiting"] + ([] if active is None else [active["packet"]])
            if active is None and link["waiting"]:
                fail("idle link has uncommitted FIFO")
            if active is not None and (active["end_at"] <= now or active["start_at"] > now):
                fail("active link deadline is not future")
            if any(routes.get(pid) != side for pid in owned):
                fail("link ownership differs from route ledger")
            inventory.extend(owned)
            due[f"link-{side}"] = None if active is None else active["end_at"]
        for row in sink["results"]:
            spec = cfg["packets"][row["packet"] - 1]
            if row["time"] > now or row["time"] < spec["at"] or routes.get(row["packet"]) != row["link"][-1]:
                fail("sink time/route differs")
            if (row["status"] == "delivered" and row["time"] > spec["deadline"]
                or row["status"] == "ttl" and row["time"] < spec["deadline"]
                or row["status"] == "overflow" and row["time"] != spec["at"]):
                fail("sink deadline/drop result differs")
            inventory.append(row["packet"])
        if sorted(inventory) != list(range(1, source["cursor"] + 1)):
            fail("packet conservation differs")
        for name, value in due.items():
            if engine_state["executor"]["models"][name]["request_time"] != ("+inf" if value is None else value):
                fail(f"{name} domain deadline and scheduler differ")
        observation = projection(model_state, now)
        try:
            m.validate_observation(observation)
            validate_policy_context(logical_context["policy_context"])
            validate_sampling_context(logical_context["sampling_context"], boundary_state["environment"]["run_id"])
        except (ValueError, TypeError) as exc:
            fail(str(exc))
        if canonical_bytes(observation) != canonical_bytes(boundary_state["environment"]["observation"]):
            fail("packet observation cache differs")
        expected_reward = {"last_delivered": observation["delivered"], "last_dropped": observation["dropped"]}
        if boundary_state["environment"]["step_id"] == 0:
            expected_reward = {"last_delivered": 0, "last_dropped": 0}
        if canonical_bytes(boundary_state["reward_state"]) != canonical_bytes(expected_reward):
            fail("packet reward baseline differs")
        if boundary_state["environment"]["seed"] != model_state["construction"]["seed"]:
            fail("packet seed/context differs")

    def restore_into(self, graph, state):
        rows = detach(state["values"])
        graph.source.cursor = rows["packet-source"]["cursor"]
        graph.router.route = rows["packet-router"]["route"]
        for key in ("pending", "routed"):
            getattr(graph.router, key)[:] = rows["packet-router"][key]
        for name, link in (("link-a", graph.link_a), ("link-b", graph.link_b)):
            link.active = rows[name]["active"]
            link.waiting[:] = rows[name]["waiting"]
            link.inbox[:] = rows[name]["inbox"]
        graph.sink.results[:] = rows["packet-sink"]["results"]
        graph.sink.pending[:] = rows["packet-sink"]["pending"]

    def rebind(self, graph, refs, services):
        if any(refs.get(name) is not obj for name, obj in self.reference_objects(graph).items()):
            fail("packet candidate reference differs", "CC_RESTORE_FAILED")
        for leaf in graph.leaves():
            leaf.clock = services.clock
        self._clock_owners[graph] = (id(services), services.clock.__func__)

    def make_binding(self, graph, services, boundary_state):
        def apply_action(_unused, action):
            action = m.closed(action, {"route"}, "routing action")
            if action["route"] not in ("a", "b"):
                raise ValueError("route must be a or b")
            services.inject("action", action)

        def reward(view):
            state, obs = boundary_state.value, view.observation
            m.validate_reward_state(state)
            value = graph.config["delivery_value"] * (obs["delivered"] - state["last_delivered"])
            value -= graph.config["drop_cost"] * (obs["dropped"] - state["last_dropped"])
            state.update(last_delivered=obs["delivered"], last_dropped=obs["dropped"])
            return value

        return services.make_binding(apply_action_fn=apply_action,
            observe_fn=lambda _engine, _events: graph.observe(services.clock()), reward_fn=reward,
            terminated_fn=lambda view: bool(view.observation["source_exhausted"] and
                view.observation["released"] == view.observation["delivered"] + view.observation["dropped"]),
            info_fn=lambda view: {"delivered": view.observation["delivered"], "dropped": view.observation["dropped"]})

    def validate_restored(self, graph, engine_view):
        violations = self.inspect(graph)
        if violations:
            fail(violations[0].message, "CC_CONFORMANCE_FAILED")
        now = engine_view["executor"]["global_time"]
        if canonical_bytes(graph.observe(now)) != canonical_bytes(projection(self.export_state(graph, None), now)):
            fail("packet restored projection differs", "CC_CONFORMANCE_FAILED")
