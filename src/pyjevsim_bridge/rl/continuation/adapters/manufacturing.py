"""Declared manufacturing domain state; engine-private fields are not written."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from random import Random
from weakref import WeakKeyDictionary

from ...models import manufacturing as m
from ..contracts import (ContinuationError, StateObligation, Violation, canonical_bytes,
                         exact_fields, fail, normalize_owned, thaw_value)
from ..registry import SourceBinding

_IMPORT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CONSTRUCTION_SCHEMA = "manufacturing-construction-v1"
NAMES = ("job-source", "stage-a", "stage-b", "tool-arbiter", "product-sink", "dc")
STAGE_FIELDS = {"duration", "waiting", "pending_requests", "current", "finish_at", "completed"}
_COMMON = {"_SystemObject__object_id", "model_type", "_name", "external_input_ports", "external_output_ports"}
_BEHAVIOR = _COMMON | {"_states", "_cur_state", "global_time", "_cancel_reschedule_f",
                      "external_transition_map_tuple", "external_transition_map_state",
                      "internal_transition_map_tuple", "internal_transition_map_state"}
_SHAPES = {
    m.ManufacturingGraph: _COMMON | {"model_map", "port_map", "config", "seed", "initial_rng_state",
                                    "ledger", "source", "stage_a", "stage_b", "arbiter", "sink"},
    m.JobSource: _BEHAVIOR | {"config", "clock", "job_cursor", "maintenance_cursor"},
    m.Stage: _BEHAVIOR | STAGE_FIELDS | {"clock", "ledger"},
    m.ToolArbiter: _BEHAVIOR | {"config", "clock", "ledger", "rng", "rng_draws", "requests", "maintenance_pending"},
    m.ProductSink: _BEHAVIOR | {"clock", "completions", "pending"},
    m.ToolLedger: {"owner", "repair_until", "trace"},
}
_METHODS = {cls: {name: (getattr(cls, name), getattr(getattr(cls, name), "__code__", None))
                  for base in cls.__mro__ for name, member in vars(base).items()
                  if callable(member) and (not name.startswith("__") or name == "__init__")}
            for cls in _SHAPES}
_HELPERS = {name: (getattr(m, name), getattr(m, name).__code__)
            for name in ("closed", "number", "integer", "validate_config", "tuple_tree",
                         "validate_rng", "validate_observation", "validate_reward_state", "emit")}
_RNG_METHODS = {name: (getattr(Random, name), getattr(getattr(Random, name), "__code__", None))
                for name in ("__new__", "__init__", "seed", "random", "getstate", "setstate")}


def detach(value):
    # This call owns a fresh value tree, never a cache of history or RNG state.
    return normalize_owned(value).to_plain()


def construction(raw):
    exact_fields(raw, {"schema_id", "config", "seed", "initial_rng_state"}, "manufacturing construction")
    if raw["schema_id"] != CONSTRUCTION_SCHEMA:
        fail("manufacturing construction schema differs")
    try:
        if canonical_bytes(m.validate_config(raw["config"])) != canonical_bytes(raw["config"]):
            fail("noncanonical manufacturing config")
        m.integer(raw["seed"], "seed")
        m.validate_rng(raw["initial_rng_state"])
    except (ValueError, TypeError, OverflowError) as exc:
        fail(str(exc))
    return raw


def topology():
    def node(type_id, inputs, outputs, schema="manufacturing-model-v1"):
        return {"type_id": type_id, "schema_id": schema, "inputs": inputs, "outputs": outputs}
    routes = [("manufacturing-cell", "action", "tool-arbiter", "action"),
              ("job-source", "job", "stage-a", "job"),
              ("job-source", "maintenance", "tool-arbiter", "maintenance"),
              ("stage-a", "completed", "stage-b", "job"),
              ("stage-b", "completed", "product-sink", "job"),
              ("product-sink", "completion", "manufacturing-cell", "completion")]
    for stage in ("stage-a", "stage-b"):
        routes.extend([(stage, "request", "tool-arbiter", "request"),
                       (stage, "release", "tool-arbiter", "release"),
                       ("tool-arbiter", "grant", stage, "grant")])
    return {"nodes": {
        "manufacturing-cell": node("pyjevsim.structural", ["action"], ["completion"], "manufacturing-graph-v1"),
        "job-source": node("manufacturing.source", [], ["job", "maintenance"]),
        "stage-a": node("manufacturing.stage", ["job", "grant"], ["request", "release", "completed"]),
        "stage-b": node("manufacturing.stage", ["job", "grant"], ["request", "release", "completed"]),
        "tool-arbiter": node("manufacturing.arbiter", ["request", "release", "maintenance", "action"], ["grant"]),
        "product-sink": node("manufacturing.sink", ["job"], ["completion"]),
        "dc": node("pyjevsim.default-message-catcher", ["uncaught"], [], "pyjevsim-default-catcher-v1"),
    }, "couplings": [{"source_node": a, "source_port": b, "target_node": c, "target_port": d}
                     for a, b, c, d in routes],
        "shared_resources": {"tool-ledger": {"type_id": "manufacturing.ToolLedger", "schema_id": "tool-ledger-v1"}},
        "aliases": [{"owner": name, "path": "ledger", "target": "tool-ledger"}
                    for name in ("manufacturing-cell", "stage-a", "stage-b", "tool-arbiter")]}


def projection(model_state, now):
    config, rows = model_state["construction"]["config"], model_state["values"]
    source, arbiter, sink = rows["job-source"], rows["tool-arbiter"], rows["product-sink"]
    ledger = arbiter["ledger"]
    inventory = [list(rows[name]["waiting"]) + ([rows[name]["current"]] if rows[name]["current"] is not None else [])
                 for name in ("stage-a", "stage-b")]
    completed = sink["completions"]
    released = source["job_cursor"]
    wip = sum(now - t for t in config["arrivals"][:released]) - sum(now - row["time"] for row in completed)
    return {"schema_version": m.OBS_SCHEMA, "logical_time": float(now), "released": released,
            "completed": len(completed), "wip_integral": wip, "cumulative_cost": wip * config["wip_weight"],
            "owner": ledger["owner"], "repair_remaining": 0.0 if ledger["repair_until"] is None else ledger["repair_until"] - now,
            "maintenance_pending": arbiter["maintenance_pending"], "stage_a_jobs": inventory[0],
            "stage_b_jobs": inventory[1], "completion_ledger": completed,
            "source_exhausted": source["job_cursor"] == len(config["arrivals"]) and
                                  source["maintenance_cursor"] == len(config["maintenance_times"]),
            "rng_draws": arbiter["rng_draws"]}


class ManufacturingModelAdapter:
    def __init__(self):
        self.source_bindings = (
            SourceBinding("continuation.adapters.manufacturing", str(Path(__file__)), _IMPORT_SHA256),
            SourceBinding("models.manufacturing", str(Path(m.__file__)), m._IMPORT_SHA256),
        )
        self._clock_owners = WeakKeyDictionary()

    def verify_identity(self):
        for cls, methods in _METHODS.items():
            if any(getattr(cls, name) is not method or getattr(method, "__code__", None) is not code
                   for name, (method, code) in methods.items()):
                fail("manufacturing implementation changed", "CC_INCOMPATIBLE_IDENTITY")
        if any(getattr(m, name) is not function or function.__code__ is not code
               for name, (function, code) in _HELPERS.items()):
            fail("manufacturing module helper changed", "CC_INCOMPATIBLE_IDENTITY")
        if m.Random is not Random or any(getattr(Random, name) is not method or
                getattr(method, "__code__", None) is not code for name, (method, code) in _RNG_METHODS.items()):
            fail("manufacturing RNG implementation changed", "CC_INCOMPATIBLE_IDENTITY")

    def descriptor(self):
        return {"schema_id": CONSTRUCTION_SCHEMA, "type_id": "manufacturing-cell-v1"}

    def topology(self):
        return topology()

    def obligations(self):
        rows = [("01", "model.manufacturing.source_cursors", "JobSource.output", "JobSource.int_trans", "next input must be unchanged"),
                ("02", "model.manufacturing.stage_inventory_deadline", "Stage.output/_schedule", "Stage.ext_trans/int_trans", "job conservation and absolute completion deadline"),
                ("03", "model.manufacturing.tool_owner_requests", "ToolArbiter._choice", "ToolArbiter.ext_trans/int_trans", "one owner and request/grant authority"),
                ("04", "model.manufacturing.maintenance_residual", "ToolArbiter._schedule", "ToolArbiter.int_trans", "nonpreemptive repair deadline"),
                ("05", "model.manufacturing.rng", "ToolArbiter.int_trans", "Random.random", "next draw and full MT19937/cache state"),
                ("06", "model.manufacturing.shared_ledger", "ManufacturingGraph.observe", "ToolArbiter.ext_trans/int_trans", "graph/stages/arbiter share one candidate ledger"),
                ("07", "model.manufacturing.sink", "ManufacturingGraph.observe", "ProductSink.ext_trans", "completion/WIP history"),
                ("08", "model.manufacturing.constants_topology", "all model callbacks", "allocate_shell", "closed static-flat topology and constants")]
        return tuple(StateObligation(f"OB-M-{i}", path, "model", (read,), (write,),
                     restore_rule="explicit domain values + reference rebind; no transition/RNG draw",
                     invariant=invariant, counterexample="test_rl_continuation_manufacturing",
                     test_ids=("TC-CC-006", "TC-CC-011", "TC-CC-012", "TC-CC-018"), evidence_status="reviewed")
                     for i, path, read, write, invariant in rows)

    def construction_from_request(self, request):
        config = m.validate_config(thaw_value(request.model_config))
        m.integer(request.seed, "seed")
        return construction(detach({"schema_id": CONSTRUCTION_SCHEMA, "config": config, "seed": request.seed,
                                    "initial_rng_state": Random(request.seed).getstate()}))

    def allocate_shell(self, descriptor, services, cleanup):
        row = construction(detach(descriptor))
        graph = m.ManufacturingGraph(row["config"], row["seed"], row["initial_rng_state"], services.clock)
        self._clock_owners[graph] = (id(services), services.clock.__func__)
        return graph

    def reference_objects(self, graph):
        return {"manufacturing-cell": graph, "tool-ledger": graph.ledger,
                **{leaf.get_name(): leaf for leaf in graph.leaves()}}

    def inspect(self, graph):
        try:
            self.verify_identity()
            objects = (graph, graph.source, graph.stage_a, graph.stage_b, graph.arbiter, graph.sink, graph.ledger)
            for obj in objects:
                if type(obj) not in _SHAPES or set(vars(obj)) != _SHAPES[type(obj)]:
                    fail("unknown manufacturing object/field", "CC_UNSUPPORTED_PROFILE")
            if any(obj.ledger is not graph.ledger for obj in (graph.stage_a, graph.stage_b, graph.arbiter)):
                fail("shared tool ledger alias differs", "CC_UNSUPPORTED_PROFILE")
            if graph.source.config is not graph.config or graph.arbiter.config is not graph.config:
                fail("model configuration alias differs", "CC_UNSUPPORTED_PROFILE")
            if type(graph.arbiter.rng) is not Random:
                fail("custom RNG unsupported", "CC_UNSUPPORTED_PROFILE")
            owner = self._clock_owners.get(graph)
            for leaf in graph.leaves():
                if owner != (id(getattr(leaf.clock, "__self__", None)), getattr(leaf.clock, "__func__", None)):
                    fail("clock does not belong to candidate", "CC_UNSUPPORTED_PROFILE")
            refs = self.reference_objects(graph)
            expected_routes = {}
            for route in topology()["couplings"]:
                key = (refs[route["source_node"]], route["source_port"])
                expected_routes.setdefault(key, []).append((refs[route["target_node"]], route["target_port"]))
            if graph.port_map != expected_routes or graph.model_map != {x.get_name(): x for x in graph.leaves()}:
                fail("manufacturing topology changed", "CC_UNSUPPORTED_PROFILE")
            for name, obj in refs.items():
                if name == "tool-ledger":
                    continue
                node = topology()["nodes"][name]
                if obj.get_name() != name or obj.external_input_ports != node["inputs"] or obj.external_output_ports != node["outputs"]:
                    fail("manufacturing node ports/name changed", "CC_UNSUPPORTED_PROFILE")
            self.validate_payload(self.export_state(graph, None), topology(), None)
        except (ContinuationError, ValueError, TypeError, AttributeError) as exc:
            return (Violation(getattr(exc, "code", "CC_INVALID_PAYLOAD"), "model.inspect", "manufacturing-cell", "OB-M-08", str(exc)),)
        return ()

    def export_state(self, graph, refs):
        return detach({"construction": {"schema_id": CONSTRUCTION_SCHEMA, "config": graph.config,
                    "seed": graph.seed, "initial_rng_state": graph.initial_rng_state},
            "values": {"dc": {}, "job-source": {"job_cursor": graph.source.job_cursor,
                          "maintenance_cursor": graph.source.maintenance_cursor},
                "stage-a": {key: getattr(graph.stage_a, key) for key in STAGE_FIELDS},
                "stage-b": {key: getattr(graph.stage_b, key) for key in STAGE_FIELDS},
                "tool-arbiter": {"requests": graph.arbiter.requests, "maintenance_pending": graph.arbiter.maintenance_pending,
                    "rng_draws": graph.arbiter.rng_draws, "rng_state": graph.arbiter.rng.getstate(),
                    "ledger": vars(graph.ledger)},
                "product-sink": {"completions": graph.sink.completions, "pending": graph.sink.pending}}})

    def validate_payload(self, state, supplied_topology, profile):
        exact_fields(state, {"construction", "values"}, "manufacturing model state")
        construction(state["construction"])
        if supplied_topology != topology():
            fail("manufacturing topology differs", "CC_UNSUPPORTED_PROFILE")
        rows = exact_fields(state["values"], set(NAMES), "manufacturing values")
        exact_fields(rows["dc"], set(), "catcher")
        cfg = state["construction"]["config"]
        try:
            source = exact_fields(rows["job-source"], {"job_cursor", "maintenance_cursor"}, "source")
            for key, seq in (("job_cursor", "arrivals"), ("maintenance_cursor", "maintenance_times")):
                if m.integer(source[key], key) > len(cfg[seq]):
                    fail("source cursor beyond input")
            for name, duration in (("stage-a", "stage_a_time"), ("stage-b", "stage_b_time")):
                row = exact_fields(rows[name], STAGE_FIELDS, name)
                if row["duration"] != cfg[duration]:
                    fail("stage duration differs")
                for key in ("waiting", "pending_requests"):
                    if type(row[key]) is not list or len(set(row[key])) != len(row[key]):
                        fail("invalid/duplicate stage job list")
                    for job in row[key]:
                        if m.integer(job, "job", 1) > len(cfg["arrivals"]):
                            fail("unknown stage job")
                if (row["current"] is None) != (row["finish_at"] is None):
                    fail("stage job/deadline mismatch")
                if row["current"] is not None:
                    m.integer(row["current"], "current", 1)
                    m.number(row["finish_at"], "finish_at")
                self._completions(row["completed"])
            arbiter = exact_fields(rows["tool-arbiter"], {"requests", "maintenance_pending", "rng_draws", "rng_state", "ledger"}, "arbiter")
            m.integer(arbiter["maintenance_pending"], "maintenance_pending")
            m.integer(arbiter["rng_draws"], "rng_draws")
            m.validate_rng(arbiter["rng_state"])
            if type(arbiter["requests"]) is not list:
                fail("request list required")
            for request in arbiter["requests"]:
                self._owner(request)
            if arbiter["requests"] != sorted(arbiter["requests"], key=lambda x: (x["job"], x["stage"])):
                fail("request semantic order differs")
            ledger = exact_fields(arbiter["ledger"], {"owner", "repair_until", "trace"}, "ledger")
            if ledger["owner"] is not None:
                self._owner(ledger["owner"])
            if ledger["repair_until"] is not None:
                m.number(ledger["repair_until"], "repair_until")
            if type(ledger["trace"]) is not list:
                fail("ledger trace must be list")
            for row in ledger["trace"]:
                if type(row) is not list or len(row) != 3 or row[0] not in ("grant", "release", "repair-start", "repair-end"):
                    fail("ledger trace row invalid")
                m.number(row[1], "trace time")
                if row[0] in ("grant", "release"):
                    self._owner(row[2])
                elif row[0] == "repair-start":
                    m.number(row[2], "repair duration", True)
                elif row[2] is not None:
                    fail("repair end payload differs")
            sink = exact_fields(rows["product-sink"], {"completions", "pending"}, "sink")
            self._completions(sink["completions"])
            self._completions(sink["pending"])
        except (ValueError, TypeError, OverflowError) as exc:
            fail(str(exc))

    @staticmethod
    def _owner(row):
        exact_fields(row, {"stage", "job"}, "owner/request")
        if row["stage"] not in ("stage-a", "stage-b"):
            fail("unknown stage")
        m.integer(row["job"], "job", 1)

    @staticmethod
    def _completions(rows):
        if type(rows) is not list:
            fail("completion list required")
        for row in rows:
            exact_fields(row, {"job", "time"}, "completion")
            m.integer(row["job"], "job", 1)
            m.number(row["time"], "time")

    def validate_composition(self, engine_state, model_state, boundary_state, logical_context):
        from ..generic_boundary import validate_policy_context, validate_sampling_context
        self.validate_payload(model_state, topology(), None)
        cfg, rows = model_state["construction"]["config"], model_state["values"]
        now = engine_state["executor"]["global_time"]
        source, arbiter, sink = rows["job-source"], rows["tool-arbiter"], rows["product-sink"]
        ledger = arbiter["ledger"]
        if arbiter["rng_draws"] != sum(row[0] == "repair-start" for row in ledger["trace"]):
            fail("repair history/RNG draw counter differs")
        if any(row[1] > now for row in ledger["trace"]):
            fail("future tool trace entry")
        for key, seq in (("job_cursor", "arrivals"), ("maintenance_cursor", "maintenance_times")):
            if source[key] != sum(t <= now for t in cfg[seq]):
                fail("committed source cursor/time differs")
        if sink["pending"] or any(rows[name]["pending_requests"] for name in ("stage-a", "stage-b")):
            fail("manufacturing zero-time protocol not drained")
        inventory, requests, active = [], [], []
        for name in ("stage-a", "stage-b"):
            stage = rows[name]
            inventory.extend(stage["waiting"])
            requests.extend({"stage": name, "job": job} for job in stage["waiting"])
            if stage["current"] is not None:
                inventory.append(stage["current"])
                active.append({"stage": name, "job": stage["current"]})
                if stage["finish_at"] <= now:
                    fail("busy stage deadline is not future")
            if any(item["time"] > now for item in stage["completed"]):
                fail("future stage completion")
        if sorted(inventory + [x["job"] for x in sink["completions"]]) != list(range(1, source["job_cursor"] + 1)):
            fail("manufacturing job conservation differs")
        if active != ([] if ledger["owner"] is None else [ledger["owner"]]):
            fail("resource owner differs from active stage")
        if ledger["repair_until"] is not None and (active or ledger["repair_until"] <= now):
            fail("repair conflicts with processing/time")
        if arbiter["requests"] != sorted(requests, key=lambda x: (x["job"], x["stage"])):
            fail("arbiter requests differ from stage waiters")
        if ledger["owner"] is None and ledger["repair_until"] is None and (requests or arbiter["maintenance_pending"]):
            fail("uncommitted ready arbiter")
        if rows["stage-b"]["completed"] != sink["completions"]:
            fail("stage B/sink completion history differs")
        stage_a_jobs = [x["job"] for x in rows["stage-a"]["completed"]]
        stage_b_jobs = rows["stage-b"]["waiting"] + ([rows["stage-b"]["current"]] if rows["stage-b"]["current"] is not None else [])
        if sorted(stage_a_jobs) != sorted(stage_b_jobs + [x["job"] for x in sink["completions"]]):
            fail("two-stage route conservation differs")
        due_source = min(cfg["arrivals"][source["job_cursor"]] if source["job_cursor"] < len(cfg["arrivals"]) else math.inf,
                         cfg["maintenance_times"][source["maintenance_cursor"]] if source["maintenance_cursor"] < len(cfg["maintenance_times"]) else math.inf)
        due = {"job-source": due_source, "stage-a": rows["stage-a"]["finish_at"], "stage-b": rows["stage-b"]["finish_at"],
               "tool-arbiter": ledger["repair_until"], "product-sink": None, "dc": None}
        for name, value in due.items():
            expected = "+inf" if value is None or value == math.inf else value
            if engine_state["executor"]["models"][name]["request_time"] != expected:
                fail(f"{name} domain deadline and scheduler differ")
        observation = projection(model_state, now)
        try:
            m.validate_observation(observation)
            validate_policy_context(logical_context["policy_context"])
            validate_sampling_context(logical_context["sampling_context"], boundary_state["environment"]["run_id"])
        except (ValueError, TypeError) as exc:
            fail(str(exc))
        if canonical_bytes(observation) != canonical_bytes(boundary_state["environment"]["observation"]):
            fail("manufacturing observation cache differs")
        expected_reward = {"last_cost": observation["cumulative_cost"], "last_completed": observation["completed"]}
        if canonical_bytes(boundary_state["reward_state"]) != canonical_bytes(expected_reward):
            fail("manufacturing reward baseline differs")
        if boundary_state["environment"]["seed"] != model_state["construction"]["seed"]:
            fail("manufacturing seed/context differs")

    def restore_into(self, graph, state):
        rows = detach(state["values"])
        for key, value in rows["job-source"].items():
            setattr(graph.source, key, value)
        for name, obj in (("stage-a", graph.stage_a), ("stage-b", graph.stage_b)):
            for key, value in rows[name].items():
                if isinstance(value, list):
                    getattr(obj, key)[:] = value
                else:
                    setattr(obj, key, value)
        arbiter = rows["tool-arbiter"]
        graph.arbiter.requests[:] = arbiter["requests"]
        graph.arbiter.maintenance_pending, graph.arbiter.rng_draws = arbiter["maintenance_pending"], arbiter["rng_draws"]
        graph.arbiter.rng.setstate(m.tuple_tree(arbiter["rng_state"]))
        graph.ledger.owner = arbiter["ledger"]["owner"]
        graph.ledger.repair_until = arbiter["ledger"]["repair_until"]
        graph.ledger.trace[:] = arbiter["ledger"]["trace"]
        graph.sink.completions[:] = rows["product-sink"]["completions"]
        graph.sink.pending[:] = rows["product-sink"]["pending"]

    def rebind(self, graph, refs, services):
        if any(refs.get(name) is not obj for name, obj in self.reference_objects(graph).items()):
            fail("manufacturing candidate reference changed", "CC_RESTORE_FAILED")
        graph.ledger = refs.get("tool-ledger")
        for obj in (graph.stage_a, graph.stage_b, graph.arbiter):
            obj.ledger = graph.ledger
        for leaf in graph.leaves():
            leaf.clock = services.clock
        self._clock_owners[graph] = (id(services), services.clock.__func__)

    def make_binding(self, graph, services, boundary_state):
        def apply_action(_unused, action):
            value = m.closed(action, {"maintenance"}, "manufacturing action")
            if type(value["maintenance"]) is not bool:
                raise ValueError("maintenance action must be bool")
            services.inject("action", value)

        def reward(view):
            state, obs = boundary_state.value, view.observation
            m.validate_reward_state(state)
            result = -(obs["cumulative_cost"] - state["last_cost"]) + graph.config["completion_value"] * (obs["completed"] - state["last_completed"])
            state.update(last_cost=obs["cumulative_cost"], last_completed=obs["completed"])
            return result

        def terminal(view):
            obs = view.observation
            return bool(obs["source_exhausted"] and obs["released"] == obs["completed"] and
                        graph.ledger.repair_until is None and not graph.arbiter.maintenance_pending)

        return services.make_binding(apply_action_fn=apply_action,
             observe_fn=lambda _ex, _events: graph.observe(services.clock()), reward_fn=reward,
             terminated_fn=terminal, info_fn=lambda view: {"cumulative_cost": view.observation["cumulative_cost"],
                                                        "completed": view.observation["completed"]})

    def validate_restored(self, graph, engine_view):
        violations = self.inspect(graph)
        if violations:
            fail(violations[0].message, "CC_CONFORMANCE_FAILED")
        now = engine_view["executor"]["global_time"]
        if canonical_bytes(graph.observe(now)) != canonical_bytes(projection(self.export_state(graph, None), now)):
            fail("manufacturing restored projection differs", "CC_CONFORMANCE_FAILED")
