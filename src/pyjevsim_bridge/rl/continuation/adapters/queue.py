"""Declared P0 queue domain adapter; no engine-private state writes.

The legacy value validator is reused for joint payload checks only. No legacy
capture, restore, allocator or runtime context is imported or invoked.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from ...models import _queue_snapshot_state as values
from ...models import queue_control as queue
from ..contracts import (
    ContinuationError, StateObligation, Violation, canonical_bytes,
    digest, exact_fields, fail, normalize_owned, thaw_value,
)
from ..registry import SourceBinding

CONSTRUCTION_SCHEMA = "queue-control-construction-v1"
_COMMON = {
    "_SystemObject__object_id", "model_type", "_name", "external_input_ports",
    "external_output_ports",
}
_BEHAVIOR = _COMMON | {
    "_states", "_cur_state", "global_time", "_cancel_reschedule_f",
    "external_transition_map_tuple", "external_transition_map_state",
    "internal_transition_map_tuple", "internal_transition_map_state",
}
_SHAPES = {
    queue.QueueControlGraph: _COMMON | {
        "model_map", "port_map", "config", "arrival_tape", "source", "server", "sink",
    },
    queue.ArrivalSource: _BEHAVIOR | {"tape", "clock", "index", "exhausted"},
    queue.BufferServer: _BEHAVIOR | values.SERVER_FIELDS | {"clock"},
    queue.CompletionSink: _BEHAVIOR | {"ledger", "pending"},
}
_METHODS = {
    cls: {
        name: (getattr(cls, name), getattr(getattr(cls, name), "__code__", None))
        for base in cls.__mro__ for name, member in vars(base).items()
        if (not name.startswith("__") or name == "__init__") and callable(member)
    }
    for cls in _SHAPES
}
_HELPERS = {
    name: (getattr(queue, name), getattr(queue, name).__code__)
    for name in ("validate_queue_config", "action_mode", "_make_tape")
}


def _detach(value: dict) -> dict:
    # Detached object-root normalization retains all closed-value/byte limits;
    # no live graph state or checked result is cached across provider calls.
    return normalize_owned(value).to_plain()


def _topology() -> dict:
    def node(type_id: str, inputs: list, outputs: list, schema: str = "queue-control-model-v1") -> dict:
        return {"type_id": type_id, "schema_id": schema, "inputs": inputs, "outputs": outputs}

    return {
        "nodes": {
            "queue-control": node("pyjevsim.structural", ["action"], ["completion"],
                                  "queue-control-graph-v1"),
            "arrival-source": node("queue.arrival-source", [], ["arrival"]),
            "buffer-server": node("queue.buffer-server", ["arrival", "action"], ["completed"]),
            "completion-sink": node("queue.completion-sink", ["completed"], ["completion"]),
            "dc": node("pyjevsim.default-message-catcher", ["uncaught"], [],
                       "pyjevsim-default-catcher-v1"),
        },
        "couplings": [
            {"source_node": source, "source_port": out, "target_node": target, "target_port": port}
            for source, out, target, port in (
                ("queue-control", "action", "buffer-server", "action"),
                ("arrival-source", "arrival", "buffer-server", "arrival"),
                ("buffer-server", "completed", "completion-sink", "completed"),
                ("completion-sink", "completion", "queue-control", "completion"),
            )
        ],
        "shared_resources": {
            "arrival-tape": {"type_id": "queue.ArrivalTape", "schema_id": "queue-arrival-tape-v1"},
        },
        "aliases": [
            {"owner": "queue-control", "path": "arrival_tape", "target": "arrival-tape"},
            {"owner": "arrival-source", "path": "tape", "target": "arrival-tape"},
        ],
    }


def _construction(raw: dict) -> dict:
    exact_fields(raw, {"schema_id", "config", "seed", "tape"}, "queue construction")
    if raw["schema_id"] != CONSTRUCTION_SCHEMA:
        fail("unknown queue construction schema", "CC_UNSUPPORTED_PROFILE")
    try:
        config = queue.validate_queue_config(raw["config"])
        if canonical_bytes(config) != canonical_bytes(raw["config"]):
            fail("queue configuration is not canonical")
        values.integer(raw["seed"], "seed")
        tape = exact_fields(raw["tape"], {"seed", "events", "end_time"}, "arrival tape")
        if tape["seed"] != raw["seed"]:
            fail("tape seed differs from model seed")
        end = values.number(tape["end_time"], "end_time")
        arrival = config["arrival_spec"]
        expected_end = (arrival["end_time"] if arrival["kind"] == "explicit"
                        else arrival["slot_count"] * arrival["slot_width"])
        if end != expected_end or type(tape["events"]) is not list or len(tape["events"]) > values.MAX_STEPS:
            fail("tape size/horizon differs")
        previous = 0.0
        seen = set(config["initial_waiting"])
        if config["initial_service"] is not None:
            seen.add(config["initial_service"])
        for event in tape["events"]:
            if type(event) is not list or len(event) != 2:
                fail("invalid tape event")
            instant, job = values.number(event[0], "arrival instant"), values.text(event[1], "job")
            if not 0 < instant <= end or instant < previous or job in seen:
                fail("invalid tape event order/identity")
            previous = instant
            seen.add(job)
        if arrival["kind"] == "explicit" and tape["events"] != [
            [row["time"], row["job_id"]] for row in arrival["events"]
        ]:
            fail("explicit tape differs from config")
        if arrival["kind"] == "bernoulli-slots":
            values.integer(arrival["slot_count"], "slot_count", 1, values.MAX_STEPS)
            previous_slot = 0
            for instant, job in tape["events"]:
                if not job.startswith("arrival-") or not job[8:].isdigit():
                    fail("invalid generated tape job")
                slot = int(job[8:])
                if not previous_slot < slot <= arrival["slot_count"] or instant != slot * arrival["slot_width"]:
                    fail("invalid generated tape slot")
                previous_slot = slot
    except (values.QueueSnapshotError, queue.QueueConfigurationError) as exc:
        raise ContinuationError("CC_INVALID_PAYLOAD", str(exc)) from exc
    return raw


def _expected_observation(model: dict, now: float) -> dict:
    """Pure value projection for composition checks; no graph/callback allocation."""
    construction, rows = model["construction"], model["values"]
    server, source = rows["buffer-server"], rows["arrival-source"]
    busy = server["current"] is not None
    elapsed = now - server["last_event_time"]
    mode = queue.MODES.index(server["mode"])
    if not math.isfinite(elapsed) or elapsed < 0:
        fail("projection time precedes internal work anchor")
    raw_work = server["remaining"] - queue.RATES[mode] * elapsed if busy else 0.0
    if not math.isfinite(raw_work) or raw_work < -queue.WORK_TOLERANCE:
        fail("projection has invalid remaining work")
    backlog = server["backlog"] + (len(server["waiting"]) + int(busy)) * elapsed
    energy = server["energy"] + (queue.POWERS[mode] * elapsed if busy else 0.0)
    weights = construction["config"]["cost_weights"]
    cost = weights["backlog"] * backlog + weights["energy"] * energy + weights["drop"] * server["dropped"]
    if not all(math.isfinite(value) for value in (backlog, energy, cost)):
        fail("nonfinite objective projection")
    return {
        "schema_version": queue.OBSERVATION_SCHEMA, "logical_time": float(now),
        "source_end_time": construction["tape"]["end_time"],
        "waiting_capacity": server["capacity"], "waiting_job_ids": list(server["waiting"]),
        "in_service_id": server["current"], "remaining_work": max(0.0, raw_work),
        "mode": server["mode"], "initial_job_count": server["initial_count"],
        "source_arrivals": server["source_arrivals"], "admitted": server["admitted"],
        "completed": server["completed"], "dropped": server["dropped"],
        "backlog_integral": backlog, "energy_integral": energy,
        "cumulative_cost": cost, "source_exhausted": source["exhausted"],
    }


class QueueModelAdapter:
    def __init__(self) -> None:
        self.source_bindings = tuple(
            SourceBinding(name, str(path), hashlib.sha256(path.read_bytes()).hexdigest())
            for name, path in (
                ("continuation.adapters.queue", Path(__file__)),
                ("models.queue_control", Path(queue.__file__)),
                ("models.queue_values", Path(values.__file__)),
            )
        )
        # Admission metadata only, never serialized or used as model state.
        # IDs here identify an already-owned live service; no PID lookup occurs.
        self._clock_owners = WeakKeyDictionary()

    def verify_identity(self) -> None:
        for cls, methods in _METHODS.items():
            if any(getattr(cls, name) is not method or getattr(method, "__code__", None) is not code
                   for name, (method, code) in methods.items()):
                fail("installed queue callback/constructor changed", "CC_INCOMPATIBLE_IDENTITY")
        if any(getattr(queue, name) is not function or function.__code__ is not code
               for name, (function, code) in _HELPERS.items()):
            fail("installed queue helper changed", "CC_INCOMPATIBLE_IDENTITY")

    def descriptor(self) -> dict:
        return {"schema_id": CONSTRUCTION_SCHEMA, "type_id": "queue-control-v1"}

    def topology(self) -> dict:
        return _topology()

    def obligations(self) -> tuple[StateObligation, ...]:
        rows = (
            ("01", "engine.clock", "engine", "SysExecutor.step", "SysExecutor.step", "TC-CC-004"),
            ("02", "model.server.work", "model", "BufferServer.project", "BufferServer._advance", "TC-CC-004"),
            ("03", "model.server.inventory", "model", "QueueControlGraph.observe", "BufferServer.ext_trans/int_trans", "TC-CC-005"),
            ("04", "model.server.cost_trace_sink", "model", "QueueControlGraph.observe", "BufferServer._advance/CompletionSink.ext_trans", "TC-CC-012"),
            ("05", "model.source.tape_cursor", "model", "ArrivalSource.output", "ArrivalSource.int_trans", "TC-CC-011"),
            ("06", "engine.behaviors_wrappers", "engine", "BehaviorExecutor.set_req_time", "BehaviorExecutor.set_req_time", "TC-CC-004"),
            ("07", "engine.calendar_tie", "engine", "SysExecutor._run_instant", "ScheduleQueue.push", "TC-CC-005"),
            ("08", "boundary.commit", "boundary", "PyJevSimEnv._step_unlocked", "PyJevSimEnv._step_unlocked", "TC-CC-003"),
            ("09", "boundary.cache_cursor", "boundary", "queue.reward", "PyJevSimEnv._step_unlocked", "TC-CC-012"),
            ("10", "composition.references", "coordinator", "model.clock/engine.routing", "adapter.rebind/engine.attach", "TC-CC-009"),
            ("11", "boundary.logical_context", "boundary", "fixed-policy caller", "explicit branch context", "TC-CC-013"),
            ("12", "model.profile_constants", "model", "callbacks/topology", "allocate_shell", "TC-CC-002"),
        )
        return tuple(StateObligation(
            f"OB-Q-{number}", path, owner, (read,), (write,),
            restore_rule="P0 declared field ownership; STATE_OBLIGATIONS.md",
            invariant="P0 composition validator and bounded queue topology",
            counterexample=f"STATE_OBLIGATIONS.md OB-Q-{number}",
            test_ids=(test,), evidence_status="reviewed",
        ) for number, path, owner, read, write, test in rows)

    def construction_from_request(self, request: Any) -> dict:
        config = queue.validate_queue_config(thaw_value(request.model_config))
        values.integer(request.seed, "seed")
        arrival = config["arrival_spec"]
        if arrival["kind"] == "bernoulli-slots":
            values.integer(arrival["slot_count"], "slot_count", 1, values.MAX_STEPS)
        elif len(arrival["events"]) > values.MAX_STEPS:
            fail("fresh explicit tape exceeds bounded profile", "CC_LIMIT")
        # Fresh path only; restoration consumes the stored tape values instead.
        tape = queue._make_tape(config, request.seed)
        return _construction({
            "schema_id": CONSTRUCTION_SCHEMA, "config": config, "seed": request.seed,
            "tape": {"seed": tape.seed, "end_time": tape.end_time,
                     "events": [list(event) for event in tape.events]},
        })

    def allocate_shell(self, descriptor: dict, services: Any, cleanup: Any) -> queue.QueueControlGraph:
        descriptor = _construction(_detach(descriptor))
        raw = descriptor["tape"]
        tape = queue.ArrivalTape(raw["seed"], tuple(tuple(event) for event in raw["events"]), raw["end_time"])
        # The inspected P0 constructors allocate values/ports and calculate an
        # initial deadline only; no reset, RNG draw, I/O or transition executes.
        graph = queue.QueueControlGraph(descriptor["config"], tape, services.clock)
        self._clock_owners[graph] = (id(services), services.clock.__func__)
        return graph

    def reference_objects(self, graph: queue.QueueControlGraph) -> dict:
        return {
            "queue-control": graph, "arrival-source": graph.source,
            "buffer-server": graph.server, "completion-sink": graph.sink,
            "arrival-tape": graph.arrival_tape,
        }

    def inspect(self, graph: queue.QueueControlGraph) -> tuple[Violation, ...]:
        try:
            if type(graph) is not queue.QueueControlGraph:
                fail("not a P0 queue graph", "CC_UNSUPPORTED_PROFILE")
            expected = (
                (graph, queue.QueueControlGraph), (graph.source, queue.ArrivalSource),
                (graph.server, queue.BufferServer), (graph.sink, queue.CompletionSink),
            )
            for obj, cls in expected:
                if type(obj) is not cls or set(vars(obj)) != _SHAPES[cls]:
                    fail("unknown model type/state field", "CC_UNSUPPORTED_PROFILE")
                if any(getattr(cls, name) is not method or getattr(method, "__code__", None) is not code
                       for name, (method, code) in _METHODS[cls].items()):
                    fail("model callback implementation changed", "CC_INCOMPATIBLE_IDENTITY")
            owner = self._clock_owners.get(graph)
            for callback in (graph.source.clock, graph.server.clock):
                if owner != (id(getattr(callback, "__self__", None)), getattr(callback, "__func__", None)):
                    fail("model clock is not its allocated candidate service", "CC_UNSUPPORTED_PROFILE")
            if type(graph.arrival_tape) is not queue.ArrivalTape or graph.source.tape is not graph.arrival_tape:
                fail("arrival tape alias differs", "CC_UNSUPPORTED_PROFILE")
            expected_models = {obj.get_name(): obj for obj in (graph.source, graph.server, graph.sink)}
            expected_routes = {
                (graph, "action"): [(graph.server, "action")],
                (graph.source, "arrival"): [(graph.server, "arrival")],
                (graph.server, "completed"): [(graph.sink, "completed")],
                (graph.sink, "completion"): [(graph, "completion")],
            }
            if graph.model_map != expected_models or graph.port_map != expected_routes:
                fail("queue semantic topology changed", "CC_UNSUPPORTED_PROFILE")
            names = ((graph, "queue-control"), (graph.source, "arrival-source"),
                     (graph.server, "buffer-server"), (graph.sink, "completion-sink"))
            nodes = self.topology()["nodes"]
            for obj, name in names:
                if (obj.get_name() != name or obj.external_input_ports != nodes[name]["inputs"]
                        or obj.external_output_ports != nodes[name]["outputs"]):
                    fail("queue node identity or ports changed", "CC_UNSUPPORTED_PROFILE")
            self.validate_payload(self.export_state(graph, None), self.topology(), None)
        except ContinuationError as exc:
            return (Violation(exc.code, "model.inspect", "queue-control", "OB-Q-12", str(exc)),)
        return ()

    def export_state(self, graph: queue.QueueControlGraph, refs: Any) -> dict:
        return _detach({
            "construction": {
                "schema_id": CONSTRUCTION_SCHEMA, "config": graph.config,
                "seed": graph.arrival_tape.seed,
                "tape": {"seed": graph.arrival_tape.seed, "end_time": graph.arrival_tape.end_time,
                         "events": graph.arrival_tape.events},
            },
            "values": {
                "dc": {}, "arrival-source": {"index": graph.source.index, "exhausted": graph.source.exhausted},
                "buffer-server": {name: getattr(graph.server, name) for name in values.SERVER_FIELDS},
                "completion-sink": {"ledger": graph.sink.ledger, "pending": graph.sink.pending},
            },
        })

    def validate_payload(self, state: dict, topology: dict, profile: Any) -> None:
        exact_fields(state, {"construction", "values"}, "queue model state")
        _construction(state["construction"])
        if topology != self.topology():
            fail("P0 queue topology differs", "CC_UNSUPPORTED_PROFILE")
        rows = exact_fields(state["values"], set(values.NAMES), "queue domain rows")
        exact_fields(rows["dc"], set(), "default catcher values")
        exact_fields(rows["arrival-source"], {"index", "exhausted"}, "source values")
        exact_fields(rows["buffer-server"], values.SERVER_FIELDS, "server values")
        exact_fields(rows["completion-sink"], {"ledger", "pending"}, "sink values")
        # Cross-domain numeric/time/conservation checks are performed on the
        # combined value payload before any runtime allocation.

    def validate_composition(self, engine_state: dict, model_state: dict,
                             boundary_state: dict, logical_context: dict) -> None:
        construction, descriptor = model_state["construction"], boundary_state["descriptor"]
        combined = {
            "config": construction["config"], "config_sha256": digest(construction["config"]),
            "seed": construction["seed"], "delta": descriptor["delta"],
            "max_steps": descriptor["max_steps"], "tape": construction["tape"],
            "models": {name: {"behavior": engine_state["behaviors"][name],
                              "values": model_state["values"][name]} for name in values.NAMES},
            "executor": engine_state["executor"], "environment": boundary_state["environment"],
        }
        try:
            values.validate_state(combined)
            values.policy_context(logical_context["policy_context"])
            values.sampling_context(logical_context["sampling_context"], combined["environment"]["run_id"])
        except values.QueueSnapshotError as exc:
            raise ContinuationError("CC_INVALID_PAYLOAD", str(exc)) from exc
        observation = _expected_observation(model_state, engine_state["executor"]["global_time"])
        if canonical_bytes(boundary_state["environment"]["observation"]) != canonical_bytes(observation):
            fail("RL cached observation/reward baseline differs from model projection")
        if boundary_state["environment"]["step_id"] == 0 and observation["logical_time"] != 0.0:
            fail("reset-committed P0 state must have logical time zero")

    def restore_into(self, graph: queue.QueueControlGraph, state: dict) -> None:
        rows = _detach(state["values"])
        for name, value in rows["arrival-source"].items():
            setattr(graph.source, name, value)
        for name, value in rows["buffer-server"].items():
            if name == "trace":
                graph.server.trace[:] = [tuple(item) for item in value]
            elif name == "waiting":
                graph.server.waiting[:] = value
            else:
                setattr(graph.server, name, value)
        graph.sink.ledger[:] = rows["completion-sink"]["ledger"]
        graph.sink.pending[:] = rows["completion-sink"]["pending"]

    def rebind(self, graph: queue.QueueControlGraph, refs: Any, services: Any) -> None:
        if any(refs.get(name) is not obj for name, obj in self.reference_objects(graph).items()):
            fail("candidate semantic reference changed", "CC_RESTORE_FAILED")
        graph.source.tape = graph.arrival_tape = refs.get("arrival-tape")
        graph.source.clock = graph.server.clock = services.clock
        self._clock_owners[graph] = (id(services), services.clock.__func__)

    def make_binding(self, graph: queue.QueueControlGraph, services: Any, boundary_state: Any) -> Any:
        def check_state() -> None:
            if boundary_state.value != {}:
                fail("P0 has no hidden mutable reward state", "CC_UNSUPPORTED_PROFILE")

        def apply_action(_unused: Any, action: Any) -> None:
            check_state()
            mode = queue.action_mode(action)
            if graph.server.current is None and mode != "idle":
                raise queue.QueueConfigurationError("empty queue permits only idle")
            services.inject("action", {"mode": mode})

        def observe(_unused: Any, _events: Any) -> dict:
            check_state()
            return graph.observe(services.clock())

        def reward(view: Any) -> float:
            check_state()
            return -(view.observation["cumulative_cost"] - view.previous_observation["cumulative_cost"])

        def terminal(view: Any) -> bool:
            check_state()
            row = view.observation
            return bool(row["source_exhausted"] and row["in_service_id"] is None
                        and not row["waiting_job_ids"] and not graph.sink.pending)

        def info(view: Any) -> dict:
            check_state()
            row = view.observation
            result = {
                "objective_cost": -reward(view), "cumulative_cost": row["cumulative_cost"],
                "arrival_tape_sha256": graph.arrival_tape.sha256,
                "unfinished_inventory": len(row["waiting_job_ids"]) + int(row["in_service_id"] is not None),
            }
            if view.step_id == 1:
                result["arrival_tape"] = graph.arrival_tape.content()
            return result

        return services.make_binding(apply_action_fn=apply_action, observe_fn=observe,
                                     reward_fn=reward, terminated_fn=terminal, info_fn=info)

    def validate_restored(self, graph: queue.QueueControlGraph, engine_view: dict) -> None:
        violations = self.inspect(graph)
        if violations:
            fail(violations[0].message, "CC_CONFORMANCE_FAILED")
        actual = graph.observe(engine_view["executor"]["global_time"])
        expected = _expected_observation(self.export_state(graph, None), engine_view["executor"]["global_time"])
        if actual != expected or graph.sink.pending:
            fail("restored queue state violates its physical projection", "CC_CONFORMANCE_FAILED")
