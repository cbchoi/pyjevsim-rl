"""Source-pinned P0 flat HLA engine state provider.

This is not a generic simulator serializer. The installed model adapter owns
domain state, callback qualification and the model-specific obligation that
same-time bag iteration does not alter the declared projection. Heapset buckets
are rebuilt as native sets, never sorted into a new tie-breaking semantics.

P0 v1 encodes infinity as ``'+inf'`` only in declared timing fields. Dynamic
creation/destruction, hierarchical executors, transport queues, external output
callbacks and rescheduling-in-progress are not admitted. Native source identity
is independent of the installed package's 2.1.1 metadata label.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import threading
from collections import deque
from pathlib import Path
from types import FunctionType

from pyjevsim.behavior_executor import BehaviorExecutor
from pyjevsim.behavior_model import BehaviorModel
from pyjevsim.core_model import CoreModel
from pyjevsim.default_message_catcher import DefaultMessageCatcher
from pyjevsim.definition import ExecutionType, ModelType, SimulationMode
from pyjevsim.executor_factory import ExecutorFactory
from pyjevsim.schedule_queue import ScheduleQueue
from pyjevsim.system_executor import SysExecutor
from pyjevsim.system_object import SystemObject

from pyjevsim_bridge.rl.adapters import FunctionalEpisodeBinding
from .contracts import ContinuationError, Violation, checked_id, exact_fields, fail
from .registry import SourceBinding

PROVIDER_ID = "pyjevsim-flat-hla-v1"
VERSION = "1"
_PINS = {
    "pyjevsim.system_executor": "b11728c9df1e0f4186a90a4d2f94a1a2e61336046ef8cd9d04a55594b81ca194",
    "pyjevsim.schedule_queue": "d7abb1e5cbb5c108d671cea0dd0085ff505dd0932fad01c4e909e824b830bd4b",
    "pyjevsim.behavior_executor": "7a5eea831841f48e9c647c92c6480a19d5111e011d3c3c7034da1a5536e29f98",
    "pyjevsim.behavior_model": "9344286c022843aefb1568c36cc2fee6894ea42132bce260d657f7d3177b238e",
    "pyjevsim.core_model": "d6c897a30cd66a831f6269973f8123b03a0f2c5ee955966203cec248a74f95c8",
    "pyjevsim.system_object": "446639952a3ae8e2b2da8ed178f3997b39ea89726412b953b3d00857852eccf5",
    "pyjevsim.system_message": "c53b589127c04b64009d48502d61394179c2d698eff93629fda941b4ba590004",
    "pyjevsim.executor": "3e61155a614c66e2151ab4fd14782b38ae92ead95d0b761b4f545743efedbc70",
    "pyjevsim.executor_factory": "eb4d5d0f2367f5dae19110a682867a1cb3566482366b1655964210c73b24c411",
    "pyjevsim.default_message_catcher": "cf70d0d054ec58fdcad776b753d8249cd5c822f47fc59f628bd267f34c46060a",
    "pyjevsim.definition": "183099a55a8f0dbbe4a115fc4d71b59aa1129911e6fee9c59adea08d1b2017b3",
    "pyjevsim.message_deliverer": "6b85305324d1ccec0daad74d8199e3d051407918d4b51269c21361b3d0a2efdc",
}

# Take in-process identities in addition to reading installed source files.
_NATIVE_METHODS = {
    cls: {name: (getattr(cls, name), getattr(cls, name).__code__)
          for base in cls.__mro__ for name, value in vars(base).items()
          if isinstance(value, FunctionType)}
    for cls in (SysExecutor, BehaviorExecutor, BehaviorModel, CoreModel,
                SystemObject, ScheduleQueue, ExecutorFactory, DefaultMessageCatcher)
}
_CORE_FIELDS = {"_SystemObject__object_id", "model_type", "_name",
                "external_input_ports", "external_output_ports"}
_BEHAVIOR_FIELDS = {"_states", "_cur_state", "global_time", "_cancel_reschedule_f",
                    "external_transition_map_tuple", "external_transition_map_state",
                    "internal_transition_map_tuple", "internal_transition_map_state"}
_ENGINE_FIELDS = _CORE_FIELDS | {
    "condition", "_track_uncaught", "global_time", "target_time", "time_resolution",
    "waiting_obj_map", "_waiting_keys", "active_obj_map", "product_port_map", "port_map",
    "hierarchical_structure", "model_map", "min_schedule_item", "_destructs_pending",
    "sim_init_time", "simulation_mode", "input_event_queue", "output_event_queue",
    "_output_event_callback", "ex_mode", "snapshot_manager", "exec_factory", "dmc",
}
_WRAPPER_FIELDS = {
    "engine_name", "_instance_t", "_destruct_t", "model", "parent", "_next_event_t",
    "_cur_state", "request_time", "behavior_model", "_cancel_reschedule_f",
    "_cached_destruct_time", "_obj_id", "global_time",
}
_WRAPPER_PAYLOAD = {"global_time", "next_event_time", "cur_state", "request_time",
                    "cancel_reschedule", "instance_time", "destruct_time", "engine_name"}
_BEHAVIOR_PAYLOAD = {"states", "cur_state", "global_time", "cancel_reschedule"}


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity() -> dict[str, str]:
    """Exact loaded implementation identity, not package-version qualification."""
    result = {}
    for name, expected in _PINS.items():
        measured = _sha(importlib.import_module(name).__file__)
        if measured != expected:
            fail(f"unsupported native source: {name}", "CC_INCOMPATIBLE_IDENTITY")
        result[name] = measured
    result[__name__] = _sha(__file__)
    return result


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        fail(f"{label} must be a finite nonnegative number")
    return value


def _time(value, label):
    return math.inf if value == "+inf" else _number(value, label)


def _encode_time(value):
    if type(value) not in (int, float) or math.isnan(value) or value < 0:
        fail("native timing field is unsupported", "CC_UNSUPPORTED_STATE")
    return "+inf" if value == math.inf else value


def _layout(topology):
    exact_fields(topology, {"nodes", "couplings", "shared_resources", "aliases"}, "topology")
    nodes = topology["nodes"]
    if type(nodes) is not dict or "dc" not in nodes:
        fail("P0 topology requires declared dc")
    roots = []
    for name, node in nodes.items():
        checked_id(name)
        exact_fields(node, {"type_id", "schema_id", "inputs", "outputs"}, "node")
        checked_id(node["type_id"])
        checked_id(node["schema_id"])
        for side in ("inputs", "outputs"):
            if type(node[side]) is not list or len(node[side]) != len(set(node[side])):
                fail("P0 ports must be distinct lists")
            for port in node[side]:
                checked_id(port)
        if node["type_id"] == "pyjevsim.structural":
            roots.append(name)
    if len(roots) != 1 or roots[0] == "dc":
        fail("P0 requires one declared flat structural boundary")
    if nodes["dc"] != {"type_id": "pyjevsim.default-message-catcher",
                        "schema_id": "pyjevsim-default-catcher-v1",
                        "inputs": ["uncaught"], "outputs": []}:
        fail("reserved dc declaration differs")
    root = roots[0]
    if type(topology["couplings"]) is not list:
        fail("couplings must be a list")
    for coupling in topology["couplings"]:
        exact_fields(coupling, {"source_node", "source_port", "target_node", "target_port"}, "coupling")
        for side in ("source", "target"):
            name, port = coupling[side + "_node"], coupling[side + "_port"]
            if name not in nodes:
                fail("undeclared coupling node")
            direction = "outputs" if side == "source" else "inputs"
            if name == root:
                direction = "inputs" if side == "source" else "outputs"
            if port not in nodes[name][direction]:
                fail("P0 coupling port direction differs")
    return root, tuple(name for name in nodes if name != root)


def _construction(descriptor):
    exact_fields(descriptor, {"provider_id", "mode", "time_resolution", "name"}, "engine construction")
    if (descriptor["provider_id"] != PROVIDER_ID or descriptor["mode"] != "HLA_TIME"
            or _number(descriptor["time_resolution"], "time resolution") != 1.0
            or descriptor["name"] != "default"):
        fail("unsupported P0 engine construction", "CC_UNSUPPORTED_PROFILE")


def _native_instance_methods(instance, cls):
    for name, (method, _) in _NATIVE_METHODS[cls].items():
        actual = getattr(instance, name, None)
        if getattr(actual, "__func__", None) is not method:
            fail(f"native callback replaced: {cls.__name__}.{name}", "CC_INCOMPATIBLE_IDENTITY")


class _EngineServices:
    """Explicit narrow services, not an automatic arbitrary-closure restore."""
    __slots__ = ("__engine",)

    def __init__(self, engine):
        self.__engine = engine

    def clock(self):
        return self.__engine.get_global_time()

    def inject(self, port, value):
        if port not in self.__engine.external_input_ports:
            fail("undeclared external input port")
        self.__engine.insert_external_event(port, value)

    def make_binding(self, **callbacks):
        if "executor" in callbacks:
            fail("binding may not replace the owned executor")
        callbacks.setdefault("initialize_fn", lambda _executor: self.__engine.step(0.0))
        callbacks.setdefault("close_fn", self.__engine.terminate_simulation)
        return FunctionalEpisodeBinding(executor=self.__engine, **callbacks)


class PyJevSimEngineProvider:
    """Only common engine state; no imports of any domain model or legacy codec."""
    def __init__(self):
        identity = source_identity()
        self.source_bindings = tuple(SourceBinding(name, str(Path(
            __file__ if name == __name__ else importlib.import_module(name).__file__
        ).resolve()), sha) for name, sha in identity.items())

    def verify_source_bytes(self):
        """Re-read every declared source; no checkpoint or file cache."""
        for source in self.source_bindings:
            if _sha(source.path) != source.sha256:
                fail(f"installed source changed: {source.logical_id}", "CC_INCOMPATIBLE_IDENTITY")

    def verify_loaded_identity(self):
        """Check loaded methods independently from the bytes on disk."""
        for cls, methods in _NATIVE_METHODS.items():
            for name, (method, code) in methods.items():
                if getattr(cls, name, None) is not method or method.__code__ is not code:
                    fail(f"native implementation replaced: {cls.__name__}.{name}", "CC_INCOMPATIBLE_IDENTITY")

    def verify_identity(self):
        # Preserve the legacy strict order and number of source observations.
        self.verify_source_bytes()
        self.verify_loaded_identity()

    def services(self, engine):
        if type(engine) is not SysExecutor:
            fail("services require the installed native engine")
        return _EngineServices(engine)

    def validate_payload(self, state, topology, profile):
        exact_fields(state, {"construction", "executor", "behaviors"}, "engine state")
        _construction(state["construction"])
        if profile is not None and profile.engine.provider_id != PROVIDER_ID:
            fail("engine profile identity differs", "CC_INCOMPATIBLE_IDENTITY")
        _, names = _layout(topology)
        saved = exact_fields(state["executor"], {"global_time", "target_time", "time_resolution",
            "simulation_mode", "calendar", "models"}, "executor state")
        now = _number(saved["global_time"], "engine clock")
        if (_number(saved["target_time"], "target time") != 0
                or _number(saved["time_resolution"], "time resolution") != 1.0
                or saved["simulation_mode"] != "SIMULATION_IDLE"):
            fail("unsupported HLA control state")
        exact_fields(saved["models"], set(names), "executor model inventory")
        exact_fields(saved["calendar"], set(names), "calendar inventory")
        exact_fields(state["behaviors"], set(names), "behavior inventory")
        for name in names:
            model = exact_fields(state["behaviors"][name], _BEHAVIOR_PAYLOAD, "common behavior")
            wrapper = exact_fields(saved["models"][name], _WRAPPER_PAYLOAD, "wrapper")
            if type(model["states"]) is not dict or not model["states"]:
                fail("common behavior requires declared states")
            for key, value in model["states"].items():
                checked_id(key, "state name")
                _time(value, "time advance")
            if model["cur_state"] not in model["states"] or model["cancel_reschedule"] is not False:
                fail("unsupported common behavior state/cancel flag")
            if (_number(model["global_time"], "model clock") > now
                    or wrapper["global_time"] != model["global_time"]):
                fail("model/wrapper callback clocks differ")
            _number(wrapper["global_time"], "wrapper clock")
            request = _time(wrapper["request_time"], "request time")
            if (request <= now or _time(wrapper["next_event_time"], "next event") != request
                    or _time(saved["calendar"][name], "calendar time") != request):
                fail("undrained or inconsistent event calendar")
            if (wrapper["cur_state"] != "" or wrapper["cancel_reschedule"] is not False
                    or _number(wrapper["instance_time"], "instance time") != 0
                    or wrapper["destruct_time"] != "+inf" or wrapper["engine_name"] != "default"):
                fail("unsupported static wrapper control state")
            if request != wrapper["global_time"] + _time(model["states"][model["cur_state"]], "deadline"):
                fail("model deadline and absolute request differ")
        dc = state["behaviors"]["dc"]
        if dc["states"] != {"IDLE": "+inf"} or dc["cur_state"] != "IDLE":
            fail("default catcher state differs")

    def allocate_empty(self, descriptor, cleanup):
        self.verify_identity()
        construction = descriptor.get("construction", descriptor)
        _construction(construction)
        engine = SysExecutor(1.0, _sim_name="default", ex_mode=ExecutionType.HLA_TIME)
        cleanup.add("engine", engine.terminate_simulation)
        return engine

    def attach(self, engine, graph, refs):
        self.verify_identity()
        topology = refs.topology
        root, names = _layout(topology)
        if (type(engine) is not SysExecutor or set(engine.model_map) != {"dc"}
                or engine.global_time != 0 or engine.active_obj_map or engine.port_map
                or engine.external_input_ports or engine.external_output_ports):
            fail("attach requires an empty owned P0 engine", "CC_UNSUPPORTED_STATE")
        boundary = refs.get(root)
        if boundary is not graph or boundary.model_type is not ModelType.STRUCTURAL:
            fail("declared boundary is not the candidate graph")
        refs.bind("dc", engine.dmc)
        for port in topology["nodes"][root]["inputs"]:
            engine.insert_input_port(port)
        for port in topology["nodes"][root]["outputs"]:
            engine.insert_output_port(port)
        for name in names:
            model = refs.get(name)
            if not isinstance(model, BehaviorModel) or model.model_type is not ModelType.BEHAVIORAL:
                fail("P0 registers only flat BehaviorModel leaves", "CC_UNSUPPORTED_STATE")
            if name != "dc":
                engine.register_entity(model)
        for row in topology["couplings"]:
            source = None if row["source_node"] == root else refs.get(row["source_node"])
            target = None if row["target_node"] == root else refs.get(row["target_node"])
            engine.coupling_relation(source, row["source_port"], target, row["target_port"])

    def _inspect_engine(self, engine, refs, topology):
        self.verify_identity()
        root, names = _layout(topology)
        if type(engine) is not SysExecutor or set(vars(engine)) != _ENGINE_FIELDS:
            fail("native engine type/field shape differs", "CC_UNSUPPORTED_STATE")
        _native_instance_methods(engine, SysExecutor)
        # Empty malformed containers must not pass merely because they are
        # falsey or compare equal to the native representation.
        for name in ("waiting_obj_map", "active_obj_map", "product_port_map", "port_map",
                     "hierarchical_structure", "model_map"):
            if type(getattr(engine, name)) is not dict:
                fail(f"native engine mapping shape differs: {name}", "CC_UNSUPPORTED_STATE")
        for name in ("_waiting_keys", "external_input_ports", "external_output_ports"):
            if type(getattr(engine, name)) is not list:
                fail(f"native engine list shape differs: {name}", "CC_UNSUPPORTED_STATE")
        if (type(engine.condition) is not threading.Condition
                or type(engine._destructs_pending) is not int or engine._destructs_pending != 0):
            fail("native condition/lifecycle representation differs", "CC_UNSUPPORTED_STATE")
        if (engine.ex_mode is not ExecutionType.HLA_TIME or engine.snapshot_manager is not None
                or engine._output_event_callback is not None or engine._track_uncaught is not False
                or engine.simulation_mode is not SimulationMode.SIMULATION_IDLE
                or engine.input_event_queue or engine.output_event_queue or engine.waiting_obj_map
                or engine._waiting_keys or engine._destructs_pending or engine.hierarchical_structure):
            fail("engine is not at a drained static HLA boundary", "CC_UNSUPPORTED_BOUNDARY")
        if type(engine.input_event_queue) is not list or type(engine.output_event_queue) is not deque:
            fail("native transport queue shape differs", "CC_UNSUPPORTED_STATE")
        if type(engine.exec_factory) is not ExecutorFactory or vars(engine.exec_factory):
            fail("native executor factory differs", "CC_UNSUPPORTED_STATE")
        _native_instance_methods(engine.exec_factory, ExecutorFactory)
        if engine._name != "default" or engine.model_type is not ModelType.UTILITY:
            fail("engine identity differs")
        if (engine.external_input_ports != topology["nodes"][root]["inputs"]
                or engine.external_output_ports != topology["nodes"][root]["outputs"]):
            fail("engine boundary ports differ")
        models = {name: refs.get(name) for name in names}
        if (refs.get("dc") is not engine.dmc or type(engine.dmc) is not DefaultMessageCatcher
                or set(vars(engine.dmc)) != _CORE_FIELDS | _BEHAVIOR_FIELDS):
            fail("reserved catcher identity/shape differs")
        _native_instance_methods(engine.dmc, DefaultMessageCatcher)
        if set(engine.model_map) != set(names) or set(engine.product_port_map) != set(models.values()):
            fail("flat model inventory differs")
        wrappers = {}
        common_methods = {name: method for name, (method, _) in _NATIVE_METHODS[BehaviorModel].items()
                          if name not in {"__init__", "ext_trans", "int_trans", "output"}}
        for name, model in models.items():
            node = topology["nodes"][name]
            if (not isinstance(model, BehaviorModel) or model.model_type is not ModelType.BEHAVIORAL
                    or model._name != name or model.external_input_ports != node["inputs"]
                    or model.external_output_ports != node["outputs"]
                    or type(model.external_input_ports) is not list
                    or type(model.external_output_ports) is not list or type(model._states) is not dict):
                fail("model name/type/declared ports differ")
            for key, method in common_methods.items():
                if getattr(getattr(model, key, None), "__func__", None) is not method:
                    fail(f"unsupported common behavior callback: {name}.{key}")
            if any(type(getattr(model, key)) is not dict or getattr(model, key) for key in (
                    "external_transition_map_tuple", "external_transition_map_state",
                    "internal_transition_map_tuple", "internal_transition_map_state")):
                fail("custom transition map is unsupported")
            found = engine.model_map[name]
            if type(found) is not list or len(found) != 1 or type(found[0]) is not BehaviorExecutor:
                fail("native wrapper inventory differs")
            wrapper = found[0]
            if set(vars(wrapper)) != _WRAPPER_FIELDS:
                fail("native wrapper state shape differs")
            _native_instance_methods(wrapper, BehaviorExecutor)
            if (wrapper.parent is not engine or wrapper.model is not model
                    or wrapper.behavior_model is not model or engine.product_port_map[model] is not wrapper
                    or type(wrapper._obj_id) is not int or wrapper._obj_id < 0
                    or wrapper._obj_id != model.get_obj_id() or wrapper._cached_destruct_time != math.inf
                    or wrapper._cached_destruct_time != wrapper._destruct_t):
                fail("wrapper ownership/lifecycle differs")
            wrappers[name] = wrapper
        if engine.active_obj_map != {wrapper._obj_id: wrapper for wrapper in wrappers.values()}:
            fail("active model inventory differs")
        expected_routes = {}
        for row in topology["couplings"]:
            source = engine if row["source_node"] == root else wrappers[row["source_node"]]
            target = engine if row["target_node"] == root else wrappers[row["target_node"]]
            expected_routes.setdefault((source, row["source_port"]), []).append((target, row["target_port"]))
        if engine.port_map != expected_routes:
            fail("routing order/multiplicity differs")
        calendar = engine.min_schedule_item
        if (type(calendar) is not ScheduleQueue or set(vars(calendar)) != {"_heap", "_mapped", "_reverse"}
                or type(calendar._heap) is not list or type(calendar._mapped) is not dict
                or type(calendar._reverse) is not dict
                or set(calendar._reverse) != {wrapper._obj_id for wrapper in wrappers.values()}):
            fail("calendar shape/inventory differs")
        _native_instance_methods(calendar, ScheduleQueue)
        for instant in calendar._heap:
            _encode_time(instant)
        members = set()
        for instant, bucket in calendar._mapped.items():
            _encode_time(instant)
            if type(bucket) is not set or (bucket and instant <= engine.global_time):
                fail("calendar bucket pending or unsupported")
            for wrapper in bucket:
                if (wrapper in members or wrapper not in wrappers.values()
                        or wrapper.request_time != instant or calendar._reverse.get(wrapper._obj_id) != instant):
                    fail("calendar membership differs")
                members.add(wrapper)
        if members != set(wrappers.values()):
            fail("calendar is incomplete")
        live_times = {instant for instant, bucket in calendar._mapped.items() if bucket}
        if not live_times.issubset(set(calendar._heap)) or any(
                calendar._heap[(index - 1) // 2] > instant
                for index, instant in enumerate(calendar._heap) if index):
            fail("calendar heap is inconsistent")

    def inspect(self, runtime, topology):
        try:
            self._inspect_engine(runtime.engine, runtime.refs, topology)
            self.validate_payload(self.export_state(runtime, runtime.refs), topology, None)
        except (ContinuationError, AttributeError, KeyError, TypeError, ValueError) as exc:
            return (Violation(getattr(exc, "code", "CC_UNSUPPORTED_STATE"), phase="inspect",
                              semantic_path="engine", message=str(exc)),)
        return ()

    def export_state(self, runtime, refs):
        engine = runtime.engine
        _, names = _layout(refs.topology)
        behaviors, wrappers, calendar = {}, {}, {}
        for name in names:
            model = refs.get(name)
            wrapper = engine.product_port_map[model]
            behaviors[name] = {"states": {key: _encode_time(value) for key, value in model._states.items()},
                "cur_state": model._cur_state, "global_time": model.global_time,
                "cancel_reschedule": model._cancel_reschedule_f}
            wrappers[name] = {"global_time": wrapper.global_time,
                "next_event_time": _encode_time(wrapper._next_event_t), "cur_state": wrapper._cur_state,
                "request_time": _encode_time(wrapper.request_time),
                "cancel_reschedule": wrapper._cancel_reschedule_f,
                "instance_time": wrapper._instance_t, "destruct_time": _encode_time(wrapper._destruct_t),
                "engine_name": wrapper.engine_name}
            calendar[name] = _encode_time(engine.min_schedule_item._reverse[wrapper._obj_id])
        return {"construction": {"provider_id": PROVIDER_ID, "mode": "HLA_TIME",
                    "time_resolution": 1.0, "name": "default"},
                "executor": {"global_time": engine.global_time, "target_time": engine.target_time,
                    "time_resolution": engine.time_resolution, "simulation_mode": engine.simulation_mode.name,
                    "calendar": calendar, "models": wrappers}, "behaviors": behaviors}

    def restore_into(self, engine, state, refs):
        self.validate_payload(state, refs.topology, None)
        saved = state["executor"]
        engine.global_time = saved["global_time"]
        engine.target_time = saved["target_time"]
        engine.time_resolution = saved["time_resolution"]
        engine.simulation_mode = SimulationMode.SIMULATION_IDLE
        engine.active_obj_map = {}
        engine.waiting_obj_map = {}
        engine._waiting_keys = []
        engine.min_schedule_item = ScheduleQueue()
        for name, behavior in state["behaviors"].items():
            model = refs.get(name)
            model._states = {key: _time(value, "deadline") for key, value in behavior["states"].items()}
            model._cur_state = behavior["cur_state"]
            model.global_time = behavior["global_time"]
            model._cancel_reschedule_f = behavior["cancel_reschedule"]
            wrapper = engine.product_port_map[model]
            values = saved["models"][name]
            wrapper.global_time = values["global_time"]
            wrapper._next_event_t = _time(values["next_event_time"], "next event")
            wrapper._cur_state = values["cur_state"]
            wrapper.request_time = _time(values["request_time"], "request")
            wrapper._cancel_reschedule_f = values["cancel_reschedule"]
            engine.active_obj_map[wrapper._obj_id] = wrapper
            # Native push only reads wrapper timing and clears already-false
            # cancellation flags. It neither calls a domain transition nor TA.
            engine.min_schedule_item.push(wrapper)

    def validate_restored(self, engine, state, refs):
        self._inspect_engine(engine, refs, refs.topology)
        runtime = type("_EngineView", (), {"engine": engine})()
        actual = self.export_state(runtime, refs)
        self.validate_payload(actual, refs.topology, None)
        if actual != state:
            fail("restored common engine state differs", "CC_CONFORMANCE_FAILED")
