"""Native PyJevSim journal plus a manually implemented Q/M continuation sidecar.

This deliberately does not use the proposed continuation providers or codecs.
The actual SnapshotManager and RestoreHandler persist/load the domain leaves.
The local sidecar repairs information those native APIs do not save: wrapper
clocks/calendar, graph/shared aliases and the RL boundary/reward cache. Files
are trusted-local dill inputs, NOT safe inputs from another party. Persistence
is cached filesystem, without fsync; save never overwrites an existing cut.

Only the declared static-flat Q/M, HLA_TIME, fixed-delta, nonterminal committed
boundary is supported. No transition, output, reset, input redraw or RNG draw
is used for restore. Pure observation is used to check the rebuilt candidate.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

from pyjevsim.default_message_catcher import DefaultMessageCatcher
from pyjevsim.definition import ExecutionType, ModelType, SimulationMode
from pyjevsim.restore_handler import RestoreHandler
from pyjevsim.schedule_queue import ScheduleQueue
from pyjevsim.snapshot_manager import SnapshotManager
from pyjevsim.structural_model import StructuralModel
from pyjevsim.system_object import SystemObject

from pyjevsim_bridge.rl.adapters import FunctionalEpisodeBinding
from pyjevsim_bridge.rl.environment import PyJevSimEnv
from pyjevsim_bridge.rl.executor import ExecutorDriver, FixedDeltaBoundary
from pyjevsim_bridge.rl.models import manufacturing as m
from pyjevsim_bridge.rl.models import queue_control as q

SCHEMA = "native-correct-qm-sidecar-v1"
_WRAPPER_FIELDS = ("global_time", "request_time", "_next_event_t", "_cur_state",
                   "_cancel_reschedule_f", "engine_name")
_ENV_FIELDS = ("_episode_number", "_step_id", "_observation", "_seed", "_done", "_failed")
_TRANSITION_MAP_FIELDS = ("external_transition_map_tuple", "external_transition_map_state",
                          "internal_transition_map_tuple", "internal_transition_map_state")
_DC_FIELDS = {"_SystemObject__object_id", "model_type", "_name", "external_input_ports",
              "external_output_ports", "_states", "_cancel_reschedule_f", "_cur_state",
              "global_time", *_TRANSITION_MAP_FIELDS}
_LEAF_TYPES = {
    "queue": {"arrival-source": q.ArrivalSource, "buffer-server": q.BufferServer,
              "completion-sink": q.CompletionSink},
    "manufacturing": {"job-source": m.JobSource, "stage-a": m.Stage,
                      "stage-b": m.Stage, "tool-arbiter": m.ToolArbiter,
                      "product-sink": m.ProductSink},
}


@dataclass(frozen=True)
class NativeCleanupReceipt:
    success: bool
    errors: tuple[str, ...] = ()


@dataclass
class NativeRuntime:
    engine: Any
    graph: Any
    env: PyJevSimEnv
    reward_state: dict
    model_kind: str
    _receipt: NativeCleanupReceipt | None = field(default=None, init=False, repr=False)

    def step(self, action):
        return self.env.step(action)

    def close(self) -> NativeCleanupReceipt:
        if self._receipt is None:
            try:
                self.env.close()
            except BaseException as exc:
                self._receipt = NativeCleanupReceipt(False, _error_messages(exc))
            else:
                self._receipt = NativeCleanupReceipt(True)
        return self._receipt


def _error_messages(exc):
    messages = (f"{type(exc).__name__}: {exc}",)
    if isinstance(exc, BaseExceptionGroup):
        for child in exc.exceptions:
            messages += _error_messages(child)
    return messages


def _dispose_failed_construction(engine, env, primary):
    """Retain the original failure together with every cleanup failure."""
    errors = []
    if env is not None:
        try:
            env.close()
        except BaseException as exc:
            errors.append(exc)
    try:
        if not engine.is_terminated():
            engine.terminate_simulation()
    except BaseException as exc:
        errors.append(exc)
    if errors:
        raise BaseExceptionGroup("native construction failed; cleanup unconfirmed",
                                 [primary, *errors]) from primary


def _staged_clock():
    raise RuntimeError("a journal staging clock cannot execute a model")


def _unseeded_random(state):
    # Random's ordinary pickle reducer constructs Random(), which seeds first.
    rng = Random.__new__(Random)
    rng.setstate(state)
    return rng


class _RandomStateCarrier:
    def __init__(self, rng):
        self.state = rng.getstate()

    def __reduce__(self):
        return _unseeded_random, (self.state,)


class _StagedWrapper:
    def __init__(self, model):
        self.model = model

    def get_core_model(self):
        return self.model


def _wire(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if math.isnan(value) or value == -math.inf:
            raise ValueError("unsupported sidecar number")
        return {"$number": "+inf"} if value == math.inf else value
    if type(value) in (list, tuple):
        return [_wire(x) for x in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _wire(v) for k, v in value.items()}
    raise TypeError(f"sidecar value is not plain data: {type(value).__name__}")


def _unwire(value):
    if type(value) is dict:
        if value == {"$number": "+inf"}:
            return math.inf
        return {k: _unwire(v) for k, v in value.items()}
    if type(value) is list:
        return [_unwire(v) for v in value]
    return value


def _target(directory, name):
    if type(name) is not str or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        raise ValueError("snapshot name must be a simple local folder name")
    return Path(directory).resolve() / name


def _factory_unavailable(_context):
    raise RuntimeError("native continuation runtime does not support reset")


def _binding(kind, engine, graph, reward_state):
    if kind == "queue":
        return q.bind_queue_system(engine, graph)

    def apply(ex, action):
        if type(action) is not dict or set(action) != {"maintenance"} or type(action["maintenance"]) is not bool:
            raise ValueError("manufacturing action is exactly {maintenance: bool}")
        ex.insert_external_event("action", dict(action), scheduled_time=0.0)

    def reward(view):
        obs = view.observation
        value = -(obs["cumulative_cost"] - reward_state["last_cost"])
        value += graph.config["completion_value"] * (obs["completed"] - reward_state["last_completed"])
        reward_state.update(last_cost=obs["cumulative_cost"], last_completed=obs["completed"])
        return value

    def terminated(view):
        obs = view.observation
        return (obs["source_exhausted"] and obs["released"] == obs["completed"]
                and graph.ledger.repair_until is None and not graph.arbiter.maintenance_pending)

    return FunctionalEpisodeBinding(
        executor=engine, apply_action_fn=apply,
        observe_fn=lambda ex, _events: graph.observe(ex.get_global_time()),
        reward_fn=reward, terminated_fn=terminated,
        info_fn=lambda view: {"cumulative_cost": view.observation["cumulative_cost"],
                              "completed": view.observation["completed"]},
        initialize_fn=lambda ex: ex.step(0.0), close_fn=engine.terminate_simulation,
    )


def create_native(model_kind, config, *, seed, delta, max_steps,
                  instance_id="native", run_id="study") -> NativeRuntime:
    if model_kind not in _LEAF_TYPES:
        raise ValueError("only queue and manufacturing are supported")
    boundary = FixedDeltaBoundary(delta)
    if type(seed) is not int or seed < 0 or type(max_steps) is not int or max_steps <= 0:
        raise ValueError("nonnegative integer seed and positive integer max_steps required")
    build = q.build_queue_system if model_kind == "queue" else m.build_manufacturing_system
    engine, graph = build(config, seed=seed)
    env = None
    try:
        reward_state = {} if model_kind == "queue" else {"last_cost": 0.0, "last_completed": 0}
        binding = _binding(model_kind, engine, graph, reward_state)
        used = False

        def factory(_context):
            nonlocal used
            if used:
                return _factory_unavailable(_context)
            used = True
            return binding

        env = PyJevSimEnv(factory, instance_id=instance_id, boundary=boundary,
                         max_steps=max_steps, run_id=run_id, plugin_version=SCHEMA)
        env.reset(seed=seed)
        return NativeRuntime(engine, graph, env, reward_state, model_kind)
    except BaseException as exc:
        _dispose_failed_construction(engine, env, exc)
        raise


def _assert_cut(runtime):
    engine, env = runtime.engine, runtime.env
    if runtime.model_kind not in _LEAF_TYPES or runtime._receipt is not None:
        raise ValueError("unsupported or closed runtime")
    if (env._binding is None or env._driver is None or env._episode_number != 1
            or any(getattr(env, flag) for flag in ("_done", "_failed", "_closed", "_closing",
                "_disposing", "_constructing", "_stepping", "_close_requested"))):
        raise ValueError("capture requires a committed nonterminal boundary")
    if (engine.ex_mode is not ExecutionType.HLA_TIME or engine.time_resolution != 1
            or engine.input_event_queue or engine.output_event_queue or engine.waiting_obj_map
            or engine._waiting_keys or engine.hierarchical_structure or engine._destructs_pending
            or engine._track_uncaught or engine._output_event_callback is not None
            or engine.simulation_mode is not SimulationMode.SIMULATION_IDLE):
        raise ValueError("engine is outside the native baseline static-flat cut")
    if (env._boundary.__class__ is not FixedDeltaBoundary or env._driver._executor is not engine
            or runtime.graph.observe(engine.global_time) != env._observation):
        raise ValueError("RL boundary/cache does not describe this engine")
    if set(engine.model_map) != {"dc", *_LEAF_TYPES[runtime.model_kind]}:
        raise ValueError("native model names differ")
    dc = engine.dmc
    dc_wrappers = engine.model_map["dc"]
    if (type(dc) is not DefaultMessageCatcher or set(vars(dc)) != _DC_FIELDS
            or dc.model_type is not ModelType.BEHAVIORAL or dc.get_name() != "dc" or dc._cur_state != "IDLE"
            or dc._states != {"IDLE": math.inf} or dc.get_cancel_flag()
            or dc.external_input_ports != ["uncaught"] or dc.external_output_ports
            or any(type(getattr(dc, key)) is not dict or getattr(dc, key) for key in _TRANSITION_MAP_FIELDS)
            or len(dc_wrappers) != 1 or dc_wrappers[0].get_core_model() is not dc
            or engine.product_port_map.get(dc) is not dc_wrappers[0]
            or engine.active_obj_map.get(dc.get_obj_id()) is not dc_wrappers[0]):
        raise ValueError("default catcher is not pristine")
    for name, wrappers in engine.model_map.items():
        if len(wrappers) != 1:
            raise ValueError("duplicate native leaf name")
        wrapper = wrappers[0]
        model = wrapper.get_core_model()
        if name != "dc" and type(model) is not _LEAF_TYPES[runtime.model_kind][name]:
            raise ValueError("native leaf type differs")
        if (wrapper.get_create_time() != 0 or wrapper.get_destruct_time() != math.inf
                or wrapper._cancel_reschedule_f or model.get_cancel_flag()
                or wrapper._next_event_t != wrapper.request_time
                or engine.active_obj_map.get(wrapper.get_obj_id()) is not wrapper
                or engine.min_schedule_item._reverse.get(wrapper.get_obj_id()) != wrapper.request_time):
            raise ValueError("wrapper/calendar cut is unsupported")
    if len(engine.active_obj_map) != len(engine.model_map):
        raise ValueError("unregistered active executor")


def save_native(runtime, directory: Path, name="cut") -> dict:
    """Write actual native leaf journal plus independent, explicit sidecar."""
    target = _target(directory, name)
    if target.exists():
        raise FileExistsError(target)
    _assert_cut(runtime)
    engine, graph, env = runtime.engine, runtime.graph, runtime.env
    wrappers = {}
    staged = {}
    for model_name, values in engine.model_map.items():
        wrapper = values[0]
        model = wrapper.get_core_model()
        wrappers[model_name] = {key: getattr(wrapper, key) for key in _WRAPPER_FIELDS}
        wrappers[model_name]["model_global_time"] = model.global_time
        if model_name == "dc":
            continue  # RestoreHandler creates exactly one pristine catcher.
        leaf = copy.copy(model)
        if hasattr(leaf, "clock"):
            leaf.clock = _staged_clock
        if runtime.model_kind == "manufacturing" and model_name == "tool-arbiter":
            leaf.rng = _RandomStateCarrier(model.rng)
        staged[model_name] = [_StagedWrapper(leaf)]
    relations = {}
    for (source, port), destinations in engine.port_map.items():
        relations[(None if source is engine else source, port)] = [
            (None if dest is engine else dest, target_port) for dest, target_port in destinations]
    sidecar = {
        "schema": SCHEMA, "model_kind": runtime.model_kind, "config": graph.config,
        "engine": {"name": engine.get_name(), "global_time": engine.global_time,
                   "target_time": engine.target_time, "time_resolution": engine.time_resolution,
                   "simulation_mode": engine.simulation_mode.name,
                   "input_ports": engine.external_input_ports, "output_ports": engine.external_output_ports},
        "wrappers": wrappers,
        "environment": {key: getattr(env, key) for key in _ENV_FIELDS},
        "boundary": {"delta": env._boundary.delta, "max_steps": env._max_steps,
                     "run_id": env._run_id, "has_advanced": env._driver._has_advanced},
        "reward_state": runtime.reward_state,
        "initial_rng_state": graph.initial_rng_state if runtime.model_kind == "manufacturing" else None,
    }
    encoded = json.dumps(_wire(sidecar), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    SnapshotManager().snapshot_simulation(relations, staged, name, str(target.parent))
    (target / "continuation-sidecar.json").write_text(encoded, encoding="utf-8")
    files = sorted(p.name for p in target.iterdir() if p.is_file())
    return {"folder": str(target), "files": files,
            "bytes": sum((target / filename).stat().st_size for filename in files),
            "schema": SCHEMA, "persistence": "cached-filesystem-no-fsync",
            "manual_repairs": ["root-endpoint-normalization", "omit-pristine-catcher",
                               "staged-clock", "fresh-physical-ids", "wrapper-calendar-sidecar",
                               "graph-shared-aliases", "rl-cache-reward-sidecar"]
                               + (["unseeded-current-rng-reducer"] if runtime.model_kind == "manufacturing" else [])}


def _graph_from_leaves(kind, config, loaded, engine, sidecar):
    if set(loaded) != set(_LEAF_TYPES[kind]):
        raise ValueError("native restored leaf set differs")
    for name, leaf in loaded.items():
        if type(leaf) is not _LEAF_TYPES[kind][name] or leaf.get_name() != name:
            raise ValueError("native restored leaf type/name differs")
        if hasattr(leaf, "clock"):
            leaf.clock = engine.get_global_time
    if kind == "queue":
        graph = q.QueueControlGraph.__new__(q.QueueControlGraph)
        StructuralModel.__init__(graph, "queue-control")
        graph.config = q.validate_queue_config(config)
        graph.source, graph.server, graph.sink = (loaded[x] for x in
                ("arrival-source", "buffer-server", "completion-sink"))
        graph.arrival_tape = graph.source.tape
        leaves = (graph.source, graph.server, graph.sink)
        routes = [(graph, "action", graph.server, "action"),
                  (graph.source, "arrival", graph.server, "arrival"),
                  (graph.server, "completed", graph.sink, "completed"),
                  (graph.sink, "completion", graph, "completion")]
    else:
        graph = m.ManufacturingGraph.__new__(m.ManufacturingGraph)
        StructuralModel.__init__(graph, "manufacturing-cell")
        graph.config = m.validate_config(config)
        graph.seed = sidecar["environment"]["_seed"]
        graph.initial_rng_state = m.tuple_tree(sidecar["initial_rng_state"])
        graph.source, graph.stage_a, graph.stage_b, graph.arbiter, graph.sink = (
            loaded[x] for x in ("job-source", "stage-a", "stage-b", "tool-arbiter", "product-sink"))
        if graph.source.config != graph.config or graph.arbiter.config != graph.config:
            raise ValueError("native copied configurations disagree")
        graph.source.config = graph.arbiter.config = graph.config
        graph.ledger = graph.arbiter.ledger
        for stage in (graph.stage_a, graph.stage_b):
            if vars(stage.ledger) != vars(graph.ledger):
                raise ValueError("native copied shared ledgers disagree")
            stage.ledger = graph.ledger
        leaves = graph.leaves()
        routes = [(graph, "action", graph.arbiter, "action"),
                  (graph.source, "job", graph.stage_a, "job"),
                  (graph.source, "maintenance", graph.arbiter, "maintenance"),
                  (graph.stage_a, "completed", graph.stage_b, "job"),
                  (graph.stage_b, "completed", graph.sink, "job"),
                  (graph.sink, "completion", graph, "completion")]
        for stage in (graph.stage_a, graph.stage_b):
            routes.extend([(stage, "request", graph.arbiter, "request"),
                           (stage, "release", graph.arbiter, "release"),
                           (graph.arbiter, "grant", stage, "grant")])
    graph.insert_input_port("action")
    graph.insert_output_port("completion")
    for leaf in leaves:
        graph.register_entity(leaf)
    for route in routes:
        graph.coupling_relation(*route)
    return graph


def load_native(directory: Path, name="cut", instance_id="branch") -> NativeRuntime:
    """Load a trusted-local native journal; no reset or simulated catch-up."""
    target = _target(directory, name)
    sidecar = _unwire(json.loads((target / "continuation-sidecar.json").read_text(encoding="utf-8")))
    if sidecar["schema"] != SCHEMA or sidecar["model_kind"] not in _LEAF_TYPES:
        raise ValueError("unsupported native sidecar schema/model")
    kind = sidecar["model_kind"]
    saved_env, boundary, saved_engine = sidecar["environment"], sidecar["boundary"], sidecar["engine"]
    if saved_env["_done"] or saved_env["_failed"] or saved_engine["simulation_mode"] != "SIMULATION_IDLE":
        raise ValueError("native sidecar is not a committed nonterminal boundary")
    handler = RestoreHandler(t_resol=saved_engine["time_resolution"], ex_mode=ExecutionType.HLA_TIME,
                             name=name, path=str(target.parent))
    engine, env = handler.engine, None
    try:
        restored = SnapshotManager(handler).get_engine()
        if restored is not engine:
            raise ValueError("native RestoreHandler returned a different engine")
        engine.set_name(saved_engine["name"])
        # Loaded leaf object IDs belong to the old process. Assign fresh
        # physical IDs before activation; semantic names and event times stay.
        for leaf in handler.model_map.values():
            SystemObject.__init__(leaf)
            engine.product_port_map[leaf]._obj_id = leaf.get_obj_id()
        graph = _graph_from_leaves(kind, sidecar["config"], handler.model_map, engine, sidecar)
        engine.external_input_ports = list(saved_engine["input_ports"])
        engine.external_output_ports = list(saved_engine["output_ports"])
        engine.global_time = saved_engine["global_time"]
        engine.target_time = saved_engine["target_time"]
        engine.create_entity()  # native wrapper activation only; no model callback
        engine.min_schedule_item = ScheduleQueue()
        if set(sidecar["wrappers"]) != set(engine.model_map):
            raise ValueError("native wrapper sidecar does not cover candidate")
        for model_name, rows in engine.model_map.items():
            if len(rows) != 1:
                raise ValueError("duplicate restored wrapper")
            wrapper, saved = rows[0], sidecar["wrappers"][model_name]
            for key in _WRAPPER_FIELDS:
                setattr(wrapper, key, saved[key])
            wrapper.get_core_model().global_time = saved["model_global_time"]
            engine.min_schedule_item.push(wrapper)
        engine.simulation_mode = SimulationMode[saved_engine["simulation_mode"]]
        reward_state = copy.deepcopy(sidecar["reward_state"])
        binding = _binding(kind, engine, graph, reward_state)
        env = PyJevSimEnv(_factory_unavailable, instance_id=instance_id,
                         boundary=FixedDeltaBoundary(boundary["delta"]),
                         max_steps=boundary["max_steps"], run_id=boundary["run_id"], plugin_version=SCHEMA)
        env._binding, env._driver = binding, ExecutorDriver(engine)
        for key in _ENV_FIELDS:
            setattr(env, key, copy.deepcopy(saved_env[key]))
        env._driver._has_advanced = boundary["has_advanced"]
        runtime = NativeRuntime(engine, graph, env, reward_state, kind)
        _assert_cut(runtime)
        if kind == "manufacturing":
            m.validate_reward_state(reward_state)
            if (reward_state["last_cost"] != env._observation["cumulative_cost"]
                    or reward_state["last_completed"] != env._observation["completed"]):
                raise ValueError("native reward baseline differs from saved boundary")
        return runtime
    except BaseException as exc:
        _dispose_failed_construction(engine, env, exc)
        raise
