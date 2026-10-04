"""Handwritten native inventory snapshot sidecar, separate from C1 providers.

The actual native SnapshotManager/RestoreHandler serialize model leaves. This
code explicitly repairs graph/config aliases, wrapper calendar and RL state.
Inputs are trusted local dill only, never untrusted exchange artifacts.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from pyjevsim.definition import ExecutionType, SimulationMode
from pyjevsim.restore_handler import RestoreHandler
from pyjevsim.schedule_queue import ScheduleQueue
from pyjevsim.snapshot_manager import SnapshotManager
from pyjevsim.structural_model import StructuralModel
from pyjevsim.system_object import SystemObject
from pyjevsim_bridge.rl.adapters import FunctionalEpisodeBinding
from pyjevsim_bridge.rl.environment import PyJevSimEnv
from pyjevsim_bridge.rl.executor import ExecutorDriver, FixedDeltaBoundary

from bench.continuation_study.native_baseline import (
    NativeCleanupReceipt, _ENV_FIELDS, _WRAPPER_FIELDS, _StagedWrapper, _staged_clock, _wire, _unwire)
from . import inventory as m
from .inventory_maintenance import validate_reward_v2

SCHEMA = "native-inventory-sidecar-v1"


@dataclass
class InventoryRuntime:
    engine: object
    graph: object
    env: object
    reward_state: dict

    def step(self, action):
        return self.env.step(action)

    def close(self):
        try:
            self.env.close()
        except BaseException as exc:
            return NativeCleanupReceipt(False, (f"{type(exc).__name__}: {exc}",))
        return NativeCleanupReceipt(True)


def binding(engine, graph, reward):
    return FunctionalEpisodeBinding(executor=engine,
        **m.callbacks(graph, engine.get_global_time, engine.insert_external_event, reward),
        initialize_fn=lambda ex: ex.step(0.), close_fn=engine.terminate_simulation)


def create_native(config, seed=0, delta=.25, max_steps=100, instance_id="inventory-native"):
    engine, graph = m.build_system(config, seed=seed)
    reward = m.initial_reward(config["version"])
    bound = binding(engine, graph, reward)
    env = PyJevSimEnv(lambda _context: bound, instance_id=instance_id, run_id="inventory-study",
                     boundary=FixedDeltaBoundary(delta), max_steps=max_steps, plugin_version=SCHEMA)
    try:
        env.reset(seed=seed)
        return InventoryRuntime(engine, graph, env, reward)
    except BaseException:
        env.close()
        engine.terminate_simulation()
        raise


def _assert_cut(runtime):
    engine, graph, env = runtime.engine, runtime.graph, runtime.env
    if (engine.input_event_queue or engine.output_event_queue or env._done or env._failed
            or engine.simulation_mode is not SimulationMode.SIMULATION_IDLE
            or engine.ex_mode is not ExecutionType.HLA_TIME
            or graph.observe(engine.global_time) != env._observation):
        raise ValueError("native inventory requires a committed live boundary")
    expected = {"last_fulfilled": graph.stock.fulfilled}
    if graph.config["version"] == 2:
        validate_reward_v2(runtime.reward_state)
        if graph.stock.cumulative_penalty != graph.stock.lost * graph.config["penalty_per_unit"]:
            raise ValueError("native inventory penalty accumulator mismatch")
        expected["last_penalty"] = graph.stock.cumulative_penalty
    else:
        m.validate_reward_v1(runtime.reward_state)
    if env._step_id == 0:
        expected = m.initial_reward(graph.config["version"])
    if runtime.reward_state != expected:
        raise ValueError("native inventory reward baseline mismatch")


def save_native(runtime, directory):
    directory = Path(directory)
    target = directory / "native-cut"
    if target.exists():
        raise FileExistsError(target)
    _assert_cut(runtime)
    engine, env, graph = runtime.engine, runtime.env, runtime.graph
    wrappers, staged = {}, {}
    for name, values in engine.model_map.items():
        wrapper, model = values[0], values[0].get_core_model()
        wrappers[name] = {key: getattr(wrapper, key) for key in _WRAPPER_FIELDS}
        wrappers[name]["model_global_time"] = model.global_time
        if name != "dc":
            leaf = copy.copy(model)
            leaf.clock = _staged_clock
            staged[name] = [_StagedWrapper(leaf)]
    relations = {(None if source is engine else source, port):
                 [(None if target is engine else target, target_port) for target, target_port in targets]
                 for (source, port), targets in engine.port_map.items()}
    sidecar = {"schema": SCHEMA, "config": graph.config, "seed": graph.seed,
        "global_time": engine.global_time, "target_time": engine.target_time,
        "wrappers": wrappers, "environment": {key: getattr(env, key) for key in _ENV_FIELDS},
        "delta": env._boundary.delta, "max_steps": env._max_steps,
        "has_advanced": env._driver._has_advanced, "reward_state": runtime.reward_state}
    encoded = json.dumps(_wire(sidecar), allow_nan=False, sort_keys=True, separators=(",", ":"))
    SnapshotManager().snapshot_simulation(relations, staged, "native-cut", str(directory))
    (target / "sidecar.json").write_text(encoded, encoding="utf-8")
    return {"bytes": sum(path.stat().st_size for path in target.iterdir()),
            "files": sorted(path.name for path in target.iterdir())}


def load_native(directory, instance_id="inventory-native-branch"):
    directory = Path(directory)
    sidecar = _unwire(json.loads((directory / "native-cut/sidecar.json").read_text(encoding="utf-8")))
    if sidecar["schema"] != SCHEMA:
        raise ValueError("native inventory sidecar schema mismatch")
    cfg = m.validate_config(sidecar["config"])
    handler = RestoreHandler(1., ex_mode=ExecutionType.HLA_TIME, name="native-cut", path=str(directory))
    engine, env = handler.engine, None
    try:
        if SnapshotManager(handler).get_engine() is not engine:
            raise ValueError("native restore engine identity mismatch")
        if set(handler.model_map) != {"demand-source", "inventory-stock"}:
            raise ValueError("native inventory leaf inventory mismatch")
        source, stock = handler.model_map["demand-source"], handler.model_map["inventory-stock"]
        if type(source) is not m.DemandSource or type(stock) is not m.stock_type(cfg["version"]):
            raise ValueError("native inventory leaf type mismatch")
        engine.set_name("default")
        graph = m.InventoryGraph.__new__(m.InventoryGraph)
        StructuralModel.__init__(graph, "inventory")
        graph.config, graph.seed, graph.source, graph.stock = cfg, sidecar["seed"], source, stock
        for leaf in graph.leaves():
            if leaf.config != cfg:
                raise ValueError("native journal and sidecar configurations differ")
            leaf.config, leaf.clock = cfg, engine.get_global_time
            SystemObject.__init__(leaf)
            engine.product_port_map[leaf]._obj_id = leaf.get_obj_id()
        graph.connect()
        engine.external_input_ports = ["action"]
        engine.external_output_ports = []
        engine.global_time, engine.target_time = sidecar["global_time"], sidecar["target_time"]
        engine.create_entity()  # Wrapper activation, not model transition/catchup.
        engine.min_schedule_item = ScheduleQueue()
        for name, values in engine.model_map.items():
            wrapper, saved = values[0], sidecar["wrappers"][name]
            for key in _WRAPPER_FIELDS:
                setattr(wrapper, key, saved[key])
            wrapper.get_core_model().global_time = saved["model_global_time"]
            engine.min_schedule_item.push(wrapper)
        engine.simulation_mode = SimulationMode.SIMULATION_IDLE
        reward = copy.deepcopy(sidecar["reward_state"])
        bound = binding(engine, graph, reward)
        def cannot_reset(_context):
            raise RuntimeError("restored inventory cannot reset")
        env = PyJevSimEnv(cannot_reset, instance_id=instance_id, run_id="inventory-study",
            boundary=FixedDeltaBoundary(sidecar["delta"]), max_steps=sidecar["max_steps"], plugin_version=SCHEMA)
        env._binding, env._driver = bound, ExecutorDriver(engine)
        for key in _ENV_FIELDS:
            setattr(env, key, copy.deepcopy(sidecar["environment"][key]))
        env._driver._has_advanced = sidecar["has_advanced"]
        runtime = InventoryRuntime(engine, graph, env, reward)
        _assert_cut(runtime)
        return runtime
    except BaseException:
        if env is not None:
            env.close()
        engine.terminate_simulation()
        raise
