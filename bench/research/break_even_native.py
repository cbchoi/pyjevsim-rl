"""Native inventory-risk journals plus handwritten calendar/boundary sidecar.

Uses actual SnapshotManager/RestoreHandler, not the common C1 serializer. Local
dill journals are trusted-input artifacts only.
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
from . import break_even_domain as m

SCHEMA = "native-inventory-risk-sidecar-v1"


@dataclass
class RiskRuntime:
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


def create_native(config, seed=None, delta=.25, max_steps=81,
                  instance_id="inventory-risk-native", run_id="break-even-study"):
    seed = config["input_seed"] if seed is None else seed
    engine, graph = m.build_system(config, seed=seed)
    reward = m.initial_reward()
    bound = binding(engine, graph, reward)
    env = PyJevSimEnv(lambda _context: bound, instance_id=instance_id, run_id=run_id,
                     boundary=FixedDeltaBoundary(delta), max_steps=max_steps, plugin_version=SCHEMA)
    try:
        env.reset(seed=seed)
        return RiskRuntime(engine, graph, env, reward)
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
        raise ValueError("native risk snapshot requires a committed live boundary")
    m.validate_domain_state(graph.stock.domain_state(), graph.config, graph.source.cursor, engine.global_time)
    m.validate_reward(runtime.reward_state)
    expected = {"last_fulfilled": graph.stock.fulfilled, "last_cumulative_risk": graph.stock.cumulative_risk}
    if env._step_id == 0:
        expected = m.initial_reward()
    if runtime.reward_state != expected:
        raise ValueError("native risk reward baseline mismatch")


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
        "run_id": env._run_id, "delta": env._boundary.delta, "max_steps": env._max_steps,
        "has_advanced": env._driver._has_advanced, "reward_state": runtime.reward_state}
    encoded = json.dumps(_wire(sidecar), allow_nan=False, sort_keys=True, separators=(",", ":"))
    SnapshotManager().snapshot_simulation(relations, staged, "native-cut", str(directory))
    (target / "sidecar.json").write_text(encoded, encoding="utf-8")
    return {"bytes": sum(path.stat().st_size for path in target.iterdir()),
            "files": sorted(path.name for path in target.iterdir())}


def load_native(directory, instance_id="inventory-risk-native-branch"):
    directory = Path(directory)
    sidecar = _unwire(json.loads((directory / "native-cut/sidecar.json").read_text(encoding="utf-8")))
    if sidecar["schema"] != SCHEMA:
        raise ValueError("native risk sidecar schema mismatch")
    cfg = m.validate_config(sidecar["config"])
    handler = RestoreHandler(1., ex_mode=ExecutionType.HLA_TIME, name="native-cut", path=str(directory))
    engine, env = handler.engine, None
    try:
        if SnapshotManager(handler).get_engine() is not engine:
            raise ValueError("native restore engine identity mismatch")
        if set(handler.model_map) != {"demand-source", "inventory-stock"}:
            raise ValueError("native risk leaf inventory mismatch")
        source, stock = handler.model_map["demand-source"], handler.model_map["inventory-stock"]
        if type(source) is not m.DemandSource or type(stock) is not m.RiskStock:
            raise ValueError("native risk leaf type mismatch")
        engine.set_name("default")
        graph = m.RiskGraph.__new__(m.RiskGraph)
        StructuralModel.__init__(graph, "inventory-risk")
        graph.config, graph.seed, graph.source, graph.stock = cfg, sidecar["seed"], source, stock
        for leaf in graph.leaves():
            if leaf.config != cfg:
                raise ValueError("native journal and sidecar configurations differ")
            leaf.config, leaf.clock = cfg, engine.get_global_time
            SystemObject.__init__(leaf)
            engine.product_port_map[leaf]._obj_id = leaf.get_obj_id()
        graph.connect()
        engine.external_input_ports, engine.external_output_ports = ["action"], []
        engine.global_time, engine.target_time = sidecar["global_time"], sidecar["target_time"]
        engine.create_entity()
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
            raise RuntimeError("restored risk runtime cannot reset")
        env = PyJevSimEnv(cannot_reset, instance_id=instance_id, run_id=sidecar["run_id"],
            boundary=FixedDeltaBoundary(sidecar["delta"]), max_steps=sidecar["max_steps"], plugin_version=SCHEMA)
        env._binding, env._driver = bound, ExecutorDriver(engine)
        for key in _ENV_FIELDS:
            setattr(env, key, copy.deepcopy(sidecar["environment"][key]))
        env._driver._has_advanced = sidecar["has_advanced"]
        runtime = RiskRuntime(engine, graph, env, reward)
        _assert_cut(runtime)
        return runtime
    except BaseException:
        if env is not None:
            env.close()
        engine.terminate_simulation()
        raise
