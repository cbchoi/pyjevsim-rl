"""Opt-in live queue continuation at a drained decision boundary.

This is a model-owned, source-pinned codec, not a generic pyjevsim serializer.
The ordinary environment, learner checkpoint, and process wire are unchanged.
Capture/restore do no file I/O other than read-only source identity checks.
Actual model equivalence is a separate test obligation, not implied by a hash.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import Any, cast

from pyjevsim.behavior_executor import BehaviorExecutor
from pyjevsim.default_message_catcher import DefaultMessageCatcher
from pyjevsim.definition import ExecutionType, ModelType, SimulationMode
from pyjevsim.schedule_queue import ScheduleQueue
from pyjevsim.system_executor import SysExecutor

from pyjevsim_bridge.rl.adapters import FunctionalEpisodeBinding
from pyjevsim_bridge.rl.environment import PyJevSimEnv
from pyjevsim_bridge.rl.executor import ExecutorDriver, FixedDeltaBoundary

from . import _queue_snapshot_state as codec
from . import queue_control as queue

QueueSnapshotError = codec.QueueSnapshotError
PROFILE_ID = codec.PROFILE

# Exact runtime implementation inspected for this profile. Package version alone
# cannot identify the scheduler or its time/ordering semantics.
_PINNED_SOURCES = {
    "pyjevsim.system_executor": "b11728c9df1e0f4186a90a4d2f94a1a2e61336046ef8cd9d04a55594b81ca194",
    "pyjevsim.schedule_queue": "d7abb1e5cbb5c108d671cea0dd0085ff505dd0932fad01c4e909e824b830bd4b",
    "pyjevsim.behavior_executor": (
        "7a5eea831841f48e9c647c92c6480a19d5111e011d3c3c7034da1a5536e29f98"
    ),
    "pyjevsim.behavior_model": "9344286c022843aefb1568c36cc2fee6894ea42132bce260d657f7d3177b238e",
    "pyjevsim.core_model": "d6c897a30cd66a831f6269973f8123b03a0f2c5ee955966203cec248a74f95c8",
    "pyjevsim.system_object": "446639952a3ae8e2b2da8ed178f3997b39ea89726412b953b3d00857852eccf5",
    "pyjevsim.system_message": "c53b589127c04b64009d48502d61394179c2d698eff93629fda941b4ba590004",
    "pyjevsim.executor": "3e61155a614c66e2151ab4fd14782b38ae92ead95d0b761b4f545743efedbc70",
    "pyjevsim.executor_factory": "eb4d5d0f2367f5dae19110a682867a1cb3566482366b1655964210c73b24c411",
    "pyjevsim.default_message_catcher": (
        "cf70d0d054ec58fdcad776b753d8249cd5c822f47fc59f628bd267f34c46060a"
    ),
    "pyjevsim.definition": "183099a55a8f0dbbe4a115fc4d71b59aa1129911e6fee9c59adea08d1b2017b3",
    "pyjevsim.structural_model": "8bb8a8159f9fb1d1b9f567c52e503ab3303a1043512fd6e7cb953892b3494e0a",
    "pyjevsim.message_deliverer": (
        "6b85305324d1ccec0daad74d8199e3d051407918d4b51269c21361b3d0a2efdc"
    ),
    "pyjevsim_bridge.rl.environment": (
        "b6355530a3d1dcb2bce5c9247e2f2f13f00c5c5674cf89720f30c085ccdcae61"
    ),
    "pyjevsim_bridge.rl.executor": (
        "f2a5ca8d1302653c52c9fd641f87cc9a640253ecd4b15c36958576584a8cb790"
    ),
    "pyjevsim_bridge.rl.contracts": (
        "45f15b7793103c5fc550e2f347a42962b11c3b877062f8e9c35ca32eb551fbbd"
    ),
    "pyjevsim_bridge.rl.adapters": (
        "f50d83027d7c181e166ddf16b4d073f6208a37a1b1edf68b0477c3746ceed686"
    ),
    "pyjevsim_bridge.rl.models.queue_control": (
        "9ae3023a1684fbbc40d21263e8232307f55ab048d86427c53fc440549a7d2d3d"
    ),
}
_ENVELOPE = {
    "schema_version",
    "profile_id",
    "sources",
    "prefix_identity",
    "policy_context",
    "sampling_context",
    "state",
    "state_sha256",
}
_LIFECYCLE_FLAGS = (
    "_failed",
    "_closed",
    "_closing",
    "_disposing",
    "_constructing",
    "_construction_cleanup_started",
    "_stepping",
    "_close_requested",
)
_MODEL_TYPES = {
    "dc": DefaultMessageCatcher,
    "arrival-source": queue.ArrivalSource,
    "buffer-server": queue.BufferServer,
    "completion-sink": queue.CompletionSink,
}
_DECLARED_CONSTANTS = {
    "dc": ("dc", ModelType.BEHAVIORAL, ("uncaught",), ()),
    "arrival-source": ("arrival-source", ModelType.BEHAVIORAL, (), ("arrival",)),
    "buffer-server": ("buffer-server", ModelType.BEHAVIORAL, ("arrival", "action"), ("completed",)),
    "completion-sink": ("completion-sink", ModelType.BEHAVIORAL, ("completed",), ("completion",)),
    "graph": ("queue-control", ModelType.STRUCTURAL, ("action",), ("completion",)),
    "executor": ("default", ModelType.UTILITY, ("action",), ("completion",)),
}
# File hashes alone do not detect an in-process class/descriptor replacement.
# In particular, con_trans is inherited and intentionally not instrumented; a
# subclass changing it must never capture as the original queue profile.
_CLASS_METHODS = {
    cls: {
        name: getattr(cls, name)
        for parent in cls.__mro__
        for name, member in vars(parent).items()
        if not name.startswith("__") and callable(member)
    }
    for cls in (*_MODEL_TYPES.values(), queue.QueueControlGraph)
}


@lru_cache(maxsize=1)
def _source_identity_items() -> tuple[tuple[str, str], ...]:
    """Read/pin the loaded implementation once per process, never auto-qualify it."""
    result: dict[str, str] = {}
    for name, expected in _PINNED_SOURCES.items():
        module = importlib.import_module(name)
        actual = hashlib.sha256(Path(cast(str, module.__file__)).read_bytes()).hexdigest()
        if actual != expected:
            codec.fail(f"unsupported source {name}", "unsupported_profile")
        result[name] = actual
    for name in (__name__, codec.__name__):
        module = importlib.import_module(name)
        result[name] = hashlib.sha256(Path(cast(str, module.__file__)).read_bytes()).hexdigest()
    return tuple(sorted(result.items()))


def source_identity() -> dict[str, str]:
    """Return a detached copy, never the mutable identity cache itself."""
    return dict(_source_identity_items())


def _detached(value: Any) -> Any:
    return json.loads(codec.canonical(value))


def _read_snapshot(data: bytes) -> dict[str, Any]:
    raw = codec.fields(codec.decode(data), _ENVELOPE, "snapshot")
    if raw["schema_version"] != codec.SCHEMA or raw["profile_id"] != PROFILE_ID:
        codec.fail("snapshot profile/schema differs")
    if raw["sources"] != source_identity():
        codec.fail("snapshot source identity differs")
    codec.sha(raw["prefix_identity"], "prefix_identity")
    unsigned = {key: value for key, value in raw.items() if key != "state_sha256"}
    if codec.sha(raw["state_sha256"], "state_sha256") != codec.digest(unsigned):
        codec.fail("snapshot payload digest differs")
    codec.validate_state(raw["state"])
    codec.policy_context(raw["policy_context"])
    codec.sampling_context(raw["sampling_context"], raw["state"]["environment"]["run_id"])
    return raw


@dataclass(frozen=True, slots=True)
class SimulatorSnapshotV1:
    """Immutable canonical bytes; all executable state is represented as values."""

    data: bytes

    def __post_init__(self) -> None:
        _read_snapshot(self.data)

    @classmethod
    def from_bytes(cls, data: bytes) -> SimulatorSnapshotV1:
        return cls(data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def prefix_identity(self) -> str:
        return cast(str, _read_snapshot(self.data)["prefix_identity"])


@dataclass(frozen=True, slots=True)
class SnapshotCapabilityV1:
    supported: bool
    reason: str | None = None
    profile_id: str = PROFILE_ID


class BranchRuntimeContextV1:
    """Own one queue graph, real environment/driver, and immutable identities.

    Model callbacks are observed, not replaced by a direct transition shortcut.
    ``event_count`` counts actual internal plus external model-transition calls;
    a confluent callback contributes its actual int/ext calls. Output calls are
    separately exposed in ``event_counts``. Restore starts work counters at zero.
    """

    # Allocated by _allocate_context before the runtime is exposed to callers.
    _executor: SysExecutor
    env: PyJevSimEnv
    graph: queue.QueueControlGraph
    _operation_lock: threading.RLock
    _policy_bytes: bytes
    _sampling_bytes: bytes
    _config_bytes: bytes
    _seed: int
    _delta: float
    _max_steps: int
    _factory_used: bool
    _binding: FunctionalEpisodeBinding
    _counts: dict[str, int]
    _observers: dict[tuple[object, str], MethodType]
    _shapes: list[tuple[object, frozenset[str]]]
    _binding_values: dict[str, Any]

    @classmethod
    def create(
        cls,
        model_config: Mapping[str, object],
        *,
        seed: int,
        instance_id: str = "queue-branch",
        run_id: str = "queue-branch-study",
        delta: float = 0.5,
        max_steps: int = 576,
        policy_context: Mapping[str, object] | None = None,
        sampling_context: Mapping[str, object] | None = None,
    ) -> BranchRuntimeContextV1:
        return _allocate_context(
            model_config,
            seed=seed,
            instance_id=instance_id,
            run_id=run_id,
            delta=delta,
            max_steps=max_steps,
            policy_context=policy_context,
            sampling_context=sampling_context,
            tape=None,
            initialize=True,
        )

    @property
    def executor(self) -> SysExecutor:
        return self._executor

    @property
    def observation(self) -> dict[str, Any]:
        return cast(dict[str, Any], _detached(self.env._observation))

    @property
    def step_id(self) -> int:
        return self.env._step_id

    @property
    def logical_time(self) -> float:
        return float(self._executor.global_time)

    @property
    def event_count(self) -> int:
        return self._counts["internal"] + self._counts["external"]

    @property
    def event_counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def physical_events(self) -> list[Any]:
        """Queue server's physical order, not a claimed generic global tie order."""
        return cast(list[Any], _detached(self.graph.server.trace))

    @property
    def physical_event_length(self) -> int:
        return len(self.graph.server.trace)

    def physical_events_since(self, index: int) -> list[Any]:
        codec.integer(index, "physical event cursor", 0, len(self.graph.server.trace))
        return cast(list[Any], _detached(self.graph.server.trace[index:]))

    @property
    def policy_context(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._policy_bytes))

    @property
    def sampling_context(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._sampling_bytes))

    def step(self, action: object) -> tuple[object, float, bool, bool, dict[str, object]]:
        if not self._operation_lock.acquire(blocking=False):
            codec.fail("another context operation is in flight", "unsupported_boundary")
        try:
            return self.env.step(action)
        finally:
            self._operation_lock.release()

    def close(self) -> None:
        with self._operation_lock:
            self.env.close()

    def _factory(self, context: Any) -> Any:
        del context
        if self._factory_used:
            codec.fail("branch context reset is not an implicit restore", "unsupported_state")
        self._factory_used = True
        return self._binding


def _observe(context: BranchRuntimeContextV1) -> None:
    context._counts = {"output": 0, "internal": 0, "external": 0}
    context._observers = {}
    for model in (context.graph.source, context.graph.server, context.graph.sink):
        for name, key in (
            ("output", "output"),
            ("int_trans", "internal"),
            ("ext_trans", "external"),
        ):
            original = getattr(type(model), name)

            def observed(self: Any, *args: Any, _fn: Any = original, _key: str = key) -> Any:
                context._counts[_key] += 1
                return _fn(self, *args)

            method = MethodType(observed, model)
            setattr(model, name, method)
            context._observers[(model, name)] = method


def _allocate_context(
    model_config: Mapping[str, object],
    *,
    seed: int,
    instance_id: str,
    run_id: str,
    delta: float,
    max_steps: int,
    policy_context: Mapping[str, object] | None,
    sampling_context: Mapping[str, object] | None,
    tape: queue.ArrivalTape | None,
    initialize: bool,
) -> BranchRuntimeContextV1:
    source_identity()
    checked = queue.validate_queue_config(model_config)
    codec.integer(seed, "seed")
    codec.text(instance_id, "instance ID")
    codec.text(run_id, "run ID")
    codec.integer(max_steps, "max_steps", 1, codec.MAX_STEPS)
    if codec.number(delta, "delta") == 0:
        codec.fail("delta must be positive")
    if checked["arrival_spec"]["kind"] == "bernoulli-slots":
        codec.integer(checked["arrival_spec"]["slot_count"], "slot_count", 1, codec.MAX_STEPS)
    elif len(checked["arrival_spec"]["events"]) > codec.MAX_STEPS:
        codec.fail("explicit tape exceeds bounded profile")
    policy = codec.policy_context(policy_context)
    sampling = codec.sampling_context(sampling_context, run_id)
    result = BranchRuntimeContextV1()
    result._operation_lock = threading.RLock()
    result._policy_bytes = codec.canonical(policy)
    result._sampling_bytes = codec.canonical(sampling)
    result._config_bytes = codec.canonical(checked)
    result._seed, result._delta, result._max_steps = seed, delta, max_steps
    result._factory_used = False
    result._executor = SysExecutor(1.0, ex_mode=ExecutionType.HLA_TIME)
    try:
        arrival = tape if tape is not None else queue._make_tape(checked, seed)
        result.graph = queue.QueueControlGraph(checked, arrival, result._executor.get_global_time)
        queue.register_queue_graph(result._executor, result.graph)
        result._binding = queue.bind_queue_system(result._executor, result.graph)
        _observe(result)
        result.env = PyJevSimEnv(
            result._factory,
            instance_id=instance_id,
            run_id=run_id,
            boundary=FixedDeltaBoundary(delta),
            max_steps=max_steps,
            plugin_version=PROFILE_ID,
        )
        if initialize:
            result.env.reset(seed=seed)
        else:
            # Allocate only. No initialize(), reset(), executor.step(), model
            # transition, RNG re-generation, or prefix replay is allowed here.
            result._factory_used = True
            result.env._binding = result._binding
            result.env._driver = ExecutorDriver(result._executor)
        _remember_shape(result)
        return result
    except BaseException as original:
        try:
            result._executor.terminate_simulation()
        except BaseException as cleanup:
            original.add_note(f"partial queue allocation cleanup failed: {cleanup!r}")
        raise


def _models(context: BranchRuntimeContextV1) -> dict[str, Any]:
    return {
        "dc": context.executor.dmc,
        "arrival-source": context.graph.source,
        "buffer-server": context.graph.server,
        "completion-sink": context.graph.sink,
    }


def _remember_shape(context: BranchRuntimeContextV1) -> None:
    objects = [context.env, context.env._driver, context.executor, context.graph, context._binding]
    objects.extend(_models(context).values())
    objects.extend(context.executor.product_port_map.values())
    context._shapes = [(obj, frozenset(vars(obj))) for obj in objects]
    context._binding_values = dict(vars(context._binding))


def _admit(context: BranchRuntimeContextV1, *, allow_terminal: bool = False) -> None:
    if type(context) is not BranchRuntimeContextV1:
        codec.fail("not a queue branch runtime", "unsupported_profile")
    source_identity()
    env, ex, graph = context.env, context.executor, context.graph
    if any(getattr(env, name) for name in _LIFECYCLE_FLAGS):
        codec.fail("environment is active, failed, or closed", "unsupported_boundary")
    if env._done and not allow_terminal:
        codec.fail("terminal/truncated snapshots are unsupported", "unsupported_boundary")
    if (
        type(env) is not PyJevSimEnv
        or type(env._driver) is not ExecutorDriver
        or type(ex) is not SysExecutor
        or type(graph) is not queue.QueueControlGraph
        or env._binding is not context._binding
        or env._driver.executor is not ex
    ):
        codec.fail("runtime ownership/type differs", "unsupported_state")
    ex = cast(SysExecutor, ex)
    for obj, expected in context._shapes:
        if frozenset(vars(obj)) != expected:
            codec.fail(f"unknown or missing state in {type(obj).__name__}", "unsupported_state")
    if vars(context._binding) != context._binding_values:
        codec.fail("binding callbacks were replaced", "unsupported_state")
    for (model, name), method in context._observers.items():
        if getattr(model, name) is not method:
            codec.fail("model callback/observer was replaced", "unsupported_state")
    if (
        ex.ex_mode != ExecutionType.HLA_TIME
        or ex.snapshot_manager is not None
        or ex._output_event_callback is not None
        or ex._track_uncaught
        or ex.input_event_queue
        or ex.output_event_queue
        or graph.sink.pending
        or ex.waiting_obj_map
        or ex._waiting_keys
        or ex._destructs_pending
        or ex.hierarchical_structure
        or ex.simulation_mode != SimulationMode.SIMULATION_IDLE
    ):
        codec.fail(
            "undrained queues, lifecycle work, or unsupported side effects", "unsupported_boundary"
        )
    if (
        env._factory != context._factory
        or type(env._boundary) is not FixedDeltaBoundary
        or env._boundary.delta != context._delta
        or env._max_steps != context._max_steps
        or env._executor_qualification is not None
        or env._require_claim_grade
        or env._driver._qualification_policy is not None
        or env._driver._semantic_evidence is not None
        or env._driver._closed
        or env._plugin_version != PROFILE_ID
    ):
        codec.fail("environment/driver control profile changed", "unsupported_state")
    if codec.canonical(graph.config) != context._config_bytes:
        codec.fail("model config changed", "unsupported_state")
    models = _models(context)
    for label, model in {**models, "graph": graph, "executor": ex}.items():
        name, model_type, inputs, outputs = _DECLARED_CONSTANTS[label]
        if (
            model._name != name
            or model.model_type is not model_type
            or type(model.external_input_ports) is not list
            or type(model.external_output_ports) is not list
            or model.external_input_ports != list(inputs)
            or model.external_output_ports != list(outputs)
        ):
            codec.fail(f"{label} declared name/type/ports differ", "unsupported_state")
    for cls, methods in _CLASS_METHODS.items():
        if any(getattr(cls, name, None) is not method for name, method in methods.items()):
            codec.fail(f"{cls.__name__} method implementation was replaced", "unsupported_state")
    for name, model in models.items():
        if type(model) is not _MODEL_TYPES[name]:
            codec.fail(f"{name} concrete model type differs", "unsupported_profile")
        for method_name, method in _CLASS_METHODS[type(model)].items():
            if (model, method_name) not in context._observers and (
                getattr(getattr(model, method_name, None), "__func__", None) is not method
            ):
                codec.fail(f"{name}.{method_name} callback was replaced", "unsupported_state")
    for name, method in _CLASS_METHODS[queue.QueueControlGraph].items():
        if getattr(getattr(graph, name, None), "__func__", None) is not method:
            codec.fail(f"graph.{name} callback was replaced", "unsupported_state")
    if set(ex.model_map) != set(codec.NAMES) or set(ex.product_port_map) != set(models.values()):
        codec.fail("dynamic model inventory is unsupported", "unsupported_state")
    wrappers = {}
    for name, model in models.items():
        found = ex.model_map[name]
        if len(found) != 1 or type(found[0]) is not BehaviorExecutor:
            codec.fail("executor wrapper differs", "unsupported_state")
        wrapper = found[0]
        wrappers[name] = wrapper
        if (
            wrapper.get_core_model() is not model
            or wrapper.parent is not ex
            or ex.product_port_map[model] is not wrapper
            or wrapper.model is not model
            or wrapper._obj_id != model.get_obj_id()
        ):
            codec.fail("model/executor ownership differs", "unsupported_state")
        if (
            wrapper._cached_destruct_time != math.inf
            or wrapper._destruct_t != math.inf
            or wrapper._cached_destruct_time != wrapper._destruct_t
            or wrapper._instance_t != 0
        ):
            codec.fail("finite or inconsistent cached model lifecycle", "unsupported_state")
        if any(
            getattr(model, key)
            for key in (
                "external_transition_map_tuple",
                "external_transition_map_state",
                "internal_transition_map_tuple",
                "internal_transition_map_state",
            )
        ):
            codec.fail("custom transition map is unsupported", "unsupported_state")
    if ex.active_obj_map != {wrapper._obj_id: wrapper for wrapper in wrappers.values()}:
        codec.fail("active model inventory differs", "unsupported_state")
    for model in (graph.source, graph.server):
        if (
            getattr(model.clock, "__self__", None) is not ex
            or getattr(model.clock, "__func__", None) is not SysExecutor.get_global_time
        ):
            codec.fail("clock callback is not bound to this engine", "unsupported_state")
    if (
        graph.source.tape is not graph.arrival_tape
        or type(graph.arrival_tape) is not queue.ArrivalTape
    ):
        codec.fail("arrival tape ownership differs", "unsupported_state")
    expected_graph = {
        (graph, "action"): [(graph.server, "action")],
        (graph.source, "arrival"): [(graph.server, "arrival")],
        (graph.server, "completed"): [(graph.sink, "completed")],
        (graph.sink, "completion"): [(graph, "completion")],
    }
    if graph.port_map != expected_graph or graph.model_map != {
        m.get_name(): m for m in (graph.source, graph.server, graph.sink)
    }:
        codec.fail("graph routing differs", "unsupported_state")
    expected_routes = {
        (ex, "action"): [(wrappers["buffer-server"], "action")],
        (wrappers["arrival-source"], "arrival"): [(wrappers["buffer-server"], "arrival")],
        (wrappers["buffer-server"], "completed"): [(wrappers["completion-sink"], "completed")],
        (wrappers["completion-sink"], "completion"): [(ex, "completion")],
    }
    if (
        ex.port_map != expected_routes
        or ex.external_input_ports != ["action"]
        or ex.external_output_ports != ["completion"]
    ):
        codec.fail("executor routing differs", "unsupported_state")
    calendar = ex.min_schedule_item
    if (
        type(calendar) is not ScheduleQueue
        or set(vars(calendar)) != {"_heap", "_mapped", "_reverse"}
        or type(calendar._heap) is not list
        or type(calendar._mapped) is not dict
        or type(calendar._reverse) is not dict
        or set(calendar._reverse) != {m._obj_id for m in wrappers.values()}
    ):
        codec.fail("event calendar inventory differs", "unsupported_state")
    if any(
        type(instant) not in (int, float) or math.isnan(instant) or instant < 0
        for instant in calendar._heap
    ):
        codec.fail("event calendar heap contains an unsupported time", "unsupported_state")
    members: set[Any] = set()
    for instant, bucket in calendar._mapped.items():
        if (
            type(bucket) is not set
            or type(instant) not in (int, float)
            or math.isnan(instant)
            or instant < 0
        ):
            codec.fail("event calendar bucket representation differs", "unsupported_state")
        if bucket and instant <= ex.global_time:
            codec.fail("unprocessed event at capture boundary", "unsupported_boundary")
        for wrapper in bucket:
            if (
                wrapper in members
                or wrapper not in wrappers.values()
                or wrapper.request_time != instant
                or calendar._reverse.get(wrapper._obj_id) != instant
            ):
                codec.fail("event calendar membership differs", "unsupported_state")
            members.add(wrapper)
    if members != set(wrappers.values()):
        codec.fail("event calendar is incomplete", "unsupported_state")
    live_times = {instant for instant, bucket in calendar._mapped.items() if bucket}
    if not live_times.issubset(set(calendar._heap)) or any(
        calendar._heap[(index - 1) // 2] > instant
        for index, instant in enumerate(calendar._heap)
        if index
    ):
        codec.fail("event calendar heap is inconsistent", "unsupported_state")
    if env._observation != graph.observe(ex.global_time):
        codec.fail("cached observation differs from physical state", "unsupported_state")


def _state(context: BranchRuntimeContextV1) -> dict[str, Any]:
    ex, env = context.executor, context.env
    models: dict[str, Any] = {}
    wrappers: dict[str, Any] = {}
    calendar: dict[str, Any] = {}
    for name, model in _models(context).items():
        behavior = {
            "states": {key: codec.time_out(value) for key, value in model._states.items()},
            "cur_state": model._cur_state,
            "global_time": model.global_time,
            "cancel_reschedule": model._cancel_reschedule_f,
        }
        if name == "arrival-source":
            values = {"index": model.index, "exhausted": model.exhausted}
        elif name == "buffer-server":
            values = {key: getattr(model, key) for key in codec.SERVER_FIELDS}
        elif name == "completion-sink":
            values = {"ledger": model.ledger, "pending": model.pending}
        else:
            values = {}
        models[name] = {"behavior": behavior, "values": values}
        wrapper = ex.product_port_map[model]
        wrappers[name] = {
            "global_time": wrapper.global_time,
            "next_event_time": codec.time_out(wrapper._next_event_t),
            "cur_state": wrapper._cur_state,
            "request_time": codec.time_out(wrapper.request_time),
            "cancel_reschedule": wrapper._cancel_reschedule_f,
            "instance_time": wrapper._instance_t,
            "destruct_time": codec.time_out(wrapper._destruct_t),
            "engine_name": wrapper.engine_name,
        }
        calendar[name] = codec.time_out(ex.min_schedule_item._reverse[wrapper._obj_id])
    tape = context.graph.arrival_tape
    return cast(
        dict[str, Any],
        _detached(
            {
                "config": json.loads(context._config_bytes),
                "config_sha256": hashlib.sha256(context._config_bytes).hexdigest(),
                "seed": context._seed,
                "delta": context._delta,
                "max_steps": context._max_steps,
                "tape": {"seed": tape.seed, "events": tape.events, "end_time": tape.end_time},
                "models": models,
                "executor": {
                    "global_time": ex.global_time,
                    "target_time": ex.target_time,
                    "time_resolution": ex.time_resolution,
                    "simulation_mode": ex.simulation_mode.name,
                    "calendar": calendar,
                    "models": wrappers,
                },
                "environment": {
                    "instance_id": env._instance_id,
                    "run_id": env._run_id,
                    "episode_number": env._episode_number,
                    "step_id": env._step_id,
                    "observation": env._observation,
                    "seed": env._seed,
                    "done": env._done,
                    "failed": env._failed,
                    "driver_has_advanced": cast(ExecutorDriver, env._driver)._has_advanced,
                },
            }
        ),
    )


def state_view(context: BranchRuntimeContextV1) -> dict[str, Any]:
    """Detached physical state; only execution IDs/work counters are omitted."""
    with context._operation_lock, context.env._lifecycle_lock:
        _admit(context, allow_terminal=True)
        state = _state(context)
        state["environment"].pop("instance_id")
        state["environment"].pop("run_id")
        return state


def supports(context: BranchRuntimeContextV1) -> SnapshotCapabilityV1:
    try:
        if type(context) is not BranchRuntimeContextV1:
            codec.fail("not a queue branch runtime", "unsupported_profile")
        if not context._operation_lock.acquire(blocking=False):
            codec.fail("context operation in flight", "unsupported_boundary")
        try:
            if not context.env._lifecycle_lock.acquire(blocking=False):
                codec.fail("environment operation in flight", "unsupported_boundary")
            try:
                _admit(context)
                codec.validate_state(_state(context))
            finally:
                context.env._lifecycle_lock.release()
        finally:
            context._operation_lock.release()
    except (QueueSnapshotError, ValueError, TypeError, AttributeError, KeyError) as exc:
        return SnapshotCapabilityV1(False, str(exc))
    return SnapshotCapabilityV1(True)


def _encode_snapshot(unsigned: dict[str, Any]) -> bytes:
    result = codec.canonical({**unsigned, "state_sha256": codec.digest(unsigned)})
    if len(result) > codec.MAX_BYTES:
        codec.fail("snapshot exceeds bounded byte limit", "capture_failure")
    return result


def capture(context: BranchRuntimeContextV1, prefix_identity: str) -> SimulatorSnapshotV1:
    codec.sha(prefix_identity, "prefix_identity")
    if type(context) is not BranchRuntimeContextV1:
        codec.fail("not a queue branch runtime", "unsupported_profile")
    if not context._operation_lock.acquire(blocking=False):
        codec.fail("context operation in flight", "unsupported_boundary")
    state: dict[str, Any] | None = None
    try:
        if not context.env._lifecycle_lock.acquire(blocking=False):
            codec.fail("environment operation in flight", "unsupported_boundary")
        try:
            _admit(context)
            state = codec.validate_state(_state(context))
            unsigned = {
                "schema_version": codec.SCHEMA,
                "profile_id": PROFILE_ID,
                "sources": source_identity(),
                "prefix_identity": prefix_identity,
                "policy_context": context.policy_context,
                "sampling_context": context.sampling_context,
                "state": state,
            }
            data = _encode_snapshot(unsigned)
            if _state(context) != state:
                context.env._failed = True
                codec.fail("source changed during capture", "capture_failure")
            return SimulatorSnapshotV1(data)
        finally:
            context.env._lifecycle_lock.release()
    except BaseException as exc:
        if state is not None:
            try:
                if _state(context) != state:
                    context.env._failed = True
                    exc.add_note("source invalidated: it changed during failed capture")
            except BaseException:
                context.env._failed = True
                exc.add_note("source invalidated: state could not be checked after failed capture")
        if isinstance(exc, QueueSnapshotError) or not isinstance(exc, Exception):
            raise
        raise QueueSnapshotError("capture_failure", str(exc)[:1024]) from exc
    finally:
        context._operation_lock.release()


def _apply_state(context: BranchRuntimeContextV1, state: dict[str, Any]) -> None:
    ex, env = context.executor, context.env
    saved = state["executor"]
    ex.global_time = saved["global_time"]
    ex.target_time = saved["target_time"]
    ex.time_resolution = saved["time_resolution"]
    ex.simulation_mode = SimulationMode[saved["simulation_mode"]]
    ex.active_obj_map = {}
    ex.waiting_obj_map = {}
    ex._waiting_keys = []
    ex.min_schedule_item = ScheduleQueue()
    for name, model in _models(context).items():
        row = state["models"][name]
        behavior = row["behavior"]
        model._states = {
            key: codec.time_in(value, "deadline") for key, value in behavior["states"].items()
        }
        model._cur_state = behavior["cur_state"]
        model.global_time = behavior["global_time"]
        model._cancel_reschedule_f = behavior["cancel_reschedule"]
        for key, value in row["values"].items():
            # Rebuild trace tuples exactly as the model ordinarily appends them.
            setattr(model, key, [tuple(item) for item in value] if key == "trace" else value)
        wrapper = ex.product_port_map[model]
        values = saved["models"][name]
        wrapper.global_time = values["global_time"]
        wrapper._next_event_t = codec.time_in(values["next_event_time"], "next event")
        wrapper._cur_state = values["cur_state"]
        wrapper.request_time = codec.time_in(values["request_time"], "request time")
        wrapper._cancel_reschedule_f = values["cancel_reschedule"]
        ex.active_obj_map[wrapper._obj_id] = wrapper
        # push() only rebuilds the future calendar; it invokes no model callback.
        ex.min_schedule_item.push(wrapper)
    values = state["environment"]
    env._episode_number = values["episode_number"]
    env._step_id = values["step_id"]
    env._observation = values["observation"]
    env._seed = values["seed"]
    env._done = values["done"]
    env._failed = values["failed"]
    cast(ExecutorDriver, env._driver)._has_advanced = values["driver_has_advanced"]
    _remember_shape(context)


def restore(
    snapshot: SimulatorSnapshotV1,
    branch_context: Mapping[str, object] | None = None,
) -> BranchRuntimeContextV1:
    if type(snapshot) is not SimulatorSnapshotV1:
        codec.fail("restore requires a SimulatorSnapshotV1")
    raw = _read_snapshot(snapshot.data)
    state = raw["state"]
    branch = {} if branch_context is None else dict(branch_context)
    if set(branch) - {"instance_id", "run_id", "sampling_context"}:
        codec.fail("branch context contains unsupported overrides", "invalid_branch")
    env = state["environment"]
    instance_id = cast(str, branch.get("instance_id", env["instance_id"]))
    run_id = cast(str, branch.get("run_id", env["run_id"]))
    sampling = codec.sampling_context(
        cast(Mapping[str, object] | None, branch.get("sampling_context", raw["sampling_context"])),
        run_id,
    )
    if any(
        sampling[key] != raw["sampling_context"][key]
        for key in ("domain", "phase", "master", "run_id", "generation")
    ):
        codec.fail("branch changed family/prefix sampling namespace", "invalid_branch")
    tape = state["tape"]
    result: BranchRuntimeContextV1 | None = None
    try:
        result = _allocate_context(
            state["config"],
            seed=state["seed"],
            instance_id=instance_id,
            run_id=run_id,
            delta=state["delta"],
            max_steps=state["max_steps"],
            policy_context=raw["policy_context"],
            sampling_context=sampling,
            tape=queue.ArrivalTape(
                tape["seed"], tuple(tuple(item) for item in tape["events"]), tape["end_time"]
            ),
            initialize=False,
        )
        _apply_state(result, state)
        _admit(result)
        expected = _detached(state)
        expected["environment"]["instance_id"] = instance_id
        expected["environment"]["run_id"] = run_id
        if _state(result) != expected or result.event_count != 0:
            codec.fail("restored state differs or executed model transitions", "restore_failure")
        return result
    except BaseException as original:
        if result is not None:
            try:
                result.close()
            except BaseException as cleanup:
                original.add_note(f"partial queue restore cleanup failed: {cleanup!r}")
        if isinstance(original, QueueSnapshotError) or not isinstance(original, Exception):
            raise
        raise QueueSnapshotError("restore_failure", str(original)[:1024]) from original
