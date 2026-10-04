"""Original finite-buffer DEVS queue; fixed one-level graph, not native hierarchy.

The source, combined FIFO/server and completion sink are real BehaviorModels.
All event arithmetic uses the executor clock; observation is a nonmutating
projection to a committed boundary. This module contains no learning algorithm.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from random import Random
from typing import Any, cast

from pyjevsim.behavior_model import BehaviorModel
from pyjevsim.definition import ExecutionType
from pyjevsim.structural_model import StructuralModel
from pyjevsim.system_executor import SysExecutor
from pyjevsim.system_message import SysMessage

from pyjevsim_bridge.rl.adapters import FunctionalEpisodeBinding
from pyjevsim_bridge.rl.contracts import EpisodeContext, StepView

MODEL_ID = "FiniteBufferQueueControl"
MODEL_VERSION = "1"
CONFIG_SCHEMA = "queue-control-config-v1"
OBSERVATION_SCHEMA = "queue-control-observation-v1"
MODES = ("idle", "normal", "fast")
RATES = (0.0, 1.0, 2.0)
POWERS = (0.0, 1.0, 4.0)
WORK_TOLERANCE = 1e-12
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "waiting_capacity",
        "initial_service",
        "initial_waiting",
        "initial_mode",
        "arrival_spec",
        "cost_weights",
    }
)
OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "logical_time",
        "source_end_time",
        "waiting_capacity",
        "waiting_job_ids",
        "in_service_id",
        "remaining_work",
        "mode",
        "initial_job_count",
        "source_arrivals",
        "admitted",
        "completed",
        "dropped",
        "backlog_integral",
        "energy_integral",
        "cumulative_cost",
        "source_exhausted",
    }
)


class QueueConfigurationError(ValueError):
    """Closed queue configuration or action is invalid."""


class QueueStateError(RuntimeError):
    """Physical queue state or time arithmetic violated an invariant."""


def _object(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise QueueConfigurationError(f"{name} fields differ from the closed schema")
    return dict(value)


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise QueueConfigurationError(f"{name} must be finite numeric, not bool")
    try:
        result = float(cast(int | float, value))
    except OverflowError as exc:
        raise QueueConfigurationError(f"{name} exceeds finite numeric range") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise QueueConfigurationError(f"{name} is outside its finite nonnegative domain")
    return result


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise QueueConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _job(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise QueueConfigurationError("job IDs must be nonempty strings")
    return value


def action_mode(action: object) -> str:
    value = _object(action, frozenset({"mode"}), "action")["mode"]
    if not isinstance(value, str) or value not in MODES:
        raise QueueConfigurationError("unknown queue service mode")
    return value


def validate_queue_config(value: object) -> dict[str, Any]:
    """Validate/detach all inputs before any SysExecutor is constructed."""
    config = _object(value, _CONFIG_FIELDS, "queue config")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise QueueConfigurationError("queue config schema differs")
    capacity = _integer(config["waiting_capacity"], "waiting_capacity", 1)
    if capacity > 16:
        raise QueueConfigurationError("waiting_capacity must not exceed16")
    service = config["initial_service"]
    if service is not None:
        _job(service)
    waiting = config["initial_waiting"]
    if not isinstance(waiting, (list, tuple)) or len(waiting) > capacity:
        raise QueueConfigurationError("initial waiting queue exceeds capacity or is not a sequence")
    waiting = [_job(item) for item in waiting]
    seen = ([service] if service is not None else []) + waiting
    if len(set(seen)) != len(seen) or (service is None and waiting):
        raise QueueConfigurationError("duplicate or orphaned initial jobs")
    mode = action_mode({"mode": config["initial_mode"]})
    if service is None and mode != "idle":
        raise QueueConfigurationError("empty initial state requires idle")
    weights = _object(config["cost_weights"], frozenset({"backlog", "energy", "drop"}), "weights")
    weights = {key: _number(item, key) for key, item in weights.items()}
    arrival = config["arrival_spec"]
    if not isinstance(arrival, Mapping):
        raise QueueConfigurationError("arrival_spec must be an object")
    if arrival.get("kind") == "bernoulli-slots":
        arrival = _object(
            arrival, frozenset({"kind", "slot_count", "slot_width", "probability"}), "arrival_spec"
        )
        count = _integer(arrival["slot_count"], "slot_count", 1)
        width = _number(arrival["slot_width"], "slot_width", positive=True)
        probability = _number(arrival["probability"], "probability")
        try:
            end_time = count * width
        except OverflowError as exc:
            raise QueueConfigurationError("source end time exceeds finite range") from exc
        if probability > 1 or not math.isfinite(end_time):
            raise QueueConfigurationError("invalid probability or source end time")
        if any(
            job.startswith("arrival-") and job[8:].isdigit() and 1 <= int(job[8:]) <= count
            for job in seen
        ):
            raise QueueConfigurationError("initial ID collides with generated arrival IDs")
        arrival = {
            "kind": "bernoulli-slots",
            "slot_count": count,
            "slot_width": width,
            "probability": probability,
        }
    elif arrival.get("kind") == "explicit":
        arrival = _object(arrival, frozenset({"kind", "events", "end_time"}), "arrival_spec")
        end_time = _number(arrival["end_time"], "end_time", positive=True)
        if not isinstance(arrival["events"], (list, tuple)):
            raise QueueConfigurationError("explicit events must be a sequence")
        events = []
        last = 0.0
        for raw in arrival["events"]:
            event = _object(raw, frozenset({"time", "job_id"}), "arrival event")
            instant = _number(event["time"], "arrival time", positive=True)
            job = _job(event["job_id"])
            if instant < last or instant > end_time or job in seen:
                raise QueueConfigurationError("arrival time order/range or job ID invalid")
            seen.append(job)
            events.append({"time": instant, "job_id": job})
            last = instant
        arrival = {"kind": "explicit", "events": events, "end_time": end_time}
    else:
        raise QueueConfigurationError("unknown arrival specification")
    return {
        "schema_version": CONFIG_SCHEMA,
        "waiting_capacity": capacity,
        "initial_service": service,
        "initial_waiting": waiting,
        "initial_mode": mode,
        "arrival_spec": arrival,
        "cost_weights": weights,
    }


def _residual(value: float) -> float:
    # Only arithmetic after a scheduled event or a nonmutating projection uses
    # this clamp. It never moves an event or decides whether a job completed.
    if not math.isfinite(value) or value < -WORK_TOLERANCE:
        raise QueueStateError("remaining work below the exact1e-12 tolerance")
    return max(0.0, value)


@dataclass(frozen=True)
class ArrivalTape:
    seed: int
    events: tuple[tuple[float, str], ...]
    end_time: float

    def content(self) -> dict[str, object]:
        return {
            "schema_version": "queue-arrival-tape-v1",
            "end_time": self.end_time,
            "events": [{"time": instant, "job_id": job} for instant, job in self.events],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.content(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


def _make_tape(config: Mapping[str, Any], seed: int) -> ArrivalTape:
    arrival = config["arrival_spec"]
    if arrival["kind"] == "explicit":
        return ArrivalTape(
            seed,
            tuple((row["time"], row["job_id"]) for row in arrival["events"]),
            arrival["end_time"],
        )
    rng = Random(seed)  # noqa: S311 -- reproducible simulation, not cryptography.
    events = tuple(
        (index * arrival["slot_width"], f"arrival-{index}")
        for index in range(1, arrival["slot_count"] + 1)
        if rng.random() < arrival["probability"]
    )
    return ArrivalTape(seed, events, arrival["slot_count"] * arrival["slot_width"])


# Upstream pyjevsim model bases have no stubs; only subclass checks are suppressed.
class ArrivalSource(BehaviorModel):  # type: ignore[misc]
    def __init__(self, tape: ArrivalTape, clock: Callable[[], float]) -> None:
        super().__init__("arrival-source")
        self.tape, self.clock = tape, clock
        self.index = 0
        self.exhausted = False
        self.insert_state("active", self._next_time())
        self.init_state("active")
        self.insert_output_port("arrival")

    def _next_time(self) -> float:
        return (
            self.tape.events[self.index][0]
            if self.index < len(self.tape.events)
            else self.tape.end_time
        )

    def output(self, deliverer: Any) -> None:
        now = float(self.clock())
        index = self.index
        while index < len(self.tape.events) and self.tape.events[index][0] == now:
            message = SysMessage(self.get_name(), "arrival")
            message.insert(self.tape.events[index][1])
            deliverer.insert_message(message)
            index += 1

    def int_trans(self) -> None:
        now = float(self.clock())
        while self.index < len(self.tape.events) and self.tape.events[self.index][0] == now:
            self.index += 1
        self.exhausted = now == self.tape.end_time
        self.update_state("active", math.inf if self.exhausted else self._next_time() - now)

    def ext_trans(self, port: str, message: Any) -> None:
        raise QueueStateError("arrival source has no input ports")


class BufferServer(BehaviorModel):  # type: ignore[misc]
    """One atomic FIFO/server, so the capacity slot is freed in delta_internal."""

    def __init__(self, config: Mapping[str, Any], clock: Callable[[], float]) -> None:
        super().__init__("buffer-server")
        self.clock = clock
        self.capacity = config["waiting_capacity"]
        self.current: str | None = config["initial_service"]
        self.waiting: list[str] = list(config["initial_waiting"])
        self.remaining = 1.0 if self.current is not None else 0.0
        self.mode: str = config["initial_mode"]
        self.initial_count = len(self.waiting) + int(self.current is not None)
        self.admitted = self.initial_count
        self.source_arrivals = self.completed = self.dropped = 0
        self.last_event_time = self.backlog = self.energy = 0.0
        self.trace: list[tuple[str, float, object]] = []
        self.insert_state("active")
        self.init_state("active")
        self.insert_input_port("arrival")
        self.insert_input_port("action")
        self.insert_output_port("completed")
        self._schedule()

    def project(self, now: float) -> tuple[float, float, float]:
        elapsed = now - self.last_event_time
        if not math.isfinite(elapsed) or elapsed < 0:
            raise QueueStateError("queue event/projection time regressed or is nonfinite")
        busy = self.current is not None
        mode = MODES.index(self.mode)
        remaining = _residual(self.remaining - RATES[mode] * elapsed) if busy else 0.0
        backlog = self.backlog + (len(self.waiting) + int(busy)) * elapsed
        energy = self.energy + (POWERS[mode] * elapsed if busy else 0.0)
        if not math.isfinite(backlog) or not math.isfinite(energy):
            raise QueueStateError("queue objective integral overflow")
        return remaining, backlog, energy

    def _advance(self) -> None:
        now = float(self.clock())
        self.remaining, self.backlog, self.energy = self.project(now)
        self.last_event_time = now

    def _schedule(self) -> None:
        rate = RATES[MODES.index(self.mode)]
        self.update_state(
            "active", self.remaining / rate if self.current is not None and rate else math.inf
        )

    def output(self, deliverer: Any) -> None:
        if self.current is None:
            raise QueueStateError("empty server cannot emit a completion")
        now = float(self.clock())
        self.trace.append(("output", now, self.current))
        message = SysMessage(self.get_name(), "completed")
        message.insert({"job_id": self.current, "completion_time": now})
        deliverer.insert_message(message)

    def int_trans(self) -> None:
        if self.current is None:
            raise QueueStateError("empty server cannot complete")
        self._advance()
        self.trace.append(("internal", self.last_event_time, self.current))
        self.completed += 1
        self.current = self.waiting.pop(0) if self.waiting else None
        self.remaining = 1.0 if self.current is not None else 0.0
        if self.current is None:
            self.mode = "idle"
        self._schedule()

    def ext_trans(self, port: str, message: Any) -> None:
        self._advance()
        for value in message.retrieve():
            self.trace.append((f"external-{port}", self.last_event_time, value))
            if port == "action":
                mode = action_mode(value)
                if self.current is None and mode != "idle":
                    raise QueueConfigurationError("empty queue permits only idle")
                self.mode = mode
            elif port == "arrival":
                job = _job(value)
                self.source_arrivals += 1
                if self.current is None:
                    self.current, self.remaining = job, 1.0
                    self.admitted += 1
                elif len(self.waiting) < self.capacity:
                    self.waiting.append(job)
                    self.admitted += 1
                else:
                    self.dropped += 1
            else:
                raise QueueStateError("unknown server input port")
        self._schedule()


class CompletionSink(BehaviorModel):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__("completion-sink")
        self.ledger: list[dict[str, object]] = []
        self.pending: list[dict[str, object]] = []
        self.insert_state("active")
        self.init_state("active")
        self.insert_input_port("completed")
        self.insert_output_port("completion")

    def ext_trans(self, port: str, message: Any) -> None:
        if port != "completed":
            raise QueueStateError("unknown sink input port")
        for item in message.retrieve():
            payload = dict(item)
            self.ledger.append(payload)
            self.pending.append(payload)
        self.update_state("active", 0.0)

    def output(self, deliverer: Any) -> None:
        for payload in self.pending:
            message = SysMessage(self.get_name(), "completion")
            message.insert(dict(payload))
            deliverer.insert_message(message)

    def int_trans(self) -> None:
        self.pending.clear()
        self.update_state("active", math.inf)


class QueueControlGraph(StructuralModel):  # type: ignore[misc]
    """Declarative one-level graph; never registered through StructuralExecutor."""

    def __init__(
        self, config: Mapping[str, Any], tape: ArrivalTape, clock: Callable[[], float]
    ) -> None:
        super().__init__("queue-control")
        self.config = config
        self.arrival_tape = tape
        self.source = ArrivalSource(tape, clock)
        self.server = BufferServer(config, clock)
        self.sink = CompletionSink()
        self.insert_input_port("action")
        self.insert_output_port("completion")
        for leaf in (self.source, self.server, self.sink):
            self.register_entity(leaf)
        self.coupling_relation(self, "action", self.server, "action")
        self.coupling_relation(self.source, "arrival", self.server, "arrival")
        self.coupling_relation(self.server, "completed", self.sink, "completed")
        self.coupling_relation(self.sink, "completion", self, "completion")

    def observe(self, now: float) -> dict[str, object]:
        server = self.server
        remaining, backlog, energy = server.project(float(now))
        if (
            server.admitted
            != len(server.waiting) + int(server.current is not None) + server.completed
            or server.initial_count + server.source_arrivals != server.admitted + server.dropped
            or len(self.sink.ledger) != server.completed
            or not 0 <= len(server.waiting) <= server.capacity
            or (server.current is None and server.waiting)
        ):
            raise QueueStateError("queue conservation invariant failed")
        weights = self.config["cost_weights"]
        cost = (
            weights["backlog"] * backlog
            + weights["energy"] * energy
            + weights["drop"] * server.dropped
        )
        if not math.isfinite(cost):
            raise QueueStateError("queue cumulative cost overflow")
        return {
            "schema_version": OBSERVATION_SCHEMA,
            "logical_time": float(now),
            "source_end_time": self.arrival_tape.end_time,
            "waiting_capacity": server.capacity,
            "waiting_job_ids": list(server.waiting),
            "in_service_id": server.current,
            "remaining_work": remaining,
            "mode": server.mode,
            "initial_job_count": server.initial_count,
            "source_arrivals": server.source_arrivals,
            "admitted": server.admitted,
            "completed": server.completed,
            "dropped": server.dropped,
            "backlog_integral": backlog,
            "energy_integral": energy,
            "cumulative_cost": cost,
            "source_exhausted": self.source.exhausted,
        }


def register_queue_graph(executor: Any, graph: QueueControlGraph) -> None:
    """Translate only this exact three-leaf declared graph using public APIs."""
    if type(graph) is not QueueControlGraph:
        raise QueueConfigurationError("registration accepts only QueueControlGraph")
    leaves: tuple[tuple[Any, type[Any], str, list[str], list[str]], ...] = (
        (graph.source, ArrivalSource, "arrival-source", [], ["arrival"]),
        (graph.server, BufferServer, "buffer-server", ["arrival", "action"], ["completed"]),
        (graph.sink, CompletionSink, "completion-sink", ["completed"], ["completion"]),
    )
    for leaf, expected_type, name, inputs, outputs in leaves:
        if (type(leaf) is not expected_type or leaf.get_name() != name
                or leaf.external_input_ports != inputs or leaf.external_output_ports != outputs):
            raise QueueConfigurationError("queue leaf type, name or declared ports differ")
    expected_models = {leaf.get_name(): leaf for leaf in (graph.source, graph.server, graph.sink)}
    expected: dict[tuple[Any, str], list[tuple[Any, str]]] = {
        (graph, "action"): [(graph.server, "action")],
        (graph.source, "arrival"): [(graph.server, "arrival")],
        (graph.server, "completed"): [(graph.sink, "completed")],
        (graph.sink, "completion"): [(graph, "completion")],
    }
    if (
        graph.get_models() != expected_models
        or len(expected_models) != 3
        or graph.get_couplings() != expected
        or graph.external_input_ports != ["action"]
        or graph.external_output_ports != ["completion"]
    ):
        raise QueueConfigurationError("queue graph or declared wiring differs")
    for leaf in (graph.source, graph.server, graph.sink):
        executor.register_entity(leaf)
    executor.insert_input_port("action")
    executor.insert_output_port("completion")
    for (source, source_port), destinations in expected.items():
        for destination, destination_port in destinations:
            executor.coupling_relation(
                None if source is graph else source,
                source_port,
                None if destination is graph else destination,
                destination_port,
            )


def build_queue_system(config: object, *, seed: int) -> tuple[Any, QueueControlGraph]:
    checked = validate_queue_config(config)
    checked_seed = _integer(seed, "seed")
    tape = _make_tape(checked, checked_seed)
    executor = SysExecutor(1.0, ex_mode=ExecutionType.HLA_TIME)
    try:
        graph = QueueControlGraph(checked, tape, executor.get_global_time)
        register_queue_graph(executor, graph)
        return executor, graph
    except BaseException:
        executor.terminate_simulation()
        raise


def _view_observations(view: StepView) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    return (
        cast(Mapping[str, Any], view.previous_observation),
        cast(Mapping[str, Any], view.observation),
    )


def bind_queue_system(executor: Any, graph: QueueControlGraph) -> FunctionalEpisodeBinding:
    """Model-owned physical/observation adapter; useful for independent drivers."""

    def apply_action(_executor: Any, action: object) -> None:
        mode = action_mode(action)
        if graph.server.current is None and mode != "idle":
            raise QueueConfigurationError("empty queue permits only idle")
        _executor.insert_external_event("action", {"mode": mode}, scheduled_time=0.0)

    def reward(view: StepView) -> float:
        previous, current = _view_observations(view)
        return cast(float, -(current["cumulative_cost"] - previous["cumulative_cost"]))

    def terminal(view: StepView) -> bool:
        _, current = _view_observations(view)
        return bool(
            current["source_exhausted"]
            and current["in_service_id"] is None
            and not current["waiting_job_ids"]
            and not graph.sink.pending
        )

    def info(view: StepView) -> dict[str, object]:
        _, current = _view_observations(view)
        result = {
            "objective_cost": -reward(view),
            "cumulative_cost": current["cumulative_cost"],
            "arrival_tape_sha256": graph.arrival_tape.sha256,
            "unfinished_inventory": len(current["waiting_job_ids"])
            + int(current["in_service_id"] is not None),
        }
        if view.step_id == 1:
            result["arrival_tape"] = graph.arrival_tape.content()
        return result

    def initialize(ex: Any) -> None:
        ex.step(0.0)

    return FunctionalEpisodeBinding(
        executor=executor,
        apply_action_fn=apply_action,
        observe_fn=lambda ex, _events: graph.observe(ex.get_global_time()),
        reward_fn=reward,
        terminated_fn=terminal,
        info_fn=info,
        initialize_fn=initialize,
        close_fn=executor.terminate_simulation,
    )


def make_queue_episode(context: EpisodeContext) -> FunctionalEpisodeBinding:
    executor, graph = build_queue_system(context.options, seed=_integer(context.seed, "seed"))
    return bind_queue_system(executor, graph)
