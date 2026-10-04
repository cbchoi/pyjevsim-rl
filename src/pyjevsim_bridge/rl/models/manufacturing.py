"""Static-flat two-stage manufacturing cell with an exclusive shared tool.

Only the arbiter writes ToolLedger. Stages receive authority through grant
messages; their ledger aliases are observation-only, never transition inputs.
The arbiter selects from requests actually received by its zero-time dispatch:
maintenance first, then job sequence and stage name. Requests arriving in a
later zero-time round compete at the next dispatch, not retroactively. Native
inherited confluence and scheduler ordering are unchanged.
"""
from __future__ import annotations

import math
import hashlib
from collections.abc import Mapping
from pathlib import Path
from random import Random

from pyjevsim.behavior_model import BehaviorModel
from pyjevsim.structural_model import StructuralModel
from pyjevsim.system_message import SysMessage
from pyjevsim.system_executor import SysExecutor
from pyjevsim.definition import ExecutionType

_IMPORT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CONFIG_SCHEMA = "manufacturing-config-v1"
OBS_SCHEMA = "manufacturing-observation-v1"
CONFIG_FIELDS = {"schema_version", "arrivals", "maintenance_times", "stage_a_time",
                 "stage_b_time", "repair_time", "repair_jitter", "wip_weight", "completion_value"}
OBS_FIELDS = {"schema_version", "logical_time", "released", "completed", "wip_integral",
              "cumulative_cost", "owner", "repair_remaining", "maintenance_pending",
              "stage_a_jobs", "stage_b_jobs", "completion_ledger", "source_exhausted", "rng_draws"}


def closed(value, keys, label):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{label}: closed fields differ")
    return dict(value)


def number(value, label, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{label}: finite nonnegative number required")
    return float(value)


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: integer required")
    return value


def validate_config(raw):
    cfg = closed(raw, CONFIG_FIELDS, "manufacturing config")
    if cfg["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("manufacturing schema differs")
    for key in CONFIG_FIELDS - {"schema_version", "arrivals", "maintenance_times"}:
        cfg[key] = number(cfg[key], key, key in {"stage_a_time", "stage_b_time", "repair_time"})
    for key in ("arrivals", "maintenance_times"):
        if type(cfg[key]) not in (list, tuple) or len(cfg[key]) > 10000:
            raise ValueError("bounded event sequence required")
        cfg[key] = [number(x, key) for x in cfg[key]]
        if cfg[key] != sorted(cfg[key]):
            raise ValueError("event times must be nondecreasing")
    if not cfg["arrivals"]:
        raise ValueError("at least one job required")
    return cfg


def hand_config():
    return {"schema_version": CONFIG_SCHEMA, "arrivals": [0.0, 1.5], "maintenance_times": [2.0],
            "stage_a_time": 1.0, "stage_b_time": 2.0, "repair_time": 1.0,
            "repair_jitter": 0.0, "wip_weight": 1.0, "completion_value": 0.0}


def tuple_tree(value):
    return tuple(tuple_tree(x) for x in value) if isinstance(value, (list, tuple)) else value


def validate_rng(raw):
    if type(raw) not in (list, tuple) or len(raw) != 3 or raw[0] != 3:
        raise ValueError("MT19937 version3 state required")
    inner = raw[1]
    if type(inner) not in (list, tuple) or len(inner) != 625:
        raise ValueError("MT19937 state length differs")
    if any(type(x) is not int or not 0 <= x <= 0xFFFFFFFF for x in inner[:-1]):
        raise ValueError("MT19937 words invalid")
    if type(inner[-1]) is not int or not 0 <= inner[-1] <= 624:
        raise ValueError("MT19937 index invalid")
    if raw[2] is not None and (type(raw[2]) not in (int, float) or not math.isfinite(raw[2])):
        raise ValueError("Gaussian cache invalid")


def validate_observation(raw):
    row = closed(raw, OBS_FIELDS, "manufacturing observation")
    if row["schema_version"] != OBS_SCHEMA or type(row["source_exhausted"]) is not bool:
        raise ValueError("observation schema/source state differs")
    for key in ("logical_time", "wip_integral", "cumulative_cost", "repair_remaining"):
        number(row[key], key)
    for key in ("released", "completed", "maintenance_pending", "rng_draws"):
        integer(row[key], key)
    if row["owner"] is not None:
        owner = closed(row["owner"], {"stage", "job"}, "tool owner")
        if owner["stage"] not in ("stage-a", "stage-b"):
            raise ValueError("unknown tool owner stage")
        integer(owner["job"], "owner job", 1)
    for key in ("stage_a_jobs", "stage_b_jobs"):
        if type(row[key]) is not list:
            raise ValueError("job inventory must be list")
        for job in row[key]:
            integer(job, "job", 1)
    if type(row["completion_ledger"]) is not list:
        raise ValueError("completion ledger must be list")
    for item in row["completion_ledger"]:
        closed(item, {"job", "time"}, "completion")
        integer(item["job"], "completed job", 1)
        number(item["time"], "completion time")


def validate_reward_state(raw):
    row = closed(raw, {"last_cost", "last_completed"}, "reward state")
    number(row["last_cost"], "last_cost")
    integer(row["last_completed"], "last_completed")


def emit(deliverer, model, port, payload):
    message = SysMessage(model.get_name(), port)
    message.insert(payload)
    deliverer.insert_message(message)


class ToolLedger:
    def __init__(self):
        self.owner = None
        self.repair_until = None
        self.trace = []


class JobSource(BehaviorModel):
    def __init__(self, config, clock):
        super().__init__("job-source")
        self.config, self.clock = config, clock
        self.job_cursor = self.maintenance_cursor = 0
        self.insert_state("active", self._due())
        self.init_state("active")
        self.insert_output_port("job")
        self.insert_output_port("maintenance")

    def _due(self):
        jobs, maintenance = self.config["arrivals"], self.config["maintenance_times"]
        return min(jobs[self.job_cursor] if self.job_cursor < len(jobs) else math.inf,
                   maintenance[self.maintenance_cursor] if self.maintenance_cursor < len(maintenance) else math.inf)

    def output(self, deliverer):
        now = self.clock()
        for index in range(self.job_cursor, len(self.config["arrivals"])):
            if self.config["arrivals"][index] != now:
                break
            emit(deliverer, self, "job", index + 1)
        for time in self.config["maintenance_times"][self.maintenance_cursor:]:
            if time != now:
                break
            emit(deliverer, self, "maintenance", True)

    def int_trans(self):
        now = self.clock()
        while self.job_cursor < len(self.config["arrivals"]) and self.config["arrivals"][self.job_cursor] == now:
            self.job_cursor += 1
        while (self.maintenance_cursor < len(self.config["maintenance_times"])
               and self.config["maintenance_times"][self.maintenance_cursor] == now):
            self.maintenance_cursor += 1
        self.update_state("active", self._due() - now)

    def ext_trans(self, port, message):
        raise RuntimeError("source has no input")


class Stage(BehaviorModel):
    def __init__(self, name, duration, clock, ledger):
        super().__init__(name)
        self.duration, self.clock, self.ledger = duration, clock, ledger
        self.waiting, self.pending_requests = [], []
        self.current = self.finish_at = None
        self.completed = []
        self.insert_state("active")
        self.init_state("active")
        for port in ("job", "grant"):
            self.insert_input_port(port)
        for port in ("request", "release", "completed"):
            self.insert_output_port(port)

    def _schedule(self):
        self.update_state("active", 0.0 if self.pending_requests else
                          self.finish_at - self.clock() if self.finish_at is not None else math.inf)

    def output(self, deliverer):
        for job in self.pending_requests:
            emit(deliverer, self, "request", {"stage": self.get_name(), "job": job})
        if self.finish_at == self.clock():
            emit(deliverer, self, "release", {"stage": self.get_name(), "job": self.current})
            emit(deliverer, self, "completed", self.current)

    def int_trans(self):
        self.pending_requests.clear()
        if self.finish_at == self.clock():
            self.completed.append({"job": self.current, "time": float(self.clock())})
            self.current = self.finish_at = None
        self._schedule()

    def ext_trans(self, port, message):
        for value in message.retrieve():
            if port == "job":
                if value in self.waiting or value == self.current:
                    raise RuntimeError("duplicate stage job")
                self.waiting.append(value)
                self.pending_requests.append(value)
            elif port == "grant":
                if value["stage"] != self.get_name():
                    continue
                if self.current is not None or value["job"] not in self.waiting:
                    raise RuntimeError("invalid/double grant")
                self.waiting.remove(value["job"])
                self.current = value["job"]
                self.finish_at = self.clock() + self.duration
            else:
                raise RuntimeError("unknown stage input")
        self._schedule()


class ToolArbiter(BehaviorModel):
    def __init__(self, config, clock, ledger, rng_state):
        super().__init__("tool-arbiter")
        self.config, self.clock, self.ledger = config, clock, ledger
        # No constructor seed/draw on restore. Initial state comes from the
        # fresh-only construction factory, current state from model payload.
        self.rng = Random.__new__(Random)
        self.rng.setstate(tuple_tree(rng_state))
        self.rng_draws = 0
        self.requests = []
        self.maintenance_pending = 0
        self.insert_state("active")
        self.init_state("active")
        for port in ("request", "release", "maintenance", "action"):
            self.insert_input_port(port)
        self.insert_output_port("grant")

    def _choice(self):
        if self.maintenance_pending:
            return "maintenance"
        return min(self.requests, key=lambda x: (x["job"], x["stage"])) if self.requests else None

    def _schedule(self):
        if self.ledger.repair_until is not None:
            delay = self.ledger.repair_until - self.clock()
        elif self.ledger.owner is None and (self.requests or self.maintenance_pending):
            delay = 0.0
        else:
            delay = math.inf
        self.update_state("active", delay)

    def output(self, deliverer):
        if self.ledger.repair_until is None:
            choice = self._choice()
            if isinstance(choice, dict):
                emit(deliverer, self, "grant", dict(choice))

    def int_trans(self):
        now = float(self.clock())
        if self.ledger.repair_until is not None:
            if self.ledger.repair_until != now:
                raise RuntimeError("early repair completion")
            self.ledger.trace.append(["repair-end", now, None])
            self.ledger.repair_until = None
        else:
            choice = self._choice()
            if choice == "maintenance":
                self.maintenance_pending -= 1
                noise = self.rng.random()
                self.rng_draws += 1
                duration = self.config["repair_time"] + self.config["repair_jitter"] * noise
                self.ledger.repair_until = now + duration
                self.ledger.trace.append(["repair-start", now, duration])
            elif isinstance(choice, dict):
                self.requests.remove(choice)
                self.ledger.owner = dict(choice)
                self.ledger.trace.append(["grant", now, dict(choice)])
        self._schedule()

    def ext_trans(self, port, message):
        for value in message.retrieve():
            if port == "request":
                if value in self.requests or value == self.ledger.owner:
                    raise RuntimeError("duplicate tool request")
                self.requests.append(dict(value))
                self.requests.sort(key=lambda row: (row["job"], row["stage"]))
            elif port == "release":
                if value != self.ledger.owner:
                    raise RuntimeError("release without ownership")
                self.ledger.trace.append(["release", float(self.clock()), dict(value)])
                self.ledger.owner = None
            elif port == "maintenance":
                self.maintenance_pending += int(bool(value))
            elif port == "action":
                value = closed(value, {"maintenance"}, "manufacturing action")
                if type(value["maintenance"]) is not bool:
                    raise ValueError("maintenance action must be bool")
                self.maintenance_pending += int(value["maintenance"])
            else:
                raise RuntimeError("unknown arbiter input")
        self._schedule()


class ProductSink(BehaviorModel):
    def __init__(self, clock):
        super().__init__("product-sink")
        self.clock = clock
        self.completions, self.pending = [], []
        self.insert_state("active")
        self.init_state("active")
        self.insert_input_port("job")
        self.insert_output_port("completion")

    def ext_trans(self, port, message):
        if port != "job":
            raise RuntimeError("unknown sink input")
        for job in message.retrieve():
            row = {"job": job, "time": float(self.clock())}
            self.completions.append(row)
            self.pending.append(row)
        self.update_state("active", 0.0)

    def output(self, deliverer):
        for row in self.pending:
            emit(deliverer, self, "completion", dict(row))

    def int_trans(self):
        self.pending.clear()
        self.update_state("active", math.inf)


class ManufacturingGraph(StructuralModel):
    def __init__(self, config, seed, initial_rng_state, clock):
        super().__init__("manufacturing-cell")
        self.config, self.seed, self.initial_rng_state = config, seed, tuple_tree(initial_rng_state)
        self.ledger = ToolLedger()
        self.source = JobSource(config, clock)
        self.stage_a = Stage("stage-a", config["stage_a_time"], clock, self.ledger)
        self.stage_b = Stage("stage-b", config["stage_b_time"], clock, self.ledger)
        self.arbiter = ToolArbiter(config, clock, self.ledger, initial_rng_state)
        self.sink = ProductSink(clock)
        self.insert_input_port("action")
        self.insert_output_port("completion")
        for model in self.leaves():
            self.register_entity(model)
        routes = [(self, "action", self.arbiter, "action"),
                  (self.source, "job", self.stage_a, "job"),
                  (self.source, "maintenance", self.arbiter, "maintenance"),
                  (self.stage_a, "completed", self.stage_b, "job"),
                  (self.stage_b, "completed", self.sink, "job"),
                  (self.sink, "completion", self, "completion")]
        for stage in (self.stage_a, self.stage_b):
            routes.extend([(stage, "request", self.arbiter, "request"),
                           (stage, "release", self.arbiter, "release"),
                           (self.arbiter, "grant", stage, "grant")])
        for route in routes:
            self.coupling_relation(*route)

    def leaves(self):
        return (self.source, self.stage_a, self.stage_b, self.arbiter, self.sink)

    def observe(self, now):
        now = float(now)
        released = self.source.job_cursor
        completions = [dict(x) for x in self.sink.completions]
        inventory = [list(stage.waiting) + ([stage.current] if stage.current is not None else [])
                     for stage in (self.stage_a, self.stage_b)]
        all_jobs = inventory[0] + inventory[1] + [x["job"] for x in completions]
        if sorted(all_jobs) != list(range(1, released + 1)):
            raise RuntimeError("manufacturing job conservation failed")
        active = [{"stage": stage.get_name(), "job": stage.current}
                  for stage in (self.stage_a, self.stage_b) if stage.current is not None]
        if active != ([] if self.ledger.owner is None else [self.ledger.owner]):
            raise RuntimeError("tool ownership / stage activity differ")
        if self.ledger.repair_until is not None and active:
            raise RuntimeError("nonpreemptive maintenance overlapped production")
        wip = sum(now - t for t in self.config["arrivals"][:released]) - sum(now - x["time"] for x in completions)
        row = {"schema_version": OBS_SCHEMA, "logical_time": now, "released": released,
               "completed": len(completions), "wip_integral": wip,
               "cumulative_cost": wip * self.config["wip_weight"],
               "owner": None if self.ledger.owner is None else dict(self.ledger.owner),
               "repair_remaining": 0.0 if self.ledger.repair_until is None else self.ledger.repair_until - now,
               "maintenance_pending": self.arbiter.maintenance_pending,
               "stage_a_jobs": inventory[0], "stage_b_jobs": inventory[1],
               "completion_ledger": completions, "source_exhausted": math.isinf(self.source._due()),
               "rng_draws": self.arbiter.rng_draws}
        validate_observation(row)
        return row


def register_graph(executor, graph):
    for leaf in graph.leaves():
        executor.register_entity(leaf)
    executor.insert_input_port("action")
    executor.insert_output_port("completion")
    for (source, source_port), targets in graph.port_map.items():
        for target, target_port in targets:
            executor.coupling_relation(None if source is graph else source, source_port,
                                       None if target is graph else target, target_port)


def build_manufacturing_system(config, *, seed):
    config = validate_config(config)
    integer(seed, "seed")
    initial_rng_state = Random(seed).getstate()
    executor = SysExecutor(1.0, ex_mode=ExecutionType.HLA_TIME)
    try:
        graph = ManufacturingGraph(config, seed, initial_rng_state, executor.get_global_time)
        register_graph(executor, graph)
        return executor, graph
    except BaseException:
        executor.terminate_simulation()
        raise
