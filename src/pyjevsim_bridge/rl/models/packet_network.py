"""Declared static packet routing with finite FIFO links and absolute deadlines.

This deterministic transfer case is not a calibrated network benchmark. Ordering
is local to an actually received bag; no global native scheduler tie is changed.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

from pyjevsim.behavior_model import BehaviorModel
from pyjevsim.structural_model import StructuralModel
from pyjevsim.system_message import SysMessage
from pyjevsim.system_executor import SysExecutor
from pyjevsim.definition import ExecutionType

_IMPORT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CONFIG_SCHEMA = "packet-network-config-v1"
OBS_SCHEMA = "packet-network-observation-v1"
CONFIG_FIELDS = {"schema_version", "packets", "rate_a", "rate_b", "capacity_a", "capacity_b",
                 "initial_route", "delivery_value", "drop_cost"}
OBS_FIELDS = {"schema_version", "logical_time", "released", "delivered", "dropped", "route",
              "link_a", "link_b", "results", "source_exhausted"}


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
    cfg = closed(raw, CONFIG_FIELDS, "packet config")
    if cfg["schema_version"] != CONFIG_SCHEMA or cfg["initial_route"] not in ("a", "b"):
        raise ValueError("packet schema/route differs")
    for key in ("rate_a", "rate_b", "delivery_value", "drop_cost"):
        cfg[key] = number(cfg[key], key, key.startswith("rate"))
    for key in ("capacity_a", "capacity_b"):
        if integer(cfg[key], key) > 10000:
            raise ValueError("FIFO capacity is unbounded")
    if type(cfg["packets"]) not in (list, tuple) or not 1 <= len(cfg["packets"]) <= 10000:
        raise ValueError("bounded nonempty packet list required")
    packets = []
    for packet in cfg["packets"]:
        row = closed(packet, {"at", "size", "deadline"}, "packet")
        row = {key: number(row[key], key, key == "size") for key in ("at", "size", "deadline")}
        if row["deadline"] < row["at"]:
            raise ValueError("deadline precedes arrival")
        packets.append(row)
    if [row["at"] for row in packets] != sorted(row["at"] for row in packets):
        raise ValueError("packet arrivals must be nondecreasing")
    cfg["packets"] = packets
    return cfg


def hand_config():
    return {"schema_version": CONFIG_SCHEMA,
        "packets": [{"at": at, "size": size, "deadline": deadline} for at, size, deadline in
                    ((0.0, 2.0, 10.0), (0.0, 1.0, 1.0), (0.0, 1.0, 10.0),
                     (1.0, 1.0, 10.0), (3.0, 1.0, 10.0))],
        "rate_a": 1.0, "rate_b": 2.0, "capacity_a": 1, "capacity_b": 1,
        "initial_route": "a", "delivery_value": 1.0, "drop_cost": 2.0}


def validate_result(raw):
    row = closed(raw, {"packet", "link", "time", "status"}, "packet result")
    integer(row["packet"], "packet", 1)
    number(row["time"], "result time")
    if row["link"] not in ("link-a", "link-b") or row["status"] not in ("delivered", "ttl", "overflow"):
        raise ValueError("packet result link/status differs")


def validate_observation(raw):
    row = closed(raw, OBS_FIELDS, "packet observation")
    if row["schema_version"] != OBS_SCHEMA or row["route"] not in ("a", "b") or type(row["source_exhausted"]) is not bool:
        raise ValueError("packet observation schema/route differs")
    number(row["logical_time"], "logical time")
    for key in ("released", "delivered", "dropped"):
        integer(row[key], key)
    for key in ("link_a", "link_b"):
        link = closed(row[key], {"active", "remaining", "waiting"}, "link view")
        if link["active"] is not None:
            integer(link["active"], "active packet", 1)
        number(link["remaining"], "remaining")
        if type(link["waiting"]) is not list:
            raise ValueError("FIFO view must be list")
        for packet in link["waiting"]:
            integer(packet, "queued packet", 1)
    if type(row["results"]) is not list:
        raise ValueError("results must be list")
    for result in row["results"]:
        validate_result(result)


def validate_reward_state(raw):
    row = closed(raw, {"last_delivered", "last_dropped"}, "packet reward state")
    integer(row["last_delivered"], "last_delivered")
    integer(row["last_dropped"], "last_dropped")


def emit(deliverer, model, port, value):
    message = SysMessage(model.get_name(), port)
    message.insert(value)
    deliverer.insert_message(message)


class PacketSource(BehaviorModel):
    def __init__(self, config, clock):
        super().__init__("packet-source")
        self.config, self.clock, self.cursor = config, clock, 0
        self.insert_state("active", self._due())
        self.init_state("active")
        self.insert_output_port("packet")

    def _due(self):
        return self.config["packets"][self.cursor]["at"] if self.cursor < len(self.config["packets"]) else math.inf

    def output(self, deliverer):
        for index in range(self.cursor, len(self.config["packets"])):
            if self.config["packets"][index]["at"] != self.clock():
                break
            emit(deliverer, self, "packet", index + 1)

    def int_trans(self):
        while self.cursor < len(self.config["packets"]) and self._due() == self.clock():
            self.cursor += 1
        self.update_state("active", self._due() - self.clock())

    def ext_trans(self, port, message):
        raise RuntimeError("source has no input")


class PacketRouter(BehaviorModel):
    def __init__(self, config, clock):
        super().__init__("packet-router")
        self.config, self.clock, self.route = config, clock, config["initial_route"]
        self.pending, self.routed = [], []
        self.insert_state("active")
        self.init_state("active")
        self.insert_input_port("packet")
        self.insert_input_port("action")
        self.insert_output_port("a")
        self.insert_output_port("b")

    def ext_trans(self, port, message):
        for value in message.retrieve():
            if port == "packet":
                self.pending.append(integer(value, "packet", 1))
            elif port == "action":
                action = closed(value, {"route"}, "routing action")
                if action["route"] not in ("a", "b"):
                    raise ValueError("route must be a or b")
                self.route = action["route"]
            else:
                raise RuntimeError("unknown router input")
        self.pending.sort()
        self.update_state("active", 0.0 if self.pending else math.inf)

    def output(self, deliverer):
        for packet in self.pending:
            emit(deliverer, self, self.route, packet)

    def int_trans(self):
        self.routed.extend({"packet": packet, "route": self.route, "time": float(self.clock())}
                           for packet in self.pending)
        self.pending.clear()
        self.update_state("active", math.inf)


class PacketLink(BehaviorModel):
    def __init__(self, name, config, clock):
        super().__init__(name)
        self.config, self.clock = config, clock
        self.active, self.waiting, self.inbox = None, [], []
        self.insert_state("active")
        self.init_state("active")
        self.insert_input_port("packet")
        self.insert_output_port("result")

    def _plan(self):
        """Pure due-step calculation, used identically by output and int_trans."""
        now, side = float(self.clock()), self.get_name()[-1]
        active = None if self.active is None else dict(self.active)
        waiting, results = list(self.waiting), []

        def result(packet, status):
            results.append({"packet": packet, "link": self.get_name(), "time": now, "status": status})

        def start(packet):
            spec = self.config["packets"][packet - 1]
            if spec["deadline"] <= now:
                result(packet, "ttl")
                return None
            finish = now + spec["size"] / self.config[f"rate_{side}"]
            return {"packet": packet, "start_at": now, "finish_at": finish,
                    "end_at": min(finish, spec["deadline"]),
                    "outcome": "delivered" if finish <= spec["deadline"] else "ttl"}

        if active is not None and active["end_at"] == now:
            result(active["packet"], active["outcome"])
            active = None
        while active is None and waiting:
            active = start(waiting.pop(0))
        for packet in self.inbox:
            if self.config["packets"][packet - 1]["deadline"] <= now:
                result(packet, "ttl")
            elif active is None:
                active = start(packet)
            elif len(waiting) < self.config[f"capacity_{side}"]:
                waiting.append(packet)
            else:
                result(packet, "overflow")
        return active, waiting, results

    def _schedule(self):
        self.update_state("active", 0.0 if self.inbox else
                          self.active["end_at"] - self.clock() if self.active is not None else math.inf)

    def ext_trans(self, port, message):
        if port != "packet":
            raise RuntimeError("unknown link input")
        self.inbox.extend(integer(value, "packet", 1) for value in message.retrieve())
        self.inbox.sort()
        self._schedule()

    def output(self, deliverer):
        for result in self._plan()[2]:
            emit(deliverer, self, "result", result)

    def int_trans(self):
        self.active, waiting, _results = self._plan()
        self.waiting[:] = waiting
        self.inbox.clear()
        self._schedule()


class PacketSink(BehaviorModel):
    def __init__(self, clock):
        super().__init__("packet-sink")
        self.clock = clock
        self.results, self.pending = [], []
        self.insert_state("active")
        self.init_state("active")
        self.insert_input_port("result")
        self.insert_output_port("result")

    def ext_trans(self, port, message):
        if port != "result":
            raise RuntimeError("unknown sink input")
        rows = [dict(value) for value in message.retrieve()]
        for row in rows:
            validate_result(row)
        self.pending.extend(rows)
        self.update_state("active", 0.0)

    def output(self, deliverer):
        # Native routing calls ext_trans per SysMessage. The actual dispatch
        # bag is pending across those calls, not each individual message.
        for result in sorted(self.pending, key=lambda row: row["packet"]):
            emit(deliverer, self, "result", dict(result))

    def int_trans(self):
        self.results.extend(sorted(self.pending, key=lambda row: row["packet"]))
        self.pending.clear()
        self.update_state("active", math.inf)


class PacketNetworkGraph(StructuralModel):
    def __init__(self, config, seed, clock):
        super().__init__("packet-network")
        self.config, self.seed = config, seed
        self.source, self.router = PacketSource(config, clock), PacketRouter(config, clock)
        self.link_a, self.link_b = PacketLink("link-a", config, clock), PacketLink("link-b", config, clock)
        self.sink = PacketSink(clock)
        self.insert_input_port("action")
        self.insert_output_port("result")
        for leaf in self.leaves():
            self.register_entity(leaf)
        for route in ((self, "action", self.router, "action"),
                      (self.source, "packet", self.router, "packet"),
                      (self.router, "a", self.link_a, "packet"),
                      (self.router, "b", self.link_b, "packet"),
                      (self.link_a, "result", self.sink, "result"),
                      (self.link_b, "result", self.sink, "result"),
                      (self.sink, "result", self, "result")):
            self.coupling_relation(*route)

    def leaves(self):
        return self.source, self.router, self.link_a, self.link_b, self.sink

    def observe(self, now):
        now = float(now)
        inventory = list(self.router.pending)
        views = {}
        for side, link in (("a", self.link_a), ("b", self.link_b)):
            active = None if link.active is None else link.active["packet"]
            inventory.extend(link.inbox + link.waiting + ([] if active is None else [active]))
            views[f"link_{side}"] = {"active": active,
                "remaining": 0.0 if active is None else link.active["end_at"] - now,
                "waiting": list(link.waiting)}
        results = [dict(row) for row in self.sink.results]
        inventory.extend(row["packet"] for row in results)
        if sorted(inventory) != list(range(1, self.source.cursor + 1)):
            raise RuntimeError("packet conservation failed")
        delivered = sum(row["status"] == "delivered" for row in results)
        row = {"schema_version": OBS_SCHEMA, "logical_time": now, "released": self.source.cursor,
               "delivered": delivered, "dropped": len(results) - delivered, "route": self.router.route,
               **views, "results": results, "source_exhausted": self.source.cursor == len(self.config["packets"])}
        validate_observation(row)
        return row


def register_graph(executor, graph):
    for leaf in graph.leaves():
        executor.register_entity(leaf)
    executor.insert_input_port("action")
    executor.insert_output_port("result")
    for (source, source_port), targets in graph.port_map.items():
        for target, target_port in targets:
            executor.coupling_relation(None if source is graph else source, source_port,
                                       None if target is graph else target, target_port)


def build_packet_system(config, *, seed):
    config = validate_config(config)
    integer(seed, "seed")
    executor = SysExecutor(1.0, ex_mode=ExecutionType.HLA_TIME)
    try:
        graph = PacketNetworkGraph(config, seed, executor.get_global_time)
        register_graph(executor, graph)
        return executor, graph
    except BaseException:
        executor.terminate_simulation()
        raise
