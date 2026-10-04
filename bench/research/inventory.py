"""New inventory domain for a research engineering transfer case.

The demand tape is deterministic; the seed labels a configuration, not a Monte
Carlo replication. A stock actor owns inventory, lost sales and ordered future
replenishments. Native int-then-ext confluence makes replenishment available to
demand at the same time. No continuation code is used by the domain model.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

from pyjevsim.behavior_model import BehaviorModel
from pyjevsim.definition import ExecutionType
from pyjevsim.structural_model import StructuralModel
from pyjevsim.system_executor import SysExecutor
from pyjevsim.system_message import SysMessage


def closed(value, fields, label):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{label}: closed fields differ")
    return dict(value)


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: nonnegative integer required")
    return value


def number(value, label, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{label}: finite nonnegative number required")
    return float(value)


def validate_config(raw):
    version = raw.get("version")
    fields = {"version", "initial_stock", "lead_time", "demands"}
    if version == 2:
        fields.add("penalty_per_unit")
    cfg = closed(raw, fields, "inventory config")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("inventory version must be 1 or 2")
    integer(cfg["initial_stock"], "initial stock")
    cfg["lead_time"] = number(cfg["lead_time"], "lead time", True)
    if version == 2:
        cfg["penalty_per_unit"] = number(cfg["penalty_per_unit"], "penalty rate", True)
    if type(cfg["demands"]) not in (list, tuple) or not 1 <= len(cfg["demands"]) <= 10000:
        raise ValueError("finite nonempty demand tape required")
    demands = []
    for item in cfg["demands"]:
        row = closed(item, {"at", "quantity"}, "demand")
        demands.append({"at": number(row["at"], "demand time", True),
                        "quantity": integer(row["quantity"], "demand quantity", 1)})
    if any(a["at"] >= b["at"] for a, b in zip(demands, demands[1:])):
        raise ValueError("demand times must be strictly increasing")
    cfg["demands"] = demands
    return cfg


def configuration(version=1, variant=0):
    cfg = {"version": version, "initial_stock": 2 + variant, "lead_time": .5,
           "demands": [{"at": at, "quantity": quantity + variant} for at, quantity in
                       ((.25, 3), (.75, 4), (1.5, 7), (2.25, 5), (3., 8), (8., 2))]}
    if version == 2:
        cfg["penalty_per_unit"] = 2.5
    return validate_config(cfg)


def validate_observation(raw):
    fields = {"version", "logical_time", "stock", "fulfilled", "lost", "received",
              "ordered", "pending", "events", "demand_cursor"}
    if raw.get("version") == 2:
        fields.add("cumulative_penalty")
    row = closed(raw, fields, "inventory observation")
    if row["version"] not in (1, 2):
        raise ValueError("observation version differs")
    number(row["logical_time"], "clock")
    for key in ("stock", "fulfilled", "lost", "received", "ordered", "demand_cursor"):
        integer(row[key], key)
    if row["version"] == 2:
        number(row["cumulative_penalty"], "penalty")
    if type(row["pending"]) is not list or type(row["events"]) is not list:
        raise ValueError("pending orders and events must be lists")


def validate_reward_v1(raw):
    integer(closed(raw, {"last_fulfilled"}, "inventory reward")["last_fulfilled"], "reward baseline")


class DemandSource(BehaviorModel):
    def __init__(self, config, clock):
        super().__init__("demand-source")
        self.config, self.clock, self.cursor = config, clock, 0
        self.insert_state("active", self.due())
        self.init_state("active")
        self.insert_output_port("demand")

    def due(self):
        return self.config["demands"][self.cursor]["at"] if self.cursor < len(self.config["demands"]) else math.inf

    def output(self, deliverer):
        message = SysMessage(self.get_name(), "demand")
        message.insert(self.config["demands"][self.cursor]["quantity"])
        deliverer.insert_message(message)

    def int_trans(self):
        self.cursor += 1
        self.update_state("active", self.due() - self.clock())

    def ext_trans(self, port, message):
        raise RuntimeError("demand source has no external input")


class InventoryStock(BehaviorModel):
    DOMAIN_FIELDS = ("stock", "fulfilled", "lost", "received", "ordered", "pending", "events")

    def __init__(self, config, clock):
        super().__init__("inventory-stock")
        self.config, self.clock = config, clock
        self.stock, self.fulfilled, self.lost = config["initial_stock"], 0, 0
        self.received, self.ordered, self.pending, self.events = 0, 0, [], []
        self.insert_state("active")
        self.init_state("active")
        self.insert_input_port("action")
        self.insert_input_port("demand")

    def lost_sale(self, quantity):
        self.lost += quantity

    def schedule(self):
        self.update_state("active", min((row["due"] for row in self.pending), default=math.inf) - self.clock())

    def ext_trans(self, port, message):
        now = float(self.clock())
        for value in message.retrieve():
            if port == "action":
                quantity = integer(closed(value, {"order"}, "inventory action")["order"], "order")
                self.events.append({"time": now, "kind": "action", "quantity": quantity})
                if quantity:
                    self.ordered += quantity
                    self.pending.append({"due": now + self.config["lead_time"], "quantity": quantity})
            elif port == "demand":
                quantity = integer(value, "demand", 1)
                filled = min(self.stock, quantity)
                self.stock -= filled
                self.fulfilled += filled
                self.lost_sale(quantity - filled)
                self.events.append({"time": now, "kind": "demand", "quantity": quantity,
                                    "fulfilled": filled, "lost": quantity - filled})
            else:
                raise ValueError("undeclared inventory input")
        self.schedule()

    def output(self, deliverer):
        pass

    def int_trans(self):
        now = float(self.clock())
        for row in self.pending:
            if row["due"] == now:
                self.stock += row["quantity"]
                self.received += row["quantity"]
                self.events.append({"time": now, "kind": "replenish", "quantity": row["quantity"]})
        self.pending[:] = [row for row in self.pending if row["due"] != now]
        self.schedule()

    def domain_state(self):
        return copy.deepcopy({key: getattr(self, key) for key in self.DOMAIN_FIELDS})

    def reward(self, state):
        value = self.fulfilled - state["last_fulfilled"]
        state["last_fulfilled"] = self.fulfilled
        return value


class InventoryGraph(StructuralModel):
    def __init__(self, config, seed, clock, stock_type=InventoryStock):
        super().__init__("inventory")
        self.config, self.seed = validate_config(config), seed
        self.source, self.stock = DemandSource(self.config, clock), stock_type(self.config, clock)
        self.connect()

    def connect(self):
        self.insert_input_port("action")
        self.register_entity(self.source)
        self.register_entity(self.stock)
        self.coupling_relation(self, "action", self.stock, "action")
        self.coupling_relation(self.source, "demand", self.stock, "demand")

    def leaves(self):
        return self.source, self.stock

    def observe(self, now):
        row = {"version": self.config["version"], "logical_time": float(now),
               **self.stock.domain_state(), "demand_cursor": self.source.cursor}
        validate_observation(row)
        if self.stock.stock != self.config["initial_stock"] + self.stock.received - self.stock.fulfilled:
            raise RuntimeError("stock conservation differs")
        return row


def stock_type(version):
    if version == 1:
        return InventoryStock
    from .inventory_maintenance import PenaltyStock
    if version == 2:
        return PenaltyStock
    raise ValueError("unknown inventory version")


def initial_reward(version):
    return {"last_fulfilled": 0} if version == 1 else {"last_fulfilled": 0, "last_penalty": 0.}


def callbacks(graph, clock, inject, reward_state):
    def apply(_engine, action):
        quantity = integer(closed(action, {"order"}, "order action")["order"], "order")
        inject("action", {"order": quantity})
    return {"apply_action_fn": apply,
            "observe_fn": lambda _engine, _events: graph.observe(clock()),
            "reward_fn": lambda _view: graph.stock.reward(reward_state),
            "terminated_fn": lambda _view: False,
            "info_fn": lambda _view: {"lost": graph.stock.lost}}


def build_system(config, *, seed):
    engine = SysExecutor(1., ex_mode=ExecutionType.HLA_TIME)
    try:
        graph = InventoryGraph(config, seed, engine.get_global_time, stock_type(config["version"]))
        for leaf in graph.leaves():
            engine.register_entity(leaf)
        engine.insert_input_port("action")
        engine.coupling_relation(None, "action", graph.stock, "action")
        engine.coupling_relation(graph.source, "demand", graph.stock, "demand")
        return engine, graph
    except BaseException:
        engine.terminate_simulation()
        raise
