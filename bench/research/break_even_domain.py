"""Versioned inventory-risk mechanism model; identical transitions for R/N/C1.

Only the external CompanionObserver retains traces and counts. The production
model has scalar observations, a fixed-shape product table and one pending order.
The forecast law is hypothetical, not a fitted real-world demand model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from random import Random

from pyjevsim.behavior_model import BehaviorModel
from pyjevsim.definition import ExecutionType
from pyjevsim.structural_model import StructuralModel
from pyjevsim.system_executor import SysExecutor
from pyjevsim.system_message import SysMessage

SCHEMA = "inventory-risk-v1"
FORECAST_ALGORITHM = "forecast-h8-lcg64-v1"
INPUT_ALGORITHM = "inventory-risk-input-v1"
PRODUCT_COLUMNS = ("stock", "regime", "fulfilled", "lost", "received")
MASK = (1 << 64) - 1
_observer = None


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: integer >= {minimum} required")
    return value


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}: finite nonnegative number required")
    return float(value)


def closed(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{label}: closed fields differ")
    return value


def namespace_seed(label, family_seed):
    integer(family_seed, "family seed")
    return int.from_bytes(hashlib.sha256(f"{label}:{family_seed}".encode()).digest()[:8], "big")


def make_input(input_structure, input_seed):
    if input_structure not in ("smooth", "bursty"):
        raise ValueError("unknown input structure")
    rng = Random(integer(input_seed, "input seed"))
    rows = [{"id": j, "at": (2*j+1)/8, "product": j % 8,
             "quantity": rng.randrange(1, 5)} for j in range(80)]
    rows[63] = {"id": 63, "at": 15.875, "product": 0, "quantity": 100}
    rows[67] = {"id": 67, "at": 16.75, "product": 0, "quantity": 100}
    if input_structure == "bursty":
        for row in rows:
            j = row["id"]
            if j not in (63, 67):
                row["at"] = j // 4 + (j % 4 + 1) / 32
    return sorted(rows, key=lambda row: (row["at"], row["id"]))


def make_action_plan(action_seed, B):
    if B not in (4, 8, 16):
        raise ValueError("declared branch count required")
    rng = Random(integer(action_seed, "action seed"))
    pairs = [(q, 17-q) for q in range(1, 9)]
    rng.shuffle(pairs)
    pairs = [pair[::-1] if rng.getrandbits(1) else pair for pair in pairs]
    quantities = [q for pair in pairs[:B//2] for q in pair]
    rng.shuffle(quantities)
    return {"prefix": [{"product": 0, "order": 0} for _ in range(64)],
            "branches": [[{"product": 0, "order": q}] +
                         [{"product": 0, "order": 0} for _ in range(15)] for q in quantities],
            "quantities": quantities}


def configuration(K=1, S=8, input_seed=981000, input_structure="smooth"):
    return validate_config({"schema": SCHEMA, "K": K, "S": S,
        "input_seed": input_seed, "input_structure": input_structure,
        "forecast_seed": namespace_seed("be-forecast-v1", input_seed),
        "demands": make_input(input_structure, input_seed)})


def create_case(K=1, S=8, input_seed=981000, input_structure="smooth"):
    config = configuration(K, S, input_seed, input_structure)
    return {"config": config, "seed": input_seed, "config_sha256": sha(config)}


def validate_config(raw):
    cfg = copy.deepcopy(closed(raw, {"schema", "K", "S", "input_seed", "input_structure",
                                    "forecast_seed", "demands"}, "risk config"))
    if cfg["schema"] != SCHEMA or cfg["input_structure"] not in ("smooth", "bursty"):
        raise ValueError("risk config identity differs")
    integer(cfg["K"], "K", 1)
    integer(cfg["S"], "S", 8)
    integer(cfg["input_seed"], "input seed")
    if cfg["forecast_seed"] != namespace_seed("be-forecast-v1", cfg["input_seed"]):
        raise ValueError("forecast seed differs")
    if cfg["demands"] != make_input(cfg["input_structure"], cfg["input_seed"]):
        raise ValueError("materialized input differs from declared generator")
    return cfg


class CompanionObserver:
    """External passive hooks; no retained observer on a restorable model.

    Counts are increments from executed callbacks/stages, never K*H estimates.
    The context is deliberately single-worker and does not monkeypatch methods.
    """
    def __init__(self, phase=lambda: "unclassified"):
        self.phase, self.counts, self.events = phase, {}, []

    def row(self):
        phase = self.phase() if callable(self.phase) else self.phase
        return self.counts.setdefault(phase, {key: 0 for key in
            ("int_trans", "ext_trans", "output", "con_trans", "risk_calls", "scenario_stages",
             "normal_risk_calls", "validation_risk_calls", "normal_scenario_stages", "validation_scenario_stages")})

    def drain_events(self):
        result, self.events = self.events, []
        return result

    def __enter__(self):
        global _observer
        if _observer is not None:
            raise RuntimeError("nested companion observers are not supported")
        _observer = self
        return self

    def __exit__(self, *_):
        global _observer
        _observer = None


def _callback(kind):
    if _observer is not None:
        _observer.row()[kind] += 1
        # The generic native provider requires inherited confluence. Observe its
        # actual entry through its first int_trans call without overriding or
        # monkeypatching the protected base callback (and without a global profiler).
        if kind == "int_trans" and sys._getframe(2).f_code is BehaviorModel.con_trans.__code__:
            _observer.row()["con_trans"] += 1


def _event(kind, now, **values):
    if _observer is not None:
        _observer.events.append({"kind": kind, "time": float(now), **values})


def mix64(value):
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK
    return value ^ (value >> 31)


def forecast_risk(product, product_id, pending, now, demand_id, K, forecast_seed, *, origin="normal"):
    """The declared genuine eight-period scalar scenario calculation."""
    counted = _observer.row() if _observer is not None else None
    if origin not in ("normal", "validation"):
        raise ValueError("unknown risk origin")
    if counted is not None:
        counted["risk_calls"] += 1
        counted[origin + "_risk_calls"] += 1
    receipts = [0] * 8
    if pending is not None and pending["product"] == product_id:
        for h in range(8):
            if now + h*.25 < pending["due"] <= now + (h+1)*.25:
                receipts[h] = pending["order"]
    seed = mix64(mix64(forecast_seed ^ 0x9e3779b97f4a7c15) ^ demand_id)
    total = 0
    for scenario in range(K):
        x, regime, stock = mix64(seed ^ scenario), product["regime"], product["stock"]
        for receipt in receipts:
            x = (6364136223846793005*x + 1442695040888963407) & MASK
            u = x >> 32
            if (u & 3) == 0:
                regime = 1-regime
            demand = 1 + ((u >> 2) & 3) + 3*regime
            stock += receipt
            served = min(stock, demand)
            stock -= served
            total += 4*(demand-served) + stock
            if counted is not None:
                counted["scenario_stages"] += 1
                counted[origin + "_scenario_stages"] += 1
    return total / K


def initial_reward():
    return {"last_fulfilled": 0, "last_cumulative_risk": 0.0}


def validate_reward(raw):
    closed(raw, {"last_fulfilled", "last_cumulative_risk"}, "risk reward")
    integer(raw["last_fulfilled"], "reward fulfilled")
    number(raw["last_cumulative_risk"], "reward risk")


def validate_observation(raw):
    closed(raw, {"logical_time", "fulfilled", "lost", "cumulative_risk", "stock", "received"}, "risk observation")
    for key in ("fulfilled", "lost", "stock", "received"):
        integer(raw[key], key)
    number(raw["logical_time"], "clock")
    number(raw["cumulative_risk"], "risk")


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
        _callback("output")
        message = SysMessage(self.get_name(), "demand")
        message.insert(dict(self.config["demands"][self.cursor]))
        deliverer.insert_message(message)

    def int_trans(self):
        _callback("int_trans")
        self.cursor += 1
        self.update_state("active", self.due() - self.clock())

    def ext_trans(self, port, message):
        _callback("ext_trans")
        raise RuntimeError("demand source has no external input")

class RiskStock(BehaviorModel):
    DOMAIN_FIELDS = ("products", "pending", "fulfilled", "lost", "cumulative_risk", "last_risk", "action_product")

    def __init__(self, config, clock):
        super().__init__("inventory-stock")
        self.config, self.clock = config, clock
        self.products = [{"stock": 20, "regime": i % 2, "fulfilled": 0, "lost": 0, "received": 0}
                         for i in range(config["S"])]
        self.pending = None
        self.fulfilled = self.lost = self.action_product = 0
        self.cumulative_risk = self.last_risk = 0.0
        self.insert_state("active")
        self.init_state("active")
        self.insert_input_port("action")
        self.insert_input_port("demand")

    def schedule(self):
        self.update_state("active", (self.pending["due"] if self.pending else math.inf) - self.clock())

    def ext_trans(self, port, message):
        _callback("ext_trans")
        now = float(self.clock())
        for value in message.retrieve():
            if port == "action":
                validate_action(value, len(self.products))
                self.action_product = value["product"]
                if value["order"]:
                    if self.pending is not None:
                        raise ValueError("at most one pending order supported")
                    self.pending = {**value, "due": now + .5}
                _event("action", now, **value)
            elif port == "demand":
                product_id, quantity = value["product"], value["quantity"]
                product = self.products[product_id]
                filled = min(product["stock"], quantity)
                product["stock"] -= filled
                product["fulfilled"] += filled
                product["lost"] += quantity-filled
                product["regime"] = (product["regime"] + quantity % 2) % 2
                self.fulfilled += filled
                self.lost += quantity-filled
                self.last_risk = forecast_risk(product, product_id, self.pending, now, value["id"],
                                               self.config["K"], self.config["forecast_seed"])
                self.cumulative_risk += self.last_risk
                _event("demand", now, demand_id=value["id"], product=product_id, quantity=quantity,
                       fulfilled=filled, lost=quantity-filled, risk=self.last_risk)
            else:
                raise ValueError("unknown stock input")
        self.schedule()

    def output(self, deliverer):
        _callback("output")

    def int_trans(self):
        _callback("int_trans")
        now = float(self.clock())
        pending = self.pending
        if pending is None or pending["due"] != now:
            raise RuntimeError("receipt deadline differs")
        product = self.products[pending["product"]]
        product["stock"] += pending["order"]
        product["received"] += pending["order"]
        _event("replenish", now, product=pending["product"], quantity=pending["order"])
        self.pending = None
        self.schedule()

    def domain_state(self):
        return copy.deepcopy({key: getattr(self, key) for key in self.DOMAIN_FIELDS})

    def reward(self, state):
        value = (self.fulfilled-state["last_fulfilled"]) - .01*(self.cumulative_risk-state["last_cumulative_risk"])
        state.update(last_fulfilled=self.fulfilled, last_cumulative_risk=self.cumulative_risk)
        return value


def validate_action(value, S):
    closed(value, {"product", "order"}, "risk action")
    if integer(value["product"], "action product") >= S:
        raise ValueError("action product outside table")
    integer(value["order"], "order")


def scalar_observation(stock_state, now):
    product = stock_state["products"][stock_state["action_product"]]
    return {"logical_time": float(now), "fulfilled": stock_state["fulfilled"], "lost": stock_state["lost"],
            "cumulative_risk": stock_state["cumulative_risk"], "stock": product["stock"], "received": product["received"]}


class RiskGraph(StructuralModel):
    def __init__(self, config, seed, clock):
        super().__init__("inventory-risk")
        self.config, self.seed = validate_config(config), seed
        self.source, self.stock = DemandSource(self.config, clock), RiskStock(self.config, clock)
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
        stock = self.stock
        row = scalar_observation(vars(stock), now)
        validate_observation(row)
        return row

    def physical_state(self, now, reward_state=None):
        # Lossless fixed-column table avoids repeating 512 dictionary schemas at
        # every observed step. Input/config identities are recorded separately.
        stock = self.stock
        state = {key: copy.deepcopy(getattr(stock, key)) for key in RiskStock.DOMAIN_FIELDS if key != "products"}
        state["products"] = [[row[key] for key in PRODUCT_COLUMNS] for row in stock.products]
        return {"logical_time": float(now), "cursor": self.source.cursor, "seed": self.seed,
                "forecast_seed": self.config["forecast_seed"], "domain": state,
                "reward_state": copy.deepcopy(reward_state)}


def validate_domain_state(state, config, cursor, now=None):
    closed(state, set(RiskStock.DOMAIN_FIELDS), "risk state")
    products = state["products"]
    if type(products) is not list or len(products) != config["S"]:
        raise ValueError("product table size differs")
    for i, row in enumerate(products):
        closed(row, {"stock", "regime", "fulfilled", "lost", "received"}, "product")
        for key, value in row.items():
            integer(value, key)
        if row["regime"] not in (0, 1) or row["stock"] != 20+row["received"]-row["fulfilled"]:
            raise ValueError("product conservation differs")
    if not 0 <= integer(cursor, "cursor") <= len(config["demands"]):
        raise ValueError("cursor outside tape")
    if integer(state["action_product"], "action product") >= config["S"]:
        raise ValueError("action product outside table")
    for key in ("fulfilled", "lost"):
        if integer(state[key], key) != sum(row[key] for row in products):
            raise ValueError("aggregate product count differs")
    actual = [0] * config["S"]
    regimes = [i % 2 for i in range(config["S"])]
    for demand in config["demands"][:cursor]:
        actual[demand["product"]] += demand["quantity"]
        regimes[demand["product"]] = (regimes[demand["product"]]+demand["quantity"] % 2) % 2
    if any(row["fulfilled"]+row["lost"] != actual[i] or row["regime"] != regimes[i]
           for i, row in enumerate(products)):
        raise ValueError("demand conservation/regime differs")
    for key in ("cumulative_risk", "last_risk"):
        number(state[key], key)
    if state["last_risk"] > state["cumulative_risk"]:
        raise ValueError("last risk exceeds aggregate")
    pending = state["pending"]
    if pending is not None:
        closed(pending, {"product", "order", "due"}, "pending")
        validate_action({"product": pending["product"], "order": pending["order"]}, config["S"])
        integer(pending["order"], "positive pending order", 1)
        number(pending["due"], "pending due")
        if now is not None and pending["due"] <= now:
            raise ValueError("committed pending receipt is overdue")
    if now is not None and cursor != sum(row["at"] <= now for row in config["demands"]):
        raise ValueError("cursor and clock differ")


def callbacks(graph, clock, inject, reward_state):
    def apply(_engine, action):
        validate_action(action, graph.config["S"])
        inject("action", dict(action))
    return {"apply_action_fn": apply, "observe_fn": lambda _engine, _events: graph.observe(clock()),
            "reward_fn": lambda _view: graph.stock.reward(reward_state),
            "terminated_fn": lambda _view: False, "info_fn": lambda _view: {"lost": graph.stock.lost}}


def build_system(config, *, seed):
    engine = SysExecutor(1., ex_mode=ExecutionType.HLA_TIME)
    try:
        graph = RiskGraph(config, seed, engine.get_global_time)
        for leaf in graph.leaves():
            engine.register_entity(leaf)
        engine.insert_input_port("action")
        engine.coupling_relation(None, "action", graph.stock, "action")
        engine.coupling_relation(graph.source, "demand", graph.stock, "demand")
        return engine, graph
    except BaseException:
        engine.terminate_simulation()
        raise
