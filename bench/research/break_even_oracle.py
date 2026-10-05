"""Independent scalar forecast and chronological event oracle.

Does not call domain forecast/transitions, snapshot export, or continuation code.
Input config and resulting dictionaries are the only shared representation.
"""
from __future__ import annotations

import copy


def forecast(product, product_id, pending, now, demand_id, scenarios, seed):
    modulus = 2 ** 64
    def scramble(value):
        value = ((value ^ (value >> 30)) * int("bf58476d1ce4e5b9", 16)) % modulus
        value = ((value ^ (value >> 27)) * int("94d049bb133111eb", 16)) % modulus
        return value ^ (value >> 31)
    cost = 0
    for scenario in range(scenarios):
        state = scramble(seed ^ int("9e3779b97f4a7c15", 16))
        state = scramble(state ^ demand_id)
        state = scramble(state ^ scenario)
        inventory, regime = product["stock"], product["regime"]
        for period in range(8):
            state = (6364136223846793005 * state + 1442695040888963407) % modulus
            upper = state // (2 ** 32)
            if upper % 4 == 0:
                regime = 1 - regime
            quantity = 1 + ((upper // 4) % 4) + 3 * regime
            if (pending is not None and pending["product"] == product_id
                    and now + period/4 < pending["due"] <= now + (period+1)/4):
                inventory += pending["order"]
            fulfilled = quantity if inventory >= quantity else inventory
            inventory -= fulfilled
            cost += 4 * (quantity-fulfilled) + inventory
    return cost / scenarios


def oracle(config, actions, delta=.25):
    products = [{"stock": 20, "regime": i % 2, "fulfilled": 0, "lost": 0, "received": 0}
                for i in range(config["S"])]
    demands = config["demands"]
    now, cursor, pending, rows = 0., 0, None, []
    fulfilled = lost = action_product = 0
    cumulative = last_risk = previous_risk = 0.
    previous_fulfilled = 0
    for action in actions:
        events = [{"kind": "action", "time": now, **action}]
        action_product = action["product"]
        if action["order"]:
            if pending is not None:
                raise ValueError("oracle accepts at most one pending order")
            pending = {**action, "due": now + .5}
        endpoint = now + delta
        while True:
            next_demand = demands[cursor]["at"] if cursor < len(demands) else float("inf")
            next_receipt = pending["due"] if pending is not None else float("inf")
            next_time = min(next_demand, next_receipt)
            if next_time > endpoint:
                break
            if next_receipt <= next_demand:
                target = products[pending["product"]]
                target["stock"] += pending["order"]
                target["received"] += pending["order"]
                events.append({"kind": "replenish", "time": next_time,
                               "product": pending["product"], "quantity": pending["order"]})
                pending = None
                continue
            demand = demands[cursor]
            target, quantity = products[demand["product"]], demand["quantity"]
            filled = min(target["stock"], quantity)
            target["stock"] -= filled
            target["fulfilled"] += filled
            target["lost"] += quantity-filled
            target["regime"] = (target["regime"] + quantity % 2) % 2
            fulfilled += filled
            lost += quantity-filled
            last_risk = forecast(target, demand["product"], pending, next_time, demand["id"],
                                 config["K"], config["forecast_seed"])
            cumulative += last_risk
            events.append({"kind": "demand", "time": next_time, "demand_id": demand["id"],
                           "product": demand["product"], "quantity": quantity, "fulfilled": filled,
                           "lost": quantity-filled, "risk": last_risk})
            cursor += 1
        now = endpoint
        reward = fulfilled-previous_fulfilled - .01*(cumulative-previous_risk)
        previous_fulfilled, previous_risk = fulfilled, cumulative
        observation = {"logical_time": now, "fulfilled": fulfilled, "lost": lost,
                       "cumulative_risk": cumulative, "stock": products[action_product]["stock"],
                       "received": products[action_product]["received"]}
        domain = {"products": [[row[key] for key in ("stock", "regime", "fulfilled", "lost", "received")]
                               for row in products], "pending": copy.deepcopy(pending),
                  "fulfilled": fulfilled, "lost": lost, "cumulative_risk": cumulative,
                  "last_risk": last_risk, "action_product": action_product}
        physical = {"logical_time": now, "cursor": cursor, "seed": config["input_seed"],
                    "forecast_seed": config["forecast_seed"], "domain": domain,
                    "reward_state": {"last_fulfilled": fulfilled, "last_cumulative_risk": cumulative}}
        rows.append({"observation": observation, "reward": reward, "physical": physical, "events": events})
    return rows
