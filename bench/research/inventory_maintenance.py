"""Concrete V2 extension: lost-sale penalty state and incremental reward cache.

The base stock transition calls lost_sale; V2 adds a new persisted accumulator.
No scheduler/coordinator/boundary implementation is modified by this extension.
"""
from .inventory import InventoryStock, closed, integer, number


def validate_reward_v2(raw):
    row = closed(raw, {"last_fulfilled", "last_penalty"}, "penalty reward")
    integer(row["last_fulfilled"], "fulfilled baseline")
    number(row["last_penalty"], "penalty baseline")


class PenaltyStock(InventoryStock):
    DOMAIN_FIELDS = InventoryStock.DOMAIN_FIELDS + ("cumulative_penalty",)

    def __init__(self, config, clock):
        super().__init__(config, clock)
        self.cumulative_penalty = 0.

    def lost_sale(self, quantity):
        super().lost_sale(quantity)
        # Linear fixed-rate cost has one canonical floating-point evaluation;
        # repeated addition would disagree with the conservation check at .1.
        self.cumulative_penalty = self.lost * self.config["penalty_per_unit"]

    def reward(self, state):
        value = super().reward(state) - (self.cumulative_penalty - state["last_penalty"])
        state["last_penalty"] = self.cumulative_penalty
        return value
