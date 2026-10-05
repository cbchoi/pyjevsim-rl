"""Direct planner checks; no model, worker, or performance cohort is executed."""
import copy
from collections import defaultdict
import unittest

from bench.research import break_even_design as design


def fake_case(**factors):
    return {"config": dict(factors), "seed": factors["input_seed"]}


def fake_actions(seed, B):
    # This is an IDD fixture, not evidence for physical intervention semantics.
    return {"prefix": [{"product": 0, "order": 0}] * 64,
            "branches": [[{"product": 0, "order": q}] + [{"product": 0, "order": 0}] * 15
                         for q in range(1, B+1)], "quantities": list(range(1, B+1)), "seed": seed}


class BreakEvenDesignTests(unittest.TestCase):
    def setUp(self):
        self.protocol = design.load_protocol()

    def plan(self, stage="calibration", selection=None):
        return design.make_plan(self.protocol, stage, selection,
                                case_factory=fake_case, action_plan_factory=fake_actions)

    @staticmethod
    def selection(N=12):
        return {"status": "succeeded", "N": N, "calibration_input_seeds": list(range(981000, 981006)),
                "coordinates": [{"K": K, "S": S, "B": B}
                                for K, S in ((4, 8), (64, 512)) for B in (4, 8, 16)]}

    def test_declared_denominators_and_no_execution_authorization(self):
        for stage, N, count, cells in (("calibration", None, 864, 144),
                                      ("validation", 12, 432, 72),
                                      ("validation", 48, 1728, 288),
                                      ("transfer", 12, 216, 36)):
            with self.subTest(stage=stage, N=N):
                plan = self.plan(stage, None if N is None else self.selection(N))
                self.assertEqual(len(plan["arms"]), count)
                self.assertEqual(len({arm["arm_id"] for arm in plan["arms"]}), count)
                self.assertEqual(len({arm["cell_id"] for arm in plan["arms"]}), cells)
                self.assertEqual(plan["planned"]["all_arms"], count)
                self.assertFalse(plan["execute_authorized"])
                self.assertTrue(plan["resource_values_are_proposals"])

    def test_six_method_permutations_and_role_balance_per_block(self):
        plan = self.plan("validation", self.selection(12))
        groups, roles = defaultdict(list), defaultdict(list)
        for arm in plan["arms"]:
            if arm["method_position"] == 0:
                key = (arm["family"] // 6, arm["K"], arm["S"], arm["B"], arm["role"])
                groups[key].append(tuple(arm["method_order"]))
                roles[key].append(arm["role_position"])
        self.assertTrue(all(len(values) == len(set(values)) == 6 for values in groups.values()))
        self.assertTrue(all(values.count(0) == values.count(1) == 3 for values in roles.values()))

    def test_all_six_arms_share_materialized_case_and_action_identity(self):
        plan = self.plan()
        grouped = defaultdict(list)
        for arm in plan["arms"]:
            grouped[arm["cell_id"]].append(arm)
        for group in grouped.values():
            self.assertEqual(len(group), 6)
            for field in ("case_id", "input_seed", "forecast_seed", "action_seed", "input_identity", "action_identity"):
                self.assertEqual(len({row[field] for row in group}), 1)
            case = plan["cases"][group[0]["case_id"]]
            self.assertEqual(case["action_identity"], design.sha(case["action_plan"]))
        self.assertEqual(design.encoded(plan), design.encoded(self.plan()))

    def test_no_seed_overlap_or_unapproved_selection_replacement(self):
        bad = self.selection()
        bad["calibration_input_seeds"] = [982000]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.plan("validation", bad)
        for selection in (None, self.selection(10), self.selection(54),
                          {**self.selection(), "status": "model_prediction_failed"}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.plan("validation", selection)
        bad = self.selection()
        bad["coordinates"][0]["K"] = 16
        with self.assertRaises(ValueError):
            self.plan("validation", bad)

    def test_protocol_factor_change_and_seed_collision_are_rejected(self):
        original = copy.deepcopy(self.protocol)
        self.protocol["factors"]["branches_B"] = [1, 4, 16]
        with self.assertRaises(ValueError):
            self.plan()
        self.protocol = original
        self.protocol["cohorts"]["transfer"]["input_seed_start"] = 981001
        with self.assertRaises(ValueError):
            self.plan()


if __name__ == "__main__":
    unittest.main()
