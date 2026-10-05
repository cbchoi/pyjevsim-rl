"""TASK218 correctness checks; their measured durations are not research data."""
import copy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.research import decision_utility as d
from bench.research.transfer import oracle


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class FakeRuntime:
    def __init__(self, config, clock, selected=()):
        self.config, self.clock, self.selected = config, clock, list(selected)
        self.closed = False

    def step(self, action):
        self.selected.append(action)
        self.clock.now += .01
        row = oracle(self.config, self.selected, d.DELTA)[-1]
        return row["observation"], row["reward"], False, False, {}

    def close(self):
        self.closed = True
        return SimpleNamespace(success=True)


class FakeNative:
    def __init__(self, clock):
        self.clock = clock

    def create_native(self, config, *args):
        return FakeRuntime(config, self.clock)


class FakeBackend:
    def __init__(self, method, family, modules):
        self.family, self.modules = family, modules
        self.saved = None

    def fresh(self, identity):
        return FakeRuntime(self.family["config"], self.modules["clock"])

    def capture(self, runtime, directory):
        self.saved = list(runtime.selected)

    def restore(self, directory, identity):
        return FakeRuntime(self.family["config"], self.modules["clock"], self.saved)


def fake_modules(clock):
    return {"clock": clock, "native": FakeNative(clock), "oracle": oracle}


class DecisionUtilityTests(unittest.TestCase):
    def test_family_namespace_and_real_prefix_conditioning(self):
        family = d.prepare_family(321, 8)
        self.assertEqual(family, d.prepare_family(321, 8))
        self.assertEqual(len(set(family["candidate_order"])), 8)
        self.assertEqual(set(family), {"family_id", "master", "seeds", "config", "candidate_order",
                                      "prefix_identity", "planning_identity", "candidate_order_identity"})
        panel = d.evaluation_panel(family, 8)
        self.assertFalse(set(panel["seeds"]) & set(family["seeds"].values()))
        for config in panel["configs"]:
            self.assertEqual(config["demands"][:2], family["config"]["demands"][:2])
            self.assertEqual(oracle(config, d.actions(0)[:4]),
                             oracle(family["config"], d.actions(0)[:4]))
        self.assertNotEqual(d.evaluation_panel(family, 8, namespace="alternate")["identity"], panel["identity"])
        with self.assertRaises(ValueError):
            d.evaluation_panel(family, 1, namespace="planning")

    def test_objective_has_interior_optimum_not_maximal_order(self):
        family = d.prepare_family(123, 33)
        losses = {}
        for q in family["candidate_order"]:
            rows = oracle(family["config"], d.actions(q))
            losses[q] = d.loss(rows[-1]["observation"], rows[3]["observation"])
        best = min(losses, key=losses.get)
        self.assertGreater(best, 0)
        self.assertLess(best, 32)
        self.assertGreater(losses[32], losses[best])

    def test_fixed_candidates_exact_and_operation_counts(self):
        family = d.prepare_family(124, 4)
        selections, traces = [], []
        for method in d.METHODS:
            clock = Clock()
            receipt, rows = d.execute_decision(method, family, modules=fake_modules(clock),
                clock=clock, backend_factory=FakeBackend)
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["operations"]["prefix"], 4 if method == "R" else 1)
            self.assertEqual(receipt["operations"]["capture"], 0 if method == "R" else 1)
            self.assertEqual(receipt["operations"]["restore"], 0 if method == "R" else 4)
            self.assertEqual(receipt["completed_before_deadline"], 4)
            selections.append(receipt["selected_quantity"])
            traces.append(rows)
        self.assertEqual(len(set(selections)), 1)
        self.assertTrue(all(rows == traces[0] for rows in traces))

    def test_deadline_counts_setup_late_candidates_and_no_decision(self):
        family = d.prepare_family(125, 4)
        counts = {}
        for method in ("R", "N"):
            clock = Clock()
            receipt, _ = d.execute_decision(method, family, mode="fixed-budget", budget_seconds=.205,
                modules=fake_modules(clock), clock=clock, backend_factory=FakeBackend)
            counts[method] = receipt["completed_before_deadline"]
            self.assertEqual(receipt["candidates"][-1]["status"], "late")
            eligible = [row["quantity"] for row in receipt["candidates"] if row["status"] == "eligible"]
            self.assertIn(receipt["selected_quantity"], eligible)
            self.assertGreater(receipt["overrun_seconds"], 0)
        self.assertEqual(counts, {"R": 1, "N": 2})
        clock = Clock()
        receipt, _ = d.execute_decision("N", family, mode="fixed-budget", budget_seconds=.02,
            modules=fake_modules(clock), clock=clock, backend_factory=FakeBackend)
        self.assertTrue(receipt["no_decision"])
        self.assertEqual(receipt["candidates"], [])

    def test_campaign_interruption_closes_and_preserves_partial(self):
        family = d.prepare_family(125, 4)
        clock = Clock()
        def check():
            if clock() >= .055:
                raise TimeoutError("test campaign bound")
        receipt, _ = d.execute_decision("R", family, modules=fake_modules(clock),
            clock=clock, backend_factory=FakeBackend, check=check)
        self.assertEqual(receipt["status"], "stopped")
        self.assertTrue(receipt["cleanup_confirmed"])
        self.assertEqual(receipt["operations"]["close"], 1)
        self.assertEqual(receipt["candidates"][0]["status"], "failed")

    def test_independent_panel_evaluation_and_prefix_rejection(self):
        family = d.prepare_family(126, 4)
        clock = Clock()
        modules = fake_modules(clock)
        panel = d.evaluation_panel(family, 3)
        result = d.evaluate_selections(family, [0, 32, None], panel, modules=modules, clock=clock)
        self.assertTrue(result["native_oracle_exact"])
        self.assertEqual(result["native_trajectories_checked"], 6)
        self.assertTrue(all(row["regret"] >= 0 for row in result["selections"].values()))
        altered = copy.deepcopy(panel)
        altered["configs"][0]["demands"][0]["quantity"] += 1
        with self.assertRaisesRegex(ValueError, "prefix"):
            d.evaluate_selections(family, [0], altered, modules=modules, clock=clock)

    def test_selection_has_no_dependency_on_holdout_namespace(self):
        family = d.prepare_family(127, 4)
        original = copy.deepcopy(family)
        for name in ("one", "two"):
            d.evaluation_panel(family, 3, namespace=name)
        self.assertEqual(family, original)
        self.assertFalse(any("heldout" in key or "evaluation" in key for key in family))

    def test_report_all_denominators_and_no_retry(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            report = d.run_decision_study(directory, families=2, candidates=4, evaluation_rollouts=2,
                decision_budget_seconds=.205, max_seconds=100, modules=fake_modules(clock),
                clock=clock, backend_factory=FakeBackend)
            self.assertEqual(report["status"], "completed", report.get("error"))
            self.assertEqual(report["denominators"]["completed_decisions"], 16)
            self.assertEqual(report["denominators"]["completed_families"], 2)
            self.assertNotEqual(report["families"][0]["method_order"], report["families"][1]["method_order"])
            self.assertTrue(all(row["fixed_candidate_exact"] for row in report["families"]))
        clock = Clock()
        report = d.run_decision_study(families=2, candidates=4, evaluation_rollouts=2,
            max_seconds=.075, modules=fake_modules(clock), clock=clock, backend_factory=FakeBackend)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["denominators"]["attempted_decisions"], 1)
        self.assertEqual(report["denominators"]["unexecuted_decisions"], 15)
        self.assertFalse(report["study_admission"])
        clock = Clock()
        original_dumps = d.json.dumps
        def slow_summary(value, *args, **kwargs):
            if isinstance(value, dict) and value.get("schema") == d.SCHEMA and "paired_summary" in value:
                clock.now += 100.
            return original_dumps(value, *args, **kwargs)
        with patch.object(d.json, "dumps", side_effect=slow_summary):
            report = d.run_decision_study(families=1, candidates=2, evaluation_rollouts=1,
                methods=("R",), max_seconds=10, modules=fake_modules(clock),
                clock=clock, backend_factory=FakeBackend)
        self.assertEqual(report["status"], "stopped")
        self.assertIn("final reporting", report["error"])
        self.assertFalse(report["study_admission"])

    def test_cleanup_failure_is_not_success(self):
        class BadRuntime(FakeRuntime):
            def close(self):
                return SimpleNamespace(success=False)
        class BadBackend(FakeBackend):
            def fresh(self, identity):
                return BadRuntime(self.family["config"], self.modules["clock"])
        clock = Clock()
        receipt, _ = d.execute_decision("R", d.prepare_family(123, 2), modules=fake_modules(clock),
            clock=clock, backend_factory=BadBackend)
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["cleanup_confirmed"])
        backend = d.InventoryBackend.__new__(d.InventoryBackend)
        backend.method = "C1A"
        mismatch = FakeRuntime(d.prepare_family(123, 2)["config"], clock)
        mismatch.execution_profile = "strict-v1"
        with self.assertRaisesRegex(RuntimeError, "execution profile"):
            backend._qualified(mismatch)
        self.assertTrue(mismatch.closed)

    def test_real_native_and_common_candidates_match_domain_oracle(self):
        # Small direct correctness test, not a timing cohort or benchmark finding.
        family = d.prepare_family(984100, 2)
        modules = d.runtime_dependencies()
        expected = {q: oracle(family["config"], d.actions(q))[4:] for q in family["candidate_order"]}
        with tempfile.TemporaryDirectory() as directory:
            for method in d.METHODS:
                receipt, actual = d.execute_decision(method, family, modules=modules, scratch_root=directory)
                self.assertEqual(receipt["status"], "completed", receipt.get("error"))
                self.assertEqual(actual, expected)
                self.assertTrue(receipt["cleanup_confirmed"])

    def test_invalid_configuration_fails_before_execution(self):
        for kwargs in ({"families": 0}, {"candidates": 1}, {"methods": ("R", "R")},
                       {"max_seconds": float("nan")}, {"decision_budget_seconds": 0},
                       {"evaluation_rollouts": 0}, {"costs": {"procurement": 1.}}):
            with self.assertRaises(ValueError):
                d.run_decision_study(**kwargs)


if __name__ == "__main__":
    unittest.main()
