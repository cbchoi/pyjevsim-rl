"""Pure cost-study design/statistics tests; no runtime, install or measurement."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.continuation_study.cases import sha
from bench.research import cost, cost_analysis as analysis


def fixture():
    manifest = cost.make_manifest()
    rows, groups = [], defaultdict(dict)
    for arm in manifest["arms"]:
        prefixes = arm["branch_count"] if arm["method"] == "R" else 1
        suffix = arm["branch_count"] * 4
        case = manifest["cases"][arm["case_id"]]
        phases = {"backend_setup": .01, "fresh": .01, "prefix": prefixes * arm["prefix_steps"] * .001,
                  "suffix": suffix * .001, "cleanup": .01, "compact_receipt": .001}
        if arm["method"] != "R":
            phases.update(capture_write=.003, restore_read=arm["branch_count"] * .002)
        work_counts = None
        if arm["purpose"] == "counting":
            work_counts = {"prefix": {"int_trans": prefixes * 100, "ext_trans": prefixes * 200,
                "output": prefixes * 100, "con_trans": prefixes * 2},
                "suffix": dict.fromkeys(analysis.KINDS, suffix)}
        row = {**arm, "status": "succeeded", "error": None, "worker_exit_code": 0,
            "cleanup_confirmed": True, "cleanup_errors": [], "seed": case["seed"],
            "config_sha256": case["config_sha256"], "prefixes_executed": prefixes,
            "suffix_steps_executed": suffix, "expected_suffix_steps": suffix,
            "step_calls_attempted": {"prefix": prefixes * arm["prefix_steps"], "suffix": suffix},
            "step_results_returned": {"prefix": prefixes * arm["prefix_steps"], "suffix": suffix},
            "branch_projection_sha256": [sha([arm["cell_id"], branch]) for branch in range(arm["branch_count"])],
            "application_wall_seconds": {"R": 40., "N": 20., "C1": 10.}[arm["method"]],
            "phase_seconds": phases, "application_cpu_seconds": 5., "process_wall_seconds": 41.,
            "cpu_scope_wall_seconds": 10., "snapshot_bytes": 0 if arm["method"] == "R" else 100,
            "work_counts": work_counts}
        rows.append(row)
        groups[arm["cell_id"]][arm["method"]] = row
    cells = []
    for identifier, group in groups.items():
        reference = next(iter(group.values()))
        cells.append({"cell_id": identifier, "purpose": reference["purpose"], "complete": True, "exact": True,
            "methods": list(cost.METHODS), "arm_ids": [row["arm_id"] for row in group.values()],
            "projection_sha256_by_method": {method: row["branch_projection_sha256"] for method, row in group.items()},
            "whole_projection_sha256_by_method": {method: sha(row["branch_projection_sha256"]) for method, row in group.items()}})
    return manifest, rows, cells


class CostDesignTests(unittest.TestCase):
    def test_manifest_unique_denominators_and_six_balanced_orders(self):
        manifest = cost.make_manifest()
        self.assertEqual(len(manifest["arms"]), 330)
        self.assertEqual(len({arm["arm_id"] for arm in manifest["arms"]}), 330)
        self.assertEqual(len({arm["cell_id"] for arm in manifest["arms"]}), 110)
        self.assertEqual(Counter(arm["purpose"] for arm in manifest["arms"]), {"timing": 324, "counting": 6})
        orders, positions = defaultdict(set), defaultdict(Counter)
        for arm in manifest["arms"]:
            if arm["purpose"] == "timing":
                key = (arm["model"], arm["prefix_steps"], arm["branch_count"])
                orders[key].add(tuple(arm["method_order"]))
                positions[key + (arm["method"],)][arm["method_position"]] += 1
        self.assertTrue(all(len(value) == 6 for value in orders.values()))
        self.assertTrue(all(value == {0: 2, 1: 2, 2: 2} for value in positions.values()))
        self.assertEqual(sha({key: value for key, value in manifest.items() if key != "manifest_sha256"}), manifest["manifest_sha256"])

    def test_sustained_inputs_exist_after_largest_cut_and_endpoint(self):
        manifest = cost.make_manifest()
        for case in manifest["cases"].values():
            values = ([event["time"] for event in case["config"]["arrival_spec"]["events"]]
                      if case["model"] == "Q" else case["config"]["arrivals"])
            self.assertTrue(any(64 < value <= 65 for value in values))
            self.assertGreater(max(values), manifest["largest_endpoint_time"])
            self.assertEqual(case["config_sha256"], sha(case["config"]))

    def test_timing_and_counting_are_separate_ids_same_declared_family_zero(self):
        manifest = cost.make_manifest()
        counters = manifest["arms"][-6:]
        self.assertTrue(all(arm["purpose"] == "counting" and arm["family"] == 0 for arm in counters))
        self.assertTrue(all((arm["prefix_steps"], arm["branch_count"]) == (256, 16) for arm in counters))
        self.assertEqual({arm["model"] for arm in counters}, {"Q", "M"})

    def test_deterministic_manifest_and_explicit_new_seed_offset(self):
        first, second = cost.make_manifest(), cost.make_manifest({"seed_offset": 200})
        self.assertEqual(first, cost.make_manifest())
        self.assertEqual(first["cases"]["Q-f00"]["seed"], 971000)
        self.assertEqual(first["cases"]["M-f05"]["seed"], 972005)
        self.assertEqual(second["cases"]["Q-f00"]["seed"], 971200)
        self.assertEqual(second["planning_seed"], 973451)

    def test_undeclared_design_parameters_rejected(self):
        for config in ({"prefix_steps": 1024}, {"seed_offset": True}, {"seed_offset": 1_000_000_001},
                       {"budget_seconds": float("nan")}, {"max_bytes": 100}, {"condition": "clean-certified"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                cost.make_manifest(config)

    def test_action_plan_all_cuts_and_independent_actions(self):
        for length in cost.PREFIXES:
            prefix, suffix = cost.action_plan("M", length, 1)
            self.assertEqual(len(prefix), length)
            self.assertEqual(suffix, [{"maintenance": True}, {"maintenance": False}, {"maintenance": False}, {"maintenance": False}])
            prefix[0]["maintenance"] = True
            self.assertFalse(prefix[1]["maintenance"])
            self.assertFalse(cost.action_plan("M", length, 0)[1][0]["maintenance"])
        with self.assertRaises(ValueError):
            cost.action_plan("Q", 1024, 0)

    def test_kernel_hook_restores_on_success_and_maps_c1(self):
        previous = object()
        spec = next(arm for arm in cost.make_manifest()["arms"] if arm["method"] == "C1")
        kernel = SimpleNamespace(action_plan=previous)
        def run(internal, *args, **kwargs):
            self.assertIs(kernel.action_plan, cost.action_plan)
            self.assertEqual(internal["method"], "C")
            self.assertEqual(len(kernel.action_plan("Q", 256, 0)[1]), 4)
            return "receipt", "projection"
        kernel.execute_arm = run
        self.assertEqual(cost.execute_arm(kernel, spec, {}, "unused", None, {}), ("receipt", "projection"))
        self.assertIs(kernel.action_plan, previous)

    def test_kernel_hook_restores_on_failure(self):
        previous = object()
        kernel = SimpleNamespace(action_plan=previous)
        def run(*args, **kwargs):
            raise RuntimeError("controlled failure")
        kernel.execute_arm = run
        with self.assertRaisesRegex(RuntimeError, "controlled failure"):
            cost.execute_arm(kernel, cost.make_manifest()["arms"][0], {}, "unused", None, {})
        self.assertIs(kernel.action_plan, previous)


class CostAnalysisTests(unittest.TestCase):
    def test_known_ratios_denominators_and_complete_family_resampling(self):
        manifest, rows, cells = fixture()
        result = analysis.analyze_records(manifest, rows, cells, bootstrap_replicates=30)
        self.assertTrue(result["study_admission"])
        self.assertEqual(result["denominators"]["timing"]["succeeded"], 324)
        self.assertEqual(result["exact_cells"]["counting"]["exact"], 2)
        for record in result["models"].values():
            self.assertEqual(record["complete_nine_condition_families"], list(range(6)))
            for key, estimate in record["estimates"].items():
                expected = .25 if key.endswith(":C1/R") else .5
                self.assertAlmostEqual(estimate["geometric_wall_ratio"], expected)
                self.assertEqual(estimate["paired_n"], 6)
                self.assertTrue(all(abs(value - expected) < 1e-12 for value in estimate["ci95"]))
                self.assertEqual(estimate["valid_replicates"], 30)

    def test_counting_times_do_not_enter_timing_estimates(self):
        manifest, rows, cells = fixture()
        for row in rows:
            if row["purpose"] == "counting":
                row["application_wall_seconds"] *= 1_000_000
        result = analysis.analyze_records(manifest, rows, cells, bootstrap_replicates=5)
        self.assertAlmostEqual(result["models"]["Q"]["estimates"]["L256-B16:C1/N"]["geometric_wall_ratio"], .5)
        self.assertEqual(result["counting"][0]["prefix_invocations"], {"R": 16, "N": 1, "C1": 1})
        self.assertEqual(result["counting"][0]["prefix_decision_steps"], {"R": 4096, "N": 256, "C1": 256})

    def test_unexecuted_counting_does_not_promote_complete_timing_to_study_admission(self):
        manifest, rows, cells = fixture()
        rows = [row for row in rows if row["purpose"] == "timing"]
        cells = [cell for cell in cells if cell["purpose"] == "timing"]
        result = analysis.analyze_records(manifest, rows, cells, bootstrap_replicates=5)
        self.assertFalse(result["study_admission"])
        self.assertEqual(result["denominators"]["counting"]["unexecuted"], 6)
        self.assertEqual(result["counting"], [])

    def test_partial_empty_data_retains_full_denominators(self):
        result = analysis.analyze_records(cost.make_manifest(), [], [], bootstrap_replicates=2)
        self.assertFalse(result["study_admission"])
        self.assertEqual(result["denominators"]["timing"]["unexecuted"], 324)
        self.assertEqual(len(result["unexecuted_arm_ids"]), 330)
        self.assertIsNone(result["models"]["Q"]["estimates"]["L64-B4:C1/N"]["ci95"])
        self.assertEqual(result["models"]["Q"]["observed_boundaries"]["C1/N"]["interpretation"], "partial_grid_no_boundary_conclusion")

    def test_no_winning_cells_is_reported_without_global_claim(self):
        manifest, rows, cells = fixture()
        for row in rows:
            if row["method"] == "C1":
                row["application_wall_seconds"] = 80.
        result = analysis.analyze_records(manifest, rows, cells, bootstrap_replicates=5)
        boundary = result["models"]["M"]["observed_boundaries"]["C1/N"]
        self.assertEqual(boundary["point_estimate_below_one_cells"], [])
        self.assertEqual(boundary["interpretation"], "no_observed_point_advantage_in_sampled_grid")
        self.assertTrue(boundary["not_simultaneous_confidence_or_confirmatory"])

    def test_duplicate_or_mismatched_identity_rejected(self):
        manifest, rows, cells = fixture()
        with self.assertRaises(ValueError):
            analysis.analyze_records(manifest, rows + [rows[0]], cells, bootstrap_replicates=1)
        rows[0]["seed"] += 1
        with self.assertRaises(ValueError):
            analysis.analyze_records(manifest, rows, cells, bootstrap_replicates=1)

    def test_unobserved_counter_or_shortened_success_rejected(self):
        for mutate in (lambda rows: rows[-1].update(work_counts=None),
                       lambda rows: rows[0].update(suffix_steps_executed=1),
                       lambda rows: rows[0]["step_results_returned"].update(prefix=True)):
            manifest, rows, cells = fixture()
            mutate(rows)
            with self.assertRaises(ValueError):
                analysis.analyze_records(manifest, rows, cells, bootstrap_replicates=1)

    def test_exact_flag_cannot_hide_digest_difference(self):
        manifest, rows, cells = fixture()
        rows[0]["branch_projection_sha256"][0] = "f" * 64
        with self.assertRaises(ValueError):
            analysis.analyze_records(manifest, rows, cells, bootstrap_replicates=1)

    def test_workload_change_rejected_even_with_recomputed_manifest_hash(self):
        manifest = cost.make_manifest()
        manifest["cases"]["Q-f00"]["config"]["arrival_spec"]["events"] = []
        manifest["cases"]["Q-f00"]["config_sha256"] = sha(manifest["cases"]["Q-f00"]["config"])
        manifest["manifest_sha256"] = sha({key: value for key, value in manifest.items() if key != "manifest_sha256"})
        with self.assertRaises(ValueError):
            analysis.analyze_records(manifest, [], [], bootstrap_replicates=1)

    def test_terminal_fallback_merge_and_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"arm_id": "one", "status": "failed"}
            (root / "arms.jsonl").write_text(json.dumps(row) + "\n{broken", encoding="utf-8")
            self.assertEqual(analysis._records(root, "arms.jsonl", [row], allow_incomplete_tail=True), [row])
            with self.assertRaises(ValueError):
                analysis._records(root, "arms.jsonl", [dict(row, status="succeeded")], allow_incomplete_tail=True)
            with self.assertRaises(ValueError):
                analysis._records(root, "arms.jsonl", [])


if __name__ == "__main__":
    unittest.main()
