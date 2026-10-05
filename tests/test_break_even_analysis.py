"""Direct statistics tests using synthetic records; no simulator or cohort run."""
from __future__ import annotations

from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import time
import unittest

from bench.research import break_even_analysis as analysis


PROTOCOL = json.loads((Path(__file__).parents[1] / "docs" / "break-even-protocol.json").read_text(encoding="utf-8"))
SOURCE = "a" * 64


def fixture(stage="calibration", coordinates=None, families=6, *, tie=False):
    if coordinates is None:
        coordinates = [{"K": K, "S": S, "B": B} for K, S, B in
                       itertools.product((1, 16, 256, 4096), (8, 512), (4, 8, 16))]
    starts = {"calibration": 981000, "validation": 982000, "transfer": 983000}
    plan = {"stage": stage, "source_identity": SOURCE, "arms": [], "cases": {}}
    rows, cells = [], []
    orders = list(itertools.permutations(analysis.METHODS))
    for family in range(families):
        case_id = f"{stage}-f{family}"
        plan["cases"][case_id] = {"seed": starts[stage] + family}
        for c in coordinates:
            K, S, B = c["K"], c["S"], c["B"]
            cell_id = f"{case_id}-S{S}-K{K}-B{B}"
            witnesses = f"normal-{cell_id}"
            for role in analysis.ROLES:
                for position, method in enumerate(orders[family % 6]):
                    arm = {"arm_id": f"{cell_id}-{role}-{method}", "cell_id": cell_id,
                           "case_id": case_id, "family_id": case_id, "family": family,
                           "K": K, "S": S, "B": B, "role": role, "method": method,
                           "method_position": position, "method_order": list(orders[family % 6]),
                           "global_order": len(plan["arms"]), "source_identity": SOURCE}
                    plan["arms"].append(arm)
                    centered = family - (families - 1) / 2
                    common = .01 * centered + S * .0001
                    replay = 2 + B * (.5 + .001 * K) + common
                    native = 1 + .0002 * K + B * (.1 + .001 * K) + common
                    continuation = 3 + .001 * K + B * (.05 + .001 * K) + common + .001 * centered
                    seconds = {"R": replay, "N": native, "C1": replay if tie else continuation}[method]
                    row = {**arm, "status": "succeeded", "actual_source_identity": SOURCE,
                           "scalar_witness_sha256": witnesses,
                           "branch_projection_sha256": [f"{cell_id}-{i}" for i in range(B)] if role == "companion" else [],
                           "workflow_wall_seconds": seconds if role == "timing" else None}
                    rows.append(row)
            cells.append({"cell_id": cell_id, "complete": True, "exact": True,
                          "companion_exact": True, "normal_output_agreement": True,
                          "identity_agreement": True})
    return plan, rows, cells


class AlgebraTests(unittest.TestCase):
    def test_every_F_D_sign_case_matches_direct_strict_inequality(self):
        for F in (-5., 0., 5.):
            for D in (-2., 0., 2.):
                result = analysis.branch_break_even(F, D)
                domain = result["integer_win"]
                for B in range(1, 20):
                    predicted = domain is not None and B >= domain["minimum"] and (domain["maximum"] is None or B <= domain["maximum"])
                    self.assertEqual(predicted, F - B * D < 0, (F, D, B))

    def test_integer_ties_excluded_and_outside_support_retained(self):
        self.assertEqual(analysis.branch_break_even(8, 2)["integer_win"]["minimum"], 5)
        self.assertEqual(analysis.branch_break_even(-8, -2)["integer_win"]["maximum"], 3)
        self.assertEqual(analysis.branch_break_even(100, 1)["root_status"], "outside_supported_domain")
        self.assertEqual(analysis.branch_break_even(-1, 1)["root_status"], "nonpositive_root")

    def test_compute_root_and_zero_slope(self):
        root = analysis.compute_break_even(5, .1, .2, .2, 4)
        self.assertAlmostEqual(root["root"], 6)
        self.assertEqual(root["status"], "advantage_begins")
        self.assertEqual(analysis.compute_break_even(2, 4, 1, 2, 2)["status"], "all_tied")
        self.assertEqual(analysis.compute_break_even(1, 4, 1, 2, 2)["status"], "always_faster")


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.rows, cls.cells = fixture()
        cls.fit = analysis.fit_calibration(cls.plan, cls.rows, cls.cells, protocol=PROTOCOL, bootstrap_replicates=80)

    def test_full_counts_and_OLS_parameters(self):
        self.assertTrue(self.fit["study_admission"])
        self.assertEqual(self.fit["denominators"]["all"]["planned"], 864)
        for S in (8, 512):
            difference = self.fit["fits"][str(S)]["differences"]["C1-R"]
            self.assertAlmostEqual(difference["alpha"], 1, places=9)
            self.assertAlmostEqual(difference["beta"], .001, places=9)
            self.assertAlmostEqual(difference["gamma"], .45, places=9)
            self.assertAlmostEqual(difference["eta"], 0, places=9)
        self.assertEqual(len(self.fit["diagnostics"]["leave_one_K_level_out"]), 72)
        self.assertTrue(all(abs(row["error_seconds"]) < 1e-7 for row in self.fit["diagnostics"]["leave_one_K_level_out"]))

    def test_calibration_selects_fixed_unseen_K_and_minimum_N(self):
        selected = analysis.select_validation(self.fit, PROTOCOL)
        self.assertEqual(selected["status"], "succeeded")
        self.assertEqual(selected["N"], 12)
        self.assertEqual(len(selected["coordinates"]), 6)
        self.assertEqual({c["K"] for c in selected["coordinates"]}, {1024})
        self.assertEqual({c["B"] for c in selected["coordinates"]}, {4, 8, 16})

    def test_cap_overflow_is_not_truncated(self):
        fit = deepcopy(self.fit)
        fit["s_max"] = 1000.
        selected = analysis.select_validation(fit, PROTOCOL)
        self.assertEqual(selected["status"], "sample_size_infeasible")
        self.assertGreater(selected["N_required"], 48)
        self.assertIsNone(selected["N"])

    def test_numeric_sample_size_overflow_reports_infeasibility(self):
        fit = deepcopy(self.fit)
        fit["s_max"] = 1e308
        selected = analysis.select_validation(fit, PROTOCOL)
        self.assertEqual(selected["status"], "sample_size_infeasible")
        self.assertIsNone(selected["N"])
        json.dumps(selected, allow_nan=False)

    def test_prediction_nonpositive_fails_without_alternative_selection(self):
        fit = deepcopy(self.fit)
        fit["fits"]["8"]["absolute"]["N"]["coefficients_scaled"] = [-1, 0, 0, 0]
        selected = analysis.select_validation(fit, PROTOCOL)
        self.assertEqual(selected["status"], "model_prediction_failed")
        self.assertEqual({c["K"] for c in selected["coordinates"]}, {1024})

    def test_missing_one_arm_blocks_planning_without_imputation(self):
        fit = analysis.fit_calibration(self.plan, self.rows[:-1], self.cells, bootstrap_replicates=8)
        self.assertFalse(fit["study_admission"])
        self.assertEqual(fit["denominators"]["all"]["unexecuted"], 1)
        self.assertEqual(len(fit["fit_families"]), 5)
        self.assertEqual(analysis.select_validation(fit, PROTOCOL)["status"], "calibration_incomplete")

    def test_timeout_not_accepted_as_timing_and_no_complete_family_is_reported(self):
        rows = deepcopy(self.rows[:6])
        rows[0].update(status="timeout", workflow_wall_seconds=120.)
        fit = analysis.fit_calibration(self.plan, rows, self.cells[:1], bootstrap_replicates=8)
        self.assertFalse(fit["study_admission"])
        self.assertEqual(fit["denominators"]["all"]["failed"], 1)
        self.assertEqual(fit["fit_status"], "no_complete_calibration_families")

    def test_source_and_witness_mismatches_are_retained(self):
        rows = deepcopy(self.rows)
        rows[0]["actual_source_identity"] = "b" * 64
        rows[6]["scalar_witness_sha256"] = "different"
        fit = analysis.fit_calibration(self.plan, rows, self.cells, bootstrap_replicates=8)
        self.assertFalse(fit["study_admission"])
        self.assertEqual(len(fit["source_issues"]), 1)
        self.assertEqual(len(fit["ineligible_cells"]), 2)

    def test_repeated_input_seed_and_campaign_source_failure_block_admission(self):
        plan = deepcopy(self.plan)
        for case in plan["cases"].values():
            case["seed"] = 981000
        plan["source_check"] = {"complete": True, "consistent": False}
        fit = analysis.fit_calibration(plan, self.rows, self.cells, bootstrap_replicates=2)
        self.assertFalse(fit["study_admission"])
        self.assertTrue(fit["design_issues"])
        self.assertTrue(fit["source_issues"])

    def test_contradictory_cleanup_receipt_not_admitted(self):
        rows = deepcopy(self.rows)
        rows[0]["cleanup_confirmed"] = False
        fit = analysis.fit_calibration(self.plan, rows, self.cells, bootstrap_replicates=2)
        self.assertFalse(fit["study_admission"])
        self.assertIn("termination_or_cleanup_not_verified", fit["ineligible_cells"][rows[0]["cell_id"]])

    def test_phases_and_measured_companion_vectors_remain_separate(self):
        rows = deepcopy(self.rows)
        timing = next(row for row in rows if row["role"] == "timing")
        timing.update(phase_seconds={"prefix": .7}, workflow_cpu_seconds=.8,
                      process_wall_seconds=2., unclassified_seconds=.02, snapshot_bytes=42)
        companion = next(row for row in rows if row["role"] == "companion")
        companion["work_counts"] = {"prefix": {"risk_calls": 64, "scenario_stages": 512}}
        fit = analysis.fit_calibration(self.plan, rows, self.cells, bootstrap_replicates=2)
        record = fit["descriptive"][analysis._key(timing)][timing["method"]]
        self.assertEqual(record["phase_seconds"]["prefix"]["mean_seconds"], .7)
        self.assertEqual(record["snapshot_bytes"]["missing_n"], 5)
        counted = next(row for row in fit["companion_counts"]["records"] if row["cell_id"] == companion["cell_id"] and row["method"] == companion["method"])
        self.assertEqual(counted["work_counts"]["prefix"]["risk_calls"], 64)
        self.assertFalse(fit["companion_counts"]["callback_vector_sum_is_unique_event_count"])

    def test_duplicate_and_wrong_cohort_rejected(self):
        with self.assertRaises(ValueError):
            analysis.fit_calibration(self.plan, self.rows + self.rows[:1], self.cells, bootstrap_replicates=1)
        plan = dict(self.plan, stage="validation")
        with self.assertRaises(ValueError):
            analysis.fit_calibration(plan, self.rows, self.cells, bootstrap_replicates=1)

    def test_no_root_draws_remain_in_denominator(self):
        plan, rows, cells = fixture(tie=True)
        fit = analysis.fit_calibration(plan, rows, cells, bootstrap_replicates=15)
        record = fit["root_uncertainty"]["S8:C1-R:B_at_K1"]
        self.assertEqual(record["status_counts"], {"all_tied": 15})
        self.assertEqual(record["no_finite_root_draws"], 15)
        self.assertIsNone(record["finite_root_conditional_ci95"])

    def test_deadline_is_respected(self):
        with self.assertRaises(TimeoutError):
            analysis.fit_calibration(self.plan, self.rows, self.cells, deadline=time.perf_counter() - 1)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan, rows, cells = fixture()
        fit = analysis.fit_calibration(plan, rows, cells, bootstrap_replicates=20)
        cls.predictions = analysis.select_validation(fit, PROTOCOL)
        cls.plan, cls.rows, cls.cells = fixture("validation", cls.predictions["coordinates"], 12)

    def test_independent_validation_fixed_prediction_CI_and_secondary_ratios(self):
        saved = deepcopy(self.predictions)
        result = analysis.analyze_validation(self.predictions, self.plan, self.rows, self.cells, bootstrap_replicates=80)
        self.assertTrue(result["study_admission"])
        self.assertEqual(result["denominators"]["all"]["planned"], 432)
        self.assertTrue(result["all_primary_prediction_tolerances_met"])
        self.assertEqual(result["primary_interval_level"], 1 - .05 / 6)
        self.assertEqual(saved, self.predictions)
        for coordinate in result["coordinates"].values():
            self.assertAlmostEqual(coordinate["mean_prediction_error_seconds"], 0, places=8)
            shift = coordinate["predicted_difference_seconds"]
            self.assertEqual(coordinate["prediction_error_ci_seconds"], [value - shift for value in coordinate["difference_ci_seconds"]])
            self.assertIn("C1/N", coordinate["secondary_exploratory"])
            self.assertIsNotNone(coordinate["supplementary_two_cohort_prediction_error_ci95"])
        self.assertTrue(all(c["status"] == "crossing_bracket_demonstrated" for c in result["crossings"].values()))

    def test_transfer_separate_and_exploratory(self):
        plan, rows, cells = fixture("transfer", self.predictions["coordinates"], 6)
        result = analysis.analyze_validation(self.predictions, plan, rows, cells, cohort="transfer", bootstrap_replicates=20)
        self.assertTrue(result["study_admission"])
        self.assertEqual(result["denominators"]["all"]["planned"], 216)
        self.assertEqual(result["primary_interval_level"], .95)
        self.assertFalse(result["all_primary_prediction_tolerances_met"])
        self.assertTrue(result["transfer_is_exploratory_not_pooled"])

    def test_calibration_seed_overlap_and_changed_coordinates_rejected(self):
        plan = deepcopy(self.plan)
        next(iter(plan["cases"].values()))["seed"] = 981000
        with self.assertRaisesRegex(ValueError, "overlap"):
            analysis.analyze_validation(self.predictions, plan, self.rows, self.cells, bootstrap_replicates=1)
        predictions = deepcopy(self.predictions)
        predictions["coordinates"][0]["K"] = 4
        with self.assertRaisesRegex(ValueError, "coordinates differ"):
            analysis.analyze_validation(predictions, self.plan, self.rows, self.cells, bootstrap_replicates=1)

    def test_partial_validation_never_claims_primary_success(self):
        result = analysis.analyze_validation(self.predictions, self.plan, self.rows[:-1], self.cells, bootstrap_replicates=10)
        self.assertFalse(result["study_admission"])
        self.assertFalse(result["all_primary_prediction_tolerances_met"])
        self.assertEqual(result["denominators"]["all"]["unexecuted"], 1)

    def test_default_twenty_thousand_is_declared_without_simulation(self):
        self.assertEqual(analysis.fit_calibration.__kwdefaults__["bootstrap_replicates"], 20000)
        self.assertEqual(analysis.analyze_validation.__kwdefaults__["bootstrap_replicates"], 20000)


if __name__ == "__main__":
    unittest.main()
