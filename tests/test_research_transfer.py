"""Direct tests of the new inventory transfer case, not runtime measurements."""
import tempfile
import time
import unittest
from unittest.mock import patch

from bench.research import inventory as m
from bench.research import inventory_native as native
from bench.research.transfer import (
    CallbackCounter, CommonBackend, _advance, _close, branch_actions,
    negative_controls, oracle, prefix_actions, run_cell, run_transfer)


class TransferTests(unittest.TestCase):
    def test_independent_oracle_replenishment_precedes_tied_demand(self):
        rows = oracle(m.configuration(2, 0), prefix_actions(4) + branch_actions(4))
        cut = rows[3]["observation"]
        self.assertEqual((cut["stock"], cut["fulfilled"], cut["lost"]), (5, 2, 5))
        tied = rows[5]["observation"]
        self.assertEqual((tied["stock"], tied["fulfilled"], tied["lost"]), (2, 9, 5))
        events = [event["kind"] for event in tied["events"] if event["time"] == 1.5]
        self.assertEqual(events, ["replenish", "demand"])
        self.assertEqual(tied["cumulative_penalty"], 12.5)

    def test_native_snapshot_is_actual_journal_and_restores_without_callbacks(self):
        cfg = m.configuration(2, 0)
        source = native.create_native(cfg)
        try:
            _advance(source, prefix_actions(3))
            with tempfile.TemporaryDirectory() as directory:
                receipt = native.save_native(source, directory)
                self.assertIn("inventory-stock.simx", receipt["files"])
                self.assertIn("demand-source.simx", receipt["files"])
                with CallbackCounter() as counted:
                    branch = native.load_native(directory)
                try:
                    self.assertEqual(counted.count, 0)
                    self.assertEqual(branch.engine.global_time, .75)
                    self.assertEqual(branch.graph.stock.pending, [{"due": 1., "quantity": 5}])
                    self.assertIs(branch.graph.stock.config, branch.graph.source.config)
                    self.assertEqual(_advance(branch, branch_actions(4)),
                                     oracle(cfg, prefix_actions(3) + branch_actions(4))[3:])
                finally:
                    _close(branch)
        finally:
            _close(source)

    def test_new_inventory_v1_and_v2_three_methods_match_oracle(self):
        for version in (1, 2):
            with self.subTest(version=version):
                outcome = run_cell(version, 0, 4)
                self.assertEqual(outcome["primary_executions"], 9)
                self.assertEqual(outcome["diagnostic_executions"], 2)
                self.assertTrue(outcome["source_invariance"])
                self.assertTrue(outcome["physical_intervention_effect"])
                self.assertEqual(len(outcome["rows"]), 3)

    def test_reset_cut_has_no_catchup_and_parameterized_intervention(self):
        outcome = run_cell(1, 1, 0)
        self.assertTrue(outcome["branch_isolation"])
        self.assertTrue(all(row["injected_action_time"] == 0 for row in outcome["rows"]))

    def test_maintenance_omission_and_wrong_version_are_rejected(self):
        receipts = negative_controls()
        self.assertEqual(len(receipts), 6)
        self.assertTrue(all(row["detected"] for row in receipts))

    def test_undeclared_domain_field_is_rejected(self):
        backend = CommonBackend(1, 0)
        runtime = backend.fresh(m.configuration())
        try:
            runtime._parts.graph.stock.unowned_state = 42
            self.assertEqual(backend.coordinator.inspect(runtime).status, "unsupported")
        finally:
            _close(runtime)

    def test_partial_failed_arm_counts_are_retained(self):
        progress = {}
        with patch.object(native, "load_native", side_effect=ValueError("deliberate restore failure")):
            with self.assertRaisesRegex(ValueError, "deliberate restore failure"):
                run_cell(1, 0, 3, progress=progress)
        self.assertEqual(progress["primary_attempted"], 2)
        self.assertEqual(progress["primary_succeeded"], 1)
        self.assertEqual(progress["active_primary"]["method"], "N")

    def test_deadline_preserves_full_unexecuted_denominator(self):
        report = run_transfer(deadline=time.perf_counter() - 1)
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["study_admission"])
        self.assertEqual(report["denominators"]["unexecuted_primary_executions"], 162)
        self.assertEqual(report["denominators"]["attempted_primary_executions"], 0)

    def test_fractional_penalty_uses_consistent_linear_cost(self):
        cfg = m.configuration(2, 0)
        cfg["penalty_per_unit"] = .1
        backend = CommonBackend(2, 0)
        runtime = backend.fresh(cfg)
        try:
            actions = prefix_actions(4) + branch_actions(4)
            self.assertEqual(_advance(runtime, actions), oracle(cfg, actions))
            snapshot = backend.capture(runtime, len(actions))
            branch = backend.restore(snapshot, "fractional-rate")
            try:
                self.assertEqual(branch._parts.graph.stock.cumulative_penalty,
                                 branch._parts.graph.stock.lost * .1)
            finally:
                _close(branch)
        finally:
            _close(runtime)


if __name__ == "__main__":
    unittest.main()
