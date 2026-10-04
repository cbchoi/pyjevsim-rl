"""Direct scientific-observer and live restoration tests, not performance data."""
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor"), str(ROOT / "bench")]

from bench.research.semantics import negative_controls, physical_effect, run_case, run_study


class PhysicalObserverTests(unittest.TestCase):
    def test_identity_and_action_echo_are_not_physical_effect(self):
        observation = {"remaining_work": .5, "completed": 0, "energy_integral": .5,
                       "backlog_integral": 1.0}
        a = [{"observation": dict(observation, mode="normal"), "reward": -1,
              "branch_id": "a"}]
        b = [{"observation": dict(observation, mode="fast"), "reward": -1,
              "branch_id": "b"}]
        self.assertFalse(physical_effect(a, b, "Q")["observed"])
        b[0]["observation"]["remaining_work"] = .25
        self.assertEqual(physical_effect(a, b, "Q")["differences"], {"remaining_work": [.5, .25]})

    def test_deadline_retains_unexecuted_denominators(self):
        report = run_study(seeds=(971000,), cuts=(1,), deadline=time.perf_counter() - 1)
        self.assertEqual((report["planned"], report["attempted"], report["unexecuted"]), (2, 0, 2))
        self.assertFalse(report["admission"])


class LiveSemanticsTests(unittest.TestCase):
    def test_queue_intervention_same_time_replay_and_isolation(self):
        with tempfile.TemporaryDirectory(prefix="semantics owned scratch ") as scratch:
            row = run_case("Q", 971000, 4, suffix_steps=8, scratch_root=scratch, max_bytes=1024**2)
            self.assertEqual(list(Path(scratch).iterdir()), [])
        self.assertEqual(row["status"], "succeeded", row)
        self.assertEqual(row["completed_trajectories"], 9)
        self.assertTrue(all(row["checks"].values()))
        self.assertGreater(row["peak_transient_observed_bytes"], 0)
        self.assertLess(row["peak_transient_observed_bytes"], 1024**2)
        self.assertFalse(row["unsampled_transient_peak_known"])

    def test_manufacturing_rng_reward_alias_and_replay(self):
        row = run_case("M", 971000, 9, suffix_steps=16)
        self.assertEqual(row["status"], "succeeded", row)
        self.assertTrue(all(row["checks"].values()))

    def test_observer_detects_live_native_fault_controls(self):
        report = negative_controls()
        self.assertEqual(report["status"], "succeeded", report)
        self.assertEqual(report["detected"], 5)
        self.assertTrue(all(c["baseline_exact"] for c in report["controls"]))
        self.assertEqual(sum(c["subsequent_physical_difference"] is True for c in report["controls"]), 2)


if __name__ == "__main__":
    unittest.main()
