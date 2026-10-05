"""TC06/10/12 clock, lifecycle and accounting tests with fake model backends."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.research import break_even_run as run
from bench.research.break_even_design import sha


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def tick(self, value):
        self.value += value


class Observer:
    def __init__(self, owner, phase):
        self.owner, self.phase, self.events, self.counts = owner, phase, [], {}

    def __enter__(self):
        self.owner.observer = self
        return self

    def __exit__(self, *_):
        self.owner.observer = None

    def count(self, key):
        row = self.counts.setdefault(self.phase(), {})
        row[key] = row.get(key, 0) + 1

    def drain_events(self):
        result, self.events = self.events, []
        return result


class FakeBudget:
    def __init__(self, owner):
        self.owner, self.clock, self.transient_bytes, self.checks = owner, owner.clock, 0, 0
        self.max_transient_bytes = 1024

    def check(self):
        self.checks += 1
        if self.owner.fail_budget_at == self.checks:
            raise TimeoutError("direct-test deadline")

    def sample_transient(self, directory):
        self.transient_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        self.check()

    def write(self, path, value):
        if self.owner.running:
            raise AssertionError("receipt is inside endpoint")
        self.owner.receipts.append(value)
        self.clock.tick(1000)
        if self.owner.fail_receipt:
            raise OSError("direct-test receipt failure")


class FakeRuntime:
    def __init__(self, owner, physical_id, *, restored=False):
        self.owner, self.physical_id, self.closed = owner, physical_id, False
        self.step_id = owner.spec["prefix_steps"] if restored else 0
        self.quantity = 0

    def step(self, action):
        if self.closed:
            raise AssertionError("stepping a closed runtime")
        self.owner.clock.tick(3)  # Includes the backend's required validation.
        self.owner.validation_calls += 1
        self.step_id += 1
        self.quantity = action["order"] or self.quantity
        if self.owner.fail_step and self.step_id > self.owner.spec["prefix_steps"]:
            raise RuntimeError("direct-test model failure")
        if self.owner.observer is not None:
            self.owner.observer.count("ext_trans")
            self.owner.observer.count("risk_calls")
            if action["order"]:
                self.owner.observer.events.append({"kind": "demand", "demand_id": 67,
                    "product": 0, "fulfilled": -1 if self.owner.wrong_quantity else action["order"]})
        observation = {"logical_time": self.step_id * .25, "received": self.quantity,
                       "fulfilled": self.quantity, "lost": 100-self.quantity,
                       "stock": 0, "cumulative_risk": 1.0}
        if self.owner.mutable_output:
            observation["stock"] = []
        info = {"run_id": self.owner.spec["cell_id"], "instance_id": self.physical_id,
            "episode_id": self.physical_id + ":episode-1", "step_id": self.step_id,
            "seed": self.owner.case["seed"], "logical_time": observation["logical_time"]}
        return observation, 1.0, False, False, info

    def close(self):
        if self.closed:
            raise AssertionError("close retried")
        self.owner.clock.tick(7)
        self.closed = True
        self.owner.close_calls += 1
        return SimpleNamespace(success=not self.owner.fail_close)


class Backend:
    def __init__(self, owner, spec, case, modules):
        self.owner = owner
        owner.running = True
        owner.clock.tick(1)

    def fresh(self, physical_id):
        self.owner.clock.tick(2)
        if self.owner.fail_fresh:
            raise RuntimeError("direct-test construction failure")
        runtime = FakeRuntime(self.owner, physical_id)
        self.owner.runtimes.append(runtime)
        return runtime

    def capture_to_file(self, runtime, directory):
        self.owner.clock.tick(5)
        if self.owner.capture_callback and self.owner.observer is not None:
            self.owner.observer.count("int_trans")
        (directory / "cut.dat").write_bytes(b"fixture")
        return {"bytes": 7}

    def restore_from_file(self, directory, branch_index, physical_id):
        self.owner.clock.tick(4)
        self.owner.restored_ids.append(physical_id)
        runtime = FakeRuntime(self.owner, physical_id, restored=True)
        self.owner.runtimes.append(runtime)
        return runtime

    def physical_state(self, runtime, now):
        if self.owner.spec["role"] == "timing":
            raise AssertionError("scientific projection called during timing")
        if runtime.closed:
            raise AssertionError("post-close graph access")
        self.owner.clock.tick(100)
        return {"step": runtime.step_id, "time": now, "quantity": runtime.quantity}


class BreakEvenRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="break even endpoint ")
        self.addCleanup(self.directory.cleanup)
        self.campaign = Path(self.directory.name)
        (self.campaign / "transient").mkdir()
        (self.campaign / "receipts").mkdir()
        self.clock, self.running, self.observer = Clock(), False, None
        self.fail_budget_at = self.fail_step = self.fail_close = self.fail_fresh = None
        self.fail_receipt = self.mutable_output = self.capture_callback = self.wrong_quantity = False
        self.receipts, self.runtimes, self.restored_ids = [], [], []
        self.validation_calls = self.close_calls = 0
        quantities = [1, 16, 2, 15]
        actions = {"prefix": [{"product": 0, "order": 0}] * 2,
            "branches": [[{"product": 0, "order": q}] + [{"product": 0, "order": 0}] * 2 for q in quantities],
            "quantities": quantities}
        config = {"fixture": True, "K": 1, "S": 8, "input_seed": 981000}
        self.case = {"config": config, "seed": 981000, "action_plan": actions,
                     "input_identity": "direct-test-input", "config_sha256": sha(config),
                     "action_identity": sha(actions)}
        self.spec = {"arm_id": "fake-arm", "cell_id": "fake-cell", "family_id": "fake-family", "K": 1,
            "S": 8, "B": 4, "branch_count": 4, "prefix_steps": 2, "suffix_steps": 3,
            "delta": .25, "method": "C1", "role": "timing",
            **{key: self.case[key] for key in ("config_sha256", "input_identity", "action_identity")}}
        self.budget = FakeBudget(self)

    def execute(self):
        original_cleanup, original_sha = run.remove_owned_directory, run.sha

        def cleanup(*args):
            self.assertTrue(self.running)
            self.clock.tick(11)
            original_cleanup(*args)
            self.running = False

        def digest(value):
            if self.running and self.spec["role"] == "timing":
                raise AssertionError("scientific hash inside timing endpoint")
            return original_sha(value)

        with patch.object(run, "remove_owned_directory", side_effect=cleanup), patch.object(run, "sha", side_effect=digest):
            return run.execute_arm(self.spec, self.case, self.campaign, self.budget,
                backend_factory=lambda *args: Backend(self, *args),
                observer_factory=lambda phase: Observer(self, phase), cpu_clock=self.clock)

    def test_contiguous_endpoint_contains_all_product_work_and_excludes_receipt(self):
        row, projection = self.execute()
        self.assertEqual(row["status"], "succeeded")
        self.assertEqual(row["workflow_wall_seconds"], 112)
        self.assertEqual(row["workflow_cpu_seconds"], 112)
        self.assertEqual(sum(row["phase_seconds"].values()), 112)
        self.assertEqual(row["unclassified_seconds"], 0)
        self.assertEqual(self.validation_calls, 14)
        self.assertEqual(row["operations"]["restore"], 4)
        self.assertEqual(self.close_calls, 5)
        self.assertEqual(len(self.restored_ids), 4)
        self.assertEqual(projection, [])
        self.assertIsNone(row["work_counts"])
        self.assertGreater(self.clock(), row["workflow_wall_seconds"])
        self.assertFalse((self.campaign / "transient/fake-arm").exists())

    def test_replay_fresh_and_prefix_are_per_branch(self):
        self.spec["method"] = "R"
        row, _ = self.execute()
        self.assertEqual(row["status"], "succeeded")
        self.assertEqual(row["workflow_wall_seconds"], 108)
        self.assertEqual(row["operations"]["fresh"], 4)
        self.assertEqual(row["operations"]["prefix"], 4)
        self.assertEqual(row["operations"]["restore"], 0)
        self.assertEqual(self.close_calls, 4)

    def test_companion_exact_shape_counts_and_no_performance_evidence(self):
        self.spec["role"] = "companion"
        row, projection = self.execute()
        self.assertEqual(row["status"], "succeeded")
        self.assertIsNone(row["workflow_wall_seconds"])
        self.assertIsNone(row["workflow_cpu_seconds"])
        self.assertIsNone(row["phase_seconds"])
        self.assertEqual([len(branch) for branch in projection], [3] * 4)
        self.assertEqual(row["work_counts"]["prefix"]["ext_trans"], 2)
        self.assertEqual(row["work_counts"]["suffix"]["ext_trans"], 12)
        self.assertEqual(row["work_counts"]["restore_read"]["ext_trans"], 0)
        self.assertFalse(row["full_trace_exact_claim"])

    def test_companion_detects_restore_callbacks_and_wrong_actual_intervention(self):
        self.spec["role"] = "companion"
        self.capture_callback = True
        row, _ = self.execute()
        self.assertEqual(row["status"], "failed")
        self.assertIn("callbacks", row["error"])

    def test_wrong_intervention_is_not_declared_exact(self):
        self.spec["role"], self.wrong_quantity = "companion", True
        row, _ = self.execute()
        self.assertEqual(row["status"], "failed")
        self.assertIn("actual declared quantity", row["error"])

    def test_model_failure_closes_only_returned_handles_without_retry(self):
        self.fail_step = True
        row, _ = self.execute()
        self.assertEqual(row["status"], "failed")
        self.assertEqual(len(self.restored_ids), 1)
        self.assertTrue(all(runtime.closed for runtime in self.runtimes))
        self.assertTrue(row["cleanup_confirmed"])

    def test_construction_failure_does_not_invent_cleanup_confirmation(self):
        self.fail_fresh = True
        row, _ = self.execute()
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["cleanup_confirmed"])
        self.assertEqual(self.close_calls, 0)

    def test_close_failure_is_retained_and_not_retried(self):
        self.fail_close = True
        row, _ = self.execute()
        self.assertEqual(row["status"], "failed")
        self.assertFalse(row["cleanup_confirmed"])
        self.assertEqual(self.close_calls, 1)

    def test_mutable_output_is_not_copied_into_timing_witness(self):
        self.mutable_output = True
        row, _ = self.execute()
        self.assertEqual(row["status"], "failed")
        self.assertIn("scalars", row["error"])

    def test_receipt_failure_is_outside_endpoint_but_fails_arm(self):
        self.fail_receipt = True
        row, _ = self.execute()
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error_phase"], "receipt")
        self.assertEqual(row["workflow_wall_seconds"], 112)

    def test_existing_scratch_is_not_deleted(self):
        scratch = self.campaign / "transient/fake-arm"
        scratch.mkdir()
        (scratch / "user-data").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual((scratch / "user-data").read_bytes(), b"preserve")

    def test_assigned_coordinate_cannot_lie_about_materialized_state(self):
        self.spec["S"] = 512
        with self.assertRaisesRegex(ValueError, "K/S"):
            self.execute()
        self.assertFalse(self.running)

    def test_budget_failure_preserves_failure_and_closes_returned_handles(self):
        self.fail_budget_at = 6
        row, _ = self.execute()
        self.assertEqual(row["status"], "failed")
        self.assertIn("deadline", row["error"])
        self.assertTrue(all(runtime.closed for runtime in self.runtimes))


class BreakEvenIntegrationTests(unittest.TestCase):
    def test_isolated_companion_worker_source_identity_and_transport(self):
        """One real owned child validates import/transport, not a timing cohort."""
        import sys
        from bench import runner
        from bench.research.break_even_design import encoded, load_protocol, make_plan
        from run_research import source_inventory
        root = Path(__file__).resolve().parents[1]
        plan = make_plan(load_protocol(), "calibration")
        selected = next(arm for arm in plan["arms"] if arm["family"] == 0
                        and (arm["K"], arm["S"], arm["B"], arm["role"], arm["method"])
                        == (1, 512, 16, "companion", "C1"))
        selected["source_identity"] = sha(source_inventory())
        with tempfile.TemporaryDirectory(prefix="break even isolated direct worker ") as temporary:
            campaign = Path(temporary)
            for directory in ("transient", "receipts", "requests"):
                (campaign / directory).mkdir()
            budget = runner.Budget(campaign, 120, 32 * 1024**2)
            packet = runner.invoke(root, campaign, plan, selected,
                plan["cases"][selected["case_id"]], budget, python=sys.executable)
            row, projection = packet["row"], packet["projection"]
            self.assertEqual(row["status"], "succeeded", row.get("error"))
            self.assertEqual(row["worker_exit_code"], 0)
            self.assertFalse(row["owned_process_terminated"])
            self.assertEqual(row["actual_source_identity"], selected["source_identity"])
            self.assertTrue(row["cleanup_confirmed"])
            self.assertIsNone(row["workflow_wall_seconds"])
            self.assertEqual([len(branch) for branch in projection], [16] * 16)
            self.assertEqual([sha(branch) for branch in projection], row["branch_projection_sha256"])
            self.assertEqual(sha(row["scalar_witness"]), row["scalar_witness_sha256"])
            self.assertLessEqual(len(encoded(packet)), 4 * 1024**2)
            self.assertTrue(packet["provenance"]["source_files"])
            self.assertEqual(list((campaign / "transient").iterdir()), [])
            self.assertEqual(list((campaign / "requests").iterdir()), [])

    def test_one_direct_six_arm_cell_with_large_state_and_bounded_transport(self):
        """TC02/03/06 integration only; all temporary timing numbers are discarded."""
        from bench.research.break_even_design import encoded, load_protocol, make_plan
        from bench.continuation_study.run import Budget
        modules = run.load_runtime_modules()
        plan = make_plan(load_protocol(), "calibration")
        chosen = next(arm["cell_id"] for arm in plan["arms"]
                      if arm["family"] == 0 and arm["K"] == 1 and arm["S"] == 512 and arm["B"] == 16)
        selected = [arm for arm in plan["arms"] if arm["cell_id"] == chosen]
        witnessed, projected = [], []
        with tempfile.TemporaryDirectory(prefix="break even direct semantic cell ") as temporary:
            campaign = Path(temporary)
            (campaign / "transient").mkdir()
            (campaign / "receipts").mkdir()
            budget = Budget(campaign, 120, 32 * 1024**2)
            for arm in selected:
                row, projection = run.execute_arm(arm, plan["cases"][arm["case_id"]], campaign, budget, modules)
                self.assertEqual(row["status"], "succeeded", row.get("error"))
                self.assertTrue(row["cleanup_confirmed"])
                self.assertEqual(row["prefixes_executed"], 16 if arm["method"] == "R" else 1)
                self.assertEqual(row["restores_executed"], 0 if arm["method"] == "R" else 16)
                self.assertLessEqual(len(encoded({"row": row, "projection": projection})), 4 * 1024**2)
                witnessed.append(encoded(row["scalar_witness"]))
                if arm["role"] == "companion":
                    self.assertIsNone(row["workflow_wall_seconds"])
                    self.assertEqual(row["work_counts"]["restore_read"]["int_trans"], 0)
                    projected.append(encoded(projection))
            self.assertEqual(len(set(witnessed)), 1)
            self.assertEqual(len(set(projected)), 1)
            self.assertEqual(list((campaign / "transient").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
