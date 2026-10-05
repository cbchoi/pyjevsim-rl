"""Direct execution-contract checks, not performance or general DEVS evidence."""
import pickle
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

from bench.research import break_even_domain as domain
from bench.research.break_even_adapter import make_bundle
from pyjevsim_bridge.rl.continuation import (
    BranchContext, CaptureRequest, ContinuationCoordinator, ContinuationError,
    ContinuationRegistry, ResetRequest)
from pyjevsim_bridge.rl.continuation import registry as registry_module
from pyjevsim_bridge.rl.continuation.contracts import canonical_bytes, decode_json, digest


POLICY = {"policy_sha256": domain.sha({"policy": "admitted-contract-test"}),
          "policy_version": 0, "feature_contract_sha256": domain.sha({"features": "risk"})}
ZERO = {"product": 0, "order": 0}


def sampling(branch="prefix"):
    return {"domain": "pyjevsim-live-branch-v1", "phase": "admitted-direct",
            "master": 981000, "segment": "prefix" if branch == "prefix" else "suffix",
            "logical_branch_id": branch, "run_id": "admitted-test", "generation": 0,
            "worker_id": "serial", "episode_id": "one", "sampling_seed": 981000}


def physical(runtime):
    parts = runtime._parts
    return parts.graph.physical_state(parts.engine.global_time, parts.boundary_state.value)


class AdmittedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.bundle = make_bundle()
        self.registry = ContinuationRegistry()
        self.registry.register(self.bundle)
        self.runtimes = []

    def tearDown(self):
        for runtime in reversed(self.runtimes):
            self.assertTrue(runtime.close().success)

    def coordinator(self, profile="admitted-runtime-v1"):
        return ContinuationCoordinator(self.registry, execution_profile=profile)

    def fresh(self, coordinator, *, max_steps=81):
        runtime = coordinator.create_fresh(ResetRequest(self.bundle.profile.profile_id,
            domain.configuration(), 981000, "admitted-instance", "admitted-test", .25,
            max_steps, POLICY, sampling()))
        self.runtimes.append(runtime)
        return runtime

    def capture(self, coordinator, runtime):
        return coordinator.capture(runtime, CaptureRequest(self.bundle.profile.profile_id,
            runtime._parts.env._step_id, "admitted-family", "admitted-prefix", POLICY))

    def context(self, branch):
        return BranchContext("admitted-family", "admitted-prefix", branch,
            POLICY, sampling(branch), "admitted-" + branch)

    def restore(self, coordinator, snapshot, branch="a"):
        runtime = coordinator.restore(snapshot, self.context(branch))
        self.runtimes.append(runtime)
        return runtime

    def test_default_strict_and_read_only_profile(self):
        coordinator = ContinuationCoordinator(self.registry)
        runtime = self.fresh(coordinator)
        self.assertEqual(coordinator.execution_profile, "strict-v1")
        self.assertEqual(runtime.execution_profile, "strict-v1")
        self.assertIsNone(runtime._admission_witness)
        for obj in (coordinator, runtime):
            with self.assertRaises(AttributeError):
                obj.execution_profile = "admitted-runtime-v1"
        with patch.object(coordinator, "_admit", wraps=coordinator._admit) as admit:
            runtime.step(ZERO)
            self.assertEqual(admit.call_count, 1)

    def test_unknown_profiles_rejected(self):
        for value in ("", "fast", None, True, 1):
            with self.subTest(value=value), self.assertRaises(ContinuationError) as error:
                self.coordinator(value)
            self.assertEqual(error.exception.code, "CC_UNSUPPORTED_PROFILE")

    def test_admitted_step_has_no_full_admission_or_source_reads(self):
        coordinator = self.coordinator()
        runtime = self.fresh(coordinator)
        self.assertEqual(runtime.execution_profile, "admitted-runtime-v1")
        with patch.object(coordinator, "_admit", side_effect=AssertionError("full admission")), \
                patch.object(registry_module, "_verify_sources", side_effect=AssertionError("source read")):
            runtime.step(ZERO)
            runtime.step(ZERO)
        self.assertEqual(runtime._parts.env._step_id, 2)

    def test_full_fresh_capture_restore_inspection_and_semantic_view_remain(self):
        coordinator = self.coordinator()
        with patch.object(coordinator, "_admit", wraps=coordinator._admit) as admit:
            runtime = self.fresh(coordinator)
            self.assertGreater(admit.call_count, 0)
        runtime.step(ZERO)
        with patch.object(coordinator, "_admit", wraps=coordinator._admit) as admit:
            snapshot = self.capture(coordinator, runtime)
            self.assertEqual(admit.call_count, 2)
        with patch.object(coordinator, "_admit", wraps=coordinator._admit) as admit:
            branch = self.restore(coordinator, snapshot)
            self.assertGreater(admit.call_count, 0)
        with patch.object(self.registry, "resolve", wraps=self.registry.resolve) as resolve:
            coordinator.inspect(branch)
            self.assertGreater(resolve.call_count, 0)
        with patch.object(coordinator, "_admit", wraps=coordinator._admit) as admit:
            coordinator.semantic_view(branch)
            self.assertEqual(admit.call_count, 1)

    def test_changed_installation_is_detected_at_full_checkpoint_not_promised_each_step(self):
        fast = self.coordinator()
        admitted = self.fresh(fast)
        strict = self.coordinator("strict-v1")
        checked = self.fresh(strict)
        failure = ContinuationError("CC_INCOMPATIBLE_IDENTITY", "simulated source change")
        with patch.object(registry_module, "_verify_sources", side_effect=failure):
            admitted.step(ZERO)
            with self.assertRaises(ContinuationError):
                fast.semantic_view(admitted)
            with self.assertRaises(ContinuationError):
                checked.step(ZERO)
        self.assertEqual(admitted._parts.env._step_id, 1)
        self.assertEqual(checked._parts.env._step_id, 0)

    def test_witness_cannot_be_swapped_between_runtimes(self):
        coordinator = self.coordinator()
        first, second = self.fresh(coordinator), self.fresh(coordinator)
        second._admission_witness = first._admission_witness
        with self.assertRaises(ContinuationError) as error:
            second.step(ZERO)
        self.assertEqual(error.exception.code, "CC_INCOMPATIBLE_IDENTITY")
        self.assertEqual(second._parts.env._step_id, 0)

    def test_foreign_registry_and_coordinator_rejected(self):
        coordinator = self.coordinator()
        runtime = self.fresh(coordinator)
        foreign = ContinuationRegistry()
        foreign.register(self.bundle)
        with self.assertRaises(ContinuationError):
            foreign._require_admitted_runtime(runtime._admission_witness,
                bundle=self.bundle, runtime=runtime)
        with self.assertRaises(ContinuationError):
            self.coordinator().semantic_view(runtime)

    def test_registry_generation_change_rejected_before_step(self):
        coordinator = self.coordinator()
        runtime = self.fresh(coordinator)
        key = self.bundle.profile.profile_id
        previous = self.registry._generations[key]
        self.registry._generations[key] += 1
        try:
            with self.assertRaises(ContinuationError):
                runtime.step(ZERO)
            self.assertEqual(runtime._parts.env._step_id, 0)
        finally:
            self.registry._generations[key] = previous

    def test_bundle_and_profile_identity_are_bound(self):
        coordinator = self.coordinator()
        runtime = self.fresh(coordinator)
        with self.assertRaises(ContinuationError):
            self.registry._require_admitted_runtime(runtime._admission_witness,
                bundle=replace(self.bundle), runtime=runtime)
        original = self.bundle.profile
        object.__setattr__(self.bundle, "profile", replace(original, version="test-replaced"))
        try:
            with self.assertRaises(ContinuationError):
                runtime.step(ZERO)
        finally:
            object.__setattr__(self.bundle, "profile", original)

    def test_witness_not_serializable(self):
        runtime = self.fresh(self.coordinator())
        with self.assertRaises(TypeError):
            pickle.dumps(runtime._admission_witness)

    def test_serialized_cross_thread_step_remains_supported(self):
        runtime = self.fresh(self.coordinator())
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(runtime.step, ZERO).result(timeout=10)
        runtime.step(ZERO)
        self.assertEqual(runtime._parts.env._step_id, 2)

    def test_close_and_step_failure_revoke_witness(self):
        coordinator = self.coordinator()
        closed = self.fresh(coordinator)
        self.assertTrue(closed.close().success)
        self.assertIsNone(closed._admission_witness)
        with self.assertRaises(ContinuationError):
            closed.step(ZERO)
        failed = self.fresh(coordinator)
        with self.assertRaises(Exception):
            failed.step({"product": 0, "order": -1})
        self.assertEqual(failed.state, "INVALID")
        self.assertIsNone(failed._admission_witness)
        with self.assertRaises(ContinuationError):
            failed.step(ZERO)

    def test_strict_admitted_transitions_snapshot_and_termination_match(self):
        strict, fast = self.coordinator("strict-v1"), self.coordinator()
        left, right = self.fresh(strict, max_steps=5), self.fresh(fast, max_steps=5)
        for _ in range(2):
            self.assertEqual(left.step(ZERO), right.step(ZERO))
            self.assertEqual(physical(left), physical(right))
        old, new = self.capture(strict, left), self.capture(fast, right)
        self.assertEqual(old.data, new.data)
        self.assertNotIn("execution_profile", decode_json(new.data))
        a, b = self.restore(strict, new), self.restore(fast, old)
        for action in ({"product": 0, "order": 3}, ZERO, ZERO):
            self.assertEqual(a.step(action), b.step(action))
            self.assertEqual(physical(a), physical(b))
        self.assertEqual((a.state, b.state), ("DONE", "DONE"))
        for runtime in (a, b):
            with self.assertRaises(ContinuationError):
                runtime.step(ZERO)

    def test_admitted_branch_isolation_and_capture_restore_no_model_callbacks(self):
        coordinator = self.coordinator()
        runtime = self.fresh(coordinator)
        runtime.step(ZERO)
        before = physical(runtime)
        with domain.CompanionObserver(lambda: "boundary") as observed:
            snapshot = self.capture(coordinator, runtime)
            a, b = self.restore(coordinator, snapshot, "a"), self.restore(coordinator, snapshot, "b")
        counts = observed.counts.get("boundary", {})
        for key in ("int_trans", "ext_trans", "output", "con_trans", "risk_calls", "scenario_stages"):
            self.assertEqual(counts.get(key, 0), 0)
        sibling = physical(b)
        frozen = bytes(snapshot.data)
        for action in ({"product": 0, "order": 3}, ZERO, ZERO):
            a.step(action)
        self.assertEqual(physical(runtime), before)
        self.assertEqual(physical(b), sibling)
        self.assertEqual(snapshot.data, frozen)
        self.assertNotEqual(physical(a), sibling)

    def test_corrupted_snapshot_rejected_in_both_profiles(self):
        fast = self.coordinator()
        source = self.fresh(fast)
        source.step(ZERO)
        snapshot = self.capture(fast, source)
        corrupt = decode_json(snapshot.data)
        corrupt["boundary_state"]["reward_state"]["last_cumulative_risk"] += 1
        # Re-sign the envelope so model/boundary composition, not merely a
        # checksum mismatch, must reject this semantically corrupted snapshot.
        corrupt["integrity"] = digest({k: v for k, v in corrupt.items() if k != "integrity"})
        encoded = canonical_bytes(corrupt)
        for coordinator in (fast, self.coordinator("strict-v1")):
            with self.assertRaises(ContinuationError):
                coordinator.restore(encoded, self.context("bad"))


if __name__ == "__main__":
    unittest.main()
