"""Direct TASK208 semantic tests, not benchmark timing observations."""
import tempfile
import unittest

from bench.research import break_even_domain as d
from bench.research import break_even_native as native
from bench.research.break_even_adapter import make_bundle
from bench.research.break_even_oracle import forecast, oracle
from pyjevsim_bridge.rl.continuation import (
    BranchContext, CaptureRequest, ContinuationCoordinator, ContinuationRegistry, ResetRequest)


POLICY = {"policy_sha256": d.sha({"policy": "test-fixed-orders"}), "policy_version": 0,
          "feature_contract_sha256": d.sha({"features": "risk-scalars"})}


def sampling(seed, segment="prefix", branch="prefix"):
    return {"domain": "pyjevsim-live-branch-v1", "phase": "break-even-direct", "master": seed,
            "segment": segment, "logical_branch_id": branch, "run_id": "break-even-study",
            "generation": 0, "worker_id": "serial", "episode_id": "one", "sampling_seed": seed}


class Common:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bundle = make_bundle()
        registry = ContinuationRegistry()
        registry.register(self.bundle)
        self.coordinator = ContinuationCoordinator(registry)

    def fresh(self):
        seed = self.cfg["input_seed"]
        return self.coordinator.create_fresh(ResetRequest(self.bundle.profile.profile_id, self.cfg, seed,
            "risk-common", "break-even-study", .25, 81, POLICY, sampling(seed)))

    def capture(self, runtime, step):
        return self.coordinator.capture(runtime, CaptureRequest(self.bundle.profile.profile_id,
            step, "risk-family", "break-even-study", POLICY))

    def restore(self, snapshot, branch):
        return self.coordinator.restore(snapshot, BranchContext("risk-family", "break-even-study", branch,
            POLICY, sampling(self.cfg["input_seed"], "suffix", branch), "risk-" + branch))


def parts(runtime):
    return runtime._parts if hasattr(runtime, "_parts") else runtime


def physical(runtime):
    p = parts(runtime)
    reward = p.boundary_state.value if hasattr(p, "boundary_state") else p.reward_state
    return p.graph.physical_state(p.engine.global_time, reward)


def advance(runtime, actions, observed):
    rows = []
    for action in actions:
        value = runtime.step(action)
        rows.append({"observation": value[0], "reward": value[1], "physical": physical(runtime),
                     "events": observed.drain_events()})
        if value[2] or value[3]:
            raise AssertionError("unexpected early end")
    return rows


class BreakEvenDomainTests(unittest.TestCase):
    def check_tc02_cut(self, prefix, suffix):
        """Fresh replay and both restored runtimes against a separate event oracle."""
        cfg = d.configuration()
        common = Common(cfg)
        phase = ["prefix"]
        runtimes = []
        traces = {}
        with tempfile.TemporaryDirectory() as folder, d.CompanionObserver(lambda: phase[0]) as observed:
            try:
                native_source = native.create_native(cfg, instance_id="tc02-native-source")
                runtimes.append(native_source)
                common_source = common.fresh()
                runtimes.append(common_source)
                prefix_rows = []
                for source in (native_source, common_source):
                    observed.drain_events()
                    prefix_rows.append(advance(source, prefix, observed))
                self.assertEqual(prefix_rows[0], prefix_rows[1])
                self.assertEqual(prefix_rows[0], oracle(cfg, prefix))
                cut_states = [physical(source) for source in (native_source, common_source)]
                self.assertEqual(cut_states[0], cut_states[1])
                self.assertEqual(cut_states[0]["logical_time"], len(prefix)*.25)

                phase[0] = "capture"
                native.save_native(native_source, folder)
                snapshot = common.capture(common_source, len(prefix))
                frozen = bytes(snapshot.data)
                phase[0] = "restore"
                nbranch = native.load_native(folder, "tc02-native-branch")
                runtimes.append(nbranch)
                cbranch = common.restore(snapshot, "tc02-branch")
                runtimes.append(cbranch)
                for branch in (nbranch, cbranch):
                    self.assertEqual(physical(branch), cut_states[0])

                phase[0] = "replay-prefix"
                replay = native.create_native(cfg, instance_id="tc02-independent-replay")
                runtimes.append(replay)
                observed.drain_events()
                advance(replay, prefix, observed)
                self.assertEqual(physical(replay), cut_states[0])
                expected = oracle(cfg, prefix + suffix)[len(prefix):]
                phase[0] = "suffix"
                for method, branch in (("R", replay), ("N", nbranch), ("C1", cbranch)):
                    observed.drain_events()
                    traces[method] = advance(branch, suffix, observed)
                    self.assertEqual(traces[method], expected)
                    first = traces[method][0]["events"][0]
                    self.assertEqual(first, {"kind": "action", "time": len(prefix)*.25, **suffix[0]})
                self.assertEqual([physical(source) for source in (native_source, common_source)], cut_states)
                self.assertEqual(snapshot.data, frozen)
                for restore_phase in ("capture", "restore"):
                    counts = observed.counts.get(restore_phase, {})
                    for key in ("int_trans", "ext_trans", "output", "con_trans", "risk_calls", "scenario_stages"):
                        self.assertEqual(counts.get(key, 0), 0)
                return cut_states[0], prefix_rows[0], traces
            finally:
                for runtime in reversed(runtimes):
                    self.assertTrue(runtime.close().success)

    def test_tc02_cut_zero_new_action_three_way(self):
        cut, before, traces = self.check_tc02_cut([], [
            {"product": 0, "order": 7}, {"product": 0, "order": 0}, {"product": 0, "order": 0}])
        self.assertEqual((cut["cursor"], cut["logical_time"]), (0, 0.))
        self.assertEqual(before, [])
        for rows in traces.values():
            receipts = [event for row in rows for event in row["events"] if event["kind"] == "replenish"]
            self.assertEqual(receipts, [{"kind": "replenish", "time": .5, "product": 0, "quantity": 7}])

    def test_tc02_after_demand_new_action_three_way(self):
        cut, before, traces = self.check_tc02_cut([{"product": 0, "order": 0}], [
            {"product": 1, "order": 5}, {"product": 1, "order": 0}, {"product": 1, "order": 0}])
        self.assertEqual((cut["cursor"], cut["logical_time"]), (1, .25))
        self.assertEqual(before[-1]["events"][-1]["demand_id"], 0)
        for rows in traces.values():
            self.assertEqual(rows[-1]["physical"]["domain"]["products"][1][4], 5)

    def test_tc02_pending_receipt_new_actions_three_way(self):
        # A zero order at the cut must retain the old pending order. The model
        # allows one pending order, so a new positive order follows its receipt.
        cut, _, traces = self.check_tc02_cut([
            {"product": 0, "order": 0}, {"product": 0, "order": 7}], [
            {"product": 1, "order": 0}, {"product": 1, "order": 3},
            {"product": 1, "order": 0}, {"product": 1, "order": 0}])
        self.assertEqual(cut["domain"]["pending"], {"product": 0, "order": 7, "due": .75})
        for rows in traces.values():
            receipts = [event for row in rows for event in row["events"] if event["kind"] == "replenish"]
            self.assertEqual(receipts, [
                {"kind": "replenish", "time": .75, "product": 0, "quantity": 7},
                {"kind": "replenish", "time": 1.25, "product": 1, "quantity": 3}])

    def test_tc02_committed_receipt_demand_tie_then_new_action_three_way(self):
        prefix = [{"product": 0, "order": 0} for _ in range(65)]
        prefix += [{"product": 0, "order": 7}, {"product": 0, "order": 0}]
        cut, before, traces = self.check_tc02_cut(prefix, [
            {"product": 0, "order": 3}, {"product": 0, "order": 0}, {"product": 0, "order": 0}])
        self.assertEqual(cut["logical_time"], 16.75)
        self.assertIsNone(cut["domain"]["pending"])
        tied = [event for event in before[-1]["events"] if event["time"] == 16.75]
        self.assertEqual([event["kind"] for event in tied], ["replenish", "demand"])
        self.assertEqual(tied[-1]["fulfilled"], 7)
        for rows in traces.values():
            self.assertEqual(rows[0]["events"][0],
                             {"kind": "action", "time": 16.75, "product": 0, "order": 3})
            self.assertFalse(any(event.get("demand_id") == 67 for row in rows for event in row["events"]))
            receipts = [event for row in rows for event in row["events"] if event["kind"] == "replenish"]
            self.assertEqual(receipts, [{"kind": "replenish", "time": 17.25, "product": 0, "quantity": 3}])

    def test_exact_input_and_nested_balanced_actions(self):
        for structure in ("smooth", "bursty"):
            rows = d.make_input(structure, 981000)
            self.assertEqual(len(rows), 80)
            self.assertEqual(sum(row["at"] <= 16 for row in rows), 64)
            self.assertEqual(next(row for row in rows if row["id"] == 63),
                             {"id": 63, "at": 15.875, "product": 0, "quantity": 100})
            self.assertEqual(next(row for row in rows if row["id"] == 67)["at"], 16.75)
        seed = d.namespace_seed("be-actions-v1", 981000)
        catalogs = [d.make_action_plan(seed, B)["quantities"] for B in (4, 8, 16)]
        self.assertTrue(set(catalogs[0]) < set(catalogs[1]) < set(catalogs[2]))
        for quantities in catalogs:
            self.assertEqual(sum(quantities)/len(quantities), 8.5)
        a, b = d.configuration(S=8), d.configuration(S=512)
        self.assertEqual(a["demands"], b["demands"])
        self.assertEqual(a["forecast_seed"], b["forecast_seed"])

    def test_forecast_independent_oracle_and_executed_stage_counters(self):
        product = {"stock": 3, "regime": 1}
        same = {"product": 0, "order": 7, "due": .5}
        other = {"product": 1, "order": 7, "due": .5}
        with d.CompanionObserver(lambda: "kernel") as observed:
            for K in (1, 16):
                for pending in (None, same, other):
                    self.assertEqual(d.forecast_risk(product, 0, pending, .125, 3, K, 8213),
                                     forecast(product, 0, pending, .125, 3, K, 8213))
        self.assertEqual(observed.counts["kernel"]["risk_calls"], 6)
        self.assertEqual(observed.counts["kernel"]["scenario_stages"], 3*(1+16)*8)
        self.assertEqual(observed.counts["kernel"]["validation_risk_calls"], 0)
        self.assertEqual(d.forecast_risk(product, 0, None, .125, 3, 16, 8213),
                         d.forecast_risk(product, 0, other, .125, 3, 16, 8213))

    def test_native_smooth_and_burst_exact_independent_event_oracle(self):
        for structure in ("smooth", "bursty"):
            cfg = d.configuration(input_structure=structure)
            actions = d.make_action_plan(d.namespace_seed("be-actions-v1", cfg["input_seed"]), 4)
            sequence = actions["prefix"] + actions["branches"][0]
            with d.CompanionObserver(lambda: "simulation") as observed:
                runtime = native.create_native(cfg)
                try:
                    actual = advance(runtime, sequence, observed)
                    self.assertEqual(actual, oracle(cfg, sequence))
                    events = [event for row in actual for event in row["events"] if event.get("demand_id") == 67]
                    self.assertEqual(events[0]["fulfilled"], actions["quantities"][0])
                    self.assertEqual(observed.counts["simulation"]["normal_risk_calls"], 80)
                    self.assertEqual(observed.counts["simulation"]["normal_scenario_stages"], 640)
                finally:
                    self.assertTrue(runtime.close().success)

    def test_replay_native_common_branches_isolation_and_no_restore_transitions(self):
        cfg = d.configuration()
        prefix = [{"product": 0, "order": 0} for _ in range(64)]
        common = Common(cfg)
        phase = ["prefix"]
        sources, branches = [], []
        with tempfile.TemporaryDirectory() as folder, d.CompanionObserver(lambda: phase[0]) as observed:
            try:
                nsource, csource = native.create_native(cfg), common.fresh()
                sources = [nsource, csource]
                for source in sources:
                    advance(source, prefix, observed)
                before = [physical(source) for source in sources]
                self.assertEqual(before[0], before[1])
                self.assertEqual(before[0]["domain"]["products"][0][0], 0)
                phase[0] = "capture"
                receipt = native.save_native(nsource, folder)
                self.assertIn("inventory-stock.simx", receipt["files"])
                snapshot = common.capture(csource, 64)
                snapshot_bytes = bytes(snapshot.data)
                phase[0] = "restore"
                na, nb = native.load_native(folder, "native-a"), native.load_native(folder, "native-b")
                ca, cb = common.restore(snapshot, "a"), common.restore(snapshot, "b")
                branches += [na, nb, ca, cb]
                sibling_before = [physical(nb), physical(cb)]
                phase[0] = "suffix"
                for q, pair in ((1, (na, ca)), (16, (nb, cb))):
                    actions = [{"product": 0, "order": q}] + [{"product": 0, "order": 0} for _ in range(15)]
                    expected = oracle(cfg, prefix + actions)[64:]
                    replay = native.create_native(cfg, instance_id="independent-replay")
                    branches.append(replay)
                    observed.drain_events()
                    advance(replay, prefix, observed)
                    self.assertEqual(advance(replay, actions, observed), expected)
                    for branch in pair:
                        observed.drain_events()
                        self.assertEqual(advance(branch, actions, observed), expected)
                    if q == 1:
                        self.assertEqual([physical(nb), physical(cb)], sibling_before)
                phase[0] = "restore"
                repeat = common.restore(snapshot, "a-repeat")
                native_repeat = native.load_native(folder, "native-a-repeat")
                branches.extend((repeat, native_repeat))
                observed.drain_events()
                phase[0] = "suffix"
                actions = [{"product": 0, "order": 1}] + [{"product": 0, "order": 0} for _ in range(15)]
                self.assertEqual(advance(repeat, actions, observed), oracle(cfg, prefix + actions)[64:])
                self.assertEqual(advance(native_repeat, actions, observed), oracle(cfg, prefix + actions)[64:])
                self.assertEqual([physical(source) for source in sources], before)
                self.assertEqual(snapshot.data, snapshot_bytes)
                for name in ("capture", "restore"):
                    row = observed.counts.get(name, {})
                    self.assertTrue(all(row.get(key, 0) == 0 for key in
                        ("int_trans", "ext_trans", "output", "con_trans", "risk_calls", "scenario_stages")))
            finally:
                for runtime in branches + sources:
                    self.assertTrue(runtime.close().success)

    def test_dormant_last_product_is_real_restored_state(self):
        cfg = d.configuration(S=512)
        common = Common(cfg)
        source = common.fresh()
        branch = None
        try:
            source.step({"product": 0, "order": 0})
            snapshot = common.capture(source, 1)
            branch = common.restore(snapshot, "last-product")
            for action in ({"product": 511, "order": 7}, {"product": 511, "order": 0},
                           {"product": 511, "order": 0}):
                result = branch.step(action)
            self.assertEqual((result[0]["stock"], result[0]["received"]), (27, 7))
            self.assertEqual(parts(source).graph.stock.products[511]["stock"], 20)
            self.assertEqual(len(physical(branch)["domain"]["products"]), 512)
        finally:
            if branch is not None:
                self.assertTrue(branch.close().success)
            self.assertTrue(source.close().success)

    def test_corrupt_reward_and_table_are_rejected(self):
        cfg = d.configuration()
        runtime = native.create_native(cfg)
        try:
            runtime.step({"product": 0, "order": 0})
            runtime.reward_state["last_cumulative_risk"] += 1
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    native.save_native(runtime, folder)
        finally:
            self.assertTrue(runtime.close().success)

        common = Common(cfg)
        runtime = common.fresh()
        try:
            parts(runtime).graph.stock.products[-1]["stock"] += 1
            with self.assertRaises(Exception):
                common.capture(runtime, 0)
        finally:
            self.assertTrue(runtime.close().success)

    def test_burst_restore_preserves_all_risk_in_a_boundary(self):
        cfg = d.configuration(input_structure="bursty", K=16)
        common = Common(cfg)
        sources, branches = [], []
        prefix = [{"product": 0, "order": 0} for _ in range(64)]
        suffix = [{"product": 0, "order": 4}] + [{"product": 0, "order": 0} for _ in range(15)]
        with tempfile.TemporaryDirectory() as folder, d.CompanionObserver() as observed:
            try:
                sources = [native.create_native(cfg), common.fresh()]
                for runtime in sources:
                    advance(runtime, prefix, observed)
                native.save_native(sources[0], folder)
                snapshot = common.capture(sources[1], 64)
                branches = [native.load_native(folder), common.restore(snapshot, "burst")]
                expected = oracle(cfg, prefix + suffix)[64:]
                for runtime in branches:
                    observed.drain_events()
                    self.assertEqual(advance(runtime, suffix, observed), expected)
            finally:
                for runtime in branches + sources:
                    self.assertTrue(runtime.close().success)

    def test_actual_confluence_receipt_before_demand(self):
        cfg = d.configuration()
        sequence = [{"product": 0, "order": 0} for _ in range(65)]
        sequence += [{"product": 0, "order": 7}, {"product": 0, "order": 0}]
        with d.CompanionObserver(lambda: "tie") as observed:
            runtime = native.create_native(cfg)
            try:
                rows = advance(runtime, sequence, observed)
                self.assertEqual(rows, oracle(cfg, sequence))
                tied = [event for event in rows[-1]["events"] if event["time"] == 16.75]
                self.assertEqual([event["kind"] for event in tied], ["replenish", "demand"])
                self.assertEqual(tied[-1]["fulfilled"], 7)
                self.assertGreater(observed.counts["tie"]["con_trans"], 0)
            finally:
                self.assertTrue(runtime.close().success)
if __name__ == "__main__":
    unittest.main()
