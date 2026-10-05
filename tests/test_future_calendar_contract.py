"""Future-calendar sensitivity versus snapshot rejection and failure isolation.

These are different evidence categories. No arbitrary ordering of native set
buckets is imposed, and no finite test proves arbitrary DEVS equivalence.
"""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor"), str(ROOT / "bench")]

from pyjevsim_bridge.rl.continuation import ContinuationCoordinator, ContinuationRegistry
from pyjevsim_bridge.rl.continuation.contracts import ContinuationError, canonical_bytes, digest
from pyjevsim_bridge.rl.continuation.generic_boundary import DeclaredFixedDeltaBoundaryProvider
from pyjevsim_bridge.rl.continuation.generic_bundle import make_declared_bundle
from bench.research import inventory as model
from bench.research import inventory_native as native
from bench.research.declared_inventory_adapter import DeclaredInventoryAdapter
from bench.research.transfer import CommonBackend, _advance, _close, branch_actions, oracle, prefix_actions


class FaultAdapter(DeclaredInventoryAdapter):
    """Trusted installed test provider with an explicitly injected failure."""
    def __init__(self):
        super().__init__(1)
        self.fail_after_restore = False
        self.released_graphs = []
        self.allocated_graphs = []

    def allocate_shell(self, descriptor, services, cleanup):
        graph = super().allocate_shell(descriptor, services, cleanup)
        self.allocated_graphs.append(graph)
        cleanup.add("test-candidate-graph", lambda: self.released_graphs.append(id(graph)))
        return graph

    def restore_into(self, graph, state):
        super().restore_into(graph, state)
        if self.fail_after_restore:
            raise ValueError("deliberate postallocation restore failure")


class FaultBackend(CommonBackend):
    def __init__(self):
        adapter = FaultAdapter()
        profile = "declared-inventory-fault-test"
        boundary = DeclaredFixedDeltaBoundaryProvider(profile_id=profile, schema_id="inventory-boundary-v1",
            observation_validator=model.validate_observation, reward_validator=model.validate_reward_v1,
            initial_reward_state=model.initial_reward(1), source_bindings=adapter.source_bindings)
        self.bundle = make_declared_bundle(profile_id=profile, model=adapter, boundary=boundary,
            model_provider_id="inventory-fault-test", projection_id="inventory-observation-v1")
        self.registry = ContinuationRegistry()
        self.registry.register(self.bundle)
        self.coordinator = ContinuationCoordinator(self.registry)
        self.seed = 0


class FutureCalendarTests(unittest.TestCase):
    def test_live_future_calendar_delay_changes_receipt_without_rewriting_history(self):
        cfg = model.configuration(1, 0)
        control, faulty = native.create_native(cfg), native.create_native(cfg)
        try:
            _advance(control, prefix_actions(3))
            _advance(faulty, prefix_actions(3))
            history = copy.deepcopy(faulty.graph.stock.events)
            wrapper = faulty.engine.product_port_map[faulty.graph.stock]
            self.assertEqual(wrapper.request_time, 1.)
            wrapper.request_time = 1.25
            faulty.engine.min_schedule_item.push(wrapper)
            self.assertEqual(faulty.graph.stock.events, history)
            self.assertEqual(faulty.graph.stock.pending, [{"due": 1., "quantity": 5}])
            # Advance the native engine directly: a new action would itself
            # reschedule the stock. This is observer sensitivity, not an API use.
            control.engine.step(1.)
            faulty.engine.step(1.)
            self.assertEqual((control.engine.global_time, faulty.engine.global_time), (1., 1.))
            self.assertEqual(control.graph.stock.received, 5)
            self.assertEqual(faulty.graph.stock.received, 0)
            self.assertEqual(faulty.graph.stock.events, history)
        finally:
            _close(control)
            _close(faulty)

    def test_consistent_engine_calendar_but_wrong_domain_deadline_rejected_without_damage(self):
        backend = CommonBackend(1, 0)
        source = backend.fresh(model.configuration(1, 0))
        sibling = None
        try:
            _advance(source, prefix_actions(3))
            snapshot = backend.capture(source, 3)
            sibling = backend.restore(snapshot, "untouched-sibling")
            before_source = backend.coordinator.semantic_view(source)
            before_sibling = backend.coordinator.semantic_view(sibling)
            payload = json.loads(snapshot.data)
            state = payload["engine_state"]
            wrapper = state["executor"]["models"]["inventory-stock"]
            behavior = state["behaviors"]["inventory-stock"]
            wrapper["request_time"] = wrapper["next_event_time"] = 1.25
            state["executor"]["calendar"]["inventory-stock"] = 1.25
            behavior["states"][behavior["cur_state"]] = 1.25 - behavior["global_time"]
            self.assertIsNone(backend.bundle.engine.validate_payload(state, payload["topology"], backend.bundle.profile))
            payload["integrity"] = digest({key: value for key, value in payload.items() if key != "integrity"})
            with self.assertRaisesRegex(ContinuationError, "domain deadline"):
                backend.restore(canonical_bytes(payload), "rejected-calendar")
            self.assertEqual(backend.coordinator.semantic_view(source), before_source)
            self.assertEqual(backend.coordinator.semantic_view(sibling), before_sibling)
            expected = oracle(model.configuration(1, 0), prefix_actions(3) + branch_actions(4))[3:]
            self.assertEqual(_advance(source, branch_actions(4)), expected)
            self.assertEqual(_advance(sibling, branch_actions(4)), expected)
        finally:
            if sibling is not None:
                _close(sibling)
            _close(source)

    def test_postallocation_failure_releases_candidate_not_source_or_sibling(self):
        backend = FaultBackend()
        source = backend.fresh(model.configuration(1, 0))
        sibling = None
        adapter = backend.bundle.model
        try:
            _advance(source, prefix_actions(3))
            snapshot = backend.capture(source, 3)
            sibling = backend.restore(snapshot, "surviving-sibling")
            before_source = backend.coordinator.semantic_view(source)
            before_sibling = backend.coordinator.semantic_view(sibling)
            adapter.fail_after_restore = True
            with self.assertRaisesRegex(ContinuationError, "postallocation") as caught:
                backend.restore(snapshot, "failed-candidate")
            self.assertEqual(caught.exception.phase, "restore")
            self.assertEqual(caught.exception.state_disposition, "candidate-not-published")
            self.assertEqual(adapter.released_graphs, [id(adapter.allocated_graphs[-1])])
            self.assertNotIn(id(source._parts.graph), adapter.released_graphs)
            self.assertNotIn(id(sibling._parts.graph), adapter.released_graphs)
            adapter.fail_after_restore = False
            self.assertEqual(backend.coordinator.semantic_view(source), before_source)
            self.assertEqual(backend.coordinator.semantic_view(sibling), before_sibling)
            expected = oracle(model.configuration(1, 0), prefix_actions(3) + branch_actions(4))[3:]
            self.assertEqual(_advance(source, branch_actions(4)), expected)
            self.assertEqual(_advance(sibling, branch_actions(4)), expected)
        finally:
            if sibling is not None:
                _close(sibling)
            _close(source)


if __name__ == "__main__":
    unittest.main()
