"""Direct helper and same-model adapter checks; no productivity/time claims."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor"), str(ROOT / "bench")]

from pyjevsim_bridge.rl.continuation import ContinuationCoordinator, ContinuationRegistry
from pyjevsim_bridge.rl.continuation.contracts import ContinuationError
from pyjevsim_bridge.rl.continuation.state_fields import DeclaredValueFields, OwnedValueField
from bench.research import inventory as model
from bench.research import inventory_native as native
from bench.research.declared_inventory_adapter import make_bundle
from bench.research.transfer import CommonBackend, _advance, _close, branch_actions, oracle, prefix_actions


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative exact integer required")


def sequence(value):
    if type(value) is not list:
        raise ValueError("list required")


class DeclaredBackend(CommonBackend):
    def __init__(self, version, seed=0):
        self.registry = ContinuationRegistry()
        self.bundle = make_bundle(version)
        self.registry.register(self.bundle)
        self.coordinator = ContinuationCoordinator(self.registry)
        self.seed = seed


class DeclaredFieldsTests(unittest.TestCase):
    def setUp(self):
        self.fields = DeclaredValueFields((OwnedValueField("count", integer), OwnedValueField("rows", sequence)))

    def test_capture_and_restore_detach_values_without_rebinding_exceptions(self):
        shared = object()
        source = SimpleNamespace(count=2, rows=[{"value": 3}], clock=shared)
        payload = self.fields.capture(source)
        payload["rows"][0]["value"] = 7
        self.assertEqual(source.rows, [{"value": 3}])
        target = SimpleNamespace(count=0, rows=[], clock=shared)
        self.fields.restore_into(target, payload)
        payload["rows"].append({"value": 8})
        self.assertEqual(target.rows, [{"value": 7}])
        self.assertIs(target.clock, shared)

    def test_missing_extra_or_invalid_field_never_partially_restores(self):
        target = SimpleNamespace(count=1, rows=["original"])
        for payload in ({"count": 8}, {"count": 8, "rows": [], "extra": 1},
                        {"count": True, "rows": []}, {"count": 8, "rows": "wrong"}):
            with self.subTest(payload=payload), self.assertRaises((ContinuationError, ValueError)):
                self.fields.restore_into(target, payload)
            self.assertEqual(vars(target), {"count": 1, "rows": ["original"]})

    def test_nonfinite_object_and_cyclic_values_are_rejected(self):
        cycle = []
        cycle.append(cycle)
        for value in ([float("nan")], [object()], cycle):
            with self.subTest(value=type(value)), self.assertRaises(ContinuationError):
                self.fields.validate({"count": 0, "rows": value})

    def test_validator_cannot_mutate_source_payload_or_target(self):
        def mutate(value):
            value.append("unexpected")
        fields = DeclaredValueFields((OwnedValueField("rows", mutate),))
        payload = {"rows": ["original"]}
        target = SimpleNamespace(rows=["target"])
        with self.assertRaisesRegex(ContinuationError, "mutated"):
            fields.restore_into(target, payload)
        self.assertEqual(payload, {"rows": ["original"]})
        self.assertEqual(target.rows, ["target"])

    def test_validator_success_must_be_none(self):
        fields = DeclaredValueFields((OwnedValueField("count", lambda value: True),))
        with self.assertRaisesRegex(ContinuationError, "return None"):
            fields.validate({"count": 1})

    def test_validator_numeric_type_mutation_is_not_hidden_by_python_equality(self):
        def mutate(value):
            value[0] = True
        fields = DeclaredValueFields((OwnedValueField("rows", mutate),))
        with self.assertRaisesRegex(ContinuationError, "mutated"):
            fields.validate({"rows": [1]})

    def test_declarations_reject_duplicate_or_indirect_attributes(self):
        for name in ("a.b", "_engine", "", 1):
            with self.subTest(name=name), self.assertRaises(ContinuationError):
                OwnedValueField(name, integer)
        with self.assertRaises(ContinuationError):
            DeclaredValueFields((OwnedValueField("count", integer), OwnedValueField("count", integer)))
        with self.assertRaises(ContinuationError):
            DeclaredValueFields([OwnedValueField("count", integer)])

    def test_missing_target_field_rejected_without_adding_attributes(self):
        target = SimpleNamespace(count=1)
        with self.assertRaises(ContinuationError):
            self.fields.restore_into(target, {"count": 2, "rows": []})
        self.assertEqual(vars(target), {"count": 1})

    def test_helper_never_invokes_property_or_setattr_callbacks(self):
        class Target:
            def __setattr__(self, name, value):
                raise AssertionError("setter must not run")
        target = Target()
        vars(target).update(count=1, rows=[])
        self.fields.restore_into(target, {"count": 2, "rows": [3]})
        self.assertEqual((target.count, target.rows), (2, [3]))


class DeclaredInventoryTests(unittest.TestCase):
    def test_separate_profile_and_explicit_source_binding(self):
        backend = DeclaredBackend(1)
        self.assertEqual(backend.bundle.profile.profile_id, "declared-inventory-C1-v1")
        names = {binding.logical_id for binding in backend.bundle.model.source_bindings}
        self.assertIn("continuation.state_fields", names)
        self.assertIn("research.declared_inventory_adapter", names)
        self.assertIn("research.inventory_adapter", names)

    def test_v1_v2_restored_branches_match_original_adapter_and_independent_oracle(self):
        for version in (1, 2):
            with self.subTest(version=version):
                backend, original = DeclaredBackend(version), CommonBackend(version, 0)
                cfg = model.configuration(version, 0)
                source, reference = backend.fresh(cfg), original.fresh(cfg)
                native_source = native.create_native(cfg)
                live = [source, reference, native_source]
                try:
                    prefix = prefix_actions(3)
                    self.assertEqual(_advance(source, prefix), _advance(reference, prefix))
                    _advance(native_source, prefix)
                    with tempfile.TemporaryDirectory() as directory:
                        native.save_native(native_source, directory)
                        native_branch = native.load_native(directory)
                    live.append(native_branch)
                    snapshot = backend.capture(source, 3)
                    original_snapshot = original.capture(reference, 3)
                    first = backend.restore(snapshot, "declared-first")
                    sibling = backend.restore(snapshot, "declared-sibling")
                    native_fields = original.restore(original_snapshot, "original-fields")
                    live += [first, sibling, native_fields]
                    actions = branch_actions(4)
                    expected = oracle(cfg, prefix + actions)[len(prefix):]
                    self.assertEqual(_advance(first, actions), expected)
                    self.assertEqual(_advance(native_fields, actions), expected)
                    self.assertEqual(_advance(native_branch, actions), expected)
                    self.assertEqual(_advance(sibling, actions), expected)
                    self.assertEqual(source._parts.graph.observe(.75), oracle(cfg, prefix)[-1]["observation"])
                    self.assertIs(first._parts.graph.stock.config, first._parts.graph.source.config)
                    self.assertIsNot(first._parts.graph.stock.events, sibling._parts.graph.stock.events)
                finally:
                    for runtime in reversed(live):
                        _close(runtime)


if __name__ == "__main__":
    unittest.main()
