import unittest
from collections import Counter

from bench.design import make_manifest


class DesignTests(unittest.TestCase):
    def test_counts_and_balanced_order(self):
        manifest = make_manifest({})
        arms = manifest["arms"]
        self.assertEqual(len(arms), 24)
        self.assertEqual(len({row["arm_id"] for row in arms}), 24)
        self.assertEqual(len({row["cell_id"] for row in arms}), 12)
        for model in ("Q", "M"):
            first = Counter(row["method"] for row in arms if row["model"] == model and row["method_position"] == 0)
            self.assertEqual(first, {"N": 3, "C1": 3})

    def test_paired_inputs_and_fresh_seeds(self):
        manifest = make_manifest({})
        for cell in {row["cell_id"] for row in manifest["arms"]}:
            pair = [row for row in manifest["arms"] if row["cell_id"] == cell]
            self.assertEqual(pair[0]["case_id"], pair[1]["case_id"])
        self.assertEqual(manifest["cases"]["Q-f00"]["seed"], 961000)
        changed = make_manifest({"seed_offset": 100})
        self.assertEqual(changed["cases"]["Q-f00"]["seed"], 961100)
        self.assertNotEqual(manifest["manifest_sha256"], changed["manifest_sha256"])

    def test_no_hidden_local_path_or_c0_dependency(self):
        import json
        text = json.dumps(make_manifest({}))
        self.assertNotIn("D:/", text)
        self.assertNotIn("C0", text)
        self.assertNotIn("site-packages", text)

    def test_budget_and_condition_validation(self):
        for config in ({"budget_seconds": 0}, {"budget_seconds": 7201}, {"budget_seconds": float("nan")},
                       {"max_bytes": 0}, {"max_mib": 129}, {"seed_offset": -1},
                       {"condition": "clean-certified"}, {"families": 3}, {"preset": "unreviewed"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                make_manifest(config)

    def test_label_is_not_interference_proof(self):
        manifest = make_manifest({"condition": "idle"})
        self.assertTrue(manifest["condition_is_user_label_not_clean_host_evidence"])
        self.assertFalse(manifest["host_interference_controlled"])


if __name__ == "__main__":
    unittest.main()
