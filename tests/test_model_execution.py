"""Explicit, tiny functional checks; never called by the experiment entry point.

Run with the prepared .venv Python. These four tiny arms are not performance data.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from bench.continuation_study.cases import sha
from bench.design import ROOT, make_manifest
from bench.runner import run_campaign


@unittest.skipUnless(importlib.util.find_spec("dill"), "use the prepared .venv for direct model checks")
class ModelExecutionTests(unittest.TestCase):
    def test_two_models_two_methods_in_path_with_spaces(self):
        manifest = make_manifest()
        manifest["arms"] = [dict(arm, prefix_steps=1, branch_count=1, suffix_steps=8,
                                cell_id=f"functional-{arm['model']}-f00-L1-B1",
                                arm_id=f"functional-{arm['model']}-f00-L1-B1-{arm['method']}")
                            for arm in manifest["arms"] if arm["family"] == 0]
        manifest["study_type"] = "functional-verification-only-not-performance-evidence"
        manifest["manifest_sha256"] = sha({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        with tempfile.TemporaryDirectory(prefix="pyjevsim functional space ") as temporary:
            campaign = Path(temporary) / "results with spaces"
            execution = run_campaign(ROOT, campaign, manifest)
            self.assertTrue(execution["study_admission"], execution)
            self.assertEqual(execution["denominators"]["timing"],
                             dict(planned=4, attempted=4, succeeded=4, failed=0, unexecuted=0))
            self.assertEqual(execution["exact_cells"]["timing"], dict(planned=2, complete=2, exact=2))
            rows = [json.loads(line) for line in (campaign / "arms.jsonl").read_text().splitlines()]
            self.assertEqual({(row["model"], row["method"]) for row in rows},
                             {("Q", "N"), ("Q", "C1"), ("M", "N"), ("M", "C1")})
            for row in rows:
                self.assertGreaterEqual(row["application_cpu_seconds"], 0)
                self.assertGreater(row["cpu_scope_wall_seconds"], 0)
                self.assertEqual(row["worker_exit_code"], 0)
                self.assertFalse(row["cpu_endpoint_matches_application_wall"])
            self.assertEqual(list((campaign / "transient").iterdir()), [])
            with self.assertRaises(FileExistsError):
                run_campaign(ROOT, campaign, manifest)


if __name__ == "__main__":
    unittest.main()
