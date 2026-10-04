"""Direct checks for research accounting; no model or timing campaign is run."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import run_research
from bench import runner


class ResearchEntryTests(unittest.TestCase):
    def test_help_does_not_prepare_environment(self):
        with patch.object(run_research, "ensure_environment") as prepare, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                run_research.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        prepare.assert_not_called()

    def test_three_method_purpose_denominators(self):
        arms = [{"arm_id": f"{purpose}-{method}", "purpose": purpose}
                for purpose in ("timing", "counting") for method in ("R", "N", "C1")]
        rows = [{**arms[0], "status": "succeeded"}, {**arms[1], "status": "failed"}]
        self.assertEqual(runner.denominators({"arms": arms}, rows), {
            "timing": dict(planned=3, attempted=2, succeeded=1, failed=1, unexecuted=1),
            "counting": dict(planned=3, attempted=0, succeeded=0, failed=0, unexecuted=3)})

    def test_failed_scientific_case_is_retained_without_retry(self):
        with tempfile.TemporaryDirectory(prefix="research entry ") as temporary:
            output = Path(temporary) / "new"
            options = run_research.parser().parse_args(["--stage", "semantics", "--output", str(output)])
            result = {"status": "failed", "study_admission": False, "cases": [{"failed": True}]}
            fake = types.SimpleNamespace(run_study=lambda **kwargs: result)
            original = run_research.importlib.import_module
            def load(name, *args, **kwargs):
                return fake if name == "bench.research.semantics" else original(name, *args, **kwargs)
            with patch.object(run_research.importlib, "import_module", side_effect=load) as imports, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = run_research.run(options)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads((output / "semantics.json").read_text()), result)
            summary = json.loads((output / "workflow.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertFalse(summary["human_productivity_measured"])
            self.assertEqual(sum(call.args[0] == "bench.research.semantics" for call in imports.call_args_list), 1)

    def test_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = run_research.parser().parse_args(["--output", temporary])
            with self.assertRaises(FileExistsError):
                run_research.run(options)

    def test_analysis_failure_does_not_relabel_executed_stage_unexecuted(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            options = run_research.parser().parse_args(["--stage", "cost", "--output", str(output)])
            execution = {"status": "completed", "denominators": {"timing": {"attempted": 324}},
                         "exact_cells": {"timing": {"exact": 108}}, "study_admission": True}
            with patch("bench.research.cost.make_manifest", return_value={}), \
                    patch("bench.runner.run_campaign", return_value=execution), \
                    patch("bench.research.cost_analysis.analyze", side_effect=TimeoutError("analysis deadline")), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run_research.run(options), 2)
            summary = json.loads((output / "workflow.json").read_text())
            self.assertEqual(summary["unexecuted_stages"], [])
            self.assertEqual(summary["stages"]["cost"]["execution_status"], "completed")
            self.assertEqual(summary["stages"]["cost"]["status"], "failed")
            self.assertEqual(summary["stages"]["cost"]["denominators"]["timing"]["attempted"], 324)

    def test_bounded_new_write_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_research.write_new(root, "receipt.json", {"ok": True}, 1000)
            with self.assertRaises(FileExistsError):
                run_research.write_new(root, "receipt.json", {"ok": False}, 1000)
            with self.assertRaises(RuntimeError):
                run_research.write_new(root, "too-big.json", {"text": "x" * 1000}, 1000)


if __name__ == "__main__":
    unittest.main()
