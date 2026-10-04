"""Owned-process and partial-accounting checks; no subprocess/model is run."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from bench import runner
from bench.design import make_manifest


ROOT = Path(__file__).resolve().parents[1]


def packet(spec, case, *, failed=False, altered=False):
    projection = [[{"step": step, "value": int(altered)} for step in range(8)] for _ in range(8)]
    row = {**spec, "status": "failed" if failed else "succeeded", "error": "synthetic failure" if failed else None,
        "seed": case["seed"], "config_sha256": case["config_sha256"],
        "application_wall_seconds": None if failed else 1.0,
        "application_cpu_seconds": None if failed else .5,
        "cleanup_confirmed": None if failed else True, "cleanup_errors": [],
        "worker_exit_code": 2 if failed else 0, "branch_projection_sha256": [runner.sha(branch) for branch in projection],
        "prefixes_executed": 1, "suffix_steps_executed": 64, "expected_suffix_steps": 64, "work_counts": None}
    return {"row": row, "projection": projection, "provenance": None}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = make_manifest()
        self.temporary = tempfile.TemporaryDirectory(prefix="portable runner with spaces ")
        self.addCleanup(self.temporary.cleanup)
        self.campaign = Path(self.temporary.name) / "new cohort"

    def run_mock(self, responder):
        with patch.object(runner, "invoke", side_effect=responder) as invoke, \
             patch.object(runner.host, "metadata", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            result = runner.run_campaign(ROOT, self.campaign, self.manifest)
        return result, invoke

    @staticmethod
    def successful(_root, _campaign, _manifest, spec, case, _budget, **_kwargs):
        return packet(spec, case)

    def test_first_failure_stops_without_retry_and_keeps_partial_cell(self):
        def fail(_root, _campaign, _manifest, spec, case, _budget, **_kwargs):
            return packet(spec, case, failed=True)
        result, invoke = self.run_mock(fail)
        self.assertFalse(result["study_admission"])
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(result["denominators"]["timing"],
            {"planned": 24, "attempted": 1, "succeeded": 0, "failed": 1, "unexecuted": 23})
        cell = json.loads((self.campaign / "cells.jsonl").read_text())
        self.assertFalse(cell["complete"])
        self.assertFalse(cell["exact"])
        self.assertEqual(len(cell["arm_ids"]), 1)

    def test_final_write_error_preserves_terminal_accounting_on_stderr(self):
        original = runner.Budget.write
        def write(budget, path, value, **kwargs):
            if Path(path).name == "execution.json":
                raise OSError("synthetic terminal disk error")
            return original(budget, path, value, **kwargs)
        with patch.object(runner.Budget, "write", write), contextlib.redirect_stderr(io.StringIO()) as errors:
            result, invoke = self.run_mock(self.successful)
        self.assertEqual(invoke.call_count, 24)
        self.assertFalse(result["study_admission"])
        self.assertTrue(result["execution_completed_before_write_error"])
        self.assertIn("synthetic terminal disk error", result["execution_write_error"])
        terminal = json.loads(errors.getvalue())["terminal_execution"]
        self.assertEqual(terminal["denominators"]["timing"]["succeeded"], 24)
        self.assertEqual(terminal["exact_cells"]["timing"]["exact"], 12)

    def invoke_mock(self, process, *, clock_values=None):
        self.campaign.mkdir()
        (self.campaign / "requests").mkdir()
        spec = self.manifest["arms"][0]
        case = self.manifest["cases"][spec["case_id"]]
        budget = runner.Budget(self.campaign, 600, 16 * 1024**2)
        clock_patch = patch.object(runner.time, "perf_counter", side_effect=clock_values) if clock_values else contextlib.nullcontext()
        if clock_values:
            budget.started = 0
        with patch.object(runner.subprocess, "Popen", return_value=process) as launch, \
             patch.object(budget, "check"), patch.object(runner.host, "snapshot", return_value={"available": False}), \
             patch.object(runner.host, "difference", return_value={"available": False}), clock_patch:
            result = runner.invoke(ROOT, self.campaign, self.manifest, spec, case, budget, python="owned python")
        return result, launch

    @staticmethod
    def owned_process(*, needs_kill=False):
        process = MagicMock(returncode=None)
        process.poll.side_effect = lambda: process.returncode
        process.terminate.side_effect = lambda: setattr(process, "returncode", None if needs_kill else -15)
        process.kill.side_effect = lambda: setattr(process, "returncode", -9)
        process.communicate.side_effect = ([subprocess.TimeoutExpired("owned", 2), (b"", b"")]
                                           if needs_kill else [(b"", b"")])
        return process

    def test_timeout_terminates_and_escalates_only_returned_owned_handle(self):
        process = self.owned_process(needs_kill=True)
        result, launch = self.invoke_mock(process, clock_values=[0.0, 0.0, 121.0, 121.0])
        self.assertEqual(launch.call_count, 1)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertTrue(result["row"]["owned_process_terminated"])
        self.assertEqual(result["row"]["status"], "failed")
        self.assertIsNone(result["row"]["cleanup_confirmed"])
        command = launch.call_args.args[0]
        self.assertEqual(command[:3], ["owned python", "-I", "-B"])
        self.assertNotIn("shell", launch.call_args.kwargs)
        self.assertFalse(any((self.campaign / "requests").iterdir()))


if __name__ == "__main__":
    unittest.main()
