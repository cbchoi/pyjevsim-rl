"""Direct break-even accounting and entry checks; no performance cohort."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import run_research
from bench.research import break_even_campaign as campaign
from bench.research.break_even_design import load_protocol, make_plan, sha


ROOT = Path(__file__).resolve().parents[1]


class BreakEvenCampaignTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol()
        self.plan = make_plan(self.protocol, "calibration")
        self.plan["arms"] = self.plan["arms"][:6]
        self.plan["source_identity"] = "synthetic-source"
        for arm in self.plan["arms"]:
            arm["source_identity"] = "synthetic-source"
        self.temporary = tempfile.TemporaryDirectory(prefix="break even accounting ")
        self.addCleanup(self.temporary.cleanup)
        self.study = Path(self.temporary.name)

    @staticmethod
    def packet(spec):
        witness = [[{"stock": step}] for step in range(spec["B"])]
        projection = [[{"state": "synthetic"}] for _ in range(spec["B"])] if spec["role"] == "companion" else []
        return {"row": {**spec, "status": "succeeded", "cleanup_confirmed": True,
            "actual_source_identity": spec["source_identity"], "process_wall_seconds": .01,
            "scalar_witness": witness, "scalar_witness_sha256": sha(witness)},
            "projection": projection, "provenance": {"synthetic": True}, "witness": witness}

    def test_six_arm_companion_and_scalar_contracts_are_separate(self):
        packets = [self.packet(spec) for spec in self.plan["arms"]]
        result = campaign._cell_summary(self.plan["arms"][0], packets)
        self.assertTrue(result["exact"])
        self.assertFalse(result["timing_full_trace_exact"])
        packets[-1]["witness"] = "changed"
        self.assertFalse(campaign._cell_summary(self.plan["arms"][0], packets)["exact"])
        packets[-1] = self.packet(self.plan["arms"][-1])
        packets[-1]["row"]["actual_source_identity"] = "other"
        self.assertFalse(campaign._cell_summary(self.plan["arms"][0], packets)["exact"])

    def run_collect(self, responder):
        with patch.object(campaign.runner, "invoke", side_effect=responder) as invoke, \
             patch.object(campaign.host, "metadata", return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = campaign.collect(ROOT, self.study, self.plan, self.protocol)
        return result, invoke

    def test_first_failure_preserves_partial_denominators_no_retry(self):
        def respond(_root, _study, _plan, spec, _case, _budget, **kwargs):
            packet = self.packet(spec)
            packet["row"].update(status="failed", error="synthetic failure")
            return packet
        (execution, rows, cells, _), invoke = self.run_collect(respond)
        self.assertEqual(invoke.call_count, 1)
        self.assertFalse(execution["study_admission"])
        self.assertEqual(sum(item["unexecuted"] for item in execution["denominators"].values()), 5)
        self.assertEqual(len(cells), 1)
        self.assertFalse(cells[0]["exact"])
        self.assertNotIn("scalar_witness", rows[0])
        self.assertTrue((self.study / "calibration/execution.json").is_file())

    def test_persisted_plan_hash_matches_effective_budget(self):
        def respond(_root, _study, _plan, spec, _case, _budget, **kwargs):
            return self.packet(spec)
        (execution, rows, cells, _), _ = self.run_collect(respond)
        self.assertTrue(execution["study_admission"])
        plan = json.loads((self.study / "calibration/protocol.json").read_bytes())
        expected = plan.pop("manifest_sha256")
        self.assertEqual(sha(plan), expected)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(cells), 1)

    def test_provenance_write_failure_does_not_erase_attempt(self):
        original = campaign._write
        def write(budget, path, value, **kwargs):
            if path.parent.name == "provenance":
                raise OSError("synthetic provenance failure")
            return original(budget, path, value, **kwargs)
        def respond(_root, _study, _plan, spec, _case, _budget, **kwargs):
            return self.packet(spec)
        with patch.object(campaign, "_write", side_effect=write):
            (execution, rows, cells, _), invoke = self.run_collect(respond)
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(cells), 1)
        self.assertEqual(len(execution["unpersisted_rows"]), 1)
        self.assertFalse(execution["study_admission"])

    def test_break_even_entry_uses_design_caps_and_rejects_overrun(self):
        options = run_research.parser().parse_args(["--stage", "break-even"])
        with patch.object(campaign, "run_workflow", return_value=0) as run:
            self.assertEqual(run_research.run(options), 0)
        self.assertEqual(run.call_args.kwargs["max_seconds"], 9000)
        self.assertEqual(run.call_args.kwargs["max_bytes"], 32 * 1024**2)
        for args in (["--budget-seconds", "9001"], ["--budget-seconds", "nan"], ["--max-mib", "33"]):
            with self.assertRaises(ValueError):
                run_research.run(run_research.parser().parse_args(["--stage", "break-even", *args]))

    def test_partial_collection_is_analyzed_but_not_promoted_to_validation(self):
        output = self.study / "new"
        execution = {"status": "failed", "study_admission": False, "stop_reason": "synthetic stop",
            "source_check": {"complete": False, "consistent": True}, "denominators": {},
            "cells": {"planned": 144, "admitted": 0}}
        def collect(*args, **kwargs):
            (output / "calibration").mkdir()
            return copy.deepcopy(execution), [], [], time.perf_counter() + 120
        with patch.object(campaign, "collect", side_effect=collect) as invoke, \
             patch("bench.research.break_even_analysis.fit_calibration", return_value={
                 "status": "partial", "study_admission": False}) as fit, \
             patch("bench.research.break_even_analysis.select_validation") as select, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(campaign.run_workflow(ROOT, output), 2)
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(fit.call_count, 1)
        select.assert_not_called()
        self.assertTrue((output / "calibration/fit.json").is_file())
        summary = json.loads((output / "workflow.json").read_bytes())
        self.assertEqual(summary["unexecuted_stages"], ["validation", "transfer"])


if __name__ == "__main__":
    unittest.main()
