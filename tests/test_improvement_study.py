"""Direct tests for the new paired comparison, not experimental observations."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from bench.research import improvement_study as study


def packets():
    result = []
    for role in ("companion", "timing"):
        for method in study.METHODS:
            result.append({"row": dict(cell_id="cell", family=0, K=1, S=8, B=4, role=role,
                method=method, status="succeeded", cleanup_confirmed=True,
                source_identity="same", actual_source_identity="same"),
                "projection": [[{"state": 1}]] if role == "companion" else [], "witness": [[1, 2]]})
    return result


class ImprovementPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = study.make_plan()

    def test_fixed_disjoint_denominators(self):
        self.assertEqual(len(self.plan["arms"]), 96)
        self.assertEqual(len(self.plan["cases"]), 12)
        self.assertEqual({row["input_seed"] for row in self.plan["arms"]}, {985000, 985001, 985002})
        self.assertEqual(len({row["arm_id"] for row in self.plan["arms"]}), 96)

    def test_profile_labels_and_same_cell_inputs(self):
        for case_id in self.plan["cases"]:
            rows = [r for r in self.plan["arms"] if r["cell_id"] == case_id]
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({r["action_identity"] for r in rows}), 1)
            self.assertEqual(len({r["input_identity"] for r in rows}), 1)
            for row in rows:
                self.assertEqual(row["execution_profile"], study.PROFILES[row["method"]])

    def test_companion_and_normal_output_exactness_are_separate(self):
        data = packets()
        self.assertTrue(study.compare_cell(data)["exact"])
        data[-1]["witness"] = [[1, 3]]
        result = study.compare_cell(data)
        self.assertFalse(result["exact"])
        self.assertTrue(result["companion_exact"])
        self.assertFalse(result["normal_output_agreement"])

    def test_source_mismatch_and_partial_fail_closed(self):
        data = packets()
        self.assertFalse(study.compare_cell(data[:-1])["exact"])
        data[0]["row"]["actual_source_identity"] = "changed"
        self.assertFalse(study.compare_cell(data)["exact"])

    def test_missing_witness_not_equal_null_success(self):
        data = packets()
        for packet in data:
            packet["witness"] = None
        self.assertFalse(study.compare_cell(data)["exact"])

    def test_duplicate_roles_cannot_complete(self):
        data = packets()
        data[-1] = copy.deepcopy(data[-2])
        self.assertFalse(study.compare_cell(data)["complete"])

    def test_profile_translation_preserves_external_method(self):
        spec = next(r for r in self.plan["arms"] if r["method"] == "C1A")
        with patch.object(study.kernel, "execute_arm", return_value=({"status": "succeeded"}, [])) as call:
            row, _ = study.execute_arm(spec, {}, None, None, {})
            self.assertEqual(call.call_args.args[0]["method"], "C1")
            self.assertIs(call.call_args.kwargs["backend_factory"], study.AdmittedBackend)
            self.assertEqual(row["method"], "C1A")
            self.assertFalse(row["strict_step_admission"])

    def test_mislabeled_profile_rejected(self):
        spec = dict(self.plan["arms"][0], execution_profile="wrong")
        with self.assertRaises(ValueError):
            study.execute_arm(spec, {}, None, None, {})

    def test_incomplete_statistical_denominator_is_not_bootstrapped(self):
        result = study.summarize([], [])
        for row in result["conditions"]:
            self.assertEqual(row["families"], 0)
            self.assertIsNone(row["ratios"]["C1A/R"]["paired_geometric_ratio"])
            self.assertIsNone(row["ratios"]["C1A/R"]["exploratory_pointwise_95_interval"])

    def test_known_family_ratios(self):
        cells, rows = [], []
        for family in range(3):
            for K, B in ((1, 4), (1, 16), (4096, 4), (4096, 16)):
                cell_id = f"f{family}-{K}-{B}"
                cells.append(dict(cell_id=cell_id, family=family, K=K, B=B, exact=True))
                for method, seconds in (("R", 8), ("N", 1), ("C1", 4), ("C1A", 2)):
                    rows.append(dict(cell_id=cell_id, method=method, role="timing", workflow_wall_seconds=seconds))
        condition = study.summarize(rows, cells)["conditions"][0]
        self.assertEqual(condition["ratios"]["C1A/C1"]["paired_geometric_ratio"], .5)
        self.assertEqual(condition["ratios"]["C1A/C1"]["exploratory_pointwise_95_interval"], [.5, .5])

    def test_invoke_exception_records_failed_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(study.runner, "invoke", side_effect=OSError("owned child observation failed")):
                result = study.run_study(Path(directory) / "new")
            counts = result["denominators"].values()
            self.assertFalse(result["study_admission"])
            self.assertEqual(sum(c["attempted"] for c in counts), 1)
            self.assertEqual(sum(c["failed"] for c in counts), 1)
            self.assertEqual(sum(c["unexecuted"] for c in counts), 95)

    def test_provenance_write_failure_retains_observed_row(self):
        original_write = study.Budget.write
        def write(budget, path, value, **kwargs):
            if Path(path).parent.name == "provenance":
                raise OSError("injected provenance persistence failure")
            return original_write(budget, path, value, **kwargs)
        def invoke(root, output, plan, spec, case, budget, **kwargs):
            return {"row": dict(spec, status="succeeded", cleanup_confirmed=True, scalar_witness=[[1]],
                                actual_source_identity=spec["source_identity"]),
                    "projection": [], "provenance": {"test": True}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(study.Budget, "write", write), patch.object(study.runner, "invoke", invoke):
                result = study.run_study(Path(directory) / "new")
            self.assertFalse(result["study_admission"])
            self.assertEqual(len(result["unpersisted_rows"]), 1)
            self.assertEqual(sum(c["attempted"] for c in result["denominators"].values()), 1)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                study.run_study(directory)

    def test_resource_limits(self):
        for seconds in (float("nan"), -1, 601):
            with self.assertRaises(ValueError):
                study.run_study("unused", max_seconds=seconds)


class ImprovementTransportTests(unittest.TestCase):
    def test_real_admitted_companion_uses_named_profile(self):
        from bench import runner
        plan = study.make_plan()
        spec = next(r for r in plan["arms"] if (r["family"], r["K"], r["B"], r["role"], r["method"]) == (0, 1, 4, "companion", "C1A"))
        spec["source_identity"] = study.design.sha(study.source_inventory())
        plan["budget"] = {"attempt_seconds": 120, "transient_max_bytes": 4 * 1024**2}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("transient", "requests", "receipts"):
                (output / name).mkdir()
            budget = runner.Budget(output, 120, 16 * 1024**2)
            packet = runner.invoke(study.ROOT, output, plan, spec, plan["cases"][spec["case_id"]], budget, python=sys.executable)
            row = packet["row"]
            self.assertEqual(row["status"], "succeeded", row.get("error"))
            self.assertEqual(row["method"], "C1A")
            self.assertEqual(row["execution_profile"], "admitted-runtime-v1")
            self.assertEqual(row["actual_source_identity"], spec["source_identity"])
            self.assertEqual(len(packet["projection"]), 4)
            self.assertIsNone(row["workflow_wall_seconds"])
            self.assertTrue(row["cleanup_confirmed"])


if __name__ == "__main__":
    unittest.main()
