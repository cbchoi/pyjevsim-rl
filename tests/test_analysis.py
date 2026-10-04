"""Synthetic receipt and optional host-counter checks; no model execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench import host
from bench.analyze import analyze, analyze_records, _bytes
from bench.design import make_manifest


def fixtures():
    manifest = make_manifest({"condition": "idle"})
    rows = []
    for spec in manifest["arms"]:
        case = manifest["cases"][spec["case_id"]]
        rows.append({**spec, "status": "succeeded", "seed": case["seed"],
            "config_sha256": case["config_sha256"], "application_wall_seconds": 2.0 if spec["method"] == "C1" else 1.0,
            "application_cpu_seconds": .75 if spec["method"] == "C1" else .5,
            "cpu_scope_wall_seconds": 2.1 if spec["method"] == "C1" else 1.1,
            "cleanup_confirmed": True, "cleanup_errors": [], "worker_exit_code": 0,
            "prefixes_executed": 1, "suffix_steps_executed": 64, "expected_suffix_steps": 64,
            "work_counts": None, "branch_projection_sha256": ["a" * 64] * 8})
    cells = comparisons(rows)
    return manifest, rows, cells


def comparisons(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["cell_id"], []).append(row)
    cells = []
    for key, group in groups.items():
        complete = len(group) == 2 and all(row["status"] == "succeeded" for row in group)
        vectors = {row["method"]: row.get("branch_projection_sha256", []) for row in group}
        whole = {method: hashlib.sha256(_bytes(value)).hexdigest() for method, value in vectors.items()}
        cells.append({"cell_id": key, "purpose": "timing", "complete": complete,
            "exact": complete and len(set(whole.values())) == 1, "methods": ["N", "C1"],
            "arm_ids": [row["arm_id"] for row in group], "projection_sha256_by_method": vectors,
            "whole_projection_sha256_by_method": whole})
    return cells


class AnalysisTests(unittest.TestCase):
    def test_known_ratios_order_and_missing_host_claim(self):
        result = analyze_records(*fixtures(), bootstrap_replicates=200)
        self.assertTrue(result["study_admission"])
        self.assertEqual(result["denominators"]["succeeded"], 24)
        self.assertEqual(result["exact_cells"]["exact"], 12)
        self.assertFalse(result["condition_is_clean_host_proof"])
        for model in result["models"].values():
            wall, cpu = model["metrics"]["wall"], model["metrics"]["cpu"]
            self.assertAlmostEqual(wall["geometric_ratio_C1_over_N"], 2.0)
            self.assertEqual(wall["ci95"], [2.0, 2.0])
            self.assertAlmostEqual(cpu["geometric_ratio_C1_over_N"], 1.5)
            self.assertEqual(wall["first3_last3"]["last3"]["n"], 3)
            self.assertEqual(wall["relative_order"]["C1-before-N"]["n"], 3)
            self.assertEqual(model["pairs"][0]["absolute_seconds"]["C1"]["worker_cpu_matched_wall"], 2.1)

    def test_partial_failed_and_missing_are_not_zero_or_promoted(self):
        manifest, rows, _ = fixtures()
        rows = rows[:3]
        rows[-1].update(status="failed", application_wall_seconds=None, application_cpu_seconds=None,
                        cleanup_confirmed=None, branch_projection_sha256=[])
        result = analyze_records(manifest, rows, comparisons(rows), bootstrap_replicates=50)
        self.assertEqual(result["denominators"], {"planned": 24, "attempted": 3, "succeeded": 2, "failed": 1, "unexecuted": 21})
        self.assertEqual(result["exact_cells"]["exact"], 1)
        self.assertFalse(result["study_admission"])

    def test_unavailable_cpu_preserves_wall_but_not_cpu_estimate(self):
        manifest, rows, cells = fixtures()
        for row in rows:
            row["application_cpu_seconds"] = None if row["method"] == "N" else 0.0
        result = analyze_records(manifest, rows, cells, bootstrap_replicates=50)
        for model in result["models"].values():
            self.assertEqual(model["metrics"]["wall"]["paired_n"], 6)
            self.assertEqual(model["metrics"]["cpu"]["paired_n"], 0)
            self.assertIsNone(model["metrics"]["cpu"]["geometric_ratio_C1_over_N"])
            self.assertIsNone(model["metrics"]["cpu"]["ci95"])

    def test_duplicate_unplanned_wrong_seed_and_short_workload_reject(self):
        for mutation in ("duplicate", "unplanned", "seed", "short"):
            manifest, rows, cells = fixtures()
            if mutation == "duplicate":
                rows.append(deepcopy(rows[0]))
            elif mutation == "unplanned":
                rows[0]["arm_id"] = "outside"
            elif mutation == "seed":
                rows[0]["seed"] += 1
            else:
                rows[0]["suffix_steps_executed"] = 63
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze_records(manifest, rows, cells, bootstrap_replicates=10)

    def test_mismatch_is_not_exact_and_forged_exact_rejects(self):
        manifest, rows, _ = fixtures()
        rows[0]["branch_projection_sha256"][0] = "b" * 64
        cells = comparisons(rows)
        result = analyze_records(manifest, rows, cells, bootstrap_replicates=50)
        self.assertFalse(result["study_admission"])
        self.assertEqual(result["exact_cells"]["mismatched_complete"], 1)
        cells[0]["exact"] = True
        with self.assertRaises(ValueError):
            analyze_records(manifest, rows, cells, bootstrap_replicates=10)

    def test_fallback_and_final_completion_status(self):
        manifest, rows, cells = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_bytes(_bytes(manifest))
            (root / "arms.jsonl").write_bytes(b"".join(_bytes(row) + b"\n" for row in rows[:-1]) + b'{"truncated":')
            (root / "cells.jsonl").write_bytes(b"".join(_bytes(cell) + b"\n" for cell in cells[:-1]))
            execution = {"campaign_wall_seconds": 1.0, "study_admission": False,
                "unpersisted_arms": [rows[-1]], "unpersisted_cells": [cells[-1]], "write_errors": ["synthetic write failure"]}
            (root / "execution.json").write_bytes(_bytes(execution))
            result = analyze(root, bootstrap_replicates=50, max_seconds=30)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["denominators"]["succeeded"], 24)
            self.assertFalse(result["study_admission"])
            self.assertEqual(json.loads((root / "analysis-status.json").read_text())["status"], "completed")
            self.assertIn("## English", (root / "findings.md").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                analyze(root, bootstrap_replicates=10)

    def test_expired_budget_has_failed_receipt_not_success(self):
        manifest, _, _ = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_bytes(_bytes(manifest))
            (root / "execution.json").write_bytes(_bytes({"campaign_wall_seconds": 1.0}))
            with self.assertRaises(TimeoutError):
                analyze(root, max_seconds=0)
            self.assertEqual(json.loads((root / "analysis-status.json").read_text())["status"], "failed")


class HostTests(unittest.TestCase):
    def test_optional_read_failure_stays_unavailable(self):
        with patch.object(host, "_system_cpu", side_effect=OSError("unavailable")), patch.object(host, "_disk_io", return_value=None):
            observed = host.snapshot()
        self.assertIsNone(observed["system_cpu"])
        self.assertIsNone(observed["disk_io"])
        self.assertFalse(observed["coverage"]["external_process_attribution"])

    def test_aggregate_delta_is_not_external_process_attribution(self):
        before = {"monotonic_seconds": 1, "observer_pid": 10, "observer_process_cpu_seconds": 2,
            "system_cpu": {"source": "synthetic", "total_seconds": 100, "idle_seconds": 50},
            "disk_io": {"disk": {"read_bytes": 100, "write_bytes": 200}}}
        after = {"monotonic_seconds": 3, "observer_pid": 10, "observer_process_cpu_seconds": 2.1,
            "system_cpu": {"source": "synthetic", "total_seconds": 110, "idle_seconds": 53},
            "disk_io": {"disk": {"read_bytes": 110, "write_bytes": 250}}}
        value = host.difference(before, after)
        self.assertAlmostEqual(value["aggregate_cpu_busy_fraction"], .7)
        self.assertEqual(value["disk_io_by_device"]["disk"], {"read_bytes": 10, "write_bytes": 50})
        self.assertFalse(value["external_interference_identified"])
        after["system_cpu"]["total_seconds"] = 99
        after["observer_pid"] = 11
        after["disk_io"]["disk"]["read_bytes"] = 99
        value = host.difference(before, after)
        self.assertIsNone(value["aggregate_cpu_busy_fraction"])
        self.assertIsNone(value["observer_process_cpu_seconds"])
        self.assertIsNone(value["disk_io_by_device"]["disk"])


if __name__ == "__main__":
    unittest.main()
