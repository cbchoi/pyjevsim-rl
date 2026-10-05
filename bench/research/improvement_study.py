"""Small paired exploration of strict and explicitly admitted execution contracts."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

from . import break_even_design as design, break_even_run as kernel
from bench.continuation_study.run import Budget, remove_owned_directory
from bench import runner, host

METHODS = ("R", "N", "C1", "C1A")
PROFILES = {"R": "native-replay", "N": "native-snapshot", "C1": "strict-v1", "C1A": "admitted-runtime-v1"}
ROOT = Path(__file__).resolve().parents[2]
load_runtime_modules = kernel.load_runtime_modules


def source_inventory():
    from run_research import source_inventory as base_inventory
    rows = base_inventory()
    rows["run_improvement.py"] = hashlib.sha256((ROOT / "run_improvement.py").read_bytes()).hexdigest()
    return rows


class CheckedBackend(kernel.Backend):
    expected_profile = "strict-v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_runtime_profiles = set()

    def _check_runtime(self, runtime):
        if self.spec["method"] == "C1":
            actual = runtime.execution_profile
            self.observed_runtime_profiles.add(actual)
            if actual != self.expected_profile:
                runtime.close()
                raise RuntimeError("actual runtime execution profile differs")
        return runtime

    def fresh(self, physical_id):
        return self._check_runtime(super().fresh(physical_id))

    def restore_from_file(self, directory, branch_index, physical_id):
        return self._check_runtime(super().restore_from_file(directory, branch_index, physical_id))


class AdmittedBackend(CheckedBackend):
    expected_profile = "admitted-runtime-v1"

    def __init__(self, spec, case, modules):
        super().__init__(spec, case, modules)
        self.coordinator = self.cc.ContinuationCoordinator(self.registry, execution_profile="admitted-runtime-v1")


def execute_arm(spec, case, campaign, budget, modules):
    label = spec["method"]
    if label not in METHODS or spec.get("execution_profile") != PROFILES[label]:
        raise ValueError("unknown method or mislabeled execution contract")
    actual = dict(spec, method="C1" if label == "C1A" else label)
    result, projection = kernel.execute_arm(actual, case, campaign, budget, modules,
        backend_factory=AdmittedBackend if label == "C1A" else CheckedBackend)
    result.update(spec)
    result["strict_step_admission"] = label == "C1"
    result["trusted_model_lifetime_contract"] = label == "C1A"
    result["runtime_execution_profile_checked"] = label in ("C1", "C1A") and result["status"] == "succeeded"
    return result, projection


def make_plan():
    protocol = design.load_protocol()
    protocol["cohorts"]["calibration"]["input_seed_start"] = 985000
    base = design.make_plan(protocol, "calibration")
    candidates = {row["cell_id"]: row for row in base["arms"] if row["family"] < 3
                  and row["K"] in (1, 4096) and row["S"] == 8 and row["B"] in (4, 16)}
    cases, arms = {}, []
    for cell_index, (cell_id, original) in enumerate(candidates.items()):
        cases[cell_id] = base["cases"][cell_id]
        for role_index, role in enumerate(("companion", "timing") if cell_index % 2 == 0 else ("timing", "companion")):
            start = (cell_index + role_index) % 4
            order = METHODS[start:] + METHODS[:start]
            if original["family"] % 2:
                order = tuple(reversed(order))
            for position, method in enumerate(order):
                arms.append(dict(original, arm_id=f"imp-{cell_id}-{role}-{method}",
                    method=method, role=role, purpose=role, execution_profile=PROFILES[method],
                    global_order=len(arms), method_order=list(order), method_position=position,
                    role_position=role_index, role_order=["companion", "timing"] if cell_index % 2 == 0 else ["timing", "companion"]))
    return {"schema": "continuation-improvement-plan-v1", "experiment_kind": "continuation-improvement-v1",
            "methods": list(METHODS), "profiles": PROFILES, "arms": arms, "cases": cases,
            "families": 3, "planned_cells": len(cases), "workers": 1, "blas_threads": 1,
            "confirmatory": False, "condition": "unspecified", "retry_or_replacement": False}


def compare_cell(packets):
    rows = [packet["row"] for packet in packets]
    keyset = {(row["role"], row["method"]) for row in rows}
    complete = len(rows) == 8 and keyset == set(itertools.product(("companion", "timing"), METHODS))
    success = complete and all(row["status"] == "succeeded" and row.get("cleanup_confirmed") is True for row in rows)
    companions = [packet for packet in packets if packet["row"]["role"] == "companion"]
    projections = success and len({design.encoded(p["projection"]) for p in companions}) == 1
    scalars = success and all(p.get("witness") is not None for p in packets) and len({design.encoded(p["witness"]) for p in packets}) == 1
    identity = success and all(row.get("actual_source_identity") == row.get("source_identity") for row in rows)
    first = rows[0]
    return {key: first[key] for key in ("cell_id", "family", "K", "S", "B")} | {
        "complete": complete, "exact": bool(success and projections and scalars and identity),
        "companion_exact": bool(projections), "normal_output_agreement": bool(scalars),
        "identity_agreement": bool(identity), "timing_full_trace_exact": False,
        "recorded_arms": len(rows), "planned_arms": 8}


def summarize(rows, cells):
    accepted = {cell["cell_id"] for cell in cells if cell["exact"]}
    timed = {row["cell_id"] + "/" + row["method"]: row for row in rows
             if row["role"] == "timing" and row["cell_id"] in accepted}
    result = []
    family_coordinates = {}
    for cell in cells:
        if cell["exact"]:
            family_coordinates.setdefault(cell["family"], set()).add((cell["K"], cell["B"]))
    complete_families = sorted(family for family, coordinates in family_coordinates.items()
                              if coordinates == set(itertools.product((1, 4096), (4, 16))))
    rng = random.Random(985901)
    draws = [[rng.randrange(3) for _ in range(3)] for _ in range(20000)]
    for K, B in itertools.product((1, 4096), (4, 16)):
        selected = sorted([cell for cell in cells if cell["exact"] and cell["K"] == K and cell["B"] == B], key=lambda x: x["family"])
        summary = {"K": K, "S": 8, "B": B, "families": len(selected),
                   "family_ids": [cell["family"] for cell in selected], "ratios": {}, "seconds": {}}
        for method in METHODS:
            values = [timed[cell["cell_id"] + "/" + method]["workflow_wall_seconds"] for cell in selected]
            summary["seconds"][method] = values
        for numerator, denominator in (("C1A", "C1"), ("C1A", "N"), ("C1A", "R"), ("C1", "R"), ("N", "R")):
            logs = [math.log(a / b) for a, b in zip(summary["seconds"][numerator], summary["seconds"][denominator])]
            bounds = None
            if len(logs) == 3 and len(complete_families) == 3:
                boot = sorted(math.exp(sum(logs[i] for i in draw) / 3) for draw in draws)
                bounds = [boot[499], boot[19499]]
            summary["ratios"][numerator + "/" + denominator] = {
                "paired_geometric_ratio": math.exp(sum(logs) / len(logs)) if logs else None,
                "exploratory_pointwise_95_interval": bounds}
        result.append(summary)
    return {"schema": "continuation-improvement-analysis-v1", "conditions": result,
        "bootstrap_draws": 20000, "resampling_unit": "whole family across all coordinates",
        "complete_family_ids": complete_families, "intervals_require_all_three_complete_families": True,
        "confirmatory": False, "small_sample_warning": "three families; intervals are exploratory, no general superiority",
        "comparability_warning": "C1A changes ordinary-step validation frequency under an explicit trusted-model contract",
        "same_guarantees_claim": False}


def run_study(output, *, max_seconds=600, max_bytes=16*1024**2):
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 600 or not 0 < max_bytes <= 16*1024**2:
        raise ValueError("exploration limits are 600 seconds and16MiB")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("transient", "requests", "provenance", "receipts"):
        (output / name).mkdir()
    started = time.perf_counter()
    budget = Budget(output, max_seconds, max_bytes)
    plan = make_plan()
    sources = source_inventory()
    identity = design.sha(sources)
    plan.update(source_identity=identity, source_sha256=sources,
                budget={"seconds": max_seconds, "max_bytes": max_bytes, "attempt_seconds": 120, "transient_max_bytes": 4*1024**2})
    for spec in plan["arms"]:
        spec["source_identity"] = identity
    plan["manifest_sha256"] = design.sha(plan)
    rows, cells, pending, provenance_keys = [], [], {}, set()
    unpersisted_rows, unpersisted_cells, write_errors = [], [], []
    stop_reason = None
    def write(name, value, *, final=False, append=False):
        runner.refresh(budget)
        budget.write(output / name, value, final=final, append=append)
    try:
        write("protocol.json", plan)
        write("environment.json", dict(host.metadata(output), condition="unspecified", workers=1, blas_threads=1,
            host_interference_controlled=False, memory_complete=False))
        for spec in plan["arms"]:
            if time.perf_counter() - started >= max_seconds - 10:
                raise TimeoutError("execution budget exhausted with analysis reserve")
            case = plan["cases"][spec["case_id"]]
            try:
                packet = runner.invoke(ROOT, output, plan, spec, case, budget, python=sys.executable)
            except BaseException as exc:
                packet = {"row": runner.failed_row(spec, case, f"invoke: {type(exc).__name__}: {exc}"),
                          "projection": [], "provenance": None}
                packet["row"]["execution_observation_incomplete"] = True
            row = packet["row"]
            packet["witness"] = row.pop("scalar_witness", None)
            rows.append(row)
            pending.setdefault(spec["cell_id"], []).append(packet)
            try:
                if packet.get("provenance") is not None:
                    key = design.sha(packet["provenance"])
                    provenance_keys.add(key)
                    row["source_metadata_key"] = key
                    if not (output / "provenance" / (key + ".json")).exists():
                        write("provenance/" + key + ".json", packet["provenance"])
                write("arms.jsonl", row, append=True, final=row["status"] != "succeeded")
            except BaseException as exc:
                unpersisted_rows.append(row)
                write_errors.append(str(exc))
                raise
            if len(pending[spec["cell_id"]]) == 8:
                cell = compare_cell(pending.pop(spec["cell_id"]))
                cells.append(cell)
                try:
                    write("cells.jsonl", cell, append=True)
                except BaseException as exc:
                    unpersisted_cells.append(cell)
                    write_errors.append(str(exc))
                    raise
                if not cell["exact"]:
                    raise RuntimeError("cell comparison failed")
            print(f"improvement {len(rows)}/{len(plan['arms'])} {spec['arm_id']}: {row['status']}", flush=True)
            if row["status"] != "succeeded":
                raise RuntimeError(row.get("error") or "arm failed")
            if len(provenance_keys) > 1:
                raise RuntimeError("loaded source identity changed")
    except BaseException as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
    for packets in pending.values():
        cell = compare_cell(packets)
        cells.append(cell)
        try:
            write("cells.jsonl", cell, final=True, append=True)
        except BaseException as exc:
            unpersisted_cells.append(cell)
            write_errors.append(str(exc))
            stop_reason = stop_reason or "partial cell write failed"
    cleanup = []
    try:
        remaining = list((output / "transient").iterdir())
    except OSError as exc:
        remaining = []
        cleanup.append({"removed": False, "error": str(exc)})
        stop_reason = stop_reason or "owned transient observation failed"
    for path in remaining:
        try:
            remove_owned_directory(path, output / "transient")
            cleanup.append({"name": path.name, "removed": True})
        except BaseException as exc:
            cleanup.append({"name": path.name, "removed": False, "error": str(exc)})
            stop_reason = stop_reason or "owned cleanup failed"
    admission = (stop_reason is None and len(rows) == 96 and len(cells) == 12
                 and all(c["exact"] for c in cells) and len(provenance_keys) == 1)
    summary = {"schema": "continuation-improvement-execution-v1", "status": "completed" if admission else "failed",
        "study_admission": admission, "stop_reason": stop_reason, "denominators": runner.denominators(plan, rows),
        "planned_cells": 12, "exact_cells": sum(c["exact"] for c in cells), "cleanup": cleanup,
        "elapsed_seconds": time.perf_counter() - started, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity": identity, "source_metadata_keys": sorted(provenance_keys),
        "unpersisted_rows": unpersisted_rows, "unpersisted_cells": unpersisted_cells, "write_errors": write_errors,
        "method_positions": dict(Counter(f"{r['method']}:{r['method_position']}:{r['role']}" for r in rows)),
        "memory_complete": False, "host_interference_controlled": False, "confirmatory": False,
        "retry_or_replacement": False, "same_guarantees_claim": False}
    # Preserve attempted denominators before optional statistical reporting.
    try:
        write("collection.json", summary, final=True)
    except BaseException as exc:
        write_errors.append(f"collection: {exc}")
        summary.update(status="failed", study_admission=False, stop_reason=stop_reason or "collection write failed")
    try:
        analysis = summarize(rows, cells)
        write("analysis.json", analysis, final=True)
        summary["analysis_status"] = "completed"
    except BaseException as exc:
        summary.update(status="failed", study_admission=False, analysis_status="failed",
                       analysis_error=f"{type(exc).__name__}: {exc}")
    summary["elapsed_seconds"] = time.perf_counter() - started
    if summary["elapsed_seconds"] > max_seconds:
        summary.update(status="failed", study_admission=False, stop_reason=stop_reason or "analysis exceeded total time")
    try:
        write("execution.json", summary, final=True)
    except BaseException as exc:
        summary.update(status="failed", study_admission=False, execution_write_error=str(exc))
        print(json.dumps({"terminal_execution": summary}, ensure_ascii=False), file=sys.stderr, flush=True)
    return summary
