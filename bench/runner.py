"""Sequential portable N/C1 study. No retries, smoke tests or staging gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .continuation_study.cases import encoded, sha
from .continuation_study.run import Budget, remove_owned_directory
from . import host


def storage_bytes(directory):
    return sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())


def refresh(budget):
    budget.written = storage_bytes(budget.root)
    budget.transient_bytes = 0
    budget.peak_observed_bytes = max(budget.peak_observed_bytes, budget.written)


def denominators(manifest, rows):
    ids = {row["arm_id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("duplicate attempts are not permitted")
    planned = len(manifest["arms"])
    return {"timing": {"planned": planned, "attempted": len(rows),
        "succeeded": sum(row["status"] == "succeeded" for row in rows),
        "failed": sum(row["status"] != "succeeded" for row in rows),
        "unexecuted": planned - len(rows)}}


def failed_row(spec, case, message):
    return {**spec, "status": "failed", "error": message, "error_phase": "owned-process",
        "config_sha256": case["config_sha256"], "seed": case["seed"],
        "application_wall_seconds": None, "application_cpu_seconds": None,
        "cleanup_confirmed": None, "cleanup_errors": [],
        "memory_complete": False, "whole_lifetime_peak_bytes": None}


def invoke(root, campaign, manifest, spec, case, budget, *, python):
    refresh(budget)
    budget.check()
    remaining = budget.seconds - (time.perf_counter() - budget.started)
    request_path = campaign / "requests" / f"{spec['arm_id']}.json"
    request = {"spec": spec, "case": case, "campaign": str(campaign),
        "remaining_seconds": min(manifest["budget"]["attempt_seconds"], remaining),
        "max_bytes": budget.max_bytes, "retained_bytes": budget.written + 65536,
        "max_transport_bytes": min(4 * 1024**2, budget.max_bytes)}
    if len(encoded(request)) + 1 > 65536:
        raise ValueError("request exceeds its preaccounted allowance")
    budget.write(request_path, request)
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    before = host.snapshot()
    started = time.perf_counter()
    process = subprocess.Popen([str(python), "-I", "-B", str(root / "bench" / "worker.py"),
        "--request", str(request_path)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, cwd=str(campaign), env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    deadline = min(started + manifest["budget"]["attempt_seconds"], budget.started + budget.seconds)
    termination = None
    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                termination = "attempt or campaign deadline reached"
                break
            try:
                output, stderr = process.communicate(timeout=min(1.0, remaining))
                break
            except subprocess.TimeoutExpired:
                observed = storage_bytes(campaign)
                budget.peak_observed_bytes = max(budget.peak_observed_bytes, observed)
                if observed > budget.max_bytes - budget.reserve:
                    termination = "observed storage budget reached"
                    break
        if termination:
            process.terminate()
            try:
                output, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                output, stderr = process.communicate(timeout=2)
        elapsed = time.perf_counter() - started
        after = host.snapshot()
        if termination or len(output) > request["max_transport_bytes"]:
            packet = {"row": failed_row(spec, case, termination or "worker transport exceeded cap"),
                      "projection": [], "provenance": None}
        else:
            try:
                packet = json.loads(output)
                row = packet["row"]
                if any(row.get(key) != value for key, value in spec.items()):
                    raise ValueError("worker identity differs from assigned arm")
                if process.returncode != 0 and row["status"] == "succeeded":
                    raise ValueError("successful receipt from failed process")
                if row["status"] == "succeeded":
                    projection = packet["projection"]
                    if (type(projection) is not list or len(projection) != spec["branch_count"]
                            or any(type(branch) is not list or len(branch) != spec["suffix_steps"] for branch in projection)
                            or [sha(branch) for branch in projection] != row.get("branch_projection_sha256")):
                        raise ValueError("successful projection differs from workload/digest receipt")
            except BaseException as exc:
                packet = {"row": failed_row(spec, case, f"result transport: {type(exc).__name__}: {exc}"),
                          "projection": [], "provenance": None}
        packet["row"].update(process_wall_seconds=elapsed, worker_exit_code=process.returncode,
            owned_process_terminated=termination is not None,
            worker_stderr=stderr.decode("utf-8", errors="replace")[:8192],
            host_observation={"before": before, "after": after, "delta": host.difference(before, after)})
        return packet
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        request_path.unlink(missing_ok=True)


def run_campaign(root, campaign, manifest, *, python=None, stage_started=None):
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    campaign.parent.mkdir(parents=True, exist_ok=True)
    campaign.mkdir(exist_ok=False)
    for name in ("transient", "receipts", "requests", "provenance"):
        (campaign / name).mkdir()
    budget = Budget(campaign, manifest["budget"]["seconds"], manifest["budget"]["max_bytes"])
    if stage_started is not None:
        budget.started = stage_started
    rows, cells, pending, unpersisted, unpersisted_cells, cleanup_records = [], [], {}, [], [], []
    stop_reason, write_errors = None, []
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        budget.write(campaign / "manifest.json", manifest)
        budget.write(campaign / "environment.json", {**host.metadata(campaign),
            "python": sys.version, "workers": 1, "blas_environment_threads": 1,
            "condition": manifest.get("condition", "unspecified"), "condition_is_user_label": True,
            "setup_in_timing_budget": False, "worker_startup_import_in_campaign_wall": True})
        for spec in manifest["arms"]:
            refresh(budget)
            budget.check()
            case = manifest["cases"][spec["case_id"]]
            try:
                packet = invoke(root, campaign, manifest, spec, case, budget, python=python or sys.executable)
            except BaseException as exc:
                packet = {"row": failed_row(spec, case, f"launch: {type(exc).__name__}: {exc}"),
                          "projection": [], "provenance": None}
            row = packet["row"]
            rows.append(row)
            identity = packet.get("provenance")
            refresh(budget)
            try:
                if identity is not None:
                    key = sha(identity)
                    row["source_metadata_key"] = key
                    path = campaign / "provenance" / f"{key}.json"
                    if not path.exists():
                        budget.write(path, identity, final=row["status"] != "succeeded")
                budget.write(campaign / "arms.jsonl", row, append=True, final=row["status"] != "succeeded")
            except BaseException as exc:
                unpersisted.append(row)
                raise RuntimeError(f"arm receipt write failed: {exc}") from exc
            cell = pending.setdefault(spec["cell_id"], {})
            cell[spec["method"]] = packet["projection"]
            expected = [arm for arm in manifest["arms"] if arm["cell_id"] == spec["cell_id"]]
            if len(cell) == len(expected):
                complete = all(item["status"] == "succeeded" for item in rows if item["cell_id"] == spec["cell_id"])
                exact = complete and len({encoded(value) for value in cell.values()}) == 1
                summary = {"cell_id": spec["cell_id"], "purpose": "timing", "complete": complete,
                    "exact": exact, "methods": manifest["methods"],
                    "arm_ids": [arm["arm_id"] for arm in expected],
                    "projection_sha256_by_method": {method: [sha(branch) for branch in value] for method, value in cell.items()},
                    "whole_projection_sha256_by_method": {method: sha(value) for method, value in cell.items()}}
                cells.append(summary)
                del pending[spec["cell_id"]]
                try:
                    budget.write(campaign / "cells.jsonl", summary, append=True, final=not exact)
                except BaseException:
                    unpersisted_cells.append(summary)
                    raise
                if not exact:
                    stop_reason = f"physical projection mismatch or failed arm: {spec['cell_id']}"
            if row["status"] != "succeeded":
                stop_reason = row.get("error") or f"failed arm: {spec['arm_id']}"
            print(f"{len(rows)}/{len(manifest['arms'])} {spec['arm_id']}: {row['status']}", flush=True)
            if stop_reason:
                break
    except BaseException as exc:
        stop_reason = f"{type(exc).__name__}: {exc}"
    finally:
        # Only our newly allocated campaign's immediate arm directories.
        try:
            remaining_directories = list((campaign / "transient").iterdir())
        except OSError as exc:
            remaining_directories = []
            cleanup_records.append({"removed": False, "error": str(exc)})
            stop_reason = stop_reason or "parent transient directory observation failed"
        for directory in remaining_directories:
            try:
                remove_owned_directory(directory, campaign / "transient")
                cleanup_records.append({"arm_id": directory.name, "removed": True})
            except BaseException as exc:
                cleanup_records.append({"arm_id": directory.name, "removed": False, "error": str(exc)})
                stop_reason = stop_reason or "parent transient cleanup failed"
    for cell_id, values in pending.items():
        summary = {"cell_id": cell_id, "purpose": "timing", "complete": False, "exact": False,
            "methods": manifest["methods"], "arm_ids": [row["arm_id"] for row in rows if row["cell_id"] == cell_id],
            "projection_sha256_by_method": {key: [sha(branch) for branch in value] for key, value in values.items()},
            "whole_projection_sha256_by_method": {key: sha(value) for key, value in values.items()}}
        cells.append(summary)
        try:
            budget.write(campaign / "cells.jsonl", summary, append=True, final=True)
        except BaseException as exc:
            write_errors.append(str(exc))
            unpersisted_cells.append(summary)
    try:
        refresh(budget)
        storage_complete = True
    except OSError as exc:
        storage_complete = False
        write_errors.append(str(exc))
    counts = denominators(manifest, rows)
    planned_cells = len({arm["cell_id"] for arm in manifest["arms"]})
    completed_cells = sum(item["complete"] for item in cells)
    exact_cells = sum(item["exact"] for item in cells)
    admission = (stop_reason is None and not write_errors and not unpersisted and not unpersisted_cells
        and counts["timing"]["succeeded"] == len(manifest["arms"]) and exact_cells == planned_cells
        and all(row.get("cleanup_confirmed") is True for row in rows))
    execution = {"status": "completed" if admission else "failed", "study_admission": admission,
        "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "stop_reason": stop_reason, "denominators": counts,
        "exact_cells": {"timing": {"planned": planned_cells, "complete": completed_cells, "exact": exact_cells}},
        "campaign_wall_seconds": time.perf_counter() - budget.started,
        "peak_storage_observed_at_boundaries_bytes": budget.peak_observed_bytes,
        "retained_bytes_before_summary": budget.written, "final_storage_observation_complete": storage_complete,
        "write_errors": write_errors, "unpersisted_arms": unpersisted,
        "unpersisted_cells": unpersisted_cells,
        "parent_cleanup_records": cleanup_records,
        "memory_complete": False, "whole_lifetime_peak_bytes": None,
        "host_interference_controlled": False,
        "budget_enforcement": "owned direct-child wall deadline; sampled/boundary storage, unsampled transient peak unknown",
        "full_success_projections_retained": False, "retry_or_replacement": False}
    try:
        budget.write(campaign / "execution.json", execution, final=True)
    except BaseException as exc:
        execution["execution_completed_before_write_error"] = admission
        execution.update(status="failed", study_admission=False,
                         execution_write_error=f"{type(exc).__name__}: {exc}")
        # Preserve the small terminal accounting even if the destination cannot
        # accept a final file. No retry and no claim that disk evidence is whole.
        print(encoded({"terminal_execution": execution}).decode("utf-8"), file=sys.stderr, flush=True)
    return execution


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["idle-primary"], default="idle-primary")
    parser.add_argument("--condition", choices=["idle", "busy", "unspecified"], default="unspecified")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument("--max-mib", type=int, default=16)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args(argv)
    from .design import make_manifest
    root = Path(__file__).resolve().parents[1]
    manifest = make_manifest({"preset": args.preset, "condition": args.condition,
        "budget_seconds": args.budget_seconds, "max_bytes": args.max_mib * 1024**2,
        "seed_offset": args.seed_offset})
    campaign = args.output or root / "results" / ("idle-primary-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:
        execution = run_campaign(root, campaign, manifest, stage_started=started)
        if execution.get("execution_write_error"):
            return 2
        from .analyze import analyze
        remaining = manifest["budget"]["seconds"] - (time.perf_counter() - started)
        analysis = analyze(campaign.resolve(), max_seconds=max(0.0, remaining), max_bytes=manifest["budget"]["max_bytes"])
        if time.perf_counter() - started > manifest["budget"]["seconds"]:
            raise TimeoutError("whole workflow exceeded the declared time budget")
        if storage_bytes(campaign) > manifest["budget"]["max_bytes"]:
            raise RuntimeError("whole workflow exceeded the declared storage budget")
        print(f"Results: {campaign.resolve()}", flush=True)
        return 0 if (execution["study_admission"] and analysis.get("study_admission") is True
                     and analysis.get("status") == "completed") else 2
    except BaseException as exc:
        print(f"Study stopped: {type(exc).__name__}: {exc}; results: {campaign.resolve()}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
