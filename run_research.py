"""Run semantics, new-model transfer and R/N/C1 cost studies sequentially."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_experiment import ensure_environment, _already_isolated, _seconds, _mib


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("all", "semantics", "transfer", "cost"), default="all")
    result.add_argument("--output", type=Path)
    result.add_argument("--budget-seconds", type=_seconds, default=600.0)
    result.add_argument("--max-mib", type=_mib, default=16)
    result.add_argument("--condition", choices=("idle", "busy", "unspecified"), default="unspecified")
    return result


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8") + b"\n"


def storage_bytes(directory):
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def write_new(root, name, value, cap):
    data = encoded(value)
    if storage_bytes(root) + len(data) > cap:
        raise RuntimeError("research output storage budget reached")
    with (root / name).open("xb") as stream:
        stream.write(data)


def source_inventory():
    paths = [ROOT / "run_research.py", ROOT / "bench" / "worker.py", ROOT / "bench" / "runner.py"]
    for name in ("environment.py", "executor.py", "adapters.py"):
        paths.append(ROOT / "src/pyjevsim_bridge/rl" / name)
    for directory in ("bench/research", "bench/continuation_study", "src/pyjevsim_bridge/rl/continuation",
                      "src/pyjevsim_bridge/rl/models", "vendor/pyjevsim"):
        paths.extend((ROOT / directory).rglob("*.py"))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def run(options):
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor"), str(ROOT / "bench")]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    started = time.perf_counter()
    deadline = started + options.budget_seconds
    cap = options.max_mib * 1024**2
    output = (options.output or ROOT / "results" / (
        "research-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "transient").mkdir()
    stages = ("semantics", "transfer", "cost") if options.stage == "all" else (options.stage,)
    summary = {"schema": "continuation-research-workflow-v1", "status": "running",
        "stages_planned": list(stages), "stages": {}, "condition": options.condition,
        "host_interference_controlled": False, "memory_complete": False,
        "human_productivity_measured": False, "stage_retries": 0,
        "budget_seconds": options.budget_seconds, "max_bytes": cap,
        "started_utc": datetime.now(timezone.utc).isoformat()}
    from bench.worker import ShortLog
    try:
        from bench import host
        write_new(output, "protocol.json", {**summary, "source_sha256": source_inventory(),
            "design_sha256": hashlib.sha256((ROOT / "docs/RESEARCH_DESIGN.md").read_bytes()).hexdigest(),
            "environment": host.metadata(output), "python": sys.version}, cap)
        for stage in stages:
            if time.perf_counter() >= deadline:
                raise TimeoutError("research workflow deadline reached")
            print(f"Starting {stage}; remaining {deadline-time.perf_counter():.1f}s", flush=True)
            summary["stages"][stage] = {"status": "running", "study_admission": False}
            if stage == "cost":
                from bench.research.cost import make_manifest
                from bench.research.cost_analysis import analyze
                from bench.runner import run_campaign
                remaining = deadline - time.perf_counter()
                if remaining <= 30:
                    raise TimeoutError("insufficient remaining time after fixed analysis reserve")
                remaining_bytes = cap - storage_bytes(output) - 65536
                manifest = make_manifest({"condition": options.condition,
                    "budget_seconds": remaining - 30, "max_bytes": remaining_bytes})
                campaign = output / "cost"
                execution = run_campaign(ROOT, campaign, manifest)
                summary["stages"][stage].update(
                    denominators=execution["denominators"], exact_cells=execution["exact_cells"],
                    execution_status=execution["status"], results_directory="cost",
                    status="execution-finished-analysis-pending")
                result = analyze(campaign, max_seconds=max(0, deadline-time.perf_counter()),
                                 max_bytes=remaining_bytes)
                summary["stages"][stage] = {"status": result.get("status"),
                    "study_admission": result.get("study_admission", False),
                    "denominators": execution["denominators"], "exact_cells": execution["exact_cells"],
                    "execution_status": execution["status"], "results_directory": "cost"}
                if not execution["study_admission"] or not result.get("study_admission"):
                    raise RuntimeError("cost study incomplete or failed; no retry")
            else:
                module = importlib.import_module("bench.research." + stage)
                function = module.run_study if stage == "semantics" else module.run_transfer
                log = ShortLog()
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = function(deadline=deadline, scratch_root=output / "transient",
                                      max_bytes=cap-storage_bytes(output)-65536)
                write_new(output, stage + ".json", result, cap)
                if log.text:
                    write_new(output, stage + "-log.json", {"text": log.text,
                        "discarded_characters": log.discarded_characters}, cap)
                summary["stages"][stage] = {"status": result.get("status"),
                    "study_admission": result.get("study_admission", False), "result_file": stage + ".json"}
                if not result.get("study_admission"):
                    raise RuntimeError(stage + " research reports failure; no retry")
            print(f"Finished {stage}", flush=True)
        if time.perf_counter() > deadline:
            raise TimeoutError("research execution and analysis exceeded declared budget")
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        if "stage" in locals() and stage in summary["stages"]:
            summary["stages"][stage].update(status="failed", study_admission=False,
                error=f"{type(error).__name__}: {error}")
    summary.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=time.perf_counter()-started,
                   unexecuted_stages=[stage for stage in stages if stage not in summary["stages"]])
    try:
        write_new(output, "workflow.json", summary, cap)
    except BaseException as error:
        summary.update(status="failed", write_error=f"{type(error).__name__}: {error}")
        print(encoded(summary).decode("utf-8"), file=sys.stderr)
    print(f"Research {summary['status']}: {output}", flush=True)
    if "error" in summary:
        print(summary["error"], file=sys.stderr)
    return 0 if summary["status"] == "completed" else 2


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    options = parser().parse_args(arguments)
    try:
        python = ensure_environment(ROOT)
        if not _already_isolated(ROOT):
            return subprocess.run([str(python), "-I", "-B", str(ROOT / "run_research.py"),
                                   *arguments], check=False).returncode
        return run(options)
    except (OSError, RuntimeError, ImportError) as error:
        print(f"Research stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
