"""Separate diagnostic, not an uninstrumented performance measurement."""
from __future__ import annotations

import argparse
import contextlib
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "vendor"), str(ROOT / "bench")]
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execution-profile", choices=("strict-v1", "admitted-runtime-v1"), default="strict-v1")
    args = parser.parse_args(argv)
    from bench.research import break_even_design as design, break_even_run as run
    from bench.continuation_study.run import Budget
    from bench.worker import ShortLog
    from run_research import source_inventory
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "transient").mkdir()
    (output / "receipts").mkdir()
    plan = design.make_plan(design.load_protocol(), "calibration")
    arm = next(row for row in plan["arms"] if row["family"] == 0 and row["K"] == 1
               and row["S"] == 8 and row["B"] == 4 and row["method"] == "C1" and row["role"] == "timing")
    case = plan["cases"][arm["case_id"]]
    sources = source_inventory()
    sources["tools/profile_continuation.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    arm = dict(arm, source_identity=design.sha(sources))
    modules = run.load_runtime_modules()
    class SelectedBackend(run.Backend):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if args.execution_profile != "strict-v1":
                self.coordinator = self.cc.ContinuationCoordinator(
                    self.registry, execution_profile=args.execution_profile)
    profiler, budget, log = cProfile.Profile(), Budget(output, 60, 4 * 1024**2), ShortLog()
    started = time.perf_counter()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        profiler.enable()
        try:
            row, _ = run.execute_arm(arm, case, output, budget, modules, backend_factory=SelectedBackend)
        finally:
            profiler.disable()
    stats = pstats.Stats(profiler, stream=io.StringIO())
    functions = []
    for (filename, lineno, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        path = Path(filename)
        location = path.relative_to(ROOT).as_posix() if path.is_absolute() and path.is_relative_to(ROOT) else filename
        functions.append(dict(path=location, line=lineno, function=name, primitive_calls=primitive,
                              calls=calls, self_seconds=own, cumulative_seconds=cumulative))
    report = {"schema": "continuation-profile-v1", "status": row["status"], "error": row["error"],
        "execution_profile": args.execution_profile, "source_sha256": sources,
        "diagnostic_only": True, "uninstrumented_performance": False,
        "condition": "unspecified", "host_interference_controlled": False,
        "elapsed_seconds": time.perf_counter() - started, "instrumented_workflow_seconds": row["workflow_wall_seconds"],
        "functions": sorted(functions, key=lambda x: x["cumulative_seconds"], reverse=True),
        "cleanup_confirmed": row["cleanup_confirmed"], "output_log": log.text}
    budget.write(output / "profile.json", report, final=True)
    print(json.dumps({key: report[key] for key in ("status", "execution_profile", "instrumented_workflow_seconds")}, ensure_ascii=False))
    for record in report["functions"][:16]:
        print(json.dumps(record, ensure_ascii=False))
    return 0 if row["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
