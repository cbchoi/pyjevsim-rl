"""Run focused continuation research using the prepared local environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "vendor"), str(ROOT / "bench")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("performance", "decision"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=None)
    parser.add_argument("--decision-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    if args.stage == "performance":
        from bench.research.improvement_study import run_study
        result = run_study(args.output, max_seconds=600 if args.budget_seconds is None else args.budget_seconds)
    else:
        from bench.research.decision_utility import run_decision_study
        from bench.research.improvement_study import source_inventory
        from bench.continuation_study.run import Budget
        from bench.worker import ShortLog
        from bench import host
        import contextlib
        cap = 120 if args.budget_seconds is None else args.budget_seconds
        if not 0 < cap <= 120:
            raise ValueError("decision exploration cap must be in (0,120] seconds")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        budget = Budget(output, cap, 8 * 1024**2)
        sources = source_inventory()
        budget.write(output / "protocol.json", {"stage": "decision", "families": 3, "candidates": 8,
            "evaluation_rollouts": 8, "base_seed": 984000, "methods": ["R", "N", "C1", "C1A"],
            "max_seconds": cap, "max_bytes": budget.max_bytes, "decision_seconds": args.decision_seconds,
            "source_sha256": sources, "environment": host.metadata(output), "condition": "unspecified",
            "host_interference_controlled": False, "memory_complete": False})
        log = ShortLog()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = run_decision_study(output, max_seconds=max(.001, cap - (time.perf_counter() - budget.started)),
                                        decision_budget_seconds=args.decision_seconds,
                                        max_storage_bytes=budget.max_bytes-budget.written)
        result["source_consistent"] = sources == source_inventory()
        result["total_cli_wall_seconds"] = time.perf_counter() - budget.started
        result["bounded_log"] = log.text
        if not result["source_consistent"] or result["total_cli_wall_seconds"] > cap:
            result.update(status="failed", study_admission=False, execution_error="source changed or total time exceeded")
        try:
            budget.write(output / "decision.json", result, final=True)
        except BaseException as exc:
            result.update(status="failed", study_admission=False, persistence_error=str(exc))
            print(json.dumps({"terminal_execution": result}, ensure_ascii=False), file=sys.stderr)
    print(json.dumps({key: result.get(key) for key in ("status", "study_admission", "denominators", "stop_reason", "error")}, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("study_admission", result.get("status") == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
