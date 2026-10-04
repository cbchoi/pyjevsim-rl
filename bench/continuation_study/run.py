"""Small sequential research runner; no smoke/freeze/staging workflow.

Invoke only after a new explicit time/storage budget is approved. Every arm is
attempted at most once. Application wall ends after an untimed-content receipt
write and cleanup; the row containing that measured time is written afterward
and belongs to campaign wall, avoiding a self-referential timing endpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "pysdk"))
    from continuation_study.cases import make_manifest, action_plan, physical_projection, encoded, sha
else:
    from .cases import make_manifest, action_plan, physical_projection, encoded, sha


class BudgetExceeded(RuntimeError):
    pass


class StudyFailure(RuntimeError):
    pass


class WorkloadShortfall(StudyFailure):
    def __init__(self, result):
        super().__init__("planned step budget shortened by terminated/truncated result")
        self.result = result


class Budget:
    def __init__(self, root, seconds, max_bytes, *, clock=time.perf_counter):
        self.root, self.seconds, self.max_bytes = Path(root).resolve(), seconds, max_bytes
        self.clock, self.started = clock, clock()
        self.written = 0
        self.transient_bytes = 0
        self.peak_observed_bytes = 0
        self.reserve = min(1_048_576, max_bytes // 8)

    def check(self):
        if self.clock() - self.started >= self.seconds:
            raise BudgetExceeded("time budget reached at operation boundary")
        if self.written + self.transient_bytes > self.max_bytes - self.reserve:
            raise BudgetExceeded("storage budget reached at operation boundary")

    def sample_transient(self, directory):
        self.transient_bytes = sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())
        self.peak_observed_bytes = max(self.peak_observed_bytes, self.written + self.transient_bytes)
        self.check()

    def write(self, path, value, *, final=False, append=False):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("receipt path escapes owned campaign")
        data = value if isinstance(value, bytes) else encoded(value) + b"\n"
        limit = self.max_bytes if final else self.max_bytes - self.reserve
        if self.written + self.transient_bytes + len(data) > limit:
            raise BudgetExceeded("retained output would exceed storage cap")
        before = path.stat().st_size if path.exists() else 0
        try:
            with path.open("ab" if append else "xb") as stream:
                stream.write(data)
        finally:
            after = path.stat().st_size if path.exists() else 0
            self.written += max(0, after - before)
        self.peak_observed_bytes = max(self.peak_observed_bytes, self.written + self.transient_bytes)


class Phases:
    def __init__(self, clock=time.perf_counter):
        self.clock, self.seconds, self.current = clock, Counter(), None
        self.failed = None

    @contextmanager
    def measure(self, name):
        if self.current is not None:
            raise RuntimeError("timing phases must not overlap")
        started, self.current = self.clock(), name
        try:
            yield
        except BaseException:
            if self.failed is None:
                self.failed = name
            raise
        finally:
            self.seconds[name] += self.clock() - started
            self.current = None


def remove_owned_directory(directory, owner_root):
    """Delete only the newly allocated arm subtree; never inferred user paths."""
    owner_root, directory = Path(owner_root).resolve(), Path(directory)
    resolved = directory.resolve()
    if resolved.parent != owner_root or resolved == owner_root or not directory.is_dir():
        raise ValueError("cleanup is not an immediate owned arm directory")
    paths = list(directory.rglob("*"))
    for path in [directory, *paths]:
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("cleanup refuses links/junctions")
        if not path.resolve().is_relative_to(resolved):
            raise ValueError("cleanup escaped arm directory")
    for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    directory.rmdir()


def load_runtime_modules():
    # Only this process changes; no external process or host BLAS setting is touched.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    names = ["continuation_study.native_baseline", "pyjevsim_bridge.rl.continuation",
             "pyjevsim_bridge.rl.continuation.queue_bundle", "pyjevsim_bridge.rl.continuation.manufacturing_bundle",
             "pyjevsim_bridge.rl.models.queue_control", "pyjevsim_bridge.rl.models.manufacturing"]
    return {name: importlib.import_module(name) for name in names}


def source_metadata(modules):
    names = ("contracts", "registry", "envelope", "coordinator", "engine_pyjevsim", "boundary",
             "generic_boundary", "generic_bundle", "references", "lifecycle", "obligations",
             "adapters.queue", "adapters.manufacturing")
    sources = {name: Path(module.__file__).resolve() for name, module in modules.items()}
    for name in names:
        module = importlib.import_module(f"pyjevsim_bridge.rl.continuation.{name}")
        sources[module.__name__] = Path(module.__file__).resolve()
    for name in ("system_executor", "behavior_executor", "behavior_model", "schedule_queue", "snapshot_manager", "restore_handler"):
        module = importlib.import_module(f"pyjevsim.{name}")
        sources[module.__name__] = Path(module.__file__).resolve()
    for filename in ("cases.py", "run.py", "analyze.py", "native_baseline.py"):
        sources[f"study.{filename}"] = Path(__file__).with_name(filename).resolve()
    return {"python": sys.version, "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "source_files": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for name, path in sources.items()},
        "host_interference_controlled": False, "memory_complete": False, "whole_lifetime_peak_bytes": None}


class Backend:
    def __init__(self, spec, case, modules):
        self.spec, self.case = spec, case
        self.native = modules["continuation_study.native_baseline"]
        self.kind = "queue" if spec["model"] == "Q" else "manufacturing"
        self.cc = modules["pyjevsim_bridge.rl.continuation"]
        self.policy = {"policy_sha256": sha({"actions": "predeclared-maintenance-first-odd-or-queue-normal-v1"}),
                       "policy_version": 0, "feature_contract_sha256": sha({"model": spec["model"], "features": "fixed-action-plan"})}
        self.run_id = spec["cell_id"]
        self.family_id = f"{spec['model']}-f{spec['family']:02}"
        self.sampling = {"domain": "pyjevsim-live-branch-v1", "phase": "exploratory-pilot", "master": case["seed"],
            "segment": "prefix", "logical_branch_id": "prefix", "run_id": self.run_id, "generation": 0,
            "worker_id": "serial-worker", "episode_id": "episode-1", "sampling_seed": case["seed"]}
        if spec["method"] == "C":
            registry = self.cc.ContinuationRegistry()
            package = modules[f"pyjevsim_bridge.rl.continuation.{self.kind}_bundle"]
            self.bundle = getattr(package, f"register_{self.kind}_bundle")(registry)
            self.coordinator = self.cc.ContinuationCoordinator(registry)

    def fresh(self, physical_id):
        case, spec = self.case, self.spec
        if spec["method"] != "C":
            return self.native.create_native(self.kind, case["config"], seed=case["seed"], delta=spec["delta"],
                    max_steps=spec["prefix_steps"] + spec["suffix_steps"] + 1, instance_id=physical_id, run_id=self.run_id)
        return self.coordinator.create_fresh(self.cc.ResetRequest(self.bundle.profile.profile_id,
            case["config"], case["seed"], physical_id, self.run_id, spec["delta"],
            spec["prefix_steps"] + spec["suffix_steps"] + 1, self.policy, self.sampling))

    @staticmethod
    def graph(runtime):
        return runtime._parts.graph if hasattr(runtime, "_parts") else runtime.graph

    def capture_to_file(self, runtime, directory):
        if self.spec["method"] == "N":
            return self.native.save_native(runtime, directory, name="cut")
        snapshot = self.coordinator.capture(runtime, self.cc.CaptureRequest(runtime.profile_id,
            self.spec["prefix_steps"], self.family_id, self.run_id, self.policy))
        path = directory / "cut.json"
        with path.open("xb") as stream:
            stream.write(snapshot.data)
        return {"bytes": len(snapshot.data), "files": [path.name], "folder": str(directory)}

    def restore_from_file(self, directory, branch_index, physical_id):
        if self.spec["method"] == "N":
            return self.native.load_native(directory, name="cut", instance_id=physical_id)
        branch_id = f"branch-{branch_index}"
        context = self.cc.BranchContext(self.family_id, self.run_id, branch_id, self.policy,
            dict(self.sampling, segment="suffix", logical_branch_id=branch_id), physical_id)
        return self.coordinator.restore((directory / "cut.json").read_bytes(), context)


class WorkCounter:
    """Python callback call vector, deliberately NOT an independent event sum."""
    def __init__(self, modules, phases):
        q, m = modules["pyjevsim_bridge.rl.models.queue_control"], modules["pyjevsim_bridge.rl.models.manufacturing"]
        classes = (q.ArrivalSource, q.BufferServer, q.CompletionSink, m.JobSource, m.Stage, m.ToolArbiter, m.ProductSink)
        self.codes = {getattr(cls, name).__code__: name for cls in classes
                      for name in ("int_trans", "ext_trans", "output", "con_trans")}
        self.phases, self.counts = phases, {}

    def __call__(self, frame, event, arg):
        if event == "call" and frame.f_code in self.codes:
            phase = self.phases.current or "unclassified"
            row = self.counts.setdefault(phase, {name: 0 for name in ("int_trans", "ext_trans", "output", "con_trans")})
            row[self.codes[frame.f_code]] += 1


def execute_arm(spec, case, campaign, budget, modules, *, backend_factory=Backend, projection=physical_projection):
    """One application-level arm, no retries; projections retained only in RAM."""
    campaign = Path(campaign).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", spec["arm_id"]):
        raise ValueError("unsafe arm identifier")
    temp_root = campaign / "transient"
    temp = temp_root / spec["arm_id"]
    phases = Phases(budget.clock)
    counts = WorkCounter(modules, phases) if spec["purpose"] == "counting" else None
    previous_profile = sys.getprofile()
    projections, runtimes, closed, cleanup_errors = [], [], set(), []
    prefixes_executed = suffix_steps_executed = snapshot_bytes = 0
    step_calls, step_results = Counter(), Counter()
    transient_observation_complete = True
    status, error, error_phase = "succeeded", None, None
    construction_cleanup = True
    started = budget.clock()
    if counts is not None:
        sys.setprofile(counts)

    def close_runtime(runtime):
        if id(runtime) in closed:
            return
        closed.add(id(runtime))
        try:
            receipt = runtime.close()
        except BaseException as exc:
            cleanup_errors.append(f"{type(exc).__name__}: {exc}")
            raise
        if not receipt.success:
            cleanup_errors.append("returned runtime did not confirm cleanup")
            cleanup_errors.extend(str(item) for item in getattr(receipt, "errors", getattr(receipt, "failed_resources", ())))
            raise StudyFailure("runtime cleanup not confirmed")

    def advance(runtime, actions, phase, *, allow_terminal=False):
        nonlocal suffix_steps_executed
        outcomes = []
        for action in actions:
            budget.check()
            with phases.measure(phase):
                step_calls[phase] += 1
                row = runtime.step(action)
                step_results[phase] += 1
            if phase == "suffix":
                suffix_steps_executed += 1
                outcomes.append(row)
            if (row[2] or row[3]) and not allow_terminal:
                phases.failed = phase
                raise WorkloadShortfall(row)
        return outcomes

    try:
        budget.check()
        with phases.measure("backend_setup"):
            temp.mkdir(exist_ok=False)
            backend = backend_factory(spec, case, modules)
        prefix_actions, _ = action_plan(spec["model"], spec["prefix_steps"], 0)
        if spec["method"] != "R":
            budget.check()
            with phases.measure("fresh"):
                source = backend.fresh(f"{spec['arm_id']}-source")
                runtimes.append(source)
            advance(source, prefix_actions, "prefix")
            prefixes_executed += 1
            with phases.measure("capture_write"):
                artifact = backend.capture_to_file(source, temp)
                snapshot_bytes = artifact["bytes"]
            budget.sample_transient(temp)
            with phases.measure("cleanup"):
                close_runtime(source)
        for branch_index in range(spec["branch_count"]):
            budget.check()
            physical_id = f"{spec['arm_id']}-branch-{branch_index}"
            if spec["method"] == "R":
                with phases.measure("fresh"):
                    runtime = backend.fresh(physical_id)
                    runtimes.append(runtime)
                advance(runtime, prefix_actions, "prefix")
                prefixes_executed += 1
            else:
                with phases.measure("restore_read"):
                    runtime = backend.restore_from_file(temp, branch_index, physical_id)
                    runtimes.append(runtime)
            _, suffix_actions = action_plan(spec["model"], spec["prefix_steps"], branch_index)
            branch_rows = []
            projections.append(branch_rows)
            for offset, action in enumerate(suffix_actions, 1):
                try:
                    row = advance(runtime, [action], "suffix", allow_terminal=offset == len(suffix_actions))[0]
                except WorkloadShortfall as exc:
                    branch_rows.append({"workload_shortfall": True, "raw_step_result": json.loads(encoded(exc.result))})
                    raise
                with phases.measure("projection_serialize"):
                    try:
                        branch_rows.append(projection(spec["model"], backend.graph(runtime), row,
                            family_id=f"{spec['model']}-f{spec['family']:02}", branch_id=f"branch-{branch_index}",
                            expected_run_id=spec["cell_id"], expected_step=spec["prefix_steps"] + offset))
                    except BaseException as exc:
                        try:
                            raw = json.loads(encoded(row))
                        except (TypeError, ValueError):
                            raw = repr(row)[:4096]
                        branch_rows.append({"projection_error": f"{type(exc).__name__}: {exc}", "raw_step_result": raw})
                        raise
            with phases.measure("cleanup"):
                close_runtime(runtime)
        with phases.measure("projection_serialize"):
            branch_digests = [sha(rows) for rows in projections]
        if counts is not None and any(sum(counts.counts.get(name, {}).values())
                                     for name in ("capture_write", "restore_read")):
            raise StudyFailure("capture/restore executed simulated callbacks")
    except BaseException as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        error_phase = phases.failed or phases.current or "between-phases"
        if error_phase in ("fresh", "restore_read", "backend_setup"):
            # An unreturned candidate cannot be treated as closed because the
            # local handle list is empty. Keep the uncertainty explicit.
            construction_cleanup = None
        pending = [exc]
        while pending:
            item = pending.pop()
            pending.extend(getattr(item, "exceptions", ()))
            details = getattr(item, "cleanup_errors", ())
            if details or getattr(item, "code", None) == "CC_CLEANUP_UNCONFIRMED":
                cleanup_errors.extend(str(value) for value in details)
                cleanup_errors.append(f"unreturned candidate cleanup: {type(item).__name__}: {item}")
                construction_cleanup = False
        branch_digests = [sha(rows) for rows in projections]
    finally:
        with phases.measure("cleanup"):
            for runtime in runtimes:
                if id(runtime) not in closed:
                    try:
                        close_runtime(runtime)
                    except BaseException as exc:
                        cleanup_errors.append(f"{type(exc).__name__}: {exc}")
            if temp.exists():
                try:
                    remove_owned_directory(temp, temp_root)
                except BaseException as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
            try:
                budget.transient_bytes = 0 if not temp.exists() else sum(p.stat().st_size for p in temp.rglob("*") if p.is_file())
            except BaseException as exc:
                transient_observation_complete = False
                cleanup_errors.append(f"remaining transient bytes unobserved: {type(exc).__name__}: {exc}")
        if counts is not None:
            sys.setprofile(previous_profile)
    if cleanup_errors:
        status = "failed"
        error = (error + "; " if error else "") + "cleanup unconfirmed"
    compact = {**spec, "status": status, "error": error, "error_phase": error_phase,
        "config_sha256": case["config_sha256"], "seed": case["seed"],
        "prefixes_executed": prefixes_executed, "suffix_steps_executed": suffix_steps_executed,
        "step_calls_attempted": dict(step_calls), "step_results_returned": dict(step_results),
        "expected_suffix_steps": spec["branch_count"] * spec["suffix_steps"],
        "branch_projection_sha256": branch_digests, "snapshot_bytes": snapshot_bytes,
        "cleanup_confirmed": False if cleanup_errors else construction_cleanup, "cleanup_errors": cleanup_errors,
        "memory_complete": False, "whole_lifetime_peak_bytes": None,
        "transient_observation_complete": transient_observation_complete,
        "work_counts": None if counts is None else counts.counts}
    try:
        with phases.measure("compact_receipt"):
            budget.write(campaign / "receipts" / f"{spec['arm_id']}.json",
                {"arm_id": spec["arm_id"], "status": status, "prefixes": prefixes_executed,
                 "suffix_steps": suffix_steps_executed, "timing_values_stored_in": "arms.jsonl"})
    except BaseException as exc:
        compact.update(status="failed", error=f"{error or ''}; receipt: {type(exc).__name__}: {exc}")
    elapsed = budget.clock() - started
    compact.update(application_wall_seconds=elapsed, phase_seconds=dict(phases.seconds),
        unclassified_seconds=elapsed - sum(phases.seconds.values()),
        endpoint="backend setup through cleanup and untimed-content receipt; arms.jsonl write excluded")
    return compact, projections


def denominators(manifest, rows):
    by_id = {row["arm_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate arm result is not a retry policy")
    result = {}
    for purpose in ("timing", "counting"):
        planned = [arm for arm in manifest["arms"] if arm["purpose"] == purpose]
        attempted = [by_id[arm["arm_id"]] for arm in planned if arm["arm_id"] in by_id]
        result[purpose] = {"planned": len(planned), "attempted": len(attempted),
            "succeeded": sum(row["status"] == "succeeded" for row in attempted),
            "failed": sum(row["status"] != "succeeded" for row in attempted),
            "unexecuted": len(planned) - len(attempted)}
    return result


def run_campaign(campaign, manifest, *, modules=None, backend_factory=Backend, projection=physical_projection):
    """Create a new campaign only; never resume/retry a partial campaign."""
    campaign = Path(campaign).resolve()
    campaign.mkdir(parents=False, exist_ok=False)
    for name in ("transient", "receipts", "failures"):
        (campaign / name).mkdir()
    budget = Budget(campaign, manifest["budget"]["seconds"], manifest["budget"]["max_bytes"])
    started_utc = datetime.now(timezone.utc).isoformat()
    rows, cells, outcomes = [], [], {}
    stop_reason = None
    write_errors, unpersisted_rows, unpersisted_cells = [], [], []
    try:
        budget.write(campaign / "manifest.json", manifest)
        modules = load_runtime_modules() if modules is None else modules
        budget.write(campaign / "source-metadata.json", source_metadata(modules))
    except BaseException as exc:
        stop_reason = f"campaign setup failed: {type(exc).__name__}: {exc}"
    for spec in (() if stop_reason else manifest["arms"]):
        try:
            budget.check()
        except BudgetExceeded as exc:
            stop_reason = str(exc)
            break
        try:
            row, full = execute_arm(spec, manifest["cases"][spec["case_id"]], campaign, budget, modules,
                                    backend_factory=backend_factory, projection=projection)
        except BaseException as exc:
            row, full = ({**spec, "config_sha256": manifest["cases"][spec["case_id"]]["config_sha256"],
                          "seed": manifest["cases"][spec["case_id"]]["seed"],
                          "status": "failed", "error": f"outside arm handler: {type(exc).__name__}: {exc}",
                          "cleanup_confirmed": None, "application_wall_seconds": None}, [])
        rows.append(row)
        try:
            budget.write(campaign / "arms.jsonl", row, append=True)
        except BaseException as exc:
            write_errors.append(f"arms.jsonl: {type(exc).__name__}: {exc}")
            unpersisted_rows.append(row)
            stop_reason = write_errors[-1]
        group = outcomes.setdefault(spec["cell_id"], {})
        group[spec["method"]] = full
        if row["status"] != "succeeded" and stop_reason is None:
            stop_reason = f"arm failed: {spec['arm_id']}: {row['error']}"
        elif stop_reason is None and set(group) == {"R", "N", "C"}:
            same = encoded(group["R"]) == encoded(group["N"]) == encoded(group["C"])
            cell = {"cell_id": spec["cell_id"], "purpose": spec["purpose"], "exact": same,
                    "methods": ["R", "N", "C"], "branches": spec["branch_count"],
                    "steps_per_branch": spec["suffix_steps"]}
            cells.append(cell)
            try:
                budget.write(campaign / "cells.jsonl", cell, append=True)
            except BaseException as exc:
                write_errors.append(f"cells.jsonl: {type(exc).__name__}: {exc}")
                unpersisted_cells.append(cell)
                stop_reason = write_errors[-1]
            if not same:
                stop_reason = f"exact semantic mismatch: {spec['cell_id']}"
        if stop_reason:
            try:
                budget.write(campaign / "failures" / f"{spec['cell_id']}.json", {
                    "cell_id": spec["cell_id"], "reason": stop_reason, "projections": group})
            except BaseException as exc:
                write_errors.append(f"failure projection: {type(exc).__name__}: {exc}")
                stop_reason += "; full failure projections not retained; terminal reserve protected"
            break
        if set(group) == {"R", "N", "C"}:
            del outcomes[spec["cell_id"]]
    summary = {"schema": "continuation-study-execution-v1", "status": "completed" if stop_reason is None else "stopped",
        "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(), "stop_reason": stop_reason,
        "denominators": denominators(manifest, rows),
        "exact_cells": {purpose: {"planned": manifest["planned"][f"{purpose}_cells"],
            "compared": sum(cell["purpose"] == purpose for cell in cells),
            "exact": sum(cell["purpose"] == purpose and cell["exact"] for cell in cells)} for purpose in ("timing", "counting")},
        "unexecuted_arm_ids": [arm["arm_id"] for arm in manifest["arms"] if arm["arm_id"] not in {row["arm_id"] for row in rows}],
        "campaign_wall_seconds": budget.clock() - budget.started, "retained_bytes_before_summary": budget.written,
        "peak_storage_observed_at_boundaries_bytes": budget.peak_observed_bytes,
        "budget_enforcement": "cooperative step/phase boundaries; no guarantee against a stuck native call or unsampled temporary peak",
        "memory_complete": False, "whole_lifetime_peak_bytes": None, "host_interference_controlled": False,
        "write_errors": write_errors, "unpersisted_arm_records": unpersisted_rows,
        "unpersisted_cell_records": unpersisted_cells,
        "discarded_evidence": "successful snapshot files and full successful projections removed; digests cannot reconstruct discarded traces",
        "study_admission": stop_reason is None and len(rows) == 312 and len(cells) == 104 and all(cell["exact"] for cell in cells)}
    try:
        budget.write(campaign / "execution.json", summary, final=True)
    except BaseException as exc:
        summary.update(status="stopped", study_admission=False,
            execution_write_error=f"{type(exc).__name__}: {exc}",
            stop_reason=f"{stop_reason or 'execution finished'}; terminal record could not be written")
        # A filesystem failure cannot be guaranteed recoverable. Preserve the
        # complete terminal denominator in the process output as best effort.
        print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--budget-seconds", required=True, type=float)
    parser.add_argument("--max-bytes", required=True, type=int)
    parser.add_argument("--planning-seed", default=930251, type=int)
    args = parser.parse_args()
    manifest = make_manifest(budget_seconds=args.budget_seconds, max_bytes=args.max_bytes,
                             planning_seed=args.planning_seed)
    result = run_campaign(args.campaign, manifest)
    analysis_status = "not-started"
    if not result.get("execution_write_error"):
        try:
            from continuation_study.analyze import analyze_campaign
            analysis = analyze_campaign(args.campaign)
            analysis_status = analysis["analysis_status"]["status"]
        except BaseException as exc:
            analysis_status = f"failed-or-budget-exhausted: {type(exc).__name__}: {exc}"
    print(json.dumps({"status": result["status"], "denominators": result["denominators"],
                      "stop_reason": result["stop_reason"], "analysis_status": analysis_status}, ensure_ascii=False))
    return 0 if result["study_admission"] and analysis_status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
