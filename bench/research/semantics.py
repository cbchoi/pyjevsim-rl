"""Direct restoration/intervention research; no performance claims or gates.

The oracle runs the same native simulator from fresh state, but never calls a
snapshot reader, serializer or continuation state provider. This establishes
independence from restoration, not independence from simulator/model defects.
Instrumentation observes Python calls without changing qualified engine methods.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import tempfile
import time

from bench.continuation_study.cases import encoded, physical_projection, sha, workload

SCHEMA = "continuation-semantics-research-v1"
METHODS = ("R", "N", "C1")
SEEDS = (971000, 971001, 971002)
CUTS = (1, 4, 9)


class StorageBudgetExceeded(RuntimeError):
    pass


def _modules():
    from continuation_study.run import load_runtime_modules
    return load_runtime_modules()


def _plain(value):
    """Detach observed physical data; infinity is an explicit clock sentinel."""
    if isinstance(value, float) and math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _parts(runtime):
    parts = getattr(runtime, "_parts", runtime)
    reward = parts.boundary_state.value if hasattr(parts, "boundary_state") else parts.reward_state
    return parts.engine, parts.graph, parts.env, reward


def runtime_state(runtime, model):
    """Independent direct observation, deliberately not an adapter state export."""
    engine, graph, env, reward = _parts(runtime)
    wrappers = {name: {key: getattr(items[0], key) for key in
                          ("global_time", "request_time", "_next_event_t", "_cur_state")}
                for name, items in sorted(engine.model_map.items())}
    state = {"logical_time": engine.global_time, "step_id": env._step_id,
             "observation_cache": env._observation, "reward_cache": reward,
             "done": env._done, "failed": env._failed, "wrappers": wrappers,
             "input_queue_empty": not engine.input_event_queue,
             "output_queue_empty": not engine.output_event_queue}
    if model == "Q":
        server = graph.server
        state["domain"] = {key: getattr(server, key) for key in
            ("current", "waiting", "remaining", "mode", "last_event_time", "backlog",
             "energy", "completed", "admitted", "dropped", "source_arrivals", "trace")}
        state["source_cursor"] = graph.source.index
        state["completion_history"] = graph.sink.ledger
        state["rng_state"] = None  # Q's explicit arrival tape makes no future RNG draw.
        state["shared_aliases"] = None
    else:
        state["domain"] = {"tool_trace": graph.ledger.trace, "owner": graph.ledger.owner,
            "repair_until": graph.ledger.repair_until,
            "maintenance_pending": graph.arbiter.maintenance_pending,
            "requests": graph.arbiter.requests,
            "stages": [{key: getattr(stage, key) for key in
                        ("waiting", "pending_requests", "current", "finish_at", "completed")}
                       for stage in (graph.stage_a, graph.stage_b)]}
        state["source_cursor"] = [graph.source.job_cursor, graph.source.maintenance_cursor]
        state["completion_history"] = graph.sink.completions
        state["rng_state"] = graph.arbiter.rng.getstate()
        state["rng_draws"] = graph.arbiter.rng_draws
        state["shared_aliases"] = all(obj.ledger is graph.ledger for obj in
                                     (graph.stage_a, graph.stage_b, graph.arbiter))
    return json.loads(encoded(_plain(state)))


class CallObserver:
    """Non-timing telemetry; no monkeypatch or callback replacement."""
    def __init__(self, modules):
        from pyjevsim.system_executor import SysExecutor
        q = modules["pyjevsim_bridge.rl.models.queue_control"]
        m = modules["pyjevsim_bridge.rl.models.manufacturing"]
        self.classes = (q.ArrivalSource, q.BufferServer, q.CompletionSink,
                        m.JobSource, m.Stage, m.ToolArbiter, m.ProductSink)
        self.codes = {getattr(cls, name).__code__: name for cls in self.classes
                      for name in ("int_trans", "ext_trans", "output", "con_trans")}
        self.insert_code = SysExecutor.insert_external_event.__code__
        self.callbacks = []
        self.injections = []
        self.action_deliveries = []

    def __call__(self, frame, event, _arg):
        if event != "call":
            return
        if frame.f_code is self.insert_code:
            v = frame.f_locals
            self.injections.append({"port": v["_port"], "payload": copy.deepcopy(v["_msg"]),
                "inserted_at": v["self"].global_time,
                "scheduled_for": v["self"].global_time + v["scheduled_time"]})
        name = self.codes.get(frame.f_code)
        obj = frame.f_locals.get("self")
        if name and isinstance(obj, self.classes):
            self.callbacks.append(name)
            if name == "ext_trans" and frame.f_locals.get("port") == "action":
                self.action_deliveries.append({"model": obj.get_name(), "time": obj.clock(),
                    "payloads": copy.deepcopy(frame.f_locals["message"].retrieve())})

    @contextmanager
    def observe(self):
        previous = sys.getprofile()
        sys.setprofile(self)
        try:
            yield self
        finally:
            sys.setprofile(previous)


def _deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("research deadline reached at operation boundary")


def _storage(directory, max_bytes, receipt):
    observed = sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())
    receipt["peak_transient_observed_bytes"] = max(receipt.get("peak_transient_observed_bytes", 0), observed)
    if max_bytes is not None and observed > max_bytes:
        raise StorageBudgetExceeded("research transient storage budget reached at operation boundary")


def _actions(model, count, intervention=False):
    if model == "Q":
        return [{"mode": "fast" if intervention else "normal"} for _ in range(count)]
    return [{"maintenance": bool(intervention and i == 0)} for i in range(count)]


def _projection(runtime, model, row, step, run_id):
    _, graph, _, _ = _parts(runtime)
    value = physical_projection(model, graph, row, family_id="semantics",
        branch_id="same-logical-branch", expected_run_id=run_id, expected_step=step)
    # Identity is checked above, but cannot manufacture physical branch effects.
    for key in ("family_id", "branch_id", "run_id"):
        value.pop(key)
    value["direct_state"] = runtime_state(runtime, model)
    return value


def physical_effect(left, right, model):
    """Exclude identities, action/mode echo and intervention trace labels."""
    keys = (("remaining_work", "completed", "energy_integral", "backlog_integral")
            if model == "Q" else
            ("repair_remaining", "maintenance_pending", "rng_draws", "completed", "wip_integral"))
    for index, (a, b) in enumerate(zip(left, right), 1):
        differences = {key: [a["observation"][key], b["observation"][key]]
                       for key in keys if a["observation"][key] != b["observation"][key]}
        if a["reward"] != b["reward"]:
            differences["reward"] = [a["reward"], b["reward"]]
        if differences:
            return {"observed": True, "first_suffix_step": index, "differences": differences}
    return {"observed": False, "first_suffix_step": None, "differences": {}}


def _first_difference(expected, observed, path="$"):
    if expected == observed:
        return None
    if isinstance(expected, dict) and isinstance(observed, dict):
        if set(expected) != set(observed):
            return {"path": path, "expected_keys": sorted(expected), "observed_keys": sorted(observed)}
        for key in expected:
            difference = _first_difference(expected[key], observed[key], f"{path}.{key}")
            if difference:
                return difference
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            return {"path": path, "expected_length": len(expected), "observed_length": len(observed)}
        for index, (left, right) in enumerate(zip(expected, observed)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
    return {"path": path, "expected": expected, "observed": observed}


def _backend(model, method, seed, cut, suffix_steps, modules):
    from continuation_study.run import Backend
    case = workload(model, 0)
    case.update(seed=seed)
    spec = {"model": model, "method": "C" if method == "C1" else method,
        "cell_id": f"semantics-{model}-{seed}-L{cut}", "family": 0,
        "prefix_steps": cut, "suffix_steps": suffix_steps, "delta": .25}
    return Backend(spec, case, modules)


def _prefix(runtime, model, count, deadline):
    for action in _actions(model, count):
        _deadline(deadline)
        row = runtime.step(action)
        if row[2] or row[3]:
            raise RuntimeError("semantics prefix ended before declared cut")


def run_case(model, seed, cut, *, suffix_steps=16, modules=None, deadline=None,
             scratch_root=None, max_bytes=None):
    """Nine trajectories: R/N/C1 each A/B/A, with live sibling isolation checks."""
    modules = modules or _modules()
    result = {"case_id": f"{model}-{seed}-L{cut}", "model": model, "seed": seed,
        "cut_steps": cut, "cut_time": cut * .25, "suffix_steps": suffix_steps,
        "status": "failed", "planned_trajectories": 9, "attempted_trajectories": 0,
        "completed_trajectories": 0,
        "checks": {}, "methods": {}, "errors": [], "cleanup_confirmed": True,
        "peak_transient_observed_bytes": 0, "unsampled_transient_peak_known": False}
    live = []
    trajectories = {}
    reference_cut = None
    try:
        with tempfile.TemporaryDirectory(prefix="pyjevsim-semantics-", dir=scratch_root) as temporary:
            for method in METHODS:
                _deadline(deadline)
                backend = _backend(model, method, seed, cut, suffix_steps, modules)
                folder = Path(temporary) / method
                folder.mkdir()
                source, source_state = None, None
                method_receipt = result["methods"][method] = {
                    "capture_callbacks": 0, "restore_callbacks": 0, "restored_cut_exact": True,
                    "source_unchanged": True, "sibling_isolation": True,
                    "first_input_at_cut": True, "first_delivery_at_cut": True,
                    "step_clocks_exact": True, "shared_aliases_preserved": True,
                    "trajectories": []}
                if method != "R":
                    source = backend.fresh(f"semantics-{method}-source")
                    live.append(source)
                    _prefix(source, model, cut, deadline)
                    source_state = runtime_state(source, model)
                    observer = CallObserver(modules)
                    _storage(temporary, max_bytes, result)
                    with observer.observe():
                        backend.capture_to_file(source, folder)
                    _storage(temporary, max_bytes, result)
                    method_receipt["capture_callbacks"] = len(observer.callbacks)
                    method_receipt["source_unchanged"] &= runtime_state(source, model) == source_state
                branches = []
                for branch_index in range(3):
                    _deadline(deadline)
                    if method == "R":
                        runtime = backend.fresh(f"semantics-{method}-branch-{branch_index}")
                        live.append(runtime)
                        _prefix(runtime, model, cut, deadline)
                    else:
                        _storage(temporary, max_bytes, result)
                        observer = CallObserver(modules)
                        with observer.observe():
                            runtime = backend.restore_from_file(folder, branch_index,
                                f"semantics-{method}-branch-{branch_index}")
                        live.append(runtime)
                        method_receipt["restore_callbacks"] += len(observer.callbacks)
                    state = runtime_state(runtime, model)
                    if reference_cut is None:
                        reference_cut = state
                    method_receipt["restored_cut_exact"] &= state == reference_cut
                    if model == "M":
                        method_receipt["shared_aliases_preserved"] &= state["shared_aliases"]
                    branches.append(runtime)
                rows = []
                for branch_index, runtime in enumerate(branches):
                    result["attempted_trajectories"] += 1
                    siblings = {i: runtime_state(other, model) for i, other in enumerate(branches)
                                if i != branch_index}
                    branch_rows = []
                    actions = _actions(model, suffix_steps, intervention=branch_index == 1)
                    first_input = first_delivery = None
                    for offset, action in enumerate(actions, 1):
                        _deadline(deadline)
                        observer = CallObserver(modules)
                        with observer.observe():
                            row = runtime.step(action)
                        if offset == 1:
                            first_input, first_delivery = observer.injections, observer.action_deliveries
                            method_receipt["first_input_at_cut"] &= (
                                len(first_input) == 1 and first_input[0]["port"] == "action"
                                and first_input[0]["payload"] == action
                                and first_input[0]["inserted_at"] == cut * .25
                                and first_input[0]["scheduled_for"] == cut * .25)
                            method_receipt["first_delivery_at_cut"] &= (
                                len(first_delivery) == 1 and first_delivery[0]["time"] == cut * .25
                                and first_delivery[0]["payloads"] == [action])
                        branch_rows.append(_projection(runtime, model, row, cut + offset, backend.run_id))
                        method_receipt["step_clocks_exact"] &= row[4]["logical_time"] == (cut + offset) * .25
                        if (row[2] or row[3]) and offset != suffix_steps:
                            raise RuntimeError("semantics suffix ended before its declared horizon")
                    rows.append(branch_rows)
                    result["completed_trajectories"] += 1
                    method_receipt["trajectories"].append({"branch": "B" if branch_index == 1 else "A",
                        "repetition": branch_index, "physical_sha256": sha(branch_rows),
                        "steps": len(branch_rows), "first_input": first_input,
                        "first_delivery": first_delivery})
                    method_receipt["sibling_isolation"] &= all(
                        runtime_state(branches[i], model) == state for i, state in siblings.items())
                    if source is not None:
                        method_receipt["source_unchanged"] &= runtime_state(source, model) == source_state
                trajectories[method] = rows
                method_receipt["repeat_A_exact"] = rows[0] == rows[2]
                method_receipt["physical_intervention_effect"] = physical_effect(rows[0], rows[1], model)
            result["checks"] = {
                "N_equals_independent_replay": trajectories["N"] == trajectories["R"],
                "C1_equals_independent_replay": trajectories["C1"] == trajectories["R"],
                "zero_capture_restore_callbacks": all(r["capture_callbacks"] == r["restore_callbacks"] == 0
                                                       for r in result["methods"].values()),
                **{key: all(r[key] for r in result["methods"].values()) for key in
                   ("restored_cut_exact", "source_unchanged", "sibling_isolation", "first_input_at_cut",
                    "first_delivery_at_cut", "step_clocks_exact", "shared_aliases_preserved", "repeat_A_exact")},
                "physical_intervention_effect": all(r["physical_intervention_effect"]["observed"]
                                                    for r in result["methods"].values())}
            result["oracle_comparisons"] = {method: {
                "compared_steps": sum(len(rows) for rows in trajectories["R"]),
                "first_difference": _first_difference(trajectories["R"], trajectories[method])}
                for method in ("N", "C1")}
            result["status"] = "succeeded" if all(result["checks"].values()) else "failed"
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["deadline_reached"] = isinstance(exc, TimeoutError)
        result["storage_budget_reached"] = isinstance(exc, StorageBudgetExceeded)
    finally:
        for runtime in reversed(live):
            try:
                receipt = runtime.close()
                if not receipt.success:
                    raise RuntimeError(str(receipt))
            except Exception as exc:
                result["cleanup_confirmed"] = False
                result["errors"].append(f"cleanup {type(exc).__name__}: {exc}")
        if not result["cleanup_confirmed"]:
            result["status"] = "failed"
    result["failed_trajectories"] = result["attempted_trajectories"] - result["completed_trajectories"]
    result["unexecuted_trajectories"] = result["planned_trajectories"] - result["attempted_trajectories"]
    return result


def negative_controls(*, modules=None, deadline=None, scratch_root=None, max_bytes=None):
    """Corrupt disposable N handles, testing observer sensitivity, not C1 rejection."""
    modules = modules or _modules()
    report = {"scope": "live disposable native restores; not C1 fault rejection",
              "planned": 5, "attempted": 0, "controls": [], "status": "failed", "errors": [],
              "peak_transient_observed_bytes": 0, "unsampled_transient_peak_known": False}
    live = []
    try:
        with tempfile.TemporaryDirectory(prefix="pyjevsim-negative-controls-", dir=scratch_root) as temporary:
            backend = _backend("M", "N", SEEDS[0], 9, 16, modules)
            source = backend.fresh("negative-source")
            live.append(source)
            _prefix(source, "M", 9, deadline)
            _storage(temporary, max_bytes, report)
            backend.capture_to_file(source, Path(temporary))
            _storage(temporary, max_bytes, report)
            oracle = _backend("M", "R", SEEDS[0], 9, 16, modules).fresh("negative-replay")
            live.append(oracle)
            _prefix(oracle, "M", 9, deadline)
            reference = runtime_state(oracle, "M")
            oracle_rows = []
            for offset, action in enumerate(_actions("M", 16, intervention=True), 1):
                _deadline(deadline)
                oracle_rows.append(_projection(oracle, "M", oracle.step(action), 9 + offset, backend.run_id))
            for index, name in enumerate(("clock_shift", "rng_advance", "reward_cache_offset",
                                          "shared_alias_split", "event_history_reorder")):
                _deadline(deadline)
                report["attempted"] += 1
                _storage(temporary, max_bytes, report)
                runtime = backend.restore_from_file(Path(temporary), index, f"negative-{index}")
                live.append(runtime)
                baseline_exact = runtime_state(runtime, "M") == reference
                if name == "clock_shift":
                    runtime.engine.global_time += .125
                elif name == "rng_advance":
                    runtime.graph.arbiter.rng.random()
                elif name == "reward_cache_offset":
                    runtime.reward_state["last_cost"] += 1.0
                elif name == "shared_alias_split":
                    runtime.graph.stage_a.ledger = copy.deepcopy(runtime.graph.ledger)
                else:
                    if len(runtime.graph.ledger.trace) < 2:
                        raise RuntimeError("event-order control has insufficient ordered history")
                    runtime.graph.ledger.trace.reverse()
                observed = runtime_state(runtime, "M")
                changed_fields = [key for key in reference if reference[key] != observed[key]]
                suffix_different = None
                if name in ("rng_advance", "reward_cache_offset"):
                    rows = []
                    for offset, action in enumerate(_actions("M", 16, intervention=True), 1):
                        _deadline(deadline)
                        rows.append(_projection(runtime, "M", runtime.step(action), 9 + offset, backend.run_id))
                    # Inspect physical outputs, not only the intentionally changed state.
                    suffix_different = any(a["observation"] != b["observation"] or a["reward"] != b["reward"]
                                           or a["history"] != b["history"]
                                           for a, b in zip(rows, oracle_rows))
                report["controls"].append({"name": name, "baseline_exact": baseline_exact,
                    "detected": bool(changed_fields), "changed_observer_fields": changed_fields,
                    "subsequent_physical_difference": suffix_different})
            report["status"] = "succeeded" if all(
                c["baseline_exact"] and c["detected"] and c["subsequent_physical_difference"] is not False
                for c in report["controls"]) else "failed"
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        for runtime in reversed(live):
            try:
                if not runtime.close().success:
                    raise RuntimeError("cleanup not confirmed")
            except Exception as exc:
                report["status"] = "failed"
                report["errors"].append(f"cleanup {type(exc).__name__}: {exc}")
    report["detected"] = sum(c["detected"] for c in report["controls"])
    report["failed"] = report["attempted"] - sum(
        c["baseline_exact"] and c["detected"] and c["subsequent_physical_difference"] is not False
        for c in report["controls"])
    report["unexecuted"] = report["planned"] - report["attempted"]
    return report


def run_study(output_dir=None, *, seeds=SEEDS, cuts=CUTS, suffix_steps=16, deadline=None,
              scratch_root=None, max_bytes=None):
    """Return compact complete/partial evidence; never retry a failed case."""
    started = time.perf_counter()
    seeds, cuts = tuple(seeds), tuple(cuts)
    if (not seeds or not cuts or len(set(seeds)) != len(seeds) or len(set(cuts)) != len(cuts)
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or any(type(cut) is not int or cut <= 0 for cut in cuts)
            or type(suffix_steps) is not int or suffix_steps <= 0):
        raise ValueError("unique nonnegative seeds, positive cuts and positive suffix required")
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("positive transient storage allowance required")
    plan = [(model, seed, cut) for model in ("Q", "M") for seed in seeds for cut in cuts]
    modules = _modules()
    report = {"schema": SCHEMA, "study_type": "exploratory-semantics-not-performance",
        "planned": len(plan), "attempted": 0, "succeeded": 0, "failed": 0, "unexecuted": len(plan),
        "planned_trajectories": 9 * len(plan), "completed_trajectories": 0, "cases": [],
        "oracle_scope": "native fresh replay independent of restoration, not of simulator/model",
        "cut_semantics": "committed drained boundary; no retroactive intervention",
        "memory_complete": False, "host_interference_controlled": False,
        "max_transient_bytes": max_bytes, "peak_transient_observed_bytes": 0,
        "unsampled_transient_peak_known": False,
        "transient_storage_scope": "all files in owned case subtree at capture/restore boundaries"}
    report["manifest"] = {"methods": list(METHODS), "seeds": list(seeds), "cuts": list(cuts),
        "delta": .25, "suffix_steps": suffix_steps, "branch_order": ["A", "B", "A"],
        "configuration_sha256": {model: workload(model, 0)["config_sha256"] for model in ("Q", "M")},
        "Q_intervention": "normal versus fast throughout suffix",
        "M_intervention": "false throughout versus true on first suffix step only",
        "negative_controls": ["clock_shift", "rng_advance", "reward_cache_offset",
                              "shared_alias_split", "event_history_reorder"]}
    report["manifest"]["sha256"] = sha(report["manifest"])
    for model, seed, cut in plan:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        case = run_case(model, seed, cut, suffix_steps=suffix_steps, modules=modules, deadline=deadline,
                        scratch_root=scratch_root, max_bytes=max_bytes)
        report["cases"].append(case)
        report["attempted"] += 1
        report["succeeded" if case["status"] == "succeeded" else "failed"] += 1
        report["completed_trajectories"] += case["completed_trajectories"]
        report["peak_transient_observed_bytes"] = max(report["peak_transient_observed_bytes"],
                                                     case["peak_transient_observed_bytes"])
        if case.get("deadline_reached") or case.get("storage_budget_reached"):
            break
    report["unexecuted"] = report["planned"] - report["attempted"]
    storage_reached = any(case.get("storage_budget_reached") for case in report["cases"])
    if not storage_reached and (deadline is None or time.perf_counter() < deadline):
        report["negative_controls"] = negative_controls(modules=modules, deadline=deadline,
                                                        scratch_root=scratch_root, max_bytes=max_bytes)
        report["peak_transient_observed_bytes"] = max(report["peak_transient_observed_bytes"],
            report["negative_controls"]["peak_transient_observed_bytes"])
    else:
        report["negative_controls"] = {"planned": 5, "attempted": 0, "status": "unexecuted"}
    report["admission"] = (report["failed"] == report["unexecuted"] == 0
                            and report["negative_controls"]["status"] == "succeeded")
    report["study_admission"] = report["admission"]
    report["status"] = "completed" if report["admission"] else "incomplete-or-failed"
    report["elapsed_seconds"] = time.perf_counter() - started
    if output_dir is not None:
        with (Path(output_dir) / "semantics.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report
