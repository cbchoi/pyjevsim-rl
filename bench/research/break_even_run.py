"""One break-even arm with a contiguous, observer-free production endpoint.

Companions use the same backend/model calls with an independent observer. Their
elapsed time is deliberately not emitted as performance evidence. This module
does not launch processes, install dependencies or run preflight test suites.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import math
from pathlib import Path
import re
import time

from .break_even_design import ENDPOINT, encoded, sha
from ..continuation_study.run import remove_owned_directory


CALLBACKS = ("int_trans", "ext_trans", "output", "con_trans")
PHASES = ("setup", "fresh", "prefix", "capture_write", "source_close",
          "restore_read", "suffix", "branch_close", "artifact_cleanup")
FEATURE_CONTRACT_SHA256 = sha({"model": "inventory-risk-v1", "actions": "open-loop"})


class ArmFailure(RuntimeError):
    pass


class PhaseClock:
    def __init__(self, clock):
        self.clock, self.current, self.failed = clock, None, None
        self.seconds = Counter({name: 0.0 for name in PHASES})
        self.invocations = Counter({name: 0 for name in PHASES})

    @contextmanager
    def measure(self, name):
        if self.current is not None:
            raise RuntimeError("phase clocks must not overlap")
        self.current = name
        self.invocations[name] += 1
        started = self.clock()
        try:
            yield
        except BaseException:
            self.failed = self.failed or name
            raise
        finally:
            self.seconds[name] += self.clock() - started
            self.current = None


def load_runtime_modules():
    # Called before the workflow timer, including when a caller injects clocks.
    from . import break_even_domain, break_even_adapter, break_even_native
    from pyjevsim_bridge.rl import continuation
    return {"domain": break_even_domain, "adapter": break_even_adapter,
            "native": break_even_native, "continuation": continuation}


class Backend:
    def __init__(self, spec, case, modules):
        self.spec, self.case, self.modules = spec, case, modules
        self.native, self.cc = modules["native"], modules["continuation"]
        self.policy = {"policy_sha256": case["action_identity"], "policy_version": 0,
                       "feature_contract_sha256": FEATURE_CONTRACT_SHA256}
        self.sampling = {"domain": "pyjevsim-live-branch-v1", "phase": "break-even-v1",
            "master": case["seed"], "segment": "prefix", "logical_branch_id": "prefix",
            "run_id": spec["cell_id"], "generation": 0, "worker_id": "serial-worker",
            "episode_id": "episode-1", "sampling_seed": case["seed"]}
        if spec["method"] == "C1":
            self.registry = self.cc.ContinuationRegistry()
            self.bundle = modules["adapter"].make_bundle()
            self.registry.register(self.bundle)
            self.coordinator = self.cc.ContinuationCoordinator(self.registry)

    def fresh(self, physical_id):
        spec, case = self.spec, self.case
        maximum = spec["prefix_steps"] + spec["suffix_steps"] + 1
        if spec["method"] != "C1":
            return self.native.create_native(case["config"], seed=case["seed"], delta=spec["delta"],
                max_steps=maximum, instance_id=physical_id, run_id=spec["cell_id"])
        return self.coordinator.create_fresh(self.cc.ResetRequest(self.bundle.profile.profile_id,
            case["config"], case["seed"], physical_id, spec["cell_id"], spec["delta"], maximum,
            self.policy, self.sampling))

    def capture_to_file(self, runtime, directory):
        if self.spec["method"] == "N":
            return self.native.save_native(runtime, directory)
        snapshot = self.coordinator.capture(runtime, self.cc.CaptureRequest(runtime.profile_id,
            self.spec["prefix_steps"], self.spec["family_id"], self.spec["cell_id"], self.policy))
        path = directory / "cut.json"
        with path.open("xb") as stream:
            stream.write(snapshot.data)
        return {"bytes": len(snapshot.data), "files": [path.name]}

    def restore_from_file(self, directory, branch_index, physical_id):
        if self.spec["method"] == "N":
            return self.native.load_native(directory, instance_id=physical_id)
        branch_id = f"branch-{branch_index}"
        context = self.cc.BranchContext(self.spec["family_id"], self.spec["cell_id"], branch_id,
            self.policy, dict(self.sampling, segment="suffix", logical_branch_id=branch_id), physical_id)
        return self.coordinator.restore((directory / "cut.json").read_bytes(), context)

    @staticmethod
    def physical_state(runtime, now):
        parts = runtime._parts if hasattr(runtime, "_parts") else runtime
        reward = parts.boundary_state.value if hasattr(parts, "boundary_state") else parts.reward_state
        state = parts.graph.physical_state(now, reward)
        # Lossless column encoding, not a digest or snapshot-provider export.
        # It avoids repeating five field names 512*16*16 times in transport.
        domain = state.get("domain", {})
        products = domain.get("products")
        if products and type(products[0]) is dict:
            fields = sorted(products[0])
            if any(sorted(product) != fields for product in products):
                raise ArmFailure("physical product fields differ")
            domain["products"] = {"fields": fields, "values": [[product[key] for key in fields] for product in products]}
        return state


def scalar_witness(result, *, run_id, physical_id, expected_step, seed):
    """Consume only already-returned immutable scalars; never inspect a graph."""
    if type(result) not in (tuple, list) or len(result) != 5:
        raise ArmFailure("a normal five-tuple result is required")
    observation, reward, terminated, truncated, info = result
    if type(observation) is not dict or type(info) is not dict:
        raise ArmFailure("normal observation/info must be dictionaries")
    pairs = []
    for key in sorted(observation):
        value = observation[key]
        if type(key) is not str or type(value) not in (int, float, bool, str, type(None)):
            raise ArmFailure("timing witness accepts returned scalars only")
        if type(value) is float and not math.isfinite(value):
            raise ArmFailure("nonfinite scalar observation")
        pairs.append((key, value))
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise ArmFailure("finite scalar reward required")
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ArmFailure("boolean completion flags required")
    if (info.get("run_id") != run_id or info.get("instance_id") != physical_id
            or info.get("step_id") != expected_step or info.get("seed") != seed
            or info.get("logical_time") != observation.get("logical_time")):
        raise ArmFailure("normal output identity, step, seed or clock differs")
    episode = info.get("episode_id")
    if episode != f"{physical_id}:episode-1":
        raise ArmFailure("unexpected physical episode identity")
    return {"observation": tuple(pairs), "reward": reward, "terminated": terminated,
            "truncated": truncated, "step_id": expected_step, "seed": seed}


def _file_identity(directory):
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ArmFailure("snapshot artifact may not contain links")
        if path.is_file():
            result.append((path.relative_to(directory).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(result)


def _prepare(spec, case, campaign):
    if spec.get("method") not in ("R", "N", "C1") or spec.get("role") not in ("companion", "timing"):
        raise ValueError("unknown break-even method/role")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", spec.get("arm_id", "")):
        raise ValueError("unsafe arm identifier")
    for field in ("prefix_steps", "suffix_steps", "B"):
        if type(spec.get(field)) is not int or spec[field] <= 0:
            raise ValueError("positive declared workload dimensions required")
    if (spec.get("branch_count", spec["B"]) != spec["B"]
            or type(spec.get("delta")) not in (int, float)
            or not math.isfinite(spec["delta"]) or spec["delta"] <= 0):
        raise ValueError("declared branch count or decision interval differs")
    actions = case["action_plan"]
    if (len(actions["prefix"]) != spec["prefix_steps"] or len(actions["branches"]) != spec["B"]
            or len(actions["quantities"]) != spec["B"]
            or any(len(branch) != spec["suffix_steps"] for branch in actions["branches"])):
        raise ValueError("materialized action plan differs from arm dimensions")
    if case.get("config_sha256") != sha(case["config"]):
        raise ValueError("configuration identity differs")
    if any(case["config"].get(key) != spec.get(key) for key in ("K", "S")):
        raise ValueError("materialized K/S differs from the assigned coordinate")
    if case["config"].get("input_seed") != case["seed"]:
        raise ValueError("materialized input seed differs")
    for key in ("config_sha256", "input_identity", "action_identity"):
        if spec.get(key) != case.get(key):
            raise ValueError(f"arm/case {key} differs")
    if case.get("action_identity") != sha(actions):
        raise ValueError("action plan identity differs")
    root = Path(campaign).resolve()
    scratch_root = root / "transient"
    if scratch_root.is_symlink() or (hasattr(scratch_root, "is_junction") and scratch_root.is_junction()):
        raise ValueError("scratch root may not be a link")
    if not scratch_root.is_dir() or scratch_root.resolve().parent != root:
        raise ValueError("campaign must own its immediate transient directory")
    scratch = scratch_root / spec["arm_id"]
    if scratch.exists() or scratch.is_symlink() or (hasattr(scratch, "is_junction") and scratch.is_junction()):
        raise FileExistsError("arm scratch already exists; no overwrite or retry")
    return root, scratch_root, scratch, actions


def execute_arm(spec, case, campaign, budget, modules=None, *, backend_factory=None,
                observer_factory=None, wall_clock=None, cpu_clock=None):
    """Return ``(compact row, ephemeral companion projections)`` for one arm.

    No scientific observer, digest, receipt or serialization executes inside a
    timing endpoint. The common scalar consumer and production validations do.
    Callers must not persist full scalar_witness matrices across the whole study;
    compare them in RAM and retain their digests/final values instead.
    """
    root, scratch_root, scratch, actions = _prepare(spec, case, campaign)
    if backend_factory is None:
        modules = load_runtime_modules() if modules is None else modules
        backend_factory = Backend
    if spec["role"] == "companion" and observer_factory is None:
        if modules is None or "domain" not in modules:
            modules = load_runtime_modules()
        observer_factory = modules["domain"].CompanionObserver
    clock = wall_clock or getattr(budget, "clock", time.perf_counter)
    cpu = cpu_clock or time.process_time
    phases = PhaseClock(clock)
    observer = observer_factory(phase=lambda: phases.current or "unclassified") if spec["role"] == "companion" else None
    projected, witnesses, runtimes, attempted_close, cleanup_errors = [], [], [], set(), []
    operations = Counter({name: 0 for name in ("fresh", "prefix", "capture", "restore", "suffix", "source_close", "branch_close")})
    steps = Counter({"prefix": 0, "suffix": 0})
    status, error, error_phase = "succeeded", None, None
    construction_cleanup, snapshot_bytes, snapshot_identity = True, 0, None
    backend = None

    def close(runtime, phase):
        if id(runtime) in attempted_close:
            return
        attempted_close.add(id(runtime))
        with phases.measure(phase):
            operations[phase] += 1
            try:
                receipt = runtime.close()
                if getattr(receipt, "success", None) is not True:
                    raise ArmFailure("runtime did not confirm cleanup")
            except BaseException as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
                raise

    def advance(runtime, selected, phase, physical_id, branch_rows=None, branch_witness=None):
        with phases.measure(phase):
            operations[phase] += 1
            for offset, action in enumerate(selected, 1):
                budget.check()
                result = runtime.step(action)
                steps[phase] += 1
                expected = offset if phase == "prefix" else spec["prefix_steps"] + offset
                witness = scalar_witness(result, run_id=spec["cell_id"], physical_id=physical_id,
                    expected_step=expected, seed=case["seed"])
                if result[4]["logical_time"] != expected * spec["delta"]:
                    raise ArmFailure("normal output clock shortened the fixed-delta workload")
                if (result[2] or result[3]) and not (phase == "suffix" and offset == len(selected)):
                    raise ArmFailure("declared workload shortened by terminal result")
                if branch_witness is not None:
                    branch_witness.append(witness)
                if observer is not None:
                    events = observer.drain_events()
                    if branch_rows is not None:
                        branch_rows.append({"normal": witness, "state": backend.physical_state(
                            runtime, result[4]["logical_time"]), "events": events})

    budget.check()
    with observer if observer is not None else nullcontext():
        started = clock()
        cpu_started = cpu()
        try:
            with phases.measure("setup"):
                scratch.mkdir(exist_ok=False)
                backend = backend_factory(spec, case, modules)
            if spec["method"] != "R":
                physical_id = spec["arm_id"] + "-source"
                with phases.measure("fresh"):
                    source = backend.fresh(physical_id)
                    operations["fresh"] += 1
                    runtimes.append((source, "source_close"))
                advance(source, actions["prefix"], "prefix", physical_id)
                before = backend.physical_state(source, spec["prefix_steps"] * spec["delta"]) if observer is not None else None
                with phases.measure("capture_write"):
                    artifact = backend.capture_to_file(source, scratch)
                    operations["capture"] += 1
                    snapshot_bytes = artifact["bytes"]
                    if type(snapshot_bytes) is not int or snapshot_bytes <= 0:
                        raise ArmFailure("capture did not report a positive artifact size")
                if observer is not None:
                    if backend.physical_state(source, spec["prefix_steps"] * spec["delta"]) != before:
                        raise ArmFailure("capture changed source physical state")
                    snapshot_identity = _file_identity(scratch)
                budget.sample_transient(scratch)
                if budget.transient_bytes > getattr(budget, "max_transient_bytes", 8 * 1024**2):
                    raise ArmFailure("transient snapshot exceeds declared allowance")
                close(source, "source_close")
            for branch, selected in enumerate(actions["branches"]):
                budget.check()
                physical_id = f"{spec['arm_id']}-branch-{branch}"
                if spec["method"] == "R":
                    with phases.measure("fresh"):
                        runtime = backend.fresh(physical_id)
                        operations["fresh"] += 1
                        runtimes.append((runtime, "branch_close"))
                    advance(runtime, actions["prefix"], "prefix", physical_id)
                else:
                    with phases.measure("restore_read"):
                        runtime = backend.restore_from_file(scratch, branch, physical_id)
                        operations["restore"] += 1
                        runtimes.append((runtime, "branch_close"))
                if observer is not None and observer.drain_events():
                    raise ArmFailure("fresh/restore emitted undrained model events")
                normal_rows, branch_rows = [], []
                witnesses.append(normal_rows)
                if observer is not None:
                    projected.append(branch_rows)
                advance(runtime, selected, "suffix", physical_id, branch_rows, normal_rows)
                if observer is not None:
                    distinguishing = [event for row in branch_rows for event in row["events"]
                                      if event.get("kind") == "demand" and event.get("demand_id") == 67]
                    if (len(distinguishing) != 1 or distinguishing[0].get("fulfilled") != actions["quantities"][branch]
                            or distinguishing[0].get("product") != 0):
                        raise ArmFailure("branch intervention did not produce its actual declared quantity")
                close(runtime, "branch_close")
            if observer is not None and snapshot_identity is not None and _file_identity(scratch) != snapshot_identity:
                raise ArmFailure("branch execution changed source snapshot files")
        except BaseException as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            error_phase = phases.failed or phases.current or "between-phases"
            if error_phase in ("setup", "fresh", "restore_read"):
                construction_cleanup = None
        finally:
            for runtime, phase in runtimes:
                if id(runtime) not in attempted_close:
                    try:
                        close(runtime, phase)
                    except BaseException:
                        pass
            with phases.measure("artifact_cleanup"):
                try:
                    if scratch.exists():
                        remove_owned_directory(scratch, scratch_root)
                    budget.transient_bytes = 0
                except BaseException as exc:
                    cleanup_errors.append(f"artifact cleanup: {type(exc).__name__}: {exc}")
            cpu_elapsed = cpu() - cpu_started
            elapsed = clock() - started
    # The endpoint has stopped. Everything below is scientific/reporting work.
    if sha(case["config"]) != case["config_sha256"] or sha(actions) != case["action_identity"]:
        status, error, error_phase = "failed", "input configuration or action plan changed during execution", "identity"
    counts = None
    if observer is not None:
        keys = set(CALLBACKS) | {"risk_calls", "scenario_stages"}
        keys.update(key for values in observer.counts.values() for key in values)
        counts = {phase: {key: observer.counts.get(phase, {}).get(key, 0) for key in sorted(keys)}
                  for phase in (*PHASES, "unclassified")}
        if any(counts[phase][key] for phase in ("capture_write", "restore_read") for key in CALLBACKS):
            status, error, error_phase = "failed", "capture/restore executed simulated model callbacks", "companion-accounting"
    if cleanup_errors:
        status, error = "failed", f"{error or ''}; cleanup unconfirmed"
    timing = spec["role"] == "timing"
    row = {**spec, "status": status, "error": error, "error_phase": error_phase,
        "seed": case["seed"], "config_sha256": case["config_sha256"],
        "endpoint_revision": ENDPOINT, "workflow_wall_seconds": elapsed if timing else None,
        "workflow_cpu_seconds": cpu_elapsed if timing else None,
        "phase_seconds": dict(phases.seconds) if timing else None,
        "phase_invocations": dict(phases.invocations),
        "unclassified_seconds": elapsed - sum(phases.seconds.values()) if timing else None,
        "operations": dict(operations), "step_results_returned": dict(steps),
        "prefixes_executed": operations["prefix"], "restores_executed": operations["restore"],
        "suffix_steps_executed": steps["suffix"], "expected_suffix_steps": spec["B"] * spec["suffix_steps"],
        "scalar_witness": witnesses, "scalar_witness_sha256": sha(witnesses),
        "scalar_branch_sha256": [sha(branch) for branch in witnesses],
        "branch_projection_sha256": [sha(branch) for branch in projected] if observer is not None else None,
        "work_counts": counts, "snapshot_bytes": snapshot_bytes,
        "cleanup_confirmed": False if cleanup_errors else construction_cleanup, "cleanup_errors": cleanup_errors,
        "full_trace_exact_claim": False, "event_counting_enabled": observer is not None,
        "memory_complete": False, "whole_lifetime_peak_bytes": None,
        "persistence": "cached-local-filesystem-no-fsync",
        "endpoint": "backend construction through owned runtime close and snapshot disposition; scientific receipts excluded"}
    try:
        budget.write(root / "receipts" / f"{spec['arm_id']}.json", {
            "arm_id": spec["arm_id"], "status": status, "role": spec["role"],
            "operations": dict(operations), "scalar_witness_sha256": row["scalar_witness_sha256"]})
    except BaseException as exc:
        row.update(status="failed", error=f"{error or ''}; receipt: {type(exc).__name__}: {exc}", error_phase="receipt")
    return row, projected
