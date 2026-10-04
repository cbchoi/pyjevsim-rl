"""New-model engineering case; no human productivity inference or timing claim."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

from pyjevsim_bridge.rl.continuation import (
    BranchContext, CaptureRequest, ContinuationCoordinator, ContinuationError,
    ContinuationRegistry, ResetRequest)
from pyjevsim_bridge.rl.continuation.contracts import canonical_bytes, digest
from pyjevsim_bridge.rl.continuation.envelope import encode_snapshot
from . import inventory as m
from . import inventory_native as native
from . import inventory_maintenance as maintenance
from .inventory_adapter import make_bundle

DELTA = .25
SUFFIX_STEPS = 8
MANIFEST = {"schema": "inventory-engineering-transfer-v1", "versions": [1, 2],
    "configuration_variants": [0, 1, 2], "cuts_steps": [0, 3, 4], "first_branch_orders": [0, 4, 9],
    "suffix_steps": SUFFIX_STEPS, "delta": DELTA, "primary_branch_cells": 54,
    "primary_branch_executions": 162, "repeat_A_diagnostic_executions": 36,
    "prefix_actions": "order5 at index2 (time .5, due1.0); otherwise order0", "future_actions": "first branch order; then seven order0",
    "methods": ["R", "N", "C1"], "oracle": "separate explicit inventory event loop",
    "study_kind": "AI-authored engineering case; not blinded or prospectively held out",
    "deterministic_variants_not_independent_random_samples": True}
POLICY = {"policy_sha256": digest({"inventory": "fixed-orders-v1"}), "policy_version": 0,
          "feature_contract_sha256": digest({"observation": "inventory"})}


def prefix_actions(cut):
    return [{"order": 5 if index == 2 else 0} for index in range(cut)]


def branch_actions(quantity):
    return [{"order": quantity}] + [{"order": 0} for _ in range(SUFFIX_STEPS - 1)]


def oracle(config, actions, delta=DELTA):
    """Independent event algorithm; no model/adapter/serializer callback used.

    Replenishment precedes demand on ties, matching the specified native
    int-then-ext confluence. Actions enter only after the prior cut was drained.
    """
    cfg = copy.deepcopy(config)
    stock, fulfilled, lost, ordered, received = cfg["initial_stock"], 0, 0, 0, 0
    pending, events, rows, cursor, last_fulfilled, last_penalty = [], [], [], 0, 0, 0.
    now = 0.
    for action in actions:
        quantity = action["order"]
        events.append({"time": now, "kind": "action", "quantity": quantity})
        if quantity:
            pending.append({"due": now + cfg["lead_time"], "quantity": quantity})
            ordered += quantity
        end = now + delta
        while True:
            receipt_time = min((row["due"] for row in pending), default=float("inf"))
            demand_time = cfg["demands"][cursor]["at"] if cursor < len(cfg["demands"]) else float("inf")
            event_time = min(receipt_time, demand_time)
            if event_time > end:
                break
            if receipt_time <= demand_time:
                for row in pending:
                    if row["due"] == event_time:
                        stock += row["quantity"]
                        received += row["quantity"]
                        events.append({"time": event_time, "kind": "replenish", "quantity": row["quantity"]})
                pending = [row for row in pending if row["due"] != event_time]
            else:
                quantity = cfg["demands"][cursor]["quantity"]
                filled = min(stock, quantity)
                stock -= filled
                fulfilled += filled
                lost += quantity - filled
                cursor += 1
                events.append({"time": event_time, "kind": "demand", "quantity": quantity,
                               "fulfilled": filled, "lost": quantity - filled})
        now = end
        observation = {"version": cfg["version"], "logical_time": now, "stock": stock,
            "fulfilled": fulfilled, "lost": lost, "received": received, "ordered": ordered,
            "pending": copy.deepcopy(pending), "events": copy.deepcopy(events), "demand_cursor": cursor}
        reward = float(fulfilled - last_fulfilled)
        last_fulfilled = fulfilled
        if cfg["version"] == 2:
            penalty = lost * cfg["penalty_per_unit"]
            observation["cumulative_penalty"] = penalty
            reward -= penalty - last_penalty
            last_penalty = penalty
        rows.append({"observation": observation, "reward": reward, "terminated": False, "truncated": False})
    return rows


def _sampling(seed, segment="prefix", branch_id="prefix"):
    return {"domain": "pyjevsim-live-branch-v1", "phase": "engineering-transfer", "master": seed,
            "segment": segment, "logical_branch_id": branch_id, "run_id": "inventory-study",
            "generation": 0, "worker_id": "single", "episode_id": "one", "sampling_seed": seed}


class CommonBackend:
    def __init__(self, version, seed):
        self.registry = ContinuationRegistry()
        self.bundle = make_bundle(version)
        self.registry.register(self.bundle)
        self.coordinator = ContinuationCoordinator(self.registry)
        self.seed = seed

    def fresh(self, cfg):
        return self.coordinator.create_fresh(ResetRequest(self.bundle.profile.profile_id, cfg, self.seed,
            "inventory-C1", "inventory-study", DELTA, 100, POLICY, _sampling(self.seed)))

    def capture(self, runtime, cut):
        return self.coordinator.capture(runtime, CaptureRequest(runtime.profile_id, cut,
            "inventory-family", "inventory-prefix", POLICY))

    def restore(self, snapshot, branch_id):
        return self.coordinator.restore(snapshot, BranchContext("inventory-family", "inventory-prefix", branch_id,
            POLICY, _sampling(self.seed, "suffix", branch_id), f"inventory-{branch_id}"))


def _parts(runtime):
    return runtime._parts if hasattr(runtime, "_parts") else runtime


def _close(runtime):
    receipt = runtime.close()
    if not receipt.success:
        raise RuntimeError(f"inventory cleanup failed: {receipt}")


def _advance(runtime, actions):
    rows = []
    for action in actions:
        observation, reward, terminated, truncated, _info = runtime.step(action)
        rows.append({"observation": copy.deepcopy(observation), "reward": reward,
                     "terminated": terminated, "truncated": truncated})
    return rows


class CallbackCounter:
    """Only a diagnostic callback vector, not a count of unique events."""
    def __enter__(self):
        self.previous = sys.getprofile()
        self.count = 0
        self.codes = {getattr(cls, name).__code__ for cls in (m.DemandSource, m.InventoryStock, maintenance.PenaltyStock)
                      for name in ("int_trans", "ext_trans", "output", "con_trans")}
        sys.setprofile(self.profile)
        return self

    def profile(self, frame, event, _argument):
        if event == "call" and frame.f_code in self.codes:
            self.count += 1

    def __exit__(self, *_exc):
        sys.setprofile(self.previous)


def _check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("research workflow deadline reached")


def _storage(directory, max_bytes, progress):
    measured = sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())
    progress["observed_transient_peak_bytes"] = max(progress.get("observed_transient_peak_bytes", 0), measured)
    if max_bytes is not None and measured > max_bytes:
        raise RuntimeError("inventory transient storage exceeds remaining research allowance")


def run_cell(version, variant, cut, deadline=None, scratch_root=None, max_bytes=None, progress=None):
    if progress is None:
        progress = {}
    progress.update(primary_attempted=0, primary_succeeded=0, diagnostic_attempted=0,
                    diagnostic_succeeded=0, rows=[], observed_transient_peak_bytes=0)
    cfg, seed = m.configuration(version, variant), variant
    prefix = prefix_actions(cut)
    backend = CommonBackend(version, seed)
    sources = {}
    rows, primary_exec, diagnostic_exec = [], 0, 0
    try:
        sources["N"] = native.create_native(cfg, seed)
        sources["C1"] = backend.fresh(cfg)
        for source in sources.values():
            _advance(source, prefix)
        cut_views = {method: copy.deepcopy(_parts(runtime).graph.observe(cut * DELTA))
                     for method, runtime in sources.items()}
        if cut_views["N"] != cut_views["C1"]:
            raise AssertionError("inventory cut differs before capture")
        expected_pending = [{"due": 1., "quantity": 5}] if cut == 3 else []
        if cut_views["N"]["pending"] != expected_pending:
            raise AssertionError("predeclared pending-order cut was not exercised")
        with tempfile.TemporaryDirectory(prefix="inventory-research-", dir=scratch_root) as directory:
            with CallbackCounter() as capture_callbacks:
                native.save_native(sources["N"], directory)
                snapshot = backend.capture(sources["C1"], cut)
            _storage(directory, max_bytes, progress)
            if capture_callbacks.count:
                raise AssertionError("capture executed domain callbacks")
            baseline_traces = {}
            for quantity in MANIFEST["first_branch_orders"]:
                _check_deadline(deadline)
                actions = branch_actions(quantity)
                expected = oracle(cfg, prefix + actions)[cut:]
                traces, restore_counts, shared_aliases = {}, {}, {}
                for method in MANIFEST["methods"]:
                    _check_deadline(deadline)
                    progress["primary_attempted"] += 1
                    progress["active_primary"] = {"method": method, "branch_order": quantity}
                    if method == "R":
                        runtime = native.create_native(cfg, seed)
                    else:
                        with CallbackCounter() as restore_callbacks:
                            runtime = (native.load_native(directory) if method == "N"
                                       else backend.restore(snapshot, f"order-{quantity}"))
                        restore_counts[method] = restore_callbacks.count
                    try:
                        if method == "R":
                            _advance(runtime, prefix)
                        _storage(directory, max_bytes, progress)
                        parts = _parts(runtime)
                        shared_aliases[method] = all(leaf.config is parts.graph.config for leaf in parts.graph.leaves())
                        if parts.engine.global_time != cut * DELTA or parts.graph.observe(cut * DELTA) != cut_views["N"]:
                            raise AssertionError("restored clock/model values differ")
                        traces[method] = _advance(runtime, actions)
                    finally:
                        _close(runtime)
                    primary_exec += 1
                    progress["primary_succeeded"] += 1
                    progress["active_primary"] = None
                exact = all(traces[method] == expected for method in MANIFEST["methods"])
                if any(restore_counts.values()) or not all(shared_aliases.values()):
                    raise AssertionError("restore executed domain callbacks or broke config aliases")
                first_event = traces["C1"][0]["observation"]["events"][len(cut_views["N"]["events"])]
                if first_event != {"time": cut * DELTA, "kind": "action", "quantity": quantity}:
                    raise AssertionError("new action was not injected at restored time")
                if quantity == 0:
                    baseline_traces = copy.deepcopy(traces)
                row = {"version": version, "variant": variant, "cut_steps": cut, "branch_order": quantity,
                    "exact": exact, "oracle_exact": exact, "trace_sha256": digest({"steps": expected}),
                    "restored_time": cut * DELTA, "injected_action_time": first_event["time"],
                    "pending_at_cut": copy.deepcopy(cut_views["N"]["pending"]),
                    "capture_callback_calls": capture_callbacks.count, "restore_callback_calls": restore_counts,
                    "shared_configuration_aliases": shared_aliases,
                    "final_stock": expected[-1]["observation"]["stock"],
                    "final_fulfilled": expected[-1]["observation"]["fulfilled"],
                    "final_lost": expected[-1]["observation"]["lost"],
                    "suffix_reward": sum(step["reward"] for step in expected)}
                rows.append(row)
                progress["rows"].append(row)
                if not exact:
                    raise AssertionError("inventory replay/native/C1/domain-oracle traces differ")
            # A/B/C/A is diagnostic and is not silently added to the primary denominator.
            for method in ("N", "C1"):
                _check_deadline(deadline)
                progress["diagnostic_attempted"] += 1
                runtime = native.load_native(directory) if method == "N" else backend.restore(snapshot, "order-0-repeat")
                try:
                    if _advance(runtime, branch_actions(0)) != baseline_traces[method]:
                        raise AssertionError("inventory branch isolation failed")
                    diagnostic_exec += 1
                    progress["diagnostic_succeeded"] += 1
                finally:
                    _close(runtime)
            for method, source in sources.items():
                if _parts(source).graph.observe(cut * DELTA) != cut_views[method]:
                    raise AssertionError("captured inventory source mutated")
            if len({(row["final_stock"], row["final_fulfilled"], row["final_lost"]) for row in rows}) < 2:
                raise AssertionError("inventory branch intervention effect is vacuous")
            return {"rows": rows, "primary_executions": primary_exec, "diagnostic_executions": diagnostic_exec,
                    "branch_isolation": True, "source_invariance": True, "physical_intervention_effect": True}
    finally:
        for source in sources.values():
            _close(source)


def negative_controls(deadline=None, receipts=None):
    if receipts is None:
        receipts = []
    _check_deadline(deadline)
    cfg = m.configuration(2, 0)
    backend = CommonBackend(2, 0)
    runtime = backend.fresh(cfg)
    try:
        _advance(runtime, prefix_actions(4))
        original = json.loads(backend.capture(runtime, 4).data)
        mutations = {
            "missing_penalty_accumulator": lambda p: p["model_state"]["values"]["inventory-stock"].pop("cumulative_penalty"),
            "reset_penalty_accumulator": lambda p: p["model_state"]["values"]["inventory-stock"].update(cumulative_penalty=0.),
            "reset_reward_penalty_baseline": lambda p: p["boundary_state"]["reward_state"].update(last_penalty=0.),
        }
        for name, mutate in mutations.items():
            _check_deadline(deadline)
            payload = copy.deepcopy(original)
            del payload["integrity"]
            mutate(payload)
            try:
                encode_snapshot(payload, registry=backend.registry)
            except (ContinuationError, ValueError) as exc:
                receipts.append({"control": name, "method": "C1", "detected": True,
                                 "error": str(exc), "kind": "schema/composition rejection before allocation"})
            else:
                raise AssertionError(f"inventory C1 missed {name}")
        other = CommonBackend(1, 0)
        _check_deadline(deadline)
        try:
            unexpected = other.restore(canonical_bytes(original), "wrong-version")
        except ContinuationError as exc:
            receipts.append({"control": "v2_snapshot_in_v1_registry", "method": "C1", "detected": True,
                             "error": str(exc), "kind": "registered profile incompatibility"})
        else:
            _close(unexpected)
            raise AssertionError("cross-version restore accepted")
    finally:
        _close(runtime)
    # N has model-specific checks too; do not weaken it to favor C1.
    for name in ("reset_penalty_accumulator", "reset_reward_penalty_baseline"):
        _check_deadline(deadline)
        runtime = native.create_native(cfg)
        try:
            _advance(runtime, prefix_actions(4))
            if name == "reset_penalty_accumulator":
                runtime.graph.stock.cumulative_penalty = 0.
            else:
                runtime.reward_state["last_penalty"] = 0.
            try:
                native._assert_cut(runtime)
            except ValueError as exc:
                receipts.append({"control": name, "method": "N", "detected": True,
                                 "error": str(exc), "kind": "handwritten live-state validation"})
            else:
                raise AssertionError(f"inventory N missed {name}")
        finally:
            _close(runtime)
    return receipts


def source_inventory():
    roles = {"inventory.py": "shared domain + binding callbacks (count once)",
             "inventory_maintenance.py": "V2 domain/reward extension",
             "inventory_adapter.py": "C1 domain obligations/composition/profile declaration",
             "inventory_native.py": "N native journal plus explicit engine/graph/RL repairs",
             "transfer.py": "common research evaluation, not method integration"}
    rows = []
    for filename, role in roles.items():
        path = Path(__file__).with_name(filename)
        data = path.read_bytes()
        tree = ast.parse(data)
        rows.append({"file": f"bench/research/{filename}", "role": role,
                     "sha256": hashlib.sha256(data).hexdigest(), "lines": len(data.splitlines()),
                     "function_definitions": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)),
                     "class_definitions": sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))})
    return rows


def run_transfer(output_directory=None, deadline=None, scratch_root=None, max_bytes=None):
    """Run sequentially; return partial receipts on failure, never retry a cell.

    output_directory is accepted for interface symmetry, but this function does
    not write permanent results: the root research entry point owns persistence.
    """
    report = {"manifest": copy.deepcopy(MANIFEST), "manifest_sha256": digest(MANIFEST),
        "status": "running", "study_admission": False, "rows": [], "completed_case_groups": 0,
        "primary_executions": 0, "diagnostic_executions": 0, "negative_controls": [],
        "primary_attempted": 0, "diagnostic_attempted": 0, "observed_transient_peak_bytes": 0,
        "transient_measurement": "native capture and branch boundaries; not a hard lifetime-peak proof",
        "source_inventory": source_inventory(), "human_productivity_measured": False,
        "core_modified_by_transfer": False,
        "limitations": ["same AI team developed model and both integrations with iterative testing",
            "no human time, independent participants or causal effort reduction measured",
            "code inventory is structural description, not a developer effort estimator",
            "deterministic inventory demand; no inventory-model RNG claim",
            "finite test cases, not a universal proof or a new empirical runtime benchmark"]}
    current = None
    try:
        for version in MANIFEST["versions"]:
            for variant in MANIFEST["configuration_variants"]:
                for cut in MANIFEST["cuts_steps"]:
                    _check_deadline(deadline)
                    current = {"version": version, "variant": variant, "cut_steps": cut}
                    progress = {}
                    try:
                        run_cell(version, variant, cut, deadline, scratch_root, max_bytes, progress)
                    finally:
                        report["rows"].extend(progress.get("rows", []))
                        report["primary_executions"] += progress.get("primary_succeeded", 0)
                        report["diagnostic_executions"] += progress.get("diagnostic_succeeded", 0)
                        report["primary_attempted"] += progress.get("primary_attempted", 0)
                        report["diagnostic_attempted"] += progress.get("diagnostic_attempted", 0)
                        report["observed_transient_peak_bytes"] = max(report["observed_transient_peak_bytes"],
                            progress.get("observed_transient_peak_bytes", 0))
                        report["last_attempt"] = progress.get("active_primary")
                    report["completed_case_groups"] += 1
        _check_deadline(deadline)
        negative_controls(deadline, report["negative_controls"])
        report.update(status="completed", study_admission=True,
            branch_isolation=True, source_invariance=True, physical_intervention_effect=True)
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc, TimeoutError) else "failed",
            failure={"case": current, "type": type(exc).__name__, "message": str(exc)},
            partially_attempted_group_retained_in_execution_denominators=True)
    report["exact_cells"] = {"planned": 54, "complete": len(report["rows"]),
                              "exact": sum(row["exact"] for row in report["rows"])}
    report["denominators"] = {"planned_case_groups": 18, "completed_case_groups": report["completed_case_groups"],
        "planned_primary_executions": 162, "attempted_primary_executions": report["primary_attempted"],
        "succeeded_primary_executions": report["primary_executions"],
        "failed_primary_executions": report["primary_attempted"] - report["primary_executions"],
        "unexecuted_primary_executions": 162 - report["primary_attempted"],
        "planned_repeat_diagnostics": 36, "attempted_repeat_diagnostics": report["diagnostic_attempted"],
        "completed_repeat_diagnostics": report["diagnostic_executions"]}
    return report
