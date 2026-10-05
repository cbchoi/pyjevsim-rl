"""Bounded exploratory inventory decisions, not policy training or confirmation.

The existing inventory transition model is unchanged.  All methods see one
planning tape and identical candidate order.  Independent future tapes share
only the observed prefix and are unavailable to the selector.  Native replay
checks the held-out domain oracle outside every decision timing endpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from random import Random
import tempfile
import time

SCHEMA = "inventory-decision-utility-v1"
METHODS = ("R", "N", "C1", "C1A")
MODES = ("fixed-candidates", "fixed-budget")
DELTA, PREFIX_STEPS, SUFFIX_STEPS = .25, 4, 8
DEFAULT_COSTS = {"procurement": 1., "terminal_holding": .5, "stockout": 4.}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def namespace_seed(label, master):
    return int.from_bytes(hashlib.sha256(f"{SCHEMA}:{label}:{master}".encode()).digest()[:8], "big")


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def validate_costs(costs):
    if type(costs) is not dict or set(costs) != set(DEFAULT_COSTS):
        raise ValueError("closed procurement/terminal_holding/stockout costs required")
    return {key: _positive(value, key) for key, value in costs.items()}


def _future(seed):
    rng = Random(seed)
    return [{"at": at, "quantity": rng.randrange(1, 7)} for at in (1.5, 2., 2.75)]


def prepare_family(master, candidates=8):
    """Return selector-visible information only, never held-out seeds/tapes."""
    _integer(master, "master", 0, (1 << 63) - 1)
    _integer(candidates, "candidates", 2, 33)
    seeds = {key: namespace_seed(key, master) for key in ("prefix", "planning", "candidate-order")}
    rng = Random(seeds["prefix"])
    prefix = [{"at": at, "quantity": rng.randrange(1, 5)} for at in (.25, .75)]
    config = {"version": 1, "initial_stock": 8, "lead_time": .5,
              "demands": prefix + _future(seeds["planning"])}
    quantities = [index * 32 // (candidates - 1) for index in range(candidates)]
    Random(seeds["candidate-order"]).shuffle(quantities)
    return {"family_id": f"du-{master}", "master": master, "seeds": seeds,
            "config": config, "candidate_order": quantities,
            "prefix_identity": digest(prefix), "planning_identity": digest(config),
            "candidate_order_identity": digest(quantities)}


def evaluation_panel(family, count=8, *, namespace="heldout"):
    _integer(count, "evaluation_rollouts", 1, 128)
    if type(namespace) is not str or not namespace or namespace in ("prefix", "planning", "candidate-order"):
        raise ValueError("independent held-out namespace required")
    configs, seeds = [], []
    for index in range(count):
        seed = namespace_seed(f"{namespace}:{index}", family["master"])
        if seed in family["seeds"].values():
            raise ValueError("seed namespace collision")
        cfg = copy.deepcopy(family["config"])
        cfg["demands"] = cfg["demands"][:2] + _future(seed)
        configs.append(cfg)
        seeds.append(seed)
    return {"configs": configs, "seeds": seeds, "identity": digest(configs),
            "namespace": namespace, "prefix_identity": family["prefix_identity"]}


def actions(quantity):
    return [{"order": 0}] * PREFIX_STEPS + [{"order": quantity}] + [{"order": 0}] * (SUFFIX_STEPS - 1)


def loss(observation, prefix_observation, costs=None):
    costs = DEFAULT_COSTS if costs is None else costs
    return (costs["procurement"] * (observation["ordered"] - prefix_observation["ordered"])
            + costs["terminal_holding"] * observation["stock"]
            + costs["stockout"] * (observation["lost"] - prefix_observation["lost"]))


def _sampling(seed, segment="prefix", branch="prefix"):
    return {"domain": "pyjevsim-live-branch-v1", "phase": SCHEMA, "master": seed,
            "segment": segment, "logical_branch_id": branch, "run_id": "inventory-study",
            "generation": 0, "worker_id": "serial", "episode_id": "one", "sampling_seed": seed}


def runtime_dependencies():
    # Imports occur before the decision endpoint, equally for all methods.
    from . import inventory_native
    from .inventory_adapter import make_bundle
    from .transfer import oracle
    from pyjevsim_bridge.rl import continuation
    return {"native": inventory_native, "bundle": make_bundle, "cc": continuation, "oracle": oracle}


class InventoryBackend:
    def __init__(self, method, family, modules):
        self.method, self.family, self.modules = method, family, modules
        self.policy = {"policy_sha256": family["candidate_order_identity"], "policy_version": 0,
                       "feature_contract_sha256": digest({"inventory": "decision-loss-v1"})}
        if method in ("C1", "C1A"):
            cc = modules["cc"]
            self.bundle = modules["bundle"](1)
            self.registry = cc.ContinuationRegistry()
            self.registry.register(self.bundle)
            self.coordinator = cc.ContinuationCoordinator(self.registry, execution_profile=(
                "admitted-runtime-v1" if method == "C1A" else "strict-v1"))

    def _qualified(self, runtime):
        expected = "admitted-runtime-v1" if self.method == "C1A" else "strict-v1"
        if getattr(runtime, "execution_profile", None) != expected:
            _close(runtime)
            raise RuntimeError(f"runtime execution profile is not {expected}")
        return runtime

    def fresh(self, identity):
        cfg, seed = self.family["config"], self.family["master"]
        if self.method in ("R", "N"):
            return self.modules["native"].create_native(cfg, seed, DELTA, 100, identity)
        cc = self.modules["cc"]
        return self._qualified(self.coordinator.create_fresh(cc.ResetRequest(self.bundle.profile.profile_id,
            cfg, seed, identity, "inventory-study", DELTA, 100, self.policy, _sampling(seed))))

    def capture(self, runtime, directory):
        if self.method == "N":
            self.modules["native"].save_native(runtime, directory)
        else:
            cc = self.modules["cc"]
            captured = self.coordinator.capture(runtime, cc.CaptureRequest(runtime.profile_id,
                PREFIX_STEPS, self.family["family_id"], "decision-prefix", self.policy))
            with (directory / "cut.json").open("xb") as stream:
                stream.write(captured.data)

    def restore(self, directory, identity):
        if self.method == "N":
            return self.modules["native"].load_native(directory, identity)
        cc = self.modules["cc"]
        return self._qualified(self.coordinator.restore((directory / "cut.json").read_bytes(), cc.BranchContext(
            self.family["family_id"], "decision-prefix", identity, self.policy,
            _sampling(self.family["master"], "suffix", identity), identity)))


def _close(runtime):
    receipt = runtime.close()
    if getattr(receipt, "success", None) is not True:
        raise RuntimeError("runtime cleanup not confirmed")


def _advance(runtime, selected, check):
    rows = []
    for action in selected:
        check()
        observation, reward, terminated, truncated, _ = runtime.step(action)
        if terminated or truncated:
            raise RuntimeError("declared decision horizon ended early")
        rows.append({"observation": observation, "reward": reward,
                     "terminated": terminated, "truncated": truncated})
    return rows


def execute_decision(method, family, *, mode="fixed-candidates", budget_seconds=1.,
                     scratch_root=None, costs=None, modules=None, backend_factory=InventoryBackend,
                     clock=time.perf_counter, check=lambda: None, storage_check=lambda _: None):
    """One actual decision; returns compact receipt plus ephemeral normal outputs.

    A deadline is a soft observation boundary, not an OS timeout.  An in-flight
    candidate finishes and closes, is marked late, and cannot alter the selection.
    Campaign ``check`` may interrupt between steps; all owned runtimes still close.
    """
    if method not in METHODS or mode not in MODES:
        raise ValueError("unknown decision method/mode")
    budget_seconds = _positive(budget_seconds, "decision budget")
    costs = validate_costs(dict(DEFAULT_COSTS) if costs is None else costs)
    modules = runtime_dependencies() if modules is None else modules
    # Full family has no held-out values; a backend cannot inspect evaluation data.
    check()
    started = clock()
    deadline = started + budget_seconds if mode == "fixed-budget" else math.inf
    operations = {key: 0 for key in ("fresh", "prefix", "capture", "restore", "suffix", "close")}
    receipt = {"family_id": family["family_id"], "method": method, "mode": mode,
        "execution_profile": "admitted-runtime-v1" if method == "C1A" else ("strict-v1" if method == "C1" else "native-fast"),
        "budget_seconds": budget_seconds if mode == "fixed-budget" else None,
        "candidate_order": list(family["candidate_order"]), "candidates": [], "selected_quantity": None,
        "selected_predicted_loss": None, "selection_timestamp_seconds": None,
        "operations": operations, "status": "completed", "cleanup_confirmed": True}
    traces, source, active = {}, None, None
    decision_end = started
    try:
        with tempfile.TemporaryDirectory(prefix="decision-", dir=scratch_root) as directory:
            directory = Path(directory)
            try:
                backend = backend_factory(method, family, modules)
                cut = None
                if method != "R":
                    check()
                    source = backend.fresh(f"{family['family_id']}-{method}-source")
                    operations["fresh"] += 1
                    prefix = _advance(source, actions(0)[:PREFIX_STEPS], check)
                    operations["prefix"] += 1
                    cut = prefix[-1]["observation"]
                    backend.capture(source, directory)
                    operations["capture"] += 1
                    storage_check(directory)
                    try:
                        _close(source)
                        operations["close"] += 1
                    except Exception:
                        receipt["cleanup_confirmed"] = False
                        raise
                    finally:
                        source = None
                for index, quantity in enumerate(family["candidate_order"]):
                    check()
                    if clock() >= deadline:
                        break
                    identity = f"{family['family_id']}-{method}-q{quantity}"
                    candidate = {"quantity": quantity, "order_index": index, "status": "running"}
                    receipt["candidates"].append(candidate)
                    if method == "R":
                        active = backend.fresh(identity)
                        operations["fresh"] += 1
                        prefix = _advance(active, actions(0)[:PREFIX_STEPS], check)
                        operations["prefix"] += 1
                        cut = prefix[-1]["observation"]
                    else:
                        active = backend.restore(directory, identity)
                        operations["restore"] += 1
                    rows = _advance(active, actions(quantity)[PREFIX_STEPS:], check)
                    operations["suffix"] += 1
                    predicted = loss(rows[-1]["observation"], cut, costs)
                    traces[quantity] = rows
                    try:
                        _close(active)
                        operations["close"] += 1
                    except Exception:
                        receipt["cleanup_confirmed"] = False
                        raise
                    finally:
                        active = None
                    improves = (receipt["selected_predicted_loss"] is None
                                or predicted < receipt["selected_predicted_loss"])
                    # Observe the boundary after scoring, comparison and branch
                    # close. The timestamp/receipt assignment is bookkeeping.
                    available = clock()
                    eligible = available <= deadline
                    # Ties use the first candidate in the common, preassigned order.
                    if eligible and improves:
                        receipt.update(selected_quantity=quantity, selected_predicted_loss=predicted,
                                       selection_timestamp_seconds=available - started)
                    candidate.update(status="eligible" if eligible else "late", predicted_loss=predicted,
                                     available_seconds=available - started)
                    if not eligible:
                        break
                decision_end = clock()
            finally:
                for runtime in (active, source):
                    if runtime is not None:
                        try:
                            _close(runtime)
                            operations["close"] += 1
                        except Exception:
                            receipt["cleanup_confirmed"] = False
                            raise
    except Exception as exc:
        receipt.update(status="stopped" if isinstance(exc, TimeoutError) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        if receipt["candidates"] and receipt["candidates"][-1]["status"] == "running":
            receipt["candidates"][-1]["status"] = "failed"
        decision_end = clock()
    finished = clock()
    receipt.update(decision_wall_seconds=decision_end - started,
        cleanup_wall_seconds=max(0., finished - decision_end), workflow_wall_seconds=finished - started,
        overrun_seconds=max(0., finished - deadline) if mode == "fixed-budget" else 0.,
        completed_before_deadline=sum(row["status"] == "eligible" for row in receipt["candidates"]),
        unattempted_candidates=len(family["candidate_order"]) - len(receipt["candidates"]),
        no_decision=receipt["selected_quantity"] is None,
        outcome_digests={str(q): digest(rows) for q, rows in traces.items()})
    return receipt, traces


def evaluate_selections(family, selected_quantities, panel, *, modules=None, costs=None,
                        check=lambda: None, clock=time.perf_counter, progress=None):
    """Independent future empirical regret; not clairvoyant per-rollout regret.

    The oracle is the best *single action* over this held-out panel.  Every
    selected action is independently replayed natively on all panel futures.
    The panel is never used to generate or change the selected action.
    """
    modules = runtime_dependencies() if modules is None else modules
    costs = validate_costs(dict(DEFAULT_COSTS) if costs is None else costs)
    started = clock()
    quantities = family["candidate_order"]
    selected = sorted(set(q for q in selected_quantities if q is not None))
    if any(q not in quantities for q in selected):
        raise ValueError("selected quantity outside declared candidates")
    expected_prefix = family["config"]["demands"][:2]
    if any(cfg["demands"][:2] != expected_prefix for cfg in panel["configs"]):
        raise ValueError("held-out evaluation changed observed prefix")
    progress = {} if progress is None else progress
    progress.update(status="running", panel_identity=panel["identity"],
        planned_oracle_cells=len(quantities) * len(panel["configs"]), completed_oracle_cells=0,
        planned_native_trajectories=len(selected) * len(panel["configs"]),
        attempted_native_trajectories=0, native_trajectories_checked=0)
    losses = {q: [] for q in quantities}
    native_checks = 0
    for index, config in enumerate(panel["configs"]):
        for quantity in quantities:
            check()
            expected = modules["oracle"](config, actions(quantity), DELTA)
            losses[quantity].append(loss(expected[-1]["observation"], expected[PREFIX_STEPS-1]["observation"], costs))
            progress["completed_oracle_cells"] += 1
            if quantity in selected:
                progress["attempted_native_trajectories"] += 1
                runtime = modules["native"].create_native(config, family["master"], DELTA, 100,
                    f"heldout-{index}-q{quantity}")
                try:
                    actual = _advance(runtime, actions(quantity), check)
                    if actual != expected:
                        raise AssertionError("held-out native outcomes disagree with independent oracle")
                    native_checks += 1
                    progress["native_trajectories_checked"] = native_checks
                finally:
                    _close(runtime)
    means = {q: sum(values) / len(values) for q, values in losses.items()}
    best = min(quantities, key=lambda q: (means[q], quantities.index(q)))
    progress.update({"status": "completed", "panel_identity": panel["identity"], "evaluation_rollouts": len(panel["configs"]),
        "oracle_quantity": best, "oracle_loss": means[best],
        "oracle_definition": "best fixed candidate in independent held-out panel, not per-future clairvoyance",
        "mean_loss_by_quantity": {str(q): means[q] for q in quantities},
        "selections": {str(q): {"heldout_loss": means[q], "regret": means[q]-means[best],
                                 "oracle_match": means[q] == means[best]} for q in selected},
        "native_oracle_exact": True, "native_trajectories_checked": native_checks,
        "native_oracle_scope": "selected actions only, each on every held-out future",
        "evaluation_wall_seconds": clock() - started})
    return progress


def run_decision_study(output_directory=None, *, families=3, candidates=8, decision_budget_seconds=1.,
                       evaluation_rollouts=8, max_seconds=120., max_storage_bytes=8 * 1024**2,
                       base_seed=984000, methods=METHODS, costs=None, clock=time.perf_counter,
                       backend_factory=InventoryBackend, modules=None):
    """Run sequentially and return compact data; caller owns durable persistence.

    Only owned TemporaryDirectory snapshots are created and cleaned.  Storage is
    observed at capture/report boundaries, not guaranteed at every instant.  No
    failed arm is retried and previous study results are never read or modified.
    """
    _integer(families, "families", 1, 24)
    _integer(candidates, "candidates", 2, 33)
    _integer(evaluation_rollouts, "evaluation_rollouts", 1, 128)
    _integer(max_storage_bytes, "max_storage_bytes", 1024, 1024**3)
    _integer(base_seed, "base_seed", 0, (1 << 63) - 25)
    decision_budget_seconds = _positive(decision_budget_seconds, "decision budget")
    max_seconds = _positive(max_seconds, "study time")
    costs = validate_costs(dict(DEFAULT_COSTS) if costs is None else costs)
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError("unique supported methods required")
    if output_directory is not None:
        output_directory = Path(output_directory)
        if not output_directory.is_dir():
            raise ValueError("caller must provide an existing output/scratch directory")
    modules = runtime_dependencies() if modules is None else modules
    source_paths = [Path(__file__)]
    for module in (modules.get("native"),):
        if getattr(module, "__file__", None):
            source_paths.append(Path(module.__file__))
    report = {"schema": SCHEMA, "status": "running", "study_admission": False,
        "manifest": {"families": families, "candidates": candidates, "methods": list(methods), "modes": list(MODES),
            "decision_budget_seconds": decision_budget_seconds, "evaluation_rollouts": evaluation_rollouts,
            "base_seed": base_seed, "max_seconds": max_seconds, "max_storage_bytes": max_storage_bytes,
            "costs": costs, "delta": DELTA, "prefix_steps": PREFIX_STEPS, "suffix_steps": SUFFIX_STEPS,
            "future_law": "three independent discrete uniform quantities 1..6 at1.5/2.0/2.75",
            "candidate_ties": "first in common preassigned order", "decision_deadline": "soft-boundary-observed"},
        "source_inventory": [{"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for path in source_paths],
        "families": [], "decisions": [], "evaluations": [], "observed_transient_peak_bytes": 0,
        "limitations": ["exploratory synthetic decision case, not RL or real inventory validation",
            "one planning future; held-out future distribution specified by researchers",
            "families, not candidates or evaluation futures, are independent repetition units",
            "native-fast and strict/admitted have different assurance contracts",
            "shared host interference and full lifetime memory are unverified",
            "temporary storage checked only at capture and report boundaries; no hard realtime claim",
            "source inventory is partial; root CLI owns whole implementation/environment provenance"]}
    report["manifest_sha256"] = digest(report["manifest"])
    started = clock()
    def check():
        if clock() - started >= max_seconds:
            raise TimeoutError("decision study total time budget reached")
    def storage_check(path=None):
        transient = sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path else 0
        report["observed_transient_peak_bytes"] = max(report["observed_transient_peak_bytes"], transient)
        retained = len(json.dumps(report, allow_nan=False, separators=(",", ":")).encode())
        if transient + retained > max_storage_bytes:
            raise RuntimeError("decision study observed storage allowance exceeded")
    try:
        for index in range(families):
            check()
            family = prepare_family(base_seed + index, candidates)
            rotation = index % len(methods)
            order = methods[rotation:] + methods[:rotation]
            if index // len(methods) % 2:
                order = tuple(reversed(order))
            family_receipt = {key: copy.deepcopy(family[key]) for key in (
                "family_id", "master", "seeds", "candidate_order", "prefix_identity", "planning_identity",
                "candidate_order_identity")}
            family_receipt.update(method_order=list(order), fixed_candidate_exact=False, completed=False)
            report["families"].append(family_receipt)
            expected = {}
            for quantity in family["candidate_order"]:
                check()
                expected[quantity] = modules["oracle"](family["config"], actions(quantity), DELTA)[PREFIX_STEPS:]
            selected = []
            for mode in MODES:
                mode_selected = []
                for method in order:
                    check()
                    receipt, traces = execute_decision(method, family, mode=mode,
                        budget_seconds=decision_budget_seconds, scratch_root=output_directory, costs=costs,
                        modules=modules, backend_factory=backend_factory, clock=clock, check=check,
                        storage_check=storage_check)
                    report["decisions"].append(receipt)
                    if receipt["status"] != "completed":
                        if receipt["status"] == "stopped":
                            raise TimeoutError(receipt.get("error"))
                        raise RuntimeError(receipt.get("error", "decision arm failed"))
                    exact = all(traces[q] == expected[q] for q in traces)
                    receipt["completed_candidate_oracle_exact"] = exact
                    if not exact or (mode == "fixed-candidates" and len(traces) != candidates):
                        receipt["status"] = "failed"
                        raise AssertionError("planning outcomes disagree with independent oracle")
                    selected.append(receipt["selected_quantity"])
                    mode_selected.append(receipt["selected_quantity"])
                    storage_check()
                if mode == "fixed-candidates":
                    if len(set(mode_selected)) != 1:
                        raise AssertionError("identical complete candidates produced different selections")
                    family_receipt["fixed_candidate_exact"] = True
            # Only after choices are finalized is the held-out panel materialized.
            panel = evaluation_panel(family, evaluation_rollouts)
            evaluation = {"family_id": family["family_id"]}
            report["evaluations"].append(evaluation)
            try:
                evaluate_selections(family, selected, panel, modules=modules, costs=costs,
                                    check=check, clock=clock, progress=evaluation)
            except Exception as exc:
                evaluation.update(status="stopped" if isinstance(exc, TimeoutError) else "failed",
                                  error=f"{type(exc).__name__}: {exc}")
                raise
            for receipt in report["decisions"]:
                if receipt["family_id"] == family["family_id"]:
                    receipt["heldout_evaluation"] = evaluation["selections"].get(str(receipt["selected_quantity"]))
            family_receipt["completed"] = True
            storage_check()
        report.update(status="completed", study_admission=True)
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc, TimeoutError) else "failed", study_admission=False,
                      error=f"{type(exc).__name__}: {exc}")
    planned = families * len(methods) * len(MODES)
    report["denominators"] = {"planned_decisions": planned, "attempted_decisions": len(report["decisions"]),
        "completed_decisions": sum(row["status"] == "completed" for row in report["decisions"]),
        "failed_or_stopped_decisions": sum(row["status"] != "completed" for row in report["decisions"]),
        "unexecuted_decisions": planned - len(report["decisions"]), "planned_families": families,
        "completed_families": sum(row["completed"] for row in report["families"]),
        "no_decision_arms": sum(row["no_decision"] for row in report["decisions"]),
        "fixed_candidate_exact_families": sum(row["fixed_candidate_exact"] for row in report["families"]),
        "planned_evaluation_panels": families, "attempted_evaluation_panels": len(report["evaluations"]),
        "completed_evaluation_panels": sum(row.get("status") == "completed" for row in report["evaluations"]),
        "unexecuted_evaluation_panels": families-len(report["evaluations"])}
    report["study_wall_seconds"] = clock() - started
    report["paired_summary"] = []
    for mode in MODES:
        for method in methods:
            if method == "R":
                continue
            pairs = []
            for family in report["families"]:
                rows = {row["method"]: row for row in report["decisions"]
                        if row["family_id"] == family["family_id"] and row["mode"] == mode
                        and row["status"] == "completed"}
                if method in rows and "R" in rows:
                    a, b = rows[method], rows["R"]
                    pairs.append({"family_id": family["family_id"],
                        "workflow_seconds_difference": a["workflow_wall_seconds"]-b["workflow_wall_seconds"],
                        "completed_candidates_difference": a["completed_before_deadline"]-b["completed_before_deadline"],
                        "both_selected": not a["no_decision"] and not b["no_decision"],
                        "heldout_loss_difference": (a["heldout_evaluation"]["heldout_loss"]-b["heldout_evaluation"]["heldout_loss"])
                            if a.get("heldout_evaluation") and b.get("heldout_evaluation") else None})
            report["paired_summary"].append({"mode": mode, "method_minus_R": method,
                "pairs": pairs, "inference": "descriptive individual family differences; no confirmatory CI"})
    final_bytes = len(json.dumps(report, allow_nan=False, separators=(",", ":")).encode())
    if final_bytes > max_storage_bytes:
        report.update(status="failed", study_admission=False, error="final report exceeds storage allowance")
    # Analysis/summary work also consumes the study budget, unlike the separate
    # per-decision deadline. A late report cannot certify a completed campaign.
    report["study_wall_seconds"] = clock() - started
    if report["status"] == "completed" and report["study_wall_seconds"] >= max_seconds:
        report.update(status="stopped", study_admission=False,
                      error="decision study total budget reached during final reporting")
    return report
