"""Predeclared three-way runtime surface; no model work at import time.

The original kernel file is unchanged. A fresh, sequential research worker
injects the declared action planner only for the duration of one arm.
"""
from __future__ import annotations

import itertools
import math
from random import Random

from ..continuation_study.cases import sha, workload


MODELS = ("Q", "M")
PREFIXES = (16, 64, 256)
BRANCHES = (1, 4, 16)
METHODS = ("R", "N", "C1")
FAMILIES = tuple(range(6))
SUFFIX_STEPS = 4
DELTA = .25
INPUT_HORIZON = 67.0
SCHEMA = "continuation-runtime-cost-v1"


def sustained_workload(model, family, *, seed_offset=0):
    if model not in MODELS or type(family) is not int or family not in FAMILIES:
        raise ValueError("unknown cost-study model/family")
    case = workload(model, family)
    seed = (971000 if model == "Q" else 972000) + seed_offset + family
    config = case["config"]
    if model == "Q":
        rng = Random(seed)
        config["arrival_spec"] = {"kind": "explicit", "end_time": INPUT_HORIZON,
            "events": [{"time": i * .5 + (.125 if rng.getrandbits(1) else 0.0),
                        "job_id": f"cost-arrival-{i}"} for i in range(1, 134)]}
    else:
        config["arrivals"] = [i * .25 for i in range(268)]
        config["maintenance_times"] = [2.0, 8.0, 12.0, 20.0, 28.0, 36.0, 44.0, 52.0, 60.0]
    case.update(seed=seed, config_sha256=sha(config))
    return case


def action_plan(model, prefix_steps, branch_index):
    if (model not in MODELS or type(prefix_steps) is not int or prefix_steps not in PREFIXES
            or type(branch_index) is not int or branch_index < 0):
        raise ValueError("undeclared runtime-cost action plan")
    if model == "Q":
        return ([{"mode": "normal"} for _ in range(prefix_steps)],
                [{"mode": "normal"} for _ in range(SUFFIX_STEPS)])
    return ([{"maintenance": False} for _ in range(prefix_steps)],
            [{"maintenance": bool(branch_index % 2 and offset == 0)} for offset in range(SUFFIX_STEPS)])


def make_manifest(config=None):
    options = dict(config or {})
    unknown = set(options) - {"budget_seconds", "max_bytes", "max_mib", "seed_offset", "condition"}
    if unknown:
        raise ValueError(f"unsupported cost configuration: {sorted(unknown)}")
    seconds = options.get("budget_seconds", 600)
    max_bytes = options.get("max_bytes", options.get("max_mib", 16) * 1024**2)
    offset = options.get("seed_offset", 0)
    condition = options.get("condition", "unspecified")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 7200:
        raise ValueError("positive cost budget at most7200 seconds required")
    if type(max_bytes) is not int or not 1024**2 <= max_bytes <= 128 * 1024**2:
        raise ValueError("cost storage budget must be1..128MiB")
    if type(offset) is not int or not 0 <= offset <= 1_000_000_000:
        raise ValueError("seed offset must be an integer in0..1000000000")
    if condition not in ("idle", "busy", "unspecified"):
        raise ValueError("invalid user-declared host condition")
    rng = Random(973251 + offset)
    conditions = list(itertools.product(MODELS, PREFIXES, BRANCHES))
    permutations = list(itertools.permutations(METHODS))
    orders = {}
    for condition_key in conditions:
        arranged = list(permutations)
        rng.shuffle(arranged)
        orders[condition_key] = arranged
    cases = {f"{model}-f{family:02}": sustained_workload(model, family, seed_offset=offset)
             for model in MODELS for family in FAMILIES}
    arms = []
    for purpose, families, grid in (("timing", FAMILIES, conditions),
                                    ("counting", (0,), [(model, 256, 16) for model in MODELS])):
        for family in families:
            arranged = list(grid)
            rng.shuffle(arranged)
            for model, prefix, branches in arranged:
                order = list(orders[model, prefix, branches][family])
                cell = f"{purpose}-{model}-f{family:02}-L{prefix}-B{branches}"
                for position, method in enumerate(order):
                    arms.append({"arm_id": f"{cell}-{method}", "cell_id": cell,
                        "purpose": purpose, "model": model, "family": family,
                        "case_id": f"{model}-f{family:02}", "prefix_steps": prefix,
                        "branch_count": branches, "suffix_steps": SUFFIX_STEPS,
                        "delta": DELTA, "method": method, "method_position": position,
                        "global_order": len(arms), "method_order": order})
    manifest = {"schema": SCHEMA, "experiment_kind": "runtime-cost-v1",
        "study_type": "exploratory-runtime-cost-surface-not-confirmation",
        "methods": list(METHODS), "models": list(MODELS), "prefixes": list(PREFIXES),
        "branches": list(BRANCHES), "timing_families": list(FAMILIES), "counting_family": 0,
        "cases": cases, "arms": arms, "planning_seed": 973251 + offset,
        "bootstrap_seed": 973252 + offset, "seed_offset": offset,
        "input_horizon": INPUT_HORIZON, "largest_endpoint_time": (256 + SUFFIX_STEPS) * DELTA,
        "arrival_intervals": {"Q": "0.5 plus seeded0-or0.125 jitter", "M": "0.25"},
        "policy": "queue-normal;manufacturing-prefix-false-suffix-first-odd-branch-true",
        "policy_seed": None, "condition": condition, "condition_is_user_label_not_clean_host_evidence": True,
        "workers": 1, "blas_threads": 1,
        "planned": {"all_arms": 330, "all_cells": 110, "timing_arms": 324, "timing_cells": 108,
                    "counting_arms": 6, "counting_cells": 2, "families_per_model": 6},
        "budget": {"seconds": float(seconds), "max_bytes": max_bytes, "attempt_seconds": 120},
        "comparisons": ["C1/R", "N/R", "C1/N"],
        "bootstrap": "20000 modelwise whole-family draws joint across9conditions and3comparisons; pointwise exploratory95%CI",
        "order_design": "each of six method permutations occurs once per model/condition across six timing families",
        "counting_is_separate_not_timing_evidence": True, "no_retry_or_replacement": True,
        "prior_campaigns_pooled": False, "host_interference_controlled": False,
        "whole_lifetime_peak_bytes": None, "persistence": "cached-local-filesystem-no-fsync",
        "wall_endpoint": "unchanged application kernel with parameterized action plan; import/provenance excluded",
        "counter_interpretation": "int/ext/output/con callback vector; decision steps and vector sum are not unique events",
        "boundary_interpretation": "observed grid crossings only; no interpolation/global threshold or simultaneous confidence",
        "evidence_limit": "successful full traces/snapshots discarded; retained digests cannot reconstruct them"}
    manifest["manifest_sha256"] = sha(manifest)
    return manifest


def execute_arm(kernel, spec, case, campaign, budget, modules, *, backend_factory=None, projection=None):
    if (spec.get("model") not in MODELS or spec.get("prefix_steps") not in PREFIXES
            or spec.get("branch_count") not in BRANCHES or spec.get("suffix_steps") != SUFFIX_STEPS
            or spec.get("delta") != DELTA or spec.get("purpose") not in ("timing", "counting")
            or spec.get("method") not in ("R", "N", "C", "C1")):
        raise ValueError("arm does not match the declared cost workload")
    internal = dict(spec, method="C" if spec["method"] == "C1" else spec["method"])
    options = {}
    if backend_factory is not None:
        options["backend_factory"] = backend_factory
    if projection is not None:
        options["projection"] = projection
    previous = kernel.action_plan
    # This explicit, same-for-all-methods hook is only for a fresh single-arm
    # worker. It is not a concurrent/public mutation of the simulation engine.
    kernel.action_plan = action_plan
    try:
        return kernel.execute_arm(internal, case, campaign, budget, modules, **options)
    finally:
        kernel.action_plan = previous
