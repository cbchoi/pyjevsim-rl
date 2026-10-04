"""Prespecified 24-arm, two-model local portability/interference study."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from random import Random

from .continuation_study.cases import sha, workload

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {"preset", "condition", "budget_seconds", "max_bytes", "max_mib", "seed_offset"}


def make_manifest(config=None):
    overrides = dict(config or {})
    unknown = set(overrides) - ALLOWED
    if unknown:
        raise ValueError(f"unsupported preset override: {sorted(unknown)}")
    if overrides.get("preset", "idle-primary") != "idle-primary":
        raise ValueError("only the declared idle-primary preset is supported")
    preset = json.loads((ROOT / "configs" / "idle-primary.json").read_text(encoding="utf-8"))
    preset.update(overrides)
    condition = preset["condition"]
    seconds = preset["budget_seconds"]
    offset = preset["seed_offset"]
    max_bytes = preset.get("max_bytes", preset["max_mib"] * 1024**2)
    if condition not in ("idle", "busy", "unspecified"):
        raise ValueError("condition must be idle, busy or unspecified")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 7200:
        raise ValueError("budget seconds must be positive and at most 7200")
    if type(max_bytes) is not int or not 1024**2 <= max_bytes <= 128 * 1024**2:
        raise ValueError("retained storage budget must be 1..128MiB")
    if type(offset) is not int or not 0 <= offset <= 1_000_000_000:
        raise ValueError("seed offset must be an integer in 0..1000000000")
    rng = Random(preset["planning_seed"] + offset)
    cases = {}
    for model, family in itertools.product(("Q", "M"), range(6)):
        case = workload(model, family)
        seed = preset["seed_bases"][model] + offset + family
        if model == "Q":
            input_rng = Random(seed)
            case["config"]["arrival_spec"]["events"] = [
                {"time": float(i) + (.125 if input_rng.getrandbits(1) else 0.0),
                 "job_id": f"study-arrival-{i}"} for i in range(1, 33)]
        case.update(seed=seed, config_sha256=sha(case["config"]))
        cases[f"{model}-f{family:02}"] = case
    orders = {}
    for model in ("Q", "M"):
        rows = [["N", "C1"] for _ in range(3)] + [["C1", "N"] for _ in range(3)]
        rng.shuffle(rows)
        orders[model] = rows
    arms = []
    for family in range(6):
        models = ["Q", "M"]
        rng.shuffle(models)
        for model in models:
            order = orders[model][family]
            cell = f"timing-{model}-f{family:02}-L64-B8"
            for position, method in enumerate(order):
                arms.append({"arm_id": f"{cell}-{method}", "cell_id": cell,
                    "purpose": "timing", "model": model, "family": family,
                    "case_id": f"{model}-f{family:02}", "prefix_steps": 64,
                    "branch_count": 8, "suffix_steps": 8, "delta": .25,
                    "method": method, "method_position": position,
                    "global_order": len(arms), "method_order": order})
    manifest = {"schema": "portable-continuation-primary-v1", "preset": "idle-primary",
        "study_type": "exploratory-host-specific-not-confirmation", "condition": condition,
        "condition_is_user_label_not_clean_host_evidence": True,
        "methods": ["N", "C1"], "models": ["Q", "M"], "timing_families": list(range(6)),
        "families": list(range(6)), "cases": cases, "arms": arms,
        "planning_seed": preset["planning_seed"] + offset,
        "bootstrap_seed": preset["bootstrap_seed"] + offset, "seed_offset": offset,
        "policy": "fixed-queue-normal;manufacturing-prefix-false-suffix-first-odd-branch-true",
        "policy_seed": None, "workers": 1, "blas_threads": 1,
        "planned": {"timing_arms": 24, "counting_arms": 0, "all_arms": 24,
                    "timing_cells": 12, "families_per_model": 6},
        "budget": {"seconds": float(seconds), "max_bytes": max_bytes, "attempt_seconds": 120},
        "primary": "modelwise-L64-B8-geometric-mean-C1-over-N-wall-ratio",
        "cpu_endpoint": "backend entry through kernel return; paired matched wall separately recorded",
        "wall_endpoint": "unchanged application kernel including setup/prefix/capture/restore/suffix/projection/cleanup/receipt; process import excluded",
        "no_retry_or_replacement": True, "prior_campaigns_pooled": False,
        "host_interference_controlled": False, "whole_lifetime_peak_bytes": None,
        "persistence": "cached local filesystem, no fsync",
        "retained_evidence_limit": "compact receipts/digests cannot reconstruct discarded successful traces"}
    manifest["manifest_sha256"] = sha(manifest)
    return manifest
