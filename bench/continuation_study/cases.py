"""Predeclared Q/M exploratory workloads and an independent physical comparator.

This module does not import the continuation serializer or state providers.
Twelve paired timing repetitions are not a claim of twelve independent systems.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from random import Random

SCHEMA = "continuation-qm-exploratory-v1"
MODELS = ("Q", "M")
PREFIXES = (1, 64)
BRANCHES = (1, 8)
METHODS = ("R", "N", "C")
SUFFIX_STEPS = 8
DELTA = .25
FAMILIES = 12
METHOD_ORDERS = tuple(itertools.permutations(METHODS))


def encoded(value):
    """Research receipt representation, not a snapshot codec."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def workload(model, family):
    if model not in MODELS or type(family) is not int or not 0 <= family < FAMILIES:
        raise ValueError("unknown predeclared model/family")
    seed = (941000 if model == "Q" else 942000) + family
    if model == "Q":
        rng = Random(seed)
        config = {"schema_version": "queue-control-config-v1", "waiting_capacity": 16,
            "initial_service": "initial-0", "initial_waiting": [f"initial-{i}" for i in range(1, 17)],
            "initial_mode": "normal", "arrival_spec": {"kind": "explicit", "end_time": 33.0,
                "events": [{"time": float(i) + (.125 if rng.getrandbits(1) else 0.0),
                            "job_id": f"study-arrival-{i}"} for i in range(1, 33)]},
            "cost_weights": {"backlog": 1.0, "energy": 1.0, "drop": 10.0}}
    else:
        config = {"schema_version": "manufacturing-config-v1", "arrivals": [i * .25 for i in range(64)],
            "maintenance_times": [2.0, 8.0, 12.0], "stage_a_time": 1.0, "stage_b_time": 2.0,
            "repair_time": 1.0, "repair_jitter": .5, "wip_weight": 1.0, "completion_value": 2.0}
    return {"model": model, "family": family, "seed": seed, "config": config, "config_sha256": sha(config)}


def action_plan(model, prefix_steps, branch_index):
    if model not in MODELS or prefix_steps not in PREFIXES or type(branch_index) is not int or branch_index < 0:
        raise ValueError("undeclared action plan")
    if model == "Q":
        return ([{"mode": "normal"} for _ in range(prefix_steps)],
                [{"mode": "normal"} for _ in range(SUFFIX_STEPS)])
    return ([{"maintenance": False} for _ in range(prefix_steps)],
            [{"maintenance": bool(branch_index % 2 and index == 0)} for index in range(SUFFIX_STEPS)])


def make_manifest(*, budget_seconds, max_bytes, planning_seed=930251):
    if type(budget_seconds) not in (int, float) or not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("explicit positive execution budget required")
    if type(max_bytes) is not int or max_bytes < 1_048_576:
        raise ValueError("explicit storage budget of at least1MiB required")
    cases = {f"{model}-f{family:02}": workload(model, family)
             for model in MODELS for family in range(FAMILIES)}
    rng = Random(planning_seed)
    conditions = list(itertools.product(MODELS, PREFIXES, BRANCHES))
    arms = []
    for purpose, families in (("timing", range(FAMILIES)), ("counting", (0,))):
        for family in families:
            arranged = list(conditions)
            rng.shuffle(arranged)
            for model, prefix, branches in arranged:
                condition_index = conditions.index((model, prefix, branches))
                order = METHOD_ORDERS[(family + condition_index) % len(METHOD_ORDERS)]
                cell = f"{purpose}-{model}-f{family:02}-L{prefix}-B{branches}"
                for position, method in enumerate(order):
                    arms.append({"arm_id": f"{cell}-{method}", "cell_id": cell, "purpose": purpose,
                        "model": model, "family": family, "case_id": f"{model}-f{family:02}",
                        "prefix_steps": prefix, "branch_count": branches, "suffix_steps": SUFFIX_STEPS,
                        "delta": DELTA, "method": method, "method_position": position,
                        "global_order": len(arms), "method_order": list(order)})
    manifest = {"schema": SCHEMA, "study_type": "exploratory-pilot-not-confirmation", "planning_seed": planning_seed,
        "budget": {"seconds": float(budget_seconds), "max_bytes": max_bytes},
        "workers": 1, "blas_threads": 1, "cases": cases, "arms": arms,
        "planned": {"timing_arms": 288, "counting_arms": 24, "all_arms": 312,
                    "timing_cells": 96, "counting_cells": 8, "families_per_model": 12},
        "persistence": "local-cached-filesystem-no-fsync", "counting_family": 0,
        "no_retry_or_replacement": True, "policy": "queue-normal;manufacturing-prefix-false-suffix-first-odd-branch-true",
        "primary": "modelwise-L64-B8-geometric-mean-C-over-N-wall-ratio",
        "projection_schema": "independent-qm-outcomes-v1", "bootstrap_seed": 930252,
        "memory_complete": False, "whole_lifetime_peak_bytes": None,
        "host_interference_controlled": False}
    manifest["manifest_sha256"] = sha(manifest)
    return manifest


def physical_projection(model, graph, result, *, family_id, branch_id, expected_run_id, expected_step):
    """Read actual output and model history independently of snapshot export.

    Ordered event histories remain ordered. Only a verified physical episode
    prefix is normalized; logical identity/time/reward/termination are retained.
    """
    if type(result) not in (tuple, list) or len(result) != 5:
        raise ValueError("environment five-tuple required")
    observation, reward, terminated, truncated, info = result
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ValueError("boolean termination flags required")
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise ValueError("finite reward required")
    if info["run_id"] != expected_run_id or info["step_id"] != expected_step:
        raise ValueError("logical run/step changed")
    prefix = f"{info['instance_id']}:episode-"
    episode_id = info["episode_id"]
    if not isinstance(episode_id, str) or not episode_id.startswith(prefix):
        raise ValueError("episode ID physical prefix does not match instance")
    suffix = episode_id[len(prefix):]
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError("invalid logical episode number")
    if observation["logical_time"] != info["logical_time"]:
        raise ValueError("observation/info clock differ")
    if model == "Q":
        server = graph.server
        if server.admitted != len(server.waiting) + int(server.current is not None) + server.completed:
            raise ValueError("queue conservation failed")
        if server.initial_count + server.source_arrivals != server.admitted + server.dropped:
            raise ValueError("queue source conservation failed")
        if server.completed != len(graph.sink.ledger):
            raise ValueError("queue completion count differs")
        history = {"server_trace": graph.server.trace, "completions": graph.sink.ledger,
                   "pending": graph.sink.pending, "source_cursor": graph.source.index}
    elif model == "M":
        stages = (graph.stage_a, graph.stage_b)
        inventory = [job for stage in stages for job in stage.waiting]
        inventory += [stage.current for stage in stages if stage.current is not None]
        completed = [row["job"] for row in graph.sink.completions]
        if sorted(inventory + completed) != list(range(1, graph.source.job_cursor + 1)):
            raise ValueError("manufacturing conservation failed")
        active = [{"stage": stage.get_name(), "job": stage.current} for stage in stages if stage.current is not None]
        if active != ([] if graph.ledger.owner is None else [graph.ledger.owner]):
            raise ValueError("manufacturing shared tool invariant failed")
        if graph.ledger.repair_until is not None and active:
            raise ValueError("repair overlapped occupied tool")
        history = {"tool_trace": graph.ledger.trace, "stage_a": graph.stage_a.completed,
                   "stage_b": graph.stage_b.completed, "completions": graph.sink.completions,
                   "pending": graph.sink.pending, "rng_draws": graph.arbiter.rng_draws}
    else:
        raise ValueError("unknown model")
    # Serialization detaches mutable histories now, before the next transition.
    return json.loads(encoded({"schema": "independent-qm-outcomes-v1", "family_id": family_id,
        "branch_id": branch_id, "run_id": expected_run_id, "episode_number": int(suffix),
        "step_id": expected_step, "logical_time": info["logical_time"], "seed": info["seed"],
        "observation": observation, "reward": reward, "terminated": terminated,
        "truncated": truncated, "history": history}))
