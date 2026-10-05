"""Pure planning for the predeclared break-even study; never starts a worker."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from random import Random


METHODS = ("R", "N", "C1")
ROLES = ("companion", "timing")
STAGES = ("calibration", "validation", "transfer")
SCHEMA = "break-even-plan-v1"
ENDPOINT = "observer-free-contiguous-v1"


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def derived_seed(label, family_seed):
    if type(label) is not str or type(family_seed) is not int or family_seed < 0:
        raise ValueError("a string namespace and nonnegative integer seed are required")
    return int.from_bytes(hashlib.sha256(f"{label}:{family_seed}".encode("utf-8")).digest()[:8], "big")


def load_protocol(path=None):
    location = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "docs/break-even-protocol.json"
    return json.loads(location.read_text(encoding="utf-8"))


def _protocol(protocol):
    if (protocol.get("schema") != "break-even-research-design-v1"
            or protocol.get("future_experiment_kind") != "break-even-v1"
            or protocol.get("methods") != list(METHODS) or protocol.get("roles") != list(ROLES)
            or protocol.get("endpoint", {}).get("revision") != ENDPOINT):
        raise ValueError("not the declared break-even protocol")
    factors = protocol["factors"]
    expected = {"calibration_scenarios_K": [1, 16, 256, 4096],
                "held_out_scenario_candidates_K": [4, 64, 1024], "product_records_S": [8, 512],
                "branches_B": [4, 8, 16], "prefix_steps": 64, "suffix_steps": 16,
                "delta": .25, "forecast_horizon": 8, "active_products": 8, "restore_count": "B"}
    if any(factors.get(key) != value for key, value in expected.items()):
        raise ValueError("workload factors differ from the predeclared design")
    starts = []
    for stage in STAGES:
        cohort = protocol["cohorts"][stage]
        seed = cohort["input_seed_start"]
        if type(seed) is not int or seed < 0:
            raise ValueError("cohort seeds must be nonnegative integers")
        starts.append(set(range(seed, seed + (48 if stage == "validation" else 6))))
    if any(starts[i] & starts[j] for i in range(3) for j in range(i)):
        raise ValueError("cohort input seed namespaces overlap")
    if (protocol["cohorts"]["calibration"]["families"] != 6
            or protocol["cohorts"]["transfer"]["families"] != 6):
        raise ValueError("calibration and transfer require six families")


def _coordinates(protocol, stage, selection):
    factors = protocol["factors"]
    if stage == "calibration":
        if selection is not None:
            raise ValueError("calibration cannot consume validation selections")
        return list(itertools.product(factors["calibration_scenarios_K"],
                                      factors["product_records_S"], factors["branches_B"])), 6
    if type(selection) is not dict or selection.get("status") != "succeeded":
        raise ValueError("an admitted calibration selection is required")
    coordinates = selection.get("coordinates")
    if type(coordinates) is not list or len(coordinates) != 6:
        raise ValueError("exactly six selected coordinates are required")
    selected = []
    for row in coordinates:
        values = tuple(row.get(key) for key in ("K", "S", "B"))
        if (any(type(value) is not int for value in values)
                or values[0] not in factors["held_out_scenario_candidates_K"]
                or values[1] not in factors["product_records_S"] or values[2] not in factors["branches_B"]):
            raise ValueError("selection is outside the held-out domain")
        selected.append(values)
    if len(set(selected)) != 6 or any(
            len({K for K, state, _ in selected if state == S}) != 1
            or {B for _, state, B in selected if state == S} != set(factors["branches_B"])
            for S in factors["product_records_S"]):
        raise ValueError("selection must use one K and all three B values for each S")
    count = selection.get("N")
    if type(count) is not int or count not in range(12, 49, 6):
        raise ValueError("validation N must be a predeclared multiple of six in 12..48")
    return sorted(selected), count if stage == "validation" else 6


def make_plan(protocol, stage, calibration_selection=None, *, case_factory=None, action_plan_factory=None):
    """Materialize inputs/actions and balanced orders without granting execution.

    Runtime/source identity and the approved resource allowance are supplied by
    the calling campaign, not inferred from a design-only protocol document.
    Factories are injectable only for direct planner tests.
    """
    _protocol(protocol)
    if stage not in STAGES:
        raise ValueError("unknown break-even stage")
    coordinates, families = _coordinates(protocol, stage, calibration_selection)
    if case_factory is None or action_plan_factory is None:
        from . import break_even_domain as domain
        case_factory = domain.create_case if case_factory is None else case_factory
        action_plan_factory = domain.make_action_plan if action_plan_factory is None else action_plan_factory
    cohort = protocol["cohorts"][stage]
    input_seeds = [cohort["input_seed_start"] + family for family in range(families)]
    if calibration_selection is not None:
        previous = set(calibration_selection.get("calibration_input_seeds", []))
        if previous & set(input_seeds):
            raise ValueError("held-out input seeds overlap calibration")
    rng = Random(protocol["ordering"]["planning_seed"] + 10000 * STAGES.index(stage))
    permutations = list(itertools.permutations(METHODS))
    orders = {}
    for block in range(families // 6):
        for coordinate in coordinates:
            for role in ROLES:
                order = list(permutations)
                rng.shuffle(order)
                orders[block, coordinate, role] = order
    cases, arms = {}, []
    for family, seed in enumerate(input_seeds):
        forecast_seed = derived_seed("be-forecast-v1", seed)
        action_seed = derived_seed("be-actions-v1", seed)
        order_seed = derived_seed("be-order-v1", seed)
        actions = {B: action_plan_factory(action_seed, B) for B in protocol["factors"]["branches_B"]}
        arranged = list(coordinates)
        Random(order_seed).shuffle(arranged)
        for coordinate in arranged:
            K, S, B = coordinate
            case_id = f"{stage}-f{family:02}-K{K}-S{S}-B{B}"
            case = case_factory(K=K, S=S, input_seed=seed, input_structure=cohort["input_structure"])
            if type(case) is not dict or type(case.get("config")) is not dict:
                raise ValueError("case factory must materialize a plain configuration")
            case = dict(case)
            case.update(seed=seed, input_seed=seed, forecast_seed=forecast_seed, action_seed=action_seed,
                        input_structure=cohort["input_structure"], action_plan=actions[B])
            case["config_sha256"] = sha(case["config"])
            case["input_identity"] = sha({"input_seed": seed, "input_structure": cohort["input_structure"],
                                           "config": case["config"]})
            case["action_identity"] = sha(actions[B])
            cases[case_id] = case
            role_order = list(ROLES if (family + coordinates.index(coordinate)) % 2 == 0 else reversed(ROLES))
            for role_position, role in enumerate(role_order):
                method_order = list(orders[family // 6, coordinate, role][family % 6])
                for position, method in enumerate(method_order):
                    arms.append({"arm_id": f"{case_id}-{role}-{method}", "cell_id": case_id,
                        "case_id": case_id, "stage": stage, "family": family,
                        "family_id": f"{stage}-f{family:02}", "input_seed": seed, "forecast_seed": forecast_seed,
                        "action_seed": action_seed, "K": K, "S": S, "B": B, "branch_count": B,
                        "model": "inventory-risk-v1", "prefix_steps": 64, "suffix_steps": 16, "delta": .25,
                        "method": method, "role": role, "purpose": role, "input_structure": cohort["input_structure"],
                        "input_identity": case["input_identity"], "action_identity": case["action_identity"],
                        "config_sha256": case["config_sha256"], "source_identity": None,
                        "method_order": method_order, "method_position": position,
                        "role_order": role_order, "role_position": role_position, "global_order": len(arms)})
    cells = families * len(coordinates)
    resources = protocol["proposed_resources_not_approved"]
    plan = {"schema": SCHEMA, "experiment_kind": "break-even-v1", "stage": stage,
        "model": "inventory-risk-v1", "methods": list(METHODS), "roles": list(ROLES),
        "endpoint_revision": ENDPOINT, "protocol_sha256": sha(protocol), "source_identity": None,
        "execute_authorized": False, "resource_values_are_proposals": True,
        "cases": cases, "arms": arms, "families": families, "input_seeds": input_seeds,
        "coordinates": [dict(zip(("K", "S", "B"), coordinate)) for coordinate in coordinates],
        "planned": {"all_arms": 6 * cells, "all_cells": cells, "timing_arms": 3 * cells,
                    "companion_arms": 3 * cells, "timing_cells": cells, "companion_cells": cells,
                    "families": families},
        "budget": {"seconds": resources[stage + "_seconds"],
                   "max_bytes": resources["new_study_shared_bytes_including_transient"],
                   "transient_max_bytes": resources["transient_bytes_within_shared_cap"],
                   "attempt_seconds": resources["arm_seconds"],
                   "analysis_reserve_seconds": resources["analysis_reserve_seconds_within_each_stage"]},
        "bootstrap_seed": protocol["ordering"]["bootstrap_seed"],
        "workers": 1, "blas_threads": 1, "no_retry_or_replacement": True,
        "calibration_selection_sha256": None if calibration_selection is None else sha(calibration_selection)}
    plan["manifest_sha256"] = sha(plan)
    return plan
