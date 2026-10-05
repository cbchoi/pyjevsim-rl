"""Prospective break-even statistics, using saved records only.

No simulation, file writes, third-party dependencies or validation-data refits.
All bootstrap indices identify entire input families, shared across coordinates
and methods. Coefficients use [1, K/4096, B/16, K*B/(4096*16)].
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from random import Random
import statistics
import time


METHODS = ("R", "N", "C1")
ROLES = ("companion", "timing")
COMPARISONS = (("C1", "R"), ("N", "R"), ("C1", "N"))
KS = (1, 16, 256, 4096)
STATES = (8, 512)
BRANCHES = (4, 8, 16)
K_SCALE, B_SCALE = 4096.0, 16.0
PRIMARY_LEVEL = 1.0 - .05 / 6


def _check(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("break-even analysis deadline reached")


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not positive or value > 0))


def _key(coordinate):
    return f"S{coordinate['S']}-K{coordinate['K']}-B{coordinate['B']}"


def _coordinate(arm):
    return {name: arm[name] for name in ("K", "S", "B")}


def _quantile_sorted(values, p):
    if not values:
        return None
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def _interval(values, level=.95):
    if not values:
        return None
    ordered = sorted(values)
    tail = (1 - level) / 2
    return [_quantile_sorted(ordered, tail), _quantile_sorted(ordered, 1 - tail)]


def _describe(values):
    return {"n": len(values), "mean_seconds": statistics.fmean(values) if values else None,
            "sd_seconds": statistics.stdev(values) if len(values) > 1 else None}


def _sign(value):
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def branch_break_even(F, D, *, supported=(4, 16)):
    """Strict integer wins for Delta(B)=F-B*D; retain every sign case."""
    if not _number(F) or not _number(D):
        return {"status": "nonfinite", "root": None, "integer_win": None}
    if D == 0:
        return {"F": F, "D": D, "F_sign": _sign(F), "D_sign": "zero",
                "status": "all_tied" if F == 0 else "always_faster" if F < 0 else "never_faster",
                "root": None, "root_status": "no_unique_root",
                "integer_win": {"minimum": 1, "maximum": None} if F < 0 else None}
    root = F / D
    if not math.isfinite(root):
        root_status, root = "nonfinite_root", None
    else:
        root_status = ("nonpositive_root" if root <= 0 else "within_supported_domain"
                       if supported[0] <= root <= supported[1] else "outside_supported_domain")
    if D > 0:
        integer_win = ({"minimum": max(1, math.floor(F / D) + 1), "maximum": None}
                       if root is not None else None)
        status = "advantage_begins"
    else:
        maximum = math.ceil(F / D) - 1 if root is not None else 0
        integer_win = {"minimum": 1, "maximum": maximum} if maximum >= 1 else None
        status = "advantage_ends" if integer_win else "never_faster"
    return {"F": F, "D": D, "F_sign": _sign(F), "D_sign": _sign(D),
            "status": status, "root": root, "root_status": root_status,
            "integer_win": integer_win, "supported_B": list(supported)}


def compute_break_even(alpha, beta, gamma, eta, B, *, supported=(1, 4096)):
    """Compute-intensity roots without treating zero slope as a finite root."""
    intercept, slope = alpha - B * gamma, beta - B * eta
    if not _number(intercept) or not _number(slope):
        return {"status": "nonfinite", "root": None}
    if slope == 0:
        return {"status": "all_tied" if intercept == 0 else "always_faster" if intercept < 0 else "never_faster",
                "root_status": "no_unique_root", "root": None,
                "intercept": intercept, "slope": slope, "slope_sign": "zero"}
    root = -intercept / slope
    state = ("nonfinite_root" if not math.isfinite(root) else "nonpositive_root" if root <= 0
             else "within_supported_domain" if supported[0] <= root <= supported[1]
             else "outside_supported_domain")
    return {"status": "advantage_begins" if slope < 0 else "advantage_ends",
            "root_status": state, "root": root if math.isfinite(root) else None,
            "intercept": intercept, "slope": slope, "slope_sign": _sign(slope),
            "supported_K": list(supported)}


def _features(K, B):
    k, b = K / K_SCALE, B / B_SCALE
    return (1.0, k, b, k * b)


def predict(coefficients, K, B):
    if len(coefficients) != 4:
        raise ValueError("four scaled coefficients required")
    return math.fsum(c * x for c, x in zip(coefficients, _features(K, B)))


def _inverse(matrix):
    n = len(matrix)
    augmented = [list(row) + [float(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-14:
            raise ValueError("singular calibration design matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [value - factor * other for value, other
                                  in zip(augmented[row], augmented[column])]
    return [row[n:] for row in augmented]


def _ols_weights(coordinates):
    design = [_features(row["K"], row["B"]) for row in coordinates]
    gram = [[math.fsum(row[a] * row[b] for row in design) for b in range(4)] for a in range(4)]
    inverse = _inverse(gram)
    return [[math.fsum(inverse[a][b] * row[b] for b in range(4)) for row in design] for a in range(4)]


def _fit(weights, values):
    return [math.fsum(weight * value for weight, value in zip(row, values)) for row in weights]


def _difference(left, right):
    return [a - b for a, b in zip(left, right)]


def _physical(coefficients):
    return {"alpha": coefficients[0], "beta": coefficients[1] / K_SCALE,
            "gamma": -coefficients[2] / B_SCALE,
            "eta": -coefficients[3] / (K_SCALE * B_SCALE)}


def _input_seeds(plan):
    seeds = set()
    cases = plan.get("cases", {})
    for case in cases.values() if isinstance(cases, dict) else cases:
        seed = case.get("input_seed", case.get("seed"))
        if seed is not None:
            seeds.add(seed)
    for arm in plan["arms"]:
        seed = arm.get("input_seed", arm.get("seed"))
        if seed is not None:
            seeds.add(seed)
    return sorted(seeds)


def _collect(plan, rows, cells, deadline):
    """Audit receipt identities; retain failures without timing-value imputation."""
    _check(deadline)
    planned, planned_cells, coordinates = {}, defaultdict(dict), {}
    for arm in plan["arms"]:
        identifier = arm["arm_id"]
        if identifier in planned or arm["method"] not in METHODS or arm["role"] not in ROLES:
            raise ValueError("duplicate arm or unknown method/role")
        if any(type(arm[key]) is not int or arm[key] <= 0 for key in ("K", "S", "B")):
            raise ValueError("positive integer K/S/B required")
        if type(arm["family"]) is not int or arm["family"] < 0:
            raise ValueError("nonnegative integer family required")
        cell, slot = arm["cell_id"], (arm["role"], arm["method"])
        if slot in planned_cells[cell]:
            raise ValueError("duplicate method/role in cell")
        identity = (arm["family"], arm["K"], arm["S"], arm["B"])
        if cell in coordinates and coordinates[cell] != identity:
            raise ValueError("cell inputs disagree")
        coordinates[cell] = identity
        planned_cells[cell][slot] = arm
        planned[identifier] = arm
    if not planned or any(len(group) != 6 for group in planned_cells.values()):
        raise ValueError("each planned cell requires all six method/role arms")
    if len(set(coordinates.values())) != len(coordinates):
        raise ValueError("duplicate family/coordinate cell")
    observed, groups, issues, design_issues = {}, defaultdict(dict), [], []
    expected_source = plan.get("source_identity")
    if not isinstance(expected_source, str) or not expected_source:
        issues.append("missing expected source identity")
    source_check = plan.get("source_check")
    if source_check is not None and (source_check.get("complete") is not True or source_check.get("consistent") is not True):
        issues.append("campaign loaded-source check incomplete or inconsistent")
    family_seeds = defaultdict(set)
    for arm in planned.values():
        case = plan.get("cases", {}).get(arm.get("case_id"), {})
        seed = arm.get("input_seed", arm.get("seed", case.get("input_seed", case.get("seed"))))
        if type(seed) is int and seed >= 0:
            family_seeds[arm["family"]].add(seed)
        else:
            design_issues.append(f"missing family input seed: {arm['arm_id']}")
    if (any(len(seeds) != 1 for seeds in family_seeds.values())
            or len(set().union(*family_seeds.values())) != len(family_seeds)):
        design_issues.append("families must have distinct input seeds, constant across their coordinates")
    for row in rows:
        _check(deadline)
        identifier = row["arm_id"]
        if identifier not in planned or identifier in observed:
            raise ValueError("unplanned/duplicate observed arm; retries are forbidden")
        arm = planned[identifier]
        required = ("cell_id", "family", "K", "S", "B", "role", "method")
        if any(row.get(key) != arm[key] for key in required):
            raise ValueError("observed arm identity differs from plan")
        for key in ("case_id", "family_id", "input_seed", "seed", "config_sha256", "input_sha256", "action_sha256",
                    "input_identity", "action_identity", "forecast_seed", "action_seed", "input_structure",
                    "stage", "method_position", "method_order", "role_position", "role_order", "global_order"):
            if key in arm and row.get(key) != arm[key]:
                raise ValueError(f"observed {key} differs from plan")
        if row.get("status") not in ("succeeded", "failed", "timeout", "timed_out"):
            raise ValueError("unknown arm status")
        observed[identifier] = row
        groups[row["cell_id"]][row["role"], row["method"]] = row
        if row["status"] == "succeeded":
            if row.get("source_identity") != expected_source or row.get("actual_source_identity") != expected_source:
                issues.append(f"source mismatch or missing actual identity: {identifier}")
            if row["role"] == "timing" and not _number(row.get("workflow_wall_seconds"), positive=True):
                issues.append(f"invalid successful timing: {identifier}")
    receipts = {}
    for cell in cells:
        key = cell["cell_id"]
        if key not in planned_cells or key in receipts:
            raise ValueError("unplanned/duplicate cell receipt")
        receipts[key] = cell
    eligible, reasons = {}, {}
    for cell_id, arm_group in planned_cells.items():
        group, receipt = groups[cell_id], receipts.get(cell_id, {})
        why = []
        if len(group) != 6 or any(row["status"] != "succeeded" for row in group.values()):
            why.append("incomplete_or_failed_arms")
        if (not all(receipt.get(key) is True for key in
                    ("complete", "exact", "companion_exact", "normal_output_agreement", "identity_agreement"))
                or receipt.get("admitted", True) is not True):
            why.append("cell_not_admitted")
        if len(group) == 6:
            if any(row.get("cleanup_confirmed") is False or row.get("cleanup_errors")
                   or ("worker_exit_code" in row and row["worker_exit_code"] != 0)
                   for row in group.values()):
                why.append("termination_or_cleanup_not_verified")
            if any(row.get("source_identity") != expected_source or row.get("actual_source_identity") != expected_source
                   for row in group.values()) or not expected_source:
                why.append("source_not_verified")
            times = [group["timing", method].get("workflow_wall_seconds") for method in METHODS]
            if not all(_number(value, positive=True) for value in times):
                why.append("timing_unavailable")
            witnesses = [row.get("scalar_witness_sha256") for row in group.values()]
            if not all(isinstance(value, str) and value for value in witnesses) or len(set(witnesses)) != 1:
                why.append("normal_output_digest_disagrees_or_missing")
            projections = [group["companion", method].get("branch_projection_sha256") for method in METHODS]
            B = next(iter(arm_group.values()))["B"]
            if (not all(isinstance(value, list) and len(value) == B for value in projections)
                    or projections[0] != projections[1] or projections[0] != projections[2]):
                why.append("companion_projection_disagrees_or_missing")
        if why:
            reasons[cell_id] = why
        else:
            eligible[cell_id] = {method: group["timing", method] for method in METHODS}
    denominators = {}
    for role in (*ROLES, "all"):
        planned_role = [arm for arm in planned.values() if role == "all" or arm["role"] == role]
        observed_role = [row for row in observed.values() if role == "all" or row["role"] == role]
        succeeded = sum(row["status"] == "succeeded" for row in observed_role)
        denominators[role] = {"planned": len(planned_role), "attempted": len(observed_role),
                              "succeeded": succeeded, "failed": len(observed_role) - succeeded,
                              "unexecuted": len(planned_role) - len(observed_role)}
    planned_family = defaultdict(set)
    for cell_id, (family, K, S, B) in coordinates.items():
        planned_family[family].add(cell_id)
    complete = sorted(family for family, identifiers in planned_family.items() if identifiers <= eligible.keys())
    admission = len(eligible) == len(planned_cells) and not issues and not design_issues
    audit = {"study_admission": admission, "denominators": denominators,
             "cells": {"planned": len(planned_cells), "recorded": len(receipts), "eligible": len(eligible)},
             "complete_families": complete, "planned_families": sorted(planned_family),
             "source_identity": expected_source, "source_issues": issues,
             "design_issues": design_issues,
             "input_seeds": _input_seeds(plan), "ineligible_cells": reasons,
             "failed_arm_ids": [key for key, row in observed.items() if row["status"] != "succeeded"],
             "unexecuted_arm_ids": [key for key in planned if key not in observed]}
    return audit, eligible


def _table(eligible):
    table = defaultdict(dict)
    for group in eligible.values():
        reference = group["R"]
        table[reference["family"]][_key(reference)] = group
    return table


def _descriptive(eligible):
    per_coordinate = defaultdict(lambda: defaultdict(list))
    for group in eligible.values():
        for method, row in group.items():
            per_coordinate[_key(row)][method].append(row)
    result = {}
    for key, group in sorted(per_coordinate.items()):
        result[key] = {}
        for method, rows in group.items():
            record = _describe([row["workflow_wall_seconds"] for row in rows])
            for metric in ("workflow_cpu_seconds", "process_wall_seconds", "unclassified_seconds"):
                record[metric] = _describe([row[metric] for row in rows if _number(row.get(metric))])
            phases = sorted({phase for row in rows for phase in (row.get("phase_seconds") or {})})
            record["phase_seconds"] = {phase: _describe([row["phase_seconds"][phase] for row in rows
                if _number((row.get("phase_seconds") or {}).get(phase))]) for phase in phases}
            sizes = [row["snapshot_bytes"] for row in rows if _number(row.get("snapshot_bytes"))]
            record["snapshot_bytes"] = {"observed_n": len(sizes), "missing_n": len(rows) - len(sizes),
                                        "mean_bytes": statistics.fmean(sizes) if sizes else None,
                                        "minimum_bytes": min(sizes) if sizes else None,
                                        "maximum_bytes": max(sizes) if sizes else None}
            result[key][method] = record
    return result


def _companion_counts(rows, eligible):
    """Retain measured vectors, never infer unique events or missing counters."""
    fields = ("work_counts", "operations", "phase_invocations", "prefixes_executed",
              "restores_executed", "step_results_returned", "suffix_steps_executed")
    records = []
    for row in rows:
        if row["role"] == "companion" and row["cell_id"] in eligible:
            records.append({"cell_id": row["cell_id"], "family": row["family"], **_coordinate(row),
                            "method": row["method"], **{key: row.get(key) for key in fields}})
    return {"scope": "separately instrumented companion, not timing evidence",
            "callback_vector_sum_is_unique_event_count": False,
            "missing_values_are_unmeasured_not_zero": True, "records": records}


def _bootstrap_options(replicates, deadline):
    if type(replicates) is not int or replicates < 1:
        raise ValueError("positive integer bootstrap_replicates required")
    _check(deadline)


def _root_accumulator(point):
    return {"point": point, "status_counts": Counter(), "root_status_counts": Counter(),
            "F_sign_counts": Counter(), "D_sign_counts": Counter(),
            "finite": [], "F": [], "D": []}


def _add_root(record, root):
    record["status_counts"][root["status"]] += 1
    record["root_status_counts"][root.get("root_status", root["status"])] += 1
    for field in ("F", "D"):
        if field in root:
            record[field].append(root[field])
            record[field + "_sign_counts"][_sign(root[field])] += 1
    if root.get("root") is not None and math.isfinite(root["root"]):
        record["finite"].append(root["root"])


def _finish_root(record, draws):
    return {"point": record["point"], "draws": draws,
            "status_counts": dict(record["status_counts"]),
            "root_status_counts": dict(record["root_status_counts"]),
            "F_sign_counts": dict(record["F_sign_counts"]), "D_sign_counts": dict(record["D_sign_counts"]),
            "F_ci95": _interval(record["F"]), "D_ci95": _interval(record["D"]),
            "finite_root_draws": len(record["finite"]),
            "no_finite_root_draws": draws - len(record["finite"]),
            "finite_root_conditional_ci95": _interval(record["finite"]),
            "conditional_interval_is_not_unconditional_root_confidence": True,
            "draw_frequencies_are_not_posterior_probabilities": True}


def fit_calibration(plan, rows, cells, *, protocol=None, bootstrap_replicates=20000, deadline=None):
    """Fit complete calibration families only; partial cohorts cannot plan holdout."""
    _bootstrap_options(bootstrap_replicates, deadline)
    if plan.get("stage") != "calibration":
        raise ValueError("fit_calibration accepts calibration stage only")
    audit, eligible = _collect(plan, rows, cells, deadline)
    factors = (protocol or {}).get("factors", {})
    Ks = tuple(factors.get("calibration_scenarios_K", KS))
    states = tuple(factors.get("product_records_S", STATES))
    branches = tuple(factors.get("branches_B", BRANCHES))
    expected = {_key({"K": K, "S": S, "B": B}) for S in states for K in Ks for B in branches}
    table = _table(eligible)
    families = [family for family in audit["complete_families"] if set(table[family]) == expected]
    expected_n = (protocol or {}).get("cohorts", {}).get("calibration", {}).get("families", 6)
    admitted = audit["study_admission"] and len(families) == expected_n and len(audit["planned_families"]) == expected_n
    result = {"schema": "break-even-calibration-analysis-v1", "cohort": "calibration", **audit,
              "study_admission": admitted, "status": "succeeded" if admitted else "partial",
              "fit_families": families, "descriptive": _descriptive(eligible),
              "companion_counts": _companion_counts(rows, eligible),
              "fits": {}, "family_coefficients": {}, "s_max": None,
              "bootstrap_replicates": bootstrap_replicates, "bootstrap_unit": "whole input family, jointly all coordinates and methods",
              "fits_use_validation": False, "coefficient_basis": ["1", "K/4096", "B/16", "K*B/65536"]}
    if not families:
        result["fit_status"] = "no_complete_calibration_families"
        return result
    family_coefficients, residuals, leave_out = {}, [], []
    sds = []
    for S in states:
        coords = [{"K": K, "S": S, "B": B} for K in Ks for B in branches]
        weights = _ols_weights(coords)
        per_method = {}
        for method in METHODS:
            per_method[method] = [_fit(weights, [table[f][_key(c)][method]["workflow_wall_seconds"] for c in coords]) for f in families]
        family_coefficients[str(S)] = per_method
        absolutes = {method: [statistics.fmean(coef[j] for coef in per_method[method]) for j in range(4)] for method in METHODS}
        differences = {f"{a}-{b}": _difference(absolutes[a], absolutes[b]) for a, b in COMPARISONS}
        result["fits"][str(S)] = {"absolute": {m: {"coefficients_scaled": c} for m, c in absolutes.items()},
                                   "differences": {name: {"coefficients_scaled": c, **_physical(c)} for name, c in differences.items()}}
        for c in coords:
            differences_by_family = [table[f][_key(c)]["C1"]["workflow_wall_seconds"] - table[f][_key(c)]["R"]["workflow_wall_seconds"] for f in families]
            if len(families) > 1:
                sds.append(statistics.stdev(differences_by_family))
            for family in families:
                for method in METHODS:
                    row = table[family][_key(c)][method]
                    predicted = predict(absolutes[method], c["K"], c["B"])
                    residuals.append({**c, "family": family, "method": method,
                                      "method_position": row.get("method_position"), "global_order": row.get("global_order"),
                                      "observed_seconds": row["workflow_wall_seconds"], "predicted_seconds": predicted,
                                      "residual_seconds": row["workflow_wall_seconds"] - predicted})
        for omitted in Ks:
            train = [c for c in coords if c["K"] != omitted]
            train_weights = _ols_weights(train)
            coefficients = {method: _fit(train_weights, [statistics.fmean(table[f][_key(c)][method]["workflow_wall_seconds"] for f in families) for c in train]) for method in METHODS}
            for c in (c for c in coords if c["K"] == omitted):
                for method in METHODS:
                    observed = statistics.fmean(table[f][_key(c)][method]["workflow_wall_seconds"] for f in families)
                    predicted = predict(coefficients[method], c["K"], c["B"])
                    leave_out.append({**c, "method": method, "observed_mean_seconds": observed,
                                      "predicted_seconds": predicted, "error_seconds": observed - predicted})
    result["family_coefficients"] = family_coefficients
    result["s_max"] = max(sds) if sds else None
    result["diagnostics"] = {"residuals": residuals, "leave_one_K_level_out": leave_out,
                             "leave_one_K_level_out_is_independent_validation": False}
    seed = (protocol or {}).get("ordering", {}).get("bootstrap_seed", 984252)
    rng = Random(seed)
    roots, coefficient_samples = {}, {}
    for S in states:
        for comparison, fit in result["fits"][str(S)]["differences"].items():
            parameters = {key: fit[key] for key in ("alpha", "beta", "gamma", "eta")}
            for K in Ks:
                key = f"S{S}:{comparison}:B_at_K{K}"
                roots[key] = _root_accumulator(branch_break_even(parameters["alpha"] + parameters["beta"] * K,
                                                                   parameters["gamma"] + parameters["eta"] * K,
                                                                   supported=(min(branches), max(branches))))
            for B in branches:
                key = f"S{S}:{comparison}:K_at_B{B}"
                roots[key] = _root_accumulator(compute_break_even(**parameters, B=B, supported=(min(Ks), max(Ks))))
        for method in METHODS:
            coefficient_samples[S, method] = [[] for _ in range(4)]
    n = len(families)
    for draw in range(bootstrap_replicates):
        if draw % 128 == 0:
            _check(deadline)
        counts = [0] * n
        for _ in range(n):
            counts[rng.randrange(n)] += 1
        for S in states:
            sampled = {}
            for method in METHODS:
                sample = [math.fsum(counts[i] * family_coefficients[str(S)][method][i][j] for i in range(n)) / n for j in range(4)]
                sampled[method] = sample
                for j, value in enumerate(sample):
                    coefficient_samples[S, method][j].append(value)
            for a, b in COMPARISONS:
                comparison = f"{a}-{b}"
                p = _physical(_difference(sampled[a], sampled[b]))
                for K in Ks:
                    _add_root(roots[f"S{S}:{comparison}:B_at_K{K}"], branch_break_even(p["alpha"] + p["beta"] * K,
                                                                                         p["gamma"] + p["eta"] * K,
                                                                                         supported=(min(branches), max(branches))))
                for B in branches:
                    _add_root(roots[f"S{S}:{comparison}:K_at_B{B}"], compute_break_even(**p, B=B, supported=(min(Ks), max(Ks))))
    for S in states:
        for method in METHODS:
            result["fits"][str(S)]["absolute"][method]["coefficient_ci95_scaled"] = [_interval(values) for values in coefficient_samples[S, method]]
    result["root_uncertainty"] = {key: _finish_root(record, bootstrap_replicates) for key, record in roots.items()}
    result["bootstrap_seed"] = seed
    result["singular_bootstrap_draws"] = 0
    result["singular_draw_note"] = "complete-family resampling preserves the fixed full-rank design matrix"
    _check(deadline)
    return result


def select_validation(fit, protocol):
    """Choose unseen coordinates and N from calibration only; never clip N."""
    result = {"schema": "break-even-locked-predictions-v1", "status": "calibration_incomplete",
              "N": None, "N_required": None, "coordinates": [],
              "calibration_source_identity": fit.get("source_identity"),
              "calibration_input_seeds": fit.get("input_seeds", []),
              "calibration_family_coefficients": fit.get("family_coefficients", {}),
              "calibration_fit_families": fit.get("fit_families", []),
              "calibration_used_only": True}
    if fit.get("study_admission") is not True:
        return result
    factors = protocol["factors"]
    candidates = factors["held_out_scenario_candidates_K"]
    coordinates = []
    for S in factors["product_records_S"]:
        state_fit = fit["fits"][str(S)]
        coefficients = state_fit["differences"]["C1-R"]["coefficients_scaled"]
        scores = [(abs(predict(coefficients, K, 8)), K) for K in candidates]
        if not all(_number(score) for score, _ in scores):
            result["status"] = "model_prediction_failed"
            return result
        K = min(scores)[1]
        for B in factors["branches_B"]:
            absolute = {method: predict(state_fit["absolute"][method]["coefficients_scaled"], K, B) for method in METHODS}
            differences = {f"{a}-{b}": absolute[a] - absolute[b] for a, b in COMPARISONS}
            coordinates.append({"K": K, "S": S, "B": B, "predicted_seconds": absolute,
                                "predicted_differences": differences, "tolerance_seconds": .10 * absolute["R"]})
    result["coordinates"] = coordinates
    if not all(_number(value, positive=True) for c in coordinates for value in c["predicted_seconds"].values()):
        result["status"] = "model_prediction_failed"
        return result
    s_max = fit.get("s_max")
    if not _number(s_max) or s_max < 0:
        result["status"] = "model_prediction_failed"
        return result
    z = statistics.NormalDist().inv_cdf(1 - .05 / 12)
    ratios = [z * s_max / c["tolerance_seconds"] for c in coordinates]
    # Multiplication becomes inf instead of raising OverflowError for extreme
    # finite planning variances; infeasibility remains a reportable result.
    needed = max(value * value for value in ratios)
    result.update(s_max=s_max, normal_quantile=z, engineering_precision_fraction=.10)
    if not math.isfinite(needed):
        result.update(status="sample_size_infeasible", N_required=None, reason="nonfinite sample-size requirement")
        return result
    required = 6 * math.ceil(max(12, needed) / 6)
    result["N_required"] = required
    if required > 48:
        result["status"] = "sample_size_infeasible"
        return result
    result.update(status="succeeded", N=required)
    return result


def _direction(interval):
    return "unavailable" if interval is None else "C1_faster" if interval[1] < 0 else "R_faster" if interval[0] > 0 else "uncertain"


def analyze_validation(predictions, plan, rows, cells, *, cohort="validation", protocol=None,
                       bootstrap_replicates=20000, deadline=None):
    """Analyze frozen predictions on fresh validation or separate transfer inputs."""
    _bootstrap_options(bootstrap_replicates, deadline)
    if cohort not in ("validation", "transfer") or plan.get("stage") != cohort:
        raise ValueError("validation/transfer stage required")
    if predictions.get("status") != "succeeded":
        raise ValueError("successful locked predictions required")
    audit, eligible = _collect(plan, rows, cells, deadline)
    if set(audit["input_seeds"]) & set(predictions["calibration_input_seeds"]):
        raise ValueError("held-out input seeds overlap calibration")
    if audit["source_identity"] != predictions["calibration_source_identity"]:
        raise ValueError("source identity changed since calibration predictions")
    coordinates = predictions["coordinates"]
    if len(coordinates) != 6 or len({_key(c) for c in coordinates}) != 6:
        raise ValueError("six unique frozen validation coordinates required")
    expected = {_key(c) for c in coordinates}
    if {_key(arm) for arm in plan["arms"]} != expected:
        raise ValueError("validation coordinates differ from frozen selection")
    table = _table(eligible)
    families = [f for f in audit["complete_families"] if set(table[f]) == expected]
    expected_n = predictions["N"] if cohort == "validation" else 6
    admitted = audit["study_admission"] and len(families) == expected_n and len(audit["planned_families"]) == expected_n
    result = {"schema": "break-even-held-out-analysis-v1", "cohort": cohort, **audit,
              "study_admission": admitted, "status": "succeeded" if admitted else "partial",
              "analysis_families": families, "descriptive": _descriptive(eligible),
              "companion_counts": _companion_counts(rows, eligible),
              "coordinates": {}, "predictions_refitted": False,
              "bootstrap_replicates": bootstrap_replicates,
              "bootstrap_unit": "whole held-out input family jointly all six coordinates",
              "primary_interval_level": PRIMARY_LEVEL if cohort == "validation" else .95,
              "transfer_is_exploratory_not_pooled": cohort == "transfer",
              "all_primary_prediction_tolerances_met": False}
    if not families:
        result["analysis_status"] = "no_complete_held_out_families"
        return result
    n = len(families)
    series = {}
    for c in coordinates:
        key = _key(c)
        series[key] = {}
        for a, b in COMPARISONS:
            left = [table[f][key][a]["workflow_wall_seconds"] for f in families]
            right = [table[f][key][b]["workflow_wall_seconds"] for f in families]
            series[key][f"{a}-{b}"] = [x - y for x, y in zip(left, right)]
            series[key][f"{a}/{b}"] = [math.log(x / y) for x, y in zip(left, right)]
    samples = {key: {comparison: [] for comparison in values} for key, values in series.items()}
    supplementary = {key: [] for key in series}
    seed = (protocol or {}).get("ordering", {}).get("bootstrap_seed", 984252) + (1 if cohort == "validation" else 2)
    rng, calibration_rng = Random(seed), Random(seed + 1000)
    cal_coef = predictions.get("calibration_family_coefficients", {})
    cal_n = len(predictions.get("calibration_fit_families", []))
    for draw in range(bootstrap_replicates):
        if draw % 128 == 0:
            _check(deadline)
        counts = [0] * n
        for _ in range(n):
            counts[rng.randrange(n)] += 1
        for key, comparisons in series.items():
            for comparison, values in comparisons.items():
                samples[key][comparison].append(math.fsum(counts[i] * values[i] for i in range(n)) / n)
        if cal_n:
            cal_counts = [0] * cal_n
            for _ in range(cal_n):
                cal_counts[calibration_rng.randrange(cal_n)] += 1
            for c in coordinates:
                coefficients = cal_coef[str(c["S"])]
                predicted = math.fsum(cal_counts[i] * (predict(coefficients["C1"][i], c["K"], c["B"])
                                     - predict(coefficients["R"][i], c["K"], c["B"])) for i in range(cal_n)) / cal_n
                supplementary[_key(c)].append(samples[_key(c)]["C1-R"][-1] - predicted)
    level = result["primary_interval_level"]
    accepted = []
    for c in coordinates:
        key = _key(c)
        predicted = c["predicted_differences"]["C1-R"]
        mean = statistics.fmean(series[key]["C1-R"])
        interval = _interval(samples[key]["C1-R"], level)
        error_interval = [value - predicted for value in interval]
        tolerance = c["tolerance_seconds"]
        within = error_interval[0] >= -tolerance and error_interval[1] <= tolerance
        accepted.append(within)
        secondary = {}
        for a, b in COMPARISONS:
            name, ratio = f"{a}-{b}", f"{a}/{b}"
            secondary[name] = {"mean_seconds": statistics.fmean(series[key][name]),
                               "pointwise_ci95_seconds": _interval(samples[key][name])}
            secondary[ratio] = {"paired_geometric_ratio": math.exp(statistics.fmean(series[key][ratio])),
                                "pointwise_ci95": [math.exp(v) for v in _interval(samples[key][ratio])]}
        order_groups = defaultdict(list)
        midpoint = n // 2
        for index, family in enumerate(families):
            reference = table[family][key]["C1"]
            label = str(reference.get("method_order", reference.get("method_position", "unrecorded")))
            order_groups[label].append(series[key]["C1-R"][index])
        result["coordinates"][key] = {**_coordinate(c), "paired_n": n,
            "predicted_difference_seconds": predicted, "mean_difference_seconds": mean,
            "difference_ci_seconds": interval, "mean_prediction_error_seconds": mean - predicted,
            "prediction_error_ci_seconds": error_interval, "tolerance_seconds": tolerance,
            "prediction_error_within_tolerance": within, "direction": _direction(interval),
            "secondary_exploratory": secondary,
            "supplementary_two_cohort_prediction_error_ci95": _interval(supplementary[key]),
            "supplementary_coordinates_fixed_not_selection_adjusted": True,
            "first_last_half_difference_seconds": {"first": _describe(series[key]["C1-R"][:midpoint]),
                                                     "last": _describe(series[key]["C1-R"][midpoint:])},
            "method_order_descriptive": {label: _describe(values) for label, values in order_groups.items()}}
    crossings = {}
    for S in sorted({c["S"] for c in coordinates}):
        ordered = sorted((row for row in result["coordinates"].values() if row["S"] == S), key=lambda row: row["B"])
        brackets = []
        for left in ordered:
            for right in ordered:
                if left["B"] < right["B"] and {left["direction"], right["direction"]} == {"C1_faster", "R_faster"}:
                    brackets.append({"lower_B": left["B"], "upper_B": right["B"],
                                     "lower_direction": left["direction"], "upper_direction": right["direction"]})
        crossings[str(S)] = {"status": "crossing_bracket_demonstrated" if brackets else "crossing_not_demonstrated",
                             "brackets": brackets, "real_valued_root_measured": False}
    result.update(bootstrap_seed=seed, crossings=crossings,
                  all_primary_prediction_tolerances_met=admitted and all(accepted) and cohort == "validation")
    _check(deadline)
    return result
