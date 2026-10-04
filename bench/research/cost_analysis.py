"""Exploratory three-way cost surface from saved records, never model execution."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

from ..analyze import _bytes, _check, _gm, _bootstrap
from ..continuation_study.cases import sha
from .cost import MODELS, METHODS, PREFIXES, BRANCHES, FAMILIES, SCHEMA, sustained_workload


RATIOS = (("C1", "R"), ("N", "R"), ("C1", "N"))
KINDS = ("int_trans", "ext_trans", "output", "con_trans")
CONDITIONS = tuple((length, branches) for length in PREFIXES for branches in BRANCHES)


def _integer(value):
    return type(value) is int and value >= 0


def _number(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _condition(length, branches):
    return f"L{length}-B{branches}"


def _manifest(manifest):
    if sha({key: value for key, value in manifest.items() if key != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("cost manifest hash differs")
    if (manifest.get("schema") != SCHEMA or manifest.get("experiment_kind") != "runtime-cost-v1"
            or manifest.get("methods") != list(METHODS) or manifest.get("timing_families") != list(FAMILIES)
            or manifest.get("counting_family") != 0 or manifest.get("workers") != 1 or manifest.get("blas_threads") != 1):
        raise ValueError("not the predeclared three-way cost design")
    planned, cells, coordinates = {}, defaultdict(dict), set()
    for index, arm in enumerate(manifest["arms"]):
        purpose, model, family = arm["purpose"], arm["model"], arm["family"]
        if (purpose not in ("timing", "counting") or model not in MODELS
                or type(family) is not int or family not in FAMILIES or arm["method"] not in METHODS
                or type(arm["prefix_steps"]) is not int or arm["prefix_steps"] not in PREFIXES
                or type(arm["branch_count"]) is not int or arm["branch_count"] not in BRANCHES
                or arm["suffix_steps"] != 4 or arm["delta"] != .25
                or type(arm["global_order"]) is not int or arm["global_order"] != index
                or (purpose == "counting" and (family != 0 or arm["prefix_steps"] != 256 or arm["branch_count"] != 16))):
            raise ValueError("arm lies outside predeclared cost grid")
        key = (purpose, model, family, arm["prefix_steps"], arm["branch_count"], arm["method"])
        order, position = arm["method_order"], arm["method_position"]
        if (type(order) is not list or len(order) != 3 or set(order) != set(METHODS)
                or not _integer(position) or position >= 3 or order[position] != arm["method"]
                or key in coordinates or arm["arm_id"] in planned
                or arm["method"] in cells[arm["cell_id"]]):
            raise ValueError("invalid/duplicate planned arm or method order")
        case = manifest["cases"][arm["case_id"]]
        expected = sustained_workload(model, family, seed_offset=manifest["seed_offset"])
        if _bytes(case) != _bytes(expected):
            raise ValueError("case differs from sustained predeclared input/seed")
        coordinates.add(key)
        planned[arm["arm_id"]] = arm
        cells[arm["cell_id"]][arm["method"]] = arm
    expected_planned = {"all_arms": 330, "all_cells": 110, "timing_arms": 324, "timing_cells": 108,
                        "counting_arms": 6, "counting_cells": 2, "families_per_model": 6}
    if manifest["planned"] != expected_planned or len(planned) != 330 or len(cells) != 110:
        raise ValueError("expected330 arms/110 cells, including separate6 counting arms")
    order_groups = defaultdict(list)
    for cell in cells.values():
        if set(cell) != set(METHODS):
            raise ValueError("planned cell lacks a method")
        reference = cell["R"]
        shared = ("purpose", "model", "family", "case_id", "prefix_steps", "branch_count", "suffix_steps", "delta", "method_order")
        if any(any(_bytes(arm[key]) != _bytes(reference[key]) for key in shared) for arm in cell.values()):
            raise ValueError("cell methods do not have identical inputs and declared order")
        if reference["purpose"] == "timing":
            order_groups[reference["model"], reference["prefix_steps"], reference["branch_count"]].append(tuple(reference["method_order"]))
    if len(order_groups) != 18 or any(len(orders) != 6 or len(set(orders)) != 6 for orders in order_groups.values()):
        raise ValueError("all six method permutations are required once per condition")
    return planned, cells


def _validate(manifest, rows, comparisons, execution, deadline):
    planned, planned_cells = _manifest(manifest)
    observed, groups = {}, defaultdict(dict)
    for row in rows:
        _check(deadline)
        identifier = row["arm_id"]
        if identifier not in planned or identifier in observed:
            raise ValueError("unplanned/duplicate result; no retries or substitution")
        arm = planned[identifier]
        case = manifest["cases"][arm["case_id"]]
        if (any(_bytes(row.get(key)) != _bytes(value) for key, value in arm.items())
                or row.get("seed") != case["seed"] or row.get("config_sha256") != case["config_sha256"]
                or row.get("status") not in ("succeeded", "failed")):
            raise ValueError("record identity/config/seed/status differs")
        observed[identifier] = row
        groups[row["cell_id"]][row["method"]] = row
        if row["status"] != "succeeded":
            continue
        prefixes = arm["branch_count"] if arm["method"] == "R" else 1
        suffix = arm["branch_count"] * arm["suffix_steps"]
        if (not _number(row.get("application_wall_seconds"), positive=True)
                or row.get("cleanup_confirmed") is not True or row.get("cleanup_errors") != []
                or row.get("error") is not None or row.get("worker_exit_code") != 0):
            raise ValueError("success contradicts time/error/worker/cleanup")
        for name, value in (("prefixes_executed", prefixes), ("suffix_steps_executed", suffix), ("expected_suffix_steps", suffix)):
            if not _integer(row.get(name)) or row[name] != value:
                raise ValueError("successful arm shortened the declared workload")
        expected_steps = {"prefix": prefixes * arm["prefix_steps"], "suffix": suffix}
        for name in ("step_calls_attempted", "step_results_returned"):
            value = row.get(name)
            if type(value) is not dict or set(value) != set(expected_steps) or any(not _integer(value[key]) or value[key] != expected_steps[key] for key in expected_steps):
                raise ValueError("successful decision-step accounting differs")
        vector = row.get("branch_projection_sha256")
        if type(vector) is not list or len(vector) != arm["branch_count"] or not all(_digest(item) for item in vector):
            raise ValueError("invalid branch projection digest vector")
        phases = row.get("phase_seconds")
        if type(phases) is not dict or not phases or any(not _number(value) for value in phases.values()):
            raise ValueError("missing/invalid phase times")
        if arm["purpose"] == "counting":
            counts = row.get("work_counts")
            if type(counts) is not dict or not counts:
                raise ValueError("unobserved callback counts are not zero")
            for vector in counts.values():
                if type(vector) is not dict or set(vector) != set(KINDS) or any(not _integer(value) for value in vector.values()):
                    raise ValueError("invalid callback vector")
            if any(sum(counts.get(phase, {}).values()) for phase in ("capture_write", "restore_read")):
                raise ValueError("capture/restore executed simulated callbacks")
        elif row.get("work_counts") is not None:
            raise ValueError("callback counting must not contaminate timing arms")
    compared, exact = {}, set()
    for cell in comparisons:
        key = cell["cell_id"]
        if key not in planned_cells or key in compared or type(cell.get("complete")) is not bool or type(cell.get("exact")) is not bool:
            raise ValueError("unplanned/duplicate/invalid comparison")
        group = groups[key]
        reference = planned_cells[key]["R"]
        digests, whole = cell.get("projection_sha256_by_method"), cell.get("whole_projection_sha256_by_method")
        if (type(digests) is not dict or set(digests) != set(group) or type(whole) is not dict
                or set(whole) != set(group) or not all(_digest(value) for value in whole.values())
                or any(_bytes(vector) != _bytes(group[method].get("branch_projection_sha256", [])) for method, vector in digests.items())):
            raise ValueError("cell digest vectors contradict arm records")
        complete = set(group) == set(METHODS) and all(row["status"] == "succeeded" for row in group.values())
        equal = complete and len({tuple(value) for value in digests.values()}) == 1 and len(set(whole.values())) == 1
        if (cell["complete"] != complete or cell["exact"] != equal or cell.get("methods") != list(METHODS)
                or cell.get("purpose") != reference["purpose"]
                or set(cell.get("arm_ids", [])) != {row["arm_id"] for row in group.values()}):
            raise ValueError("exact-cell status contradicts three-way evidence")
        compared[key] = cell
        if equal:
            exact.add(key)
    denominators, cell_counts = {}, {}
    for purpose in ("timing", "counting"):
        selected = [row for row in rows if row["purpose"] == purpose]
        recorded = [cell for cell in comparisons if cell["purpose"] == purpose]
        count = manifest["planned"][purpose + "_arms"]
        cell_count = manifest["planned"][purpose + "_cells"]
        denominators[purpose] = {"planned": count, "attempted": len(selected),
            "succeeded": sum(row["status"] == "succeeded" for row in selected),
            "failed": sum(row["status"] == "failed" for row in selected), "unexecuted": count - len(selected)}
        cell_counts[purpose] = {"planned": cell_count, "recorded": len(recorded),
            "complete": sum(cell["complete"] for cell in recorded), "exact": sum(cell["exact"] for cell in recorded),
            "mismatched_complete": sum(cell["complete"] and not cell["exact"] for cell in recorded),
            "incomplete": cell_count - sum(cell["complete"] for cell in recorded), "unrecorded": cell_count - len(recorded)}
    admitted = len(rows) == 330 and len(exact) == 110 and all(row["status"] == "succeeded" for row in rows)
    if execution is not None:
        if execution.get("denominators") != denominators:
            raise ValueError("execution/analysis arm denominators differ")
        reduced = {purpose: {key: value for key, value in record.items() if key in ("planned", "complete", "exact")}
                   for purpose, record in cell_counts.items()}
        if execution.get("exact_cells") != reduced:
            raise ValueError("execution/analysis cell denominators differ")
        if execution.get("study_admission") is True and not admitted:
            raise ValueError("execution falsely admits incomplete data")
        admitted = admitted and execution.get("study_admission") is True
    return planned, observed, groups, exact, denominators, cell_counts, admitted


def _description(values):
    observed = [value for value in values if _number(value)]
    return {"n": len(observed), "arithmetic_mean_seconds": statistics.fmean(observed) if observed else None,
            "median_seconds": statistics.median(observed) if observed else None,
            "geometric_mean_seconds": _gm([math.log(value) for value in observed]) if observed and all(value > 0 for value in observed) else None}


def _strata(values):
    return {label: {"n": len(logs), "geometric_wall_ratio": _gm(logs)} for label, logs in sorted(values.items())}


def _model(manifest, groups, exact, model, replicates, deadline):
    logs = {f"{_condition(length, branches)}:{numerator}/{denominator}": {}
            for length, branches in CONDITIONS for numerator, denominator in RATIOS}
    paired, complete_conditions = defaultdict(dict), defaultdict(set)
    for key in exact:
        group = groups[key]
        reference = group["R"]
        if reference["purpose"] != "timing" or reference["model"] != model:
            continue
        label, family = _condition(reference["prefix_steps"], reference["branch_count"]), reference["family"]
        paired[label][family] = group
        complete_conditions[family].add(label)
        for numerator, denominator in RATIOS:
            logs[f"{label}:{numerator}/{denominator}"][family] = math.log(group[numerator]["application_wall_seconds"] / group[denominator]["application_wall_seconds"])
    seed = manifest["bootstrap_seed"] + (0 if model == "Q" else 1)
    intervals = _bootstrap(logs, list(FAMILIES), seed, replicates, deadline)
    complete = [family for family in FAMILIES if len(complete_conditions[family]) == 9]
    complete_logs = {key: {family: value for family, value in values.items() if family in complete} for key, values in logs.items()}
    complete_intervals = _bootstrap(complete_logs, complete, seed + 100, replicates, deadline) if complete else {
        key: {"ci95": None, "valid_replicates": 0, "no_pair_replicates": replicates} for key in logs}
    estimates, absolutes = {}, {}
    for key, values in logs.items():
        label, comparison = key.split(":")
        numerator, denominator = comparison.split("/")
        order, relative = defaultdict(list), defaultdict(list)
        for family, value in values.items():
            group = paired[label][family]
            order["-".join(group["R"]["method_order"])].append(value)
            before = group[numerator]["method_position"] < group[denominator]["method_position"]
            relative["numerator-before" if before else "numerator-after"].append(value)
        estimates[key] = {"paired_n": len(values), "paired_families": sorted(values),
            "unavailable_families": [family for family in FAMILIES if family not in values],
            "geometric_wall_ratio": _gm(list(values.values())), **intervals[key],
            "first3_last3": {label: {"n": sum(family in values for family in members),
                "geometric_wall_ratio": _gm([values[family] for family in members if family in values])}
                for label, members in (("first3", (0, 1, 2)), ("last3", (3, 4, 5)))},
            "method_order": _strata(order), "relative_order": _strata(relative),
            "complete_family_sensitivity": {"n": len(complete_logs[key]),
                "geometric_wall_ratio": _gm(list(complete_logs[key].values())), **complete_intervals[key]}}
    for length, branches in CONDITIONS:
        label = _condition(length, branches)
        family_groups = paired[label]
        absolutes[label] = {"paired_families": sorted(family_groups), "methods": {}}
        for method in METHODS:
            rows = [group[method] for group in family_groups.values()]
            phases = sorted({phase for row in rows for phase in row["phase_seconds"]})
            absolutes[label]["methods"][method] = {
                "application_wall": _description([row["application_wall_seconds"] for row in rows]),
                "worker_cpu": _description([row.get("application_cpu_seconds") for row in rows]),
                "cpu_matched_wall": _description([row.get("cpu_scope_wall_seconds") for row in rows]),
                "process_wall": _description([row.get("process_wall_seconds") for row in rows]),
                "phase_seconds": {phase: _description([row["phase_seconds"][phase] for row in rows if phase in row["phase_seconds"]]) for phase in phases},
                "prefix_per_invocation": _description([row["phase_seconds"]["prefix"] / row["prefixes_executed"] for row in rows]),
                "suffix_per_branch": _description([row["phase_seconds"]["suffix"] / branches for row in rows]),
                "restore_per_branch": _description([row["phase_seconds"]["restore_read"] / branches for row in rows if "restore_read" in row["phase_seconds"]]),
                "snapshot_bytes": [row.get("snapshot_bytes") for row in rows]}
    boundaries = {}
    for numerator, denominator in RATIOS:
        comparison = f"{numerator}/{denominator}"
        available = {label: estimates[f"{label}:{comparison}"] for label in (_condition(*condition) for condition in CONDITIONS)
                     if estimates[f"{label}:{comparison}"]["paired_n"] > 0}
        wins = [label for label, row in available.items() if row["geometric_wall_ratio"] < 1]
        narrow = [label for label, row in available.items() if row["ci95"] is not None and row["ci95"][1] < 1]
        full = len(available) == 9 and all(row["paired_n"] == 6 for row in available.values())
        boundaries[comparison] = {"grid_complete": full, "point_estimate_below_one_cells": wins,
            "pointwise_exploratory_upper95_below_one_cells": narrow,
            "interpretation": ("no_observed_point_advantage_in_sampled_grid" if full and not wins
                               else "observed_grid_only_no_global_threshold" if full else "partial_grid_no_boundary_conclusion"),
            "not_simultaneous_confidence_or_confirmatory": True}
    return {"planned_families": list(FAMILIES), "complete_nine_condition_families": complete,
        "bootstrap_seed": seed, "complete_sensitivity_seed": seed + 100,
        "estimates": estimates, "absolute_costs": absolutes, "observed_boundaries": boundaries,
        "clean_only_sensitivity": "unavailable: host clean/interference not classified"}


def _counting(groups, exact):
    output = []
    for key in sorted(exact):
        group = groups[key]
        reference = group["R"]
        if reference["purpose"] != "counting":
            continue
        totals = {method: {kind: sum(vector[kind] for vector in row["work_counts"].values()) for kind in KINDS}
                  for method, row in group.items()}
        output.append({"cell_id": key, "model": reference["model"], "family": reference["family"],
            "prefix_steps": 256, "branch_count": 16, "callback_vectors": totals,
            "phase_callback_vectors": {method: row["work_counts"] for method, row in group.items()},
            "actual_R_minus_callbacks": {method: {kind: totals["R"][kind] - totals[method][kind] for kind in KINDS} for method in ("N", "C1")},
            "prefix_invocations": {method: row["prefixes_executed"] for method, row in group.items()},
            "prefix_decision_steps": {method: row["step_results_returned"]["prefix"] for method, row in group.items()},
            "interpretation": "separate instrumentation; vectors/decision steps are not unique events; reset/RNG draws not directly counted"})
    return output


def analyze_records(manifest, rows, cells, execution=None, *, bootstrap_replicates=20000, deadline=None):
    if type(bootstrap_replicates) is not int or bootstrap_replicates <= 0:
        raise ValueError("positive bootstrap replicate count required")
    planned, observed, groups, exact, denominators, cell_counts, admitted = _validate(manifest, rows, cells, execution, deadline)
    return {"schema": "runtime-cost-analysis-v1", "manifest_sha256": manifest["manifest_sha256"],
        "study_type": "exploratory-runtime-cost-surface-not-confirmation", "study_admission": admitted,
        "denominators": denominators, "exact_cells": cell_counts,
        "failed_arm_ids": [row["arm_id"] for row in rows if row["status"] == "failed"],
        "unexecuted_arm_ids": [key for key in planned if key not in observed],
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_unit": "whole model-specific six-family index jointly across nine conditions and three ratios",
        "models": {model: _model(manifest, groups, exact, model, bootstrap_replicates, deadline) for model in MODELS},
        "counting": _counting(groups, exact), "condition": manifest["condition"],
        "cpu_scope": "backend entry through kernel return; distinct from original application wall; wall-minus-CPU does not isolate external interference",
        "memory_complete": False, "whole_lifetime_peak_bytes": None, "host_interference_controlled": False,
        "cost_identity": "R=B*(prefix_R+suffix_R); C1=prefix_C1+capture+B*(restore_C1+suffix_C1); add setup, projection, cleanup, receipt and other measured work for total time",
        "limitations": ["two particular models, not an independently sampled model population", "pointwise exploratory intervals, not simultaneous confidence or confirmation",
            "no global crossover inferred from a finite grid; never observed is not impossible", "partial exact-pair summaries never make incomplete study admissible",
            "order/first-last strata are descriptive, not causal adjustments", "no pooling of historical, Linux, or other-host cohorts",
            "full traces/snapshots ephemeral; digests cannot reconstruct them", "callback sums and decision steps are not unique simulated events",
            "no learning, policy-quality, federation, general parallel-speedup or human productivity claim"]}


def _sources(root, rows):
    cache, issues, reference = {}, [], None
    for row in rows:
        if row["status"] != "succeeded":
            continue
        key = row.get("source_metadata_key")
        try:
            if not _digest(key):
                raise ValueError("missing/invalid provenance key")
            if key not in cache:
                value = json.loads((root / "provenance" / f"{key}.json").read_text(encoding="utf-8"))
                if sha(value) != key:
                    raise ValueError("provenance contents differ from retained key")
                cache[key] = value
            value = cache[key]
            if not value.get("source_files"):
                raise ValueError("empty source identity inventory")
            identity = {"python": value.get("python"), "implementation": value.get("implementation"),
                "dill_version": value.get("dill_version"), "dill_init_sha256": value.get("dill_init_sha256"),
                "source_files": value["source_files"], "native_source_identity": value.get("native_source_identity")}
            if reference is None:
                reference = identity
            elif _bytes(reference) != _bytes(identity):
                raise ValueError("recorded model/method/runtime sources changed within cohort")
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append({"arm_id": row["arm_id"], "error": f"{type(error).__name__}: {error}"})
    return {"checked_successful_arms": sum(row["status"] == "succeeded" for row in rows),
            "unique_records": len(cache), "issues": issues,
            "scope": "recorded loaded sources and Python/dill only, not a complete environment certificate"}


def _findings(result):
    lines = ["# Runtime cost surface / 실행 비용 경계", "", "탐색 결과이며 확증실험이 아닙니다.",
        f"분모: {result['denominators']}", f"exact 비교: {result['exact_cells']}",
        f"전체 연구 수용: {result['study_admission']}", "", "시간비가1 미만이면 분자의 방법이 빠릅니다. 모든 조건을 보존합니다.", ""]
    for model, record in result["models"].items():
        lines.extend([f"## {model}", "", "| L/B | C1/R | N/R | C1/N |", "|---|---|---|---|"])
        for length, branches in CONDITIONS:
            values = []
            for numerator, denominator in RATIOS:
                row = record["estimates"][f"{_condition(length, branches)}:{numerator}/{denominator}"]
                values.append(f"{row['geometric_wall_ratio']} [{row['ci95']}; n={row['paired_n']}]")
            lines.append(f"| {length}/{branches} | " + " | ".join(values) + " |")
        lines.extend(["", f"Observed boundaries: {record['observed_boundaries']}", ""])
    lines.extend(["계수는 별도6회이며 int/ext/output/con 벡터와 실제 prefix 호출만 보고합니다. 합계를 고유 사건 수로 부르지 않습니다.",
        "전체9조건을 가진 family, 절대 비용/phase·CPU·process wall, 실행순서/전후반 민감도는 analysis.json에 있습니다.",
        "기존 자료와 합치지 않았습니다. 불완전 코호트의 성공 쌍만으로 수용이나 경계를 확정하지 않습니다.",
        "95% 구간은 점별 탐색 구간으로 동시 신뢰·일반 임계값·항상 빠름을 보장하지 않습니다.",
        "시뮬레이션 호출을 생략해도 검증·저장·복원 비용 때문에 총시간 이득이 없을 수 있습니다.",
        "호스트 clean 상태/전체 메모리 관측/학습 성능/일반 병렬성은 입증하지 않습니다.",
        "완료는 analysis-status.json의 completed로 확인합니다."])
    return "\n".join(lines) + "\n"


def _records(root, name, fallback, *, allow_incomplete_tail=False):
    path = root / name
    lines = path.read_bytes().splitlines() if path.exists() else []
    records = []
    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if not allow_incomplete_tail or index != len(lines) - 1:
                raise
    key = "arm_id" if name == "arms.jsonl" else "cell_id"
    mapped = {row[key]: row for row in records}
    if len(mapped) != len(records):
        raise ValueError("duplicate persisted record")
    for row in fallback:
        if row[key] in mapped and _bytes(mapped[row[key]]) != _bytes(row):
            raise ValueError("terminal fallback conflicts with persisted record")
        mapped[row[key]] = row
    return list(mapped.values())


def analyze(campaign, *, max_seconds=None, max_bytes=None):
    started, root = time.perf_counter(), Path(campaign).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    execution = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    # The orchestrator reserves analysis time outside the model-stage allowance.
    # An explicit remaining-workflow budget must not be reduced by that smaller
    # model-stage budget for a second time.
    remaining = max_seconds if max_seconds is not None else manifest["budget"]["seconds"] - execution["campaign_wall_seconds"]
    if not _number(remaining, positive=True):
        raise TimeoutError("no finite positive analysis allowance remains")
    deadline = started + remaining
    cap = manifest["budget"]["max_bytes"] if max_bytes is None else min(max_bytes, manifest["budget"]["max_bytes"])
    if any((root / name).exists() for name in ("analysis.json", "findings.md", "analysis-status.json", "analysis-status.tmp")):
        raise FileExistsError("analysis attempt already exists; no implicit overwrite/retry")

    def write(name, data, *, reserve=4096):
        retained = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if retained + len(data) + reserve > cap:
            raise ValueError("analysis would exceed combined retained storage cap")
        with (root / name).open("xb") as stream:
            stream.write(data)

    write("analysis-status.json", _bytes({"status": "running", "remaining_workflow_seconds": remaining}), reserve=0)
    try:
        _check(deadline)
        fallback_arms = execution.get("unpersisted_arms", execution.get("unpersisted_arm_records", []))
        fallback_cells = execution.get("unpersisted_cells", execution.get("unpersisted_cell_records", []))
        io_failure = bool(execution.get("write_errors") or fallback_arms or fallback_cells)
        rows = _records(root, "arms.jsonl", fallback_arms, allow_incomplete_tail=io_failure)
        cells = _records(root, "cells.jsonl", fallback_cells, allow_incomplete_tail=io_failure)
        result = analyze_records(manifest, rows, cells, execution, deadline=deadline)
        result["source_receipts"] = _sources(root, rows)
        result["study_admission"] = result["study_admission"] and not result["source_receipts"]["issues"]
        result["completion_record"] = "analysis-status.json must say completed"
        _check(deadline)
        write("analysis.json", _bytes(result))
        write("findings.md", _findings(result).encode("utf-8"))
        _check(deadline)
        status = {"status": "completed", "analysis_wall_seconds": time.perf_counter() - started,
            "explicit_remaining_workflow_seconds": max_seconds,
            "endpoint": "input/read/validation/bootstrap/result writes; final status write excluded"}
        write("analysis-status.tmp", _bytes(status), reserve=0)
        (root / "analysis-status.tmp").replace(root / "analysis-status.json")
    except BaseException as error:
        status = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                  "partial_outputs_are_incomplete": True, "analysis_wall_seconds": time.perf_counter() - started}
        try:
            write("analysis-status.tmp", _bytes(status), reserve=0)
            (root / "analysis-status.tmp").replace(root / "analysis-status.json")
        except (OSError, ValueError):
            print(json.dumps(status), file=sys.stderr)
        raise
    result.update(status="completed", analysis_status=status)
    return result
