"""Single-host paired exploratory analysis; never pools campaigns or runs models."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from random import Random
import re
import time


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _check(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("execution plus analysis budget exhausted")


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _gm(values):
    return None if not values else math.exp(sum(values) / len(values))


def _percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _bootstrap(logs, families, seed, replicates, deadline):
    rng, draws = Random(seed), {metric: [] for metric in logs}
    for i in range(replicates):
        if i % 250 == 0:
            _check(deadline)
        selected = [rng.choice(families) for _ in families]
        for metric, pairs in logs.items():
            values = [pairs[family] for family in selected if family in pairs]
            if values:
                draws[metric].append(_gm(values))
    return {metric: {"ci95": [_percentile(values, .025), _percentile(values, .975)]
                    if len(logs[metric]) >= 2 and values else None,
                    "valid_replicates": len(values), "no_pair_replicates": replicates - len(values)}
            for metric, values in draws.items()}


def analyze_records(manifest, rows, cells, execution=None, *, bootstrap_replicates=20000, deadline=None):
    if hashlib.sha256(_bytes({key: value for key, value in manifest.items() if key != "manifest_sha256"})).hexdigest() != manifest["manifest_sha256"]:
        raise ValueError("manifest hash differs")
    if manifest["methods"] != ["N", "C1"] or manifest["timing_families"] != list(range(6)):
        raise ValueError("only the declared two-method six-family design is supported")
    if type(bootstrap_replicates) is not int or bootstrap_replicates <= 0:
        raise ValueError("positive bootstrap replicate count required")
    planned = {row["arm_id"]: row for row in manifest["arms"]}
    planned_cells = defaultdict(dict)
    for arm in manifest["arms"]:
        case = manifest["cases"][arm["case_id"]]
        if (arm["purpose"] != "timing" or arm["model"] not in ("Q", "M")
                or arm["family"] not in range(6) or arm["method"] not in ("N", "C1")
                or arm["prefix_steps"] != 64 or arm["branch_count"] != 8 or arm["suffix_steps"] != 8
                or case["model"] != arm["model"] or case["family"] != arm["family"]
                or hashlib.sha256(_bytes(case["config"])).hexdigest() != case["config_sha256"]):
            raise ValueError("planned case differs from the fixed workload")
        if arm["method"] in planned_cells[arm["cell_id"]]:
            raise ValueError("duplicate planned method")
        planned_cells[arm["cell_id"]][arm["method"]] = arm
    if (len(planned) != len(manifest["arms"]) or len(planned) != 24 or len(planned_cells) != 12
            or any(set(group) != {"N", "C1"} for group in planned_cells.values())):
        raise ValueError("expected 24 unique arms and 12 matched cells")
    family_cells = set()
    for group in planned_cells.values():
        n, c = group["N"], group["C1"]
        shared = ("model", "family", "case_id", "prefix_steps", "branch_count", "suffix_steps", "delta", "method_order")
        if (any(_bytes(n[key]) != _bytes(c[key]) for key in shared)
                or sorted(n["method_order"]) != ["C1", "N"]
                or any(row["method_order"][row["method_position"]] != row["method"] for row in group.values())):
            raise ValueError("cell does not pair the same workload and declared order")
        family_cells.add((n["model"], n["family"]))
    if family_cells != {(model, family) for model in ("Q", "M") for family in range(6)}:
        raise ValueError("model/family cell coverage differs")
    if (manifest["planned"]["all_arms"] != 24 or manifest["planned"]["timing_arms"] != 24
            or manifest["planned"]["timing_cells"] != 12 or manifest["planned"]["families_per_model"] != 6):
        raise ValueError("planned denominator declaration differs")
    observed, groups = {}, defaultdict(dict)
    for row in rows:
        _check(deadline)
        identifier = row["arm_id"]
        if identifier not in planned or identifier in observed:
            raise ValueError("unplanned/duplicate result; retries are not substitutions")
        arm = planned[identifier]
        case = manifest["cases"][arm["case_id"]]
        if (any(_bytes(row.get(key)) != _bytes(value) for key, value in arm.items())
                or row.get("seed") != case["seed"] or row.get("config_sha256") != case["config_sha256"]):
            raise ValueError("result metadata/config/seed differs from manifest")
        if row["status"] not in ("succeeded", "failed"):
            raise ValueError("result must have terminal status")
        observed[identifier], groups[row["cell_id"]][row["method"]] = row, row
        if row["status"] == "succeeded":
            if (not _positive(row.get("application_wall_seconds")) or row.get("cleanup_confirmed") is not True
                    or row.get("cleanup_errors") != [] or row.get("worker_exit_code") != 0
                    or row.get("suffix_steps_executed") != 64 or row.get("expected_suffix_steps") != 64
                    or row.get("prefixes_executed") != 1 or row.get("work_counts") is not None):
                raise ValueError("success contradicts wall/workload/cleanup/worker receipt")
            vector = row.get("branch_projection_sha256")
            if type(vector) is not list or len(vector) != 8 or any(type(item) is not str or not re.fullmatch(r"[a-f0-9]{64}", item) for item in vector):
                raise ValueError("successful branch digest vector invalid")
    compared, exact = {}, set()
    for cell in cells:
        key = cell["cell_id"]
        if key not in planned_cells or key in compared or type(cell.get("complete")) is not bool or type(cell.get("exact")) is not bool:
            raise ValueError("unplanned/duplicate/invalid comparison cell")
        group = groups[key]
        retained = cell.get("projection_sha256_by_method")
        whole = cell.get("whole_projection_sha256_by_method")
        if (type(retained) is not dict or not set(retained).issubset(group)
                or type(whole) is not dict or set(whole) != set(retained)
                or any(type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value) for value in whole.values())):
            raise ValueError("cell digest methods/whole hashes invalid")
        if any(_bytes(vector) != _bytes(group[method].get("branch_projection_sha256", [])) for method, vector in retained.items()):
            raise ValueError("cell projection digests contradict arm records")
        complete = set(retained) == {"N", "C1"} and all(row["status"] == "succeeded" for row in group.values())
        equal = complete and retained["N"] == retained["C1"] and whole["N"] == whole["C1"]
        if (cell["complete"] != complete or cell["exact"] != equal or cell.get("methods") != ["N", "C1"]
                or cell.get("purpose") != "timing" or set(cell.get("arm_ids", [])) != {row["arm_id"] for row in group.values()}):
            raise ValueError("comparison receipt contradicts observed planned methods")
        compared[key] = cell
        if equal:
            exact.add(key)
    counts = {"planned": 24, "attempted": len(rows), "succeeded": sum(row["status"] == "succeeded" for row in rows),
              "failed": sum(row["status"] == "failed" for row in rows), "unexecuted": 24 - len(rows)}
    results = {}
    for model in ("Q", "M"):
        logs, pair_rows = {"wall": {}, "cpu": {}}, []
        for cell_id in sorted(exact):
            group = groups[cell_id]
            n, c = group["N"], group["C1"]
            if n["model"] != model:
                continue
            family = n["family"]
            logs["wall"][family] = math.log(c["application_wall_seconds"] / n["application_wall_seconds"])
            if _positive(c.get("application_cpu_seconds")) and _positive(n.get("application_cpu_seconds")):
                logs["cpu"][family] = math.log(c["application_cpu_seconds"] / n["application_cpu_seconds"])
            pair_rows.append({"family": family, "wall_ratio_C1_over_N": math.exp(logs["wall"][family]),
                "cpu_ratio_C1_over_N": math.exp(logs["cpu"][family]) if family in logs["cpu"] else None,
                "method_order": n["method_order"], "C1_before_N": c["method_position"] < n["method_position"],
                "absolute_seconds": {method: {"application_wall": row["application_wall_seconds"],
                    "worker_cpu": row.get("application_cpu_seconds"),
                    "worker_cpu_matched_wall": row.get("cpu_scope_wall_seconds")}
                    for method, row in (("N", n), ("C1", c))}})
        seed = manifest["bootstrap_seed"] + (0 if model == "Q" else 1)
        intervals = _bootstrap(logs, list(range(6)), seed, bootstrap_replicates, deadline)
        metrics = {}
        for metric, values in logs.items():
            metrics[metric] = {"paired_n": len(values), "paired_families": sorted(values),
                "unavailable_families": [family for family in range(6) if family not in values],
                "geometric_ratio_C1_over_N": _gm(list(values.values())), **intervals[metric],
                "first3_last3": {label: {"n": sum(f in values for f in members),
                    "ratio": _gm([values[f] for f in members if f in values])}
                    for label, members in (("first3", range(3)), ("last3", range(3, 6)))},
                "relative_order": {label: {"n": sum(pair["family"] in values and pair["C1_before_N"] == before for pair in pair_rows),
                    "ratio": _gm([values[pair["family"]] for pair in pair_rows if pair["family"] in values and pair["C1_before_N"] == before])}
                    for label, before in (("C1-before-N", True), ("C1-after-N", False))}}
        results[model] = {"planned_families": list(range(6)), "complete_exact_families": sorted(logs["wall"]),
                          "bootstrap_seed": seed, "metrics": metrics, "pairs": pair_rows}
    admitted = counts["succeeded"] == 24 and len(exact) == 12
    complete_cells = sum(cell["complete"] for cell in compared.values())
    if execution is not None:
        declared = execution.get("denominators")
        if declared is not None and declared not in (counts, {"timing": counts}):
            raise ValueError("execution and analysis denominators differ")
        if execution.get("study_admission") is True and not admitted:
            raise ValueError("execution falsely claims complete admission")
        declared_exact = execution.get("exact_cells")
        if declared_exact is not None and declared_exact != {"timing": {"planned": 12, "complete": complete_cells, "exact": len(exact)}}:
            raise ValueError("execution and analysis comparison denominators differ")
        admitted = admitted and execution.get("study_admission", False)
    return {"schema": "portable-paired-analysis-v1", "manifest_sha256": manifest["manifest_sha256"],
        "condition": manifest.get("condition", "unspecified"), "condition_is_clean_host_proof": False,
        "denominators": counts, "exact_cells": {"planned": 12, "recorded": len(compared), "complete": complete_cells, "exact": len(exact),
            "mismatched_complete": complete_cells - len(exact), "incomplete": 12 - complete_cells,
            "unrecorded": 12 - len(compared)}, "failed_arm_ids": [row["arm_id"] for row in rows if row["status"] == "failed"],
        "unexecuted_arm_ids": [arm for arm in planned if arm not in observed], "study_admission": admitted,
        "models": results, "bootstrap_replicates": bootstrap_replicates, "interpretation": "exploratory single-host paired-family percentile intervals; ratio<1 means C1 faster",
        "cpu_scope": "worker CPU sampled separately from original application wall; matched wall is separate; wall-minus-CPU does not isolate interference",
        "host_observations": [{"arm_id": row["arm_id"], "observation": row.get("host_observation")} for row in rows],
        "source_metadata_keys": sorted({row["source_metadata_key"] for row in rows if row.get("source_metadata_key")}),
        "source_metadata_scope": "worker-selected file hashes retained in provenance; not a complete dependency or loaded-code inventory",
        "limitations": ["no cross-host/OS/prior-cohort pooling", "two particular models, six paired families each",
            "small order/block groups are descriptive, not causal adjustments", "aggregate host counters include benchmark activity",
            "no host-clean proof or complete process/memory observation", "successful projections discarded; equality crosschecked by retained digests",
            "no RL training, convergence, policy quality, federation or general parallel-speedup claim"]}


def _findings(result):
    lines = ["# 결과 / Findings", "", "## 한국어", "", "단일 호스트의 탐색 결과이며 확증실험이 아닙니다.",
             f"호스트 상태 사용자 표기: {result['condition']} (간섭 없음의 증거가 아님).",
             f"실행 분모: {result['denominators']}; exact: {result['exact_cells']}; 수용: {result['study_admission']}."]
    for model, values in result["models"].items():
        for metric, row in values["metrics"].items():
            lines.append(f"- {model} {metric} C1/N: n={row['paired_n']}, 기하평균비={row['geometric_ratio_C1_over_N']}, 탐색95%CI={row['ci95']}.")
    lines.extend(["", "시스템 CPU/I/O에는 벤치마크도 포함됩니다. CPU와 원래 wall endpoint는 다르며 그 차이를 외부 간섭으로 단정할 수 없습니다.",
                  "서로 다른 호스트·OS·이전 코호트를 합치지 않았습니다. 작은 표본의 순서/전후반 분석은 인과 검증이 아닙니다.",
                  "", "## English", "", "This is a single-host exploratory paired-family study, not confirmation.",
                  "Ratios below one favor C1. The user-supplied condition is not proof of an interference-free host.",
                  "Worker CPU and the original application-wall endpoint differ; a separately matched wall is retained when available.",
                  "Aggregate CPU/I/O includes benchmark activity and does not identify competing processes or establish causality.",
                  "No cross-host, cross-OS, or historical cohorts are pooled. Missing observations remain unavailable.",
                  "See analysis.json for absolute paired times, intervals, first/last-three and execution-order sensitivity.",
                  "Neither learning quality nor federation/general parallel performance is evaluated.",
                  "", "Completion requires analysis-status.json to say completed; partial output is not a completed analysis."])
    return "\n".join(lines) + "\n"


def analyze(directory, *, bootstrap_replicates=20000, max_seconds=None, max_bytes=None):
    started, root = time.perf_counter(), Path(directory).resolve()
    if max_seconds is not None and (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds < 0):
        raise ValueError("remaining seconds must be finite and nonnegative")
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("storage cap must be a positive integer")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    execution = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    remaining = manifest["budget"]["seconds"] - execution["campaign_wall_seconds"]
    if max_seconds is not None:
        remaining = min(remaining, max_seconds)
    deadline = started + remaining
    cap = manifest["budget"]["max_bytes"] if max_bytes is None else min(max_bytes, manifest["budget"]["max_bytes"])
    if any((root / name).exists() for name in ("analysis.json", "findings.md", "analysis-status.json", "analysis-status.tmp")):
        raise FileExistsError("analysis output exists; no implicit overwrite or retry")

    def write(name, data, reserve=4096):
        size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if size + len(data) + reserve > cap:
            raise ValueError("analysis exceeds campaign storage cap")
        with (root / name).open("xb") as stream:
            stream.write(data)

    write("analysis-status.json", _bytes({"status": "running"}), 0)
    try:
        _check(deadline)
        def read_records(name, fallback, identity):
            path = root / name
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            records = []
            for index, line in enumerate(lines):
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines) - 1 or not execution.get(fallback):
                        raise
                    # The failed final append is preserved by the terminal receipt.
            indexed = {row[identity]: row for row in records}
            if len(indexed) != len(records):
                raise ValueError("duplicate persisted result")
            for row in execution.get(fallback, []):
                if row[identity] in indexed and _bytes(indexed[row[identity]]) != _bytes(row):
                    raise ValueError("conflicting terminal fallback result")
                indexed[row[identity]] = row
            return list(indexed.values())
        rows = read_records("arms.jsonl", "unpersisted_arms", "arm_id")
        cells = read_records("cells.jsonl", "unpersisted_cells", "cell_id")
        result = analyze_records(manifest, rows, cells, execution, bootstrap_replicates=bootstrap_replicates, deadline=deadline)
        metadata_path = root / "environment.json"
        if not metadata_path.exists():
            metadata_path = root / "host.json"
        result["environment"] = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
        result["status"] = "computed-awaiting-final-status"
        result["completion_record"] = "analysis-status.json must say completed"
        _check(deadline)
        write("analysis.json", _bytes(result))
        write("findings.md", _findings(result).encode("utf-8"))
        _check(deadline)
        elapsed = time.perf_counter() - started
        status = {"status": "completed", "analysis_wall_seconds": elapsed,
            "execution_plus_analysis_wall_seconds": execution["campaign_wall_seconds"] + elapsed,
            "endpoint": "includes read/validation/analysis/result writes; final self-report write excluded"}
        write("analysis-status.tmp", _bytes(status), 0)
        (root / "analysis-status.tmp").replace(root / "analysis-status.json")
    except BaseException as exc:
        status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "partial_outputs_are_incomplete": True}
        try:
            write("analysis-status.tmp", _bytes(status), 0)
            (root / "analysis-status.tmp").replace(root / "analysis-status.json")
        except (OSError, ValueError):
            print(json.dumps(status), file=__import__("sys").stderr)
        raise
    result["analysis_status"] = status
    result["status"] = status["status"]
    return result
