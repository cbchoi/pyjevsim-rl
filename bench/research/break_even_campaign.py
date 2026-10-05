"""Sequential break-even research collection; no test/install/staging workflow."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import time

from bench.continuation_study.cases import encoded, sha
from bench.continuation_study.run import Budget, remove_owned_directory
from bench import host, runner


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _write(budget, path, value, *, final=False):
    runner.refresh(budget)
    budget.write(path, value, final=final)


def _check_space(study, protocol):
    limits = protocol["proposed_resources_not_approved"]
    if shutil.disk_usage(study).free < limits["minimum_free_bytes"]:
        raise RuntimeError("less than declared minimum free disk space")
    if runner.storage_bytes(study) > limits["new_study_shared_bytes_including_transient"]:
        raise RuntimeError("shared new-study storage limit exceeded")


def _cell_summary(spec, packets):
    """Compare complete values in memory; hashes are compact retained evidence."""
    rows = [packet["row"] for packet in packets]
    by_role = {role: {p["row"]["method"]: p for p in packets if p["row"]["role"] == role}
               for role in ("companion", "timing")}
    successful = all(row.get("status") == "succeeded" and row.get("cleanup_confirmed") is True for row in rows)
    companion_complete = set(by_role["companion"]) == {"R", "N", "C1"}
    timing_complete = set(by_role["timing"]) == {"R", "N", "C1"}
    complete = successful and companion_complete and timing_complete and len(rows) == 6
    companion_exact = (successful and companion_complete and
        len({encoded(p["projection"]) for p in by_role["companion"].values()}) == 1)
    witness_equal = complete and len({encoded(p.get("witness")) for p in packets}) == 1
    identity_equal = complete and all(row.get("actual_source_identity") == spec["source_identity"]
        and row.get("source_identity") == spec["source_identity"] for row in rows)
    return {key: spec[key] for key in ("cell_id", "stage", "family", "family_id", "K", "S", "B")} | {
        "complete": complete, "exact": bool(complete and companion_exact and witness_equal and identity_equal),
        "companion_complete": companion_complete, "companion_exact": bool(companion_exact),
        "timing_complete": timing_complete, "normal_output_agreement": bool(witness_equal),
        "identity_agreement": bool(identity_equal), "methods": ["R", "N", "C1"],
        "arm_ids": [row["arm_id"] for row in rows],
        "scalar_witness_sha256_by_role_method": {
            row["role"] + "/" + row["method"]: row.get("scalar_witness_sha256") for row in rows},
        "timing_full_trace_exact": False}


def _pilot_projection(plan, rows, elapsed, analysis_reserve, stage_cap, multiplier):
    first_family = min(arm["family"] for arm in plan["arms"])
    first = [row for row in rows if row["family"] == first_family]
    lookup = {(row["K"], row["S"], row["B"], row["method"], row["role"]): row["process_wall_seconds"] for row in first}
    done = {row["arm_id"] for row in rows}
    remaining = sum(lookup[(arm["K"], arm["S"], arm["B"], arm["method"], arm["role"])]
                    for arm in plan["arms"] if arm["arm_id"] not in done)
    projected = multiplier * remaining + analysis_reserve
    return {"kind": "first_complete_family_resource_projection", "observed_elapsed_seconds": elapsed,
        "remaining_available_seconds": max(0, stage_cap - elapsed),
        "projected_remaining_seconds": projected, "safety_multiplier": multiplier,
        "feasible": projected <= stage_cap - elapsed, "completion_guaranteed": False,
        "pilot_added_to_denominator_twice": False}


def collect(root, study, plan, protocol, *, python=None, started=None, whole_deadline=None):
    """Attempt each predeclared arm once and preserve partial denominators."""
    stage = plan["stage"]
    campaign = study / stage
    campaign.mkdir(exist_ok=False)
    for name in ("transient", "requests", "provenance", "receipts"):
        (campaign / name).mkdir()
    limits = protocol["proposed_resources_not_approved"]
    stage_cap = limits[stage + "_seconds"]
    stage_started = time.perf_counter() if started is None else started
    stage_end = min(stage_started + stage_cap, whole_deadline or math.inf)
    reserve = limits["analysis_reserve_seconds_within_each_stage"]
    budget = Budget(study, max(0, stage_end - stage_started - reserve), limits["new_study_shared_bytes_including_transient"])
    budget.started = stage_started
    plan["budget"] = {"seconds": budget.seconds, "max_bytes": budget.max_bytes,
        "attempt_seconds": limits["arm_seconds"], "transient_max_bytes": limits["transient_bytes_within_shared_cap"]}
    plan.pop("manifest_sha256", None)
    plan["manifest_sha256"] = sha(plan)
    rows, cells, pending, source_keys = [], [], {}, set()
    stop_reason, unpersisted_rows, unpersisted_cells = None, [], []
    started_utc = datetime.now(timezone.utc).isoformat()
    pilot_checked = False
    try:
        _write(budget, campaign / "protocol.json", plan)
        _write(budget, campaign / "environment.json", {**host.metadata(campaign), "python": sys.version,
            "condition": plan.get("condition", "unspecified"), "condition_is_user_label": True,
            "host_interference_controlled": False, "memory_complete": False,
            "workers": 1, "blas_threads": 1, "existing_cohorts_in_new_storage_cap": False})
        for spec in plan["arms"]:
            runner.refresh(budget)
            budget.check()
            _check_space(study, protocol)
            case = plan["cases"][spec["case_id"]]
            try:
                packet = runner.invoke(root, campaign, plan, spec, case, budget, python=python or sys.executable)
            except BaseException as error:
                packet = {"row": runner.failed_row(spec, case, f"launch: {type(error).__name__}: {error}"),
                          "projection": [], "provenance": None}
            row = packet["row"]
            witness = row.pop("scalar_witness", None)
            packet["witness"] = witness
            rows.append(row)
            pending.setdefault(spec["cell_id"], []).append(packet)
            identity = packet.get("provenance")
            try:
                if identity is not None:
                    key = sha(identity)
                    source_keys.add(key)
                    row["source_metadata_key"] = key
                    destination = campaign / "provenance" / (key + ".json")
                    if not destination.exists():
                        _write(budget, destination, identity, final=row["status"] != "succeeded")
                runner.refresh(budget)
                budget.write(campaign / "arms.jsonl", row, append=True, final=row["status"] != "succeeded")
            except BaseException:
                unpersisted_rows.append(row)
                raise
            if len(pending[spec["cell_id"]]) == 6:
                summary = _cell_summary(spec, pending.pop(spec["cell_id"]))
                cells.append(summary)
                try:
                    runner.refresh(budget)
                    budget.write(campaign / "cells.jsonl", summary, append=True, final=not summary["exact"])
                except BaseException:
                    unpersisted_cells.append(summary)
                    raise
                if not summary["exact"]:
                    stop_reason = "scientific comparison or source identity failed: " + spec["cell_id"]
            if row["status"] != "succeeded":
                stop_reason = row.get("error") or "arm failed"
            if len(source_keys) > 1:
                stop_reason = "actual loaded-source identity changed within cohort"
            print(f"{stage} {len(rows)}/{len(plan['arms'])} {spec['arm_id']}: {row['status']}", flush=True)
            if stop_reason:
                break
            first_family = min(arm["family"] for arm in plan["arms"])
            first_count = sum(arm["family"] == first_family for arm in plan["arms"])
            if stage == "calibration" and not pilot_checked and sum(row["family"] == first_family for row in rows) == first_count:
                feasibility = _pilot_projection(plan, rows, time.perf_counter() - stage_started,
                    reserve, stage_end-stage_started, limits["feasibility_safety_multiplier"])
                _write(budget, campaign / "resource-feasibility.json", feasibility)
                pilot_checked = True
                if not feasibility["feasible"]:
                    stop_reason = "remaining calibration infeasible under approved time cap; no automatic extension"
                    break
    except BaseException as error:
        stop_reason = f"{type(error).__name__}: {error}"
    cleanup = []
    try:
        remaining = list((campaign / "transient").iterdir())
    except OSError as error:
        remaining = []
        cleanup.append({"removed": False, "error": str(error)})
        stop_reason = stop_reason or "owned transient observation failed"
    for directory in remaining:
        try:
            remove_owned_directory(directory, campaign / "transient")
            cleanup.append({"arm": directory.name, "removed": True})
        except BaseException as error:
            cleanup.append({"arm": directory.name, "removed": False, "error": str(error)})
            stop_reason = stop_reason or "owned transient cleanup failed"
    for cell_id, packets in pending.items():
        summary = _cell_summary(packets[0]["row"], packets)
        cells.append(summary)
        try:
            runner.refresh(budget)
            budget.write(campaign / "cells.jsonl", summary, append=True, final=True)
        except BaseException:
            unpersisted_cells.append(summary)
            stop_reason = stop_reason or "partial cell write failed"
    counts = runner.denominators(plan, rows)
    planned_cells = len({arm["cell_id"] for arm in plan["arms"]})
    admitted = (stop_reason is None and not unpersisted_rows and not unpersisted_cells
                and len(rows) == len(plan["arms"]) and len(cells) == planned_cells
                and all(cell["exact"] for cell in cells) and len(source_keys) == 1)
    execution = {"schema": "break-even-execution-v1", "stage": stage,
        "status": "completed" if admitted else "failed", "study_admission": admitted,
        "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "stop_reason": stop_reason, "denominators": counts,
        "cells": {"planned": planned_cells, "recorded": len(cells),
                  "complete": sum(cell["complete"] for cell in cells),
                  "companion_exact": sum(cell["companion_exact"] for cell in cells),
                  "admitted": sum(cell["exact"] for cell in cells)},
        "source_check": {"complete": len(rows) == len(plan["arms"]), "consistent": len(source_keys) == 1,
                         "source_identity": plan["source_identity"], "metadata_keys": sorted(source_keys)},
        "collection_wall_seconds": time.perf_counter()-stage_started,
        "unpersisted_rows": unpersisted_rows, "unpersisted_cells": unpersisted_cells,
        "cleanup": cleanup, "full_timing_trace_exact": False, "memory_complete": False,
        "whole_lifetime_peak_bytes": None, "host_interference_controlled": False,
        "budget_enforcement": "owned direct-child deadline; sampled/boundary storage, unsampled transient peak unknown",
        "retry_or_replacement": False, "counting_companions_are_performance_evidence": False}
    try:
        _write(budget, campaign / "execution.json", execution, final=True)
    except BaseException as error:
        execution.update(status="failed", study_admission=False,
                         execution_write_error=f"{type(error).__name__}: {error}")
        print(encoded({"terminal_execution": execution}).decode("utf-8"), file=sys.stderr, flush=True)
    return execution, rows, cells, stage_end


def _subsequent_feasibility(plan, calibration_rows, cap, reserve, multiplier):
    """Conservative maximum observed arm by role/method, with no throughput claim."""
    maxima = defaultdict(float)
    for row in calibration_rows:
        maxima[row["role"], row["method"]] = max(maxima[row["role"], row["method"]], row["process_wall_seconds"])
    projected = multiplier * sum(maxima[arm["role"], arm["method"]] for arm in plan["arms"]) + reserve
    return {"projected_seconds": projected, "approved_seconds": cap, "safety_multiplier": multiplier,
            "feasible": projected <= cap, "completion_guaranteed": False,
            "basis": "maximum observed calibration process wall by role and method"}


def run_workflow(root, output, *, condition="unspecified", max_seconds=9000, max_bytes=32*1024**2):
    root, output = Path(root).resolve(), Path(output).resolve()
    from run_research import source_inventory
    from .break_even_design import make_plan
    from .break_even_domain import FORECAST_ALGORITHM, INPUT_ALGORITHM
    from .break_even_analysis import fit_calibration, select_validation, analyze_validation
    protocol = json.loads((root / "docs/break-even-protocol.json").read_bytes())
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 9000 or not 1024**2 <= max_bytes <= 32*1024**2:
        raise ValueError("break-even budget exceeds approved design caps")
    protocol = copy.deepcopy(protocol)
    protocol.update(status="execution_requested", execute_authorized=True,
        execution_supported_by_current_runner=True,
        authority="explicit user request to proceed with the designed experiment, 2026-10-05")
    protocol["workload"].update(risk_kernel_version=FORECAST_ALGORITHM,
                                actual_demand_generator_version=INPUT_ALGORITHM)
    limits = protocol["proposed_resources_not_approved"]
    limits["new_study_shared_bytes_including_transient"] = max_bytes
    protocol["approved_execution_limits"] = dict(limits, total_seconds=max_seconds)
    started = time.perf_counter()
    deadline = started + max_seconds
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(output, max_seconds, max_bytes)
    budget.started = started
    sources = source_inventory()
    source_id = sha(sources)
    summary = {"schema": "break-even-workflow-v1", "status": "running", "stages": {},
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_identity": source_id,
        "budget_seconds": max_seconds, "new_shared_max_bytes": max_bytes,
        "host_interference_controlled": False, "whole_lifetime_memory_verified": False,
        "retries": 0, "old_cohorts_pooled": False, "human_productivity_measured": False}
    calibration_rows, predictions = [], None
    try:
        _check_space(output, protocol)
        historical = root / "results"
        summary["storage_at_start"] = {
            "existing_results_bytes_excluding_new_study": sum(path.stat().st_size
                for path in historical.rglob("*") if path.is_file() and not path.is_relative_to(output)),
            "free_bytes": shutil.disk_usage(output).free,
            "existing_results_counted_in_new_cap": False}
        _write(budget, output / "protocol.json", {**protocol, "source_sha256": sources, "source_identity": source_id})
        for stage in ("calibration", "validation", "transfer"):
            stage_started = time.perf_counter()
            if stage_started >= deadline:
                raise TimeoutError("whole-study deadline reached")
            plan = make_plan(protocol, stage, calibration_selection=predictions)
            plan.update(source_identity=source_id, condition=condition,
                        execute_authorized=True, resource_values_are_proposals=False)
            for spec in plan["arms"]:
                spec["source_identity"] = source_id
            if stage != "calibration":
                feasibility = _subsequent_feasibility(plan, calibration_rows,
                    min(limits[stage+"_seconds"], deadline-time.perf_counter()),
                    limits["analysis_reserve_seconds_within_each_stage"], limits["feasibility_safety_multiplier"])
                _write(budget, output / (stage + "-feasibility.json"), feasibility)
                if not feasibility["feasible"]:
                    raise RuntimeError(stage + " infeasible under approved cap; no automatic extension")
            summary["stages"][stage] = {"status": "running", "study_admission": False}
            execution, rows, cells, stage_end = collect(root, output, plan, protocol,
                started=stage_started, whole_deadline=deadline)
            summary["stages"][stage] = execution
            plan["source_check"] = execution["source_check"]
            if stage == "calibration":
                calibration_rows = rows
                analysis = fit_calibration(plan, rows, cells, protocol=protocol, deadline=stage_end)
                _write(budget, output / stage / "fit.json", analysis)
            else:
                analysis = analyze_validation(predictions, plan, rows, cells, cohort=stage,
                                               protocol=protocol, deadline=stage_end)
                _write(budget, output / stage / "validation.json", analysis)
            if time.perf_counter() > stage_end:
                raise TimeoutError(stage + " analysis exceeded its stage cap")
            summary["stages"][stage]["analysis_status"] = analysis.get("status", "completed")
            _write(budget, output / stage / "findings.ko.md", _stage_findings(stage, execution, analysis).encode("utf-8"))
            if not execution["study_admission"]:
                raise RuntimeError(stage + " collection stopped: " + str(execution["stop_reason"]))
            if analysis.get("status") != "succeeded" or analysis.get("study_admission") is not True:
                raise RuntimeError(stage + " analysis did not admit the complete cohort")
            if stage == "calibration":
                predictions = select_validation(analysis, protocol)
                _write(budget, output / "predictions.json", predictions)
                if predictions.get("status") != "succeeded":
                    raise RuntimeError("validation planning requires judgment: " + str(predictions.get("reason", predictions.get("status"))))
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
    summary.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        unexecuted_stages=[stage for stage in ("calibration", "validation", "transfer") if stage not in summary["stages"]])
    try:
        _write(budget, output / "workflow.json", summary, final=True)
        if summary["status"] != "completed":
            text = "# 손익분기 연구 중단 기록\n\n" + summary.get("error", "incomplete") + "\n\n"
            text += "재시도·대체·이전 코호트 혼합 없이 부분 자료를 보존했습니다. 완료된 성능 확인 연구가 아닙니다.\n"
            text += "\n```json\n" + json.dumps({name: item.get("denominators") for name, item in summary["stages"].items()}, ensure_ascii=False, indent=2) + "\n```\n"
        else:
            text = "# 손익분기 연구 결과\n\n세 단계의 실행·분석이 완료되었습니다. 단계별 findings.ko.md와 원자료를 참조하세요.\n"
            text += "\n실행 완료는 성능 우위 또는 예측 정확도 목표 달성을 뜻하지 않습니다.\n"
        _write(budget, output / "findings.ko.md", text.encode("utf-8"), final=True)
    except BaseException as error:
        summary.update(status="failed", final_write_error=f"{type(error).__name__}: {error}")
        print(encoded({"terminal_workflow": summary}).decode("utf-8"), file=sys.stderr, flush=True)
    print(f"Break-even {summary['status']}: {output}", flush=True)
    if summary.get("error"):
        print(summary["error"], file=sys.stderr, flush=True)
    return 0 if summary["status"] == "completed" else 2


def _stage_findings(stage, execution, analysis):
    lines = ["# 손익분기 " + stage + " 결과", "",
        "실행·분석 수용: " + str(execution["study_admission"] and analysis.get("study_admission") is True), "",
        "이 수용은 실행·동등성 완료이며 C1의 성능 우위를 뜻하지 않습니다.", "",
        "companion은 전체 상태·사건 비교용, timing은 정상 반환값 비교와 실행시간 측정용입니다.",
        "timing 실행의 전체 trace 동등성·호스트 무간섭·전체 lifetime memory는 확인하지 않았습니다.", ""]
    for role, counts in execution["denominators"].items():
        lines.append(f"- {role}: 성공 {counts['succeeded']}/{counts['planned']}, 실패 {counts['failed']}, 미실행 {counts['unexecuted']}")
    lines.extend(["", f"셀 수용 {execution['cells']['admitted']}/{execution['cells']['planned']}.", ""])
    if stage == "calibration":
        lines.extend(["calibration은 예측모형 적합용이며 독립적인 손익분기점 검증이 아닙니다.",
            "계수·잔차·범위 밖/근 없음 draw를 포함한 불확실성은 fit.json에 보존했습니다.",
            "검증 좌표와 표본 수는 상위 predictions.json에 고정됩니다."])
    else:
        lines.extend(["| K/S/B | C1−R 평균(s) | 차이 CI(s) | 방향 | 예측오차 허용폭 충족 |",
            "|---|---:|---|---|---|"])
        for item in analysis.get("coordinates", {}).values():
            interval = item["difference_ci_seconds"]
            lines.append(f"| {item['K']}/{item['S']}/{item['B']} | {item['mean_difference_seconds']:.6g} | [{interval[0]:.6g}, {interval[1]:.6g}] | {item['direction']} | {item['prediction_error_within_tolerance']} |")
        lines.extend(["", "모든 primary 허용폭 충족: " + str(analysis.get("all_primary_prediction_tolerances_met")),
            "CI 수준: " + str(analysis.get("primary_interval_level")),
            "전이 단계는 탐색적이며 검증 분모에 합치지 않습니다." if stage == "transfer" else "6개 좌표별 primary 구간은 Bonferroni 근사 조정입니다."])
    lines.extend(["", "학습수렴·정책품질·federation·개발자 생산성·일반적인 병렬 우위는 평가하지 않았습니다.", ""])
    return "\n".join(lines)
