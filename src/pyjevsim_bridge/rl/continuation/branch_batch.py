"""Explicit local branch batches over trusted, installed bundle factories.

This is not a learner, federation backend, timeout supervisor, or performance
certificate. Workers own their restored runtime and close it in finally. There
is no retry or fallback. Spawned work is bounded to the worker count; after an
observed cleanup failure no more requests are submitted. Already submitted work
is collected. A hung callback can still block completion: no timeout, forced
termination, or cleanup guarantee after abrupt process loss is provided here.

Only the trusted factory is pickled by multiprocessing. Snapshots, requests and
results cross the process boundary as closed canonical bytes, never live handles
or MappingProxyType objects. Factories must construct bundles, not independently
allocate simulator resources outside the coordinator's cleanup ownership.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import multiprocessing
import os
import pickle

from .contracts import (
    BranchContext, CleanupReceipt, ContinuationError, ContinuationSnapshot,
    canonical_bytes, decode_json, exact_fields,
)
from .coordinator import ContinuationCoordinator
from .registry import ContinuationRegistry

_RESULT_FIELDS = {
    "branch_id", "logical_context", "snapshot_sha256", "profile_id", "submitted",
    "attempted", "status", "transitions", "actions_requested", "actions_attempted",
    "actions_completed", "actions_not_executed", "transitions_omitted", "stop_condition",
    "cleanup", "error", "worker_pid",
}


def _error(exc: Exception, phase: str) -> dict:
    return {
        "code": str(getattr(exc, "code", "CC_BRANCH_FAILED")),
        "type": type(exc).__name__, "message": str(exc)[:4096],
        "phase": str(getattr(exc, "phase", "") or phase),
        "state_disposition": str(getattr(exc, "state_disposition", "")),
        "cleanup_errors": [str(item)[:4096] for item in getattr(exc, "cleanup_errors", ())],
    }


def _record(wire: bytes, snapshot: bytes) -> dict:
    request = decode_json(wire)
    branch = request["branch"]
    count = len(request["actions"])
    return {
        "branch_id": branch["branch_id"], "logical_context": branch,
        "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(), "profile_id": None,
        "submitted": False, "attempted": False, "status": "not_started",
        "transitions": [], "actions_requested": count, "actions_attempted": 0,
        "actions_completed": 0, "actions_not_executed": count,
        "transitions_omitted": 0, "stop_condition": "not-submitted",
        "cleanup": {"success": None, "failed_resources": [], "errors": [],
                    "scope": "not-started"},
        "error": None, "worker_pid": None,
    }


def _run_one(bundle_factory, snapshot: bytes, wire: bytes) -> bytes:
    """Top-level spawn target: exactly one restore, one action plan, one close."""
    result = _record(wire, snapshot)
    request = decode_json(wire)
    branch = BranchContext(**request["branch"])
    result.update(submitted=True, attempted=True, status="failed", worker_pid=os.getpid())
    runtime = None
    phase = "bundle-construction"
    # Restore owns and cleans any unpublished candidate. A successful close of
    # a published handle is recorded separately below.
    result["cleanup"] = {
        "success": True, "failed_resources": [], "errors": [],
        "scope": "not-created-or-coordinator-cleaned",
    }
    try:
        registry = ContinuationRegistry()
        bundle = bundle_factory()
        registry.register(bundle)
        result["profile_id"] = bundle.profile.profile_id
        phase = "restore"
        runtime = ContinuationCoordinator(registry).restore(snapshot, branch)
        phase = "step"
        for action in request["actions"]:
            result["actions_attempted"] += 1
            row = runtime.step(action)
            result["actions_completed"] += 1
            # Preserve all five fields. Do not strip step, reward, logical
            # identity or provenance while normalizing a comparison here.
            encoded = canonical_bytes({"transition": row})
            result["transitions"].append(decode_json(encoded)["transition"])
            if row[2] or row[3]:
                result["stop_condition"] = "terminated" if row[2] else "truncated"
                break
        else:
            result["stop_condition"] = "actions-exhausted"
        result["status"] = "success"
    except Exception as exc:
        result["error"] = _error(exc, phase)
        result["stop_condition"] = "error"
        if isinstance(exc, ContinuationError) and exc.code == "CC_CLEANUP_UNCONFIRMED":
            result["cleanup"] = {
                "success": False, "failed_resources": [],
                "errors": [str(item) for item in exc.cleanup_errors] or [str(exc)],
                "scope": "coordinator-candidate",
            }
    finally:
        if runtime is not None:
            try:
                receipt = runtime.close()
                if type(receipt) is not CleanupReceipt:
                    raise TypeError("runtime.close did not return a CleanupReceipt")
                result["cleanup"] = dict(receipt.to_payload(), scope="runtime.close")
            except Exception as exc:
                result["cleanup"] = {
                    "success": False, "failed_resources": [], "errors": [str(exc)[:4096]],
                    "scope": "runtime.close",
                }
                if result["error"] is None:
                    result["error"] = _error(exc, "close")
        result["actions_not_executed"] = result["actions_requested"] - result["actions_completed"]
        # step may commit successfully before its value encoding fails. Keep
        # that physical completion in the ledger instead of silently losing it.
        result["transitions_omitted"] = result["actions_completed"] - len(result["transitions"])
        if result["cleanup"]["success"] is not True:
            result["status"] = "cleanup_unconfirmed"
            if result["error"] is None:
                result["error"] = _error(ContinuationError(
                    "CC_CLEANUP_UNCONFIRMED", "owned runtime cleanup was not confirmed",
                    cleanup_errors=tuple(result["cleanup"]["errors"])), "close")
    try:
        return canonical_bytes(result)
    except ContinuationError as exc:
        # A closed result still has a bounded wire size. Do not turn oversized
        # output into a successful arm or pretend no physical work occurred.
        result["transitions_omitted"] = result["actions_completed"]
        result["transitions"] = []
        if result["status"] != "cleanup_unconfirmed":
            result["status"] = "failed"
        result["error"] = _error(exc, "result-encoding")
        result["stop_condition"] = "result-encoding-failed"
        return canonical_bytes(result)


def _unobserved(wire: bytes, snapshot: bytes, exc: Exception) -> dict:
    result = _record(wire, snapshot)
    result.update(submitted=True, attempted=None, status="cleanup_unconfirmed",
                  stop_condition="worker-result-unobserved")
    # Submitted is known; worker entry, actions and runtime cleanup are not.
    for key in ("actions_attempted", "actions_completed", "actions_not_executed"):
        result[key] = None
    result["cleanup"] = {
        "success": None, "failed_resources": [], "errors": [str(exc)[:4096]],
        "scope": "worker-result-unobserved",
    }
    result["error"] = _error(exc, "worker-result")
    return result


def _read_result(data: bytes, wire: bytes, snapshot: bytes) -> dict:
    result = exact_fields(decode_json(data), _RESULT_FIELDS, "branch result")
    expected = _record(wire, snapshot)
    if (result["branch_id"] != expected["branch_id"]
            or result["logical_context"] != expected["logical_context"]
            or result["snapshot_sha256"] != expected["snapshot_sha256"]
            or result["status"] not in ("success", "failed", "cleanup_unconfirmed")
            or result["submitted"] is not True or result["attempted"] is not True
            or type(result["worker_pid"]) is not int or result["worker_pid"] <= 0):
        raise ContinuationError("CC_INVALID_PAYLOAD", "worker result identity/status differs")
    cleanup = exact_fields(result["cleanup"], {"success", "failed_resources", "errors", "scope"},
                           "branch cleanup")
    if (type(cleanup["success"]) is not bool or type(cleanup["scope"]) is not str
            or any(type(cleanup[key]) is not list or any(type(item) is not str for item in cleanup[key])
                   for key in ("failed_resources", "errors"))
            or (cleanup["success"] and (cleanup["failed_resources"] or cleanup["errors"]))):
        raise ContinuationError("CC_INVALID_PAYLOAD", "worker cleanup receipt differs")
    counts = ("actions_requested", "actions_attempted", "actions_completed",
              "actions_not_executed", "transitions_omitted")
    if any(type(result[key]) is not int or result[key] < 0 for key in counts):
        raise ContinuationError("CC_INVALID_PAYLOAD", "worker action counts are not nonnegative integers")
    if (result["actions_requested"] != expected["actions_requested"]
            or not result["actions_completed"] <= result["actions_attempted"] <= result["actions_requested"]
            or result["actions_not_executed"] != result["actions_requested"] - result["actions_completed"]
            or type(result["transitions"]) is not list
            or len(result["transitions"]) + result["transitions_omitted"] != result["actions_completed"]):
        raise ContinuationError("CC_INVALID_PAYLOAD", "worker action/transition denominator differs")
    for row in result["transitions"]:
        if (type(row) is not list or len(row) != 5 or type(row[2]) is not bool
                or type(row[3]) is not bool or type(row[4]) is not dict):
            raise ContinuationError("CC_INVALID_PAYLOAD", "worker transition shape differs")
    if result["status"] == "success":
        if result["error"] is not None or not cleanup["success"] or result["transitions_omitted"]:
            raise ContinuationError("CC_INVALID_PAYLOAD", "successful branch contains an error or omission")
        stop = result["stop_condition"]
        if stop == "actions-exhausted":
            valid_stop = result["actions_not_executed"] == 0
        elif stop in ("terminated", "truncated"):
            flag = 2 if stop == "terminated" else 3
            valid_stop = bool(result["transitions"]) and result["transitions"][-1][flag] is True
        else:
            valid_stop = False
        if not valid_stop:
            raise ContinuationError("CC_INVALID_PAYLOAD", "successful branch stop condition differs")
    elif (type(result["error"]) is not dict
          or cleanup["success"] != (result["status"] == "failed")):
        raise ContinuationError("CC_INVALID_PAYLOAD", "failed branch error/cleanup status differs")
    return result


def _summary(results: list[dict], workers: int, stop_reason, batch_error) -> dict:
    return {
        "workers": workers, "planned": len(results),
        "submitted": sum(row["submitted"] is True for row in results),
        "attempted": sum(row["attempted"] is True for row in results),
        "attempted_unknown": sum(row["attempted"] is None for row in results),
        "succeeded": sum(row["status"] == "success" for row in results),
        "failed": sum(row["status"] in ("failed", "cleanup_unconfirmed") for row in results),
        "not_started": sum(row["status"] == "not_started" for row in results),
        "cleanup_unconfirmed": sum(row["status"] == "cleanup_unconfirmed" for row in results),
        "stop_reason": stop_reason, "error": batch_error, "results": results,
    }


def run_branches(bundle_factory, snapshot, requests, *, workers=1) -> dict:
    """Run (BranchContext, actions) pairs and retain the complete input ledger.

    Results remain in request order, independent of completion order. All input
    shape/closed-value/duplicate checks precede factory calls or submission.
    Terminal/truncated results stop that branch and report unused actions; a
    caller requiring a fixed action count must check actions_not_executed == 0.
    Workers 2..4 require the standard spawn-safe __main__ entry point. Factory
    code is explicitly trusted and must be picklable, even in serial mode.
    """
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ContinuationError("CC_INVALID_PAYLOAD", "workers must be an integer in 1..4")
    if not callable(bundle_factory):
        raise ContinuationError("CC_INVALID_PAYLOAD", "installed bundle factory must be callable")
    try:
        pickle.dumps(bundle_factory)
    except Exception as exc:
        raise ContinuationError("CC_INVALID_PAYLOAD", "bundle factory must be picklable") from exc
    data = snapshot.data if type(snapshot) is ContinuationSnapshot else snapshot
    decode_json(data)  # Bounded generic parse; provider admission remains in restore.
    if type(requests) not in (list, tuple):
        raise ContinuationError("CC_INVALID_PAYLOAD", "requests must be a list or tuple")
    wires, identities = [], set()
    for request in requests:
        if type(request) not in (tuple, list) or len(request) != 2:
            raise ContinuationError("CC_INVALID_PAYLOAD", "each request is a context/actions pair")
        branch, actions = request
        if type(branch) is not BranchContext or type(actions) not in (tuple, list):
            raise ContinuationError("CC_INVALID_PAYLOAD", "BranchContext and action sequence required")
        if branch.branch_id in identities:
            raise ContinuationError("CC_INVALID_PAYLOAD", "duplicate branch ID before submission")
        identities.add(branch.branch_id)
        wires.append(canonical_bytes({"branch": branch.to_payload(), "actions": actions}))
    results = [_record(wire, data) for wire in wires]
    stop_reason = batch_error = None
    if workers == 1:
        for index, wire in enumerate(wires):
            try:
                results[index] = _read_result(_run_one(bundle_factory, data, wire), wire, data)
            except Exception as exc:
                results[index] = _unobserved(wire, data, exc)
            if results[index]["status"] == "cleanup_unconfirmed":
                stop_reason = "cleanup-unconfirmed"
                break
    elif wires:
        executor = None
        pending = {}
        cursor = 0
        try:
            executor = ProcessPoolExecutor(max_workers=workers,
                                           mp_context=multiprocessing.get_context("spawn"))
            while pending or (cursor < len(wires) and stop_reason is None):
                while stop_reason is None and cursor < len(wires) and len(pending) < workers:
                    try:
                        future = executor.submit(_run_one, bundle_factory, data, wires[cursor])
                    except Exception as exc:
                        stop_reason = "submission-failed"
                        batch_error = _error(exc, "submit")
                        results[cursor]["error"] = batch_error
                        results[cursor]["stop_condition"] = "submission-failed"
                        break
                    pending[future] = cursor
                    results[cursor]["submitted"] = True
                    cursor += 1
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Drain every currently observable completion before refilling.
                completed.update(future for future in pending if future.done())
                for future in sorted(completed, key=pending.__getitem__):
                    index = pending.pop(future)
                    try:
                        results[index] = _read_result(future.result(), wires[index], data)
                    except Exception as exc:
                        results[index] = _unobserved(wires[index], data, exc)
                    if results[index]["status"] == "cleanup_unconfirmed":
                        stop_reason = "cleanup-unconfirmed"
        except Exception as exc:
            stop_reason = "pool-failed"
            batch_error = _error(exc, "pool")
            # No retries. Preserve or collect all jobs already submitted.
            for future, index in pending.items():
                try:
                    results[index] = _read_result(future.result(), wires[index], data)
                except Exception as worker_exc:
                    results[index] = _unobserved(wires[index], data, worker_exc)
        finally:
            if executor is not None:
                try:
                    executor.shutdown(wait=True, cancel_futures=False)
                except Exception as exc:
                    stop_reason = "pool-shutdown-failed"
                    batch_error = _error(exc, "pool-shutdown")
    for row in results:
        if row["status"] == "not_started" and stop_reason is not None and row["error"] is None:
            row["stop_condition"] = stop_reason
    return _summary(results, workers, stop_reason, batch_error)
