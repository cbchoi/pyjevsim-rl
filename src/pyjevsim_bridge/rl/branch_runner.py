"""Opt-in, frozen-policy queue branching research; no learner updates or RTI.

The parent owns outer launch-to-exit timing, independent physical verification,
host resource observation, campaign deadlines and archival. This module never
launches on import. A branch command is a frozen-policy whole continuation, not
an adaptive-training rollout or the experimental batched-dispatch protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .learning import LoadedPolicy
    from .models.queue_snapshot import BranchRuntimeContextV1, SimulatorSnapshotV1
    from .reference_ppo import ReferencePPORolloutScope

DOMAIN = "pyjevsim-live-branch-v1"
CONFIG_SCHEMA = "live-branch-config-v1"
RESULT_SCHEMA = "live-branch-result-v1"
CONFIG_SCHEMA_V2 = "live-branch-config-v2"
RESULT_SCHEMA_V2 = "live-branch-result-v2"
VALIDATION_PROFILES = ("strict", "rollout-scoped-v1")
ARMS = ("S0", "S1", "P0", "P1")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def derive_seed(phase: str, purpose: str, master: int, *parts: Any) -> int:
    return int.from_bytes(
        hashlib.sha256(canonical([DOMAIN, phase, purpose, master, *parts])).digest()[:8], "big"
    ) & ((1 << 63) - 1)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate before allocating a graph/process; no implicit timing defaults."""
    from .models.queue_control import validate_queue_config

    if not isinstance(config, Mapping):
        raise ValueError("branch config must be a mapping")
    required = {
        "schema_version",
        "phase",
        "master",
        "prefix_steps",
        "branches",
        "model_config",
        "policy_seed",
    }
    optional = {"suffix_steps", "horizon", "operation_timeout_seconds", "branch_ids"}
    if config.get("schema_version") == CONFIG_SCHEMA_V2:
        optional |= {"validation_profile", "inference_metrics"}
    if not required <= set(config) or set(config) - required - optional:
        raise ValueError("branch config fields differ")
    cfg: dict[str, Any] = json.loads(canonical(dict(config)))
    if cfg["schema_version"] not in (CONFIG_SCHEMA, CONFIG_SCHEMA_V2) or cfg["phase"] not in (
        "pilot",
        "final",
        "correctness",
    ):
        raise ValueError("unknown branch schema or phase")
    if cfg["schema_version"] == CONFIG_SCHEMA_V2:
        cfg.setdefault("validation_profile", "strict")
        cfg.setdefault("inference_metrics", False)
        if cfg["validation_profile"] not in VALIDATION_PROFILES:
            raise ValueError("unknown validation profile")
        if type(cfg["inference_metrics"]) is not bool:
            raise ValueError("inference_metrics must be a boolean")
    for key in ("master", "policy_seed", "prefix_steps"):
        _integer(cfg[key], key)
    for key in ("branches",):
        _integer(cfg[key], key, 1)
    cfg.setdefault("suffix_steps", 64)
    cfg.setdefault("horizon", 576)
    cfg.setdefault("operation_timeout_seconds", 1800.0)
    _integer(cfg["suffix_steps"], "suffix_steps", 1)
    _integer(cfg["horizon"], "horizon", 1)
    if cfg["branches"] > 16 or cfg["prefix_steps"] + cfg["suffix_steps"] > cfg["horizon"]:
        raise ValueError("branch count or decision horizon exceeded")
    timeout = cfg["operation_timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 1800:
        raise ValueError("operation timeout must be finite in (0,1800]")
    if cfg["phase"] != "correctness" and (
        cfg["prefix_steps"] not in (0, 64, 512)
        or cfg["branches"] not in (1, 4, 16)
        or cfg["suffix_steps"] != 64
        or cfg["horizon"] != 576
    ):
        raise ValueError("research cell differs from the prespecified grid")
    cfg["model_config"] = validate_queue_config(cfg["model_config"])
    if cfg["phase"] != "correctness":
        model = cfg["model_config"]
        if (
            model["waiting_capacity"] != 3
            or model["initial_service"] is not None
            or model["initial_waiting"]
            or model["initial_mode"] != "idle"
            or model["arrival_spec"]
            != {"kind": "bernoulli-slots", "slot_count": 576, "slot_width": 0.5, "probability": 0.5}
            or model["cost_weights"] != {"backlog": 1.0, "energy": 1.0, "drop": 10.0}
        ):
            raise ValueError("research queue differs from the prespecified profile")
    expected = [f"branch-{index:04d}" for index in range(1, cfg["branches"] + 1)]
    cfg.setdefault("branch_ids", expected)
    ids = cfg["branch_ids"]
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise ValueError("branch_ids must be an ordered list")
    if len(ids) != len(expected) or sorted(ids) != expected:
        raise ValueError("branch_ids must contain each declared branch exactly once")
    return cfg


def logical_identity(config: Mapping[str, Any], branch_id: str | None = None) -> dict[str, Any]:
    """Physical PID, arm, pool width, dispatch order and prefix length are absent."""
    segment = "prefix" if branch_id is None else "suffix"
    label = "prefix" if branch_id is None else branch_id
    run_id = f"{DOMAIN}-{config['phase']}-{config['master']}"
    return {
        "domain": DOMAIN,
        "phase": config["phase"],
        "master": config["master"],
        "segment": segment,
        "logical_branch_id": label,
        "run_id": run_id,
        "generation": 0,
        "worker_id": label,
        "episode_id": f"{run_id}:{label}",
        "sampling_seed": derive_seed(config["phase"], "sampling", config["master"], label),
    }


def _error(error: BaseException, phase: str) -> dict[str, Any]:
    def describe(value: BaseException, depth: int = 0) -> dict[str, Any]:
        body: dict[str, Any] = {"type": type(value).__name__, "message": str(value)[:4096]}
        notes = getattr(value, "__notes__", None)
        if notes:
            body["notes"] = [str(note)[:4096] for note in notes[:8]]
        if isinstance(value, BaseExceptionGroup):
            if depth < 3:
                body["exceptions"] = [describe(child, depth + 1) for child in value.exceptions[:8]]
            if depth >= 3 or len(value.exceptions) > 8:
                body["details_truncated"] = True
        return body

    result = {**describe(error), "phase": phase}
    if getattr(error, "diagnostics", None) is not None:
        result["diagnostics"] = getattr(error, "diagnostics", None)
    cause = error.__cause__ or error.__context__
    if cause is not None and cause is not error:
        result["cause"] = describe(cause)
    return result


class BranchExecutionError(RuntimeError):
    def __init__(self, message: str, diagnostics: Any = None) -> None:
        self.diagnostics = diagnostics
        super().__init__(message)


class _Engine:
    """One process-local frozen policy and at most one live simulator graph."""

    def __init__(self, config: dict[str, Any]) -> None:
        from .learning import EMPTY_TRANSITION_BATCH_SHA256, PolicyArtifact, PolicyCompatibility
        from .models.queue_control import MODEL_ID, MODEL_VERSION
        from .models.queue_features import make_queue_features
        from .reference_ppo import (
            REFERENCE_PPO_ALGORITHM_ID,
            REFERENCE_PPO_ALGORITHM_VERSION,
            ReferencePPOConfigV1,
            ReferencePPOLearnerAdapter,
            ReferencePPOPolicyLoader,
        )

        self.config = config
        self.loop = asyncio.new_event_loop()
        self.active: BranchRuntimeContextV1 | None = None
        self.snapshot: SimulatorSnapshotV1 | None = None
        self.prefix: dict[str, Any] | None = None
        self.partial_trace: dict[str, Any] | None = None
        self.deadline = math.inf
        self.work = {"prefix": 0, "suffix": 0, "lifecycle": 0, "prefix_steps": 0, "suffix_steps": 0}
        self.features = make_queue_features()
        self.context = logical_identity(config)
        compatibility = PolicyCompatibility(
            REFERENCE_PPO_ALGORITHM_ID,
            REFERENCE_PPO_ALGORITHM_VERSION,
            MODEL_ID,
            MODEL_VERSION,
            self.features.contract_sha256,
            self.features.action_schema_sha256,
        )
        learner = ReferencePPOLearnerAdapter(
            self.features,
            run_id=self.context["run_id"],
            generation=0,
            compatibility=compatibility,
            config=ReferencePPOConfigV1(
                observation_size=7,
                action_count=3,
                hidden_size=16,
                batch_size=256,
                minibatch_size=64,
                epochs=4,
                initialization_seed=config["policy_seed"],
                shuffle_seed=derive_seed(config["phase"], "unused-shuffle", config["master"]),
            ),
        )
        candidate = learner.initial_policy()
        artifact = PolicyArtifact(
            run_id=self.context["run_id"],
            generation=0,
            policy_version=0,
            payload=candidate.payload,
            media_type=candidate.media_type,
            compatibility=candidate.compatibility,
            source_batch_sha256=EMPTY_TRANSITION_BATCH_SHA256,
            provenance=candidate.provenance,
        )
        self.policy = self.loop.run_until_complete(
            ReferencePPOPolicyLoader(self.features).load(artifact)
        )
        self.policy_context = {
            "policy_sha256": artifact.sha256,
            "policy_version": 0,
            "feature_contract_sha256": self.features.contract_sha256,
        }
        self.model_seed = derive_seed(config["phase"], "model", config["master"])

    def create(self) -> BranchRuntimeContextV1:
        from .models.queue_snapshot import BranchRuntimeContextV1

        self.active = BranchRuntimeContextV1.create(
            self.config["model_config"],
            seed=self.model_seed,
            instance_id="logical-prefix",
            run_id=self.context["run_id"],
            delta=0.5,
            max_steps=self.config["horizon"],
            policy_context=self.policy_context,
            sampling_context=self.context,
        )
        self.work["lifecycle"] += self.active.event_count
        return self.active

    def close_context(self) -> int:
        started = time.perf_counter_ns()
        if self.active is not None:
            self.active.close()
            self.active = None
        return time.perf_counter_ns() - started

    def close(self) -> None:
        try:
            self.close_context()
        finally:
            self.loop.close()

    def advance(
        self, runtime: BranchRuntimeContextV1, count: int, identity: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from .inference_metrics import InferenceMetrics, metric_span

        profile = self.config.get("validation_profile", "strict")
        scope: AbstractContextManager[LoadedPolicy | ReferencePPORolloutScope] = nullcontext(
            self.policy
        )
        if profile == "rollout-scoped-v1":
            from .reference_ppo import open_rollout_scope

            scope = open_rollout_scope(self.policy)
        meter = InferenceMetrics(
            phase=identity["segment"], enabled=self.config.get("inference_metrics", False)
        )
        costs: dict[str, Any] = {}
        bind_ns = final_ns = 0
        self.partial_trace = {"logical_identity": identity, "records": []}
        try:
            with meter:
                tick = time.perf_counter_ns()
                try:
                    with scope as policy:
                        bind_ns = time.perf_counter_ns() - tick
                        try:
                            rows, costs = self._advance_steps(
                                runtime, count, identity, policy, metric_span
                            )
                        finally:
                            tick = time.perf_counter_ns()
                finally:
                    final_ns = time.perf_counter_ns() - tick
        except BaseException:
            # The collector has closed even on body/final-validation failure.
            # Keep failed work diagnostic-only; do not add it to successful costs.
            self.partial_trace["failed_inference_metrics"] = meter.snapshot()
            raise
        costs["scope_bind_ns"] = bind_ns if profile != "strict" else 0
        costs["scope_final_verify_ns"] = final_ns if profile != "strict" else 0
        costs["policy_ns"] += costs["scope_bind_ns"] + costs["scope_final_verify_ns"]
        costs["inference_metrics"] = meter.snapshot()
        return rows, costs

    def _advance_steps(
        self,
        runtime: BranchRuntimeContextV1,
        count: int,
        identity: dict[str, Any],
        policy: Any,
        metric_span: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from .reference_ppo import PPOInferenceInputV1

        records: list[dict[str, Any]] = []
        policy_ns = step_ns = 0
        features = (
            policy.features
            if self.config.get("validation_profile") == "rollout-scoped-v1"
            else self.features
        )
        self.partial_trace = {"logical_identity": identity, "records": records}
        event_start, step_start = runtime.event_count, runtime.step_id
        try:
            for _ in range(count):
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("frozen branch decision deadline exceeded")
                step_id = runtime.step_id
                observation = runtime.observation
                tick = time.perf_counter_ns()
                with metric_span("policy_total"):
                    inference = PPOInferenceInputV1(
                        observation=observation,
                        action_mask=features.observation_action_mask(observation),
                        sampling_seed=identity["sampling_seed"],
                        run_id=identity["run_id"],
                        generation=0,
                        worker_id=identity["worker_id"],
                        episode_id=identity["episode_id"],
                        step_id=step_id,
                        explore=True,
                    )
                    action = self.loop.run_until_complete(
                        policy.actions({identity["worker_id"]: inference})
                    )[identity["worker_id"]]
                policy_ns += time.perf_counter_ns() - tick
                event_cursor = runtime.physical_event_length
                tick = time.perf_counter_ns()
                following, reward, terminated, truncated, _info = runtime.step(action)
                step_ns += time.perf_counter_ns() - tick
                events = runtime.physical_events_since(event_cursor)
                records.append(
                    {
                        "step_id": runtime.step_id,
                        "inference": json.loads(canonical(inference.to_dict())),
                        "action": action,
                        "observation": following,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "logical_time": runtime.logical_time,
                        "physical_events": events,
                    }
                )
                if (terminated or truncated) and len(records) != count:
                    raise BranchExecutionError(
                        "episode ended before the planned decision boundary", self.partial_trace
                    )
        finally:
            self.work[identity["segment"]] += runtime.event_count - event_start
            self.work[identity["segment"] + "_steps"] += runtime.step_id - step_start
        return records, {"policy_ns": policy_ns, "step_ns": step_ns}

    def run_prefix(self) -> tuple[dict[str, Any], dict[str, Any]]:
        from .models.queue_snapshot import state_view

        tick = time.perf_counter_ns()
        runtime = self.create()
        initialize_ns = time.perf_counter_ns() - tick
        initial = json.loads(canonical(runtime.observation))
        lifecycle_events = runtime.event_count
        tick = time.perf_counter_ns()
        rows, costs = self.advance(runtime, self.config["prefix_steps"], self.context)
        prefix_ns = time.perf_counter_ns() - tick
        prefix = {
            "schema_version": "live-branch-prefix-trace-v1",
            "model_seed": self.model_seed,
            "arrival_tape": runtime.graph.arrival_tape.content(),
            "initial_observation": initial,
            "logical_identity": self.context,
            "policy_context": self.policy_context,
            "records": rows,
            "prefix_reward": sum(row["reward"] for row in rows),
            "end_observation": runtime.observation,
            "end_state": state_view(runtime),
            "events": runtime.event_count - lifecycle_events,
            "decision_steps": self.config["prefix_steps"],
        }
        prefix["prefix_identity"] = digest(
            {
                "config": self.config["model_config"],
                "model_seed": self.model_seed,
                "policy_context": self.policy_context,
                "logical_identity": self.context,
                "records_sha256": digest(rows),
                "prefix_steps": self.config["prefix_steps"],
            }
        )
        return prefix, {
            "initialize_ns": initialize_ns,
            "prefix_ns": prefix_ns,
            "prefix_policy_ns": costs["policy_ns"],
            "prefix_step_ns": costs["step_ns"],
            "lifecycle_events": lifecycle_events,
            "scope_bind_ns": costs["scope_bind_ns"],
            "scope_final_verify_ns": costs["scope_final_verify_ns"],
            "prefix_inference_metrics": costs["inference_metrics"],
        }

    def capture_prefix(self) -> dict[str, Any]:
        from .models.queue_snapshot import capture

        result: dict[str, Any] = {}
        try:
            prefix, costs = self.run_prefix()
            tick = time.perf_counter_ns()
            self.snapshot = capture(
                cast("BranchRuntimeContextV1", self.active), prefix["prefix_identity"]
            )
            costs["capture_ns"] = time.perf_counter_ns() - tick
            self.prefix = prefix
            result.update(prefix=prefix, snapshot=self.snapshot, costs=costs)
            return result
        finally:
            result.setdefault("costs", {})["close_ns"] = self.close_context()


def _execute_branch(engine: _Engine, branch_id: str, snapshot_mode: bool) -> dict[str, Any]:
    from .models.queue_snapshot import restore, state_view

    cfg = engine.config
    identity = logical_identity(cfg, branch_id)
    costs: dict[str, Any] = {}
    result: dict[str, Any] = {}
    engine.partial_trace = None
    try:
        if snapshot_mode:
            if engine.snapshot is None or engine.prefix is None:
                raise ValueError("snapshot not installed")
            if engine.prefix["policy_context"] != engine.policy_context:
                raise ValueError("snapshot policy differs from frozen worker policy")
            prefix = engine.prefix
            tick = time.perf_counter_ns()
            engine.active = restore(
                engine.snapshot,
                {
                    "instance_id": branch_id,
                    "run_id": identity["run_id"],
                    "sampling_context": identity,
                },
            )
            costs["restore_ns"] = time.perf_counter_ns() - tick
            costs["lifecycle_events"] = engine.active.event_count
            engine.work["lifecycle"] += engine.active.event_count
            actual_prefix_events = 0
        else:
            prefix, costs = engine.run_prefix()
            actual_prefix_events = prefix["events"]
        runtime = cast("BranchRuntimeContextV1", engine.active)
        if canonical(runtime.observation) != canonical(prefix["end_observation"]):
            raise ValueError("branch starts from a different prefix observation")
        prior_events = runtime.event_count
        tick = time.perf_counter_ns()
        rows, suffix_costs = engine.advance(runtime, cfg["suffix_steps"], identity)
        costs["suffix_ns"] = time.perf_counter_ns() - tick
        costs["suffix_policy_ns"] = suffix_costs["policy_ns"]
        costs["suffix_step_ns"] = suffix_costs["step_ns"]
        costs["scope_bind_ns"] = costs.get("scope_bind_ns", 0) + suffix_costs["scope_bind_ns"]
        costs["scope_final_verify_ns"] = (
            costs.get("scope_final_verify_ns", 0) + suffix_costs["scope_final_verify_ns"]
        )
        costs["suffix_inference_metrics"] = suffix_costs["inference_metrics"]
        suffix_events = runtime.event_count - prior_events
        final = state_view(runtime)
        trace = {
            "schema_version": "live-branch-suffix-trace-v1",
            "branch_id": branch_id,
            "prefix_sha256": digest(prefix),
            "logical_identity": identity,
            "records": rows,
            "final_state": final,
        }
        prefix_reward = prefix["prefix_reward"]
        suffix_reward = sum(row["reward"] for row in rows)
        summary = {
            "branch_id": branch_id,
            "logical_identity": identity,
            "logical_identity_sha256": digest(identity),
            "semantic_sha256": digest(trace),
            "prefix_sha256": digest(prefix),
            "prefix_identity": prefix["prefix_identity"],
            "policy_sha256": engine.policy_context["policy_sha256"],
            "prefix_reward": prefix_reward,
            "suffix_reward": suffix_reward,
            "return": prefix_reward + suffix_reward,
            "final_observation": runtime.observation,
            "final_state_sha256": digest(final),
            "terminated": rows[-1]["terminated"],
            "truncated": rows[-1]["truncated"],
            "research_cut": not (rows[-1]["terminated"] or rows[-1]["truncated"]),
            "prefix_events": prefix["events"],
            "suffix_events": suffix_events,
            "actual_prefix_events": actual_prefix_events,
            "actual_decision_steps": cfg["suffix_steps"]
            + (0 if snapshot_mode else cfg["prefix_steps"]),
            "action_sha256": digest([row["action"] for row in rows]),
        }
        result.update(summary=summary, prefix=prefix, trace=trace, costs=costs)
        return result
    finally:
        costs["close_ns"] = engine.close_context()


def _worker_main(connection: Any, config: dict[str, Any], deadline: float) -> None:
    engine = None
    try:
        engine = _Engine(config)
        engine.deadline = deadline
        connection.send({"ok": True, "pid": os.getpid(), "ready": engine.policy_context})
        while True:
            request = connection.recv()
            request_id, op = request["request_id"], request["op"]
            try:
                if op == "close":
                    engine.close()
                    work = engine.work
                    engine = None
                    connection.send(
                        {"ok": True, "request_id": request_id, "closed": True, "work": work}
                    )
                    break
                if op == "capture":
                    payload = engine.capture_prefix()
                elif op == "install":
                    engine.snapshot, engine.prefix = request["snapshot"], request["prefix"]
                    if (
                        cast(dict[str, Any], engine.prefix)["policy_context"]
                        != engine.policy_context
                    ):
                        raise ValueError("installed snapshot policy differs")
                    payload = {
                        "snapshot_sha256": cast("SimulatorSnapshotV1", engine.snapshot).sha256
                    }
                elif op == "branch":
                    payload = _execute_branch(
                        engine, request["branch_id"], request["snapshot_mode"]
                    )
                else:
                    raise ValueError("unknown branch operation")
                connection.send(
                    {"ok": True, "request_id": request_id, "payload": payload, "work": engine.work}
                )
            except BaseException as exc:
                connection.send(
                    {
                        "ok": False,
                        "request_id": request_id,
                        "failure": _error(exc, op),
                        "diagnostics": getattr(exc, "diagnostics", None)
                        or cast(_Engine, engine).partial_trace,
                        "work": cast(_Engine, engine).work,
                    }
                )
                break
    except BaseException as exc:
        with suppress(OSError, EOFError, BrokenPipeError):
            connection.send({"ok": False, "failure": _error(exc, "worker")})
    finally:
        if engine is not None:
            # A prior failed request remains failed; parent checks process exit.
            with suppress(BaseException):
                engine.close()
        connection.close()


class _Pool:
    def __init__(self, config: dict[str, Any], deadline: float) -> None:
        self.deadline = deadline
        self.workers: list[dict[str, Any]] = []
        self.next_id = 0
        context = mp.get_context("spawn")
        for index in range(4):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(child, config, self.deadline),
                name=f"live-branch-worker-{index}",
            )
            worker = {
                "process": process,
                "connection": parent,
                "pending": None,
                "started": False,
                "work": {
                    "prefix": 0,
                    "suffix": 0,
                    "lifecycle": 0,
                    "prefix_steps": 0,
                    "suffix_steps": 0,
                },
                "work_complete": False,
            }
            self.workers.append(worker)
            try:
                process.start()
                worker["started"] = True
            finally:
                child.close()

    def ready(self) -> dict[str, Any]:
        policy = None
        for worker in self.workers:
            response = self.receive(worker, handshake=True)
            if response["pid"] != worker["process"].pid:
                raise ValueError("worker handshake PID differs")
            if policy is not None and policy != response["ready"]:
                raise ValueError("worker frozen policy differs")
            policy = response["ready"]
        return cast(dict[str, Any], policy)

    def send(self, worker: dict[str, Any], op: str, **payload: Any) -> None:
        if worker["pending"] is not None:
            raise ValueError("worker already has a pending individual request")
        self.next_id += 1
        worker["connection"].send({"request_id": self.next_id, "op": op, **payload})
        worker["pending"] = self.next_id
        if op != "close":
            worker["work_complete"] = False

    def receive(self, worker: dict[str, Any], *, handshake: bool = False) -> dict[str, Any]:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or not worker["connection"].poll(remaining):
            raise TimeoutError("individual branch worker response deadline exceeded")
        response = worker["connection"].recv()
        if "work" in response:
            worker["work"] = response["work"]
            worker["work_complete"] = True
        if not handshake and response.get("request_id") != worker["pending"]:
            raise ValueError("branch worker response identity differs")
        worker["pending"] = None
        if not response.get("ok"):
            raise BranchExecutionError("worker operation failed", response)
        return cast(dict[str, Any], response if handshake else response["payload"])

    def close(self, final_deadline: float) -> list[dict[str, Any]]:
        receipts = []
        natural_deadline = min(final_deadline - 2.0, time.monotonic() + 8.0)
        for worker in self.workers:
            if worker["started"] and worker["process"].is_alive() and worker["pending"] is None:
                with suppress(OSError, EOFError, BrokenPipeError):
                    self.send(worker, "close")
        for worker in self.workers:
            process = worker["process"]
            if not worker["started"]:
                worker["connection"].close()
                receipts.append(
                    {
                        "pid": None,
                        "disposition": "not-started",
                        "exitcode": None,
                        "exited": True,
                        "close_reply": False,
                    }
                )
                continue
            close_reply = False
            try:
                if worker["connection"].poll(max(0.0, min(1.0, final_deadline - time.monotonic()))):
                    response = worker["connection"].recv()
                    if "work" in response:
                        worker["work"] = response["work"]
                        worker["work_complete"] = True
                    close_reply = bool(
                        response.get("closed")
                        and response.get("ok")
                        and response.get("request_id") == worker["pending"]
                    )
            except (OSError, EOFError, BrokenPipeError):
                pass
            disposition = "natural"
            process.join(timeout=max(0.0, natural_deadline - time.monotonic()))
            if process.is_alive():
                disposition = "terminate"
                worker["work_complete"] = False
                process.terminate()
                process.join(timeout=max(0.0, min(1.0, final_deadline - time.monotonic())))
            if process.is_alive():
                disposition = "kill"
                process.kill()
                process.join(timeout=max(0.0, min(1.0, final_deadline - time.monotonic())))
            exited = not process.is_alive()
            receipts.append(
                {
                    "pid": process.pid,
                    "disposition": disposition if exited else "unconfirmed",
                    "exitcode": process.exitcode,
                    "exited": exited,
                    "close_reply": close_reply,
                }
            )
            worker["connection"].close()
            if exited:
                process.close()
        return receipts


def _write_new(path: Path, payload: Any) -> dict[str, Any]:
    body = canonical(payload)
    with path.open("xb") as stream:
        stream.write(body)
    return {"path": path.name, "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}


def run_arm(
    config: Mapping[str, Any], arm: str, *, trace_dir: Path | None = None
) -> dict[str, Any]:
    """Execute one arm only. No automatic retry or independent-learning claim."""
    cfg = validate_config(config)
    if arm not in ARMS:
        raise ValueError("unknown branch arm")
    if trace_dir is not None:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter_ns()
    final_deadline = time.monotonic() + cfg["operation_timeout_seconds"]
    operation_deadline = final_deadline - min(10.0, cfg["operation_timeout_seconds"] / 10)
    engine = pool = None
    branches: dict[str, dict[str, Any]] = {}
    traces: dict[str, Any] = {"prefix": None, "branches": {}}
    costs: list[dict[str, Any]] = []
    cleanup: list[dict[str, Any]] = []
    failure = None
    snapshot = None
    policy_context = None
    warm_start = warm_ns = None
    transfer_ns = 0
    phase = "startup"

    def admit(result: dict[str, Any]) -> None:
        branch = result["summary"]
        branch_id = branch["branch_id"]
        if branch_id in branches or branch["logical_identity"] != logical_identity(cfg, branch_id):
            raise ValueError("duplicate or wrong logical branch admission")
        prefix = result["prefix"]
        if traces["prefix"] is None:
            traces["prefix"] = prefix
        elif canonical(traces["prefix"]) != canonical(prefix):
            raise ValueError("full replay does not reproduce the same prefix")
        branches[branch_id] = branch
        traces["branches"][branch_id] = result["trace"]
        costs.append(result["costs"])

    try:
        if arm.startswith("P"):
            # Allocate before start(), so partially constructed pools remain owned.
            pool = _Pool.__new__(_Pool)
            pool.workers = []
            _Pool.__init__(pool, cfg, operation_deadline)
            policy_context = pool.ready()
        else:
            engine = _Engine(cfg)
            engine.deadline = operation_deadline
            policy_context = engine.policy_context
        startup_ns = time.perf_counter_ns() - started
        warm_start = time.perf_counter_ns()
        phase = "prefix"
        if arm.endswith("1"):
            if pool:
                pool.send(pool.workers[0], "capture")
                prepared = pool.receive(pool.workers[0])
            else:
                prepared = cast(_Engine, engine).capture_prefix()
            snapshot, traces["prefix"] = prepared["snapshot"], prepared["prefix"]
            costs.append(prepared["costs"])
            if pool:
                tick = time.perf_counter_ns()
                for worker in pool.workers:
                    pool.send(worker, "install", snapshot=snapshot, prefix=traces["prefix"])
                for worker in pool.workers:
                    installed = pool.receive(worker)
                    if installed["snapshot_sha256"] != snapshot.sha256:
                        raise ValueError("worker installed another snapshot")
                transfer_ns += time.perf_counter_ns() - tick
        phase = "branches"
        ids = cfg["branch_ids"]
        for start in range(0, len(ids), 4 if pool else 1):
            if time.monotonic() >= operation_deadline:
                raise TimeoutError("branch arm operation deadline exceeded")
            wave = ids[start : start + (4 if pool else 1)]
            if pool:
                for worker, branch_id in zip(pool.workers, wave, strict=False):
                    pool.send(
                        worker, "branch", branch_id=branch_id, snapshot_mode=arm.endswith("1")
                    )
                errors = []
                for worker in pool.workers[: len(wave)]:
                    try:
                        admit(pool.receive(worker))
                    except BaseException as exc:
                        errors.append(_error(exc, "branch"))
                if errors:
                    raise BranchExecutionError("one or more branch workers failed", errors)
            else:
                admit(_execute_branch(cast(_Engine, engine), wave[0], arm.endswith("1")))
        warm_ns = time.perf_counter_ns() - warm_start
    except BaseException as exc:
        startup_ns = locals().get("startup_ns", time.perf_counter_ns() - started)
        failure = _error(exc, phase)
        failure["diagnostics"] = getattr(exc, "diagnostics", None) or (
            engine.partial_trace if engine is not None else None
        )
    finally:
        close_start = time.perf_counter_ns()
        if pool is not None:
            cleanup = pool.close(final_deadline)
            if any(
                not row["exited"]
                or row["disposition"] != "natural"
                or row["exitcode"] != 0
                or not row["close_reply"]
                for row in cleanup
            ):
                failure = failure or {
                    "type": "BranchCloseError",
                    "phase": "cleanup",
                    "message": "one or more workers did not close cleanly",
                }
        if engine is not None:
            try:
                engine.close()
                cleanup = [
                    {
                        "pid": os.getpid(),
                        "disposition": "inprocess-resources-closed",
                        "exited": True,
                        "exitcode": None,
                        "close_reply": True,
                    }
                ]
            except BaseException as exc:
                failure = failure or _error(exc, "cleanup")
                cleanup = [
                    {
                        "pid": os.getpid(),
                        "disposition": "unconfirmed",
                        "exited": False,
                        "exitcode": None,
                        "close_reply": False,
                    }
                ]
        close_ns = time.perf_counter_ns() - close_start
    ordered = [branches[key] for key in sorted(branches)]
    work_sources = (
        [worker["work"] for worker in pool.workers] if pool else ([engine.work] if engine else [])
    )
    work = {
        key: sum(source[key] for source in work_sources)
        for key in ("prefix", "suffix", "lifecycle", "prefix_steps", "suffix_steps")
    }
    work_complete = (
        all(worker["work_complete"] for worker in pool.workers) if pool else engine is not None
    )
    work_complete = work_complete and failure is None
    timing = {
        key: sum(item.get(key, 0) for item in costs)
        for key in (
            "initialize_ns",
            "prefix_ns",
            "prefix_policy_ns",
            "prefix_step_ns",
            "capture_ns",
            "restore_ns",
            "suffix_ns",
            "suffix_policy_ns",
            "suffix_step_ns",
            "close_ns",
            "scope_bind_ns",
            "scope_final_verify_ns",
        )
    }
    timing.update(
        startup_ns=startup_ns,
        warm_ns=warm_ns,
        transfer_ns=transfer_ns,
        supervisor_close_ns=close_ns,
    )
    semantic_config = {
        key: value
        for key, value in cfg.items()
        if key not in ("branch_ids", "operation_timeout_seconds")
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "domain": DOMAIN,
        "arm": arm,
        "master": cfg["master"],
        "phase": cfg["phase"],
        "prefix_steps": cfg["prefix_steps"],
        "branch_count": cfg["branches"],
        "suffix_steps": cfg["suffix_steps"],
        "model_seed": derive_seed(cfg["phase"], "model", cfg["master"]),
        "prefix_logical_identity": logical_identity(cfg),
        "config_sha256": digest(semantic_config),
        "policy_context": policy_context,
        "execution_complete": failure is None and len(ordered) == cfg["branches"],
        "failure": failure,
        "branches": ordered,
        "completed_branches": len(ordered),
        "requested_branch_order": cfg["branch_ids"],
        "simulation_workers": 4 if pool else 1,
        "cleanup": cleanup,
        "all_owned_processes_exited": all(row["exited"] for row in cleanup),
        "process_exit_scope": "owned worker children only; caller verifies outer arm process exit",
        "actual_events": {
            "prefix": work["prefix"],
            "suffix": work["suffix"],
            "total": work["prefix"] + work["suffix"],
            "lifecycle": work["lifecycle"],
            "measurement_complete": work_complete,
        },
        "actual_decision_steps": work["prefix_steps"] + work["suffix_steps"],
        "event_count_scope": (
            "internal plus external model transition callbacks, including reported failed partial "
            "work; incomplete if worker forcibly lost"
        ),
        "timing_ns": timing,
        "timing_scope": (
            "stage totals may overlap across workers; warm includes branch cleanup; "
            "cold owned by caller"
        ),
        "snapshot": None
        if snapshot is None
        else {"sha256": snapshot.sha256, "size_bytes": snapshot.size_bytes},
        "unique_suffix_actions": len({row["action_sha256"] for row in ordered}),
        "unique_final_states": len({row["final_state_sha256"] for row in ordered}),
    }
    if cfg["schema_version"] == CONFIG_SCHEMA_V2:
        result["schema_version"] = RESULT_SCHEMA_V2
        result["policy_initialization_seed"] = cfg["policy_seed"]
        result["execution_profile"] = {
            "validation_profile": cfg["validation_profile"],
            "inference_metrics": cfg["inference_metrics"],
        }
        result["inference_metrics"] = {
            phase: [
                item[phase + "_inference_metrics"]
                for item in costs
                if phase + "_inference_metrics" in item
            ]
            for phase in ("prefix", "suffix")
        }
    io_started = time.perf_counter_ns()
    if trace_dir is None:
        result["traces"] = traces
    else:
        files = {}
        if traces["prefix"] is not None:
            files["prefix"] = _write_new(trace_dir / "prefix.json", traces["prefix"])
        for branch_id, trace in sorted(traces["branches"].items()):
            files[branch_id] = _write_new(trace_dir / f"{branch_id}.json", trace)
        if snapshot is not None:
            path = trace_dir / f"snapshot-{snapshot.sha256}.json"
            with path.open("xb") as stream:
                stream.write(snapshot.data)
            files["snapshot"] = {
                "path": path.name,
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
            }
        result["trace_files"] = files
    result["timing_ns"]["trace_io_ns"] = time.perf_counter_ns() - io_started
    result["inprocess_ns"] = time.perf_counter_ns() - started
    return result


def compare_arms(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail closed on identity/missing/failed rows before physical projections."""
    arms = [result.get("arm") for result in results]
    if (
        not 2 <= len(results) <= 4
        or len(set(arms)) != len(arms)
        or any(arm not in ARMS for arm in arms)
    ):
        raise ValueError("comparison requires two to four distinct declared arms")
    reference = results[0]
    identity_fields = (
        "schema_version",
        "domain",
        "master",
        "phase",
        "prefix_steps",
        "branch_count",
        "suffix_steps",
        "model_seed",
        "config_sha256",
        "policy_context",
        "execution_profile",
        "policy_initialization_seed",
    )
    branch_fields = (
        "branch_id",
        "logical_identity",
        "logical_identity_sha256",
        "prefix_sha256",
        "prefix_identity",
        "policy_sha256",
        "semantic_sha256",
        "prefix_reward",
        "suffix_reward",
        "return",
        "final_observation",
        "final_state_sha256",
        "terminated",
        "truncated",
        "research_cut",
        "prefix_events",
        "suffix_events",
    )
    for result in results:
        if result.get("schema_version") == RESULT_SCHEMA_V2:
            profile = result.get("execution_profile")
            if (
                not isinstance(profile, dict)
                or set(profile) != {"validation_profile", "inference_metrics"}
                or profile["validation_profile"] not in VALIDATION_PROFILES
                or type(profile["inference_metrics"]) is not bool
            ):
                raise ValueError("v2 execution profile identity is missing or malformed")
            _integer(result.get("policy_initialization_seed"), "policy_initialization_seed")
        elif result.get("schema_version") != RESULT_SCHEMA:
            raise ValueError("unknown arm result schema")
        if not result.get("execution_complete") or result.get("failure") is not None:
            raise ValueError("failed or incomplete arm is not a semantic comparison sample")
        if not result.get("all_owned_processes_exited"):
            raise ValueError("unconfirmed owned-process exit prevents admission")
        if any(
            canonical(result.get(key)) != canonical(reference.get(key)) for key in identity_fields
        ):
            raise ValueError("arm work/policy identity differs")
        rows = result.get("branches", [])
        if len(rows) != reference["branch_count"] or [row["branch_id"] for row in rows] != [
            f"branch-{index:04d}" for index in range(1, reference["branch_count"] + 1)
        ]:
            raise ValueError("missing, duplicate or unordered logical branches")
        for row, expected in zip(rows, reference["branches"], strict=False):
            correct = logical_identity(result, row["branch_id"])
            if canonical(row.get("logical_identity")) != canonical(correct) or row.get(
                "logical_identity_sha256"
            ) != digest(correct):
                raise ValueError("branch logical sampling identity differs")
            if any(
                canonical(row.get(key)) != canonical(expected.get(key)) for key in branch_fields
            ):
                raise ValueError("branch physical or logical result differs")
        events = result.get("actual_events", {})
        prefix_repetitions = 1 if result["arm"].endswith("1") else len(rows)
        expected_prefix = prefix_repetitions * rows[0]["prefix_events"]
        expected_suffix = sum(row["suffix_events"] for row in rows)
        if (
            events.get("measurement_complete") is not True
            or events.get("prefix") != expected_prefix
            or events.get("suffix") != expected_suffix
            or events.get("total") != expected_prefix + expected_suffix
            or result.get("actual_decision_steps")
            != (prefix_repetitions * result["prefix_steps"] + len(rows) * result["suffix_steps"])
        ):
            raise ValueError("actual executor work differs from the declared branch strategy")
        if (
            "traces" in reference
            and "traces" in result
            and canonical(result["traces"]) != canonical(reference["traces"])
        ):
            raise ValueError("full physical trace differs")
    return {
        "equivalent": True,
        "arms": arms,
        "branches": reference["branch_count"],
        "scope": (
            "same logical identities and physical projections; "
            "not independent equation verification"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("branch output must be new; failed attempts are never replaced")
    result = run_arm(
        json.loads(args.config.read_text(encoding="utf-8")), args.arm, trace_dir=args.trace_dir
    )
    _write_new(args.output, result)
    return 0 if result["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
