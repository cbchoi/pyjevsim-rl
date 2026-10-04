"""Installed fixed-delta local PPO runner with episode/update boundary recovery.

This module never selects model behavior by model ID. Unqualified synthetic
executors are explicit development fixtures, not real-model qualification.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import math
import platform
import sys
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar, cast

from . import _runner_records as wire
from . import _runner_storage as storage
from .contracts import EpisodeContext, StepView
from .environment import PyJevSimEnv
from .executor import (
    PINNED_PYJEVSIM_2_1_2_PROFILE,
    ExecutorQualificationPolicy,
    ExecutorSemanticCapability,
    FixedDeltaBoundary,
)
from .learning import (
    ActorPolicy,
    InMemoryPolicyArtifactStore,
    LearnerRecoveryState,
    LearnerSession,
    PolicyArtifact,
    PolicyArtifactRef,
    PolicyCompatibility,
    ValidatedTransitionBatch,
)
from .local import LocalRolloutPool
from .ppo_features import bind_feature_contract, validate_action_mask
from .records import TransitionRecord
from .reference_ppo import (
    REFERENCE_PPO_ALGORITHM_ID,
    REFERENCE_PPO_ALGORITHM_VERSION,
    PPOInferenceInputV1,
    ReferencePPOConfigV1,
    ReferencePPOLearnerAdapter,
    ReferencePPOPolicyLoader,
    numpy_runtime_sha256,
)

RunnerConfig = wire.RunnerConfig
RunnerContractError = wire.RunnerContractError
load_config = wire.load_config
export_checkpoint = storage.export_checkpoint
validate_archive = storage.validate_archive
_T = TypeVar("_T")

_FRAMEWORK_MODULES = (
    "_runner_records",
    "_runner_storage",
    "training_runner",
    "ppo_features",
    "reference_ppo",
    "learning",
    "local",
    "local_process",
    "local_wire",
    "_local_process_worker",
    "environment",
    "executor",
    "contracts",
    "adapters",
    "records",
)
_VOLATILE = frozenset(
    {
        "_runner_attempt",
        "reset_attempt",
        "pid",
        "process_id",
        "process_incarnation",
        "incarnation",
        "wall_time",
        "wall_clock",
        "timestamp",
        "source_path",
        "local_episode_id",
    }
)
_RESERVED_INFO = _VOLATILE | frozenset(
    {
        "run_id",
        "generation",
        "episode_id",
        "instance_id",
        "worker_id",
        "step_id",
        "logical_time",
        "seed",
        "executor_qualified",
        "executor_qualification_policy_id",
        "previous_action_mask",
        "next_action_mask",
        "behavior_artifact_sha256",
        "evaluation",
        "environment_terminated",
        "environment_truncated",
        "administrative_cut",
    }
)
_EPISODE_FIELDS = frozenset(
    {
        "job_index",
        "worker_id",
        "episode_id",
        "seed",
        "case_index",
        "config_sha256",
        "final_step",
        "logical_time",
        "terminated",
        "truncated",
        "cut_reason",
        "return",
        "objective_cost",
        "policy_version",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "run_id",
        "generation",
        "policy_version",
        "payload_hex",
        "media_type",
        "compatibility",
        "source_batch_sha256",
        "provenance",
        "sha256",
    }
)


class RunnerExecutionError(RuntimeError):
    """An invocation failed; its immutable receipt retains the cause and cleanup."""

    def __init__(self, message: str, receipt: Mapping[str, object]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


@dataclass(frozen=True)
class RunnerResult:
    output: Path
    checkpoint: Path | None
    receipt: Mapping[str, object]


def _framework_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: wire.byte_digest((root / f"{name}.py").read_bytes()) for name in _FRAMEWORK_MODULES
    }


def _executor_policy(profile_id: str | None) -> ExecutorQualificationPolicy | None:
    if profile_id is None:
        return None
    profile = PINNED_PYJEVSIM_2_1_2_PROFILE
    if profile_id != profile.profile_id:
        raise RunnerContractError("executor profile is not the reviewed pinned profile")
    module = importlib.import_module(profile.executor_module)
    executor_type = getattr(module, profile.executor_qualname)
    policy = ExecutorQualificationPolicy(
        profile.issue(executor_type),
        (
            ExecutorSemanticCapability.CONFLUENT_TRANSITION,
            ExecutorSemanticCapability.ZERO_TIME_CASCADE,
        ),
    )
    policy.preflight()
    return policy


@dataclass
class _Runtime:
    config: RunnerConfig

    def __post_init__(self) -> None:
        self.data = self.config.content()
        modules = wire.verify_sources(self.config)
        self.feature = wire.import_callable(self.data["feature_factory"])()
        self.binding = bind_feature_contract(self.feature)
        if type(self.feature).__module__ not in modules:
            raise RunnerContractError("feature defining module absent from source inventory")
        self.ppo = ReferencePPOConfigV1.from_dict(self.data["ppo_config"])
        if (self.ppo.observation_size, self.ppo.action_count) != (
            self.binding.feature_size,
            self.binding.action_count,
        ):
            raise RunnerContractError("config and feature dimensions differ")
        self.policy = _executor_policy(self.data["executor_profile_id"])
        self.compatibility = PolicyCompatibility(
            REFERENCE_PPO_ALGORITHM_ID,
            REFERENCE_PPO_ALGORITHM_VERSION,
            self.data["model_id"],
            self.data["model_version"],
            self.feature.contract_sha256,
            self.feature.action_schema_sha256,
        )
        self.source_identity = {
            "model_modules": modules,
            "feature_profile": self.binding.profile,
            "framework_modules": _framework_sources(),
        }
        self.runtime_identity = {
            "python": platform.python_version(),
            "numpy_runtime_sha256": numpy_runtime_sha256(),
            "executor_profile_id": self.data["executor_profile_id"],
        }
        self.schedule = wire.build_schedule(self.config)

    def assert_current(self) -> None:
        if wire.verify_sources(self.config) != self.source_identity["model_modules"]:
            raise RunnerContractError("model sources changed")
        if _framework_sources() != self.source_identity["framework_modules"]:
            raise RunnerContractError("framework sources changed during invocation")
        if numpy_runtime_sha256() != self.runtime_identity["numpy_runtime_sha256"]:
            raise RunnerContractError("runtime changed during invocation")
        self.binding.assert_current()


class _GlobalBinding:
    def __init__(self, binding: Any, episode_id: str) -> None:
        self._binding = binding
        self._episode_id = episode_id
        self.executor = binding.executor
        self._original: StepView | None = None
        self._mapped: StepView | None = None

    def initialize(self) -> None:
        callback = getattr(self._binding, "initialize", None)
        if callable(callback):
            callback()

    def _view(self, view: StepView) -> StepView:
        if view is not self._original:
            self._original = view
            self._mapped = replace(view, episode_id=self._episode_id)
        return cast(StepView, self._mapped)

    def apply_action(self, action: object) -> None:
        self._binding.apply_action(action)

    def next_decision_time(self) -> float:
        return float(self._binding.next_decision_time())

    def observe(self, events: object) -> object:
        return self._binding.observe(events)

    def reward(self, view: StepView) -> float:
        return cast(float, self._binding.reward(self._view(view)))

    def terminated(self, view: StepView) -> bool:
        return cast(bool, self._binding.terminated(self._view(view)))

    def info(self, view: StepView) -> Mapping[str, object]:
        result = self._binding.info(self._view(view))
        if not isinstance(result, Mapping) or _RESERVED_INFO.intersection(result):
            raise RunnerContractError("model info overrides reserved framework fields")
        return cast(Mapping[str, object], result)

    def close(self) -> None:
        self._binding.close()


class _ScheduledEnvironment:
    def __init__(self, config: RunnerConfig, worker_id: str, phase: str) -> None:
        self.config = config
        self.worker_id = worker_id
        self.data = config.content()
        wire.verify_sources(config)
        self.feature = bind_feature_contract(wire.import_callable(self.data["feature_factory"])())
        self.jobs = {row["seed"]: row for row in wire.build_schedule(config)[phase]}
        self.current: dict[str, Any] | None = None
        self.environment = PyJevSimEnv(
            self._factory,
            instance_id=worker_id,
            run_id=self.data["run_id"],
            boundary=FixedDeltaBoundary(self.data["boundary_delta"]),
            max_steps=self.data["max_steps"],
            executor_qualification=_executor_policy(self.data["executor_profile_id"]),
            require_claim_grade=self.data["executor_profile_id"] is not None,
        )

    def _factory(self, context: EpisodeContext) -> _GlobalBinding:
        if self.current is None or context.seed != self.current["seed"]:
            raise RunnerContractError("factory context differs from scheduled job")
        wire.verify_sources(self.config)
        translated = EpisodeContext(
            episode_id=self.current["episode_id"],
            instance_id=context.instance_id,
            seed=context.seed,
            options=MappingProxyType(dict(self.current["model_config"])),
        )
        binding = wire.import_callable(self.data["model_factory"])(translated)
        return _GlobalBinding(binding, translated.episode_id)

    def _info(self, observation: object, raw: Mapping[str, object]) -> dict[str, object]:
        if self.current is None:
            raise RunnerContractError("environment has no scheduled job")
        mask = self.feature.observation_action_mask(observation)
        if (
            "action_mask" in raw
            and validate_action_mask(raw["action_mask"], self.feature.action_count) != mask
        ):
            raise RunnerContractError("environment info mask differs from model feature mask")
        info = {key: value for key, value in raw.items() if key not in _VOLATILE}
        info["episode_id"] = self.current["episode_id"]
        info["action_mask"] = mask
        info["_runner_attempt"] = {
            "local_episode_id": raw.get("episode_id"),
            **{key: value for key, value in raw.items() if key in _VOLATILE},
        }
        return info

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, object] | None = None
    ) -> tuple[object, dict[str, object]]:
        if options is not None:
            raise RunnerContractError("mutable reset options are outside the frozen config")
        if type(seed) is not int or seed not in self.jobs:
            raise RunnerContractError("reset seed is absent from immutable job schedule")
        row = self.jobs[seed]
        if row["worker_id"] != self.worker_id:
            raise RunnerContractError("job seed assigned to another worker")
        self.current = row
        observation, info = self.environment.reset(seed=seed, options=row["model_config"])
        return observation, self._info(observation, info)

    def step(self, action: object) -> tuple[object, float, bool, bool, dict[str, object]]:
        observation, reward, terminal, truncated, info = self.environment.step(action)
        return observation, reward, terminal, truncated, self._info(observation, info)

    def close(self) -> None:
        self.environment.close()


@dataclass(frozen=True)
class _EnvironmentProvider:
    config: RunnerConfig
    worker_id: str
    phase: str

    def __call__(self) -> _ScheduledEnvironment:
        return _ScheduledEnvironment(self.config, self.worker_id, self.phase)


def _pool(runtime: _Runtime, workers: Sequence[str], phase: str) -> LocalRolloutPool:
    return LocalRolloutPool(
        {worker: _EnvironmentProvider(runtime.config, worker, phase) for worker in workers},
        run_id=runtime.data["run_id"],
        generation=runtime.data["generation"],
        backend=runtime.data["backend"],
    )


def _inference(
    runtime: _Runtime,
    observation: object,
    info: Mapping[str, object],
    worker: str,
    episode: str,
    step: int,
    seed: int,
    explore: bool,
) -> PPOInferenceInputV1:
    mask = runtime.binding.observation_action_mask(observation)
    if (
        "action_mask" in info
        and validate_action_mask(info["action_mask"], runtime.binding.action_count) != mask
    ):
        raise RunnerContractError("mask receipt differs from observed state")
    result = PPOInferenceInputV1(
        observation,
        mask,
        seed,
        runtime.data["run_id"],
        runtime.data["generation"],
        worker,
        episode,
        step,
        explore,
    )
    runtime.binding.action_mask(result.to_dict())
    return result


def _record(
    raw: TransitionRecord,
    prior: PPOInferenceInputV1,
    following: PPOInferenceInputV1,
    artifact: str,
    cut: str | None,
    evaluation: bool,
) -> TransitionRecord:
    info = {key: value for key, value in raw.info.items() if key not in _VOLATILE}
    info.update(
        {
            "previous_action_mask": prior.action_mask,
            "next_action_mask": following.action_mask,
            "behavior_artifact_sha256": artifact,
            "evaluation": evaluation,
            "environment_terminated": raw.terminated,
            "environment_truncated": raw.truncated,
            "administrative_cut": cut,
        }
    )
    return replace(
        raw,
        previous_observation=prior.to_dict(),
        next_observation=following.to_dict(),
        truncated=raw.truncated or cut is not None,
        info=info,
    )


def _transport_actions(actions: Mapping[str, object]) -> dict[str, object]:
    # ActorPolicy freezes JSON dictionaries as MappingProxyType; process transport
    # needs a detached, picklable JSON value. Use the same representation for every backend.
    return {worker: wire.decode_json(wire.canonical(value)) for worker, value in actions.items()}


async def _retain_call(
    operation: Awaitable[_T], attempts: list[object], phase: str
) -> _T:
    try:
        return await operation
    except BaseException as error:
        snapshot = _failure(error)
        error._runner_failure_snapshot = snapshot  # type: ignore[attr-defined]
        attempts.append({"phase": phase, "failure": snapshot, "learner_admissible": False})
        raise


def _episode(job: Mapping[str, Any], records: Sequence[TransitionRecord]) -> dict[str, Any]:
    final = records[-1]
    if not (final.terminated or final.truncated):
        raise RunnerContractError("episode is not closed at checkpoint cut")
    costs = [record.info.get("objective_cost") for record in records]
    if any(
        value is not None
        and (type(value) not in (int, float) or not math.isfinite(cast(float, value)))
        for value in costs
    ):
        raise RunnerContractError("objective cost must be finite when provided")
    return {
        **{
            key: job[key]
            for key in (
                "job_index",
                "worker_id",
                "episode_id",
                "seed",
                "case_index",
                "config_sha256",
            )
        },
        "final_step": final.step_id,
        "logical_time": final.logical_time,
        "terminated": final.terminated,
        "truncated": final.truncated,
        "cut_reason": final.info["administrative_cut"]
        or ("domain-termination" if final.terminated else "max-steps"),
        "return": sum(record.reward for record in records),
        "objective_cost": sum(cast(list[float], costs))
        if all(value is not None for value in costs)
        else None,
        "policy_version": final.policy_version,
    }


async def _collect(
    runtime: _Runtime,
    pool: LocalRolloutPool,
    actor: ActorPolicy,
    start_job: int,
    attempts: list[object],
) -> tuple[ValidatedTransitionBatch, list[dict[str, Any]], int]:
    records: list[TransitionRecord] = []
    episodes: list[dict[str, Any]] = []
    fragment: dict[str, Any] = {
        "phase": "training",
        "disposition": "unconsumed-attempt",
        "learner_admissible": False,
        "admitted_jobs": [],
        "valid_records": [],
        "returned_worker_results": [],
    }
    attempts.append(fragment)
    cursor = start_job
    width = runtime.data["worker_count"]
    while len(records) < runtime.ppo.batch_size:
        jobs = runtime.schedule["training"][cursor : cursor + width]
        if len(jobs) != width:
            raise RunnerContractError("training max_episode_jobs exhausted before exact batch")
        fragment["admitted_jobs"].extend(jobs)
        assignment = {job["worker_id"]: job for job in jobs}
        resets = await _retain_call(
            pool.reset(episode_seeds={worker: job["seed"] for worker, job in assignment.items()}),
            attempts, "training-reset-failure",
        )
        current = {
            row.worker_id: _inference(
                runtime,
                row.observation,
                row.info,
                row.worker_id,
                row.episode_id,
                0,
                assignment[row.worker_id]["seed"],
                True,
            )
            for row in resets
        }
        for row in resets:
            attempts.append(
                {
                    "job_index": assignment[row.worker_id]["job_index"],
                    "reset_attempt": row.info.get("reset_attempt"),
                    "diagnostic": row.info.get("_runner_attempt"),
                }
            )
        wave: dict[str, list[TransitionRecord]] = {worker: [] for worker in assignment}
        while True:
            selection = await actor.actions(current)
            returned = await _retain_call(
                pool.step(_transport_actions(selection.actions),
                          policy_version=selection.policy_version),
                attempts, "training-step-failure",
            )
            fragment["returned_worker_results"].extend(row.to_dict() for row in returned)
            peer_done = any(row.terminated or row.truncated for row in returned)
            budget_done = len(records) + len(returned) == runtime.ppo.batch_size
            for raw in returned:
                previous = current[raw.worker_id]
                following = _inference(
                    runtime,
                    raw.next_observation,
                    raw.info,
                    raw.worker_id,
                    raw.episode_id,
                    raw.step_id,
                    previous.sampling_seed,
                    True,
                )
                cut = None
                if not (raw.terminated or raw.truncated):
                    if budget_done:
                        cut = "rollout-budget"
                    elif peer_done:
                        cut = "peer-cut"
                transition = _record(
                    raw, previous, following, selection.artifact_sha256, cut, False
                )
                records.append(transition)
                fragment["valid_records"].append(transition.to_dict())
                wave[raw.worker_id].append(transition)
                current[raw.worker_id] = following
            if peer_done or budget_done:
                break
        episodes.extend(_episode(assignment[worker], wave[worker]) for worker in sorted(assignment))
        cursor += width
    fragment["disposition"] = "complete-batch-candidate"
    fragment["completed_episodes"] = episodes
    return ValidatedTransitionBatch.build(records), episodes, cursor


def _policy_bytes(artifact: PolicyArtifact) -> bytes:
    return wire.canonical(
        {
            "run_id": artifact.run_id,
            "generation": artifact.generation,
            "policy_version": artifact.policy_version,
            "payload_hex": artifact.payload.hex(),
            "media_type": artifact.media_type,
            "compatibility": artifact.compatibility.to_dict(),
            "source_batch_sha256": artifact.source_batch_sha256,
            "provenance": dict(artifact.provenance),
            "sha256": artifact.sha256,
        }
    )


def _parse_policy(body: bytes) -> PolicyArtifact:
    raw = wire.object_fields(wire.decode_json(body), _POLICY_FIELDS, "policy file")
    artifact = PolicyArtifact(
        run_id=raw["run_id"],
        generation=raw["generation"],
        policy_version=raw["policy_version"],
        payload=bytes.fromhex(raw["payload_hex"]),
        media_type=raw["media_type"],
        compatibility=PolicyCompatibility.from_dict(raw["compatibility"]),
        source_batch_sha256=raw["source_batch_sha256"],
        provenance=raw["provenance"],
    )
    if artifact.sha256 != raw["sha256"]:
        raise RunnerContractError("policy file digest differs")
    return artifact


@dataclass
class _State:
    learner: ReferencePPOLearnerAdapter
    session: LearnerSession
    actor: ActorPolicy
    store: InMemoryPolicyArtifactStore
    references: list[PolicyArtifactRef]
    episodes: list[dict[str, Any]]
    next_job: int = 0


async def _fresh(runtime: _Runtime) -> _State:
    data = runtime.data
    learner = ReferencePPOLearnerAdapter(
        runtime.feature,
        run_id=data["run_id"],
        generation=data["generation"],
        compatibility=runtime.compatibility,
        config=runtime.ppo,
    )
    store = InMemoryPolicyArtifactStore()
    session = LearnerSession(learner, store, run_id=data["run_id"], generation=data["generation"])
    reference = await session.publish_initial(learner.initial_policy())
    learner.bind_published(reference, await store.get(reference))
    actor = ActorPolicy(
        ReferencePPOPolicyLoader(runtime.feature),
        store,
        run_id=data["run_id"],
        generation=data["generation"],
        compatibility=runtime.compatibility,
    )
    await actor.activate(reference)
    return _State(learner, session, actor, store, [reference], [])


async def _save(
    runtime: _Runtime,
    state: _State,
    output: Path,
    storage_profile: str = storage.STANDALONE_PROFILE,
    artifact_entries: dict[str, dict[str, Any]] | None = None,
    raw_batches: list[dict[str, Any]] | None = None,
) -> Path:
    runtime.assert_current()
    count = state.learner.update_count
    root = output / "checkpoints" / f"update-{count:06d}"
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for reference in state.references:
        if storage_profile == storage.SHARED_PROFILE:
            if artifact_entries is None:
                raise RunnerContractError("shared storage cache is missing")
            entry = artifact_entries.get(reference.sha256)
            if entry is None:
                body = _policy_bytes(await state.store.get(reference))
                relative = f"blobs/{wire.byte_digest(body)}.json"
                entry = {**storage.file_entry(relative, body), "reference": reference.to_dict()}
                wire.atomic_write(output / relative, body)
                artifact_entries[reference.sha256] = entry
            else:
                storage.checked_bytes(output, entry, artifact=True)
                if entry["reference"] != reference.to_dict():
                    raise RunnerContractError("shared artifact reference changed")
            entries.append(entry)
        else:
            body = _policy_bytes(await state.store.get(reference))
            relative = f"artifacts/policy-{reference.policy_version:06d}.json"
            wire.atomic_write(root / relative, body)
            entries.append(
                {
                    "path": relative,
                    "size_bytes": len(body),
                    "sha256": wire.byte_digest(body),
                    "reference": reference.to_dict(),
                }
            )
    episode_bytes = wire.canonical(state.episodes)
    wire.atomic_write(root / "episodes.json", episode_bytes)
    recovery = await state.session.snapshot_recovery_state()
    manifest = {
        "schema_version": storage.SHARED_SCHEMA
        if storage_profile == storage.SHARED_PROFILE else wire.CHECKPOINT_SCHEMA,
        "config_sha256": runtime.config.sha256,
        "source_identity": runtime.source_identity,
        "runtime_identity": runtime.runtime_identity,
        "objective_id": "decision-index-v1",
        "run_id": runtime.data["run_id"],
        "generation": runtime.data["generation"],
        "completed_updates": count,
        "completed_transitions": count * runtime.ppo.batch_size,
        "next_job_index": state.next_job,
        "seed_schedule_sha256": wire.digest(runtime.schedule),
        "active_policy_reference": state.references[-1].to_dict(),
        "learner_recovery_state": recovery.to_dict(),
        "artifacts": entries,
        "episode_dispositions_sha256": wire.byte_digest(episode_bytes),
    }
    if storage_profile == storage.SHARED_PROFILE:
        if raw_batches is None or len(raw_batches) != count:
            raise RunnerContractError("shared checkpoint raw history is incomplete")
        for entry in raw_batches:
            storage.checked_bytes(output, entry)
        manifest["storage_profile"] = storage.SHARED_PROFILE
        manifest["raw_batches"] = list(raw_batches)
    manifest["checkpoint_sha256"] = wire.digest(manifest)
    path = root / "manifest.json"
    wire.atomic_write(path, wire.canonical(manifest))
    return path


def _check_episodes(
    runtime: _Runtime,
    manifest: Mapping[str, Any],
    episodes: object,
    recovery: LearnerRecoveryState,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(episodes, list):
        raise RunnerContractError("checkpoint episodes must be a list")
    cursor = wire.integer(manifest["next_job_index"], "next_job_index")
    if len(episodes) != cursor or cursor > len(runtime.schedule["training"]):
        raise RunnerContractError("next job cursor differs from actual dispositions")
    positions = {}
    steps = 0
    validated = []
    for index, item in enumerate(episodes):
        row = wire.object_fields(item, _EPISODE_FIELDS, "episode disposition")
        job = runtime.schedule["training"][index]
        for key in ("job_index", "worker_id", "episode_id", "seed", "case_index", "config_sha256"):
            if type(row[key]) is not type(job[key]) or row[key] != job[key]:
                raise RunnerContractError("episode disposition differs from scheduled job")
        final = wire.integer(row["final_step"], "final_step", 1)
        if final > runtime.data["max_steps"]:
            raise RunnerContractError("episode step count exceeds cutoff")
        for key in ("logical_time", "return", "objective_cost"):
            value = row[key]
            if key == "objective_cost" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise RunnerContractError(f"episode {key} must be a finite number")
        if (
            type(row["terminated"]) is not bool
            or type(row["truncated"]) is not bool
            or not (row["terminated"] or row["truncated"])
        ):
            raise RunnerContractError("checkpoint contains an open episode")
        if row["cut_reason"] not in (
            "domain-termination",
            "max-steps",
            "peer-cut",
            "rollout-budget",
        ):
            raise RunnerContractError("unknown episode cut reason")
        if row["cut_reason"] == "domain-termination" and not row["terminated"]:
            raise RunnerContractError("domain cut requires true termination")
        if row["cut_reason"] != "domain-termination" and (
            row["terminated"] or not row["truncated"]
        ):
            raise RunnerContractError("external cut requires truncation without termination")
        if row["cut_reason"] == "max-steps" and final != runtime.data["max_steps"]:
            raise RunnerContractError("max-steps cut differs from configured cutoff")
        version = wire.integer(row["policy_version"], "episode policy version")
        if version >= manifest["completed_updates"]:
            raise RunnerContractError("episode policy version is beyond completed updates")
        positions[(row["worker_id"], row["episode_id"])] = (final, row["logical_time"])
        steps += final
        validated.append(row)
    closed = {(item["worker_id"], item["episode_id"]) for item in payload["closed_streams"]}
    if positions != dict(recovery.stream_positions) or set(positions) != closed:
        raise RunnerContractError("episode dispositions differ from closed-stream cursors")
    if steps != manifest["completed_transitions"] or steps != payload["environment_steps"]:
        raise RunnerContractError("episode steps differ from committed learner count")
    return validated


async def _restore(
    runtime: _Runtime, checkpoint: str | Path
) -> tuple[_State, dict[str, Any], Path]:
    path = Path(checkpoint).resolve()  # noqa: ASYNC240 - synchronous checkpoint admission.
    if path.is_dir():
        path = path / "manifest.json"
    decoded = wire.decode_json(path.read_bytes())
    shared = isinstance(decoded, dict) and decoded.get("schema_version") == storage.SHARED_SCHEMA
    if shared:
        raw = storage.read_checkpoint(path)
    else:
        raw = wire.object_fields(decoded, wire.CHECKPOINT_FIELDS, "checkpoint")
    expected_digest = raw["checkpoint_sha256"]
    unsigned = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    if expected_digest != wire.digest(unsigned):
        raise RunnerContractError("checkpoint digest differs")
    for key, expected in {
        "schema_version": storage.SHARED_SCHEMA if shared else wire.CHECKPOINT_SCHEMA,
        "config_sha256": runtime.config.sha256,
        "source_identity": runtime.source_identity,
        "runtime_identity": runtime.runtime_identity,
        "objective_id": "decision-index-v1",
        "run_id": runtime.data["run_id"],
        "generation": runtime.data["generation"],
        "seed_schedule_sha256": wire.digest(runtime.schedule),
    }.items():
        if wire.canonical(raw[key]) != wire.canonical(expected):
            raise RunnerContractError(f"checkpoint {key} differs")
    updates = wire.integer(raw["completed_updates"], "completed_updates")
    transitions = wire.integer(raw["completed_transitions"], "completed_transitions")
    if (
        updates > runtime.data["training"]["total_updates"]
        or transitions != updates * runtime.ppo.batch_size
    ):
        raise RunnerContractError("checkpoint update/transition budget differs")
    entries = raw["artifacts"]
    if not isinstance(entries, list) or len(entries) != updates + 1:
        raise RunnerContractError("checkpoint omits historical policy artifacts")
    store = InMemoryPolicyArtifactStore()
    references = []
    for version, item in enumerate(entries):
        entry = wire.object_fields(
            item, {"path", "size_bytes", "sha256", "reference"}, "artifact entry"
        )
        body = (
            storage.checked_bytes(storage.archive_root(path), entry, artifact=True)
            if shared else wire.contained_file(path.parent, entry["path"]).read_bytes()
        )
        if (
            len(body) != wire.integer(entry["size_bytes"], "artifact length", 1)
            or wire.byte_digest(body) != entry["sha256"]
        ):
            raise RunnerContractError("checkpoint artifact bytes differ")
        artifact = _parse_policy(body)
        if artifact.policy_version != version or artifact.compatibility != runtime.compatibility:
            raise RunnerContractError("checkpoint artifact version/compatibility differs")
        reference = await store.put(artifact)
        if reference.to_dict() != entry["reference"]:
            raise RunnerContractError("checkpoint artifact reference differs")
        references.append(reference)
    active = PolicyArtifactRef.from_dict(raw["active_policy_reference"])
    if active != references[-1]:
        raise RunnerContractError("active reference is not the final checkpoint policy")
    artifact = await store.get(active)
    learner = ReferencePPOLearnerAdapter.from_artifact(artifact, runtime.feature)
    learner.bind_published(active, artifact)
    recovery = LearnerRecoveryState.from_dict(raw["learner_recovery_state"])
    session = await LearnerSession.restore(
        learner,
        store,
        state=recovery,
        run_id=runtime.data["run_id"],
        generation=runtime.data["generation"],
    )
    if learner.update_count != updates or len(recovery.consumed_batches) != updates:
        raise RunnerContractError("checkpoint session/update counts disagree")
    episode_bytes = (path.parent / "episodes.json").read_bytes()
    if wire.byte_digest(episode_bytes) != raw["episode_dispositions_sha256"]:
        raise RunnerContractError("checkpoint episode bytes differ")
    episodes = _check_episodes(
        runtime, raw, wire.decode_json(episode_bytes), recovery, wire.decode_json(artifact.payload)
    )
    actor = ActorPolicy(
        ReferencePPOPolicyLoader(runtime.feature),
        store,
        run_id=runtime.data["run_id"],
        generation=runtime.data["generation"],
        compatibility=runtime.compatibility,
    )
    await actor.activate(active)
    return (
        _State(learner, session, actor, store, references, episodes, raw["next_job_index"]),
        raw,
        path,
    )


def _failure(error: BaseException) -> dict[str, object]:
    snapshot = getattr(error, "_runner_failure_snapshot", None)
    if isinstance(snapshot, dict):
        return snapshot
    result: dict[str, object] = {"type": type(error).__name__, "message": str(error)}
    for name in ("phase", "completion_boundary"):
        value = getattr(error, name, None)
        if value is not None:
            result[name] = str(value)
    errors = getattr(error, "errors", None)
    if isinstance(errors, Mapping):
        result["causes"] = {str(key): _failure(value) for key, value in errors.items()}
    else:
        causes = getattr(error, "causes", ())
        result["causes"] = [_failure(value) for value in causes]
        if not causes and error.__cause__ is not None:
            result["causes"] = [_failure(error.__cause__)]
    returned = getattr(error, "returned_results", None)
    if isinstance(returned, Mapping):
        result["returned_results"] = {
            str(worker): _diagnostic_value(value) for worker, value in returned.items()
        }
        result["returned_results_learner_admissible"] = False
    return result


def _diagnostic_value(value: object) -> object:
    try:
        return wire.decode_json(wire.canonical(value))
    except (TypeError, ValueError) as error:
        return {
            "unserializable_type": type(value).__qualname__,
            "serialization_error": type(error).__name__,
        }


async def _close(pool: LocalRolloutPool, cleanup: list[object]) -> None:
    try:
        await pool.close()
    finally:
        cleanup.append(
            {
                "backend": pool.backend,
                "state": pool.state.value,
                "workers": [
                    {
                        "worker_id": worker,
                        "attempt": row.attempt,
                        "succeeded": row.succeeded,
                        "error": None if row.error is None else _failure(row.error),
                    }
                    for worker, row in pool.close_receipts.items()
                ],
                "pids": dict(pool.process_pids),
            }
        )


async def _evaluate_job(
    runtime: _Runtime,
    state: _State,
    job: dict[str, Any],
    cleanup: list[object],
    attempts: list[object],
) -> tuple[dict[str, Any], list[TransitionRecord]]:
    worker = job["worker_id"]
    pool = _pool(runtime, [worker], "evaluation")
    records = []
    fragment: dict[str, Any] = {
        "phase": "evaluation",
        "disposition": "attempted",
        "learner_admissible": False,
        "admitted_jobs": [job],
        "valid_records": [],
        "returned_worker_results": [],
    }
    attempts.append(fragment)
    try:
        reset = (await _retain_call(
            pool.reset(episode_seeds={worker: job["seed"]}), attempts,
            "evaluation-reset-failure",
        ))[0]
        current = _inference(
            runtime, reset.observation, reset.info, worker, reset.episode_id, 0, job["seed"], False
        )
        while True:
            selection = await state.actor.actions({worker: current})
            raw = (await _retain_call(
                pool.step(_transport_actions(selection.actions),
                          policy_version=selection.policy_version),
                attempts, "evaluation-step-failure",
            ))[0]
            fragment["returned_worker_results"].append(raw.to_dict())
            following = _inference(
                runtime,
                raw.next_observation,
                raw.info,
                worker,
                raw.episode_id,
                raw.step_id,
                job["seed"],
                False,
            )
            records.append(_record(raw, current, following, selection.artifact_sha256, None, True))
            fragment["valid_records"].append(records[-1].to_dict())
            current = following
            if raw.terminated or raw.truncated:
                break
        fragment["disposition"] = "completed-evaluation"
        return {
            **_episode(job, records),
            "backend": pool.backend,
            "effective_pool_width": 1,
        }, records
    finally:
        await _close(pool, cleanup)


async def _invoke(
    operation: str,
    config: RunnerConfig | Mapping[str, object],
    output: str | Path,
    checkpoint: str | Path | None = None,
    stop_after_updates: int | None = None,
    storage_profile: str | None = None,
) -> RunnerResult:
    if storage_profile is not None:
        storage.profile(storage_profile)
    selected_profile = storage_profile or storage.STANDALONE_PROFILE
    runtime = _Runtime(wire.ensure_config(config))
    data = runtime.data
    target = data["training"]["total_updates"]
    if stop_after_updates is not None:
        if (
            operation != "train"
            or wire.integer(stop_after_updates, "stop-after-updates", 1) > target
        ):
            raise RunnerContractError("stop-after-updates outside the frozen training budget")
        target = stop_after_updates
    root = Path(output).resolve()  # noqa: ASYNC240 - synchronous immutable output admission.
    root.mkdir(parents=True, exist_ok=False)
    cleanup: list[object] = []
    attempts: list[object] = []
    evaluation_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, object]] = []
    raw_batches: list[dict[str, Any]] = []
    artifact_entries: dict[str, dict[str, Any]] = {}
    state: _State | None = None
    result_checkpoint: Path | None = None
    input_digest: str | None = None
    output_digest: str | None = None
    accepted: dict[str, Any] | None = None
    error: BaseException | None = None
    try:
        wire.atomic_write(root / "config.json", runtime.config.payload)
        wire.atomic_write(root / "schedule.json", wire.canonical(runtime.schedule))
        if checkpoint is None:
            state = await _fresh(runtime)
            result_checkpoint = await _save(
                runtime, state, root, selected_profile, artifact_entries, raw_batches
            )
            accepted = wire.decode_json(result_checkpoint.read_bytes())
            output_digest = accepted["checkpoint_sha256"]
        else:
            state, manifest, _source_path = await _restore(runtime, checkpoint)
            source_profile = (
                storage.SHARED_PROFILE if manifest["schema_version"] == storage.SHARED_SCHEMA
                else storage.STANDALONE_PROFILE
            )
            if storage_profile is not None and storage_profile != source_profile:
                raise RunnerContractError("explicit storage profile cannot migrate a checkpoint")
            selected_profile = source_profile
            input_digest = manifest["checkpoint_sha256"]
            accepted = manifest
            if operation == "resume" and selected_profile == storage.SHARED_PROFILE:
                raw_batches.extend(storage.seed_shared_resume(_source_path, root))
                inventory.extend(raw_batches)
                artifact_entries.update(
                    (entry["reference"]["sha256"], entry) for entry in manifest["artifacts"]
                )
                result_checkpoint = root / _source_path.relative_to(
                    storage.archive_root(_source_path)
                )
            source_run = _source_path.parent.parent.parent
            parent_inventory = source_run / "inventory.json"
            lineage = {
                "input_checkpoint_path": str(_source_path),
                "input_checkpoint_sha256": input_digest,
                "parent_inventory": None
                if not parent_inventory.is_file()
                else {
                    "path": str(parent_inventory),
                    "sha256": wire.byte_digest(parent_inventory.read_bytes()),
                },
                "raw_history_complete_in_this_output": operation == "resume"
                and selected_profile == storage.SHARED_PROFILE,
            }
            body = wire.canonical(lineage)
            wire.atomic_write(root / "lineage.json", body)
            inventory.append(
                {"path": "lineage.json", "size_bytes": len(body), "sha256": wire.byte_digest(body)}
            )
        if operation == "evaluate":
            before = (await state.session.snapshot_recovery_state()).to_dict()
            before_parameters = state.learner.parameter_digests
            concurrency = 1 if data["backend"] == "serial" else data["worker_count"]
            semaphore = asyncio.Semaphore(concurrency)
            active_jobs = 0
            peak_jobs = 0

            async def one(job: dict[str, Any]) -> tuple[dict[str, Any], list[TransitionRecord]]:
                nonlocal active_jobs, peak_jobs
                async with semaphore:
                    active_jobs += 1
                    peak_jobs = max(active_jobs, peak_jobs)
                    try:
                        return await _evaluate_job(runtime, state, job, cleanup, attempts)
                    finally:
                        active_jobs -= 1

            rows = await asyncio.gather(
                *(one(job) for job in runtime.schedule["evaluation"]), return_exceptions=True
            )
            failures = []
            for index, value in enumerate(rows):
                if isinstance(value, BaseException):
                    attempts.append({"job_index": index, "failure": _failure(value)})
                    failures.append(value)
                    continue
                row, records = value
                evaluation_rows.append(row)
                relative = f"evaluation/job-{index:08d}.json"
                body = wire.canonical([record.to_dict() for record in records])
                wire.atomic_write(root / relative, body)
                inventory.append(
                    {"path": relative, "size_bytes": len(body), "sha256": wire.byte_digest(body)}
                )
            cleanup.append(
                {"configured_job_concurrency": concurrency, "observed_peak_jobs": peak_jobs}
            )
            if failures:
                raise RunnerContractError(f"{len(failures)} evaluation jobs failed") from failures[
                    0
                ]
            if (
                before != (await state.session.snapshot_recovery_state()).to_dict()
                or before_parameters != state.learner.parameter_digests
            ):
                raise RunnerContractError("evaluation modified training state")
        else:
            if operation == "resume" and state.learner.update_count >= target:
                raise RunnerContractError("checkpoint already completed the frozen training budget")
            workers = [f"worker-{index:04d}" for index in range(data["worker_count"])]
            pool = _pool(runtime, workers, "training")
            try:
                while state.learner.update_count < target:
                    runtime.assert_current()
                    batch, episodes, cursor = await _collect(
                        runtime, pool, state.actor, state.next_job, attempts
                    )
                    number = state.learner.update_count + 1
                    relative = f"batches/batch-{number:06d}.json"
                    body = wire.canonical(
                        {
                            "schema_version": "local-runner-batch-v1",
                            "config_sha256": runtime.config.sha256,
                            "batch_index": number,
                            "batch_sha256": batch.sha256,
                            "behavior_reference": state.references[-1].to_dict(),
                            "records": [record.to_dict() for record in batch.records],
                        }
                    )
                    wire.atomic_write(root / relative, body)
                    inventory.append(
                        {
                            "path": relative,
                            "size_bytes": len(body),
                            "sha256": wire.byte_digest(body),
                        }
                    )
                    raw_batches.append(storage.file_entry(relative, body))
                    reference = await state.session.consume(batch)
                    if reference is None:
                        raise RunnerContractError("PPO did not publish for a complete batch")
                    state.learner.bind_published(reference, await state.store.get(reference))
                    await state.actor.activate(reference)
                    state.references.append(reference)
                    state.episodes.extend(episodes)
                    state.next_job = cursor
                    result_checkpoint = await _save(
                        runtime, state, root, selected_profile, artifact_entries, raw_batches
                    )
                    accepted = wire.decode_json(result_checkpoint.read_bytes())
                    output_digest = accepted["checkpoint_sha256"]
            finally:
                await _close(pool, cleanup)
        runtime.assert_current()
        if result_checkpoint is not None:
            output_digest = wire.decode_json(result_checkpoint.read_bytes())["checkpoint_sha256"]
    except BaseException as caught:
        error = caught
        attempts.append(
            {
                "phase": "invocation-failure",
                "failure": _failure(caught),
                "attempted_updates": None if state is None else state.learner.update_count,
                "accepted_checkpoint_sha256": None
                if accepted is None
                else accepted["checkpoint_sha256"],
            }
        )
    receipt = {
        "schema_version": wire.RECEIPT_SCHEMA,
        "operation": operation,
        "config_sha256": runtime.config.sha256,
        "input_checkpoint_sha256": input_digest,
        "output_checkpoint_sha256": output_digest,
        "execution_complete": error is None,
        "contract_passed": error is None,
        "qualification_scope": "synthetic-unqualified"
        if runtime.policy is None
        else "pinned-local-runner",
        "completed_updates": 0 if accepted is None else accepted["completed_updates"],
        "completed_transitions": 0 if accepted is None else accepted["completed_transitions"],
        "episode_dispositions": []
        if state is None or accepted is None
        else state.episodes[: accepted["next_job_index"]],
        "evaluation_rows": evaluation_rows,
        "cleanup": cleanup,
        "failure": None if error is None else _failure(error),
    }
    body = wire.canonical(attempts)
    wire.atomic_write(root / "attempts.json", body)
    inventory.append(
        {"path": "attempts.json", "size_bytes": len(body), "sha256": wire.byte_digest(body)}
    )
    wire.atomic_write(root / "inventory.json", wire.canonical(inventory))
    wire.atomic_write(root / "receipt.json", wire.canonical(receipt))
    if error is not None:
        raise RunnerExecutionError(f"runner {operation} failed: {error}", receipt) from error
    return RunnerResult(root, result_checkpoint, receipt)


async def train(
    config: RunnerConfig | Mapping[str, object],
    output: str | Path,
    *,
    stop_after_updates: int | None = None,
    storage_profile: str = storage.STANDALONE_PROFILE,
) -> RunnerResult:
    """Train using an immutable config, optionally stopping at an update cut."""
    return await _invoke("train", config, output, stop_after_updates=stop_after_updates,
                         storage_profile=storage_profile)


async def resume(
    config: RunnerConfig | Mapping[str, object], checkpoint: str | Path, output: str | Path,
    *, storage_profile: str | None = None,
) -> RunnerResult:
    """Restore the full learner/coordinator cut and consume its frozen suffix."""
    return await _invoke("resume", config, output, checkpoint=checkpoint,
                         storage_profile=storage_profile)


async def evaluate(
    config: RunnerConfig | Mapping[str, object], checkpoint: str | Path, output: str | Path
) -> RunnerResult:
    """Evaluate full independent episodes without altering training state."""
    return await _invoke("evaluate", config, output, checkpoint=checkpoint)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("train", "evaluate", "resume"):
        child = subparsers.add_parser(operation)
        child.add_argument("--config", required=True)
        child.add_argument("--output", required=True)
        if operation == "train":
            child.add_argument("--stop-after-updates", type=int)
            child.add_argument("--storage-profile", choices=(storage.STANDALONE_PROFILE,
                                storage.SHARED_PROFILE), default=storage.STANDALONE_PROFILE)
        else:
            child.add_argument("--checkpoint", required=True)
            if operation == "resume":
                child.add_argument("--storage-profile", choices=(storage.STANDALONE_PROFILE,
                                    storage.SHARED_PROFILE))
    child = subparsers.add_parser("export")
    child.add_argument("--checkpoint", required=True)
    child.add_argument("--output", required=True)
    child = subparsers.add_parser("validate-archive")
    child.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "export":
            checkpoint = export_checkpoint(args.checkpoint, args.output)
            print(wire.canonical({"checkpoint": str(checkpoint)}).decode())
            return 0
        if args.operation == "validate-archive":
            print(wire.canonical(validate_archive(args.output)).decode())
            return 0
        config = load_config(args.config)
        if args.operation == "train":
            result = asyncio.run(
                train(config, args.output, stop_after_updates=args.stop_after_updates,
                      storage_profile=args.storage_profile)
            )
        elif args.operation == "resume":
            result = asyncio.run(resume(config, args.checkpoint, args.output,
                                        storage_profile=args.storage_profile))
        else:
            result = asyncio.run(evaluate(config, args.checkpoint, args.output))
        print(wire.canonical(dict(result.receipt)).decode())
    except (RunnerContractError, RunnerExecutionError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
