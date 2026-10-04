"""Bounded workload-profile qualification for the anti-torpedo factorial.

This module qualifies only a source-bound workload profile.  It deliberately
does not train a learner and cannot admit PPO, gorti, RLlib, DEXSim, performance,
generalization, or SCIE claims.  Its actual-v2 runner executes the source-locked
AT/SIM adapter, while an injected runner Protocol keeps the evidence schema
testable without allowing a deterministic fake to impersonate AT/SIM evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, Protocol, Self, cast

from pyjevsim_bridge.rl.qualification_models.anti_torpedo_profile import (
    CONFIG_COUNT,
    EVALUATION_CONFIG_COUNT,
    FACTOR_SPECS,
    AntiTorpedoScenarioProfileV1,
    ScenarioConfigV1,
    SensitivityStage,
    VariationClass,
    build_anti_torpedo_scenario_profile,
    gf2_fold,
)

SCHEMA_VERSION: Final = "anti-torpedo-workload-qualification-v1"
PROFILE_ID: Final = "atsim-effective-replicate-factorial-v2"
CANONICAL_JSON_ID: Final = "json-sort-keys-compact-utf8-no-nan-v1"
CASE_COUNT: Final = CONFIG_COUNT
REPLAY_COUNT: Final = EVALUATION_CONFIG_COUNT
MAX_MODEL_CONSTRUCTIONS: Final = 320
MAX_ENVIRONMENT_STEPS: Final = 2_432
ACTION_TRACE: Final = (0, 3, 0)
PLANNED_MODEL_CONSTRUCTIONS: Final = CASE_COUNT + REPLAY_COUNT
PLANNED_ENVIRONMENT_STEPS: Final = PLANNED_MODEL_CONSTRUCTIONS * len(ACTION_TRACE)
FACTOR_NAMES: Final = tuple(factor.factor_id for factor in FACTOR_SPECS)
EXPECTED_MASKS: Final = (
    (True, True, True, True, True, True),
    (True, True, True, True, True, True),
    (True, True, True, False, False, False),
    (True, True, True, False, False, False),
)
REQUIRED_ARTIFACT_ROLES: Final = (
    "capability",
    "configs",
    "mask",
    "partition",
    "plan",
    "probes",
    "replays",
    "sensitivity",
    "source-profile",
)
ACTUAL_V2_RUNNER_ID: Final = "anti-torpedo-v2-qualification-runner-v1"
DISALLOWED_CLAIMS: Final = (
    "ppo-quality",
    "gorti-gain",
    "rllib-comparison",
    "dexsim-comparison",
    "continuous-domain-generalization",
    "cross-model-generalization",
    "scie-conclusion",
)


class WorkloadQualificationError(ValueError):
    """Raised when bounded qualification evidence is invalid or incomplete."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkloadQualificationError(f"value is not canonical JSON: {exc}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json(value))


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise WorkloadQualificationError(f"{name} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise WorkloadQualificationError(f"{name} must be a SHA-256 hex digest") from exc
    return value.lower()


def _non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkloadQualificationError(f"{name} must be a non-empty string")
    return value


def _exact_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise WorkloadQualificationError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadQualificationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WorkloadQualificationError(f"{name} must be a finite number")
    return result


def _reject_constant(value: str) -> object:
    raise WorkloadQualificationError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WorkloadQualificationError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _canonical_object(name: str, value: str) -> Mapping[str, object]:
    if not isinstance(value, str):
        raise WorkloadQualificationError(f"{name} must be canonical JSON text")
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise WorkloadQualificationError(f"{name} is invalid JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise WorkloadQualificationError(f"{name} must contain one JSON object")
    if value.encode("utf-8") != _canonical_json(decoded):
        raise WorkloadQualificationError(f"{name} is not canonical JSON")
    return cast(Mapping[str, object], decoded)


def _validate_projection(name: str, value: str) -> None:
    root = _canonical_object(name, value)
    forbidden = {
        "case_id",
        "config_sha256",
        "episode_seed",
        "scenario_id",
        "source_profile_sha256",
        "timestamp",
        "worker_id",
        "worker_slot",
    }

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise WorkloadQualificationError(f"{name} contains a non-string key")
                if key in forbidden or key.endswith("_sha256"):
                    raise WorkloadQualificationError(
                        f"{name} contains forbidden provenance key: {key}"
                    )
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, float) and not math.isfinite(item):
            raise WorkloadQualificationError(f"{name} contains a non-finite number")

    visit(root)


def _json_text(value: object) -> str:
    return _canonical_json(value).decode("utf-8")


def qualification_implementation_sha256() -> str:
    """Return the loaded qualification runner's source identity."""

    return _sha256_bytes(Path(__file__).resolve().read_bytes())


@dataclass(frozen=True, slots=True)
class SourceProfileLock:
    """Exact source and semantic identities expected from an injected runner."""

    pyjevsim_executor_source_sha256: str
    atsim_source_sha256: str
    adapter_source_sha256: str
    scenario_generator_source_sha256: str
    scenario_bank_sha256: str
    scenario_family_sha256: str
    scenario_source_sha256: str
    factor_schema_sha256: str
    environment_contract_sha256: str
    projection_contract_sha256: str
    qualification_runner_source_sha256: str
    runner_id: str
    schema_version: str = SCHEMA_VERSION
    source_profile_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise WorkloadQualificationError("unsupported source-profile schema")
        for name in (
            "pyjevsim_executor_source_sha256",
            "atsim_source_sha256",
            "adapter_source_sha256",
            "scenario_generator_source_sha256",
            "scenario_bank_sha256",
            "scenario_family_sha256",
            "scenario_source_sha256",
            "factor_schema_sha256",
            "environment_contract_sha256",
            "projection_contract_sha256",
            "qualification_runner_source_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        _non_empty("runner_id", self.runner_id)
        object.__setattr__(self, "source_profile_sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "adapter_source_sha256": self.adapter_source_sha256,
            "atsim_source_sha256": self.atsim_source_sha256,
            "environment_contract_sha256": self.environment_contract_sha256,
            "factor_schema_sha256": self.factor_schema_sha256,
            "projection_contract_sha256": self.projection_contract_sha256,
            "pyjevsim_executor_source_sha256": self.pyjevsim_executor_source_sha256,
            "qualification_runner_source_sha256": self.qualification_runner_source_sha256,
            "runner_id": self.runner_id,
            "scenario_bank_sha256": self.scenario_bank_sha256,
            "scenario_family_sha256": self.scenario_family_sha256,
            "scenario_generator_source_sha256": self.scenario_generator_source_sha256,
            "scenario_source_sha256": self.scenario_source_sha256,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        value = self.content()
        value["source_profile_sha256"] = self.source_profile_sha256
        return value


def build_actual_v2_source_profile() -> SourceProfileLock:
    """Bind the actual v2 runner to loaded adapter, model, and profile sources."""

    from pyjevsim_bridge.rl.qualification_models.anti_torpedo import (
        LOADED_ADAPTER_SOURCE_SHA256,
        PYJEVSIM_EXECUTOR_SOURCE_SHA256,
        V2_ENVIRONMENT_CONTRACT_SHA256,
        atsim_source_sha256,
    )
    from pyjevsim_bridge.rl.scientific import SEMANTIC_PROJECTION_CONTRACT_SHA256

    profile = build_anti_torpedo_scenario_profile()
    return SourceProfileLock(
        pyjevsim_executor_source_sha256=PYJEVSIM_EXECUTOR_SOURCE_SHA256,
        atsim_source_sha256=atsim_source_sha256(),
        adapter_source_sha256=LOADED_ADAPTER_SOURCE_SHA256,
        scenario_generator_source_sha256=profile.bank.generator_source_sha256,
        scenario_bank_sha256=profile.bank.sha256,
        scenario_family_sha256=profile.bank.family_sha256,
        scenario_source_sha256=profile.bank.scenario_source_sha256,
        factor_schema_sha256=profile.bank.factor_schema_sha256,
        environment_contract_sha256=V2_ENVIRONMENT_CONTRACT_SHA256,
        projection_contract_sha256=SEMANTIC_PROJECTION_CONTRACT_SHA256,
        qualification_runner_source_sha256=qualification_implementation_sha256(),
        runner_id=ACTUAL_V2_RUNNER_ID,
    )


@dataclass(frozen=True, slots=True)
class ActualV2RuntimeReceipt:
    """Source-bound reset/step evidence emitted only by the actual v2 runner."""

    scenario_ordinal: int
    config_sha256: str
    factor_vector: tuple[int, ...]
    scenario_bank_sha256: str
    scenario_family_sha256: str
    scenario_source_sha256: str
    factor_schema_sha256: str
    profile_generator_source_sha256: str
    atsim_source_sha256: str
    adapter_source_sha256: str
    environment_contract_sha256: str
    projection_contract_sha256: str
    pyjevsim_executor_source_sha256: str
    qualification_runner_source_sha256: str
    workload_version: str
    executor_qualified: bool
    reset_observation_identity_sha256: str
    reset_info_identity_sha256: str

    def __post_init__(self) -> None:
        _exact_int("actual v2 scenario_ordinal", self.scenario_ordinal)
        if self.scenario_ordinal >= CASE_COUNT:
            raise WorkloadQualificationError("actual v2 ordinal is outside the profile")
        if len(self.factor_vector) != len(FACTOR_SPECS) or any(
            type(value) is not int or value not in (0, 1)
            for value in self.factor_vector
        ):
            raise WorkloadQualificationError("actual v2 factor vector is not binary")
        for name in (
            "config_sha256",
            "scenario_bank_sha256",
            "scenario_family_sha256",
            "scenario_source_sha256",
            "factor_schema_sha256",
            "profile_generator_source_sha256",
            "atsim_source_sha256",
            "adapter_source_sha256",
            "environment_contract_sha256",
            "projection_contract_sha256",
            "pyjevsim_executor_source_sha256",
            "qualification_runner_source_sha256",
            "reset_observation_identity_sha256",
            "reset_info_identity_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        if self.workload_version != "AntiTorpedoCountermeasure-v2":
            raise WorkloadQualificationError("actual runner workload version is not v2")
        if type(self.executor_qualified) is not bool:
            raise WorkloadQualificationError("executor_qualified must be bool")

    def content(self) -> dict[str, object]:
        return {
            "adapter_source_sha256": self.adapter_source_sha256,
            "atsim_source_sha256": self.atsim_source_sha256,
            "config_sha256": self.config_sha256,
            "environment_contract_sha256": self.environment_contract_sha256,
            "executor_qualified": self.executor_qualified,
            "factor_schema_sha256": self.factor_schema_sha256,
            "factor_vector": list(self.factor_vector),
            "profile_generator_source_sha256": (
                self.profile_generator_source_sha256
            ),
            "projection_contract_sha256": self.projection_contract_sha256,
            "pyjevsim_executor_source_sha256": (
                self.pyjevsim_executor_source_sha256
            ),
            "qualification_runner_source_sha256": (
                self.qualification_runner_source_sha256
            ),
            "reset_info_identity_sha256": self.reset_info_identity_sha256,
            "reset_observation_identity_sha256": (
                self.reset_observation_identity_sha256
            ),
            "scenario_bank_sha256": self.scenario_bank_sha256,
            "scenario_family_sha256": self.scenario_family_sha256,
            "scenario_ordinal": self.scenario_ordinal,
            "scenario_source_sha256": self.scenario_source_sha256,
            "workload_version": self.workload_version,
        }


@dataclass(frozen=True, slots=True)
class ConfigCase:
    ordinal: int
    case_id: str
    factors: tuple[int, ...]
    config: ScenarioConfigV1
    config_sha256: str
    fold: int
    partition: str
    episode_seed: int

    def __post_init__(self) -> None:
        _exact_int("case ordinal", self.ordinal)
        if self.ordinal >= CASE_COUNT:
            raise WorkloadQualificationError("case ordinal is outside the factorial")
        if self.config.ordinal != self.ordinal or self.case_id != self.config.config_id:
            raise WorkloadQualificationError("case identity differs from source-locked config")
        if len(self.factors) != len(FACTOR_NAMES) or any(
            type(value) is not int or value not in (0, 1) for value in self.factors
        ):
            raise WorkloadQualificationError("factors must contain eight binary integers")
        if self.factors != self.config.level_bits:
            raise WorkloadQualificationError("factor vector differs from source-locked config")
        if self.config_sha256 != self.config.config_sha256:
            raise WorkloadQualificationError("config digest differs from source-locked config")
        if self.fold != gf2_fold(self.ordinal):
            raise WorkloadQualificationError("case fold differs from source-locked profile")
        if self.partition not in {"tuning", "measured-training", "evaluation"}:
            raise WorkloadQualificationError("case partition is unknown")
        _exact_int("case episode_seed", self.episode_seed)

    def content(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "config": self.config.materialize(),
            "config_identity": self.config.content(),
            "config_sha256": self.config_sha256,
            "episode_seed": self.episode_seed,
            "factors": dict(zip(FACTOR_NAMES, self.factors, strict=True)),
            "fold": self.fold,
            "ordinal": self.ordinal,
            "partition": self.partition,
        }


def _cases_from_profile(
    profile: AntiTorpedoScenarioProfileV1,
) -> tuple[ConfigCase, ...]:
    assignment: dict[int, str] = {}
    for config in profile.partition.tuning:
        assignment[config.ordinal] = "tuning"
    for config in profile.partition.measured:
        assignment[config.ordinal] = "measured-training"
    for config in profile.partition.evaluation:
        assignment[config.ordinal] = "evaluation"
    seeds = profile.seeds.qualification_episodes
    return tuple(
        ConfigCase(
            ordinal=config.ordinal,
            case_id=config.config_id,
            factors=config.level_bits,
            config=config,
            config_sha256=config.config_sha256,
            fold=gf2_fold(config.ordinal),
            partition=assignment[config.ordinal],
            episode_seed=seeds[config.ordinal].seed,
        )
        for config in profile.bank.configs
    )


def _episode_seed(case: ConfigCase) -> int:
    return case.episode_seed


@dataclass(frozen=True, slots=True)
class QualificationPlan:
    source_profile: SourceProfileLock
    scenario_profile: AntiTorpedoScenarioProfileV1
    cases: tuple[ConfigCase, ...]
    replay_case_ids: tuple[str, ...]
    action_trace: tuple[int, ...] = ACTION_TRACE
    planned_model_constructions: int = PLANNED_MODEL_CONSTRUCTIONS
    planned_environment_steps: int = PLANNED_ENVIRONMENT_STEPS
    schema_version: str = SCHEMA_VERSION
    plan_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise WorkloadQualificationError("unsupported qualification plan schema")
        if not isinstance(self.scenario_profile, AntiTorpedoScenarioProfileV1):
            raise TypeError("scenario_profile must be AntiTorpedoScenarioProfileV1")
        expected_profile_locks = {
            "scenario_generator_source_sha256": (
                self.scenario_profile.bank.generator_source_sha256
            ),
            "scenario_bank_sha256": self.scenario_profile.bank.sha256,
            "scenario_family_sha256": self.scenario_profile.bank.family_sha256,
            "scenario_source_sha256": self.scenario_profile.bank.scenario_source_sha256,
            "factor_schema_sha256": self.scenario_profile.bank.factor_schema_sha256,
        }
        for name, expected in expected_profile_locks.items():
            if getattr(self.source_profile, name) != expected:
                raise WorkloadQualificationError(
                    f"source profile does not bind {name}"
                )
        cases = tuple(self.cases)
        if len(cases) != CASE_COUNT or cases != _cases_from_profile(self.scenario_profile):
            raise WorkloadQualificationError("plan must contain the exact 256-case factorial")
        if len({case.case_id for case in cases}) != CASE_COUNT:
            raise WorkloadQualificationError("case IDs must be unique")
        if len({case.config_sha256 for case in cases}) != CASE_COUNT:
            raise WorkloadQualificationError("config digests must be unique")
        expected_replays = tuple(
            case.case_id for case in cases if case.partition == "evaluation"
        )
        if tuple(self.replay_case_ids) != expected_replays or len(expected_replays) != REPLAY_COUNT:
            raise WorkloadQualificationError("replays must cover the exact 64 evaluation cases")
        if tuple(self.action_trace) != ACTION_TRACE:
            raise WorkloadQualificationError("qualification action trace differs")
        if self.planned_model_constructions != PLANNED_MODEL_CONSTRUCTIONS:
            raise WorkloadQualificationError("qualification construction budget differs")
        if self.planned_environment_steps != PLANNED_ENVIRONMENT_STEPS:
            raise WorkloadQualificationError("qualification step budget differs")
        if self.planned_model_constructions > MAX_MODEL_CONSTRUCTIONS:
            raise WorkloadQualificationError("qualification construction hard bound exceeded")
        if self.planned_environment_steps > MAX_ENVIRONMENT_STEPS:
            raise WorkloadQualificationError("qualification step hard bound exceeded")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "replay_case_ids", tuple(self.replay_case_ids))
        object.__setattr__(self, "plan_sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "action_trace": list(self.action_trace),
            "cases": [case.content() for case in self.cases],
            "planned_environment_steps": self.planned_environment_steps,
            "planned_model_constructions": self.planned_model_constructions,
            "profile_id": PROFILE_ID,
            "replay_case_ids": list(self.replay_case_ids),
            "scenario_profile": self.scenario_profile.content(),
            "schema_version": self.schema_version,
            "source_profile": self.source_profile.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        result = self.content()
        result["plan_sha256"] = self.plan_sha256
        return result


def build_qualification_plan(source_profile: SourceProfileLock) -> QualificationPlan:
    """Build the exact deterministic, source-bound 256-case qualification plan."""

    if not isinstance(source_profile, SourceProfileLock):
        raise TypeError("source_profile must be SourceProfileLock")
    scenario_profile = build_anti_torpedo_scenario_profile()
    cases = _cases_from_profile(scenario_profile)
    return QualificationPlan(
        source_profile=source_profile,
        scenario_profile=scenario_profile,
        cases=cases,
        replay_case_ids=tuple(
            case.case_id for case in cases if case.partition == "evaluation"
        ),
    )


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    case: ConfigCase
    episode_seed: int
    worker_slot: int
    replay: bool
    action_trace: tuple[int, ...] = ACTION_TRACE

    def __post_init__(self) -> None:
        if self.episode_seed != _episode_seed(self.case):
            raise WorkloadQualificationError("episode seed differs from worker-independent plan")
        _exact_int("worker_slot", self.worker_slot)
        if type(self.replay) is not bool:
            raise WorkloadQualificationError("replay must be bool")
        if tuple(self.action_trace) != ACTION_TRACE:
            raise WorkloadQualificationError("probe action trace differs")

    def content(self) -> dict[str, object]:
        return {
            "action_trace": list(self.action_trace),
            "case_id": self.case.case_id,
            "config_sha256": self.case.config_sha256,
            "episode_seed": self.episode_seed,
            "replay": self.replay,
            "worker_slot": self.worker_slot,
        }


@dataclass(frozen=True, slots=True)
class RejectionWitness:
    action: int
    rejected: bool
    logical_time_before: float
    logical_time_after: float
    physics_before: str
    physics_after: str

    def __post_init__(self) -> None:
        _exact_int("rejected action", self.action)
        if type(self.rejected) is not bool:
            raise WorkloadQualificationError("rejected must be bool")
        before = _finite("logical_time_before", self.logical_time_before)
        after = _finite("logical_time_after", self.logical_time_after)
        _validate_projection("rejection physics_before", self.physics_before)
        _validate_projection("rejection physics_after", self.physics_after)
        if not self.rejected or before != after or self.physics_before != self.physics_after:
            raise WorkloadQualificationError(
                "rejected action must not advance logical time or mutate physics"
            )

    def content(self) -> dict[str, object]:
        return {
            "action": self.action,
            "logical_time_after": self.logical_time_after,
            "logical_time_before": self.logical_time_before,
            "physics_after": self.physics_after,
            "physics_before": self.physics_before,
            "rejected": self.rejected,
        }


@dataclass(frozen=True, slots=True)
class ProbeObservation:
    materialized_config: str
    initial_projection: str
    sensor_projection: str
    dynamics_projection: str
    decoy_projection: str
    action_masks: tuple[tuple[bool, ...], ...]
    invalid_action_witness: RejectionWitness
    masked_action_witness: RejectionWitness
    steps: int = len(ACTION_TRACE)

    def __post_init__(self) -> None:
        _canonical_object("materialized_config", self.materialized_config)
        for name in (
            "initial_projection",
            "sensor_projection",
            "dynamics_projection",
            "decoy_projection",
        ):
            _validate_projection(name, cast(str, getattr(self, name)))
        masks = tuple(tuple(mask) for mask in self.action_masks)
        if any(any(type(value) is not bool for value in mask) for mask in masks):
            raise WorkloadQualificationError("action masks must contain only bool values")
        if masks != EXPECTED_MASKS:
            raise WorkloadQualificationError("action-mask witness differs from contract")
        if self.invalid_action_witness.action in range(6):
            raise WorkloadQualificationError("invalid-action witness used a valid action")
        if self.masked_action_witness.action not in (3, 4, 5):
            raise WorkloadQualificationError("masked-action witness used an unmasked action")
        if self.steps != len(ACTION_TRACE):
            raise WorkloadQualificationError("probe consumed an unexpected number of steps")
        object.__setattr__(self, "action_masks", masks)

    def semantic_content(self) -> dict[str, object]:
        return {
            "action_masks": [list(mask) for mask in self.action_masks],
            "decoy_projection": self.decoy_projection,
            "dynamics_projection": self.dynamics_projection,
            "initial_projection": self.initial_projection,
            "invalid_action_witness": self.invalid_action_witness.content(),
            "masked_action_witness": self.masked_action_witness.content(),
            "materialized_config": self.materialized_config,
            "sensor_projection": self.sensor_projection,
            "steps": self.steps,
        }


@dataclass(frozen=True, slots=True)
class ProbeExecution:
    request: ProbeRequest
    source_profile_sha256: str
    observation: ProbeObservation
    actual_v2_runtime: ActualV2RuntimeReceipt | None = None

    def __post_init__(self) -> None:
        _sha256("source_profile_sha256", self.source_profile_sha256)


class WorkloadProbeRunner(Protocol):
    """Construction/step seam shared by the actual runner and test doubles."""

    runner_id: str
    runner_source_sha256: str

    def probe(self, request: ProbeRequest) -> ProbeExecution:
        """Construct one fresh model and execute the fixed three-step trace."""


class _V2DiagnosticSnapshot(Protocol):
    canonical_observation: bytes
    logical_time: float
    action_mask: tuple[bool, ...]


class _V2Environment(Protocol):
    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[object, dict[str, object]]: ...

    def step(
        self,
        action: object,
    ) -> tuple[object, float, bool, bool, dict[str, object]]: ...

    def diagnostic_snapshot(self) -> _V2DiagnosticSnapshot: ...

    def close(self) -> None: ...


def _as_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WorkloadQualificationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _physics_projection(
    observation: Mapping[str, object],
    *keys: str,
) -> str:
    missing = [key for key in keys if key not in observation]
    if missing:
        raise WorkloadQualificationError(
            f"v2 observation lacks physics fields: {missing}"
        )
    return _json_text({key: observation[key] for key in keys})


def _snapshot_physics(snapshot: _V2DiagnosticSnapshot) -> str:
    try:
        value = json.loads(
            snapshot.canonical_observation.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkloadQualificationError(
            f"v2 diagnostic observation is invalid: {exc}"
        ) from exc
    observation = _as_mapping("v2 diagnostic observation", value)
    return _physics_projection(
        observation,
        "tick",
        "ship",
        "torpedo",
        "relative",
        "pending_target",
        "decoy_launched",
        "decoys",
        "previous_action",
    )


def _expect_identity(
    name: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise WorkloadQualificationError(f"{name} differs: {key}")


class AntiTorpedoV2ProbeRunner:
    """Execute source-locked ordinal-selected probes on the actual v2 factory."""

    runner_id = ACTUAL_V2_RUNNER_ID

    def __init__(self, source_profile: SourceProfileLock) -> None:
        if not isinstance(source_profile, SourceProfileLock):
            raise TypeError("source_profile must be SourceProfileLock")
        expected = build_actual_v2_source_profile()
        if source_profile != expected:
            raise WorkloadQualificationError(
                "actual v2 runner source profile differs from loaded sources"
            )
        self._source_profile = source_profile
        self.runner_source_sha256 = qualification_implementation_sha256()

    def probe(self, request: ProbeRequest) -> ProbeExecution:
        from pyjevsim_bridge.rl.qualification_models.anti_torpedo import (
            V2_OBSERVATION_SCHEMA_VERSION,
            V2_WORKLOAD_VERSION,
            anti_torpedo_v2_environment_factory,
        )

        environment = cast(
            _V2Environment,
            anti_torpedo_v2_environment_factory(
                instance_id=f"qualification-v2-{request.worker_slot}",
                run_id="TASK-RL-104-workload-profile",
                expected_atsim_model_source_sha256=(
                    self._source_profile.atsim_source_sha256
                ),
                expected_adapter_source_sha256=(
                    self._source_profile.adapter_source_sha256
                ),
                expected_environment_contract_sha256=(
                    self._source_profile.environment_contract_sha256
                ),
                expected_projection_contract_sha256=(
                    self._source_profile.projection_contract_sha256
                ),
                expected_pyjevsim_executor_source_sha256=(
                    self._source_profile.pyjevsim_executor_source_sha256
                ),
                expected_profile_generator_source_sha256=(
                    self._source_profile.scenario_generator_source_sha256
                ),
                expected_scenario_bank_sha256=(
                    self._source_profile.scenario_bank_sha256
                ),
                expected_scenario_family_sha256=(
                    self._source_profile.scenario_family_sha256
                ),
                expected_scenario_source_sha256=(
                    self._source_profile.scenario_source_sha256
                ),
                expected_factor_schema_sha256=(
                    self._source_profile.factor_schema_sha256
                ),
            ),
        )
        try:
            raw_initial, reset_info = environment.reset(
                seed=request.episode_seed,
                options={"scenario_ordinal": request.case.ordinal},
            )
            initial = _as_mapping("v2 reset observation", raw_initial)
            reset_observation_identity = {
                "config_sha256": request.case.config_sha256,
                "factor_vector": request.case.factors,
                "scenario_family_sha256": (
                    self._source_profile.scenario_family_sha256
                ),
                "scenario_ordinal": request.case.ordinal,
                "schema_version": V2_OBSERVATION_SCHEMA_VERSION,
            }
            reset_info_identity = {
                "config_sha256": request.case.config_sha256,
                "environment_contract_sha256": (
                    self._source_profile.environment_contract_sha256
                ),
                "factor_schema_sha256": self._source_profile.factor_schema_sha256,
                "factor_vector": request.case.factors,
                "scenario_bank_sha256": self._source_profile.scenario_bank_sha256,
                "scenario_family_sha256": (
                    self._source_profile.scenario_family_sha256
                ),
                "scenario_id": request.case.case_id,
                "scenario_ordinal": request.case.ordinal,
                "scenario_source_sha256": (
                    self._source_profile.scenario_source_sha256
                ),
                "workload_version": V2_WORKLOAD_VERSION,
            }
            _expect_identity(
                "v2 reset observation", initial, reset_observation_identity
            )
            _expect_identity("v2 reset info", reset_info, reset_info_identity)

            initial_snapshot = environment.diagnostic_snapshot()
            try:
                environment.step(6)
            except ValueError:
                pass
            else:
                raise WorkloadQualificationError(
                    "actual v2 accepted an out-of-range action"
                )
            invalid_after = environment.diagnostic_snapshot()
            if invalid_after != initial_snapshot:
                raise WorkloadQualificationError(
                    "actual v2 invalid-action rejection mutated the episode"
                )
            invalid_physics = _snapshot_physics(initial_snapshot)
            invalid_witness = RejectionWitness(
                action=6,
                rejected=True,
                logical_time_before=initial_snapshot.logical_time,
                logical_time_after=invalid_after.logical_time,
                physics_before=invalid_physics,
                physics_after=_snapshot_physics(invalid_after),
            )

            raw_sensor, _reward, terminated, truncated, first_info = environment.step(0)
            if terminated or truncated:
                raise WorkloadQualificationError(
                    "v2 probe terminated before its sensor witness"
                )
            sensor = _as_mapping("v2 sensor observation", raw_sensor)
            _expect_identity("v2 first-step info", first_info, reset_info_identity)

            raw_launch, _reward, terminated, truncated, launch_info = environment.step(3)
            if terminated or truncated:
                raise WorkloadQualificationError(
                    "v2 probe terminated before its mask witness"
                )
            launch = _as_mapping("v2 launch observation", raw_launch)
            masked_before = environment.diagnostic_snapshot()
            try:
                environment.step(3)
            except ValueError:
                pass
            else:
                raise WorkloadQualificationError("actual v2 accepted a masked action")
            masked_after = environment.diagnostic_snapshot()
            if masked_after != masked_before:
                raise WorkloadQualificationError(
                    "actual v2 masked-action rejection mutated the episode"
                )
            masked_physics = _snapshot_physics(masked_before)
            masked_witness = RejectionWitness(
                action=3,
                rejected=True,
                logical_time_before=masked_before.logical_time,
                logical_time_after=masked_after.logical_time,
                physics_before=masked_physics,
                physics_after=_snapshot_physics(masked_after),
            )

            raw_final, _reward, _terminated, _truncated, final_info = environment.step(0)
            final = _as_mapping("v2 dynamics observation", raw_final)
            masks = (
                cast(tuple[bool, ...], reset_info["action_mask"]),
                cast(tuple[bool, ...], first_info["action_mask"]),
                cast(tuple[bool, ...], launch_info["action_mask"]),
                cast(tuple[bool, ...], final_info["action_mask"]),
            )
            runtime = ActualV2RuntimeReceipt(
                scenario_ordinal=request.case.ordinal,
                config_sha256=cast(str, reset_info["config_sha256"]),
                factor_vector=cast(tuple[int, ...], reset_info["factor_vector"]),
                scenario_bank_sha256=cast(str, reset_info["scenario_bank_sha256"]),
                scenario_family_sha256=cast(
                    str, reset_info["scenario_family_sha256"]
                ),
                scenario_source_sha256=cast(
                    str, reset_info["scenario_source_sha256"]
                ),
                factor_schema_sha256=cast(str, reset_info["factor_schema_sha256"]),
                profile_generator_source_sha256=(
                    self._source_profile.scenario_generator_source_sha256
                ),
                atsim_source_sha256=cast(
                    str, first_info["atsim_model_source_sha256"]
                ),
                adapter_source_sha256=cast(
                    str, first_info["adapter_source_sha256"]
                ),
                environment_contract_sha256=cast(
                    str, first_info["environment_contract_sha256"]
                ),
                projection_contract_sha256=cast(
                    str, first_info["projection_contract_sha256"]
                ),
                pyjevsim_executor_source_sha256=cast(
                    str, first_info["pyjevsim_executor_source_sha256"]
                ),
                qualification_runner_source_sha256=self.runner_source_sha256,
                workload_version=cast(str, first_info["workload_version"]),
                executor_qualified=cast(bool, first_info["executor_qualified"]),
                reset_observation_identity_sha256=_content_sha256(
                    reset_observation_identity
                ),
                reset_info_identity_sha256=_content_sha256(reset_info_identity),
            )
            observation = ProbeObservation(
                materialized_config=request.case.config.canonical_json,
                initial_projection=_physics_projection(
                    initial, "ship", "torpedo", "relative"
                ),
                sensor_projection=_physics_projection(
                    sensor, "pending_target", "relative"
                ),
                dynamics_projection=_physics_projection(
                    final, "ship", "torpedo", "relative"
                ),
                decoy_projection=_physics_projection(
                    launch, "decoy_launched", "decoys"
                ),
                action_masks=masks,
                invalid_action_witness=invalid_witness,
                masked_action_witness=masked_witness,
            )
            return ProbeExecution(
                request=request,
                source_profile_sha256=self._source_profile.source_profile_sha256,
                observation=observation,
                actual_v2_runtime=runtime,
            )
        finally:
            environment.close()


@dataclass(frozen=True, slots=True)
class ProbeReceipt:
    request: ProbeRequest
    status: str
    execution: ProbeExecution | None
    error: str | None
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed"}:
            raise WorkloadQualificationError("probe receipt status is not terminal")
        if self.status == "completed":
            if self.execution is None or self.error is not None:
                raise WorkloadQualificationError("completed probe receipt is inconsistent")
            if self.execution.request != self.request:
                raise WorkloadQualificationError("runner returned a different probe request")
        elif self.execution is not None or not self.error:
            raise WorkloadQualificationError("failed probe receipt is inconsistent")
        object.__setattr__(self, "receipt_sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "error": self.error,
            "execution": None
            if self.execution is None
            else {
                "actual_v2_runtime": None
                if self.execution.actual_v2_runtime is None
                else self.execution.actual_v2_runtime.content(),
                "observation": self.execution.observation.semantic_content(),
                "source_profile_sha256": self.execution.source_profile_sha256,
            },
            "request": self.request.content(),
            "status": self.status,
        }

    def to_dict(self) -> dict[str, object]:
        value = self.content()
        value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class FactorSensitivityReceipt:
    factor: str
    projection: str
    edge_count: int
    changed_edge_count: int
    passed: bool

    def __post_init__(self) -> None:
        if self.factor not in FACTOR_NAMES:
            raise WorkloadQualificationError("unknown sensitivity factor")
        if self.projection not in {"initial", "sensor", "dynamics", "decoy"}:
            raise WorkloadQualificationError("unknown sensitivity projection")
        if self.edge_count != 128:
            raise WorkloadQualificationError("every factor requires 128 matched edges")
        if not 0 <= self.changed_edge_count <= self.edge_count:
            raise WorkloadQualificationError("changed sensitivity edge count is invalid")
        if self.passed != (self.changed_edge_count == self.edge_count):
            raise WorkloadQualificationError("sensitivity disposition differs from evidence")

    def content(self) -> dict[str, object]:
        return {
            "changed_edge_count": self.changed_edge_count,
            "edge_count": self.edge_count,
            "factor": self.factor,
            "passed": self.passed,
            "projection": self.projection,
        }


@dataclass(frozen=True, slots=True)
class WorkloadProfileCapability:
    exact_factorial_passed: bool
    partition_passed: bool
    materialization_passed: bool
    sensitivity_passed: bool
    deterministic_replay_passed: bool
    mask_passed: bool
    source_identity_bound: bool
    receipts_complete: bool
    budget_respected: bool
    ledger_verified: bool
    actual_v2_integration_accepted: bool
    actual_v2_evidence_sha256: str | None
    blockers: tuple[str, ...]
    workload_profile_qualified: bool
    claim_scope: str = "workload-profile-only"

    def __post_init__(self) -> None:
        values = (
            self.exact_factorial_passed,
            self.partition_passed,
            self.materialization_passed,
            self.sensitivity_passed,
            self.deterministic_replay_passed,
            self.mask_passed,
            self.source_identity_bound,
            self.receipts_complete,
            self.budget_respected,
            self.ledger_verified,
            self.actual_v2_integration_accepted,
            self.workload_profile_qualified,
        )
        if any(type(value) is not bool for value in values):
            raise WorkloadQualificationError("capability gates must be bool")
        expected = all(values[:-1])
        if self.workload_profile_qualified != expected:
            raise WorkloadQualificationError("capability is not computed from all gates")
        if self.workload_profile_qualified == bool(self.blockers):
            raise WorkloadQualificationError("capability blockers differ from disposition")
        if self.actual_v2_integration_accepted:
            if self.actual_v2_evidence_sha256 is None:
                raise WorkloadQualificationError(
                    "caller cannot fabricate actual v2 acceptance"
                )
            _sha256("actual_v2_evidence_sha256", self.actual_v2_evidence_sha256)
        elif self.actual_v2_evidence_sha256 is not None:
            raise WorkloadQualificationError(
                "unaccepted actual v2 evidence digest is forbidden"
            )
        if self.claim_scope != "workload-profile-only":
            raise WorkloadQualificationError("qualification claim scope widened")

    def content(self) -> dict[str, object]:
        return {
            "actual_v2_integration_accepted": self.actual_v2_integration_accepted,
            "actual_v2_evidence_sha256": self.actual_v2_evidence_sha256,
            "blockers": list(self.blockers),
            "budget_respected": self.budget_respected,
            "claim_scope": self.claim_scope,
            "deterministic_replay_passed": self.deterministic_replay_passed,
            "disallowed_claims": list(DISALLOWED_CLAIMS),
            "exact_factorial_passed": self.exact_factorial_passed,
            "ledger_verified": self.ledger_verified,
            "mask_passed": self.mask_passed,
            "materialization_passed": self.materialization_passed,
            "partition_passed": self.partition_passed,
            "receipts_complete": self.receipts_complete,
            "sensitivity_passed": self.sensitivity_passed,
            "source_identity_bound": self.source_identity_bound,
            "workload_profile_qualified": self.workload_profile_qualified,
        }


@dataclass(frozen=True, slots=True)
class QualificationResult:
    plan: QualificationPlan
    primary_receipts: tuple[ProbeReceipt, ...]
    replay_receipts: tuple[ProbeReceipt, ...]
    sensitivity_receipts: tuple[FactorSensitivityReceipt, ...]
    capability: WorkloadProfileCapability
    model_constructions: int
    environment_steps: int
    result_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if len(self.primary_receipts) != CASE_COUNT:
            raise WorkloadQualificationError("result requires 256 primary receipts")
        if len(self.replay_receipts) != REPLAY_COUNT:
            raise WorkloadQualificationError("result requires 64 replay receipts")
        all_receipts = self.primary_receipts + self.replay_receipts
        keys = [
            (receipt.request.case.case_id, receipt.request.replay)
            for receipt in all_receipts
        ]
        if len(set(keys)) != len(keys):
            raise WorkloadQualificationError("result contains duplicate terminal receipts")
        if len(self.sensitivity_receipts) != len(FACTOR_NAMES):
            raise WorkloadQualificationError("result requires eight sensitivity receipts")
        if {item.factor for item in self.sensitivity_receipts} != set(FACTOR_NAMES):
            raise WorkloadQualificationError("sensitivity factors are incomplete")
        if self.model_constructions != len(all_receipts):
            raise WorkloadQualificationError("construction count differs from receipts")
        if self.model_constructions > MAX_MODEL_CONSTRUCTIONS:
            raise WorkloadQualificationError("construction hard bound exceeded")
        if not 0 <= self.environment_steps <= MAX_ENVIRONMENT_STEPS:
            raise WorkloadQualificationError("environment step hard bound exceeded")
        object.__setattr__(self, "result_sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "capability": self.capability.content(),
            "environment_steps": self.environment_steps,
            "model_constructions": self.model_constructions,
            "plan_sha256": self.plan.plan_sha256,
            "primary_receipts": [item.to_dict() for item in self.primary_receipts],
            "replay_receipts": [item.to_dict() for item in self.replay_receipts],
            "schema_version": SCHEMA_VERSION,
            "sensitivity_receipts": [
                item.content() for item in self.sensitivity_receipts
            ],
        }


def _terminal_receipt(runner: WorkloadProbeRunner, request: ProbeRequest) -> ProbeReceipt:
    try:
        execution = runner.probe(request)
    except Exception as exc:
        return ProbeReceipt(
            request=request,
            status="failed",
            execution=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ProbeReceipt(request=request, status="completed", execution=execution, error=None)


def _completed_by_case(receipts: Sequence[ProbeReceipt]) -> dict[str, ProbeObservation]:
    return {
        receipt.request.case.case_id: receipt.execution.observation
        for receipt in receipts
        if receipt.execution is not None
    }


def _sensitivity_receipts(
    plan: QualificationPlan,
    primary: Sequence[ProbeReceipt],
) -> tuple[FactorSensitivityReceipt, ...]:
    observations = _completed_by_case(primary)
    results: list[FactorSensitivityReceipt] = []
    for factor_index, factor in enumerate(FACTOR_SPECS):
        factor_name = factor.factor_id
        if factor.sensitivity_stage is SensitivityStage.POST_DEPLOY_TRACE:
            projection_name = "decoy"
        elif factor.variation_class is VariationClass.SENSOR:
            projection_name = "sensor"
        elif factor.variation_class is VariationClass.DYNAMICS:
            projection_name = "dynamics"
        else:
            projection_name = "initial"
        changed = 0
        edges = 0
        for case in plan.cases:
            if case.factors[factor_index] != 0:
                continue
            paired = plan.cases[case.ordinal | (1 << factor_index)]
            edges += 1
            left = observations.get(case.case_id)
            right = observations.get(paired.case_id)
            if left is None or right is None:
                continue
            attribute = f"{projection_name}_projection"
            if getattr(left, attribute) != getattr(right, attribute):
                changed += 1
        results.append(
            FactorSensitivityReceipt(
                factor=factor_name,
                projection=projection_name,
                edge_count=edges,
                changed_edge_count=changed,
                passed=changed == edges,
            )
        )
    return tuple(results)


def _runtime_receipt_matches(
    plan: QualificationPlan,
    receipt: ProbeReceipt,
) -> bool:
    if receipt.execution is None or receipt.execution.actual_v2_runtime is None:
        return False
    runtime = receipt.execution.actual_v2_runtime
    case = receipt.request.case
    source = plan.source_profile
    expected_observation_identity = {
        "config_sha256": case.config_sha256,
        "factor_vector": case.factors,
        "scenario_family_sha256": source.scenario_family_sha256,
        "scenario_ordinal": case.ordinal,
        "schema_version": "anti-torpedo-observation-v2",
    }
    expected_info_identity = {
        "config_sha256": case.config_sha256,
        "environment_contract_sha256": source.environment_contract_sha256,
        "factor_schema_sha256": source.factor_schema_sha256,
        "factor_vector": case.factors,
        "scenario_bank_sha256": source.scenario_bank_sha256,
        "scenario_family_sha256": source.scenario_family_sha256,
        "scenario_id": case.case_id,
        "scenario_ordinal": case.ordinal,
        "scenario_source_sha256": source.scenario_source_sha256,
        "workload_version": "AntiTorpedoCountermeasure-v2",
    }
    return (
        runtime.scenario_ordinal == case.ordinal
        and runtime.config_sha256 == case.config_sha256
        and runtime.factor_vector == case.factors
        and runtime.scenario_bank_sha256 == source.scenario_bank_sha256
        and runtime.scenario_family_sha256 == source.scenario_family_sha256
        and runtime.scenario_source_sha256 == source.scenario_source_sha256
        and runtime.factor_schema_sha256 == source.factor_schema_sha256
        and runtime.profile_generator_source_sha256
        == source.scenario_generator_source_sha256
        and runtime.atsim_source_sha256 == source.atsim_source_sha256
        and runtime.adapter_source_sha256 == source.adapter_source_sha256
        and runtime.environment_contract_sha256 == source.environment_contract_sha256
        and runtime.projection_contract_sha256 == source.projection_contract_sha256
        and runtime.pyjevsim_executor_source_sha256
        == source.pyjevsim_executor_source_sha256
        and runtime.qualification_runner_source_sha256
        == source.qualification_runner_source_sha256
        and runtime.workload_version == "AntiTorpedoCountermeasure-v2"
        and runtime.executor_qualified
        and runtime.reset_observation_identity_sha256
        == _content_sha256(expected_observation_identity)
        and runtime.reset_info_identity_sha256
        == _content_sha256(expected_info_identity)
    )


def _capability(
    plan: QualificationPlan,
    primary: Sequence[ProbeReceipt],
    replays: Sequence[ProbeReceipt],
    sensitivities: Sequence[FactorSensitivityReceipt],
    *,
    constructions: int,
    steps: int,
    ledger_verified: bool,
    actual_v2_runner_used: bool,
) -> WorkloadProfileCapability:
    primary_map = _completed_by_case(primary)
    replay_map = _completed_by_case(replays)
    exact = (
        len(plan.cases) == CASE_COUNT
        and len({case.config_sha256 for case in plan.cases}) == CASE_COUNT
    )
    partition = plan.cases == _cases_from_profile(plan.scenario_profile)
    receipts_complete = (
        len(primary) == CASE_COUNT
        and len(replays) == REPLAY_COUNT
        and all(item.status == "completed" for item in (*primary, *replays))
    )
    materialization = receipts_complete and all(
        primary_map[case.case_id].materialized_config == case.config.canonical_json
        for case in plan.cases
    )
    sensitivity = len(sensitivities) == 8 and all(item.passed for item in sensitivities)
    replay = receipts_complete and all(
        case_id in replay_map
        and case_id in primary_map
        and replay_map[case_id].semantic_content() == primary_map[case_id].semantic_content()
        for case_id in plan.replay_case_ids
    )
    mask = receipts_complete and all(
        observation.action_masks == EXPECTED_MASKS
        for observation in (*primary_map.values(), *replay_map.values())
    )
    source = receipts_complete and all(
        receipt.execution is not None
        and receipt.execution.source_profile_sha256
        == plan.source_profile.source_profile_sha256
        for receipt in (*primary, *replays)
    )
    budget = (
        constructions == PLANNED_MODEL_CONSTRUCTIONS
        and constructions <= MAX_MODEL_CONSTRUCTIONS
        and steps == PLANNED_ENVIRONMENT_STEPS
        and steps <= MAX_ENVIRONMENT_STEPS
    )
    actual_v2 = actual_v2_runner_used and all(
        _runtime_receipt_matches(plan, receipt)
        for receipt in (*primary, *replays)
    )
    actual_v2_evidence_sha256 = (
        _content_sha256(
            [
                cast(ActualV2RuntimeReceipt, receipt.execution.actual_v2_runtime).content()
                for receipt in (*primary, *replays)
                if receipt.execution is not None
            ]
        )
        if actual_v2
        else None
    )
    gates = {
        "exact-factorial-failed": exact,
        "partition-failed": partition,
        "materialization-failed": materialization,
        "factor-sensitivity-failed": sensitivity,
        "deterministic-replay-failed": replay,
        "action-mask-failed": mask,
        "source-identity-not-bound": source,
        "terminal-receipts-incomplete": receipts_complete,
        "bounded-execution-failed": budget,
        "immutable-ledger-not-verified": ledger_verified,
        "actual-v2-integration-not-executed": actual_v2,
    }
    blockers = tuple(name for name, passed in gates.items() if not passed)
    return WorkloadProfileCapability(
        exact_factorial_passed=exact,
        partition_passed=partition,
        materialization_passed=materialization,
        sensitivity_passed=sensitivity,
        deterministic_replay_passed=replay,
        mask_passed=mask,
        source_identity_bound=source,
        receipts_complete=receipts_complete,
        budget_respected=budget,
        ledger_verified=ledger_verified,
        actual_v2_integration_accepted=actual_v2,
        actual_v2_evidence_sha256=actual_v2_evidence_sha256,
        blockers=blockers,
        workload_profile_qualified=not blockers,
    )


def run_workload_qualification(
    plan: QualificationPlan,
    runner: WorkloadProbeRunner,
    *,
    expected_plan_sha256: str,
) -> QualificationResult:
    """Execute exactly 256 probes and 64 deterministic replays within hard bounds."""

    if not isinstance(plan, QualificationPlan):
        raise TypeError("plan must be QualificationPlan")
    if _sha256("expected_plan_sha256", expected_plan_sha256) != plan.plan_sha256:
        raise WorkloadQualificationError("qualification plan digest mismatch")
    if runner.runner_id != plan.source_profile.runner_id:
        raise WorkloadQualificationError("runner identity differs from source profile")
    if (
        _sha256("runner_source_sha256", runner.runner_source_sha256)
        != plan.source_profile.qualification_runner_source_sha256
    ):
        raise WorkloadQualificationError("runner source differs from source profile")

    primary: list[ProbeReceipt] = []
    for case in plan.cases:
        primary.append(
            _terminal_receipt(
                runner,
                ProbeRequest(
                    case=case,
                    episode_seed=_episode_seed(case),
                    worker_slot=0,
                    replay=False,
                ),
            )
        )
    case_by_id = {case.case_id: case for case in plan.cases}
    replays: list[ProbeReceipt] = []
    for index, case_id in enumerate(reversed(plan.replay_case_ids)):
        case = case_by_id[case_id]
        replays.append(
            _terminal_receipt(
                runner,
                ProbeRequest(
                    case=case,
                    episode_seed=_episode_seed(case),
                    worker_slot=(index % 4) + 1,
                    replay=True,
                ),
            )
        )
    constructions = len(primary) + len(replays)
    steps = sum(
        receipt.execution.observation.steps
        for receipt in (*primary, *replays)
        if receipt.execution is not None
    )
    sensitivities = _sensitivity_receipts(plan, primary)
    capability = _capability(
        plan,
        primary,
        replays,
        sensitivities,
        constructions=constructions,
        steps=steps,
        ledger_verified=False,
        actual_v2_runner_used=type(runner) is AntiTorpedoV2ProbeRunner,
    )
    return QualificationResult(
        plan=plan,
        primary_receipts=tuple(primary),
        replay_receipts=tuple(replays),
        sensitivity_receipts=sensitivities,
        capability=capability,
        model_constructions=constructions,
        environment_steps=steps,
    )


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    role: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in REQUIRED_ARTIFACT_ROLES:
            raise WorkloadQualificationError("unknown ledger artifact role")
        _non_empty("ledger path", self.path)
        pure = PurePosixPath(self.path)
        if pure.is_absolute() or ".." in pure.parts or self.path != pure.as_posix():
            raise WorkloadQualificationError("ledger path must be normalized and relative")
        if self.media_type != "application/json":
            raise WorkloadQualificationError("qualification artifacts must be JSON")
        _exact_int("artifact size_bytes", self.size_bytes)
        object.__setattr__(self, "sha256", _sha256("artifact sha256", self.sha256))

    def content(self) -> dict[str, object]:
        return {
            "media_type": self.media_type,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class QualificationLedger:
    plan_sha256: str
    result_sha256: str
    entries: tuple[LedgerEntry, ...]
    schema_version: str = SCHEMA_VERSION
    ledger_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise WorkloadQualificationError("unsupported ledger schema")
        object.__setattr__(self, "plan_sha256", _sha256("plan_sha256", self.plan_sha256))
        object.__setattr__(self, "result_sha256", _sha256("result_sha256", self.result_sha256))
        entries = tuple(sorted(self.entries, key=lambda item: item.path))
        if len({item.path for item in entries}) != len(entries):
            raise WorkloadQualificationError("ledger contains duplicate artifact paths")
        if len({item.role for item in entries}) != len(entries):
            raise WorkloadQualificationError("ledger contains duplicate artifact roles")
        if {item.role for item in entries} != set(REQUIRED_ARTIFACT_ROLES):
            raise WorkloadQualificationError("ledger artifact roles are incomplete")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "ledger_sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "canonical_json_id": CANONICAL_JSON_ID,
            "entries": [entry.content() for entry in self.entries],
            "plan_sha256": self.plan_sha256,
            "result_sha256": self.result_sha256,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        value = self.content()
        value["ledger_sha256"] = self.ledger_sha256
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        fields = {
            "canonical_json_id",
            "entries",
            "ledger_sha256",
            "plan_sha256",
            "result_sha256",
            "schema_version",
        }
        if set(value) != fields or value.get("canonical_json_id") != CANONICAL_JSON_ID:
            raise WorkloadQualificationError("ledger fields differ from schema")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list):
            raise WorkloadQualificationError("ledger entries must be an array")
        entries: list[LedgerEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, Mapping) or set(raw) != {
                "media_type",
                "path",
                "role",
                "sha256",
                "size_bytes",
            }:
                raise WorkloadQualificationError("ledger entry fields differ from schema")
            entries.append(
                LedgerEntry(
                    role=cast(str, raw["role"]),
                    path=cast(str, raw["path"]),
                    media_type=cast(str, raw["media_type"]),
                    size_bytes=cast(int, raw["size_bytes"]),
                    sha256=cast(str, raw["sha256"]),
                )
            )
        ledger = cls(
            plan_sha256=cast(str, value["plan_sha256"]),
            result_sha256=cast(str, value["result_sha256"]),
            entries=tuple(entries),
            schema_version=cast(str, value["schema_version"]),
        )
        if value["ledger_sha256"] != ledger.ledger_sha256:
            raise WorkloadQualificationError("ledger content digest differs")
        return ledger


@dataclass(frozen=True, slots=True)
class WrittenQualificationLedger:
    ledger: QualificationLedger
    path: Path
    file_sha256: str


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def write_qualification_bundle(
    root: Path,
    result: QualificationResult,
) -> WrittenQualificationLedger:
    """Create a canonical artifact bundle and ledger without replacing any file."""

    if not isinstance(root, Path) or not isinstance(result, QualificationResult):
        raise TypeError("root/result types differ from qualification bundle API")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise WorkloadQualificationError("artifact root must be a real directory")
    artifacts: dict[str, tuple[str, object]] = {
        "capability": ("capability.json", result.capability.content()),
        "configs": ("configs.json", [case.content() for case in result.plan.cases]),
        "mask": (
            "mask.json",
            {
                "expected_masks": [list(mask) for mask in EXPECTED_MASKS],
                "passed": result.capability.mask_passed,
            },
        ),
        "partition": (
            "partition.json",
            {
                "assignments": [
                    {
                        "case_id": case.case_id,
                        "config_sha256": case.config_sha256,
                        "fold": case.fold,
                        "partition": case.partition,
                    }
                    for case in result.plan.cases
                ],
                "gf2_equations": [
                    "b0^b3^b4^b6^b7",
                    "b1^b3^b5^b6",
                    "b2^b4^b5^b6",
                ],
            },
        ),
        "plan": ("plan.json", result.plan.to_dict()),
        "probes": ("probes.json", [item.to_dict() for item in result.primary_receipts]),
        "replays": ("replays.json", [item.to_dict() for item in result.replay_receipts]),
        "sensitivity": (
            "sensitivity.json",
            [item.content() for item in result.sensitivity_receipts],
        ),
        "source-profile": ("source-profile.json", result.plan.source_profile.to_dict()),
    }
    all_paths = [resolved / filename for filename, _ in artifacts.values()]
    ledger_path = resolved / "ledger.json"
    if any(path.exists() for path in (*all_paths, ledger_path)):
        raise FileExistsError("qualification bundle never replaces existing artifacts")
    entries: list[LedgerEntry] = []
    for role in REQUIRED_ARTIFACT_ROLES:
        filename, content = artifacts[role]
        payload = _canonical_json(content)
        path = resolved / filename
        _write_exclusive(path, payload)
        entries.append(
            LedgerEntry(
                role=role,
                path=filename,
                media_type="application/json",
                size_bytes=len(payload),
                sha256=_sha256_bytes(payload),
            )
        )
    ledger = QualificationLedger(
        plan_sha256=result.plan.plan_sha256,
        result_sha256=result.result_sha256,
        entries=tuple(entries),
    )
    ledger_payload = _canonical_json(ledger.to_dict())
    _write_exclusive(ledger_path, ledger_payload)
    written = WrittenQualificationLedger(
        ledger=ledger,
        path=ledger_path,
        file_sha256=_sha256_bytes(ledger_payload),
    )
    verify_qualification_bundle(
        resolved,
        expected_ledger_file_sha256=written.file_sha256,
        expected_result_sha256=result.result_sha256,
    )
    return written


def verify_qualification_bundle(
    root: Path,
    *,
    expected_ledger_file_sha256: str,
    expected_result_sha256: str,
) -> QualificationLedger:
    """Read canonical ledger bytes and resolve/re-hash every indexed artifact."""

    expected_file = _sha256("expected_ledger_file_sha256", expected_ledger_file_sha256)
    expected_result = _sha256("expected_result_sha256", expected_result_sha256)
    resolved = root.resolve(strict=True)
    ledger_path = resolved / "ledger.json"
    payload = ledger_path.read_bytes()
    if _sha256_bytes(payload) != expected_file:
        raise WorkloadQualificationError("ledger file digest differs")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkloadQualificationError(f"ledger JSON is invalid: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise WorkloadQualificationError("ledger root must be an object")
    ledger = QualificationLedger.from_dict(cast(Mapping[str, object], decoded))
    if payload != _canonical_json(ledger.to_dict()):
        raise WorkloadQualificationError("ledger JSON is not canonical")
    if ledger.result_sha256 != expected_result:
        raise WorkloadQualificationError("ledger result digest differs")
    for entry in ledger.entries:
        target = (resolved / Path(*PurePosixPath(entry.path).parts)).resolve(strict=True)
        try:
            target.relative_to(resolved)
        except ValueError as exc:
            raise WorkloadQualificationError("ledger artifact escapes root") from exc
        if not target.is_file() or target.is_symlink():
            raise WorkloadQualificationError("ledger artifact is not a regular file")
        artifact = target.read_bytes()
        if len(artifact) != entry.size_bytes or _sha256_bytes(artifact) != entry.sha256:
            raise WorkloadQualificationError(f"ledger artifact differs: {entry.path}")
    return ledger


def admit_verified_bundle(
    result: QualificationResult,
    root: Path,
    *,
    expected_ledger_file_sha256: str,
) -> WorkloadProfileCapability:
    """Re-hash the bundle and recompute the workload-only capability."""

    verify_qualification_bundle(
        root,
        expected_ledger_file_sha256=expected_ledger_file_sha256,
        expected_result_sha256=result.result_sha256,
    )
    return _capability(
        result.plan,
        result.primary_receipts,
        result.replay_receipts,
        result.sensitivity_receipts,
        constructions=result.model_constructions,
        steps=result.environment_steps,
        ledger_verified=True,
        actual_v2_runner_used=result.capability.actual_v2_integration_accepted,
    )


__all__ = [
    "ACTION_TRACE",
    "ACTUAL_V2_RUNNER_ID",
    "ActualV2RuntimeReceipt",
    "AntiTorpedoV2ProbeRunner",
    "CASE_COUNT",
    "ConfigCase",
    "DISALLOWED_CLAIMS",
    "FACTOR_NAMES",
    "MAX_ENVIRONMENT_STEPS",
    "MAX_MODEL_CONSTRUCTIONS",
    "PLANNED_ENVIRONMENT_STEPS",
    "PLANNED_MODEL_CONSTRUCTIONS",
    "ProbeExecution",
    "ProbeObservation",
    "ProbeReceipt",
    "ProbeRequest",
    "QualificationLedger",
    "QualificationPlan",
    "QualificationResult",
    "RejectionWitness",
    "SourceProfileLock",
    "WorkloadProbeRunner",
    "WorkloadProfileCapability",
    "WorkloadQualificationError",
    "admit_verified_bundle",
    "build_actual_v2_source_profile",
    "build_qualification_plan",
    "qualification_implementation_sha256",
    "run_workload_qualification",
    "verify_qualification_bundle",
    "write_qualification_bundle",
]
