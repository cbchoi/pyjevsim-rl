"""Executable, non-running protocol for the MS-RL-11 matched experiment.

This module freezes identities, budgets, admission evidence, and the evidence
ledger.  It deliberately does not import a learner, PyJevSim, Ray, or gorti and
does not execute an experiment.  A missing native implementation is evidence,
not a zero-valued sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, Self, cast

MATCHED_MANIFEST_SCHEMA_VERSION: Final = "matched-native-learning-v1"
MATCHED_LEDGER_SCHEMA_VERSION: Final = "matched-native-ledger-v1"
CANONICAL_JSON_ID: Final = "sorted-compact-utf8-no-nan-v1"
BOOTSTRAP_METHOD: Final = "deterministic-sha256-paired-bca-v1"
BOOTSTRAP_SEED_DERIVATION_ID: Final = "matched-native-bca-v1"
MULTIPLICITY_METHOD: Final = "bonferroni-two-primary-one-sided-v1"
EXCLUSION_POLICY: Final = "retain-all-terminal-receipts-v1"
RERUN_POLICY: Final = "pre-first-commit-infrastructure-once-v1"
FAILURE_POLICY: Final = "terminal-nonsample-never-zero-v1"

TUNING_CONFIG_COUNT: Final = 32
MEASURED_CONFIG_COUNT: Final = 160
EVALUATION_CONFIG_COUNT: Final = 64
TUNING_MASTER_COUNT: Final = 5
TUNING_CANDIDATE_COUNT: Final = 8
TUNING_STEPS_PER_CANDIDATE: Final = 20_000
TUNING_LEARNER_ENGINE_COUNT: Final = 2
TOTAL_TUNING_STEP_BUDGET: Final = 1_600_000
MEASURED_MASTER_COUNT: Final = 20
MEASURED_STEPS_PER_SESSION: Final = 50_000
CHECKPOINT_INTERVAL_STEPS: Final = 5_000
CHECKPOINT_STEPS: Final = tuple(range(0, 50_001, 5_000))
EVALUATION_EPISODES_PER_CHECKPOINT: Final = 64
EVALUATION_EPISODES_PER_SESSION: Final = 704
MAX_EVALUATION_STEPS_PER_EPISODE: Final = 30
EVALUATION_STEP_BUDGET_PER_SESSION: Final = 21_120
TOTAL_MEASURED_TRAINING_BUDGET: Final = 4_000_000
MEASURED_SESSION_COUNT: Final = 80
MEASURED_CHECKPOINT_COUNT: Final = 880
MEASURED_EVALUATION_EPISODE_COUNT: Final = 56_320
BOOTSTRAP_SAMPLES: Final = 50_000

PRIMARY_CELL_IDS: Final = (
    "local-reference",
    "joined-gorti-reference",
    "local-rllib",
    "joined-gorti-rllib",
)
REQUIRED_SOURCE_COMPONENTS: Final = frozenset(
    {
        "adapter",
        "analysis",
        "atsim",
        "feature-contract",
        "fom",
        "gorti",
        "pyjevsim",
        "reference-learner",
        "reward-termination-contract",
        "rllib",
        "scenario-family",
        "semantic-projection",
    }
)
REQUIRED_LEDGER_ROLES: Final = frozenset(
    {
        "manifest",
        "plan",
        "terminal-receipts",
        "raw-transition-shard",
        "raw-metric-shard",
        "policy-artifact",
        "checkpoint-artifact",
        "capability-probe",
        "semantic-parity",
        "analysis-input",
        "analysis-output",
        "review-disposition",
        "external-registration",
    }
)
REPORTED_ENDPOINTS: Final = (
    "normalized-holdout-capture-auc",
    "final-holdout-return",
    "return-auc",
    "steps-to-threshold",
    "valid-transition-throughput",
    "valid-episode-throughput",
    "step-latency-p50",
    "step-latency-p95",
    "step-latency-p99",
    "policy-lag",
    "duplicate-rate",
    "reject-rate",
    "failure-rate",
    "semantic-parity",
)

_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class MatchedProtocolError(ValueError):
    """A value cannot enter the matched native experiment protocol."""


class RolloutPlane(StrEnum):
    LOCAL_SPAWN = "local-spawn"
    JOINED_GORTI = "joined-gorti"


class LearnerEngine(StrEnum):
    REFERENCE_PPO = "reference-ppo"
    NATIVE_RLLIB_PPO = "native-rllib-ppo"


class RunStatus(StrEnum):
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED_NATIVE_UNAVAILABLE = "skipped-native-unavailable"
    EXCLUDED_BY_PROTOCOL = "excluded-by-protocol"


class CapabilityKind(StrEnum):
    RLLIB = "rllib"
    DEXSIM = "dexsim"


class CapabilityDisposition(StrEnum):
    NATIVE_ADMITTED = "native-admitted"
    SKIPPED_NATIVE_UNAVAILABLE = "skipped-native-unavailable"
    EMULATED = "emulated"
    REJECTED = "rejected"


class RegistrationStage(StrEnum):
    PROTOCOL = "protocol-before-tuning"
    MEASURED = "measured-after-tuning-before-measured"


class ClaimKind(StrEnum):
    FULL_2X2 = "full-2x2-learning"
    REFERENCE_BACKEND = "reference-joined-vs-local"
    RLLIB_BACKEND = "rllib-joined-vs-local"
    LEARNER_DESCRIPTIVE = "within-backend-learner-descriptions"
    DEXSIM = "native-dexsim"
    TIMING = "joined-timing-gain"
    CROSS_HOST = "cross-host-scalability"
    CROSS_MODEL = "cross-model-generalization"
    SCIE_PACKAGE = "scie-evidence-package"


def _non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MatchedProtocolError(f"{name} must be a non-empty string")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MatchedProtocolError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    result = _non_negative_int(name, value)
    if result == 0:
        raise MatchedProtocolError(f"{name} must be a positive integer")
    return result


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatchedProtocolError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MatchedProtocolError(f"{name} must be a finite number")
    return result


def _sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MatchedProtocolError(
            f"{name} must be a lowercase SHA-256 hex digest"
        )
    return value


def _strings(name: str, values: Sequence[str], *, exact: int | None = None) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MatchedProtocolError(f"{name} must be a sequence of strings")
    result = tuple(_non_empty(f"{name}[{index}]", item) for index, item in enumerate(values))
    if exact is not None and len(result) != exact:
        raise MatchedProtocolError(f"{name} must contain exactly {exact} values")
    if len(result) != len(set(result)):
        raise MatchedProtocolError(f"{name} must not contain duplicates")
    return result


def _seeds(name: str, values: Sequence[int], *, exact: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MatchedProtocolError(f"{name} must be a sequence of seeds")
    result = tuple(
        _non_negative_int(f"{name}[{index}]", item)
        for index, item in enumerate(values)
    )
    if len(result) != exact:
        raise MatchedProtocolError(f"{name} must contain exactly {exact} seeds")
    if len(result) != len(set(result)):
        raise MatchedProtocolError(f"{name} must not contain duplicate seeds")
    return result


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
        raise MatchedProtocolError(f"value is not canonical JSON: {exc}") from exc


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _parse_rfc3339(name: str, value: object) -> datetime:
    text = _non_empty(name, value)
    if _RFC3339.fullmatch(text) is None:
        raise MatchedProtocolError(f"{name} must be an RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MatchedProtocolError(f"{name} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MatchedProtocolError(f"{name} must contain a timezone")
    return parsed


@dataclass(frozen=True, slots=True)
class SourceLock:
    component_id: str
    version: str
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        _non_empty("component_id", self.component_id)
        _non_empty("source version", self.version)
        _non_empty("source revision", self.revision)
        _sha256("source sha256", self.sha256)

    def content(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "revision": self.revision,
            "sha256": self.sha256,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ScenarioCase:
    seed: int
    config_sha256: str

    def __post_init__(self) -> None:
        _non_negative_int("scenario seed", self.seed)
        _sha256("scenario config_sha256", self.config_sha256)

    def content(self) -> dict[str, object]:
        return {"config_sha256": self.config_sha256, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class ScenarioPartition:
    family_id: str
    family_sha256: str
    variation_sources: tuple[str, ...]
    tuning: tuple[ScenarioCase, ...]
    measured_training: tuple[ScenarioCase, ...]
    evaluation: tuple[ScenarioCase, ...]
    effective_analysis_unit: str = "measured-master-seed"
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty("scenario family_id", self.family_id)
        _sha256("scenario family_sha256", self.family_sha256)
        variation = _strings("variation_sources", self.variation_sources)
        if set(variation) != {"initial-state", "sensor", "dynamics"}:
            raise MatchedProtocolError(
                "variation_sources must declare initial-state, sensor, and dynamics"
            )
        if self.effective_analysis_unit != "measured-master-seed":
            raise MatchedProtocolError(
                "effective_analysis_unit must be measured-master-seed"
            )
        partitions = (
            ("tuning", tuple(self.tuning), TUNING_CONFIG_COUNT),
            ("measured_training", tuple(self.measured_training), MEASURED_CONFIG_COUNT),
            ("evaluation", tuple(self.evaluation), EVALUATION_CONFIG_COUNT),
        )
        all_seeds: set[int] = set()
        all_digests: set[str] = set()
        for name, cases, expected in partitions:
            if len(cases) != expected:
                raise MatchedProtocolError(
                    f"scenario {name} must contain exactly {expected} cases"
                )
            if any(not isinstance(case, ScenarioCase) for case in cases):
                raise MatchedProtocolError(
                    f"scenario {name} may contain only ScenarioCase values"
                )
            seeds = {case.seed for case in cases}
            digests = {case.config_sha256 for case in cases}
            if len(seeds) != expected or len(digests) != expected:
                raise MatchedProtocolError(
                    f"scenario {name} seeds and config digests must be distinct"
                )
            if all_seeds.intersection(seeds) or all_digests.intersection(digests):
                raise MatchedProtocolError(
                    "scenario seed and config-digest partitions must be disjoint"
                )
            all_seeds.update(seeds)
            all_digests.update(digests)
            object.__setattr__(self, name, cases)
        object.__setattr__(self, "variation_sources", variation)
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "effective_analysis_unit": self.effective_analysis_unit,
            "evaluation": [case.content() for case in self.evaluation],
            "family_id": self.family_id,
            "family_sha256": self.family_sha256,
            "measured_training": [
                case.content() for case in self.measured_training
            ],
            "tuning": [case.content() for case in self.tuning],
            "variation_sources": list(self.variation_sources),
        }


@dataclass(frozen=True, slots=True)
class CellSpec:
    cell_id: str
    rollout_plane: RolloutPlane
    learner_engine: LearnerEngine
    training_environment_steps: int = MEASURED_STEPS_PER_SESSION
    evaluation_episodes: int = EVALUATION_EPISODES_PER_SESSION
    evaluation_environment_step_budget: int = EVALUATION_STEP_BUDGET_PER_SESSION
    warmup_environment_steps: int = 0
    actor_workers: int = 4
    learner_workers: int = 1
    max_cpu_cores: int = 5
    threads_per_process: int = 1
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        expected = {
            "local-reference": (
                RolloutPlane.LOCAL_SPAWN,
                LearnerEngine.REFERENCE_PPO,
            ),
            "joined-gorti-reference": (
                RolloutPlane.JOINED_GORTI,
                LearnerEngine.REFERENCE_PPO,
            ),
            "local-rllib": (
                RolloutPlane.LOCAL_SPAWN,
                LearnerEngine.NATIVE_RLLIB_PPO,
            ),
            "joined-gorti-rllib": (
                RolloutPlane.JOINED_GORTI,
                LearnerEngine.NATIVE_RLLIB_PPO,
            ),
        }
        if self.cell_id not in expected:
            raise MatchedProtocolError(f"unknown primary cell {self.cell_id!r}")
        if (self.rollout_plane, self.learner_engine) != expected[self.cell_id]:
            raise MatchedProtocolError("cell identity differs from its 2x2 factors")
        exact_values = {
            "training_environment_steps": (
                self.training_environment_steps,
                MEASURED_STEPS_PER_SESSION,
            ),
            "evaluation_episodes": (
                self.evaluation_episodes,
                EVALUATION_EPISODES_PER_SESSION,
            ),
            "evaluation_environment_step_budget": (
                self.evaluation_environment_step_budget,
                EVALUATION_STEP_BUDGET_PER_SESSION,
            ),
            "warmup_environment_steps": (self.warmup_environment_steps, 0),
            "actor_workers": (self.actor_workers, 4),
            "learner_workers": (self.learner_workers, 1),
            "max_cpu_cores": (self.max_cpu_cores, 5),
            "threads_per_process": (self.threads_per_process, 1),
        }
        for name, (actual, expected_value) in exact_values.items():
            if actual != expected_value:
                raise MatchedProtocolError(
                    f"{name} must equal the frozen value {expected_value}"
                )
        if self.dtype != "float32" or self.device != "cpu":
            raise MatchedProtocolError("matched cells require CPU-only float32")

    def content(self) -> dict[str, object]:
        return {
            "actor_workers": self.actor_workers,
            "cell_id": self.cell_id,
            "device": self.device,
            "dtype": self.dtype,
            "evaluation_environment_step_budget": (
                self.evaluation_environment_step_budget
            ),
            "evaluation_episodes": self.evaluation_episodes,
            "learner_engine": self.learner_engine.value,
            "learner_workers": self.learner_workers,
            "max_cpu_cores": self.max_cpu_cores,
            "rollout_plane": self.rollout_plane.value,
            "threads_per_process": self.threads_per_process,
            "training_environment_steps": self.training_environment_steps,
            "warmup_environment_steps": self.warmup_environment_steps,
        }


@dataclass(frozen=True, slots=True)
class RegistrationRequirement:
    registration_id: str
    stage: RegistrationStage

    def __post_init__(self) -> None:
        _non_empty("registration_id", self.registration_id)

    def content(self) -> dict[str, object]:
        return {"registration_id": self.registration_id, "stage": self.stage.value}


@dataclass(frozen=True, slots=True)
class MatchedLearningManifest:
    experiment_id: str
    sources: tuple[SourceLock, ...]
    scenarios: ScenarioPartition
    cells: tuple[CellSpec, ...]
    tuning_master_seeds: tuple[int, ...]
    measured_master_seeds: tuple[int, ...]
    registrations: tuple[RegistrationRequirement, ...]
    manifest_kind: str = "protocol"
    dexsim_contender_declared: bool = False
    tuning_candidates_per_engine: int = TUNING_CANDIDATE_COUNT
    tuning_steps_per_candidate: int = TUNING_STEPS_PER_CANDIDATE
    total_tuning_environment_steps: int = TOTAL_TUNING_STEP_BUDGET
    measured_steps_per_session: int = MEASURED_STEPS_PER_SESSION
    total_measured_training_steps: int = TOTAL_MEASURED_TRAINING_BUDGET
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS
    evaluation_episodes_per_checkpoint: int = EVALUATION_EPISODES_PER_CHECKPOINT
    primary_endpoint: str = "normalized-holdout-capture-auc"
    reported_endpoints: tuple[str, ...] = REPORTED_ENDPOINTS
    final_capture_rate_threshold: float = 0.60
    floor_separation_margin: float = 0.05
    backend_noninferiority_margin: float = -0.05
    required_completed_pairs: int = MEASURED_MASTER_COUNT
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    bootstrap_method: str = BOOTSTRAP_METHOD
    bootstrap_seed_sha256: str = ""
    confidence_bound: float = 0.975
    multiplicity_method: str = MULTIPLICITY_METHOD
    exclusion_policy: str = EXCLUSION_POLICY
    rerun_policy: str = RERUN_POLICY
    failure_policy: str = FAILURE_POLICY
    schema_version: str = MATCHED_MANIFEST_SCHEMA_VERSION
    manifest_sha256: str = field(init=False)
    protocol_registration_subject_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != MATCHED_MANIFEST_SCHEMA_VERSION:
            raise MatchedProtocolError("unsupported matched manifest schema_version")
        if self.manifest_kind != "protocol":
            raise MatchedProtocolError("protocol manifest_kind must be protocol")
        _non_empty("experiment_id", self.experiment_id)
        sources = tuple(self.sources)
        if any(not isinstance(source, SourceLock) for source in sources):
            raise MatchedProtocolError("sources may contain only SourceLock values")
        by_id = {source.component_id: source for source in sources}
        if len(by_id) != len(sources):
            raise MatchedProtocolError("source component IDs must be unique")
        required = set(REQUIRED_SOURCE_COMPONENTS)
        if self.dexsim_contender_declared:
            required.add("dexsim")
        if set(by_id) != required:
            raise MatchedProtocolError(
                f"source components differ from the closed set: {sorted(required)}"
            )
        object.__setattr__(
            self,
            "sources",
            tuple(sorted(sources, key=lambda item: item.component_id)),
        )

        if not isinstance(self.scenarios, ScenarioPartition):
            raise MatchedProtocolError("scenarios must be ScenarioPartition")
        scenario_source = by_id["scenario-family"]
        if scenario_source.sha256 != self.scenarios.family_sha256:
            raise MatchedProtocolError("scenario family source digest differs")

        cells = tuple(self.cells)
        if len(cells) != 4 or {cell.cell_id for cell in cells} != set(PRIMARY_CELL_IDS):
            raise MatchedProtocolError("manifest must contain the exact four primary cells")
        if len({(cell.rollout_plane, cell.learner_engine) for cell in cells}) != 4:
            raise MatchedProtocolError("manifest 2x2 factors must be unique")
        object.__setattr__(
            self,
            "cells",
            tuple(sorted(cells, key=lambda cell: PRIMARY_CELL_IDS.index(cell.cell_id))),
        )

        tuning = _seeds(
            "tuning_master_seeds", self.tuning_master_seeds, exact=TUNING_MASTER_COUNT
        )
        measured = _seeds(
            "measured_master_seeds",
            self.measured_master_seeds,
            exact=MEASURED_MASTER_COUNT,
        )
        if set(tuning).intersection(measured):
            raise MatchedProtocolError("tuning and measured master seeds must be disjoint")
        object.__setattr__(self, "tuning_master_seeds", tuning)
        object.__setattr__(self, "measured_master_seeds", measured)

        registrations = tuple(self.registrations)
        if len(registrations) != 2 or {
            item.stage for item in registrations
        } != {RegistrationStage.PROTOCOL, RegistrationStage.MEASURED}:
            raise MatchedProtocolError(
                "manifest requires protocol and measured registration requirements"
            )
        if len({item.registration_id for item in registrations}) != 2:
            raise MatchedProtocolError("registration IDs must be distinct")
        object.__setattr__(
            self, "registrations", tuple(sorted(registrations, key=lambda item: item.stage.value))
        )

        frozen_ints = {
            "tuning_candidates_per_engine": (self.tuning_candidates_per_engine, 8),
            "tuning_steps_per_candidate": (self.tuning_steps_per_candidate, 20_000),
            "total_tuning_environment_steps": (
                self.total_tuning_environment_steps,
                TOTAL_TUNING_STEP_BUDGET,
            ),
            "measured_steps_per_session": (self.measured_steps_per_session, 50_000),
            "total_measured_training_steps": (
                self.total_measured_training_steps,
                TOTAL_MEASURED_TRAINING_BUDGET,
            ),
            "evaluation_episodes_per_checkpoint": (
                self.evaluation_episodes_per_checkpoint,
                64,
            ),
            "required_completed_pairs": (self.required_completed_pairs, 20),
            "bootstrap_samples": (self.bootstrap_samples, 50_000),
        }
        for name, (actual, expected) in frozen_ints.items():
            if actual != expected:
                raise MatchedProtocolError(f"{name} must equal {expected}")
        if self.total_tuning_environment_steps != (
            TUNING_LEARNER_ENGINE_COUNT
            * TUNING_MASTER_COUNT
            * TUNING_CANDIDATE_COUNT
            * TUNING_STEPS_PER_CANDIDATE
        ):
            raise MatchedProtocolError("local-only tuning budget does not reconcile")
        if self.total_measured_training_steps != (
            len(self.cells) * len(measured) * self.measured_steps_per_session
        ):
            raise MatchedProtocolError("measured training budget does not reconcile")
        checkpoints = tuple(self.checkpoint_steps)
        if checkpoints != CHECKPOINT_STEPS:
            raise MatchedProtocolError("checkpoint_steps must be 0..50000 every 5000")
        object.__setattr__(self, "checkpoint_steps", checkpoints)
        if self.primary_endpoint != "normalized-holdout-capture-auc":
            raise MatchedProtocolError("primary endpoint differs from the frozen endpoint")
        if tuple(self.reported_endpoints) != REPORTED_ENDPOINTS:
            raise MatchedProtocolError("reported endpoints differ from the frozen set")
        exact_floats = {
            "final_capture_rate_threshold": (self.final_capture_rate_threshold, 0.60),
            "floor_separation_margin": (self.floor_separation_margin, 0.05),
            "backend_noninferiority_margin": (
                self.backend_noninferiority_margin,
                -0.05,
            ),
            "confidence_bound": (self.confidence_bound, 0.975),
        }
        for name, (float_actual, float_expected) in exact_floats.items():
            if _finite(name, float_actual) != float_expected:
                raise MatchedProtocolError(f"{name} must equal {float_expected}")
        if self.bootstrap_method != BOOTSTRAP_METHOD:
            raise MatchedProtocolError("bootstrap method differs")
        expected_bootstrap_seed = hashlib.sha256(
            f"{self.schema_version}:{self.experiment_id}:{BOOTSTRAP_METHOD}".encode()
        ).hexdigest()
        if self.bootstrap_seed_sha256 != expected_bootstrap_seed:
            raise MatchedProtocolError(
                "bootstrap_seed_sha256 must use the frozen deterministic derivation"
            )
        if self.multiplicity_method != MULTIPLICITY_METHOD:
            raise MatchedProtocolError("multiplicity method differs")
        if (
            self.exclusion_policy != EXCLUSION_POLICY
            or self.rerun_policy != RERUN_POLICY
            or self.failure_policy != FAILURE_POLICY
        ):
            raise MatchedProtocolError("exclusion/rerun/failure policy differs")

        manifest_sha256 = _content_sha256(self.content())
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(
            self, "protocol_registration_subject_sha256", manifest_sha256
        )

    @property
    def source_by_id(self) -> Mapping[str, SourceLock]:
        return {source.component_id: source for source in self.sources}

    @property
    def source_set_sha256(self) -> str:
        return _content_sha256([source.content() for source in self.sources])

    @property
    def measured_registration_subject_sha256(self) -> str:
        return self.manifest_sha256

    def content(self) -> dict[str, object]:
        return {
            "backend_noninferiority_margin": self.backend_noninferiority_margin,
            "bootstrap_method": self.bootstrap_method,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed_sha256": self.bootstrap_seed_sha256,
            "cells": [cell.content() for cell in self.cells],
            "checkpoint_steps": list(self.checkpoint_steps),
            "confidence_bound": self.confidence_bound,
            "dexsim_contender_declared": self.dexsim_contender_declared,
            "evaluation_episodes_per_checkpoint": self.evaluation_episodes_per_checkpoint,
            "exclusion_policy": self.exclusion_policy,
            "experiment_id": self.experiment_id,
            "failure_policy": self.failure_policy,
            "final_capture_rate_threshold": self.final_capture_rate_threshold,
            "floor_separation_margin": self.floor_separation_margin,
            "measured_master_seeds": list(self.measured_master_seeds),
            "measured_steps_per_session": self.measured_steps_per_session,
            "manifest_kind": self.manifest_kind,
            "multiplicity_method": self.multiplicity_method,
            "primary_endpoint": self.primary_endpoint,
            "registrations": [item.content() for item in self.registrations],
            "reported_endpoints": list(self.reported_endpoints),
            "required_completed_pairs": self.required_completed_pairs,
            "rerun_policy": self.rerun_policy,
            "scenario_partition": self.scenarios.content(),
            "scenario_partition_sha256": self.scenarios.sha256,
            "schema_version": self.schema_version,
            "sources": [source.content() for source in self.sources],
            "total_measured_training_steps": self.total_measured_training_steps,
            "total_tuning_environment_steps": self.total_tuning_environment_steps,
            "tuning_candidates_per_engine": self.tuning_candidates_per_engine,
            "tuning_master_seeds": list(self.tuning_master_seeds),
            "tuning_steps_per_candidate": self.tuning_steps_per_candidate,
        }


def bootstrap_seed_for(experiment_id: str) -> str:
    """Return the only bootstrap seed accepted for an experiment identity."""

    identity = _non_empty("experiment_id", experiment_id)
    return hashlib.sha256(
        f"{MATCHED_MANIFEST_SCHEMA_VERSION}:{identity}:{BOOTSTRAP_METHOD}".encode()
    ).hexdigest()


def bootstrap_resample_seed(
    measured_manifest_sha256: str,
    contrast_id: str,
    resample_index: int,
) -> int:
    """Apply the registered BCa resample seed derivation exactly."""

    manifest_digest = _sha256(
        "measured_manifest_sha256", measured_manifest_sha256
    )
    contrast = _non_empty("contrast_id", contrast_id)
    index = _non_negative_int("resample_index", resample_index)
    digest = hashlib.sha256()
    for part in (
        BOOTSTRAP_SEED_DERIVATION_ID.encode(),
        manifest_digest.encode("ascii"),
        contrast.encode(),
        str(index).encode("ascii"),
    ):
        digest.update(part)
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


@dataclass(frozen=True, slots=True)
class MatchedMeasuredManifestV1:
    """Post-tuning manifest linked to, but never mutating, its protocol parent."""

    experiment_id: str
    parent_protocol_sha256: str
    tuning_ledger_root_sha256: str
    selection_result_sha256: str
    selected_hyperparameter_sha256: str
    analysis_source_sha256: str
    measured_plan_sha256: str
    manifest_kind: str = "measured"
    schema_version: str = MATCHED_MANIFEST_SCHEMA_VERSION
    manifest_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty("measured experiment_id", self.experiment_id)
        if self.manifest_kind != "measured":
            raise MatchedProtocolError("measured manifest_kind must be measured")
        if self.schema_version != MATCHED_MANIFEST_SCHEMA_VERSION:
            raise MatchedProtocolError("unsupported measured manifest schema_version")
        for name in (
            "parent_protocol_sha256",
            "tuning_ledger_root_sha256",
            "selection_result_sha256",
            "selected_hyperparameter_sha256",
            "analysis_source_sha256",
            "measured_plan_sha256",
        ):
            _sha256(name, getattr(self, name))
        object.__setattr__(self, "manifest_sha256", _content_sha256(self.content()))

    def validate_parent(self, protocol: MatchedLearningManifest) -> None:
        if self.experiment_id != protocol.experiment_id:
            raise MatchedProtocolError("measured experiment differs from protocol")
        if self.parent_protocol_sha256 != protocol.manifest_sha256:
            raise MatchedProtocolError("measured parent protocol digest differs")
        if self.analysis_source_sha256 != protocol.source_by_id["analysis"].sha256:
            raise MatchedProtocolError("measured analysis source differs from protocol")

    def content(self) -> dict[str, object]:
        return {
            "analysis_source_sha256": self.analysis_source_sha256,
            "experiment_id": self.experiment_id,
            "manifest_kind": self.manifest_kind,
            "measured_plan_sha256": self.measured_plan_sha256,
            "parent_protocol_sha256": self.parent_protocol_sha256,
            "schema_version": self.schema_version,
            "selected_hyperparameter_sha256": (
                self.selected_hyperparameter_sha256
            ),
            "selection_result_sha256": self.selection_result_sha256,
            "tuning_ledger_root_sha256": self.tuning_ledger_root_sha256,
        }


MatchedProtocolManifestV1 = MatchedLearningManifest


@dataclass(frozen=True, slots=True)
class TuningCompletionReceipt:
    """Artifact-backed reconciliation of the closed local-only tuning phase."""

    session_count: int
    training_environment_steps: int
    evaluation_episode_count: int
    tuning_plan: ArtifactReference
    session_receipts: ArtifactReference
    episode_receipts: ArtifactReference
    tuning_ledger: ArtifactReference
    selection_result: ArtifactReference
    selected_hyperparameters: ArtifactReference
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.session_count != 80:
            raise MatchedProtocolError("tuning requires exactly 80 sessions")
        if self.training_environment_steps != TOTAL_TUNING_STEP_BUDGET:
            raise MatchedProtocolError("tuning steps do not reconcile to 1,600,000")
        if self.evaluation_episode_count != 2_560:
            raise MatchedProtocolError("tuning evaluations do not reconcile to 2,560")
        for name in (
            "tuning_plan",
            "session_receipts",
            "episode_receipts",
            "tuning_ledger",
            "selection_result",
            "selected_hyperparameters",
        ):
            if not isinstance(getattr(self, name), ArtifactReference):
                raise MatchedProtocolError(f"{name} must be an ArtifactReference")
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def validate_measured_manifest(
        self, measured: MatchedMeasuredManifestV1
    ) -> None:
        if self.tuning_ledger.sha256 != measured.tuning_ledger_root_sha256:
            raise MatchedProtocolError("tuning ledger root differs from measured manifest")
        if self.selection_result.sha256 != measured.selection_result_sha256:
            raise MatchedProtocolError("selection result differs from measured manifest")
        if (
            self.selected_hyperparameters.sha256
            != measured.selected_hyperparameter_sha256
        ):
            raise MatchedProtocolError(
                "selected hyperparameters differ from measured manifest"
            )

    def content(self) -> dict[str, object]:
        return {
            "episode_receipts": self.episode_receipts.content(),
            "evaluation_episode_count": self.evaluation_episode_count,
            "selected_hyperparameters": self.selected_hyperparameters.content(),
            "selection_result": self.selection_result.content(),
            "session_count": self.session_count,
            "session_receipts": self.session_receipts.content(),
            "training_environment_steps": self.training_environment_steps,
            "tuning_ledger": self.tuning_ledger.content(),
            "tuning_plan": self.tuning_plan.content(),
        }

    @property
    def artifacts(self) -> tuple[ArtifactReference, ...]:
        return (
            self.tuning_plan,
            self.session_receipts,
            self.episode_receipts,
            self.tuning_ledger,
            self.selection_result,
            self.selected_hyperparameters,
        )


@dataclass(frozen=True, slots=True)
class MeasuredSessionPlan:
    session_id: str
    cell_id: str
    master_seed: int
    training_configs: tuple[str, ...]
    evaluation_configs: tuple[str, ...]
    attempt: int = 0
    status: RunStatus = RunStatus.PLANNED
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty("session_id", self.session_id)
        if self.cell_id not in PRIMARY_CELL_IDS:
            raise MatchedProtocolError("session cell_id is not a primary cell")
        _non_negative_int("session master_seed", self.master_seed)
        if self.attempt != 0:
            raise MatchedProtocolError("initial measured plan attempt must be zero")
        if self.status is not RunStatus.PLANNED:
            raise MatchedProtocolError("measured session plan status must be planned")
        training = tuple(self.training_configs)
        evaluation = tuple(self.evaluation_configs)
        if len(training) != 8 or len(set(training)) != 8:
            raise MatchedProtocolError("each measured session requires 8 training configs")
        if len(evaluation) != 64 or len(set(evaluation)) != 64:
            raise MatchedProtocolError("each measured session requires 64 evaluation configs")
        for index, digest in enumerate((*training, *evaluation)):
            _sha256(f"session config digest[{index}]", digest)
        if set(training).intersection(evaluation):
            raise MatchedProtocolError("session training/evaluation configs must be disjoint")
        object.__setattr__(self, "training_configs", training)
        object.__setattr__(self, "evaluation_configs", evaluation)
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "cell_id": self.cell_id,
            "evaluation_configs": list(self.evaluation_configs),
            "master_seed": self.master_seed,
            "session_id": self.session_id,
            "status": self.status.value,
            "training_configs": list(self.training_configs),
        }


def build_measured_plan(
    manifest: MatchedLearningManifest,
) -> tuple[MeasuredSessionPlan, ...]:
    """Build the exact 80-session counterbalanced measured plan."""

    measured = manifest.scenarios.measured_training
    evaluation = tuple(case.config_sha256 for case in manifest.scenarios.evaluation)
    plans: list[MeasuredSessionPlan] = []
    for master_index, master_seed in enumerate(manifest.measured_master_seeds):
        start = master_index * 8
        training = tuple(case.config_sha256 for case in measured[start : start + 8])
        for cell_index in range(len(PRIMARY_CELL_IDS)):
            cell_id = PRIMARY_CELL_IDS[(cell_index + master_index) % len(PRIMARY_CELL_IDS)]
            plans.append(
                MeasuredSessionPlan(
                    session_id=f"{manifest.experiment_id}:m{master_seed}:{cell_id}",
                    cell_id=cell_id,
                    master_seed=master_seed,
                    training_configs=training,
                    evaluation_configs=evaluation,
                )
            )
    result = tuple(plans)
    validate_measured_plan(manifest, result)
    return result


def measured_plan_sha256(plan: Sequence[MeasuredSessionPlan]) -> str:
    return _content_sha256([item.content() for item in plan])


def validate_measured_plan(
    manifest: MatchedLearningManifest,
    plan: Sequence[MeasuredSessionPlan],
) -> None:
    items = tuple(plan)
    if len(items) != MEASURED_SESSION_COUNT:
        raise MatchedProtocolError("measured plan must contain exactly 80 sessions")
    if len({item.session_id for item in items}) != len(items):
        raise MatchedProtocolError("measured session IDs must be unique")
    expected_pairs = {
        (cell_id, master_seed)
        for cell_id in PRIMARY_CELL_IDS
        for master_seed in manifest.measured_master_seeds
    }
    if {(item.cell_id, item.master_seed) for item in items} != expected_pairs:
        raise MatchedProtocolError("measured plan does not cover the exact 4x20 pairs")
    measured_digests = {
        case.config_sha256 for case in manifest.scenarios.measured_training
    }
    evaluation_digests = {
        case.config_sha256 for case in manifest.scenarios.evaluation
    }
    for master_seed in manifest.measured_master_seeds:
        cohort = [item for item in items if item.master_seed == master_seed]
        if len({item.training_configs for item in cohort}) != 1:
            raise MatchedProtocolError("training config cohort differs across cells")
        if len({item.evaluation_configs for item in cohort}) != 1:
            raise MatchedProtocolError("evaluation config cohort differs across cells")
    for cell_id in PRIMARY_CELL_IDS:
        used = {
            digest
            for item in items
            if item.cell_id == cell_id
            for digest in item.training_configs
        }
        if used != measured_digests:
            raise MatchedProtocolError(
                "each cell must use every measured config exactly once"
            )
    if any(set(item.evaluation_configs) != evaluation_digests for item in items):
        raise MatchedProtocolError("every session must use the same holdout configs")


@dataclass(frozen=True, slots=True)
class CapabilityReceipt:
    capability_id: str
    kind: CapabilityKind
    source: SourceLock | None
    probe_implementation_sha256: str
    probe_artifact: ArtifactReference | None = None
    unavailable_reason: str | None = None
    emulated: bool = False
    algorithm_class: str | None = None
    learner_class: str | None = None
    env_runner_class: str | None = None
    algorithm_instantiated: bool = False
    learner_instantiated: bool = False
    env_runner_instantiated: bool = False
    update_count: int = 0
    initial_checkpoint_sha256: str | None = None
    final_checkpoint_sha256: str | None = None
    initial_policy_parameter_sha256: str | None = None
    final_policy_parameter_sha256: str | None = None
    checkpoint_reloaded: bool = False
    observed_cell_ids: tuple[str, ...] = ()
    config_contract_sha256: str | None = None
    observation_contract_sha256: str | None = None
    action_mask_contract_sha256: str | None = None
    ray_task_only: bool = False
    executable_path: str | None = None
    executable_adapter: bool = False
    workload_completed: bool = False
    semantic_parity_passed: bool = False
    semantic_projection_contract_sha256: str | None = None
    disposition: CapabilityDisposition = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty("capability_id", self.capability_id)
        _sha256("probe_implementation_sha256", self.probe_implementation_sha256)
        if self.probe_artifact is not None and (
            self.probe_artifact.sha256 != self.probe_implementation_sha256
        ):
            raise MatchedProtocolError("capability probe artifact digest differs")
        _non_negative_int("update_count", self.update_count)
        if self.source is not None and self.source.component_id != self.kind.value:
            raise MatchedProtocolError("capability source component differs from kind")
        if self.unavailable_reason is not None:
            _non_empty("unavailable_reason", self.unavailable_reason)
        if self.emulated:
            disposition = CapabilityDisposition.EMULATED
        elif self.kind is CapabilityKind.RLLIB:
            native = self._rllib_native()
            if native:
                disposition = CapabilityDisposition.NATIVE_ADMITTED
            elif self.unavailable_reason is not None:
                disposition = CapabilityDisposition.SKIPPED_NATIVE_UNAVAILABLE
            else:
                disposition = CapabilityDisposition.REJECTED
        else:
            native = self._dexsim_native()
            if native:
                disposition = CapabilityDisposition.NATIVE_ADMITTED
            elif self.unavailable_reason is not None:
                disposition = CapabilityDisposition.SKIPPED_NATIVE_UNAVAILABLE
            else:
                disposition = CapabilityDisposition.REJECTED
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def _rllib_native(self) -> bool:
        checkpoints = (
            self.initial_checkpoint_sha256,
            self.final_checkpoint_sha256,
        )
        policy_parameters = (
            self.initial_policy_parameter_sha256,
            self.final_policy_parameter_sha256,
        )
        contract_digests = (
            self.config_contract_sha256,
            self.observation_contract_sha256,
            self.action_mask_contract_sha256,
        )
        for group_name, values in (
            ("checkpoint", checkpoints),
            ("policy parameter", policy_parameters),
            ("contract", contract_digests),
        ):
            if any(value is not None for value in values):
                for index, value in enumerate(values):
                    _sha256(f"RLlib {group_name} digest[{index}]", value)
        cells = _strings("observed_cell_ids", self.observed_cell_ids)
        object.__setattr__(self, "observed_cell_ids", tuple(sorted(cells)))
        projection = self.semantic_projection_contract_sha256
        if projection is not None:
            _sha256("semantic_projection_contract_sha256", projection)
        return bool(
            self.source is not None
            and self.probe_artifact is not None
            and self.algorithm_class
            and self.learner_class
            and self.env_runner_class
            and self.algorithm_instantiated
            and self.learner_instantiated
            and self.env_runner_instantiated
            and self.update_count > 0
            and checkpoints[0]
            and checkpoints[1]
            and checkpoints[0] != checkpoints[1]
            and policy_parameters[0]
            and policy_parameters[1]
            and policy_parameters[0] != policy_parameters[1]
            and self.checkpoint_reloaded
            and set(cells) == {"local-rllib", "joined-gorti-rllib"}
            and all(contract_digests)
            and not self.ray_task_only
            and projection
        )

    def _dexsim_native(self) -> bool:
        projection = self.semantic_projection_contract_sha256
        if projection is not None:
            _sha256("semantic_projection_contract_sha256", projection)
        return bool(
            self.source is not None
            and self.probe_artifact is not None
            and self.executable_path
            and self.executable_adapter
            and self.workload_completed
            and self.semantic_parity_passed
            and projection
        )

    @property
    def sample_eligible(self) -> bool:
        return self.disposition is CapabilityDisposition.NATIVE_ADMITTED

    def content(self) -> dict[str, object]:
        return {
            "algorithm_class": self.algorithm_class,
            "algorithm_instantiated": self.algorithm_instantiated,
            "capability_id": self.capability_id,
            "disposition": self.disposition.value,
            "emulated": self.emulated,
            "env_runner_class": self.env_runner_class,
            "env_runner_instantiated": self.env_runner_instantiated,
            "executable_adapter": self.executable_adapter,
            "executable_path": self.executable_path,
            "final_checkpoint_sha256": self.final_checkpoint_sha256,
            "final_policy_parameter_sha256": self.final_policy_parameter_sha256,
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "initial_policy_parameter_sha256": self.initial_policy_parameter_sha256,
            "checkpoint_reloaded": self.checkpoint_reloaded,
            "observed_cell_ids": list(self.observed_cell_ids),
            "config_contract_sha256": self.config_contract_sha256,
            "observation_contract_sha256": self.observation_contract_sha256,
            "action_mask_contract_sha256": self.action_mask_contract_sha256,
            "ray_task_only": self.ray_task_only,
            "kind": self.kind.value,
            "learner_class": self.learner_class,
            "learner_instantiated": self.learner_instantiated,
            "probe_implementation_sha256": self.probe_implementation_sha256,
            "probe_artifact": (
                None if self.probe_artifact is None else self.probe_artifact.content()
            ),
            "semantic_parity_passed": self.semantic_parity_passed,
            "semantic_projection_contract_sha256": (
                self.semantic_projection_contract_sha256
            ),
            "source": None if self.source is None else self.source.content(),
            "unavailable_reason": self.unavailable_reason,
            "update_count": self.update_count,
            "workload_completed": self.workload_completed,
        }


def validate_native_capability(
    manifest: MatchedLearningManifest, receipt: CapabilityReceipt
) -> None:
    """Require computed native admission and the exact manifest source locks."""

    if not receipt.sample_eligible or receipt.source is None:
        raise MatchedProtocolError(
            f"{receipt.kind.value} capability is not a native sample"
        )
    if receipt.kind is CapabilityKind.DEXSIM and not manifest.dexsim_contender_declared:
        raise MatchedProtocolError("DEXSim was not declared in the manifest")
    expected = manifest.source_by_id.get(receipt.kind.value)
    if expected is None or expected != receipt.source:
        raise MatchedProtocolError("capability source differs from manifest source lock")
    projection = manifest.source_by_id["semantic-projection"].sha256
    if receipt.semantic_projection_contract_sha256 != projection:
        raise MatchedProtocolError("capability semantic projection differs")
    if receipt.kind is CapabilityKind.RLLIB:
        if set(receipt.observed_cell_ids) != {
            "local-rllib",
            "joined-gorti-rllib",
        }:
            raise MatchedProtocolError("RLlib capability did not observe both cells")
        if receipt.config_contract_sha256 != manifest.scenarios.family_sha256:
            raise MatchedProtocolError("RLlib config contract differs")
        feature_contract = manifest.source_by_id["feature-contract"].sha256
        if receipt.observation_contract_sha256 != feature_contract:
            raise MatchedProtocolError("RLlib observation contract differs")
        if receipt.action_mask_contract_sha256 != feature_contract:
            raise MatchedProtocolError("RLlib action-mask contract differs")


JOINED_EVIDENCE_PHASES: Final = frozenset(
    {
        "membership",
        "declaration",
        "time-enable",
        "synchronization",
        "assignment-policy",
        "raw-delivery",
        "per-episode-parity",
        "cleanup",
    }
)


@dataclass(frozen=True, slots=True)
class JoinedPhaseArtifact:
    phase: str
    artifact: ArtifactReference

    def __post_init__(self) -> None:
        if self.phase not in JOINED_EVIDENCE_PHASES:
            raise MatchedProtocolError("joined phase artifact has an unknown phase")
        if not isinstance(self.artifact, ArtifactReference):
            raise MatchedProtocolError("joined phase evidence must be an artifact")

    def content(self) -> dict[str, object]:
        return {"artifact": self.artifact.content(), "phase": self.phase}


@dataclass(frozen=True, slots=True)
class JoinedCapabilityReceipt:
    """Computed admission predicate for one actual joined-gorti primary cell."""

    capability_id: str
    cell_id: str
    execution_mode: str
    diagnostic_emulation: bool
    manifest_sha256: str
    plan_sha256: str
    run_id: str
    attempt: int
    generation: int
    rti_executable_sha256: str
    rti_process_id: int
    runtime_process_ids: tuple[int, ...]
    phase_artifacts: tuple[JoinedPhaseArtifact, ...]
    gorti_source_sha256: str
    runtime_source_sha256: str
    fom_sha256: str
    federation_joined: bool
    participant_count: int
    required_roles: tuple[str, ...]
    declarations_complete: bool
    time_regulating: bool
    time_constrained: bool
    sync_registered: bool
    sync_announced_all: bool
    sync_achieved_all: bool
    sync_synchronized: bool
    assignment_acknowledged: bool
    policy_activation_acknowledged: bool
    time_request_count: int
    time_grant_count: int
    exact_time_grants: bool
    expected_transition_count: int
    raw_transition_count: int
    semantic_transition_count: int
    unique_transition_count: int
    duplicate_count: int
    reject_count: int
    conflict_count: int
    per_episode_local_parity: bool
    aggregate_local_parity: bool
    terminal_receipts_complete: bool
    budget_respected: bool
    resigned: bool
    cleanup_complete: bool
    ledger_bound: bool
    unavailable_reason: str | None = None
    disposition: CapabilityDisposition = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty("joined capability_id", self.capability_id)
        if self.cell_id not in {
            "joined-gorti-reference",
            "joined-gorti-rllib",
        }:
            raise MatchedProtocolError("joined capability requires a joined primary cell")
        if self.execution_mode != "actual-federation":
            raise MatchedProtocolError("joined execution_mode must be actual-federation")
        if type(self.diagnostic_emulation) is not bool or self.diagnostic_emulation:
            raise MatchedProtocolError("diagnostic/emulated joined evidence is excluded")
        for name in (
            "manifest_sha256",
            "plan_sha256",
            "rti_executable_sha256",
            "gorti_source_sha256",
            "runtime_source_sha256",
            "fom_sha256",
        ):
            _sha256(name, getattr(self, name))
        _non_empty("joined run_id", self.run_id)
        _non_negative_int("joined attempt", self.attempt)
        _non_negative_int("joined generation", self.generation)
        _positive_int("joined rti_process_id", self.rti_process_id)
        process_ids = tuple(
            _positive_int(f"runtime_process_ids[{index}]", process_id)
            for index, process_id in enumerate(self.runtime_process_ids)
        )
        if len(process_ids) < 3 or len(set(process_ids)) != len(process_ids):
            raise MatchedProtocolError("joined runtime process identities are incomplete")
        object.__setattr__(self, "runtime_process_ids", process_ids)
        phases = tuple(self.phase_artifacts)
        if any(not isinstance(item, JoinedPhaseArtifact) for item in phases):
            raise MatchedProtocolError("joined phase evidence has an invalid type")
        if {item.phase for item in phases} != JOINED_EVIDENCE_PHASES or len(
            phases
        ) != len(JOINED_EVIDENCE_PHASES):
            raise MatchedProtocolError("joined phase evidence is incomplete or duplicated")
        object.__setattr__(self, "phase_artifacts", tuple(sorted(phases, key=lambda x: x.phase)))
        for name in (
            "participant_count",
            "time_request_count",
            "time_grant_count",
            "expected_transition_count",
            "raw_transition_count",
            "semantic_transition_count",
            "unique_transition_count",
            "duplicate_count",
            "reject_count",
            "conflict_count",
        ):
            _non_negative_int(name, getattr(self, name))
        roles = _strings("joined required_roles", self.required_roles)
        object.__setattr__(self, "required_roles", roles)
        if self.unavailable_reason is not None:
            _non_empty("joined unavailable_reason", self.unavailable_reason)
        admitted = self._computed_admission()
        disposition = (
            CapabilityDisposition.NATIVE_ADMITTED
            if admitted
            else (
                CapabilityDisposition.SKIPPED_NATIVE_UNAVAILABLE
                if self.unavailable_reason is not None
                else CapabilityDisposition.REJECTED
            )
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def _computed_admission(self) -> bool:
        counts_match = (
            self.expected_transition_count > 0
            and self.raw_transition_count == self.expected_transition_count
            and self.semantic_transition_count == self.expected_transition_count
            and self.unique_transition_count == self.expected_transition_count
        )
        return bool(
            self.federation_joined
            and self.participant_count >= 3
            and {"coordinator", "worker", "learner"}.issubset(self.required_roles)
            and self.declarations_complete
            and self.time_regulating
            and self.time_constrained
            and self.sync_registered
            and self.sync_announced_all
            and self.sync_achieved_all
            and self.sync_synchronized
            and self.assignment_acknowledged
            and self.policy_activation_acknowledged
            and self.time_request_count > 0
            and self.time_grant_count == self.time_request_count
            and self.exact_time_grants
            and counts_match
            and self.duplicate_count == 0
            and self.reject_count == 0
            and self.conflict_count == 0
            and self.per_episode_local_parity
            and self.aggregate_local_parity
            and self.terminal_receipts_complete
            and self.budget_respected
            and self.resigned
            and self.cleanup_complete
            and self.ledger_bound
            and self.execution_mode == "actual-federation"
            and not self.diagnostic_emulation
        )

    @property
    def sample_eligible(self) -> bool:
        return self.disposition is CapabilityDisposition.NATIVE_ADMITTED

    def content(self) -> dict[str, object]:
        names = (
            "aggregate_local_parity",
            "assignment_acknowledged",
            "budget_respected",
            "capability_id",
            "cell_id",
            "cleanup_complete",
            "conflict_count",
            "declarations_complete",
            "duplicate_count",
            "exact_time_grants",
            "expected_transition_count",
            "federation_joined",
            "fom_sha256",
            "gorti_source_sha256",
            "ledger_bound",
            "participant_count",
            "per_episode_local_parity",
            "policy_activation_acknowledged",
            "raw_transition_count",
            "reject_count",
            "resigned",
            "runtime_source_sha256",
            "semantic_transition_count",
            "sync_achieved_all",
            "sync_announced_all",
            "sync_registered",
            "sync_synchronized",
            "terminal_receipts_complete",
            "time_constrained",
            "time_grant_count",
            "time_regulating",
            "time_request_count",
            "unavailable_reason",
            "unique_transition_count",
        )
        result = {name: getattr(self, name) for name in names}
        result.update(
            {
                "attempt": self.attempt,
                "diagnostic_emulation": self.diagnostic_emulation,
                "execution_mode": self.execution_mode,
                "generation": self.generation,
                "manifest_sha256": self.manifest_sha256,
                "phase_artifacts": [item.content() for item in self.phase_artifacts],
                "plan_sha256": self.plan_sha256,
                "rti_executable_sha256": self.rti_executable_sha256,
                "rti_process_id": self.rti_process_id,
                "run_id": self.run_id,
                "runtime_process_ids": list(self.runtime_process_ids),
            }
        )
        result["disposition"] = self.disposition.value
        result["required_roles"] = list(self.required_roles)
        return result


def validate_joined_capability(
    manifest: MatchedLearningManifest,
    receipt: JoinedCapabilityReceipt,
) -> None:
    if not receipt.sample_eligible:
        raise MatchedProtocolError("joined-gorti capability is not admitted")
    if receipt.gorti_source_sha256 != manifest.source_by_id["gorti"].sha256:
        raise MatchedProtocolError("joined gorti source differs from manifest")
    if receipt.runtime_source_sha256 != manifest.source_set_sha256:
        raise MatchedProtocolError("joined runtime source set differs from manifest")
    if receipt.fom_sha256 != manifest.source_by_id["fom"].sha256:
        raise MatchedProtocolError("joined FOM source differs from manifest")
    if receipt.manifest_sha256 != manifest.manifest_sha256:
        raise MatchedProtocolError("joined manifest digest differs")
    expected_plan_sha256 = measured_plan_sha256(build_measured_plan(manifest))
    if receipt.plan_sha256 != expected_plan_sha256:
        raise MatchedProtocolError("joined plan digest differs")


def cell_capability_sha256(
    manifest: MatchedLearningManifest,
    cell_id: str,
    *,
    rllib: CapabilityReceipt | None = None,
    joined: JoinedCapabilityReceipt | None = None,
) -> str:
    """Derive the only runtime-capability identity accepted by a cell."""

    if cell_id not in PRIMARY_CELL_IDS:
        raise MatchedProtocolError("runtime capability cell is not primary")
    needs_rllib = cell_id.endswith("rllib")
    needs_joined = cell_id.startswith("joined-")
    if needs_rllib and rllib is None:
        raise MatchedProtocolError("RLlib cell requires its capability receipt")
    if not needs_rllib and rllib is not None:
        raise MatchedProtocolError("reference cell cannot bind an RLlib capability")
    if needs_joined and joined is None:
        raise MatchedProtocolError("joined cell requires its capability receipt")
    if not needs_joined and joined is not None:
        raise MatchedProtocolError("local cell cannot bind a joined capability")
    if rllib is not None and rllib.kind is not CapabilityKind.RLLIB:
        raise MatchedProtocolError("runtime capability is not RLlib")
    if joined is not None and joined.cell_id != cell_id:
        raise MatchedProtocolError("joined runtime capability cell differs")
    return _content_sha256(
        {
            "cell_id": cell_id,
            "joined_capability_sha256": None if joined is None else joined.sha256,
            "protocol_source_set_sha256": manifest.source_set_sha256,
            "rllib_capability_sha256": None if rllib is None else rllib.sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class ExternalRegistrationReceipt:
    registration_id: str
    stage: RegistrationStage
    authority: str
    uri: str
    registered_at: str
    media_type: str
    size_bytes: int
    artifact_sha256: str
    receipt_sha256: str
    resolved_and_rehashed: bool
    registered_object: ArtifactReference | None = None
    authority_receipt: ArtifactReference | None = None
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty("registration_id", self.registration_id)
        _non_empty("registration authority", self.authority)
        _non_empty("registration URI", self.uri)
        _parse_rfc3339("registered_at", self.registered_at)
        _non_empty("registration media_type", self.media_type)
        _positive_int("registration size_bytes", self.size_bytes)
        _sha256("registered artifact_sha256", self.artifact_sha256)
        _sha256("registration receipt_sha256", self.receipt_sha256)
        if type(self.resolved_and_rehashed) is not bool:
            raise MatchedProtocolError("resolved_and_rehashed must be bool")
        if (
            self.registered_object is not None
            and self.registered_object.sha256 != self.artifact_sha256
        ):
            raise MatchedProtocolError("registered object artifact digest differs")
        if (
            self.authority_receipt is not None
            and self.authority_receipt.sha256 != self.receipt_sha256
        ):
            raise MatchedProtocolError("authority receipt artifact digest differs")
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "authority": self.authority,
            "media_type": self.media_type,
            "receipt_sha256": self.receipt_sha256,
            "registered_at": self.registered_at,
            "registration_id": self.registration_id,
            "resolved_and_rehashed": self.resolved_and_rehashed,
            "registered_object": (
                None if self.registered_object is None else self.registered_object.content()
            ),
            "authority_receipt": (
                None if self.authority_receipt is None else self.authority_receipt.content()
            ),
            "size_bytes": self.size_bytes,
            "stage": self.stage.value,
            "uri": self.uri,
        }


@dataclass(frozen=True, slots=True)
class ExecutionTimeline:
    tuning_started_at: str
    tuning_completed_at: str
    measured_started_at: str

    def __post_init__(self) -> None:
        tuning_start = _parse_rfc3339("tuning_started_at", self.tuning_started_at)
        tuning_end = _parse_rfc3339("tuning_completed_at", self.tuning_completed_at)
        measured_start = _parse_rfc3339("measured_started_at", self.measured_started_at)
        if not tuning_start < tuning_end < measured_start:
            raise MatchedProtocolError(
                "execution timeline must order tuning start, tuning end, measured start"
            )


def validate_external_registrations(
    manifest: MatchedLearningManifest,
    measured_manifest: MatchedMeasuredManifestV1,
    registrations: Sequence[ExternalRegistrationReceipt],
    timeline: ExecutionTimeline,
) -> None:
    """Gate measured execution on ordered, externally identified registrations."""

    measured_manifest.validate_parent(manifest)
    if measured_manifest.measured_plan_sha256 == "0" * 64:
        raise MatchedProtocolError("measured manifest plan digest is not resolved")
    by_stage = {receipt.stage: receipt for receipt in registrations}
    if len(by_stage) != 2:
        raise MatchedProtocolError(
            "exactly one protocol and one measured registration are required"
        )
    requirements = {item.stage: item for item in manifest.registrations}
    for stage in RegistrationStage:
        receipt = by_stage.get(stage)
        requirement = requirements[stage]
        if receipt is None or receipt.registration_id != requirement.registration_id:
            raise MatchedProtocolError(f"{stage.value} registration is missing")
    protocol = by_stage[RegistrationStage.PROTOCOL]
    measured = by_stage[RegistrationStage.MEASURED]
    if protocol.artifact_sha256 != manifest.protocol_registration_subject_sha256:
        raise MatchedProtocolError("protocol registered artifact digest differs")
    if measured.artifact_sha256 != measured_manifest.manifest_sha256:
        raise MatchedProtocolError("measured registered artifact digest differs")
    if not protocol.resolved_and_rehashed or not measured.resolved_and_rehashed:
        raise MatchedProtocolError("external registrations must resolve and re-hash")
    if any(
        receipt.registered_object is None or receipt.authority_receipt is None
        for receipt in (protocol, measured)
    ):
        raise MatchedProtocolError(
            "external registration objects and authority receipts are required"
        )
    protocol_time = _parse_rfc3339("protocol registered_at", protocol.registered_at)
    measured_time = _parse_rfc3339("measured registered_at", measured.registered_at)
    tuning_start = _parse_rfc3339("tuning_started_at", timeline.tuning_started_at)
    tuning_end = _parse_rfc3339("tuning_completed_at", timeline.tuning_completed_at)
    measured_start = _parse_rfc3339("measured_started_at", timeline.measured_started_at)
    if not protocol_time < tuning_start:
        raise MatchedProtocolError("protocol registration must precede tuning")
    if not tuning_end < measured_time < measured_start:
        raise MatchedProtocolError(
            "measured registration must follow tuning and precede measured execution"
        )


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: str
    media_type: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_relative_path("artifact path", self.path)
        _non_empty("artifact media_type", self.media_type)
        _positive_int("artifact size_bytes", self.size_bytes)
        _sha256("artifact sha256", self.sha256)

    def content(self) -> dict[str, object]:
        return {
            "media_type": self.media_type,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ResourceCounters:
    wall_time_seconds: float
    cpu_time_seconds: float
    peak_rss_bytes: int
    valid_transition_count: int
    learner_update_count: int
    network_bytes: int
    artifact_bytes: int
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_p99_seconds: float
    max_policy_lag: int
    max_queue_depth: int

    def __post_init__(self) -> None:
        for name in (
            "wall_time_seconds",
            "cpu_time_seconds",
            "latency_p50_seconds",
            "latency_p95_seconds",
            "latency_p99_seconds",
        ):
            if _finite(name, getattr(self, name)) < 0:
                raise MatchedProtocolError(f"{name} must be non-negative")
        for name in (
            "peak_rss_bytes",
            "valid_transition_count",
            "learner_update_count",
            "network_bytes",
            "artifact_bytes",
            "max_policy_lag",
            "max_queue_depth",
        ):
            _non_negative_int(name, getattr(self, name))
        if not (
            self.latency_p50_seconds
            <= self.latency_p95_seconds
            <= self.latency_p99_seconds
        ):
            raise MatchedProtocolError("latency quantiles must be monotonic")

    def content(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "artifact_bytes",
                "cpu_time_seconds",
                "latency_p50_seconds",
                "latency_p95_seconds",
                "latency_p99_seconds",
                "learner_update_count",
                "max_policy_lag",
                "max_queue_depth",
                "network_bytes",
                "peak_rss_bytes",
                "valid_transition_count",
                "wall_time_seconds",
            )
        }


@dataclass(frozen=True, slots=True)
class SessionReceipt:
    receipt_id: str
    session_id: str
    cell_id: str
    attempt: int
    master_seed: int
    status: RunStatus
    training_environment_steps: int
    training_episode_receipt_count: int
    checkpoint_receipt_count: int
    evaluation_episode_receipt_count: int
    policy_version: int | None
    logical_time_start: float | None
    logical_time_end: float | None
    final_return: float | None
    outcome: str | None
    transition_sha256: str | None
    action_sha256: str | None
    semantic_sha256: str | None
    source_set_sha256: str | None
    runtime_capability_sha256: str | None
    federation_id: str | None
    generation: int | None
    worker_ids: tuple[str, ...]
    host_ids: tuple[str, ...]
    duplicate_count: int
    reject_count: int
    failure_count: int
    mask_violation_count: int
    evaluation_leak_count: int
    missing_checkpoint_count: int
    semantic_parity_passed: bool
    budget_respected: bool
    cleanup_complete: bool
    failure_phase: str | None
    failure_reason: str | None
    exclusion_reason: str | None
    rerun_of_receipt_id: str | None
    first_step_committed: bool
    resources: ResourceCounters
    artifacts: tuple[ArtifactReference, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty("receipt_id", self.receipt_id)
        _non_empty("session_id", self.session_id)
        if self.cell_id not in PRIMARY_CELL_IDS:
            raise MatchedProtocolError("terminal receipt cell is not primary")
        _non_negative_int("receipt attempt", self.attempt)
        _non_negative_int("receipt master_seed", self.master_seed)
        if self.status is RunStatus.PLANNED:
            raise MatchedProtocolError("terminal receipt cannot remain planned")
        for name in (
            "training_environment_steps",
            "training_episode_receipt_count",
            "checkpoint_receipt_count",
            "evaluation_episode_receipt_count",
            "duplicate_count",
            "reject_count",
            "failure_count",
            "mask_violation_count",
            "evaluation_leak_count",
            "missing_checkpoint_count",
        ):
            _non_negative_int(name, getattr(self, name))
        if self.policy_version is not None:
            _non_negative_int("policy_version", self.policy_version)
        if self.generation is not None:
            _non_negative_int("generation", self.generation)
        for name in ("logical_time_start", "logical_time_end", "final_return"):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value)
        if (
            self.logical_time_start is not None
            and self.logical_time_end is not None
            and self.logical_time_end < self.logical_time_start
        ):
            raise MatchedProtocolError("terminal logical time regressed")
        for name in (
            "transition_sha256",
            "action_sha256",
            "semantic_sha256",
            "source_set_sha256",
            "runtime_capability_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _sha256(name, value)
        workers = _strings("worker_ids", self.worker_ids)
        hosts = _strings("host_ids", self.host_ids)
        object.__setattr__(self, "worker_ids", workers)
        object.__setattr__(self, "host_ids", hosts)
        if self.cell_id.startswith("local-"):
            if self.federation_id is not None or self.generation is not None:
                raise MatchedProtocolError(
                    "local receipts must keep federation fields null"
                )
        elif self.federation_id is None or self.generation is None:
            raise MatchedProtocolError(
                "joined receipts require federation_id and generation"
            )
        for name in (
            "failure_phase",
            "failure_reason",
            "exclusion_reason",
            "rerun_of_receipt_id",
        ):
            value = getattr(self, name)
            if value is not None:
                _non_empty(name, value)
        if self.attempt > 0:
            if self.rerun_of_receipt_id is None or self.first_step_committed:
                raise MatchedProtocolError(
                    "rerun is allowed once only before the first committed step"
                )
            if self.attempt != 1:
                raise MatchedProtocolError("only one infrastructure rerun is permitted")
        elif self.rerun_of_receipt_id is not None:
            raise MatchedProtocolError("attempt zero cannot reference a rerun receipt")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactReference) for item in artifacts):
            raise MatchedProtocolError(
                "terminal artifacts may contain only ArtifactReference values"
            )
        if len({item.path for item in artifacts}) != len(artifacts):
            raise MatchedProtocolError("terminal artifact paths must be unique")
        object.__setattr__(self, "artifacts", artifacts)
        if not isinstance(self.resources, ResourceCounters):
            raise MatchedProtocolError("resources must be ResourceCounters")
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    @property
    def sample_eligible(self) -> bool:
        return bool(
            self.status is RunStatus.COMPLETED
            and self.training_environment_steps == MEASURED_STEPS_PER_SESSION
            and self.training_episode_receipt_count > 0
            and self.checkpoint_receipt_count == len(CHECKPOINT_STEPS)
            and self.evaluation_episode_receipt_count
            == EVALUATION_EPISODES_PER_SESSION
            and self.policy_version is not None
            and self.transition_sha256
            and self.action_sha256
            and self.semantic_sha256
            and self.source_set_sha256
            and self.runtime_capability_sha256
            and self.duplicate_count == 0
            and self.reject_count == 0
            and self.failure_count == 0
            and self.mask_violation_count == 0
            and self.evaluation_leak_count == 0
            and self.missing_checkpoint_count == 0
            and self.semantic_parity_passed
            and self.budget_respected
            and self.cleanup_complete
            and self.exclusion_reason is None
        )

    def content(self) -> dict[str, object]:
        return {
            "action_sha256": self.action_sha256,
            "artifacts": [item.content() for item in self.artifacts],
            "attempt": self.attempt,
            "budget_respected": self.budget_respected,
            "cell_id": self.cell_id,
            "checkpoint_receipt_count": self.checkpoint_receipt_count,
            "cleanup_complete": self.cleanup_complete,
            "duplicate_count": self.duplicate_count,
            "evaluation_episode_receipt_count": self.evaluation_episode_receipt_count,
            "evaluation_leak_count": self.evaluation_leak_count,
            "exclusion_reason": self.exclusion_reason,
            "failure_count": self.failure_count,
            "failure_phase": self.failure_phase,
            "failure_reason": self.failure_reason,
            "federation_id": self.federation_id,
            "final_return": self.final_return,
            "first_step_committed": self.first_step_committed,
            "generation": self.generation,
            "host_ids": list(self.host_ids),
            "logical_time_end": self.logical_time_end,
            "logical_time_start": self.logical_time_start,
            "mask_violation_count": self.mask_violation_count,
            "master_seed": self.master_seed,
            "missing_checkpoint_count": self.missing_checkpoint_count,
            "outcome": self.outcome,
            "policy_version": self.policy_version,
            "receipt_id": self.receipt_id,
            "reject_count": self.reject_count,
            "rerun_of_receipt_id": self.rerun_of_receipt_id,
            "resources": self.resources.content(),
            "runtime_capability_sha256": self.runtime_capability_sha256,
            "semantic_parity_passed": self.semantic_parity_passed,
            "semantic_sha256": self.semantic_sha256,
            "session_id": self.session_id,
            "source_set_sha256": self.source_set_sha256,
            "status": self.status.value,
            "training_environment_steps": self.training_environment_steps,
            "training_episode_receipt_count": self.training_episode_receipt_count,
            "transition_sha256": self.transition_sha256,
            "worker_ids": list(self.worker_ids),
        }


# Compatibility name retained for callers created before IF-RL-015 split the
# session and episode schemas explicitly.
TerminalReceipt = SessionReceipt


@dataclass(frozen=True, slots=True)
class EpisodeReceipt:
    """Immutable child evidence for one executed measured evaluation episode."""

    receipt_id: str
    parent_session_id: str
    cell_id: str
    status: RunStatus
    episode_seed: int
    scenario_id: str
    config_sha256: str
    checkpoint_step: int
    worker_id: str
    federation_id: str | None
    generation: int | None
    policy_version: int
    logical_time_start: float
    logical_time_end: float
    steps: int
    total_return: float
    outcome: str
    raw_transition_count: int
    transition_sha256: str
    action_sha256: str
    semantic_sha256: str
    duplicate_count: int
    reject_count: int
    failure_count: int
    terminal: bool
    artifacts: tuple[ArtifactReference, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty("episode receipt_id", self.receipt_id)
        _non_empty("episode parent_session_id", self.parent_session_id)
        if self.cell_id not in PRIMARY_CELL_IDS:
            raise MatchedProtocolError("episode receipt cell is not primary")
        if self.status is RunStatus.PLANNED:
            raise MatchedProtocolError("episode receipt cannot remain planned")
        _non_negative_int("episode_seed", self.episode_seed)
        _non_empty("scenario_id", self.scenario_id)
        _sha256("episode config_sha256", self.config_sha256)
        if self.checkpoint_step not in CHECKPOINT_STEPS:
            raise MatchedProtocolError("episode checkpoint_step is not registered")
        _non_empty("episode worker_id", self.worker_id)
        _non_negative_int("episode policy_version", self.policy_version)
        start = _finite("episode logical_time_start", self.logical_time_start)
        end = _finite("episode logical_time_end", self.logical_time_end)
        if end < start:
            raise MatchedProtocolError("episode logical time regressed")
        _positive_int("episode steps", self.steps)
        if self.steps > MAX_EVALUATION_STEPS_PER_EPISODE:
            raise MatchedProtocolError("evaluation episode exceeded its step limit")
        _finite("episode total_return", self.total_return)
        _non_empty("episode outcome", self.outcome)
        _non_negative_int("episode raw_transition_count", self.raw_transition_count)
        if self.raw_transition_count != self.steps:
            raise MatchedProtocolError("episode raw transition count differs from steps")
        for name in ("transition_sha256", "action_sha256", "semantic_sha256"):
            _sha256(f"episode {name}", getattr(self, name))
        for name in ("duplicate_count", "reject_count", "failure_count"):
            _non_negative_int(f"episode {name}", getattr(self, name))
        if type(self.terminal) is not bool or not self.terminal:
            raise MatchedProtocolError("episode receipt must prove a terminal state")
        if self.cell_id.startswith("local-"):
            if self.federation_id is not None or self.generation is not None:
                raise MatchedProtocolError(
                    "local episode receipts must keep federation fields null"
                )
        elif self.federation_id is None or self.generation is None:
            raise MatchedProtocolError(
                "joined episode receipts require federation_id and generation"
            )
        if self.generation is not None:
            _non_negative_int("episode generation", self.generation)
        artifacts = tuple(self.artifacts)
        if not artifacts or any(
            not isinstance(item, ArtifactReference) for item in artifacts
        ):
            raise MatchedProtocolError(
                "episode artifacts require ArtifactReference values"
            )
        if len({item.path for item in artifacts}) != len(artifacts):
            raise MatchedProtocolError("episode artifact paths must be unique")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    @property
    def sample_eligible(self) -> bool:
        return bool(
            self.status is RunStatus.COMPLETED
            and self.duplicate_count == 0
            and self.reject_count == 0
            and self.failure_count == 0
            and self.terminal
        )

    def content(self) -> dict[str, object]:
        return {
            "action_sha256": self.action_sha256,
            "artifacts": [item.content() for item in self.artifacts],
            "cell_id": self.cell_id,
            "checkpoint_step": self.checkpoint_step,
            "config_sha256": self.config_sha256,
            "duplicate_count": self.duplicate_count,
            "episode_seed": self.episode_seed,
            "failure_count": self.failure_count,
            "federation_id": self.federation_id,
            "generation": self.generation,
            "logical_time_end": self.logical_time_end,
            "logical_time_start": self.logical_time_start,
            "outcome": self.outcome,
            "parent_session_id": self.parent_session_id,
            "policy_version": self.policy_version,
            "raw_transition_count": self.raw_transition_count,
            "receipt_id": self.receipt_id,
            "reject_count": self.reject_count,
            "scenario_id": self.scenario_id,
            "semantic_sha256": self.semantic_sha256,
            "status": self.status.value,
            "steps": self.steps,
            "terminal": self.terminal,
            "total_return": self.total_return,
            "transition_sha256": self.transition_sha256,
            "worker_id": self.worker_id,
        }


def validate_terminal_receipts(
    manifest: MatchedLearningManifest,
    plan: Sequence[MeasuredSessionPlan],
    receipts: Sequence[SessionReceipt],
) -> None:
    validate_measured_plan(manifest, plan)
    values = tuple(receipts)
    if len(values) != MEASURED_SESSION_COUNT:
        raise MatchedProtocolError("exactly 80 measured session receipts are required")
    if len({item.receipt_id for item in values}) != len(values):
        raise MatchedProtocolError("terminal receipt IDs must be unique")
    if len({item.session_id for item in values}) != len(values):
        raise MatchedProtocolError(
            "one terminal receipt is required for every measured session"
        )
    plan_by_id = {item.session_id: item for item in plan}
    for receipt in values:
        planned = plan_by_id.get(receipt.session_id)
        if planned is None or (
            planned.cell_id,
            planned.master_seed,
        ) != (receipt.cell_id, receipt.master_seed):
            raise MatchedProtocolError("terminal receipt differs from measured plan")
    if sum(item.checkpoint_receipt_count for item in values) != MEASURED_CHECKPOINT_COUNT:
        raise MatchedProtocolError("checkpoint receipt count does not reconcile to 880")
    if (
        sum(item.evaluation_episode_receipt_count for item in values)
        != MEASURED_EVALUATION_EPISODE_COUNT
    ):
        raise MatchedProtocolError(
            "evaluation episode receipt count does not reconcile to 56,320"
        )


def validate_evaluation_episode_receipts(
    manifest: MatchedLearningManifest,
    plan: Sequence[MeasuredSessionPlan],
    sessions: Sequence[SessionReceipt],
    episodes: Sequence[EpisodeReceipt],
) -> None:
    """Reconcile every holdout child to its immutable session and plan."""

    validate_terminal_receipts(manifest, plan, sessions)
    values = tuple(episodes)
    if len(values) != MEASURED_EVALUATION_EPISODE_COUNT:
        raise MatchedProtocolError(
            "exactly 56,320 measured evaluation episode receipts are required"
        )
    if len({item.receipt_id for item in values}) != len(values):
        raise MatchedProtocolError("evaluation episode receipt IDs must be unique")
    plan_by_id = {item.session_id: item for item in plan}
    session_by_id = {item.session_id: item for item in sessions}
    expected_config_digests = {
        item.config_sha256 for item in manifest.scenarios.evaluation
    }
    observed_keys: set[tuple[str, int, str]] = set()
    child_count: dict[str, int] = {}
    for episode in values:
        planned = plan_by_id.get(episode.parent_session_id)
        session = session_by_id.get(episode.parent_session_id)
        if planned is None or session is None:
            raise MatchedProtocolError("orphan evaluation episode receipt")
        if episode.cell_id != planned.cell_id or episode.cell_id != session.cell_id:
            raise MatchedProtocolError("evaluation episode cell differs from parent")
        if episode.config_sha256 not in expected_config_digests or (
            episode.config_sha256 not in planned.evaluation_configs
        ):
            raise MatchedProtocolError("evaluation episode config is not holdout")
        if not episode.sample_eligible:
            raise MatchedProtocolError("evaluation episode is not sample eligible")
        key = (
            episode.parent_session_id,
            episode.checkpoint_step,
            episode.config_sha256,
        )
        if key in observed_keys:
            raise MatchedProtocolError("duplicate parent/checkpoint/config episode")
        observed_keys.add(key)
        child_count[episode.parent_session_id] = (
            child_count.get(episode.parent_session_id, 0) + 1
        )
    for session in sessions:
        actual = child_count.get(session.session_id, 0)
        if actual != session.evaluation_episode_receipt_count:
            raise MatchedProtocolError(
                "evaluation child count differs from parent session"
            )
        if actual != EVALUATION_EPISODES_PER_SESSION:
            raise MatchedProtocolError(
                "each measured session requires exactly 704 evaluation children"
            )


def _validate_relative_path(name: str, value: object) -> str:
    text = _non_empty(name, value)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or text != pure.as_posix():
        raise MatchedProtocolError(f"{name} must be a normalized relative POSIX path")
    return text


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    path: str
    role: str
    media_type: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_relative_path("ledger entry path", self.path)
        if self.role not in REQUIRED_LEDGER_ROLES:
            raise MatchedProtocolError(f"unsupported ledger role {self.role!r}")
        _non_empty("ledger media_type", self.media_type)
        _positive_int("ledger size_bytes", self.size_bytes)
        _sha256("ledger entry sha256", self.sha256)

    def content(self) -> dict[str, object]:
        return {
            "media_type": self.media_type,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        fields = {"media_type", "path", "role", "sha256", "size_bytes"}
        if set(value) != fields:
            raise MatchedProtocolError("ledger entry fields differ from schema")
        return cls(
            path=cast(str, value["path"]),
            role=cast(str, value["role"]),
            media_type=cast(str, value["media_type"]),
            size_bytes=cast(int, value["size_bytes"]),
            sha256=cast(str, value["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class EvidenceLedger:
    experiment_id: str
    manifest_sha256: str
    plan_sha256: str
    entries: tuple[LedgerEntry, ...]
    claim_grade: bool = False
    schema_version: str = MATCHED_LEDGER_SCHEMA_VERSION
    canonical_json_id: str = CANONICAL_JSON_ID
    ledger_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != MATCHED_LEDGER_SCHEMA_VERSION:
            raise MatchedProtocolError("unsupported evidence ledger schema_version")
        if self.canonical_json_id != CANONICAL_JSON_ID:
            raise MatchedProtocolError("unsupported canonical JSON contract")
        _non_empty("ledger experiment_id", self.experiment_id)
        _sha256("ledger manifest_sha256", self.manifest_sha256)
        _sha256("ledger plan_sha256", self.plan_sha256)
        if type(self.claim_grade) is not bool:
            raise MatchedProtocolError("claim_grade must be bool")
        entries = tuple(self.entries)
        if not entries:
            raise MatchedProtocolError("ledger entries must not be empty")
        if any(not isinstance(item, LedgerEntry) for item in entries):
            raise MatchedProtocolError("ledger may contain only LedgerEntry values")
        if len({item.path for item in entries}) != len(entries):
            raise MatchedProtocolError("ledger entry paths must be unique")
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda item: item.path)))
        object.__setattr__(self, "ledger_sha256", _content_sha256(self.content()))

    @property
    def complete_for_claim(self) -> bool:
        return {item.role for item in self.entries} == set(REQUIRED_LEDGER_ROLES)

    def content(self) -> dict[str, object]:
        return {
            "canonical_json_id": self.canonical_json_id,
            "claim_grade": self.claim_grade,
            "entries": [item.content() for item in self.entries],
            "experiment_id": self.experiment_id,
            "manifest_sha256": self.manifest_sha256,
            "plan_sha256": self.plan_sha256,
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
            "claim_grade",
            "entries",
            "experiment_id",
            "ledger_sha256",
            "manifest_sha256",
            "plan_sha256",
            "schema_version",
        }
        if set(value) != fields:
            raise MatchedProtocolError("ledger fields differ from schema")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list):
            raise MatchedProtocolError("ledger entries must be an array")
        entries: list[LedgerEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, Mapping):
                raise MatchedProtocolError("ledger entry must be an object")
            entries.append(LedgerEntry.from_dict(cast(Mapping[str, object], raw)))
        ledger = cls(
            experiment_id=cast(str, value["experiment_id"]),
            manifest_sha256=cast(str, value["manifest_sha256"]),
            plan_sha256=cast(str, value["plan_sha256"]),
            entries=tuple(entries),
            claim_grade=cast(bool, value["claim_grade"]),
            schema_version=cast(str, value["schema_version"]),
            canonical_json_id=cast(str, value["canonical_json_id"]),
        )
        if value["ledger_sha256"] != ledger.ledger_sha256:
            raise MatchedProtocolError("ledger content digest differs")
        return ledger


def write_evidence_ledger(path: Path, ledger: EvidenceLedger) -> str:
    """Create one canonical ledger without ever replacing an existing path."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(ledger, EvidenceLedger):
        raise TypeError("ledger must be EvidenceLedger")
    payload = _canonical_json(ledger.to_dict())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _reject_constant(value: str) -> object:
    raise MatchedProtocolError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MatchedProtocolError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def read_evidence_ledger(path: Path, *, expected_file_sha256: str) -> EvidenceLedger:
    """Read canonical bytes, reject schema drift, and verify both digest layers."""

    expected = _sha256("expected ledger file SHA-256", expected_file_sha256)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise MatchedProtocolError("ledger file digest differs")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatchedProtocolError(f"ledger JSON is invalid: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise MatchedProtocolError("ledger root must be an object")
    ledger = EvidenceLedger.from_dict(cast(Mapping[str, object], decoded))
    if payload != _canonical_json(ledger.to_dict()):
        raise MatchedProtocolError("ledger JSON is not canonical")
    return ledger


def verify_ledger_artifacts(root: Path, ledger: EvidenceLedger) -> None:
    """Resolve every immutable entry below ``root`` and re-hash its bytes."""

    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise MatchedProtocolError("ledger artifact root must be a directory")
    for entry in ledger.entries:
        target = (resolved_root / Path(*PurePosixPath(entry.path).parts)).resolve(strict=True)
        try:
            target.relative_to(resolved_root)
        except ValueError as exc:
            raise MatchedProtocolError("ledger artifact escapes its root") from exc
        if not target.is_file() or target.is_symlink():
            raise MatchedProtocolError(f"ledger artifact is not a regular file: {entry.path}")
        payload = target.read_bytes()
        if len(payload) != entry.size_bytes:
            raise MatchedProtocolError(f"ledger artifact size differs: {entry.path}")
        if hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise MatchedProtocolError(f"ledger artifact digest differs: {entry.path}")


def verify_indexed_artifacts(
    root: Path,
    ledger: EvidenceLedger,
    references: Sequence[ArtifactReference],
) -> None:
    """Require every evidence reference to be exactly indexed and re-hashed."""

    verify_ledger_artifacts(root, ledger)
    indexed = {entry.path: entry for entry in ledger.entries}
    for reference in references:
        entry = indexed.get(reference.path)
        if entry is None:
            raise MatchedProtocolError(
                f"referenced artifact is not ledger-indexed: {reference.path}"
            )
        if (
            entry.media_type,
            entry.size_bytes,
            entry.sha256,
        ) != (
            reference.media_type,
            reference.size_bytes,
            reference.sha256,
        ):
            raise MatchedProtocolError(
                f"referenced artifact metadata differs: {reference.path}"
            )


@dataclass(frozen=True, slots=True)
class ClaimDecision:
    claim: ClaimKind
    admitted: bool
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        blockers = tuple(self.blockers)
        if self.admitted == bool(blockers):
            raise MatchedProtocolError("claim admission must be the inverse of blockers")
        object.__setattr__(self, "blockers", blockers)

    def content(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "blockers": list(self.blockers),
            "claim": self.claim.value,
        }


@dataclass(frozen=True, slots=True)
class AnalysisReceipt:
    """Computed learner-specific confirmatory and quality admission."""

    analysis_id: str
    contrast_id: str
    measured_manifest_sha256: str
    paired_master_count: int
    bootstrap_samples: int
    bootstrap_method: str
    bootstrap_seed_derivation_id: str
    point_estimate: float
    interval_low: float
    interval_high: float
    minimum_final_capture_rate: float
    trained_minus_frozen_low: float
    trained_minus_random_low: float
    mask_violation_count: int
    evaluation_leak_count: int
    missing_checkpoint_count: int
    paired_input_artifact: ArtifactReference
    analysis_output_artifact: ArtifactReference
    analysis_code_artifact: ArtifactReference
    admitted: bool = field(init=False)
    blockers: tuple[str, ...] = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty("analysis_id", self.analysis_id)
        if self.contrast_id not in {
            "reference-joined-minus-local",
            "rllib-joined-minus-local",
        }:
            raise MatchedProtocolError("analysis contrast_id is not registered")
        _sha256("analysis measured_manifest_sha256", self.measured_manifest_sha256)
        _non_negative_int("analysis paired_master_count", self.paired_master_count)
        _non_negative_int("analysis bootstrap_samples", self.bootstrap_samples)
        for name in (
            "point_estimate",
            "interval_low",
            "interval_high",
            "minimum_final_capture_rate",
            "trained_minus_frozen_low",
            "trained_minus_random_low",
        ):
            _finite(name, getattr(self, name))
        for name in (
            "mask_violation_count",
            "evaluation_leak_count",
            "missing_checkpoint_count",
        ):
            _non_negative_int(name, getattr(self, name))
        for name in (
            "paired_input_artifact",
            "analysis_output_artifact",
            "analysis_code_artifact",
        ):
            if not isinstance(getattr(self, name), ArtifactReference):
                raise MatchedProtocolError(f"{name} must be an ArtifactReference")
        blockers: list[str] = []
        if self.paired_master_count != MEASURED_MASTER_COUNT:
            blockers.append("requires-20-of-20-paired-masters")
        if self.bootstrap_samples != BOOTSTRAP_SAMPLES:
            blockers.append("bootstrap-sample-count-differs")
        if self.bootstrap_method != BOOTSTRAP_METHOD:
            blockers.append("bootstrap-method-differs")
        if self.bootstrap_seed_derivation_id != BOOTSTRAP_SEED_DERIVATION_ID:
            blockers.append("bootstrap-seed-derivation-differs")
        if self.interval_high <= self.interval_low:
            blockers.append("interval-is-zero-width-or-reversed")
        if self.interval_low <= -0.05:
            blockers.append("noninferiority-lower-bound-not-above-margin")
        if self.minimum_final_capture_rate < 0.60:
            blockers.append("final-capture-rate-below-threshold")
        if self.trained_minus_frozen_low <= 0.05:
            blockers.append("trained-minus-frozen-floor-not-separated")
        if self.trained_minus_random_low <= 0.05:
            blockers.append("trained-minus-random-floor-not-separated")
        if self.mask_violation_count:
            blockers.append("action-mask-violations")
        if self.evaluation_leak_count:
            blockers.append("evaluation-leakage")
        if self.missing_checkpoint_count:
            blockers.append("missing-checkpoints")
        frozen = tuple(blockers)
        object.__setattr__(self, "blockers", frozen)
        object.__setattr__(self, "admitted", not frozen)
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def validate_measured_manifest(self, manifest: MatchedMeasuredManifestV1) -> None:
        if self.measured_manifest_sha256 != manifest.manifest_sha256:
            raise MatchedProtocolError("analysis measured manifest digest differs")

    def content(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "analysis_id": self.analysis_id,
            "analysis_code_artifact": self.analysis_code_artifact.content(),
            "analysis_output_artifact": self.analysis_output_artifact.content(),
            "blockers": list(self.blockers),
            "bootstrap_method": self.bootstrap_method,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed_derivation_id": self.bootstrap_seed_derivation_id,
            "contrast_id": self.contrast_id,
            "evaluation_leak_count": self.evaluation_leak_count,
            "interval_high": self.interval_high,
            "interval_low": self.interval_low,
            "mask_violation_count": self.mask_violation_count,
            "measured_manifest_sha256": self.measured_manifest_sha256,
            "minimum_final_capture_rate": self.minimum_final_capture_rate,
            "missing_checkpoint_count": self.missing_checkpoint_count,
            "paired_master_count": self.paired_master_count,
            "point_estimate": self.point_estimate,
            "paired_input_artifact": self.paired_input_artifact.content(),
            "trained_minus_frozen_low": self.trained_minus_frozen_low,
            "trained_minus_random_low": self.trained_minus_random_low,
        }

    @property
    def artifacts(self) -> tuple[ArtifactReference, ...]:
        return (
            self.paired_input_artifact,
            self.analysis_output_artifact,
            self.analysis_code_artifact,
        )


@dataclass(frozen=True, slots=True)
class ClaimMatrix:
    decisions: tuple[ClaimDecision, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        if {item.claim for item in decisions} != set(ClaimKind):
            raise MatchedProtocolError("claim matrix must contain every claim kind")
        object.__setattr__(
            self,
            "decisions",
            tuple(sorted(decisions, key=lambda item: item.claim.value)),
        )
        object.__setattr__(
            self,
            "sha256",
            _content_sha256([item.content() for item in self.decisions]),
        )


def build_claim_matrix(
    manifest: MatchedLearningManifest,
    measured_manifest: MatchedMeasuredManifestV1,
    plan: Sequence[MeasuredSessionPlan],
    receipts: Sequence[TerminalReceipt],
    capabilities: Sequence[CapabilityReceipt],
    joined_capabilities: Sequence[JoinedCapabilityReceipt],
    analyses: Sequence[AnalysisReceipt],
    ledger: EvidenceLedger,
    registrations: Sequence[ExternalRegistrationReceipt],
    timeline: ExecutionTimeline,
    episodes: Sequence[EpisodeReceipt],
    tuning_completion: TuningCompletionReceipt,
    artifact_root: Path,
) -> ClaimMatrix:
    """Compute separate fail-closed claim decisions; no global success boolean."""

    common: list[str] = []
    try:
        validate_external_registrations(
            manifest, measured_manifest, registrations, timeline
        )
    except MatchedProtocolError as exc:
        common.append(f"external-registration:{exc}")
    try:
        validate_terminal_receipts(manifest, plan, receipts)
    except MatchedProtocolError as exc:
        common.append(f"receipt-reconciliation:{exc}")
    try:
        validate_evaluation_episode_receipts(
            manifest, plan, receipts, episodes
        )
    except MatchedProtocolError as exc:
        common.append(f"episode-reconciliation:{exc}")
    try:
        tuning_completion.validate_measured_manifest(measured_manifest)
    except MatchedProtocolError as exc:
        common.append(f"tuning-completion:{exc}")
    if ledger.experiment_id != manifest.experiment_id:
        common.append("ledger-experiment-mismatch")
    if measured_manifest.measured_plan_sha256 != measured_plan_sha256(plan):
        common.append("measured-manifest-plan-mismatch")
    if ledger.manifest_sha256 != measured_manifest.manifest_sha256:
        common.append("ledger-manifest-mismatch")
    if ledger.plan_sha256 != measured_plan_sha256(plan):
        common.append("ledger-plan-mismatch")
    if not ledger.complete_for_claim:
        common.append("ledger-incomplete")
    if not ledger.claim_grade:
        common.append("ledger-review-not-claim-grade")

    artifact_references: list[ArtifactReference] = []
    for registration in registrations:
        if registration.registered_object is not None:
            artifact_references.append(registration.registered_object)
        if registration.authority_receipt is not None:
            artifact_references.append(registration.authority_receipt)
    artifact_references.extend(
        artifact for session in receipts for artifact in session.artifacts
    )
    artifact_references.extend(
        artifact for episode in episodes for artifact in episode.artifacts
    )
    artifact_references.extend(tuning_completion.artifacts)
    artifact_references.extend(
        capability.probe_artifact
        for capability in capabilities
        if capability.probe_artifact is not None
    )
    artifact_references.extend(
        phase.artifact
        for joined_capability in joined_capabilities
        for phase in joined_capability.phase_artifacts
    )
    artifact_references.extend(
        artifact for analysis in analyses for artifact in analysis.artifacts
    )
    try:
        verify_indexed_artifacts(artifact_root, ledger, artifact_references)
    except (MatchedProtocolError, FileNotFoundError, OSError) as exc:
        common.append(f"artifact-verification:{exc}")

    capability_values = tuple(capabilities)
    if len({item.kind for item in capability_values}) != len(capability_values):
        common.append("duplicate-capability-kind")
    capability_by_kind: dict[CapabilityKind, CapabilityReceipt] = {}
    for capability in capability_values:
        capability_by_kind.setdefault(capability.kind, capability)
    rllib_blockers: list[str] = []
    rllib = capability_by_kind.get(CapabilityKind.RLLIB)
    if rllib is None:
        rllib_blockers.append("native-rllib-capability-missing")
    else:
        try:
            validate_native_capability(manifest, rllib)
        except MatchedProtocolError as exc:
            rllib_blockers.append(f"native-rllib-not-admitted:{exc}")

    values = tuple(receipts)

    joined_values = tuple(joined_capabilities)
    if len({item.cell_id for item in joined_values}) != len(joined_values):
        common.append("duplicate-joined-capability-cell")
    joined_by_cell: dict[str, JoinedCapabilityReceipt] = {}
    for joined_capability in joined_values:
        joined_by_cell.setdefault(joined_capability.cell_id, joined_capability)

    def joined_blockers(cell_id: str) -> list[str]:
        receipt = joined_by_cell.get(cell_id)
        if receipt is None:
            return [f"joined-capability-missing:{cell_id}"]
        try:
            validate_joined_capability(manifest, receipt)
        except MatchedProtocolError as exc:
            return [f"joined-capability-not-admitted:{cell_id}:{exc}"]
        return []

    analysis_values = tuple(analyses)
    if len({item.contrast_id for item in analysis_values}) != len(analysis_values):
        common.append("duplicate-analysis-contrast")
    analysis_by_contrast: dict[str, AnalysisReceipt] = {}
    for analysis in analysis_values:
        analysis_by_contrast.setdefault(analysis.contrast_id, analysis)

    def analysis_blockers(contrast_id: str) -> list[str]:
        receipt = analysis_by_contrast.get(contrast_id)
        if receipt is None:
            return [f"analysis-missing:{contrast_id}"]
        try:
            receipt.validate_measured_manifest(measured_manifest)
        except MatchedProtocolError as exc:
            return [f"analysis-manifest-mismatch:{contrast_id}:{exc}"]
        return [f"analysis:{contrast_id}:{item}" for item in receipt.blockers]

    def backend_blockers(engine: LearnerEngine) -> list[str]:
        blockers: list[str] = []
        engine_cells = {
            cell.cell_id for cell in manifest.cells if cell.learner_engine is engine
        }
        eligible = [
            item
            for item in values
            if item.cell_id in engine_cells and item.sample_eligible
        ]
        by_master: dict[int, set[str]] = {
            item.master_seed: set() for item in eligible
        }
        for item in eligible:
            by_master[item.master_seed].add(item.cell_id)
        paired = sum(1 for cell_ids in by_master.values() if cell_ids == engine_cells)
        if paired != MEASURED_MASTER_COUNT:
            blockers.append(f"requires-20-of-20-post-start-pairs:observed-{paired}")
        return blockers

    rllib_receipt = capability_by_kind.get(CapabilityKind.RLLIB)
    for session in values:
        if not session.sample_eligible:
            continue
        joined_receipt = joined_by_cell.get(session.cell_id)
        try:
            expected_capability = cell_capability_sha256(
                manifest,
                session.cell_id,
                rllib=(
                    rllib_receipt if session.cell_id.endswith("rllib") else None
                ),
                joined=(
                    joined_receipt
                    if session.cell_id.startswith("joined-")
                    else None
                ),
            )
        except MatchedProtocolError as exc:
            common.append(f"session-capability-unresolved:{session.session_id}:{exc}")
            continue
        if session.source_set_sha256 != manifest.source_set_sha256:
            common.append(f"session-source-set-mismatch:{session.session_id}")
        if session.runtime_capability_sha256 != expected_capability:
            common.append(f"session-capability-mismatch:{session.session_id}")

    reference = (
        common
        + backend_blockers(LearnerEngine.REFERENCE_PPO)
        + joined_blockers("joined-gorti-reference")
        + analysis_blockers("reference-joined-minus-local")
        + [
            "reference-learner-update-capability-not-implemented",
            "raw-auc-bca-recomputation-not-implemented",
        ]
    )
    rllib_claim = common + rllib_blockers + backend_blockers(
        LearnerEngine.NATIVE_RLLIB_PPO
    )
    rllib_claim += joined_blockers("joined-gorti-rllib")
    rllib_claim += analysis_blockers("rllib-joined-minus-local")
    rllib_claim.append("raw-auc-bca-recomputation-not-implemented")
    full = list(dict.fromkeys(reference + rllib_claim))

    dex_blockers = list(common)
    dexsim = capability_by_kind.get(CapabilityKind.DEXSIM)
    if not manifest.dexsim_contender_declared:
        dex_blockers.append("dexsim-contender-not-declared")
    elif dexsim is None:
        dex_blockers.append("native-dexsim-capability-missing")
    else:
        try:
            validate_native_capability(manifest, dexsim)
        except MatchedProtocolError as exc:
            dex_blockers.append(f"native-dexsim-not-admitted:{exc}")

    descriptive = list(common)
    if any(not item.sample_eligible for item in values):
        descriptive.append("learner-description-requires-complete-primary-cells")

    timing = list(common)
    if any(not item.sample_eligible for item in values):
        timing.append("timing-requires-all-primary-samples")
    timing.extend(joined_blockers("joined-gorti-reference"))
    timing.extend(joined_blockers("joined-gorti-rllib"))
    timing.append("timing-superiority-hypothesis-not-registered")
    cross_host = list(timing)
    joined_hosts = {
        host
        for item in values
        if item.cell_id.startswith("joined-")
        for host in item.host_ids
    }
    if len(joined_hosts) < 2:
        cross_host.append("cross-host-execution-not-observed")
    cross_model = list(common)
    cross_model.append("cross-model-family-not-executed")
    scie = list(dict.fromkeys(full))
    scie.append("effective-workload-source-profile-capability-not-implemented")

    decisions = []
    for claim, blockers in (
        (ClaimKind.FULL_2X2, full),
        (ClaimKind.REFERENCE_BACKEND, reference),
        (ClaimKind.RLLIB_BACKEND, rllib_claim),
        (ClaimKind.LEARNER_DESCRIPTIVE, descriptive),
        (ClaimKind.DEXSIM, dex_blockers),
        (ClaimKind.TIMING, timing),
        (ClaimKind.CROSS_HOST, cross_host),
        (ClaimKind.CROSS_MODEL, cross_model),
        (ClaimKind.SCIE_PACKAGE, scie),
    ):
        unique = tuple(dict.fromkeys(blockers))
        decisions.append(ClaimDecision(claim, not unique, unique))
    return ClaimMatrix(tuple(decisions))


__all__ = [
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_SEED_DERIVATION_ID",
    "AnalysisReceipt",
    "CapabilityDisposition",
    "CapabilityKind",
    "CapabilityReceipt",
    "CellSpec",
    "ClaimDecision",
    "ClaimKind",
    "ClaimMatrix",
    "EvidenceLedger",
    "EpisodeReceipt",
    "ExecutionTimeline",
    "ExternalRegistrationReceipt",
    "JoinedCapabilityReceipt",
    "JoinedPhaseArtifact",
    "LedgerEntry",
    "LearnerEngine",
    "MATCHED_LEDGER_SCHEMA_VERSION",
    "MATCHED_MANIFEST_SCHEMA_VERSION",
    "MatchedLearningManifest",
    "MatchedMeasuredManifestV1",
    "MatchedProtocolManifestV1",
    "MatchedProtocolError",
    "MeasuredSessionPlan",
    "RegistrationRequirement",
    "RegistrationStage",
    "ResourceCounters",
    "RolloutPlane",
    "RunStatus",
    "ScenarioCase",
    "ScenarioPartition",
    "SessionReceipt",
    "SourceLock",
    "TerminalReceipt",
    "TuningCompletionReceipt",
    "ArtifactReference",
    "bootstrap_seed_for",
    "bootstrap_resample_seed",
    "build_claim_matrix",
    "build_measured_plan",
    "cell_capability_sha256",
    "measured_plan_sha256",
    "read_evidence_ledger",
    "validate_external_registrations",
    "validate_evaluation_episode_receipts",
    "validate_joined_capability",
    "validate_measured_plan",
    "validate_native_capability",
    "validate_terminal_receipts",
    "verify_ledger_artifacts",
    "verify_indexed_artifacts",
    "write_evidence_ledger",
]
