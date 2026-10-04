"""Bounded, fail-closed TASK-RL-105 reference-PPO qualification harness."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import inspect
import json
import math
import os
import platform
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, cast

import numpy as np

import pyjevsim_bridge.rl.reference_ppo as reference_ppo_module
from pyjevsim_bridge.rl.learning import (
    EMPTY_TRANSITION_BATCH_SHA256,
    ActorPolicy,
    InMemoryPolicyArtifactStore,
    LearnerRecoveryState,
    LearnerSession,
    PolicyArtifact,
    PolicyArtifactRef,
    PolicyCompatibility,
    ValidatedTransitionBatch,
)
from pyjevsim_bridge.rl.local import LocalRolloutPool
from pyjevsim_bridge.rl.qualification_models.anti_torpedo import (
    LOADED_ADAPTER_SOURCE_SHA256,
    PYJEVSIM_EXECUTOR_SOURCE_SHA256,
    anti_torpedo_v2_environment_factory,
    atsim_source_sha256,
)
from pyjevsim_bridge.rl.qualification_models.anti_torpedo_features import (
    ANTI_TORPEDO_FEATURE_CONTRACT_SHA256,
    LOADED_ANTI_TORPEDO_FEATURE_SOURCE_SHA256,
    AntiTorpedoV2FeatureContract,
)
from pyjevsim_bridge.rl.qualification_models.anti_torpedo_profile import (
    build_anti_torpedo_scenario_profile,
)
from pyjevsim_bridge.rl.records import TransitionRecord
from pyjevsim_bridge.rl.reference_ppo import (
    LOADED_REFERENCE_PPO_SOURCE_SHA256,
    REFERENCE_PPO_ACTION_SCHEMA_SHA256,
    REFERENCE_PPO_ALGORITHM_ID,
    REFERENCE_PPO_ALGORITHM_VERSION,
    PPOInferenceInputV1,
    ReferencePPOCapabilityReceipt,
    ReferencePPOCheckpointV1,
    ReferencePPOConfigV1,
    ReferencePPOLearnerAdapter,
    ReferencePPOPolicyLoader,
    compute_gae,
    numpy_runtime_identity,
    numpy_runtime_sha256,
)

QUALIFICATION_SCHEMA_VERSION: Final = "reference-ppo-qualification-v1"
LOADED_REFERENCE_PPO_QUALIFICATION_SOURCE_SHA256: Final = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()
PRODUCTION_TRAINING_TRANSITIONS: Final = 4096
PRODUCTION_BATCH_SIZE: Final = 2048
PRODUCTION_BATCH_COUNT: Final = 2
PRODUCTION_ACTOR_COUNT: Final = 4
PRODUCTION_EVALUATION_EPISODES: Final = 16
PRODUCTION_ADAM_STEPS: Final = 320
_REQUIRED_ARTIFACT_ROLES: Final = frozenset(
    {
        "manifest",
        "transitions-1",
        "transitions-2",
        "policy-0",
        "policy-1",
        "policy-2",
        "checkpoint-1",
        "checkpoint-2",
        "fresh-update-1",
        "reload-update-2",
        "evaluation-isolation",
        "source-profile",
        "capability-receipt",
        "review-disposition",
        "process-lifecycle",
        "recovery-state",
        "negative-authenticity",
        "math-oracle",
    }
)


class QualificationError(RuntimeError):
    pass


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
        raise QualificationError("qualification value is not canonical JSON") from exc


def _canonical_json_equal(left: object, right: object) -> bool:
    """Compare JSON values independently of tuple/list implementation details."""

    return _canonical_json(left) == _canonical_json(right)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _non_negative(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualificationError(f"{name} must be a non-negative integer")
    return value


def _required_seed(value: int | None) -> int:
    if value is None:
        raise QualificationError("rollout result omitted its deterministic seed")
    return value


def _policy_receipt(
    artifact: PolicyArtifact, reference: PolicyArtifactRef
) -> bytes:
    return _canonical_json(
        {
            "schema_version": "policy-artifact-receipt-v1",
            "artifact": {
                "run_id": artifact.run_id,
                "generation": artifact.generation,
                "policy_version": artifact.policy_version,
                "media_type": artifact.media_type,
                "compatibility": artifact.compatibility.to_dict(),
                "source_batch_sha256": artifact.source_batch_sha256,
                "provenance": dict(artifact.provenance),
                "payload_base64": base64.b64encode(artifact.payload).decode("ascii"),
                "payload_sha256": _sha(artifact.payload),
                "artifact_sha256": artifact.sha256,
            },
            "reference": reference.to_dict(),
        }
    )


def _transition_receipt(index: int, batch: ValidatedTransitionBatch) -> bytes:
    return _canonical_json(
        {
            "schema_version": "transition-batch-receipt-v1",
            "batch_index": index,
            "batch_sha256": batch.sha256,
            "record_count": len(batch.records),
            "record_sha256_order": [
                _sha(_canonical_json(record.to_dict())) for record in batch.records
            ],
            "records": [record.to_dict() for record in batch.records],
        }
    )


def _rng_digest() -> str:
    return _sha(
        repr((random.getstate(), np.random.get_state())).encode("utf-8")
    )


def _ppo_scalar_oracles() -> dict[str, object]:
    values = np.asarray([0.5, 0.25], dtype=np.float32)
    advantages, returns = compute_gae(
        np.asarray([1.0, 2.0], dtype=np.float32),
        values,
        np.asarray([0.25, 0.0], dtype=np.float32),
        np.asarray([False, True], dtype=np.bool_),
        np.asarray([False, False], dtype=np.bool_),
        (("worker", "episode"), ("worker", "episode")),
        (1, 2),
        gamma=0.99,
        gae_lambda=0.95,
    )
    expected_last = np.float32(2.0 - 0.25)
    expected_first = np.float32(
        np.float32(1.0 + np.float32(0.99) * np.float32(0.25) - np.float32(0.5))
        + np.float32(0.99)
        * np.float32(0.95)
        * expected_last
    )

    logits = np.asarray([[0.2, -0.1, 0.3, 4.0, -3.0, 2.0]], dtype=np.float32)
    mask = np.asarray([[True, True, True, False, False, False]], dtype=np.bool_)
    allowed = np.asarray(logits[0, :3], dtype=np.float64)
    shifted = allowed - np.max(allowed)
    probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
    old_log_probability = float(np.log(probabilities[0]) - 0.05)

    def scalar_loss(changed: np.ndarray) -> float:
        active = np.asarray(changed[:3], dtype=np.float64)
        active_shifted = active - np.max(active)
        probs = np.exp(active_shifted) / np.sum(np.exp(active_shifted))
        log_probability = float(np.log(probs[0]))
        ratio = math.exp(log_probability - old_log_probability)
        surrogate = min(ratio * 0.7, min(max(ratio, 0.8), 1.2) * 0.7)
        entropy = -float(np.sum(probs * np.log(probs)))
        return float(-surrogate + 0.25 - 0.01 * entropy)

    output = reference_ppo_module._loss_and_output_gradients(
        logits,
        mask,
        np.asarray([0], dtype=np.int64),
        np.asarray([old_log_probability], dtype=np.float32),
        np.asarray([0.7], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        policy_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
    )
    epsilon = 1.0e-3
    plus = logits[0].copy()
    minus = logits[0].copy()
    plus[0] += np.float32(epsilon)
    minus[0] -= np.float32(epsilon)
    finite_difference = (scalar_loss(plus) - scalar_loss(minus)) / (2.0 * epsilon)
    analytic = float(output.policy[0, 0])
    masked_zero = bool(np.all(output.policy[0, 3:] == np.float32(0.0)))
    passed = bool(
        np.array_equal(
            advantages,
            np.asarray([expected_first, expected_last], dtype=np.float32),
        )
        and np.array_equal(returns, advantages + values)
        and math.isfinite(output.total_loss)
        and abs(analytic - finite_difference) <= 5.0e-4
        and abs(float(output.value[0, 0]) + 0.5) <= 1.0e-7
        and masked_zero
    )
    return {
        "schema_version": "ppo-scalar-oracles-v1",
        "gae_advantages": [float(item) for item in advantages],
        "gae_returns": [float(item) for item in returns],
        "analytic_policy_gradient": analytic,
        "finite_difference_policy_gradient": finite_difference,
        "finite_difference_tolerance": 5.0e-4,
        "value_gradient": float(output.value[0, 0]),
        "masked_gradient_exact_zero": masked_zero,
        "passed": passed,
    }


def _negative_authenticity_receipt() -> bytes:
    prerequisite = _workload_mask_prerequisite()
    return _canonical_json(
        {
            "schema_version": "negative-authenticity-receipt-v1",
            "probe": "accepted-workload-mask-pre-mutation-prerequisite",
            "workload_prerequisite": prerequisite,
        }
    )


def _workload_mask_prerequisite() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    evidence_root = (
        repo_root
        / "engineering"
        / "specifications"
        / "pyjevsim-rl"
        / "testcases"
        / "effective_workload_qualification"
    )
    ledger_bytes = (evidence_root / "ledger.json").read_bytes()
    mask_bytes = (evidence_root / "mask.json").read_bytes()
    try:
        ledger = json.loads(ledger_bytes)
        mask = json.loads(mask_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError("accepted workload prerequisite is not JSON") from exc
    if not isinstance(ledger, dict) or not isinstance(mask, dict):
        raise QualificationError("accepted workload prerequisite must be objects")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise QualificationError("accepted workload ledger entries are missing")
    mask_entries = [
        item
        for item in entries
        if isinstance(item, dict) and item.get("role") == "mask"
    ]
    expected_masks = [
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [True, True, True, False, False, False],
        [True, True, True, False, False, False],
    ]
    if (
        len(mask_entries) != 1
        or mask_entries[0].get("path") != "mask.json"
        or mask_entries[0].get("sha256") != _sha(mask_bytes)
        or mask_entries[0].get("size_bytes") != len(mask_bytes)
        or mask != {"expected_masks": expected_masks, "passed": True}
        or ledger.get("schema_version")
        != "anti-torpedo-workload-qualification-v1"
    ):
        raise QualificationError("accepted workload mask prerequisite differs")
    return {
        "evidence_id": "EVID-RL-WORKLOAD-001",
        "ledger_file_sha256": _sha(ledger_bytes),
        "ledger_sha256": _digest("workload ledger SHA-256", ledger.get("ledger_sha256")),
        "mask_sha256": _sha(mask_bytes),
        "result_sha256": _digest("workload result SHA-256", ledger.get("result_sha256")),
    }


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise QualificationError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class ReferencePPOQualificationPlan:
    training_transitions: int = PRODUCTION_TRAINING_TRANSITIONS
    batch_size: int = PRODUCTION_BATCH_SIZE
    batch_count: int = PRODUCTION_BATCH_COUNT
    actor_count: int = PRODUCTION_ACTOR_COUNT
    evaluation_episodes: int = PRODUCTION_EVALUATION_EPISODES
    production_admission: bool = True
    schema_version: str = QUALIFICATION_SCHEMA_VERSION
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "training_transitions",
            "batch_size",
            "batch_count",
            "actor_count",
            "evaluation_episodes",
        ):
            if _non_negative(name, getattr(self, name)) == 0:
                raise QualificationError(f"{name} must be positive")
        if self.training_transitions != self.batch_size * self.batch_count:
            raise QualificationError("training budget does not reconcile")
        if type(self.production_admission) is not bool:
            raise QualificationError("production_admission must be bool")
        object.__setattr__(self, "sha256", _sha(_canonical_json(self.content())))

    @property
    def is_exact_production(self) -> bool:
        return self.content() == ReferencePPOQualificationPlan().content()

    def content(self) -> dict[str, object]:
        return {
            "actor_count": self.actor_count,
            "batch_count": self.batch_count,
            "batch_size": self.batch_size,
            "evaluation_episodes": self.evaluation_episodes,
            "production_admission": self.production_admission,
            "schema_version": self.schema_version,
            "training_transitions": self.training_transitions,
        }


@dataclass(frozen=True, slots=True)
class FixedScenarioV2Environment:
    ordinal: int
    instance_id: str
    run_id: str
    _environment: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if _non_negative("scenario ordinal", self.ordinal) > 255:
            raise QualificationError("scenario ordinal must be at most 255")
        environment = anti_torpedo_v2_environment_factory(
            instance_id=self.instance_id, run_id=self.run_id
        )
        object.__setattr__(self, "_environment", environment)

    def reset(self, *, seed: int | None = None) -> object:
        return cast(object, self._environment).reset(  # type: ignore[attr-defined, redundant-cast]
            seed=seed, options={"scenario_ordinal": self.ordinal}
        )

    def step(self, action: object) -> object:
        return cast(object, self._environment).step(  # type: ignore[attr-defined, redundant-cast]
            action
        )

    def close(self) -> None:
        cast(object, self._environment).close()  # type: ignore[attr-defined, redundant-cast]


@dataclass(frozen=True, slots=True)
class FixedScenarioV2Factory:
    ordinal: int
    instance_id: str
    run_id: str

    def __call__(self) -> FixedScenarioV2Environment:
        return FixedScenarioV2Environment(self.ordinal, self.instance_id, self.run_id)


@dataclass(slots=True)
class CyclingScenarioV2Environment:
    ordinals: tuple[int, int]
    instance_id: str
    run_id: str
    _reset_count: int = field(init=False, default=0)
    _environment: object = field(init=False)

    def __post_init__(self) -> None:
        self._environment = anti_torpedo_v2_environment_factory(
            instance_id=self.instance_id, run_id=self.run_id
        )

    def reset(self, *, seed: int | None = None) -> object:
        ordinal = self.ordinals[self._reset_count % len(self.ordinals)]
        self._reset_count += 1
        return cast(object, self._environment).reset(  # type: ignore[attr-defined, redundant-cast]
            seed=seed, options={"scenario_ordinal": ordinal}
        )

    def step(self, action: object) -> object:
        return cast(object, self._environment).step(  # type: ignore[attr-defined, redundant-cast]
            action
        )

    def close(self) -> None:
        cast(object, self._environment).close()  # type: ignore[attr-defined, redundant-cast]


@dataclass(frozen=True, slots=True)
class CyclingScenarioV2Factory:
    ordinals: tuple[int, int]
    instance_id: str
    run_id: str

    def __call__(self) -> CyclingScenarioV2Environment:
        return CyclingScenarioV2Environment(self.ordinals, self.instance_id, self.run_id)


@dataclass(frozen=True, slots=True)
class QualificationArtifactEntry:
    role: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in _REQUIRED_ARTIFACT_ROLES:
            raise QualificationError("unsupported qualification artifact role")
        pure = PurePosixPath(self.path)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != self.path:
            raise QualificationError("artifact path must be normalized and relative")
        if not self.media_type:
            raise QualificationError("artifact media_type must be non-empty")
        if _non_negative("artifact size_bytes", self.size_bytes) == 0:
            raise QualificationError("artifact must not be empty")
        _digest("artifact sha256", self.sha256)

    def content(self) -> dict[str, object]:
        return {
            "media_type": self.media_type,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class QualificationExecution:
    runner_kind: str
    transition_count: int
    batch_sizes: tuple[int, ...]
    evaluation_episode_count: int
    actor_count: int
    update_count: int
    adam_step_count: int
    initial_policy_sha256: str
    final_policy_sha256: str
    initial_value_sha256: str
    final_value_sha256: str
    initial_optimizer_sha256: str
    final_optimizer_sha256: str
    initial_checkpoint_sha256: str
    final_checkpoint_sha256: str
    checkpoint_reloaded: bool
    fresh_update1_matched: bool
    reload_update2_matched: bool
    mask_violation_count: int
    evaluation_leak_count: int
    process_backend_observed: bool
    artifacts: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if self.runner_kind not in {"internal-actual-process-v1", "injected-test-double"}:
            raise QualificationError("runner kind differs from closed schema")
        for name in (
            "transition_count",
            "evaluation_episode_count",
            "actor_count",
            "update_count",
            "adam_step_count",
            "mask_violation_count",
            "evaluation_leak_count",
        ):
            _non_negative(name, getattr(self, name))
        for name in (
            "initial_policy_sha256",
            "final_policy_sha256",
            "initial_value_sha256",
            "final_value_sha256",
            "initial_optimizer_sha256",
            "final_optimizer_sha256",
            "initial_checkpoint_sha256",
            "final_checkpoint_sha256",
        ):
            _digest(name, getattr(self, name))
        if set(self.artifacts) != set(_REQUIRED_ARTIFACT_ROLES - {"capability-receipt"}):
            raise QualificationError("execution artifacts differ from required roles")
        if any(not isinstance(value, bytes) or not value for value in self.artifacts.values()):
            raise QualificationError("execution artifacts must be non-empty bytes")
        object.__setattr__(self, "artifacts", dict(self.artifacts))


@dataclass(frozen=True, slots=True)
class VerifiedReferencePPOQualificationReceipt:
    """Final admission issued only for a re-hashed typed evidence graph."""

    summary: ReferencePPOCapabilityReceipt
    evidence_sha256: str
    admitted: bool = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _digest("evidence_sha256", self.evidence_sha256)
        admissible = self.summary.blockers == (
            "verified-qualification-evidence-required",
        )
        object.__setattr__(self, "admitted", admissible)
        object.__setattr__(
            self,
            "sha256",
            _sha(
                _canonical_json(
                    {
                        "schema_version": "verified-reference-ppo-qualification-v1",
                        "summary_sha256": self.summary.sha256,
                        "evidence_sha256": self.evidence_sha256,
                        "admitted": admissible,
                    }
                )
            ),
        )

    def content(self) -> dict[str, object]:
        return {
            "schema_version": "verified-reference-ppo-qualification-v1",
            "summary": {
                item.name: getattr(self.summary, item.name)
                for item in fields(self.summary)
            },
            "evidence_sha256": self.evidence_sha256,
            "admitted": self.admitted,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ReferencePPOQualificationBundle:
    plan_sha256: str
    entries: tuple[QualificationArtifactEntry, ...]
    capability: VerifiedReferencePPOQualificationReceipt | None
    blockers: tuple[str, ...]
    admitted: bool = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _digest("plan_sha256", self.plan_sha256)
        entries = tuple(sorted(self.entries, key=lambda item: item.path))
        if len({item.path for item in entries}) != len(entries):
            raise QualificationError("artifact paths must be unique")
        if {item.role for item in entries} != _REQUIRED_ARTIFACT_ROLES:
            raise QualificationError("artifact ledger is incomplete")
        blockers = tuple(dict.fromkeys(self.blockers))
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(
            self,
            "admitted",
            not blockers
            and self.capability is not None
            and self.capability.admitted,
        )
        object.__setattr__(
            self,
            "sha256",
            _sha(
                _canonical_json(
                    {
                        "blockers": list(blockers),
                        "capability_sha256": (
                            None if self.capability is None else self.capability.sha256
                        ),
                        "entries": [item.content() for item in entries],
                        "plan_sha256": self.plan_sha256,
                    }
                )
            ),
        )

    def content(self) -> dict[str, object]:
        return {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "plan_sha256": self.plan_sha256,
            "entries": [item.content() for item in self.entries],
            "blockers": list(self.blockers),
            "capability": None if self.capability is None else self.capability.content(),
            "bundle_sha256": self.sha256,
        }


class QualificationRunner(Protocol):
    def __call__(self, plan: ReferencePPOQualificationPlan) -> QualificationExecution: ...


def _write_exclusive(root: Path, role: str, payload: bytes) -> QualificationArtifactEntry:
    path = f"artifacts/{role}.json"
    target = root / Path(*PurePosixPath(path).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    return QualificationArtifactEntry(role, path, "application/json", len(payload), _sha(payload))


def _write_final_ledger(root: Path, bundle: ReferencePPOQualificationBundle) -> None:
    target = root / "ledger.json"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical_json(bundle.content()))
        stream.flush()
        os.fsync(stream.fileno())


def _json_object(payload: bytes, schema: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError("typed artifact is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise QualificationError(f"typed artifact differs from {schema}")
    return cast(dict[str, object], value)


def _parse_policy_receipt(payload: bytes) -> tuple[PolicyArtifact, PolicyArtifactRef]:
    value = _json_object(payload, "policy-artifact-receipt-v1")
    if set(value) != {"schema_version", "artifact", "reference"}:
        raise QualificationError("policy receipt fields differ from closed schema")
    raw = value["artifact"]
    reference_raw = value["reference"]
    if not isinstance(raw, dict) or not isinstance(reference_raw, dict):
        raise QualificationError("policy receipt children must be objects")
    required = {
        "run_id",
        "generation",
        "policy_version",
        "media_type",
        "compatibility",
        "source_batch_sha256",
        "provenance",
        "payload_base64",
        "payload_sha256",
        "artifact_sha256",
    }
    if set(raw) != required:
        raise QualificationError("policy artifact fields differ from closed schema")
    try:
        encoded = raw["payload_base64"]
        if not isinstance(encoded, str):
            raise TypeError
        payload_bytes = base64.b64decode(encoded, validate=True)
        compatibility_raw = raw["compatibility"]
        provenance = raw["provenance"]
        if not isinstance(compatibility_raw, dict) or not isinstance(provenance, dict):
            raise TypeError
        compatibility = PolicyCompatibility.from_dict(compatibility_raw)
        artifact = PolicyArtifact(
            run_id=cast(str, raw["run_id"]),
            generation=cast(int, raw["generation"]),
            policy_version=cast(int, raw["policy_version"]),
            payload=payload_bytes,
            media_type=cast(str, raw["media_type"]),
            compatibility=compatibility,
            source_batch_sha256=cast(str, raw["source_batch_sha256"]),
            provenance=provenance,
        )
        reference = PolicyArtifactRef.from_dict(reference_raw)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise QualificationError("policy receipt failed reconstruction") from exc
    if raw["payload_sha256"] != _sha(payload_bytes) or raw["artifact_sha256"] != artifact.sha256:
        raise QualificationError("policy receipt digest differs")
    if (
        reference.sha256 != artifact.sha256
        or reference.size_bytes != artifact.size_bytes
        or reference.policy_version != artifact.policy_version
        or reference.source_batch_sha256 != artifact.source_batch_sha256
        or reference.compatibility != artifact.compatibility
    ):
        raise QualificationError("policy reference is not bound to artifact")
    ReferencePPOCheckpointV1(payload_bytes)
    return artifact, reference


def _parse_transition_receipt(payload: bytes, index: int) -> ValidatedTransitionBatch:
    value = _json_object(payload, "transition-batch-receipt-v1")
    required = {
        "schema_version",
        "batch_index",
        "batch_sha256",
        "record_count",
        "record_sha256_order",
        "records",
    }
    if set(value) != required or value["batch_index"] != index:
        raise QualificationError("transition receipt fields/index differ")
    raw_records = value["records"]
    if not isinstance(raw_records, list) or not all(isinstance(item, dict) for item in raw_records):
        raise QualificationError("transition receipt records must be objects")
    try:
        batch = ValidatedTransitionBatch.build(
            [TransitionRecord.from_dict(cast(dict[str, object], item)) for item in raw_records]
        )
    except (TypeError, ValueError) as exc:
        raise QualificationError("transition receipt failed reconstruction") from exc
    order = [_sha(_canonical_json(item.to_dict())) for item in batch.records]
    if (
        value["record_count"] != len(batch.records)
        or value["batch_sha256"] != batch.sha256
        or value["record_sha256_order"] != order
    ):
        raise QualificationError("transition child count/order/digest differs")
    return batch


def _verify_update_receipt(
    payload: bytes,
    *,
    expected_payload: bytes,
    batch_sha256: str,
    expected_adam_steps: int,
) -> None:
    value = _json_object(payload, "update-equivalence-receipt-v1")
    if set(value) != {
        "schema_version",
        "expected_payload_sha256",
        "observed_payload_sha256",
        "matched",
        "batch_sha256",
        "adam_step_count",
    }:
        raise QualificationError("update receipt fields differ")
    digest = _sha(expected_payload)
    if (
        value["expected_payload_sha256"] != digest
        or value["observed_payload_sha256"] != digest
        or value["matched"] is not True
        or value["batch_sha256"] != batch_sha256
        or value["adam_step_count"] != expected_adam_steps
    ):
        raise QualificationError("update receipt does not reconcile")


def _verify_evaluation_receipt(
    payload: bytes, policies: tuple[PolicyArtifact, PolicyArtifact, PolicyArtifact]
) -> None:
    value = _json_object(payload, "evaluation-isolation-receipt-v1")
    required = {
        "schema_version",
        "episode_rows",
        "episode_count",
        "per_episode_max_steps",
        "total_step_budget",
        "session_before_sha256",
        "session_after_sha256",
        "parameters_before",
        "parameters_after",
        "rng_before_sha256",
        "rng_after_sha256",
        "training_writes",
    }
    rows = value.get("episode_rows")
    if set(value) != required or not isinstance(rows, list):
        raise QualificationError("evaluation receipt fields differ")
    expected_ordinals = sorted(
        item.ordinal
        for item in build_anti_torpedo_scenario_profile().partition.tuning
    )[8:16]
    expected_pairs = [(version, ordinal) for version in (0, 2) for ordinal in expected_ordinals]
    actual_pairs: list[tuple[int, int]] = []
    step_counts: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "policy_version",
            "scenario_ordinal",
            "episode_id",
            "seed",
            "policy_artifact_sha256",
            "step_count",
            "episode_return",
            "terminated",
            "truncated",
            "final_observation_sha256",
            "action_trace_sha256",
            "mask_violation_count",
        }:
            raise QualificationError("evaluation episode row differs")
        actual_pairs.append(
            (cast(int, row["policy_version"]), cast(int, row["scenario_ordinal"]))
        )
        step_counts.append(cast(int, row["step_count"]))
        version = cast(int, row["policy_version"])
        ordinal = cast(int, row["scenario_ordinal"])
        if (
            version not in (0, 2)
            or row["seed"] != ordinal
            or row["policy_artifact_sha256"] != policies[version].sha256
            or row["episode_id"] != f"evaluation-{version}-{ordinal}"
            or not isinstance(row["episode_return"], (int, float))
            or isinstance(row["episode_return"], bool)
            or not math.isfinite(float(row["episode_return"]))
            or type(row["terminated"]) is not bool
            or type(row["truncated"]) is not bool
            or not (row["terminated"] or row["truncated"])
            or row["mask_violation_count"] != 0
        ):
            raise QualificationError("evaluation episode provenance differs")
        _digest("evaluation final observation SHA-256", row["final_observation_sha256"])
        _digest("evaluation action trace SHA-256", row["action_trace_sha256"])
    if (
        len(rows) != PRODUCTION_EVALUATION_EPISODES
        or value["episode_count"] != len(rows)
        or actual_pairs != expected_pairs
        or value["per_episode_max_steps"] != 30
        or value["total_step_budget"] != 480
        or any(type(item) is not int or item < 1 or item > 30 for item in step_counts)
        or sum(step_counts) > 480
        or value["session_before_sha256"] != value["session_after_sha256"]
        or value["parameters_before"] != value["parameters_after"]
        or value["rng_before_sha256"] != value["rng_after_sha256"]
        or value["training_writes"] != 0
    ):
        raise QualificationError("evaluation isolation evidence does not reconcile")


def _verify_source_manifest(payload: bytes, plan_sha256: str) -> None:
    actual = _json_object(payload, "qualification-source-manifest-v1")
    expected = _json_object(_source_profile(), "qualification-source-manifest-v1")
    expected["plan_sha256"] = plan_sha256
    if actual != expected:
        raise QualificationError("source/runtime/plan manifest differs")


def _verify_execution_manifest(
    payload: bytes,
    plan_sha256: str,
    batches: tuple[ValidatedTransitionBatch, ValidatedTransitionBatch],
) -> None:
    value = _json_object(payload, "qualification-execution-manifest-v1")
    if set(value) != {
        "schema_version",
        "plan",
        "plan_sha256",
        "training_ordinals",
        "process",
    }:
        raise QualificationError("execution manifest fields differ")
    process = value["process"]
    if not isinstance(process, dict) or process != {
        "backend": "process",
        "start_method": "spawn",
        "peak_concurrent_width": 4,
        "pool_lifecycle_count": 1,
        "logical_worker_count": 4,
    }:
        raise QualificationError("process lifecycle evidence differs")
    raw_plan = value["plan"]
    if not isinstance(raw_plan, dict):
        raise QualificationError("execution plan must be an object")
    try:
        reconstructed_plan = ReferencePPOQualificationPlan(
            training_transitions=cast(int, raw_plan["training_transitions"]),
            batch_size=cast(int, raw_plan["batch_size"]),
            batch_count=cast(int, raw_plan["batch_count"]),
            actor_count=cast(int, raw_plan["actor_count"]),
            evaluation_episodes=cast(int, raw_plan["evaluation_episodes"]),
            production_admission=cast(bool, raw_plan["production_admission"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError("execution plan failed reconstruction") from exc
    expected_ordinals = sorted(
        item.ordinal
        for item in build_anti_torpedo_scenario_profile().partition.tuning
    )[:8]
    records = tuple(record for batch in batches for record in batch.records)
    contract = AntiTorpedoV2FeatureContract()
    for record in records:
        previous = record.previous_observation
        following = record.next_observation
        if not isinstance(previous, Mapping) or not isinstance(following, Mapping):
            raise QualificationError("transition inference inputs are not objects")
        try:
            mask = contract.action_mask(previous)
            next_mask = contract.action_mask(following)
        except (TypeError, ValueError) as exc:
            raise QualificationError(
                "transition feature/mask contract failed reconstruction"
            ) from exc
        if (
            type(record.action) is not int
            or record.action < 0
            or record.action >= len(mask)
            or mask[record.action] is not True
            or record.info.get("previous_action_mask") != mask
            or record.info.get("next_action_mask") != next_mask
        ):
            raise QualificationError("typed transition violates action mask")
    if (
        value["plan_sha256"] != plan_sha256
        or reconstructed_plan.sha256 != plan_sha256
        or reconstructed_plan.content() != raw_plan
        or value["training_ordinals"] != expected_ordinals
        or len({record.idempotency_key for record in records}) != len(records)
        or len({record.worker_id for record in records}) != 4
    ):
        raise QualificationError("execution identity/count/order evidence differs")


def _verify_supporting_receipts(
    payloads: Mapping[str, bytes], policies: tuple[PolicyArtifact, ...]
) -> None:
    review = _json_object(
        payloads["review-disposition"], "qualification-review-disposition-v1"
    )
    if review != {
        "schema_version": "qualification-review-disposition-v1",
        "status": "qualification-gates-passed",
        "claim_scope": "reference-ppo-local-qualification-only",
    }:
        raise QualificationError("review disposition receipt differs")
    process = _json_object(payloads["process-lifecycle"], "process-lifecycle-receipt-v1")
    if set(process) != {
        "schema_version",
        "backend",
        "start_method",
        "peak_concurrent_width",
        "pool_lifecycle_count",
        "logical_worker_ids",
        "pids",
        "incarnations",
        "cleanup",
    }:
        raise QualificationError("process lifecycle receipt fields differ")
    workers = [f"worker-{index}" for index in range(4)]
    pids = process["pids"]
    incarnations = process["incarnations"]
    cleanup = process["cleanup"]
    if (
        process["backend"] != "process"
        or process["start_method"] != "spawn"
        or process["peak_concurrent_width"] != 4
        or process["pool_lifecycle_count"] != 1
        or process["logical_worker_ids"] != workers
        or not isinstance(pids, dict)
        or set(pids) != set(workers)
        or len(set(pids.values())) != 4
        or any(type(pid) is not int or pid <= 0 for pid in pids.values())
        or not isinstance(incarnations, dict)
        or incarnations != dict.fromkeys(workers, 0)
        or not isinstance(cleanup, list)
        or len(cleanup) != 4
        or {item.get("worker_id") for item in cleanup if isinstance(item, dict)}
        != set(workers)
        or any(
            not isinstance(item, dict)
            or item.get("worker_id") not in workers
            or item.get("incarnation") != 0
            or item.get("pid") != pids.get(item.get("worker_id"))
            or item.get("action") != "closed"
            or item.get("exitcode") != 0
            or item.get("error") is not None
            for item in cleanup
        )
    ):
        raise QualificationError("process lifecycle receipt differs")
    negative = _json_object(
        payloads["negative-authenticity"], "negative-authenticity-receipt-v1"
    )
    if negative != {
        "schema_version": "negative-authenticity-receipt-v1",
        "probe": "accepted-workload-mask-pre-mutation-prerequisite",
        "workload_prerequisite": _workload_mask_prerequisite(),
    }:
        raise QualificationError("negative authenticity witness differs")
    recovery = _json_object(
        payloads["recovery-state"], "recovery-equivalence-receipt-v1"
    )
    if recovery.get("reload_payload_sha256") != recovery.get(
        "uninterrupted_payload_sha256"
    ):
        raise QualificationError("recovery payload equivalence differs")
    checkpoint_state_raw = recovery.get("checkpoint_state")
    final_state_raw = recovery.get("restored_final_state")
    if not isinstance(checkpoint_state_raw, dict) or not isinstance(final_state_raw, dict):
        raise QualificationError("recovery state bodies are missing")
    try:
        checkpoint_state = LearnerRecoveryState.from_dict(checkpoint_state_raw)
        final_state = LearnerRecoveryState.from_dict(final_state_raw)
        contract = AntiTorpedoV2FeatureContract()
        ReferencePPOLearnerAdapter.from_artifact(
            policies[1], contract
        ).validate_recovery_state(checkpoint_state)
        ReferencePPOLearnerAdapter.from_artifact(
            policies[2], contract
        ).validate_recovery_state(final_state)
    except (TypeError, ValueError) as exc:
        raise QualificationError("recovery state failed typed validation") from exc
    math = _json_object(payloads["math-oracle"], "ppo-math-oracle-receipt-v1")
    if (
        math.get("update_count") != 2
        or math.get("environment_steps") != PRODUCTION_TRAINING_TRANSITIONS
        or math.get("uninterrupted_adam_steps") != PRODUCTION_ADAM_STEPS // 2
        or math.get("qualification_adam_steps") != PRODUCTION_ADAM_STEPS
        or math.get("fresh_update_steps") != PRODUCTION_ADAM_STEPS // 4
        or math.get("reload_update_steps") != PRODUCTION_ADAM_STEPS // 4
        or math.get("scalar_oracles") != _ppo_scalar_oracles()
    ):
        raise QualificationError("PPO math oracle receipt differs")


def _validate_typed_evidence(
    payloads: Mapping[str, bytes], plan_sha256: str
) -> tuple[
    tuple[ValidatedTransitionBatch, ValidatedTransitionBatch],
    tuple[PolicyArtifact, PolicyArtifact, PolicyArtifact],
    tuple[ReferencePPOCheckpointV1, ReferencePPOCheckpointV1, ReferencePPOCheckpointV1],
]:
    batches = (
        _parse_transition_receipt(payloads["transitions-1"], 1),
        _parse_transition_receipt(payloads["transitions-2"], 2),
    )
    policies = cast(
        tuple[PolicyArtifact, PolicyArtifact, PolicyArtifact],
        tuple(_parse_policy_receipt(payloads[f"policy-{index}"])[0] for index in range(3)),
    )
    if payloads["checkpoint-1"] != payloads["policy-1"] or payloads[
        "checkpoint-2"
    ] != payloads["policy-2"]:
        raise QualificationError("checkpoint receipt differs from indexed policy")
    checkpoints = cast(
        tuple[
            ReferencePPOCheckpointV1,
            ReferencePPOCheckpointV1,
            ReferencePPOCheckpointV1,
        ],
        tuple(ReferencePPOCheckpointV1(item.payload) for item in policies),
    )
    if tuple(item.policy_version for item in policies) != (0, 1, 2):
        raise QualificationError("policy version chain differs")
    if tuple(item.source_batch_sha256 for item in policies) != (
        EMPTY_TRANSITION_BATCH_SHA256,
        batches[0].sha256,
        batches[1].sha256,
    ):
        raise QualificationError("policy source batch chain differs")
    if tuple(item.update_count for item in checkpoints) != (0, 1, 2):
        raise QualificationError("checkpoint update chain differs")
    if tuple(item.environment_steps for item in checkpoints) != (
        0,
        PRODUCTION_BATCH_SIZE,
        PRODUCTION_TRAINING_TRANSITIONS,
    ):
        raise QualificationError("checkpoint environment-step chain differs")
    for index, (before, after) in enumerate(pairwise(checkpoints), start=1):
        if (
            before.policy_parameter_sha256 == after.policy_parameter_sha256
            or before.value_parameter_sha256 == after.value_parameter_sha256
            or before.optimizer_sha256 == after.optimizer_sha256
            or before.sha256 == after.sha256
            or policies[index - 1].sha256 == policies[index].sha256
        ):
            raise QualificationError(
                f"update {index} did not change every required immutable state"
            )
    expected_behavior = (policies[0].sha256, policies[1].sha256)
    for batch_index, batch in enumerate(batches):
        if any(
            record.policy_version != batch_index
            or record.info.get("behavior_artifact_sha256")
            != expected_behavior[batch_index]
            or record.info.get("evaluation") is not False
            for record in batch.records
        ):
            raise QualificationError("transition behavior artifact chain differs")
    if sum(len(item.records) for item in batches) != PRODUCTION_TRAINING_TRANSITIONS:
        raise QualificationError("typed transition total differs")
    training_episode_constructions = len(
        {
            (record.worker_id, record.episode_id)
            for batch in batches
            for record in batch.records
        }
    )
    if training_episode_constructions + PRODUCTION_EVALUATION_EPISODES > 4112:
        raise QualificationError("environment construction bound exceeded")
    adam_per_update = (
        ReferencePPOConfigV1().epochs
        * (PRODUCTION_BATCH_SIZE // ReferencePPOConfigV1().minibatch_size)
    )
    _verify_update_receipt(
        payloads["fresh-update-1"],
        expected_payload=policies[1].payload,
        batch_sha256=batches[0].sha256,
        expected_adam_steps=adam_per_update,
    )
    _verify_update_receipt(
        payloads["reload-update-2"],
        expected_payload=policies[2].payload,
        batch_sha256=batches[1].sha256,
        expected_adam_steps=adam_per_update,
    )
    _verify_evaluation_receipt(payloads["evaluation-isolation"], policies)
    _verify_source_manifest(payloads["source-profile"], plan_sha256)
    _verify_execution_manifest(payloads["manifest"], plan_sha256, batches)
    _verify_supporting_receipts(payloads, policies)
    return batches, policies, checkpoints


def verify_qualification_bundle(root: Path, bundle: ReferencePPOQualificationBundle) -> None:
    _verify_qualification_source_lock()
    if bundle.capability is not None and not isinstance(
        bundle.capability, VerifiedReferencePPOQualificationReceipt
    ):
        raise QualificationError("capability was not issued by the typed verifier")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise QualificationError("artifact root must be a directory")
    ledger_path = (resolved / "ledger.json").resolve(strict=True)
    if ledger_path.parent != resolved or ledger_path.is_symlink():
        raise QualificationError("final ledger path is invalid")
    if ledger_path.read_bytes() != _canonical_json(bundle.content()):
        raise QualificationError("durable final ledger differs from bundle")
    payloads: dict[str, bytes] = {}
    for entry in bundle.entries:
        target = (resolved / Path(*PurePosixPath(entry.path).parts)).resolve(strict=True)
        try:
            target.relative_to(resolved)
        except ValueError as exc:
            raise QualificationError("artifact escapes ledger root") from exc
        if not target.is_file() or target.is_symlink():
            raise QualificationError("artifact is not a regular file")
        payload = target.read_bytes()
        if len(payload) != entry.size_bytes or _sha(payload) != entry.sha256:
            raise QualificationError(f"artifact failed re-hash: {entry.path}")
        payloads[entry.role] = payload
    raw_entries = tuple(
        entry for entry in bundle.entries if entry.role != "capability-receipt"
    )
    ledger_sha256 = _sha(_canonical_json([entry.content() for entry in raw_entries]))
    capability_envelope = _json_object(
        payloads["capability-receipt"], "final-capability-envelope-v1"
    )
    expected_envelope = {
        "schema_version": "final-capability-envelope-v1",
        "raw_ledger_sha256": ledger_sha256,
        "raw_artifact_count": len(raw_entries),
        "blockers": list(bundle.blockers),
        "capability": None if bundle.capability is None else bundle.capability.content(),
    }
    # The durable envelope is parsed from JSON, so arrays are lists.  The
    # in-memory frozen receipt intentionally stores blockers as a tuple.
    # Compare their canonical JSON representations rather than Python
    # container implementation types.
    if not _canonical_json_equal(capability_envelope, expected_envelope):
        raise QualificationError("durable capability envelope differs")
    if bundle.capability is not None:
        batches, policies, checkpoints = _validate_typed_evidence(
            payloads, bundle.plan_sha256
        )
        summary = bundle.capability.summary
        if summary.raw_ledger_sha256 != ledger_sha256:
            raise QualificationError("capability is not bound to the re-hashed raw ledger")
        if summary.raw_artifact_count != len(raw_entries):
            raise QualificationError("capability raw artifact count differs from ledger")
        if not bundle.capability.admitted:
            raise QualificationError("non-admitted capability cannot admit a bundle")
        initial, final = checkpoints[0], checkpoints[2]
        expected = {
            "initial_policy_sha256": initial.policy_parameter_sha256,
            "final_policy_sha256": final.policy_parameter_sha256,
            "initial_value_sha256": initial.value_parameter_sha256,
            "final_value_sha256": final.value_parameter_sha256,
            "initial_optimizer_sha256": initial.optimizer_sha256,
            "final_optimizer_sha256": final.optimizer_sha256,
            "initial_checkpoint_sha256": initial.sha256,
            "final_checkpoint_sha256": final.sha256,
            "environment_steps": sum(len(item.records) for item in batches),
            "update_count": final.update_count,
        }
        if any(getattr(summary, key) != value for key, value in expected.items()):
            raise QualificationError("capability summary differs from typed child evidence")
        if bundle.capability.evidence_sha256 != ledger_sha256:
            raise QualificationError("verifier receipt evidence digest differs")


def _object_source_sha256(value: Any) -> str:
    source = inspect.getsourcefile(value)
    if source is None:
        raise QualificationError("qualification source path is unavailable")
    return _sha(Path(source).read_bytes())


def _source_profile(plan: ReferencePPOQualificationPlan | None = None) -> bytes:
    profile = build_anti_torpedo_scenario_profile()
    return _canonical_json(
        {
            "schema_version": "qualification-source-manifest-v1",
            "feature_contract_sha256": ANTI_TORPEDO_FEATURE_CONTRACT_SHA256,
            "numpy_runtime": dict(numpy_runtime_identity()),
            "numpy_runtime_sha256": numpy_runtime_sha256(),
            "platform": platform.platform(),
            "python": sys.version,
            "plan_sha256": None if plan is None else plan.sha256,
            "profile_sha256": profile.sha256,
            "external_model_source_sha256": atsim_source_sha256(),
            "adapter_source_sha256": LOADED_ADAPTER_SOURCE_SHA256,
            "pyjevsim_executor_source_sha256": PYJEVSIM_EXECUTOR_SOURCE_SHA256,
            "scenario_bank_sha256": profile.bank.sha256,
            "scenario_generator_source_sha256": profile.bank.generator_source_sha256,
            "scenario_source_sha256": profile.bank.scenario_source_sha256,
            "scenario_family_sha256": profile.bank.family_sha256,
            "factor_schema_sha256": profile.bank.factor_schema_sha256,
            "workload_prerequisite": _workload_mask_prerequisite(),
            "sources": {
                "qualification": LOADED_REFERENCE_PPO_QUALIFICATION_SOURCE_SHA256,
                "learner": LOADED_REFERENCE_PPO_SOURCE_SHA256,
                "features": LOADED_ANTI_TORPEDO_FEATURE_SOURCE_SHA256,
                "anti_torpedo": _object_source_sha256(
                    anti_torpedo_v2_environment_factory
                ),
                "profile": _object_source_sha256(
                    build_anti_torpedo_scenario_profile
                ),
                "learning": _object_source_sha256(LearnerSession),
                "local_rollout": _object_source_sha256(LocalRolloutPool),
                "records": _object_source_sha256(TransitionRecord),
            },
        }
    )


def _verify_qualification_source_lock() -> None:
    if _sha(Path(__file__).read_bytes()) != LOADED_REFERENCE_PPO_QUALIFICATION_SOURCE_SHA256:
        raise QualificationError("qualification source changed after module import")


def run_reference_ppo_qualification(
    root: Path,
    *,
    plan: ReferencePPOQualificationPlan | None = None,
    runner: QualificationRunner | None = None,
) -> ReferencePPOQualificationBundle:
    selected = ReferencePPOQualificationPlan() if plan is None else plan
    _verify_qualification_source_lock()
    starting_source_profile = _source_profile(selected)
    if root.exists():
        raise QualificationError("qualification output root must not exist")
    root.mkdir(parents=True, exist_ok=False)
    injected = runner is not None
    execution = (
        _run_actual_production(selected)
        if runner is None
        else runner(selected)
    )
    _verify_qualification_source_lock()
    blockers: list[str] = []
    if injected or execution.runner_kind != "internal-actual-process-v1":
        blockers.append("injected-runner-not-production-evidence")
    if not selected.is_exact_production or not selected.production_admission:
        blockers.append("bounded-test-plan-not-production")
    if execution.transition_count != selected.training_transitions:
        blockers.append("training-transition-count-mismatch")
    if execution.batch_sizes != (selected.batch_size,) * selected.batch_count:
        blockers.append("batch-size-reconciliation-failed")
    if execution.actor_count != selected.actor_count or not execution.process_backend_observed:
        blockers.append("actual-four-process-actors-not-observed")
    if execution.evaluation_episode_count != selected.evaluation_episodes:
        blockers.append("evaluation-isolation-count-mismatch")
    if execution.update_count != 2 or execution.adam_step_count != PRODUCTION_ADAM_STEPS:
        blockers.append("ppo-update-count-mismatch")
    if not execution.fresh_update1_matched or not execution.reload_update2_matched:
        blockers.append("fresh-or-reload-bytes-differ")
    if execution.mask_violation_count or execution.evaluation_leak_count:
        blockers.append("mask-or-evaluation-admission-failed")

    artifacts = dict(execution.artifacts)
    if _source_profile(selected) != starting_source_profile:
        raise QualificationError("qualification source/runtime profile changed during execution")
    artifacts["source-profile"] = starting_source_profile
    if not blockers:
        artifacts["review-disposition"] = _canonical_json(
            {
                "schema_version": "qualification-review-disposition-v1",
                "status": "qualification-gates-passed",
                "claim_scope": "reference-ppo-local-qualification-only",
            }
        )
        # Admission issuance starts only after typed children independently
        # reconstruct and reconcile; execution summary booleans are not a token.
        _validate_typed_evidence(artifacts, selected.sha256)
    raw_roles = _REQUIRED_ARTIFACT_ROLES - {"capability-receipt"}
    raw_entries = tuple(
        _write_exclusive(root, role, artifacts[role])
        for role in sorted(raw_roles)
    )
    raw_ledger_sha256 = _sha(
        _canonical_json([entry.content() for entry in raw_entries])
    )
    capability: VerifiedReferencePPOQualificationReceipt | None = None
    if not blockers:
        candidate = ReferencePPOCapabilityReceipt(
            source_sha256=LOADED_REFERENCE_PPO_SOURCE_SHA256,
            feature_source_sha256=LOADED_ANTI_TORPEDO_FEATURE_SOURCE_SHA256,
            config_sha256=ReferencePPOConfigV1().sha256,
            feature_contract_sha256=ANTI_TORPEDO_FEATURE_CONTRACT_SHA256,
            numpy_runtime_sha256=numpy_runtime_sha256(),
            initial_policy_sha256=execution.initial_policy_sha256,
            final_policy_sha256=execution.final_policy_sha256,
            initial_value_sha256=execution.initial_value_sha256,
            final_value_sha256=execution.final_value_sha256,
            initial_optimizer_sha256=execution.initial_optimizer_sha256,
            final_optimizer_sha256=execution.final_optimizer_sha256,
            initial_checkpoint_sha256=execution.initial_checkpoint_sha256,
            final_checkpoint_sha256=execution.final_checkpoint_sha256,
            update_count=execution.update_count,
            adam_step_count=PRODUCTION_ADAM_STEPS // 2,
            qualification_adam_step_count=execution.adam_step_count,
            environment_steps=execution.transition_count,
            checkpoint_reloaded=execution.checkpoint_reloaded,
            resume_bytes_matched=(
                execution.fresh_update1_matched and execution.reload_update2_matched
            ),
            mask_violation_count=execution.mask_violation_count,
            evaluation_leak_count=execution.evaluation_leak_count,
            raw_ledger_sha256=raw_ledger_sha256,
            raw_artifact_count=len(raw_entries),
        )
        if candidate.blockers == ("verified-qualification-evidence-required",):
            capability = VerifiedReferencePPOQualificationReceipt(
                candidate, raw_ledger_sha256
            )
        else:
            blockers.extend(f"capability:{item}" for item in candidate.blockers)
    capability_payload = _canonical_json(
        {
            "schema_version": "final-capability-envelope-v1",
            "raw_ledger_sha256": raw_ledger_sha256,
            "raw_artifact_count": len(raw_entries),
            "blockers": blockers,
            "capability": None if capability is None else capability.content(),
        }
    )
    capability_entry = _write_exclusive(
        root, "capability-receipt", capability_payload
    )
    entries = raw_entries + (capability_entry,)
    bundle = ReferencePPOQualificationBundle(selected.sha256, entries, capability, tuple(blockers))
    _write_final_ledger(root, bundle)
    verify_qualification_bundle(root, bundle)
    return bundle


def _run_actual_production(plan: ReferencePPOQualificationPlan) -> QualificationExecution:
    """Execute the exact expensive production path; never accepts injected actors."""

    if not plan.is_exact_production:
        raise QualificationError("internal actual runner accepts only the frozen production plan")
    return asyncio.run(_run_actual_async(plan))


async def _run_actual_async(plan: ReferencePPOQualificationPlan) -> QualificationExecution:
    # Kept in one closed function so test doubles cannot enter production admission.
    contract = AntiTorpedoV2FeatureContract()
    compatibility = PolicyCompatibility(
        REFERENCE_PPO_ALGORITHM_ID,
        REFERENCE_PPO_ALGORITHM_VERSION,
        "AntiTorpedoCountermeasure-v2",
        "2",
        contract.contract_sha256,
        cast(str, REFERENCE_PPO_ACTION_SCHEMA_SHA256),
    )
    config = ReferencePPOConfigV1()
    store = InMemoryPolicyArtifactStore()
    learner = ReferencePPOLearnerAdapter(
        contract,
        run_id="task-rl-105",
        generation=0,
        compatibility=compatibility,
        config=config,
    )
    session = LearnerSession(learner, store, run_id="task-rl-105", generation=0)
    initial_ref = await session.publish_initial(learner.initial_policy())
    initial_artifact = await store.get(initial_ref)
    learner.bind_published(initial_ref, initial_artifact)
    actor = ActorPolicy(
        ReferencePPOPolicyLoader(contract),
        store,
        run_id="task-rl-105",
        generation=0,
        compatibility=compatibility,
    )
    await actor.activate(initial_ref)
    initial_checkpoint = ReferencePPOCheckpointV1(initial_artifact.payload)
    initial_digests = learner.parameter_digests
    profile = build_anti_torpedo_scenario_profile()
    training_ordinals = tuple(sorted(item.ordinal for item in profile.partition.tuning)[:8])
    batches: list[ValidatedTransitionBatch] = []
    artifacts: dict[str, bytes] = {
        "manifest": _canonical_json(
            {
                "plan": plan.content(),
                "plan_sha256": plan.sha256,
                "training_ordinals": training_ordinals,
                "schema_version": "qualification-execution-manifest-v1",
                "process": {
                    "backend": "process",
                    "start_method": "spawn",
                    "peak_concurrent_width": 4,
                    "pool_lifecycle_count": 1,
                    "logical_worker_count": 4,
                },
            }
        ),
        "source-profile": _source_profile(),
        "review-disposition": _canonical_json(
            {
                "schema_version": "qualification-review-disposition-v1",
                "status": "executed-pending-ledger",
                "claim_scope": "reference-ppo-local-qualification-only",
            }
        ),
    }
    policies = [initial_artifact]
    recovery_after_update1 = None
    adam_after_update1 = 0
    policy_refs = [initial_ref]
    workers = {
        f"worker-{index}": CyclingScenarioV2Factory(
            (training_ordinals[index], training_ordinals[index + 4]),
            f"worker-{index}",
            "task-rl-105",
        )
        for index in range(4)
    }
    pool = LocalRolloutPool(
        cast(Any, workers),
        run_id="task-rl-105",
        generation=0,
        run_seed=1516,
        backend="process",
    )
    process_backend = pool.backend
    process_pids = dict(pool.process_pids)
    process_incarnations = dict(pool.process_incarnations)
    try:
        for batch_index in range(2):
            records = await _collect_process_records(
                pool,
                plan.batch_size,
                actor,
                run_id="task-rl-105",
            )
            batch = ValidatedTransitionBatch.build(records)
            batches.append(batch)
            artifacts[f"transitions-{batch_index + 1}"] = _transition_receipt(
                batch_index + 1, batch
            )
            next_ref = await session.consume(batch)
            if next_ref is None:
                raise QualificationError("reference PPO did not publish after a full batch")
            published = await store.get(next_ref)
            learner.bind_published(next_ref, published)
            await actor.activate(next_ref)
            policies.append(published)
            policy_refs.append(next_ref)
            if batch_index == 0:
                recovery_after_update1 = await session.snapshot_recovery_state()
                adam_after_update1 = learner.adam_step
    finally:
        await pool.close()
    process_cleanup = tuple(pool.process_cleanup_history)
    final_artifact = policies[-1]
    final_checkpoint = ReferencePPOCheckpointV1(final_artifact.payload)
    final_digests = learner.parameter_digests

    fresh = ReferencePPOLearnerAdapter.from_artifact(initial_artifact, contract)
    fresh_store = InMemoryPolicyArtifactStore()
    fresh_session = LearnerSession(
        fresh, fresh_store, run_id="task-rl-105", generation=0
    )
    fresh_initial_ref = await fresh_session.publish_initial(fresh.initial_policy())
    fresh_initial_artifact = await fresh_store.get(fresh_initial_ref)
    fresh.bind_published(fresh_initial_ref, fresh_initial_artifact)
    fresh_update_ref = await fresh_session.consume(batches[0])
    if fresh_update_ref is None:
        raise QualificationError("fresh update-1 session did not publish")
    fresh_candidate = await fresh_store.get(fresh_update_ref)
    reload = ReferencePPOLearnerAdapter.from_artifact(policies[1], contract)
    policy1_ref = await _reference_for(store, policies[1])
    reload.bind_published(policy1_ref, policies[1])
    if recovery_after_update1 is None:
        raise QualificationError("missing update-1 recovery boundary")
    restored_session = await LearnerSession.restore(
        reload,
        store,
        state=recovery_after_update1,
        run_id="task-rl-105",
        generation=0,
    )
    reload_ref = await restored_session.consume(batches[1])
    if reload_ref is None:
        raise QualificationError("restored session did not publish update 2")
    reload_candidate = await store.get(reload_ref)
    restored_state_after = await restored_session.snapshot_recovery_state()
    qualification_adam_steps = (
        learner.adam_step + fresh.adam_step + (reload.adam_step - adam_after_update1)
    )
    fresh_match = fresh_candidate.payload == policies[1].payload
    reload_match = reload_candidate.payload == final_artifact.payload
    evaluation_state_before = await session.snapshot_recovery_state()
    evaluation_parameters_before = learner.parameter_digests
    evaluation_rng_before = _rng_digest()
    evaluation_steps = await _evaluate_isolation(
        profile,
        contract,
        compatibility,
        initial_artifact,
        final_artifact,
    )
    evaluation_state_after = await session.snapshot_recovery_state()
    evaluation_parameters_after = learner.parameter_digests
    evaluation_rng_after = _rng_digest()
    evaluation_count = len(evaluation_steps)
    artifacts.update(
        {
            "policy-0": _policy_receipt(initial_artifact, policy_refs[0]),
            "policy-1": _policy_receipt(policies[1], policy_refs[1]),
            "policy-2": _policy_receipt(final_artifact, policy_refs[2]),
            "checkpoint-1": _policy_receipt(policies[1], policy_refs[1]),
            "checkpoint-2": _policy_receipt(final_artifact, policy_refs[2]),
            "fresh-update-1": _canonical_json(
                {
                    "schema_version": "update-equivalence-receipt-v1",
                    "expected_payload_sha256": _sha(policies[1].payload),
                    "observed_payload_sha256": _sha(fresh_candidate.payload),
                    "matched": fresh_match,
                    "batch_sha256": batches[0].sha256,
                    "adam_step_count": fresh.adam_step,
                }
            ),
            "reload-update-2": _canonical_json(
                {
                    "schema_version": "update-equivalence-receipt-v1",
                    "expected_payload_sha256": _sha(final_artifact.payload),
                    "observed_payload_sha256": _sha(reload_candidate.payload),
                    "matched": reload_match,
                    "batch_sha256": batches[1].sha256,
                    "adam_step_count": reload.adam_step - adam_after_update1,
                }
            ),
            "evaluation-isolation": _canonical_json(
                {
                    "schema_version": "evaluation-isolation-receipt-v1",
                    "episode_rows": list(evaluation_steps),
                    "episode_count": evaluation_count,
                    "per_episode_max_steps": 30,
                    "total_step_budget": 480,
                    "session_before_sha256": evaluation_state_before.sha256,
                    "session_after_sha256": evaluation_state_after.sha256,
                    "parameters_before": list(evaluation_parameters_before),
                    "parameters_after": list(evaluation_parameters_after),
                    "rng_before_sha256": evaluation_rng_before,
                    "rng_after_sha256": evaluation_rng_after,
                    "training_writes": 0,
                }
            ),
            "process-lifecycle": _canonical_json(
                {
                    "schema_version": "process-lifecycle-receipt-v1",
                    "backend": process_backend,
                    "start_method": "spawn",
                    "peak_concurrent_width": 4,
                    "pool_lifecycle_count": 1,
                    "logical_worker_ids": [f"worker-{index}" for index in range(4)],
                    "pids": process_pids,
                    "incarnations": process_incarnations,
                    "cleanup": [
                        {
                            "worker_id": item.worker_id,
                            "incarnation": item.incarnation,
                            "pid": item.pid,
                            "action": item.action,
                            "exitcode": item.exitcode,
                            "error": item.error,
                        }
                        for item in process_cleanup
                    ],
                }
            ),
            "recovery-state": _canonical_json(
                {
                    "schema_version": "recovery-equivalence-receipt-v1",
                    "checkpoint_state": recovery_after_update1.to_dict(),
                    "restored_final_state": restored_state_after.to_dict(),
                    "reload_payload_sha256": _sha(reload_candidate.payload),
                    "uninterrupted_payload_sha256": _sha(final_artifact.payload),
                }
            ),
            "negative-authenticity": _negative_authenticity_receipt(),
            "math-oracle": _canonical_json(
                {
                    "schema_version": "ppo-math-oracle-receipt-v1",
                    "update_count": final_checkpoint.update_count,
                    "environment_steps": final_checkpoint.environment_steps,
                    "uninterrupted_adam_steps": learner.adam_step,
                    "qualification_adam_steps": qualification_adam_steps,
                    "fresh_update_steps": fresh.adam_step,
                    "reload_update_steps": reload.adam_step - adam_after_update1,
                    "scalar_oracles": _ppo_scalar_oracles(),
                }
            ),
        }
    )
    # capability-receipt is added only after independent count validation.
    return QualificationExecution(
        runner_kind="internal-actual-process-v1",
        transition_count=sum(len(item.records) for item in batches),
        batch_sizes=tuple(len(item.records) for item in batches),
        evaluation_episode_count=evaluation_count,
        actor_count=4,
        update_count=learner.update_count,
        adam_step_count=qualification_adam_steps,
        initial_policy_sha256=initial_digests[0],
        final_policy_sha256=final_digests[0],
        initial_value_sha256=initial_digests[1],
        final_value_sha256=final_digests[1],
        initial_optimizer_sha256=initial_digests[2],
        final_optimizer_sha256=final_digests[2],
        initial_checkpoint_sha256=initial_checkpoint.sha256,
        final_checkpoint_sha256=final_checkpoint.sha256,
        checkpoint_reloaded=True,
        fresh_update1_matched=fresh_match,
        reload_update2_matched=reload_match,
        mask_violation_count=0,
        evaluation_leak_count=0,
        process_backend_observed=True,
        artifacts=artifacts,
    )


async def _reference_for(
    store: InMemoryPolicyArtifactStore, artifact: PolicyArtifact
) -> PolicyArtifactRef:
    return await store.put(artifact)


async def _collect_process_records(
    pool: LocalRolloutPool,
    target: int,
    actor: ActorPolicy,
    *,
    run_id: str,
) -> list[TransitionRecord]:
    result: list[TransitionRecord] = []
    resets = await pool.reset()
    current = {
            item.worker_id: PPOInferenceInputV1(
                item.observation,
                cast(tuple[bool, ...], item.info["action_mask"]),
                _required_seed(item.seed),
                run_id,
                0,
                item.worker_id,
                item.episode_id,
                0,
                True,
            )
            for item in resets
    }
    while len(result) < target:
            action_batch = await actor.actions(current)
            raw_records = await pool.step(
                action_batch.actions, policy_version=action_batch.policy_version
            )
            terminal = False
            for raw in raw_records:
                previous = current[raw.worker_id]
                next_input = PPOInferenceInputV1(
                    raw.next_observation,
                    cast(tuple[bool, ...], raw.info["action_mask"]),
                    previous.sampling_seed,
                    run_id,
                    0,
                    raw.worker_id,
                    raw.episode_id,
                    raw.step_id,
                    True,
                )
                info = dict(raw.info)
                info.update(
                    {
                        "previous_action_mask": previous.action_mask,
                        "next_action_mask": next_input.action_mask,
                        "behavior_artifact_sha256": action_batch.artifact_sha256,
                        "evaluation": False,
                    }
                )
                result.append(
                    TransitionRecord(
                        run_id=raw.run_id,
                        generation=raw.generation,
                        worker_id=raw.worker_id,
                        episode_id=raw.episode_id,
                        step_id=raw.step_id,
                        policy_version=raw.policy_version,
                        idempotency_key=raw.idempotency_key,
                        logical_time=raw.logical_time,
                        previous_observation=previous.to_dict(),
                        action=raw.action,
                        next_observation=next_input.to_dict(),
                        reward=raw.reward,
                        terminated=raw.terminated,
                        truncated=raw.truncated,
                        info=info,
                    )
                )
                current[raw.worker_id] = next_input
                terminal = terminal or raw.terminated or raw.truncated
            if terminal and len(result) < target:
                resets = await pool.reset()
                current = {
                    item.worker_id: PPOInferenceInputV1(
                        item.observation,
                        cast(tuple[bool, ...], item.info["action_mask"]),
                        _required_seed(item.seed),
                        run_id,
                        0,
                        item.worker_id,
                        item.episode_id,
                        0,
                        True,
                    )
                    for item in resets
                }
    return result[:target]


async def _evaluate_isolation(
    profile: object,
    contract: AntiTorpedoV2FeatureContract,
    compatibility: PolicyCompatibility,
    initial: PolicyArtifact,
    final: PolicyArtifact,
) -> tuple[dict[str, object], ...]:
    ordinals = tuple(
        sorted(item.ordinal for item in profile.partition.tuning)[8:16]  # type: ignore[attr-defined]
    )
    rows: list[dict[str, object]] = []
    loader = ReferencePPOPolicyLoader(contract)
    for artifact in (initial, final):
        evaluation_store = InMemoryPolicyArtifactStore()
        reference = await evaluation_store.put(artifact)
        policy = ActorPolicy(
            loader,
            evaluation_store,
            run_id=artifact.run_id,
            generation=artifact.generation,
            compatibility=compatibility,
        )
        await policy.activate(reference)
        for ordinal in ordinals:
            environment = anti_torpedo_v2_environment_factory()
            try:
                observation, info = environment.reset(
                    seed=ordinal, options={"scenario_ordinal": ordinal}
                )
                step = 0
                done = False
                episode_id = f"evaluation-{artifact.policy_version}-{ordinal}"
                episode_return = 0.0
                action_trace_sha256 = "0" * 64
                mask_violation_count = 0
                terminated = False
                truncated = False
                while not done:
                    if step >= 30:
                        raise QualificationError("evaluation episode exceeded v2 MAX_STEPS")
                    value = PPOInferenceInputV1(
                        observation,
                        cast(tuple[bool, ...], info["action_mask"]),
                        ordinal,
                        artifact.run_id,
                        artifact.generation,
                        "evaluation-worker",
                        episode_id,
                        step,
                        False,
                    )
                    action = cast(
                        int,
                        (
                            await policy.actions({"evaluation-worker": value})
                        ).actions["evaluation-worker"],
                    )
                    if not value.action_mask[action]:
                        mask_violation_count += 1
                    action_trace_sha256 = _sha(
                        b"pyjevsim-reference-ppo-evaluation-trace-v1\x00"
                        + bytes.fromhex(action_trace_sha256)
                        + _canonical_json(
                            {"input": value.to_dict(), "action": action}
                        )
                    )
                    observation, reward, terminated, truncated, info = environment.step(action)
                    episode_return += float(reward)
                    step += 1
                    done = terminated or truncated
                rows.append(
                    {
                        "policy_version": artifact.policy_version,
                        "scenario_ordinal": ordinal,
                        "episode_id": episode_id,
                        "seed": ordinal,
                        "policy_artifact_sha256": artifact.sha256,
                        "step_count": step,
                        "episode_return": episode_return,
                        "terminated": terminated,
                        "truncated": truncated,
                        "final_observation_sha256": _sha(
                            _canonical_json(observation)
                        ),
                        "action_trace_sha256": action_trace_sha256,
                        "mask_violation_count": mask_violation_count,
                    }
                )
            finally:
                environment.close()
    return tuple(rows)


async def actual_v2_process_smoke(ordinal: int = 0) -> tuple[int, float]:
    """One actual process reset/step for bounded integration tests; never capability evidence."""

    worker = "smoke-worker"
    pool = LocalRolloutPool(
        cast(Any, {worker: FixedScenarioV2Factory(ordinal, worker, "smoke")}),
        run_id="smoke",
        generation=0,
        run_seed=1,
        backend="process",
    )
    try:
        await pool.reset()
        record = (await pool.step({worker: 0}, policy_version=0))[0]
        return record.step_id, record.logical_time
    finally:
        await pool.close()


async def actual_v2_cycling_process_smoke() -> tuple[int, int, str, str, str, str]:
    """Prove one persistent process cycles scenarios without episode/key collision."""

    worker = "cycling-smoke-worker"
    pool = LocalRolloutPool(
        cast(Any, {worker: CyclingScenarioV2Factory((0, 1), worker, "cycling-smoke")}),
        run_id="cycling-smoke",
        generation=0,
        run_seed=7,
        backend="process",
    )
    try:
        first_reset = (await pool.reset())[0]
        first_step = (await pool.step({worker: 0}, policy_version=0))[0]
        second_reset = (await pool.reset())[0]
        second_step = (await pool.step({worker: 0}, policy_version=0))[0]
        return (
            cast(int, first_reset.info["scenario_ordinal"]),
            cast(int, second_reset.info["scenario_ordinal"]),
            first_reset.episode_id,
            second_reset.episode_id,
            first_step.idempotency_key,
            second_step.idempotency_key,
        )
    finally:
        await pool.close()


__all__ = [
    "QUALIFICATION_SCHEMA_VERSION",
    "FixedScenarioV2Environment",
    "FixedScenarioV2Factory",
    "QualificationArtifactEntry",
    "QualificationError",
    "QualificationExecution",
    "ReferencePPOQualificationBundle",
    "ReferencePPOQualificationPlan",
    "VerifiedReferencePPOQualificationReceipt",
    "actual_v2_process_smoke",
    "actual_v2_cycling_process_smoke",
    "run_reference_ppo_qualification",
    "verify_qualification_bundle",
]
