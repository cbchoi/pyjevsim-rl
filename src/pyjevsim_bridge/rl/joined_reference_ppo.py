"""Fail-closed evidence primitives for joined reference-PPO rollouts."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Final, cast

from .learning import (
    ActorPolicy,
    InMemoryPolicyArtifactStore,
    LearnerSession,
    PolicyAnnouncement,
    PolicyArtifact,
    PolicyArtifactRef,
    PolicyCompatibility,
    PolicyIntegrityError,
    PolicyVersionError,
    ValidatedTransitionBatch,
)
from .records import TransitionRecord

HARNESS_SCHEMA: Final = "joined-reference-ppo-harness-v1"
CLAIM_SCOPE: Final = "reference-ppo-joined-rollout-qualification-only"
WORKER_IDS: Final = tuple(f"worker-{index}" for index in range(4))
TASK105_PLAN_SHA256: Final = "c967e8e3e5d4c572fbc79e5f8f6efb917dfb1f2a2595e9912dca454cf03ba9b3"
TASK105_BUNDLE_SHA256: Final = "c8a2a42929340e7defd7b749d8b4697efc98b2efc2923de8ab155309d35c8792"
TASK105_CAPABILITY_SHA256: Final = (
    "c25af8f706495fc1ea4568914e766cbb2343c6eb550854ae361e3b52105158d2"
)
_THREAD_ENV: Final = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
)

TIME_STRIDE: Final = 32.0
PHASE_BASES: Final = (0.0, 32768.0, 65536.0, 98304.0)


class JoinedAdmissionError(ValueError):
    """Joined evidence is incomplete, inconsistent, or non-unique."""


def _require_frozen_thread_environment() -> None:
    if any(os.environ.get(name) != "1" for name in _THREAD_ENV):
        raise JoinedAdmissionError("reference PPO frozen thread environment differs")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _uint(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sha(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be SHA-256")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be SHA-256") from exc
    return value


class JoinedPhase(IntEnum):
    TRAINING_0 = 0
    TRAINING_1 = 1
    EVALUATION_0 = 2
    EVALUATION_2 = 3


@dataclass(frozen=True, slots=True)
class EpisodeFederationTimeMapper:
    """Injectively embed resettable episode time in monotonic RTI time."""

    stride: float = TIME_STRIDE
    phase_bases: tuple[float, ...] = PHASE_BASES

    def __post_init__(self) -> None:
        if self.stride != TIME_STRIDE or self.phase_bases != PHASE_BASES:
            raise ValueError("joined time mapping differs from version 1")

    def map(self, phase: JoinedPhase, episode_sequence: int, inner_time: float) -> float:
        sequence = _uint("episode_sequence", episode_sequence)
        if sequence == 0 or sequence > 1024:
            raise ValueError("episode_sequence must be in [1, 1024]")
        if isinstance(inner_time, bool) or not isinstance(inner_time, (int, float)):
            raise TypeError("inner_time must be numeric")
        inner = float(inner_time)
        if not math.isfinite(inner) or inner < 1.0 or inner > 30.0:
            raise ValueError("inner_time must be finite and in [1, 30]")
        return self.phase_bases[int(phase)] + sequence * self.stride + inner

    def validate_sequence(
        self, values: Sequence[tuple[str, JoinedPhase, int, float, float]]
    ) -> None:
        previous: dict[str, float] = {}
        observed: set[tuple[str, JoinedPhase, int, float]] = set()
        for worker, phase, sequence, inner, outer in values:
            _text("worker", worker)
            expected = self.map(phase, sequence, inner)
            if outer != expected:
                raise JoinedAdmissionError("federation time differs from declared mapping")
            identity = (worker, phase, sequence, inner)
            if identity in observed or outer <= previous.get(worker, -math.inf):
                raise JoinedAdmissionError("worker federation time is not injective and monotonic")
            observed.add(identity)
            previous[worker] = outer


@dataclass(frozen=True, slots=True)
class AssignmentReceipt:
    run_id: str
    generation: int
    worker_id: str
    episode_id: str
    episode_sequence: int
    phase: JoinedPhase
    policy_version: int
    policy_sha256: str

    def __post_init__(self) -> None:
        for name in ("run_id", "worker_id", "episode_id"):
            _text(name, getattr(self, name))
        _uint("generation", self.generation)
        if _uint("episode_sequence", self.episode_sequence) == 0:
            raise ValueError("episode_sequence must be one-based")
        _uint("policy_version", self.policy_version)
        _sha("policy_sha256", self.policy_sha256)

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class PolicyActivationReceipt:
    run_id: str
    generation: int
    worker_id: str
    phase: JoinedPhase
    policy_version: int
    policy_sha256: str
    assignment_set_sha256: str

    def __post_init__(self) -> None:
        for name in ("run_id", "worker_id"):
            _text(name, getattr(self, name))
        _uint("generation", self.generation)
        _uint("policy_version", self.policy_version)
        _sha("policy_sha256", self.policy_sha256)
        _sha("assignment_set_sha256", self.assignment_set_sha256)

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class TimeGrantReceipt:
    worker_id: str
    requested: float
    granted: float
    previous_grant: float

    def __post_init__(self) -> None:
        _text("worker_id", self.worker_id)
        values = (self.previous_grant, self.requested, self.granted)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("grant times must be finite and non-negative")
        if self.requested < self.previous_grant or not (
            self.previous_grant <= self.granted <= self.requested
        ):
            raise ValueError("grant regresses or exceeds its request")

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class TerminalReceipt:
    run_id: str
    generation: int
    worker_id: str
    episode_id: str
    final_step_id: int
    transition_count: int
    terminated: bool
    truncated: bool
    phase_cut: bool
    transition_chain_sha256: str

    def __post_init__(self) -> None:
        for name in ("run_id", "worker_id", "episode_id"):
            _text(name, getattr(self, name))
        _uint("generation", self.generation)
        _uint("final_step_id", self.final_step_id)
        _uint("transition_count", self.transition_count)
        _sha("transition_chain_sha256", self.transition_chain_sha256)
        if sum((self.terminated, self.truncated, self.phase_cut)) != 1:
            raise ValueError("terminal receipt requires exactly one terminal disposition")

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


class FilesystemPolicyArtifactStore:
    """Immutable process-shared policy CAS with exclusive creation and rehash."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._versions = self.root / "versions"
        self._versions.mkdir(exist_ok=True)

    def _path(self, digest: str) -> Path:
        _sha("artifact digest", digest)
        return self.root / f"{digest}.json"

    def _version_path(self, artifact: PolicyArtifact) -> Path:
        identity = _digest(
            [artifact.run_id, artifact.generation, artifact.policy_version]
        )
        return self._versions / identity

    @staticmethod
    def _write_exclusive(path: Path, body: bytes) -> bool:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        return True

    async def put(self, artifact: PolicyArtifact) -> PolicyArtifactRef:
        if not isinstance(artifact, PolicyArtifact):
            raise TypeError("artifact must be PolicyArtifact")
        body = _canonical(
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
        path = self._path(artifact.sha256)
        version_path = self._version_path(artifact)
        digest_bytes = artifact.sha256.encode("ascii")
        if (
            not self._write_exclusive(version_path, digest_bytes)
            and version_path.read_bytes() != digest_bytes
        ):
            raise PolicyVersionError(
                "policy version is already bound to a different artifact digest"
            )
        if not self._write_exclusive(path, body) and path.read_bytes() != body:
            raise PolicyIntegrityError("occupied policy CAS path differs")
        return PolicyArtifactRef(
            artifact.run_id, artifact.generation, artifact.policy_version,
            path.as_uri(), artifact.sha256, artifact.size_bytes, artifact.media_type,
            artifact.compatibility, artifact.source_batch_sha256,
        )

    async def get(self, reference: PolicyArtifactRef) -> PolicyArtifact:
        if not isinstance(reference, PolicyArtifactRef):
            raise TypeError("reference must be PolicyArtifactRef")
        path = self._path(reference.sha256)
        if path.as_uri() != reference.uri:
            raise PolicyIntegrityError("policy URI is outside the configured CAS")
        try:
            raw = json.loads(path.read_bytes())
            artifact = PolicyArtifact(
                run_id=raw["run_id"], generation=raw["generation"],
                policy_version=raw["policy_version"], payload=bytes.fromhex(raw["payload_hex"]),
                media_type=raw["media_type"],
                compatibility=PolicyCompatibility.from_dict(raw["compatibility"]),
                source_batch_sha256=raw["source_batch_sha256"], provenance=raw["provenance"],
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PolicyIntegrityError("policy CAS object is malformed") from exc
        if artifact.sha256 != reference.sha256 or artifact.size_bytes != reference.size_bytes:
            raise PolicyIntegrityError("policy CAS object differs from reference")
        if (
            artifact.run_id != reference.run_id
            or artifact.generation != reference.generation
            or artifact.policy_version != reference.policy_version
            or artifact.media_type != reference.media_type
            or artifact.compatibility != reference.compatibility
            or artifact.source_batch_sha256 != reference.source_batch_sha256
        ):
            raise PolicyIntegrityError("policy CAS metadata differs from reference")
        version_path = self._version_path(artifact)
        if version_path.read_bytes() != artifact.sha256.encode("ascii"):
            raise PolicyIntegrityError("policy CAS version binding differs")
        return artifact


@dataclass(frozen=True, slots=True)
class JoinedAdmissionReceipt:
    admitted: bool
    blockers: tuple[str, ...]
    raw_count: int
    semantic_count: int
    unique_count: int
    transition_sha256: str


@dataclass(frozen=True, slots=True)
class JoinedPhaseCutV1:
    batch_index: int
    phase: JoinedPhase
    worker_id: str
    episode_sequence: int
    episode_id: str
    terminal_logical_time: float
    records: tuple[TransitionRecord, ...]


@dataclass(frozen=True, slots=True)
class JoinedEvaluationAssignmentV1:
    worker_id: str
    phase: JoinedPhase
    policy_version: int
    scenario_ordinal: int
    seed: int
    episode_sequence: int
    terminal_logical_time: float
    oracle: Mapping[str, object]


def _policy_from_receipt(path: Path) -> tuple[PolicyArtifact, PolicyArtifactRef]:
    value = json.loads(path.read_bytes())
    artifact_value = value["artifact"]
    reference_value = value["reference"]
    compatibility = PolicyCompatibility.from_dict(artifact_value["compatibility"])
    artifact = PolicyArtifact(
        run_id=artifact_value["run_id"],
        generation=artifact_value["generation"],
        policy_version=artifact_value["policy_version"],
        payload=base64.b64decode(artifact_value["payload_base64"], validate=True),
        media_type=artifact_value["media_type"],
        compatibility=compatibility,
        source_batch_sha256=artifact_value["source_batch_sha256"],
        provenance=artifact_value["provenance"],
    )
    if artifact.sha256 != artifact_value["artifact_sha256"]:
        raise PolicyIntegrityError("accepted policy receipt artifact digest differs")
    reference = PolicyArtifactRef.from_dict(reference_value)
    if reference.sha256 != artifact.sha256:
        raise PolicyIntegrityError("accepted policy receipt reference differs")
    return artifact, reference


@dataclass(frozen=True, slots=True)
class JoinedReferenceLaunchPlanV1:
    oracle_root: Path
    run_id: str
    generation: int
    worker_ids: tuple[str, ...]
    source_lock_sha256: str
    artifacts: Mapping[str, Path]
    artifact_sha256: Mapping[str, str]
    policies: tuple[PolicyArtifact, PolicyArtifact, PolicyArtifact]
    policy_references: tuple[PolicyArtifactRef, PolicyArtifactRef, PolicyArtifactRef]
    transition_batches: tuple[tuple[TransitionRecord, ...], tuple[TransitionRecord, ...]]

    def evaluation_assignments(self) -> tuple[JoinedEvaluationAssignmentV1, ...]:
        value = json.loads(self.artifacts["evaluation-isolation"].read_bytes())
        rows = value.get("episode_rows")
        if not isinstance(rows, list) or len(rows) != 16:
            raise JoinedAdmissionError("TASK-RL-105 evaluation row set differs")
        result = []
        per_worker_phase: dict[tuple[str, int], int] = {}
        per_policy_index: dict[int, int] = {}
        for row in rows:
            if not isinstance(row, Mapping) or row.get("policy_version") not in (0, 2):
                raise JoinedAdmissionError("TASK-RL-105 evaluation row is malformed")
            version = cast(int, row["policy_version"])
            policy_index = per_policy_index.get(version, 0)
            if policy_index >= len(self.worker_ids) * 2:
                raise JoinedAdmissionError("TASK-RL-105 evaluation allocation differs")
            worker = self.worker_ids[policy_index // 2]
            per_policy_index[version] = policy_index + 1
            key = (worker, version)
            sequence = per_worker_phase.get(key, 0) + 1
            per_worker_phase[key] = sequence
            result.append(
                JoinedEvaluationAssignmentV1(
                    worker,
                    JoinedPhase.EVALUATION_0 if version == 0 else JoinedPhase.EVALUATION_2,
                    version, cast(int, row["scenario_ordinal"]), cast(int, row["seed"]),
                    sequence, float(cast(int, row["step_count"])),
                    dict(row),
                )
            )
        if {item.policy_version for item in result} != {0, 2}:
            raise JoinedAdmissionError("TASK-RL-105 evaluation policies differ")
        if per_policy_index != {0: 8, 2: 8}:
            raise JoinedAdmissionError("TASK-RL-105 evaluation allocation differs")
        return tuple(result)

    @classmethod
    def from_task105_evidence(
        cls,
        *,
        oracle_root: Path,
        run_id: str,
        generation: int,
        worker_ids: Sequence[str],
        source_lock: Mapping[str, object],
    ) -> JoinedReferenceLaunchPlanV1:
        root = oracle_root.resolve()
        if run_id != "task-rl-105" or generation != 0 or tuple(worker_ids) != WORKER_IDS:
            raise JoinedAdmissionError("joined launch identity differs from TASK-RL-105")
        ledger_path = root / "ledger.json"
        ledger = json.loads(ledger_path.read_bytes())
        capability = ledger.get("capability")
        if (
            ledger.get("schema_version") != "reference-ppo-qualification-v1"
            or ledger.get("plan_sha256") != TASK105_PLAN_SHA256
            or ledger.get("bundle_sha256") != TASK105_BUNDLE_SHA256
            or not isinstance(capability, Mapping)
            or capability.get("sha256") != TASK105_CAPABILITY_SHA256
            or capability.get("admitted") is not True
            or ledger.get("blockers") != []
        ):
            raise JoinedAdmissionError("accepted TASK-RL-105 ledger identity differs")
        entries = ledger.get("entries")
        if not isinstance(entries, list) or len(entries) != 18:
            raise JoinedAdmissionError("TASK-RL-105 ledger entry set differs")
        paths: dict[str, Path] = {}
        digests: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise JoinedAdmissionError("TASK-RL-105 ledger entry is malformed")
            role = _text("artifact role", entry.get("role"))
            relative = _text("artifact path", entry.get("path"))
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise JoinedAdmissionError("TASK-RL-105 artifact escapes or is missing")
            expected = _sha("artifact sha256", entry.get("sha256"))
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != expected or len(body) != entry.get("size_bytes"):
                raise JoinedAdmissionError("TASK-RL-105 artifact rehash differs")
            if role in paths:
                raise JoinedAdmissionError("TASK-RL-105 artifact role is duplicated")
            paths[role], digests[role] = path, expected
        raw_roles = set(paths) - {"capability-receipt"}
        if len(raw_roles) != 17:
            raise JoinedAdmissionError("TASK-RL-105 raw artifact set is not exact")
        policy_pairs = tuple(_policy_from_receipt(paths[f"policy-{index}"]) for index in range(3))
        policies = (policy_pairs[0][0], policy_pairs[1][0], policy_pairs[2][0])
        references = (policy_pairs[0][1], policy_pairs[1][1], policy_pairs[2][1])
        batches: list[tuple[TransitionRecord, ...]] = []
        for index in (1, 2):
            raw = json.loads(paths[f"transitions-{index}"].read_bytes())
            records = tuple(TransitionRecord.from_dict(item) for item in raw["records"])
            if len(records) != 2048:
                raise JoinedAdmissionError("TASK-RL-105 transition batch size differs")
            batches.append(records)
        lock_sha = _sha("source lock sha256", source_lock.get("sha256"))
        return cls(
            root, run_id, generation, tuple(worker_ids), lock_sha, paths, digests,
            policies, references, (batches[0], batches[1]),
        )

    def to_harness_config(
        self,
        *,
        url: str,
        federation_name: str,
        fom: Path,
        process_dir: Path,
        lookahead: float,
        timeout_seconds: float,
        claim_scope: str,
    ) -> dict[str, object]:
        if claim_scope != CLAIM_SCOPE:
            raise JoinedAdmissionError("joined claim scope differs")
        barriers = []
        phase_specs = (
            ("training-0", "training", 0, 1, 0),
            ("training-1", "training", 1, 2, 1),
            ("evaluation-0", "evaluation", 2, None, 0),
            ("evaluation-2", "evaluation", 3, None, 2),
        )
        for phase_name, phase_kind, phase_index, batch_index, policy_version in phase_specs:
            artifact = self.policies[policy_version]
            phase_base = PHASE_BASES[phase_index]
            announcement = 1.0 if phase_index == 0 else phase_base
            barriers.append(
                {
                    "phase_name": phase_name,
                    "phase_kind": phase_kind,
                    "phase_index": phase_index,
                    "batch_index": batch_index,
                    "policy_version": policy_version,
                    "announcement_time": announcement,
                    "ack_time": announcement + 1.0,
                    "phase_base": phase_base,
                    "cutoff_time": phase_base + 32767.0,
                    "artifact_sha256": artifact.sha256,
                    "checkpoint_sha256": hashlib.sha256(artifact.payload).hexdigest(),
                }
            )
        return {
            "schema_version": HARNESS_SCHEMA, "run_id": self.run_id,
            "generation": self.generation, "url": url,
            "federation_name": federation_name, "fom": str(fom.resolve()),
            "lookahead": lookahead, "timeout_seconds": timeout_seconds,
            "process_dir": str(process_dir.resolve()), "oracle_root": str(self.oracle_root),
            "worker_ids": list(self.worker_ids), "batch_count": 2,
            "records_per_worker_per_batch": 512, "training_records": 4096,
            "policy_barriers": barriers,
            "evaluation_ordinals": sorted(
                {item.scenario_ordinal for item in self.evaluation_assignments()}
            ),
            "source_lock_sha256": self.source_lock_sha256,
            "claim_scope": claim_scope,
        }


def _load_plan_from_config(value: Mapping[str, object]) -> JoinedReferenceLaunchPlanV1:
    return JoinedReferenceLaunchPlanV1.from_task105_evidence(
        oracle_root=Path(_text("oracle_root", value.get("oracle_root"))),
        run_id=_text("run_id", value.get("run_id")),
        generation=cast(int, value.get("generation")),
        worker_ids=cast(Sequence[str], value.get("worker_ids")),
        source_lock={"sha256": value.get("source_lock_sha256")},
    )


class JoinedReferenceWorkerRuntime:
    """Join-first worker replaying accepted transitions with the unchanged actor."""

    def __init__(self, plan: JoinedReferenceLaunchPlanV1, worker_id: str, cas_root: Path) -> None:
        _require_frozen_thread_environment()
        if worker_id not in plan.worker_ids:
            raise JoinedAdmissionError("worker is not assigned by the launch plan")
        self.plan, self.worker_id = plan, worker_id
        self.store = FilesystemPolicyArtifactStore(cas_root)
        from .qualification_models.anti_torpedo_features import AntiTorpedoV2FeatureContract
        from .reference_ppo import ReferencePPOPolicyLoader

        feature = AntiTorpedoV2FeatureContract()
        self.actor = ActorPolicy(
            ReferencePPOPolicyLoader(feature), self.store, run_id=plan.run_id,
            generation=plan.generation, compatibility=plan.policies[0].compatibility,
        )
        self.evaluation_actor = ActorPolicy(
            ReferencePPOPolicyLoader(feature), self.store, run_id=plan.run_id,
            generation=plan.generation, compatibility=plan.policies[0].compatibility,
        )
        self._active_phase: str | None = None
        self._environment: Any | None = None
        self._sent = {1: 0, 2: 0}
        self._evaluated = {"evaluation-0": 0, "evaluation-2": 0}
        self._evaluation_receipts: list[dict[str, object]] = []

    @classmethod
    def from_launch_plan(
        cls, value: Mapping[str, object], *, worker_id: str
    ) -> JoinedReferenceWorkerRuntime:
        plan = _load_plan_from_config(value)
        process_dir = Path(_text("process_dir", value.get("process_dir")))
        return cls(plan, worker_id, process_dir / "policy-cas")

    async def select_policy_announcement(
        self, phase_name: str, granted: object
    ) -> PolicyAnnouncement:
        expected = {"training-0": 0, "training-1": 1, "evaluation-0": 0,
                    "evaluation-2": 2}.get(phase_name)
        if expected is None:
            raise JoinedAdmissionError("unknown joined phase")
        interactions = getattr(granted, "interactions", ())
        candidates = [
            PolicyAnnouncement.from_envelope(item)
            for item in interactions
            if getattr(item, "interaction_class", None) == "RLPolicyAnnouncement"
        ]
        if len(candidates) != 1 or candidates[0].policy_version != expected:
            raise JoinedAdmissionError("policy announcement cut differs")
        return candidates[0]

    async def activate_policy(
        self, phase_name: str, announcement: PolicyAnnouncement
    ) -> dict[str, object]:
        phase_index = ("training-0", "training-1", "evaluation-0", "evaluation-2").index(
            phase_name
        )
        policy_version = (0, 1, 0, 2)[phase_index]
        artifact = self.plan.policies[policy_version]
        reference = await self.store.put(artifact)
        if announcement.artifact_ref.sha256 != reference.sha256:
            raise JoinedAdmissionError("announced policy differs from accepted policy")
        actor = self.actor if phase_name.startswith("training") else self.evaluation_actor
        await actor.activate(reference)
        self._active_phase = phase_name
        payload = {
            "schema_version": HARNESS_SCHEMA, "run_id": self.plan.run_id,
            "generation": self.plan.generation, "worker_id": self.worker_id,
            "phase_name": phase_name, "phase_index": phase_index,
            "batch_index": phase_index + 1 if phase_index < 2 else None,
            "policy_version": reference.policy_version,
            "artifact_sha256": reference.sha256,
            "checkpoint_sha256": hashlib.sha256(artifact.payload).hexdigest(),
            "resolved_sha256": reference.sha256, "activated": True,
        }
        announcement_time = 1.0 if phase_index == 0 else PHASE_BASES[phase_index]
        logical_time = announcement_time + 1.0
        return {
            "schema_version": 1, "run_id": self.plan.run_id,
            "generation": self.plan.generation, "worker_id": self.worker_id,
            "episode_id": "__policy_activation__", "step_id": phase_index,
            "policy_version": reference.policy_version,
            "idempotency_key": _digest(payload), "logical_time": logical_time,
            "payload": payload,
        }

    async def phase_cuts(self, batch_index: int) -> tuple[JoinedPhaseCutV1, ...]:
        records = [
            item for item in self.plan.transition_batches[batch_index - 1]
            if item.worker_id == self.worker_id
        ]
        grouped: list[list[TransitionRecord]] = []
        for record in records:
            if not grouped or grouped[-1][0].episode_id != record.episode_id:
                grouped.append([])
            grouped[-1].append(record)
        phase = JoinedPhase.TRAINING_0 if batch_index == 1 else JoinedPhase.TRAINING_1
        return tuple(
            JoinedPhaseCutV1(
                batch_index, phase, self.worker_id, sequence, group[0].episode_id,
                group[-1].logical_time, tuple(group),
            )
            for sequence, group in enumerate(grouped, 1)
        )

    async def replay_phase_cut_after_activation(
        self, batch_index: int, phase_cut: JoinedPhaseCutV1
    ) -> tuple[TransitionRecord, ...]:
        if self._active_phase != f"training-{batch_index - 1}":
            raise JoinedAdmissionError("rollout attempted before exact policy activation")
        from .qualification_models.anti_torpedo import anti_torpedo_v2_environment_factory

        first = phase_cut.records[0]
        previous_wrapper = cast(Mapping[str, object], first.previous_observation)
        physical_previous = previous_wrapper["observation"]
        first_info = first.info
        if self._environment is None:
            self._environment = anti_torpedo_v2_environment_factory(
                instance_id=self.worker_id, run_id=self.plan.run_id
            )
        if first.step_id == 1:
            observation, _info = self._environment.reset(
                seed=cast(int, first_info["seed"]),
                options={"scenario_ordinal": cast(int, first_info["scenario_ordinal"])},
            )
            if observation != physical_previous:
                raise JoinedAdmissionError("joined reset differs from accepted local oracle")
        replayed: list[TransitionRecord] = []
        for oracle in phase_cut.records:
            from .reference_ppo import PPOInferenceInputV1

            raw_input = cast(Mapping[str, object], oracle.previous_observation)
            inference = PPOInferenceInputV1(
                raw_input["observation"], cast(tuple[bool, ...], raw_input["action_mask"]),
                cast(int, raw_input["sampling_seed"]), cast(str, raw_input["run_id"]),
                cast(int, raw_input["generation"]), cast(str, raw_input["worker_id"]),
                cast(str, raw_input["episode_id"]), cast(int, raw_input["step_id"]),
                cast(bool, raw_input["explore"]),
            )
            action_batch = await self.actor.actions({self.worker_id: inference})
            action = action_batch.actions[self.worker_id]
            if action != oracle.action:
                raise JoinedAdmissionError("joined actor action differs from accepted oracle")
            observation, reward, terminated, truncated, info = self._environment.step(action)
            next_wrapper = cast(Mapping[str, object], oracle.next_observation)
            if (
                observation != next_wrapper["observation"]
                or reward != oracle.reward
                or terminated != oracle.terminated
                or truncated != oracle.truncated
                or any(info.get(key) != oracle.info.get(key) for key in info)
            ):
                raise JoinedAdmissionError("joined environment step differs from accepted oracle")
            replayed.append(oracle)
        self._sent[batch_index] += len(replayed)
        return tuple(replayed)

    async def evaluation_episodes(
        self, phase_name: str
    ) -> tuple[JoinedEvaluationAssignmentV1, ...]:
        phase = (
            JoinedPhase.EVALUATION_0
            if phase_name == "evaluation-0"
            else JoinedPhase.EVALUATION_2
        )
        values = tuple(
            item for item in self.plan.evaluation_assignments()
            if item.worker_id == self.worker_id and item.phase == phase
        )
        if len(values) != 2:
            raise JoinedAdmissionError("worker evaluation assignment count differs")
        return values

    async def evaluate_after_activation(
        self, phase_name: str, episode: JoinedEvaluationAssignmentV1
    ) -> dict[str, object]:
        if self._active_phase != phase_name:
            raise JoinedAdmissionError("evaluation attempted before exact activation")
        from .qualification_models.anti_torpedo import anti_torpedo_v2_environment_factory
        from .reference_ppo import PPOInferenceInputV1

        environment = anti_torpedo_v2_environment_factory(
            instance_id=self.worker_id, run_id=self.plan.run_id
        )
        try:
            observation, info = environment.reset(
                seed=episode.seed, options={"scenario_ordinal": episode.scenario_ordinal}
            )
            step = 0
            episode_return = 0.0
            trace = "0" * 64
            mask_violations = 0
            terminated = truncated = False
            episode_id = f"evaluation-{episode.policy_version}-{episode.scenario_ordinal}"
            while not (terminated or truncated):
                value = PPOInferenceInputV1(
                    observation, cast(tuple[bool, ...], info["action_mask"]), episode.seed,
                    self.plan.run_id, self.plan.generation, "evaluation-worker",
                    episode_id, step, False,
                )
                selected = await self.evaluation_actor.actions({"evaluation-worker": value})
                action = cast(int, selected.actions["evaluation-worker"])
                if not value.action_mask[action]:
                    mask_violations += 1
                trace = hashlib.sha256(
                    b"pyjevsim-reference-ppo-evaluation-trace-v1\x00"
                    + bytes.fromhex(trace)
                    + _canonical({"input": value.to_dict(), "action": action})
                ).hexdigest()
                observation, reward, terminated, truncated, info = environment.step(action)
                episode_return += float(reward)
                step += 1
            oracle = episode.oracle
            observed = {
                "action_trace": trace, "episode_return": episode_return, "steps": step,
                "terminal": terminated or truncated,
                "final_observation_sha256": _digest(observation),
                "mask_violation_count": mask_violations,
            }
            expected = {
                "action_trace": oracle["action_trace_sha256"],
                "episode_return": oracle["episode_return"], "steps": oracle["step_count"],
                "terminal": bool(oracle["terminated"] or oracle["truncated"]),
                "final_observation_sha256": oracle["final_observation_sha256"],
                "mask_violation_count": oracle["mask_violation_count"],
            }
            if observed != expected:
                raise JoinedAdmissionError("joined evaluation differs from TASK-RL-105")
            self._evaluated[phase_name] += 1
            payload = {
                "schema_version": HARNESS_SCHEMA, "run_id": self.plan.run_id,
                "generation": self.plan.generation, "worker_id": self.worker_id,
                "phase_name": phase_name, "policy_version": episode.policy_version,
                "scenario_ordinal": episode.scenario_ordinal, "episode_id": episode_id,
                **observed, "artifact_sha256": self.plan.policies[episode.policy_version].sha256,
            }
            self._evaluation_receipts.append(dict(payload))
            logical_time = PHASE_BASES[int(episode.phase)] + episode.episode_sequence * 32 + step
            return {
                "schema_version": 1, "run_id": self.plan.run_id,
                "generation": self.plan.generation, "worker_id": self.worker_id,
                "episode_id": episode_id, "step_id": step,
                "policy_version": episode.policy_version,
                "idempotency_key": _digest(payload), "logical_time": logical_time,
                "payload": payload,
            }
        finally:
            environment.close()

    async def terminal_receipt(self) -> dict[str, object]:
        complete = self._sent == {1: 512, 2: 512} and self._evaluated == {
            "evaluation-0": 2, "evaluation-2": 2
        }
        return {
            "schema_version": HARNESS_SCHEMA, "status": "completed" if complete else "failed",
            "worker_id": self.worker_id, "records_by_batch": dict(self._sent),
            "evaluations_by_phase": dict(self._evaluated),
            "evaluation_receipts": list(self._evaluation_receipts),
            "environment_constructed_after_activation": self._environment is not None,
        }


class JoinedReferenceCoordinatorRuntime:
    """Coordinator-side oracle, policy and unchanged learner lifecycle boundary."""

    def __init__(self, plan: JoinedReferenceLaunchPlanV1, cas_root: Path | None = None) -> None:
        _require_frozen_thread_environment()
        self.plan = plan
        self._raw: dict[int, list[TransitionRecord]] = {1: [], 2: []}
        self._store = InMemoryPolicyArtifactStore()
        self._session: LearnerSession | None = None
        self._learner: Any | None = None
        self._cas_root = plan.oracle_root / "joined-policy-cas" if cas_root is None else cas_root
        self._cas = FilesystemPolicyArtifactStore(self._cas_root)
        self._actual_policies: dict[int, PolicyArtifact] = {}
        self._completed_phases: list[str] = []
        self._synchronized_labels: list[str] = []
        self._recovery_after_update1: Any | None = None
        self._evaluation_state_sha256: str | None = None
        self._evaluation_raw: list[dict[str, object]] = []

    @classmethod
    def from_launch_plan(
        cls, value: Mapping[str, object]
    ) -> JoinedReferenceCoordinatorRuntime:
        return cls(
            _load_plan_from_config(value),
            Path(_text("process_dir", value.get("process_dir"))) / "policy-cas",
        )

    async def policy_announcement(
        self, phase_name: str, *, announcement_time: float
    ) -> PolicyAnnouncement:
        await self._ensure_learner_initialized()
        policy_version = {"training-0": 0, "training-1": 1, "evaluation-0": 0,
                          "evaluation-2": 2}.get(phase_name)
        if policy_version is None or policy_version not in self._actual_policies:
            raise JoinedAdmissionError("phase policy is not an actual joined learner output")
        artifact = self._actual_policies[policy_version]
        reference = await self._cas.put(artifact)
        if (await self._cas.get(reference)).payload != artifact.payload:
            raise PolicyIntegrityError("published joined policy CAS rehash differs")
        return PolicyAnnouncement(
            self.plan.run_id, self.plan.generation, "joined-reference-coordinator",
            announcement_time, reference,
        )

    async def expected_cut(self, batch_index: int) -> tuple[dict[str, object], ...]:
        return tuple(item.to_dict() for item in self.plan.transition_batches[batch_index - 1])

    async def validate_worker_receipt(
        self, phase_name: str, envelope: Mapping[str, object]
    ) -> dict[str, object]:
        payload = envelope.get("payload")
        if not isinstance(payload, Mapping):
            raise JoinedAdmissionError("worker receipt payload is missing")
        result = dict(payload)
        if result.get("phase_name") != phase_name:
            raise JoinedAdmissionError("worker receipt phase differs")
        return result

    async def complete_phase(self, phase_name: str) -> None:
        expected = ("training-0", "training-1", "evaluation-0", "evaluation-2")
        if phase_name != expected[len(self._completed_phases)]:
            raise JoinedAdmissionError("joined phase completion order differs")
        self._completed_phases.append(phase_name)

    async def record_synchronization(self, label: str) -> None:
        """Record the one exact ready/activation/terminal synchronization order."""

        if not isinstance(label, str) or not label:
            raise JoinedAdmissionError("joined synchronization label must be non-empty")
        expected = tuple(
            f"{self.plan.run_id}-{suffix}"
            for suffix in (
                "ready",
                "training-0-activated",
                "training-1-activated",
                "evaluation-0-activated",
                "evaluation-2-activated",
                "terminal",
            )
        )
        if label in self._synchronized_labels:
            raise JoinedAdmissionError("joined synchronization label is duplicated")
        position = len(self._synchronized_labels)
        if position >= len(expected) or label not in expected:
            raise JoinedAdmissionError("joined synchronization label is unknown")
        if label != expected[position]:
            raise JoinedAdmissionError("joined synchronization label is out of order")
        required_completed = 0 if position == 0 else min(position - 1, 4)
        if len(self._completed_phases) != required_completed:
            raise JoinedAdmissionError(
                "joined synchronization differs from phase completion boundary"
            )
        self._synchronized_labels.append(label)

    async def expected_evaluation_receipts(
        self, phase_name: str
    ) -> tuple[dict[str, object], ...]:
        version = 0 if phase_name == "evaluation-0" else 2
        result = []
        for item in self.plan.evaluation_assignments():
            if item.policy_version != version:
                continue
            oracle = item.oracle
            result.append(
                {
                    "schema_version": HARNESS_SCHEMA, "run_id": self.plan.run_id,
                    "generation": self.plan.generation, "worker_id": item.worker_id,
                    "phase_name": phase_name, "policy_version": version,
                    "scenario_ordinal": item.scenario_ordinal,
                    "episode_id": oracle["episode_id"],
                    "action_trace": oracle["action_trace_sha256"],
                    "episode_return": oracle["episode_return"], "steps": oracle["step_count"],
                    "terminal": bool(oracle["terminated"] or oracle["truncated"]),
                    "final_observation_sha256": oracle["final_observation_sha256"],
                    "mask_violation_count": oracle["mask_violation_count"],
                    "artifact_sha256": self.plan.policies[version].sha256,
                }
            )
        return tuple(result)

    def expand_raw_receipt_callback(
        self, envelope: Mapping[str, object]
    ) -> tuple[int, dict[str, object]] | None:
        payload = envelope.get("payload")
        if not isinstance(payload, Mapping) or payload.get("phase_name") not in {
            "evaluation-0", "evaluation-2"
        } or "scenario_ordinal" not in payload:
            return None
        phase_index = 2 if payload["phase_name"] == "evaluation-0" else 3
        self._evaluation_raw.append(dict(payload))
        return phase_index, dict(payload)

    def expand_raw_transition_callback(
        self, batch_index: int, envelope: Mapping[str, object]
    ) -> tuple[dict[str, object], ...]:
        payload = envelope.get("payload")
        if isinstance(payload, Mapping) and isinstance(payload.get("batch_records"), list):
            values = cast(list[dict[str, object]], payload["batch_records"])
        else:
            values = [dict(envelope)]
        normalized = []
        for value in values:
            record = TransitionRecord.from_dict(value)
            self._raw[batch_index].append(record)
            normalized.append(record.to_dict())
        return tuple(normalized)

    async def validate_granted_batch(self, batch_index: int, granted: object) -> None:
        grant = getattr(granted, "granted_time", None)
        expected_minimum = PHASE_BASES[batch_index - 1]
        if not isinstance(grant, (int, float)) or float(grant) < expected_minimum:
            raise JoinedAdmissionError("joined grant does not cover its phase")
        expected = self.plan.transition_batches[batch_index - 1]
        observed = {item.idempotency_key: item.to_dict() for item in self._raw[batch_index]}
        accepted = {item.idempotency_key: item.to_dict() for item in expected}
        if len(observed) != len(self._raw[batch_index]) or observed != accepted:
            missing = sorted(set(accepted) - set(observed))
            extra = sorted(set(observed) - set(accepted))
            changed = sorted(
                key
                for key in set(observed) & set(accepted)
                if observed[key] != accepted[key]
            )
            raise JoinedAdmissionError(
                "joined raw cut differs from TASK-RL-105: "
                f"raw={len(self._raw[batch_index])}, unique={len(observed)}, "
                f"expected={len(accepted)}, missing={missing[:1]}, "
                f"extra={extra[:1]}, changed={changed[:1]}"
            )
        self._raw[batch_index] = list(expected)
        await self._consume_exact(batch_index, expected)

    async def _consume_exact(
        self, batch_index: int, records: Sequence[TransitionRecord]
    ) -> None:
        await self._ensure_learner_initialized()
        if self._session is None:
            raise JoinedAdmissionError("joined learner initialization failed")
        reference = await self._session.consume(ValidatedTransitionBatch.build(records))
        if reference is None:
            raise JoinedAdmissionError("unchanged learner did not publish")
        actual = await self._store.get(reference)
        accepted = self.plan.policies[batch_index]
        if actual.payload != accepted.payload or actual.sha256 != accepted.sha256:
            raise JoinedAdmissionError("joined learner bytes differ from TASK-RL-105")
        cast(Any, self._learner).bind_published(reference, actual)
        self._actual_policies[batch_index] = actual
        if batch_index == 1:
            self._recovery_after_update1 = await self._session.snapshot_recovery_state()
        else:
            self._evaluation_state_sha256 = (
                await self._session.snapshot_recovery_state()
            ).sha256

    async def _ensure_learner_initialized(self) -> None:
        from .qualification_models.anti_torpedo_features import AntiTorpedoV2FeatureContract
        from .reference_ppo import ReferencePPOLearnerAdapter

        if self._session is None:
            feature = AntiTorpedoV2FeatureContract()
            learner = ReferencePPOLearnerAdapter(
                feature, run_id=self.plan.run_id, generation=self.plan.generation,
                compatibility=self.plan.policies[0].compatibility,
            )
            self._learner = learner
            self._session = LearnerSession(
                learner, self._store, run_id=self.plan.run_id,
                generation=self.plan.generation,
            )
            initial_ref = await self._session.publish_initial(learner.initial_policy())
            initial_artifact = await self._store.get(initial_ref)
            if initial_artifact.payload != self.plan.policies[0].payload:
                raise JoinedAdmissionError("joined initial policy differs from TASK-RL-105")
            learner.bind_published(initial_ref, initial_artifact)
            self._actual_policies[0] = initial_artifact

    async def finalize_evidence(
        self, cuts: tuple[dict[str, object], ...],
        evaluations: tuple[dict[str, object], ...],
    ) -> dict[str, object]:
        if tuple(len(self._raw[index]) for index in (1, 2)) != (2048, 2048):
            raise JoinedAdmissionError("joined coordinator raw budget differs")
        if self._completed_phases != [
            "training-0", "training-1", "evaluation-0", "evaluation-2"
        ] or self._synchronized_labels != [
            f"{self.plan.run_id}-{suffix}"
            for suffix in (
                "ready", "training-0-activated", "training-1-activated",
                "evaluation-0-activated", "evaluation-2-activated", "terminal",
            )
        ] or len(evaluations) != 2:
            raise JoinedAdmissionError("joined four-phase evidence is incomplete")
        recovery = await self._verify_fresh_and_recovered_branches()
        return {
            "schema_version": HARNESS_SCHEMA, "status": "completed",
            "run_id": self.plan.run_id, "training_records": 4096,
            "cuts": list(cuts), "evaluations": list(evaluations),
            "raw_transition_sha256": {
                str(index): _digest([item.to_dict() for item in self._raw[index]])
                for index in (1, 2)
            },
            "policy_bytes_matched": True,
            "evaluation_receipts": list(self._evaluation_raw),
            "evaluation_state_before_sha256": self._evaluation_state_sha256,
            "evaluation_state_after_sha256": (
                await cast(LearnerSession, self._session).snapshot_recovery_state()
            ).sha256,
            **recovery,
        }

    async def _verify_fresh_and_recovered_branches(self) -> dict[str, object]:
        from .qualification_models.anti_torpedo_features import AntiTorpedoV2FeatureContract
        from .reference_ppo import ReferencePPOLearnerAdapter

        if self._recovery_after_update1 is None or self._learner is None:
            raise JoinedAdmissionError("joined recovery boundary is missing")
        feature = AntiTorpedoV2FeatureContract()
        fresh_store = InMemoryPolicyArtifactStore()
        fresh = ReferencePPOLearnerAdapter(
            feature, run_id=self.plan.run_id, generation=self.plan.generation,
            compatibility=self.plan.policies[0].compatibility,
        )
        fresh_session = LearnerSession(
            fresh, fresh_store, run_id=self.plan.run_id, generation=self.plan.generation
        )
        fresh_initial = await fresh_session.publish_initial(fresh.initial_policy())
        fresh.bind_published(fresh_initial, await fresh_store.get(fresh_initial))
        fresh_ref = await fresh_session.consume(
            ValidatedTransitionBatch.build(self.plan.transition_batches[0])
        )
        if fresh_ref is None or (
            await fresh_store.get(fresh_ref)
        ).payload != self.plan.policies[1].payload:
            raise JoinedAdmissionError("fresh joined update one differs")

        reload = ReferencePPOLearnerAdapter.from_artifact(self.plan.policies[1], feature)
        policy1_ref = await self._store.put(self.plan.policies[1])
        reload.bind_published(policy1_ref, self.plan.policies[1])
        restored = await LearnerSession.restore(
            reload, self._store, state=self._recovery_after_update1,
            run_id=self.plan.run_id, generation=self.plan.generation,
        )
        reload_ref = await restored.consume(
            ValidatedTransitionBatch.build(self.plan.transition_batches[1])
        )
        if reload_ref is None or (
            await self._store.get(reload_ref)
        ).payload != self.plan.policies[2].payload:
            raise JoinedAdmissionError("recovered joined update two differs")
        main_adam = cast(Any, self._learner).adam_step
        qualification_adam = main_adam + fresh.adam_step + (reload.adam_step - 80)
        if qualification_adam != 320:
            raise JoinedAdmissionError("joined qualification Adam reconciliation differs")
        restored_final = await restored.snapshot_recovery_state()
        uninterrupted_state = await cast(
            LearnerSession, self._session
        ).snapshot_recovery_state()
        fresh_state = await fresh_session.snapshot_recovery_state()

        def cursor_summary(state: Any) -> dict[str, object]:
            return {
                "policy_version": state.last_published_version,
                "next_policy_version": state.next_policy_version,
                "consumed_batches": len(state.consumed_batches),
                "consumed_records": len(state.consumed_records),
                "stream_count": len(state.stream_positions),
                "sha256": state.sha256,
            }
        artifact_pairs = [
            {
                "role": role, "local_sha256": digest, "joined_sha256": digest,
            }
            for role, digest in (
                ("policy-0", self.plan.policies[0].sha256),
                ("policy-1", self.plan.policies[1].sha256),
                ("policy-2", self.plan.policies[2].sha256),
                ("checkpoint-1", hashlib.sha256(self.plan.policies[1].payload).hexdigest()),
                ("checkpoint-2", hashlib.sha256(self.plan.policies[2].payload).hexdigest()),
            )
        ]
        policy_recovery = {
            "schema_version": "joined-policy-recovery-v1",
            "versions": [0, 1, 2], "batch_sizes": [2048, 2048],
            "uninterrupted_adam_steps": main_adam, "fresh_adam_steps": fresh.adam_step,
            "reload_adam_steps": reload.adam_step - 80,
            "qualification_adam_steps": qualification_adam,
            "artifact_pairs": artifact_pairs,
            "branches": [
                {
                    "branch": "uninterrupted",
                    "batch_sha256": ValidatedTransitionBatch.build(
                        self.plan.transition_batches[1]
                    ).sha256,
                    "cursor": cursor_summary(uninterrupted_state),
                    "adam_steps": main_adam,
                    "payload_sha256": hashlib.sha256(self.plan.policies[2].payload).hexdigest(),
                },
                {
                    "branch": "fresh",
                    "batch_sha256": ValidatedTransitionBatch.build(
                        self.plan.transition_batches[0]
                    ).sha256,
                    "cursor": cursor_summary(fresh_state),
                    "adam_steps": fresh.adam_step,
                    "payload_sha256": hashlib.sha256(self.plan.policies[1].payload).hexdigest(),
                },
                {
                    "branch": "reload",
                    "batch_sha256": ValidatedTransitionBatch.build(
                        self.plan.transition_batches[1]
                    ).sha256,
                    "cursor": cursor_summary(restored_final),
                    "adam_steps": reload.adam_step - 80,
                    "payload_sha256": hashlib.sha256(self.plan.policies[2].payload).hexdigest(),
                },
            ],
            "recovery_cursor": {
                "policy_version": 1, "batch_index": 1, "restored": True,
                "sha256": self._recovery_after_update1.sha256,
            },
        }
        return {
            "adam_step_count": main_adam,
            "qualification_adam_step_count": qualification_adam,
            "fresh_update1_matched": True,
            "recovered_update2_matched": True,
            "recovery_cursor_sha256": self._recovery_after_update1.sha256,
            "policy_recovery_raw": policy_recovery,
        }


def verify_joined_reference_process_results(
    config: Mapping[str, object],
    coordinator: Mapping[str, object],
    workers: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if config.get("run_id") != "task-rl-105" or coordinator.get("status") != "completed":
        raise JoinedAdmissionError("joined process result identity/status differs")
    expected = set(WORKER_IDS)
    if len(workers) != 4 or {item.get("worker_id") for item in workers} != expected:
        raise JoinedAdmissionError("joined worker terminal set differs")
    if any(item.get("status") != "completed" or item.get("records_by_batch") != {"1": 512, "2": 512}
           and item.get("records_by_batch") != {1: 512, 2: 512} for item in workers):
        raise JoinedAdmissionError("joined worker terminal budget differs")
    expected_evaluations = {"evaluation-0": 2, "evaluation-2": 2}
    if any(item.get("evaluations_by_phase") != expected_evaluations for item in workers):
        raise JoinedAdmissionError("joined worker evaluation budget differs")
    evaluations = coordinator.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 2:
        raise JoinedAdmissionError("joined coordinator evaluation evidence differs")
    return {"verified": True, "run_id": "task-rl-105", "worker_count": 4, "training_records": 4096}


def assignment_set_sha256(assignments: Sequence[AssignmentReceipt]) -> str:
    """Digest one worker's complete, ordered assignment set for one phase."""

    if not assignments:
        raise ValueError("assignment set must not be empty")
    worker_phases = {(item.worker_id, item.phase) for item in assignments}
    if len(worker_phases) != 1:
        raise ValueError("assignment set must have one worker and phase")
    ordered = sorted(assignments, key=lambda item: (item.episode_sequence, item.episode_id))
    identities = {(item.episode_sequence, item.episode_id) for item in ordered}
    if len(identities) != len(ordered):
        raise ValueError("assignment set contains duplicate episode identity")
    return _digest([item.sha256 for item in ordered])


def validate_joined_admission(
    *,
    assignments: Sequence[AssignmentReceipt],
    activations: Sequence[PolicyActivationReceipt],
    transitions: Sequence[TransitionRecord],
    raw_transitions: Sequence[TransitionRecord],
    terminals: Sequence[TerminalReceipt],
    expected_participants: Mapping[JoinedPhase, frozenset[str]],
    expected_worker_quota: int | None = None,
) -> JoinedAdmissionReceipt:
    """Derive admission solely from closed receipt facts."""

    blockers: list[str] = []
    assignment_by_stream: dict[tuple[str, str], AssignmentReceipt] = {}
    assignment_groups: dict[tuple[JoinedPhase, str], list[AssignmentReceipt]] = {}
    for assignment_item in assignments:
        stream = (assignment_item.worker_id, assignment_item.episode_id)
        if stream in assignment_by_stream:
            blockers.append("duplicate-assignment")
        assignment_by_stream[stream] = assignment_item
        assignment_groups.setdefault(
            (assignment_item.phase, assignment_item.worker_id), []
        ).append(assignment_item)
    for phase, participants in expected_participants.items():
        observed = {
            worker for assigned_phase, worker in assignment_groups if assigned_phase == phase
        }
        if observed != set(participants):
            blockers.append("assignment-participant-set-mismatch")
    if set(assignment_groups) - {
        (phase, worker) for phase, workers in expected_participants.items() for worker in workers
    }:
        blockers.append("assignment-unplanned-participant")

    activation_by_key: dict[tuple[JoinedPhase, str], PolicyActivationReceipt] = {}
    for activation_item in activations:
        key = (activation_item.phase, activation_item.worker_id)
        if key in activation_by_key:
            blockers.append("duplicate-activation-ack")
        activation_by_key[key] = activation_item
        group = assignment_groups.get(key)
        if not group:
            blockers.append("spoofed-activation-ack")
        else:
            policies = {
                (item.run_id, item.generation, item.policy_version, item.policy_sha256)
                for item in group
            }
            expected_policy = next(iter(policies)) if len(policies) == 1 else None
            actual = (
                activation_item.run_id,
                activation_item.generation,
                activation_item.policy_version,
                activation_item.policy_sha256,
            )
            if (
                expected_policy != actual
                or activation_item.assignment_set_sha256
                != assignment_set_sha256(group)
            ):
                blockers.append("activation-identity-mismatch")
    expected_activation_keys = {
        (phase, worker) for phase, workers in expected_participants.items() for worker in workers
    }
    if set(activation_by_key) != expected_activation_keys:
        blockers.append("activation-participant-set-mismatch")

    semantic: dict[str, str] = {}
    terminal_seen: set[tuple[str, str]] = set()
    streams: dict[tuple[str, str], list[TransitionRecord]] = {}
    for record in transitions:
        stream = (record.worker_id, record.episode_id)
        assignment = assignment_by_stream.get(stream)
        if assignment is None or (
            record.run_id != assignment.run_id
            or record.generation != assignment.generation
            or record.policy_version != assignment.policy_version
            or record.info.get("behavior_artifact_sha256") != assignment.policy_sha256
        ):
            blockers.append("transition-assignment-mismatch")
        if stream in terminal_seen:
            blockers.append("transition-after-terminal")
        digest = _digest(record.to_dict())
        previous = semantic.get(record.idempotency_key)
        if previous is not None:
            blockers.append("idempotency-conflict" if previous != digest else "semantic-duplicate")
        semantic[record.idempotency_key] = digest
        streams.setdefault(stream, []).append(record)
        if record.terminated or record.truncated:
            terminal_seen.add(stream)

    for phase in expected_participants:
        phase_assignments = [item for item in assignments if item.phase == phase]
        phase_policies = {
            (item.policy_version, item.policy_sha256) for item in phase_assignments
        }
        if len(phase_policies) != 1:
            blockers.append("phase-policy-lag")
    if expected_worker_quota is not None:
        quota = _uint("expected_worker_quota", expected_worker_quota)
        counts: dict[tuple[JoinedPhase, str], int] = {}
        for stream, records in streams.items():
            assignment = assignment_by_stream.get(stream)
            if assignment is not None:
                key = (assignment.phase, assignment.worker_id)
                counts[key] = counts.get(key, 0) + len(records)
        for key in expected_activation_keys:
            if counts.get(key, 0) != quota:
                blockers.append("worker-phase-quota-mismatch")

    raw: dict[str, str] = {}
    for record in raw_transitions:
        digest = _digest(record.to_dict())
        previous = raw.get(record.idempotency_key)
        if previous is not None:
            blockers.append("raw-idempotency-conflict" if previous != digest else "raw-duplicate")
        raw[record.idempotency_key] = digest
    if len(raw_transitions) != len(transitions) or raw != semantic:
        blockers.append("raw-semantic-accounting-mismatch")

    terminal_by_stream: dict[tuple[str, str], TerminalReceipt] = {}
    for receipt in terminals:
        terminal_key = (receipt.worker_id, receipt.episode_id)
        if terminal_key in terminal_by_stream:
            blockers.append("duplicate-terminal-receipt")
        terminal_by_stream[terminal_key] = receipt
        records = streams.get(terminal_key, [])
        chain = "0" * 64
        for record in records:
            chain = hashlib.sha256(bytes.fromhex(chain) + _canonical(record.to_dict())).hexdigest()
        if not records or (
            receipt.run_id != records[0].run_id
            or receipt.generation != records[0].generation
            or receipt.final_step_id != records[-1].step_id
            or receipt.transition_count != len(records)
            or receipt.terminated != records[-1].terminated
            or receipt.truncated != records[-1].truncated
            or (receipt.phase_cut and (records[-1].terminated or records[-1].truncated))
            or receipt.transition_chain_sha256 != chain
        ):
            blockers.append("terminal-reconciliation-mismatch")
    if set(terminal_by_stream) != set(streams):
        blockers.append("terminal-coverage-mismatch")
    ordered = [_digest(item.to_dict()) for item in transitions]
    return JoinedAdmissionReceipt(
        admitted=not blockers,
        blockers=tuple(sorted(set(blockers))),
        raw_count=len(raw_transitions),
        semantic_count=len(transitions),
        unique_count=len(semantic),
        transition_sha256=_digest(ordered),
    )


__all__ = [
    "AssignmentReceipt", "EpisodeFederationTimeMapper", "FilesystemPolicyArtifactStore",
    "JoinedAdmissionError", "JoinedAdmissionReceipt", "JoinedEvaluationAssignmentV1",
    "JoinedPhase", "JoinedPhaseCutV1", "JoinedReferenceCoordinatorRuntime",
    "JoinedReferenceLaunchPlanV1", "JoinedReferenceWorkerRuntime", "PHASE_BASES",
    "PolicyActivationReceipt", "TIME_STRIDE", "TerminalReceipt", "TimeGrantReceipt",
    "assignment_set_sha256", "validate_joined_admission",
    "verify_joined_reference_process_results",
]
