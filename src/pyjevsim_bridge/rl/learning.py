"""Transport-neutral learner, policy artifact, and actor activation contracts.

The module deliberately uses only the Python standard library and canonical
``TransitionRecord`` values.  Framework-specific learners and policy loaders
therefore do not depend on pyjevsim execution objects or on a gorti federate.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, Self, cast

from pyjevsim_bridge.rl.records import SCHEMA_VERSION, TransitionRecord

POLICY_ARTIFACT_SCHEMA_VERSION = 1
LEARNER_RECOVERY_SCHEMA_VERSION = 1
POLICY_REGISTRY_EPISODE_ID = "policy-registry"
EMPTY_TRANSITION_BATCH_SHA256 = hashlib.sha256(b"[]").hexdigest()
TABULAR_Q_ALGORITHM_ID = "tabular-q"
TABULAR_Q_ALGORITHM_VERSION = "2"
TABULAR_Q_OBJECTIVE_ID = "decision-index-v1"
TABULAR_Q_MEDIA_TYPE = "application/vnd.pyjevsim.tabular-q+json"


class LearningContractError(RuntimeError):
    """Base failure for learner and actor contract violations."""


class TransitionBatchValidationError(LearningContractError, ValueError):
    """A candidate learner batch is incomplete, mixed, or unordered."""


class PolicyArtifactError(LearningContractError):
    """A policy artifact cannot be stored or resolved."""


class PolicyIntegrityError(PolicyArtifactError):
    """An artifact differs from its immutable reference."""


class PolicyCompatibilityError(PolicyArtifactError):
    """An artifact is not compatible with the configured actor or loader."""


class PolicyVersionError(PolicyArtifactError):
    """A policy version conflicts with an existing version or rolls it back."""


def _require_non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _freeze_json(value: object, *, path: str = "value") -> object:
    """Validate and deeply freeze a canonical-JSON-compatible value."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if key in frozen:
                raise ValueError(f"{path} contains duplicate key {key!r}")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} is not canonical JSON compatible: {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_json_mapping(name: str, value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen = _freeze_json(value, path=name)
    return cast(Mapping[str, object], frozen)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            _thaw_json(_freeze_json(value)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON compatible: {exc}") from exc


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _hash_parts(domain: bytes, *parts: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _freeze_action_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("actions must be a mapping")
    result: dict[str, object] = {}
    for worker_id, action in value.items():
        _require_non_empty("action worker_id", worker_id)
        result[worker_id] = _freeze_action_value(action)
    return MappingProxyType(result)


def _freeze_action_value(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("nested action mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze_action_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_action_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_action_value(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class PolicyCompatibility:
    """Exact algorithm/model/space identity required to load a policy."""

    algorithm_id: str
    algorithm_version: str
    model_id: str
    model_version: str
    observation_schema_sha256: str
    action_schema_sha256: str

    def __post_init__(self) -> None:
        _require_non_empty("algorithm_id", self.algorithm_id)
        _require_non_empty("algorithm_version", self.algorithm_version)
        _require_non_empty("model_id", self.model_id)
        _require_non_empty("model_version", self.model_version)
        _require_sha256("observation_schema_sha256", self.observation_schema_sha256)
        _require_sha256("action_schema_sha256", self.action_schema_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "observation_schema_sha256": self.observation_schema_sha256,
            "action_schema_sha256": self.action_schema_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("policy compatibility must be a mapping")
        fields = {
            "algorithm_id",
            "algorithm_version",
            "model_id",
            "model_version",
            "observation_schema_sha256",
            "action_schema_sha256",
        }
        if set(value) != fields:
            raise ValueError("policy compatibility fields differ from schema version 1")
        return cls(
            algorithm_id=_require_non_empty("algorithm_id", value["algorithm_id"]),
            algorithm_version=_require_non_empty(
                "algorithm_version", value["algorithm_version"]
            ),
            model_id=_require_non_empty("model_id", value["model_id"]),
            model_version=_require_non_empty("model_version", value["model_version"]),
            observation_schema_sha256=_require_sha256(
                "observation_schema_sha256", value["observation_schema_sha256"]
            ),
            action_schema_sha256=_require_sha256(
                "action_schema_sha256", value["action_schema_sha256"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidatedTransitionBatch:
    """A non-empty, single-run batch admitted to a learner adapter."""

    records: tuple[TransitionRecord, ...]
    run_id: str = field(init=False)
    generation: int = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records:
            raise TransitionBatchValidationError("transition batch must not be empty")
        if any(not isinstance(record, TransitionRecord) for record in records):
            raise TransitionBatchValidationError(
                "transition batch may contain only TransitionRecord values"
            )
        first = records[0]
        stream_positions: dict[tuple[str, str], tuple[int, float]] = {}
        idempotency_keys: set[str] = set()
        for record in records:
            if record.run_id != first.run_id:
                raise TransitionBatchValidationError("transition batch mixes run identities")
            if record.generation != first.generation:
                raise TransitionBatchValidationError("transition batch mixes generations")
            if record.idempotency_key in idempotency_keys:
                raise TransitionBatchValidationError(
                    f"transition batch repeats idempotency key {record.idempotency_key!r}"
                )
            idempotency_keys.add(record.idempotency_key)
            stream = (record.worker_id, record.episode_id)
            previous = stream_positions.get(stream)
            if previous is not None and (
                record.step_id <= previous[0] or record.logical_time < previous[1]
            ):
                raise TransitionBatchValidationError(
                    "transition batch step/time order regresses within a worker episode"
                )
            stream_positions[stream] = (record.step_id, record.logical_time)
        try:
            encoded = _canonical_json([record.to_dict() for record in records])
        except (TypeError, ValueError) as exc:
            raise TransitionBatchValidationError(
                f"transition batch is not canonically encodable: {exc}"
            ) from exc
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "run_id", first.run_id)
        object.__setattr__(self, "generation", first.generation)
        object.__setattr__(self, "sha256", hashlib.sha256(encoded).hexdigest())

    @classmethod
    def build(cls, records: Sequence[TransitionRecord]) -> Self:
        if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
            raise TypeError("records must be a non-string sequence")
        return cls(tuple(records))


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
    """Opaque framework output before the framework assigns a policy version."""

    payload: bytes
    media_type: str
    compatibility: PolicyCompatibility
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("policy payload must be non-empty bytes")
        _require_non_empty("media_type", self.media_type)
        if not isinstance(self.compatibility, PolicyCompatibility):
            raise TypeError("compatibility must be PolicyCompatibility")
        object.__setattr__(self, "payload", bytes(self.payload))
        object.__setattr__(
            self,
            "provenance",
            _freeze_json_mapping("policy provenance", self.provenance),
        )


def _policy_artifact_digest(
    *,
    run_id: str,
    generation: int,
    policy_version: int,
    media_type: str,
    compatibility: PolicyCompatibility,
    source_batch_sha256: str,
    provenance: Mapping[str, object],
    payload: bytes,
) -> str:
    manifest = {
        "schema_version": POLICY_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "generation": generation,
        "policy_version": policy_version,
        "media_type": media_type,
        "compatibility": compatibility.to_dict(),
        "source_batch_sha256": source_batch_sha256,
        "provenance": _thaw_json(provenance),
        "payload_size_bytes": len(payload),
    }
    return _hash_parts(
        b"pyjevsim-rl-policy-artifact-v1",
        _canonical_json(manifest),
        payload,
    )


@dataclass(frozen=True, slots=True)
class PolicyArtifact:
    """An immutable, content-addressed policy payload and manifest."""

    run_id: str
    generation: int
    policy_version: int
    payload: bytes
    media_type: str
    compatibility: PolicyCompatibility
    source_batch_sha256: str
    provenance: Mapping[str, object] = field(default_factory=dict)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_non_empty("run_id", self.run_id)
        _require_non_negative_int("generation", self.generation)
        _require_non_negative_int("policy_version", self.policy_version)
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("policy payload must be non-empty bytes")
        _require_non_empty("media_type", self.media_type)
        if not isinstance(self.compatibility, PolicyCompatibility):
            raise TypeError("compatibility must be PolicyCompatibility")
        _require_sha256("source_batch_sha256", self.source_batch_sha256)
        payload = bytes(self.payload)
        provenance = _freeze_json_mapping("policy provenance", self.provenance)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "sha256",
            _policy_artifact_digest(
                run_id=self.run_id,
                generation=self.generation,
                policy_version=self.policy_version,
                media_type=self.media_type,
                compatibility=self.compatibility,
                source_batch_sha256=self.source_batch_sha256,
                provenance=provenance,
                payload=payload,
            ),
        )

    @property
    def size_bytes(self) -> int:
        return len(self.payload)

    @property
    def artifact_digest(self) -> str:
        return self.sha256


@dataclass(frozen=True, slots=True)
class PolicyArtifactRef:
    """Small immutable reference suitable for an HLA policy announcement."""

    run_id: str
    generation: int
    policy_version: int
    uri: str
    sha256: str
    size_bytes: int
    media_type: str
    compatibility: PolicyCompatibility
    source_batch_sha256: str
    schema_version: int = POLICY_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("run_id", self.run_id)
        _require_non_negative_int("generation", self.generation)
        _require_non_negative_int("policy_version", self.policy_version)
        _require_non_empty("uri", self.uri)
        _require_sha256("sha256", self.sha256)
        _require_positive_int("size_bytes", self.size_bytes)
        _require_non_empty("media_type", self.media_type)
        if not isinstance(self.compatibility, PolicyCompatibility):
            raise TypeError("compatibility must be PolicyCompatibility")
        _require_sha256("source_batch_sha256", self.source_batch_sha256)
        if self.schema_version != POLICY_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported policy artifact schema_version: {self.schema_version}")

    @property
    def artifact_digest(self) -> str:
        return self.sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "policy_version": self.policy_version,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "compatibility": self.compatibility.to_dict(),
            "source_batch_sha256": self.source_batch_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("policy artifact reference must be a mapping")
        fields = {
            "schema_version",
            "run_id",
            "generation",
            "policy_version",
            "uri",
            "sha256",
            "size_bytes",
            "media_type",
            "compatibility",
            "source_batch_sha256",
        }
        if set(value) != fields:
            raise ValueError("policy artifact reference fields differ from schema version 1")
        compatibility = value["compatibility"]
        if not isinstance(compatibility, Mapping):
            raise TypeError("policy artifact compatibility must be a mapping")
        return cls(
            schema_version=_require_non_negative_int("schema_version", value["schema_version"]),
            run_id=_require_non_empty("run_id", value["run_id"]),
            generation=_require_non_negative_int("generation", value["generation"]),
            policy_version=_require_non_negative_int(
                "policy_version", value["policy_version"]
            ),
            uri=_require_non_empty("uri", value["uri"]),
            sha256=_require_sha256("sha256", value["sha256"]),
            size_bytes=_require_positive_int("size_bytes", value["size_bytes"]),
            media_type=_require_non_empty("media_type", value["media_type"]),
            compatibility=PolicyCompatibility.from_dict(
                cast(Mapping[str, object], compatibility)
            ),
            source_batch_sha256=_require_sha256(
                "source_batch_sha256", value["source_batch_sha256"]
            ),
        )


class PolicyArtifactStore(Protocol):
    """Pluggable immutable storage used by learner and actor processes."""

    async def put(self, artifact: PolicyArtifact) -> PolicyArtifactRef: ...

    async def get(self, reference: PolicyArtifactRef) -> PolicyArtifact: ...


def _validate_artifact_reference(
    reference: PolicyArtifactRef,
    artifact: PolicyArtifact,
) -> None:
    if not isinstance(reference, PolicyArtifactRef):
        raise TypeError("reference must be PolicyArtifactRef")
    if not isinstance(artifact, PolicyArtifact):
        raise TypeError("artifact store must return PolicyArtifact")
    recomputed_digest = _policy_artifact_digest(
        run_id=artifact.run_id,
        generation=artifact.generation,
        policy_version=artifact.policy_version,
        media_type=artifact.media_type,
        compatibility=artifact.compatibility,
        source_batch_sha256=artifact.source_batch_sha256,
        provenance=artifact.provenance,
        payload=artifact.payload,
    )
    if recomputed_digest != artifact.sha256:
        raise PolicyIntegrityError("policy artifact fields differ from its digest")
    identity = (artifact.run_id, artifact.generation, artifact.policy_version)
    if identity != (reference.run_id, reference.generation, reference.policy_version):
        raise PolicyIntegrityError("policy artifact identity differs from its reference")
    if artifact.sha256 != reference.sha256:
        raise PolicyIntegrityError("policy artifact digest differs from its reference")
    if artifact.size_bytes != reference.size_bytes:
        raise PolicyIntegrityError("policy artifact size differs from its reference")
    if artifact.media_type != reference.media_type:
        raise PolicyIntegrityError("policy artifact media type differs from its reference")
    if artifact.compatibility != reference.compatibility:
        raise PolicyIntegrityError("policy artifact compatibility differs from its reference")
    if artifact.source_batch_sha256 != reference.source_batch_sha256:
        raise PolicyIntegrityError("policy artifact provenance differs from its reference")


class InMemoryPolicyArtifactStore:
    """Deterministic process-local store for tests and local execution profiles."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._by_uri: dict[str, PolicyArtifact] = {}
        self._versions: dict[tuple[str, int, int], PolicyArtifactRef] = {}

    async def put(self, artifact: PolicyArtifact) -> PolicyArtifactRef:
        if not isinstance(artifact, PolicyArtifact):
            raise TypeError("artifact must be PolicyArtifact")
        key = (artifact.run_id, artifact.generation, artifact.policy_version)
        async with self._lock:
            previous = self._versions.get(key)
            if previous is not None:
                if previous.sha256 != artifact.sha256:
                    raise PolicyVersionError(
                        "policy version is already bound to a different artifact digest"
                    )
                return previous
            uri = f"memory://pyjevsim-policy/{artifact.sha256}"
            reference = PolicyArtifactRef(
                run_id=artifact.run_id,
                generation=artifact.generation,
                policy_version=artifact.policy_version,
                uri=uri,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                media_type=artifact.media_type,
                compatibility=artifact.compatibility,
                source_batch_sha256=artifact.source_batch_sha256,
            )
            occupied = self._by_uri.get(uri)
            if occupied is not None and occupied != artifact:
                raise PolicyIntegrityError("content-addressed policy URI collision")
            self._by_uri[uri] = artifact
            self._versions[key] = reference
            return reference

    async def get(self, reference: PolicyArtifactRef) -> PolicyArtifact:
        if not isinstance(reference, PolicyArtifactRef):
            raise TypeError("reference must be PolicyArtifactRef")
        async with self._lock:
            artifact = self._by_uri.get(reference.uri)
        if artifact is None:
            raise PolicyArtifactError(f"policy artifact is not available: {reference.uri}")
        _validate_artifact_reference(reference, artifact)
        return artifact


def _policy_announcement_idempotency_key(
    run_id: str,
    generation: int,
    policy_version: int,
    artifact_sha256: str,
    logical_time: float,
) -> str:
    return _hash_parts(
        b"pyjevsim-rl-policy-announcement-event-v1",
        run_id.encode("utf-8"),
        str(generation).encode("ascii"),
        str(policy_version).encode("ascii"),
        artifact_sha256.encode("ascii"),
        float(logical_time).hex().encode("ascii"),
    )


@dataclass(frozen=True, slots=True)
class PolicyAnnouncement:
    """Version-1 common envelope carrying one external policy reference."""

    run_id: str
    generation: int
    publisher_id: str
    logical_time: float
    artifact_ref: PolicyArtifactRef
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("run_id", self.run_id)
        _require_non_negative_int("generation", self.generation)
        _require_non_empty("publisher_id", self.publisher_id)
        logical_time = _require_finite("logical_time", self.logical_time)
        if logical_time < 0:
            raise ValueError("logical_time must be non-negative")
        if not isinstance(self.artifact_ref, PolicyArtifactRef):
            raise TypeError("artifact_ref must be PolicyArtifactRef")
        if (self.run_id, self.generation) != (
            self.artifact_ref.run_id,
            self.artifact_ref.generation,
        ):
            raise ValueError("policy announcement identity differs from artifact reference")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")

    @property
    def policy_version(self) -> int:
        return self.artifact_ref.policy_version

    @property
    def idempotency_key(self) -> str:
        return _policy_announcement_idempotency_key(
            self.run_id,
            self.generation,
            self.policy_version,
            self.artifact_ref.sha256,
            self.logical_time,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "worker_id": self.publisher_id,
            "episode_id": POLICY_REGISTRY_EPISODE_ID,
            "step_id": self.policy_version,
            "policy_version": self.policy_version,
            "idempotency_key": self.idempotency_key,
            "logical_time": self.logical_time,
            "payload": {"artifact_ref": self.artifact_ref.to_dict()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("policy announcement must be a mapping")
        fields = {
            "schema_version",
            "run_id",
            "generation",
            "worker_id",
            "episode_id",
            "step_id",
            "policy_version",
            "idempotency_key",
            "logical_time",
            "payload",
        }
        if set(value) != fields:
            raise ValueError("policy announcement fields differ from envelope schema version 1")
        payload = value["payload"]
        if not isinstance(payload, Mapping) or set(payload) != {"artifact_ref"}:
            raise ValueError("policy announcement payload must contain only artifact_ref")
        raw_reference = payload["artifact_ref"]
        if not isinstance(raw_reference, Mapping):
            raise TypeError("policy announcement artifact_ref must be a mapping")
        reference = PolicyArtifactRef.from_dict(
            cast(Mapping[str, object], raw_reference)
        )
        announcement = cls(
            schema_version=_require_non_negative_int("schema_version", value["schema_version"]),
            run_id=_require_non_empty("run_id", value["run_id"]),
            generation=_require_non_negative_int("generation", value["generation"]),
            publisher_id=_require_non_empty("worker_id", value["worker_id"]),
            logical_time=_require_finite("logical_time", value["logical_time"]),
            artifact_ref=reference,
        )
        if value["episode_id"] != POLICY_REGISTRY_EPISODE_ID:
            raise ValueError("policy announcement episode_id differs")
        step_id = _require_non_negative_int("step_id", value["step_id"])
        policy_version = _require_non_negative_int(
            "policy_version", value["policy_version"]
        )
        if step_id != reference.policy_version or policy_version != reference.policy_version:
            raise ValueError("policy announcement version fields differ")
        if value["idempotency_key"] != announcement.idempotency_key:
            raise ValueError("policy announcement idempotency key differs")
        return announcement

    @classmethod
    def from_envelope(cls, value: object) -> Self:
        if isinstance(value, Mapping):
            return cls.from_dict(cast(Mapping[str, object], value))
        interaction_class = getattr(value, "interaction_class", None)
        if interaction_class is not None and interaction_class != "RLPolicyAnnouncement":
            raise ValueError(
                "policy announcement interaction_class must be RLPolicyAnnouncement"
            )
        envelope = getattr(value, "envelope", None)
        if not isinstance(envelope, Mapping):
            raise TypeError("policy announcement must be a mapping or expose .envelope")
        return cls.from_dict(cast(Mapping[str, object], envelope))


class LearnerAdapter(Protocol):
    """Algorithm/framework seam; implementations see no simulator or RTI object."""

    async def update(self, batch: ValidatedTransitionBatch) -> PolicyCandidate | None: ...


@dataclass(frozen=True, slots=True)
class LearnerRecoveryState:
    """Canonical immutable learner-session state at a checkpoint boundary.

    Algorithm-specific parameters and optimizer state remain the adapter's
    responsibility.  This value preserves the generic admission state that
    prevents an already committed transition from being applied again after a
    process restart.
    """

    run_id: str
    generation: int
    next_policy_version: int
    last_published_version: int | None
    last_published_reference: PolicyArtifactRef | None
    consumed_batches: Mapping[str, PolicyArtifactRef | None]
    consumed_records: Mapping[str, str]
    stream_positions: Mapping[tuple[str, str], tuple[int, float]]
    failed: bool
    schema_version: int = LEARNER_RECOVERY_SCHEMA_VERSION
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_non_empty("run_id", self.run_id)
        _require_non_negative_int("generation", self.generation)
        next_version = _require_non_negative_int(
            "next_policy_version", self.next_policy_version
        )
        if self.last_published_version is None:
            last_version = None
        else:
            last_version = _require_non_negative_int(
                "last_published_version", self.last_published_version
            )
        if not isinstance(self.failed, bool):
            raise TypeError("failed must be a boolean")
        if self.schema_version != LEARNER_RECOVERY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported learner recovery schema_version: "
                f"{self.schema_version}"
            )

        last_reference = self.last_published_reference
        if last_reference is not None and not isinstance(
            last_reference, PolicyArtifactRef
        ):
            raise TypeError(
                "last_published_reference must be PolicyArtifactRef or None"
            )
        if last_version is None:
            if next_version != 0 or last_reference is not None:
                raise PolicyVersionError(
                    "a recovery state without a published policy must resume at version zero"
                )
        else:
            if next_version != last_version + 1:
                raise PolicyVersionError(
                    "next policy version must immediately follow the last published version"
                )
            if last_reference is None:
                raise PolicyVersionError(
                    "a published policy version requires its immutable artifact reference"
                )
            if (
                last_reference.run_id,
                last_reference.generation,
                last_reference.policy_version,
            ) != (self.run_id, self.generation, last_version):
                raise PolicyVersionError(
                    "last published reference differs from recovery identity or version"
                )

        if not isinstance(self.consumed_batches, Mapping):
            raise TypeError("consumed_batches must be a mapping")
        frozen_batches: dict[str, PolicyArtifactRef | None] = {}
        references_by_version: dict[int, PolicyArtifactRef] = {}
        for raw_digest, raw_reference in self.consumed_batches.items():
            batch_digest = _require_sha256("consumed batch SHA-256", raw_digest)
            if raw_reference is not None and not isinstance(
                raw_reference, PolicyArtifactRef
            ):
                raise TypeError(
                    "consumed batch results must be PolicyArtifactRef or None"
                )
            if raw_reference is not None:
                if (
                    raw_reference.run_id,
                    raw_reference.generation,
                ) != (self.run_id, self.generation):
                    raise PolicyVersionError(
                        "consumed batch policy reference differs from recovery identity"
                    )
                if raw_reference.source_batch_sha256 != batch_digest:
                    raise PolicyVersionError(
                        "consumed batch digest differs from policy source batch"
                    )
                if last_version is None or raw_reference.policy_version > last_version:
                    raise PolicyVersionError(
                        "consumed batch policy version exceeds the recovery cursor"
                    )
                occupied = references_by_version.get(raw_reference.policy_version)
                if occupied is not None and occupied != raw_reference:
                    raise PolicyVersionError(
                        "one policy version is bound to multiple consumed batch results"
                    )
                if occupied is not None:
                    raise PolicyVersionError(
                        "multiple consumed batches claim the same published policy version"
                    )
                references_by_version[raw_reference.policy_version] = raw_reference
            frozen_batches[batch_digest] = raw_reference

        if not isinstance(self.consumed_records, Mapping):
            raise TypeError("consumed_records must be a mapping")
        frozen_records: dict[str, str] = {}
        for raw_key, raw_digest in self.consumed_records.items():
            key = _require_non_empty("consumed record idempotency key", raw_key)
            frozen_records[key] = _require_sha256(
                "consumed record SHA-256", raw_digest
            )

        if not isinstance(self.stream_positions, Mapping):
            raise TypeError("stream_positions must be a mapping")
        frozen_positions: dict[tuple[str, str], tuple[int, float]] = {}
        for raw_stream, raw_position in self.stream_positions.items():
            if not isinstance(raw_stream, tuple) or len(raw_stream) != 2:
                raise TypeError(
                    "stream position keys must be (worker_id, episode_id) tuples"
                )
            worker_id = _require_non_empty("stream worker_id", raw_stream[0])
            episode_id = _require_non_empty("stream episode_id", raw_stream[1])
            if not isinstance(raw_position, tuple) or len(raw_position) != 2:
                raise TypeError("stream positions must be (step_id, logical_time) tuples")
            step_id = _require_non_negative_int("stream step_id", raw_position[0])
            logical_time = _require_finite("stream logical_time", raw_position[1])
            if logical_time < 0:
                raise ValueError("stream logical_time must be non-negative")
            frozen_positions[(worker_id, episode_id)] = (step_id, logical_time)

        if frozen_batches and not frozen_records:
            raise TransitionBatchValidationError(
                "consumed batch results require consumed record digests"
            )
        if bool(frozen_records) != bool(frozen_positions):
            raise TransitionBatchValidationError(
                "consumed record digests and stream cursors must both be present"
            )
        published_versions = sorted(references_by_version)
        if last_version is None:
            if published_versions:
                raise PolicyVersionError(
                    "consumed batch results exist without a published policy version"
                )
        elif published_versions:
            first_version = published_versions[0]
            if first_version not in (0, 1):
                raise PolicyVersionError(
                    "consumed policy versions must begin at version zero or one"
                )
            expected_versions = list(range(first_version, last_version + 1))
            if published_versions != expected_versions:
                raise PolicyVersionError(
                    "consumed policy versions are not contiguous through the last version"
                )
            if last_reference is None:
                raise PolicyVersionError(
                    "published policy versions require a latest artifact reference"
                )
            if references_by_version[last_version] != last_reference:
                raise PolicyVersionError(
                    "last published reference differs from the latest batch result"
                )
        elif last_version != 0:
            raise PolicyVersionError(
                "a recovery state above version zero requires consumed batch results"
            )

        object.__setattr__(self, "consumed_batches", MappingProxyType(frozen_batches))
        object.__setattr__(self, "consumed_records", MappingProxyType(frozen_records))
        object.__setattr__(self, "stream_positions", MappingProxyType(frozen_positions))
        object.__setattr__(
            self,
            "sha256",
            hashlib.sha256(_canonical_json(self._content())).hexdigest(),
        )

    def _content(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "next_policy_version": self.next_policy_version,
            "last_published_version": self.last_published_version,
            "last_published_reference": (
                None
                if self.last_published_reference is None
                else self.last_published_reference.to_dict()
            ),
            "consumed_batches": [
                {
                    "batch_sha256": batch_digest,
                    "result": None if reference is None else reference.to_dict(),
                }
                for batch_digest, reference in sorted(self.consumed_batches.items())
            ],
            "consumed_records": [
                {
                    "idempotency_key": idempotency_key,
                    "record_sha256": record_digest,
                }
                for idempotency_key, record_digest in sorted(
                    self.consumed_records.items()
                )
            ],
            "stream_positions": [
                {
                    "worker_id": worker_id,
                    "episode_id": episode_id,
                    "step_id": position[0],
                    "logical_time": position[1],
                }
                for (worker_id, episode_id), position in sorted(
                    self.stream_positions.items()
                )
            ],
            "failed": self.failed,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content()
        result["sha256"] = self.sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("learner recovery state must be a mapping")
        fields = {
            "schema_version",
            "run_id",
            "generation",
            "next_policy_version",
            "last_published_version",
            "last_published_reference",
            "consumed_batches",
            "consumed_records",
            "stream_positions",
            "failed",
            "sha256",
        }
        if set(value) != fields:
            raise ValueError("learner recovery fields differ from schema version 1")

        raw_last_version = value["last_published_version"]
        last_version = (
            None
            if raw_last_version is None
            else _require_non_negative_int(
                "last_published_version", raw_last_version
            )
        )
        raw_last_reference = value["last_published_reference"]
        if raw_last_reference is None:
            last_reference = None
        elif isinstance(raw_last_reference, Mapping):
            last_reference = PolicyArtifactRef.from_dict(
                cast(Mapping[str, object], raw_last_reference)
            )
        else:
            raise TypeError("last_published_reference must be a mapping or null")

        raw_batches = value["consumed_batches"]
        if not isinstance(raw_batches, list):
            raise TypeError("consumed_batches must be an array")
        consumed_batches: dict[str, PolicyArtifactRef | None] = {}
        for index, raw_row in enumerate(raw_batches):
            if not isinstance(raw_row, Mapping) or set(raw_row) != {
                "batch_sha256",
                "result",
            }:
                raise ValueError(f"consumed_batches[{index}] fields differ")
            batch_digest = _require_sha256(
                f"consumed_batches[{index}].batch_sha256",
                raw_row["batch_sha256"],
            )
            if batch_digest in consumed_batches:
                raise ValueError("consumed_batches repeats a batch SHA-256")
            raw_result = raw_row["result"]
            if raw_result is None:
                reference = None
            elif isinstance(raw_result, Mapping):
                reference = PolicyArtifactRef.from_dict(
                    cast(Mapping[str, object], raw_result)
                )
            else:
                raise TypeError(
                    f"consumed_batches[{index}].result must be a mapping or null"
                )
            consumed_batches[batch_digest] = reference

        raw_records = value["consumed_records"]
        if not isinstance(raw_records, list):
            raise TypeError("consumed_records must be an array")
        consumed_records: dict[str, str] = {}
        for index, raw_row in enumerate(raw_records):
            if not isinstance(raw_row, Mapping) or set(raw_row) != {
                "idempotency_key",
                "record_sha256",
            }:
                raise ValueError(f"consumed_records[{index}] fields differ")
            idempotency_key = _require_non_empty(
                f"consumed_records[{index}].idempotency_key",
                raw_row["idempotency_key"],
            )
            if idempotency_key in consumed_records:
                raise ValueError("consumed_records repeats an idempotency key")
            consumed_records[idempotency_key] = _require_sha256(
                f"consumed_records[{index}].record_sha256",
                raw_row["record_sha256"],
            )

        raw_positions = value["stream_positions"]
        if not isinstance(raw_positions, list):
            raise TypeError("stream_positions must be an array")
        stream_positions: dict[tuple[str, str], tuple[int, float]] = {}
        for index, raw_row in enumerate(raw_positions):
            if not isinstance(raw_row, Mapping) or set(raw_row) != {
                "worker_id",
                "episode_id",
                "step_id",
                "logical_time",
            }:
                raise ValueError(f"stream_positions[{index}] fields differ")
            stream = (
                _require_non_empty(
                    f"stream_positions[{index}].worker_id", raw_row["worker_id"]
                ),
                _require_non_empty(
                    f"stream_positions[{index}].episode_id", raw_row["episode_id"]
                ),
            )
            if stream in stream_positions:
                raise ValueError("stream_positions repeats a worker/episode identity")
            stream_positions[stream] = (
                _require_non_negative_int(
                    f"stream_positions[{index}].step_id", raw_row["step_id"]
                ),
                _require_finite(
                    f"stream_positions[{index}].logical_time",
                    raw_row["logical_time"],
                ),
            )

        failed = value["failed"]
        if not isinstance(failed, bool):
            raise TypeError("failed must be a boolean")
        state = cls(
            schema_version=_require_non_negative_int(
                "schema_version", value["schema_version"]
            ),
            run_id=_require_non_empty("run_id", value["run_id"]),
            generation=_require_non_negative_int("generation", value["generation"]),
            next_policy_version=_require_non_negative_int(
                "next_policy_version", value["next_policy_version"]
            ),
            last_published_version=last_version,
            last_published_reference=last_reference,
            consumed_batches=consumed_batches,
            consumed_records=consumed_records,
            stream_positions=stream_positions,
            failed=failed,
        )
        if _require_sha256("sha256", value["sha256"]) != state.sha256:
            raise PolicyIntegrityError("learner recovery state digest differs")
        return state


class LearnerSession:
    """Serialize learner updates and atomically bind successful policy versions."""

    def __init__(
        self,
        adapter: LearnerAdapter,
        store: PolicyArtifactStore,
        *,
        run_id: str,
        generation: int,
        next_policy_version: int = 0,
    ) -> None:
        self.adapter = adapter
        self.store = store
        self.run_id = _require_non_empty("run_id", run_id)
        self.generation = _require_non_negative_int("generation", generation)
        self._next_policy_version = _require_non_negative_int(
            "next_policy_version", next_policy_version
        )
        self._last_published_version: int | None = None
        self._last_published_reference: PolicyArtifactRef | None = None
        self._consumed_batches: dict[str, PolicyArtifactRef | None] = {}
        self._consumed_records: dict[str, str] = {}
        self._stream_positions: dict[tuple[str, str], tuple[int, float]] = {}
        self._failed = False
        self._lock = asyncio.Lock()

    @property
    def next_policy_version(self) -> int:
        return self._next_policy_version

    @property
    def last_published_version(self) -> int | None:
        return self._last_published_version

    @property
    def last_published_reference(self) -> PolicyArtifactRef | None:
        return self._last_published_reference

    @property
    def failed(self) -> bool:
        return self._failed

    async def snapshot_recovery_state(self) -> LearnerRecoveryState:
        """Capture one internally consistent immutable recovery boundary."""

        async with self._lock:
            return LearnerRecoveryState(
                run_id=self.run_id,
                generation=self.generation,
                next_policy_version=self._next_policy_version,
                last_published_version=self._last_published_version,
                last_published_reference=self._last_published_reference,
                consumed_batches=dict(self._consumed_batches),
                consumed_records=dict(self._consumed_records),
                stream_positions=dict(self._stream_positions),
                failed=self._failed,
            )

    @classmethod
    async def restore(
        cls,
        adapter: LearnerAdapter,
        store: PolicyArtifactStore,
        *,
        state: LearnerRecoveryState,
        run_id: str,
        generation: int,
    ) -> Self:
        """Restore a session only after all referenced artifacts re-validate."""

        if not isinstance(state, LearnerRecoveryState):
            raise TypeError("state must be LearnerRecoveryState")
        expected_run_id = _require_non_empty("run_id", run_id)
        expected_generation = _require_non_negative_int("generation", generation)
        if (state.run_id, state.generation) != (
            expected_run_id,
            expected_generation,
        ):
            raise LearningContractError(
                "learner recovery identity differs from the requested session"
            )

        references: dict[str, PolicyArtifactRef] = {}
        if state.last_published_reference is not None:
            references[
                hashlib.sha256(
                    _canonical_json(state.last_published_reference.to_dict())
                ).hexdigest()
            ] = (
                state.last_published_reference
            )
        for reference in state.consumed_batches.values():
            if reference is not None:
                references[
                    hashlib.sha256(
                        _canonical_json(reference.to_dict())
                    ).hexdigest()
                ] = reference
        for reference in references.values():
            artifact = await store.get(reference)
            _validate_artifact_reference(reference, artifact)

        raw_validator = getattr(adapter, "validate_recovery_state", None)
        if raw_validator is not None:
            if not callable(raw_validator):
                raise TypeError("learner recovery validator must be callable")
            validator = cast(Callable[[LearnerRecoveryState], None], raw_validator)
            validator(state)

        session = cls(
            adapter,
            store,
            run_id=expected_run_id,
            generation=expected_generation,
            next_policy_version=state.next_policy_version,
        )
        session._last_published_version = state.last_published_version
        session._last_published_reference = state.last_published_reference
        session._consumed_batches = dict(state.consumed_batches)
        session._consumed_records = dict(state.consumed_records)
        session._stream_positions = dict(state.stream_positions)
        session._failed = state.failed
        return session

    async def publish_initial(self, candidate: PolicyCandidate) -> PolicyArtifactRef:
        """Publish version zero before admitting experience to this session."""

        if not isinstance(candidate, PolicyCandidate):
            raise TypeError("candidate must be PolicyCandidate")
        async with self._lock:
            if self._failed:
                raise LearningContractError(
                    "learner session is failed; recover from an explicit checkpoint boundary"
                )
            if (
                self._next_policy_version != 0
                or self._last_published_version is not None
                or self._consumed_batches
            ):
                raise PolicyVersionError(
                    "initial policy may be published only at a fresh version-zero session"
                )
            try:
                artifact = PolicyArtifact(
                    run_id=self.run_id,
                    generation=self.generation,
                    policy_version=0,
                    payload=candidate.payload,
                    media_type=candidate.media_type,
                    compatibility=candidate.compatibility,
                    source_batch_sha256=EMPTY_TRANSITION_BATCH_SHA256,
                    provenance={
                        **dict(candidate.provenance),
                        "initial_policy": True,
                    },
                )
                reference = await self.store.put(artifact)
                _validate_artifact_reference(reference, artifact)
            except BaseException:
                self._failed = True
                raise
            self._last_published_version = 0
            self._last_published_reference = reference
            self._next_policy_version = 1
            return reference

    async def consume(
        self,
        records: Sequence[TransitionRecord] | ValidatedTransitionBatch,
    ) -> PolicyArtifactRef | None:
        batch = (
            records
            if isinstance(records, ValidatedTransitionBatch)
            else ValidatedTransitionBatch.build(records)
        )
        if (batch.run_id, batch.generation) != (self.run_id, self.generation):
            raise TransitionBatchValidationError(
                "learner session identity differs from transition batch"
            )
        async with self._lock:
            if self._failed:
                raise LearningContractError(
                    "learner session is failed; recover from an explicit checkpoint boundary"
                )
            if batch.sha256 in self._consumed_batches:
                return self._consumed_batches[batch.sha256]

            record_digests = {
                record.idempotency_key: hashlib.sha256(
                    _canonical_json(record.to_dict())
                ).hexdigest()
                for record in batch.records
            }
            for idempotency_key, digest in record_digests.items():
                previous_record_digest = self._consumed_records.get(idempotency_key)
                if previous_record_digest is None:
                    continue
                if previous_record_digest != digest:
                    raise TransitionBatchValidationError(
                        "transition idempotency key was previously consumed with "
                        f"different content: {idempotency_key!r}"
                    )
                raise TransitionBatchValidationError(
                    "transition batch partially overlaps or reorders previously "
                    f"consumed experience: {idempotency_key!r}"
                )
            for record in batch.records:
                stream = (record.worker_id, record.episode_id)
                previous_position = self._stream_positions.get(stream)
                if previous_position is not None and (
                    record.step_id <= previous_position[0]
                    or record.logical_time < previous_position[1]
                ):
                    raise TransitionBatchValidationError(
                        "transition batch regresses behind the learner session cursor"
                    )
            try:
                candidate = await self.adapter.update(batch)
                if candidate is None:
                    self._consumed_records.update(record_digests)
                    self._consumed_batches[batch.sha256] = None
                    self._remember_stream_positions(batch)
                    return None
                if not isinstance(candidate, PolicyCandidate):
                    raise TypeError("learner adapter must return PolicyCandidate or None")
                artifact = PolicyArtifact(
                    run_id=self.run_id,
                    generation=self.generation,
                    policy_version=self._next_policy_version,
                    payload=candidate.payload,
                    media_type=candidate.media_type,
                    compatibility=candidate.compatibility,
                    source_batch_sha256=batch.sha256,
                    provenance=candidate.provenance,
                )
                reference = await self.store.put(artifact)
                _validate_artifact_reference(reference, artifact)
            except BaseException:
                # An adapter may have mutated optimizer state before failing or
                # being cancelled.  Retrying implicitly could apply a batch twice.
                self._failed = True
                raise
            self._last_published_version = self._next_policy_version
            self._last_published_reference = reference
            self._next_policy_version += 1
            self._consumed_records.update(record_digests)
            self._consumed_batches[batch.sha256] = reference
            self._remember_stream_positions(batch)
            return reference

    def _remember_stream_positions(self, batch: ValidatedTransitionBatch) -> None:
        for record in batch.records:
            self._stream_positions[(record.worker_id, record.episode_id)] = (
                record.step_id,
                record.logical_time,
            )


class LoadedPolicy(Protocol):
    """Prepared actor-side policy that returns actions for one worker batch."""

    async def actions(
        self, observations: Mapping[str, object]
    ) -> Mapping[str, object]: ...


class PolicyLoader(Protocol):
    """Framework-specific conversion from verified bytes to an inference policy."""

    async def load(self, artifact: PolicyArtifact) -> LoadedPolicy: ...


@dataclass(frozen=True, slots=True)
class PolicyActionBatch:
    """Actions and the indivisible policy identity that selected them."""

    actions: Mapping[str, object]
    policy_version: int
    artifact_sha256: str

    def __post_init__(self) -> None:
        actions = _freeze_action_mapping(self.actions)
        if not actions:
            raise ValueError("actions must not be empty")
        object.__setattr__(self, "actions", actions)
        _require_non_negative_int("policy_version", self.policy_version)
        _require_sha256("artifact_sha256", self.artifact_sha256)


class ActorPolicy:
    """Verify, prepare, and atomically activate immutable actor policies."""

    def __init__(
        self,
        loader: PolicyLoader,
        store: PolicyArtifactStore,
        *,
        run_id: str,
        generation: int,
        compatibility: PolicyCompatibility,
    ) -> None:
        self.loader = loader
        self.store = store
        self.run_id = _require_non_empty("run_id", run_id)
        self.generation = _require_non_negative_int("generation", generation)
        if not isinstance(compatibility, PolicyCompatibility):
            raise TypeError("compatibility must be PolicyCompatibility")
        self.compatibility = compatibility
        self._active_reference: PolicyArtifactRef | None = None
        self._active_policy: LoadedPolicy | None = None
        self._activation_lock = asyncio.Lock()
        self._lock = asyncio.Lock()

    @property
    def active_reference(self) -> PolicyArtifactRef | None:
        return self._active_reference

    async def activate(self, reference: PolicyArtifactRef) -> bool:
        if not isinstance(reference, PolicyArtifactRef):
            raise TypeError("reference must be PolicyArtifactRef")
        if (reference.run_id, reference.generation) != (self.run_id, self.generation):
            raise PolicyCompatibilityError("policy run/generation differs from actor")
        if reference.compatibility != self.compatibility:
            raise PolicyCompatibilityError("policy compatibility differs from actor")

        async with self._activation_lock:
            async with self._lock:
                active = self._active_reference
                if active is not None:
                    if reference.policy_version < active.policy_version:
                        raise PolicyVersionError("policy activation would roll back the actor")
                    if reference.policy_version == active.policy_version:
                        if reference != active:
                            raise PolicyVersionError(
                                "active policy version conflicts with a different reference"
                            )
                        return False

            # Fetch, integrity-check, and load before taking the state lock.
            # Any failure therefore leaves the active policy untouched.
            artifact = await self.store.get(reference)
            _validate_artifact_reference(reference, artifact)
            loaded = await self.loader.load(artifact)
            if not callable(getattr(loaded, "actions", None)):
                raise TypeError("policy loader must return an object with actions()")

            async with self._lock:
                self._active_reference = reference
                self._active_policy = loaded
                return True

    async def actions(
        self,
        observations: Mapping[str, object],
    ) -> PolicyActionBatch:
        if not isinstance(observations, Mapping) or not observations:
            raise ValueError("observations must be a non-empty mapping")
        expected_workers = set(observations)
        if any(not isinstance(worker_id, str) or not worker_id for worker_id in expected_workers):
            raise ValueError("observation worker IDs must be non-empty strings")
        async with self._lock:
            reference = self._active_reference
            policy = self._active_policy
            if reference is None or policy is None:
                raise PolicyVersionError("actor has no active policy")
        selected = await policy.actions(observations)
        if not isinstance(selected, Mapping):
            raise TypeError("loaded policy actions() must return a mapping")
        if set(selected) != expected_workers:
            raise ValueError("loaded policy action worker IDs differ from observations")
        return PolicyActionBatch(
            actions=selected,
            policy_version=reference.policy_version,
            artifact_sha256=reference.sha256,
        )


def _tabular_key(value: object, *, name: str) -> str:
    try:
        return _canonical_json(value).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LearningContractError(f"{name} is not a tabular JSON value: {exc}") from exc


class TabularQLearnerAdapter:
    """Deterministic stdlib Q-learning reference adapter.

    It intentionally performs greedy publication only; exploration belongs to
    an actor strategy or experiment runner so the artifact is deterministic for
    a fixed ordered transition batch.
    """

    def __init__(
        self,
        actions: Sequence[object],
        *,
        compatibility: PolicyCompatibility,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
    ) -> None:
        if isinstance(actions, (str, bytes, bytearray)) or not isinstance(actions, Sequence):
            raise TypeError("tabular actions must be a non-string sequence")
        if not actions:
            raise ValueError("tabular actions must not be empty")
        if not isinstance(compatibility, PolicyCompatibility):
            raise TypeError("compatibility must be PolicyCompatibility")
        if (
            compatibility.algorithm_id != TABULAR_Q_ALGORITHM_ID
            or compatibility.algorithm_version != TABULAR_Q_ALGORITHM_VERSION
        ):
            raise PolicyCompatibilityError(
                "tabular learner requires algorithm compatibility tabular-q version 2"
            )
        alpha = _require_finite("learning_rate", learning_rate)
        gamma = _require_finite("discount_factor", discount_factor)
        if not 0 < alpha <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= gamma <= 1:
            raise ValueError("discount_factor must be in [0, 1]")
        frozen_actions = tuple(
            _freeze_json(action, path=f"actions[{index}]")
            for index, action in enumerate(actions)
        )
        action_keys = tuple(
            _tabular_key(action, name=f"actions[{index}]")
            for index, action in enumerate(frozen_actions)
        )
        if len(set(action_keys)) != len(action_keys):
            raise ValueError("tabular actions must be canonically unique")
        self.compatibility = compatibility
        self.learning_rate = alpha
        self.discount_factor = gamma
        self._actions = frozen_actions
        self._action_indices = {
            action_key: index for index, action_key in enumerate(action_keys)
        }
        self._q_values: dict[str, list[float]] = {}
        self._transitions_applied = 0
        self._stream_cursors: dict[tuple[str, int, str, str], tuple[int, float]] = {}
        self._closed_streams: set[tuple[str, int, str, str]] = set()

    async def update(self, batch: ValidatedTransitionBatch) -> PolicyCandidate:
        if not isinstance(batch, ValidatedTransitionBatch):
            raise TypeError("batch must be ValidatedTransitionBatch")
        # Keep a rejected late record or overflow from partially training state.
        q_values = {key: list(values) for key, values in self._q_values.items()}
        cursors = dict(self._stream_cursors)
        closed = set(self._closed_streams)
        for record in batch.records:
            stream = (record.run_id, record.generation, record.worker_id, record.episode_id)
            if stream in closed:
                raise LearningContractError("tabular learner cannot reopen a closed episode stream")
            cursor = cursors.get(stream)
            if cursor is not None and (
                record.step_id != cursor[0] + 1 or record.logical_time < cursor[1]
            ):
                raise LearningContractError(
                    "tabular stream must remain step-contiguous and time-ordered"
                )
            state_key = _tabular_key(record.previous_observation, name="previous_observation")
            next_state_key = _tabular_key(record.next_observation, name="next_observation")
            action_key = _tabular_key(record.action, name="action")
            action_index = self._action_indices.get(action_key)
            if action_index is None:
                raise LearningContractError("transition action is outside tabular action space")
            previous_mask = self._action_mask(record, "previous_action_mask", allow_empty=False)
            next_mask = self._action_mask(
                record, "next_action_mask", allow_empty=record.terminated
            )
            if not previous_mask[action_index]:
                raise LearningContractError("transition action is masked in previous observation")
            state_values = q_values.setdefault(
                state_key, [0.0] * len(self._actions)
            )
            next_values = q_values.setdefault(
                next_state_key, [0.0] * len(self._actions)
            )
            continuation = 0.0 if record.terminated else max(
                value for value, allowed in zip(next_values, next_mask, strict=True) if allowed
            )
            target = record.reward + self.discount_factor * continuation
            previous = state_values[action_index]
            updated = previous + self.learning_rate * (target - previous)
            if not math.isfinite(updated):
                raise LearningContractError("tabular Q update produced a non-finite value")
            state_values[action_index] = updated
            cursors[stream] = (record.step_id, record.logical_time)
            if record.terminated or record.truncated:
                closed.add(stream)
        self._q_values = q_values
        self._stream_cursors = cursors
        self._closed_streams = closed
        self._transitions_applied += len(batch.records)
        return self._candidate(
            batch_sha256=batch.sha256,
            initial_policy=False,
        )

    def _action_mask(
        self, record: TransitionRecord, name: str, *, allow_empty: bool
    ) -> tuple[bool, ...]:
        if name not in record.info:
            return (True,) * len(self._actions)
        value = record.info[name]
        if (
            not isinstance(value, tuple)
            or len(value) != len(self._actions)
            or any(type(item) is not bool for item in value)
            or (not allow_empty and not any(value))
        ):
            raise LearningContractError(f"{name} is not a valid tabular action mask")
        return cast(tuple[bool, ...], value)

    def initial_policy(self) -> PolicyCandidate:
        """Return the deterministic all-zero policy before any experience."""

        if self._transitions_applied != 0 or self._q_values:
            raise LearningContractError(
                "initial policy is available only before the first learner update"
            )
        return self._candidate(
            batch_sha256=EMPTY_TRANSITION_BATCH_SHA256,
            initial_policy=True,
        )

    def _candidate(
        self,
        *,
        batch_sha256: str,
        initial_policy: bool,
    ) -> PolicyCandidate:
        policy = {
            "schema_version": 1,
            "algorithm_id": TABULAR_Q_ALGORITHM_ID,
            "algorithm_version": TABULAR_Q_ALGORITHM_VERSION,
            "actions": [_thaw_json(action) for action in self._actions],
            "q_values": [
                {
                    "state": json.loads(state_key),
                    "values": list(self._q_values[state_key]),
                }
                for state_key in sorted(self._q_values)
            ],
        }
        payload = _canonical_json(policy)
        return PolicyCandidate(
            payload=payload,
            media_type=TABULAR_Q_MEDIA_TYPE,
            compatibility=self.compatibility,
            provenance={
                "adapter": "pyjevsim_bridge.rl.TabularQLearnerAdapter",
                "batch_sha256": batch_sha256,
                "discount_factor": self.discount_factor,
                "objective_id": TABULAR_Q_OBJECTIVE_ID,
                "initial_policy": initial_policy,
                "learning_rate": self.learning_rate,
                "transitions_applied": self._transitions_applied,
            },
        )


@dataclass(frozen=True, slots=True)
class _LoadedTabularPolicy:
    actions_by_index: tuple[object, ...]
    q_values: Mapping[str, tuple[float, ...]]

    async def actions(
        self,
        observations: Mapping[str, object],
    ) -> Mapping[str, object]:
        selected: dict[str, object] = {}
        for worker_id, observation in observations.items():
            state_key = _tabular_key(observation, name="observation")
            values = self.q_values.get(state_key)
            action_index = 0
            if values is not None:
                action_index = max(range(len(values)), key=lambda index: (values[index], -index))
            selected[worker_id] = _thaw_json(self.actions_by_index[action_index])
        return selected


class TabularPolicyLoader:
    """Strict loader for artifacts emitted by ``TabularQLearnerAdapter``."""

    async def load(self, artifact: PolicyArtifact) -> LoadedPolicy:
        if not isinstance(artifact, PolicyArtifact):
            raise TypeError("artifact must be PolicyArtifact")
        compatibility = artifact.compatibility
        if (
            compatibility.algorithm_id != TABULAR_Q_ALGORITHM_ID
            or compatibility.algorithm_version != TABULAR_Q_ALGORITHM_VERSION
        ):
            raise PolicyCompatibilityError(
                "tabular loader requires algorithm compatibility tabular-q version 2"
            )
        if artifact.media_type != TABULAR_Q_MEDIA_TYPE:
            raise PolicyCompatibilityError("tabular policy media type differs")
        try:
            decoded = json.loads(
                artifact.payload.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise PolicyIntegrityError(f"tabular policy payload is invalid JSON: {exc}") from exc
        if not isinstance(decoded, dict) or set(decoded) != {
            "schema_version",
            "algorithm_id",
            "algorithm_version",
            "actions",
            "q_values",
        }:
            raise PolicyIntegrityError("tabular policy fields differ from schema version 1")
        if (
            isinstance(decoded["schema_version"], bool)
            or decoded["schema_version"] != 1
            or decoded["algorithm_id"] != TABULAR_Q_ALGORITHM_ID
            or decoded["algorithm_version"] != TABULAR_Q_ALGORITHM_VERSION
        ):
            raise PolicyIntegrityError("tabular policy schema or algorithm identity differs")
        raw_actions = decoded["actions"]
        raw_rows = decoded["q_values"]
        if not isinstance(raw_actions, list) or not raw_actions:
            raise PolicyIntegrityError("tabular policy actions must be a non-empty array")
        if not isinstance(raw_rows, list):
            raise PolicyIntegrityError("tabular policy q_values must be an array")
        try:
            actions = tuple(
                _freeze_json(action, path=f"actions[{index}]")
                for index, action in enumerate(raw_actions)
            )
            action_keys = [
                _tabular_key(action, name="policy action") for action in actions
            ]
        except (LearningContractError, TypeError, ValueError) as exc:
            raise PolicyIntegrityError(
                f"tabular policy actions are invalid: {exc}"
            ) from exc
        if len(set(action_keys)) != len(action_keys):
            raise PolicyIntegrityError("tabular policy actions are not unique")
        rows: dict[str, tuple[float, ...]] = {}
        for row_index, row in enumerate(raw_rows):
            if not isinstance(row, dict) or set(row) != {"state", "values"}:
                raise PolicyIntegrityError(f"tabular q_values[{row_index}] fields differ")
            try:
                state_key = _tabular_key(
                    row["state"], name=f"q_values[{row_index}].state"
                )
            except (LearningContractError, TypeError, ValueError) as exc:
                raise PolicyIntegrityError(
                    f"tabular policy state is invalid: {exc}"
                ) from exc
            if state_key in rows:
                raise PolicyIntegrityError("tabular policy repeats a state")
            raw_values = row["values"]
            if not isinstance(raw_values, list) or len(raw_values) != len(actions):
                raise PolicyIntegrityError("tabular policy Q row width differs from actions")
            try:
                values = tuple(
                    _require_finite(f"q_values[{row_index}].values[{index}]", item)
                    for index, item in enumerate(raw_values)
                )
            except ValueError as exc:
                raise PolicyIntegrityError(
                    f"tabular policy Q values are invalid: {exc}"
                ) from exc
            rows[state_key] = values
        return _LoadedTabularPolicy(
            actions_by_index=actions,
            q_values=MappingProxyType(rows),
        )


__all__ = [
    "EMPTY_TRANSITION_BATCH_SHA256",
    "LEARNER_RECOVERY_SCHEMA_VERSION",
    "POLICY_ARTIFACT_SCHEMA_VERSION",
    "POLICY_REGISTRY_EPISODE_ID",
    "TABULAR_Q_ALGORITHM_ID",
    "TABULAR_Q_ALGORITHM_VERSION",
    "TABULAR_Q_OBJECTIVE_ID",
    "TABULAR_Q_MEDIA_TYPE",
    "ActorPolicy",
    "InMemoryPolicyArtifactStore",
    "LearnerAdapter",
    "LearnerRecoveryState",
    "LearnerSession",
    "LearningContractError",
    "LoadedPolicy",
    "PolicyActionBatch",
    "PolicyAnnouncement",
    "PolicyArtifact",
    "PolicyArtifactError",
    "PolicyArtifactRef",
    "PolicyArtifactStore",
    "PolicyCandidate",
    "PolicyCompatibility",
    "PolicyCompatibilityError",
    "PolicyIntegrityError",
    "PolicyLoader",
    "PolicyVersionError",
    "TabularPolicyLoader",
    "TabularQLearnerAdapter",
    "TransitionBatchValidationError",
    "ValidatedTransitionBatch",
]
