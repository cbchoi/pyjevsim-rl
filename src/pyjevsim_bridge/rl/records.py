"""Immutable records shared by local and federated RL rollout paths.

The records intentionally depend only on the Python standard library.  They
are transport-neutral: a gorti codec may encode them later without making the
local rollout implementation depend on the RTI SDK.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

SCHEMA_VERSION = 1


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _require_schema_version(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {value}")


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return sorted((_thaw(item) for item in value), key=repr)
    return copy.deepcopy(value)


def _frozen_info(info: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(info, Mapping):
        raise TypeError("info must be a mapping")
    frozen = _freeze(info)
    return cast(Mapping[str, object], frozen)


@dataclass(frozen=True, slots=True)
class ActionCommand:
    """A policy action addressed to exactly one environment episode."""

    run_id: str
    generation: int
    worker_id: str
    episode_id: str
    step_id: int
    policy_version: int
    idempotency_key: str
    logical_time: float
    payload: object
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("run_id", self.run_id)
        _require_non_negative_int("generation", self.generation)
        _require_non_empty("worker_id", self.worker_id)
        _require_non_empty("episode_id", self.episode_id)
        _require_non_negative_int("step_id", self.step_id)
        _require_non_negative_int("policy_version", self.policy_version)
        _require_non_empty("idempotency_key", self.idempotency_key)
        _require_finite("logical_time", self.logical_time)
        _require_schema_version(self.schema_version)
        object.__setattr__(self, "payload", _freeze(self.payload))

    @property
    def action(self) -> object:
        """Local-friendly alias for the transport envelope payload."""

        return self.payload

    def to_dict(self) -> dict[str, object]:
        """Return the version-1 federation envelope shape."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "worker_id": self.worker_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "policy_version": self.policy_version,
            "idempotency_key": self.idempotency_key,
            "logical_time": self.logical_time,
            "payload": _thaw(self.payload),
        }


@dataclass(frozen=True, slots=True)
class ResetResult:
    """Initial observation and provenance returned for a worker reset."""

    run_id: str
    generation: int
    worker_id: str
    episode_id: str
    seed: int | None
    observation: object
    info: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("run_id", self.run_id)
        _require_non_negative_int("generation", self.generation)
        _require_non_empty("worker_id", self.worker_id)
        _require_non_empty("episode_id", self.episode_id)
        if self.seed is not None:
            _require_non_negative_int("seed", self.seed)
        _require_schema_version(self.schema_version)
        object.__setattr__(self, "observation", _freeze(self.observation))
        object.__setattr__(self, "info", _frozen_info(self.info))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly local reset record."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "worker_id": self.worker_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "observation": _thaw(self.observation),
            "info": _thaw(self.info),
        }


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """A committed environment transition with complete learning provenance."""

    run_id: str
    generation: int
    worker_id: str
    episode_id: str
    step_id: int
    policy_version: int
    idempotency_key: str
    logical_time: float
    previous_observation: object
    action: object
    next_observation: object
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("run_id", self.run_id)
        _require_non_negative_int("generation", self.generation)
        _require_non_empty("worker_id", self.worker_id)
        _require_non_empty("episode_id", self.episode_id)
        _require_non_negative_int("step_id", self.step_id)
        _require_non_negative_int("policy_version", self.policy_version)
        _require_non_empty("idempotency_key", self.idempotency_key)
        _require_finite("logical_time", self.logical_time)
        _require_finite("reward", self.reward)
        if not isinstance(self.terminated, bool):
            raise TypeError("terminated must be bool")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be bool")
        _require_schema_version(self.schema_version)
        object.__setattr__(
            self, "previous_observation", _freeze(self.previous_observation)
        )
        object.__setattr__(self, "action", _freeze(self.action))
        object.__setattr__(self, "next_observation", _freeze(self.next_observation))
        object.__setattr__(self, "info", _frozen_info(self.info))

    @property
    def observation(self) -> object:
        """Gym-compatible alias for the transition's next observation."""

        return self.next_observation

    def to_dict(self) -> dict[str, object]:
        """Return the version-1 envelope used by ``GortiRolloutChannel``."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "worker_id": self.worker_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "policy_version": self.policy_version,
            "idempotency_key": self.idempotency_key,
            "logical_time": self.logical_time,
            "payload": {
                "previous_observation": _thaw(self.previous_observation),
                "action": _thaw(self.action),
                "observation": _thaw(self.next_observation),
                "reward": self.reward,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "info": _thaw(self.info),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TransitionRecord:
        """Build a typed transition from a local or federation envelope.

        A federation sender may move the original model-committed logical time
        into ``payload.simulation_time`` and replace the outer ``logical_time``
        with the later HLA delivery timestamp.  Learning provenance must retain
        the former.  The outer timestamp is still validated before it is
        discarded so a malformed wire envelope cannot enter a learner batch.
        """

        if not isinstance(value, Mapping):
            raise TypeError("transition envelope must be a mapping")
        required_fields = {
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
        supplied_fields = set(value)
        missing = sorted(required_fields - supplied_fields)
        unknown = sorted(supplied_fields - required_fields)
        if missing:
            raise ValueError(f"transition envelope is missing fields: {missing}")
        if unknown:
            raise ValueError(f"transition envelope has unknown fields: {unknown}")

        payload_value = value["payload"]
        if not isinstance(payload_value, Mapping):
            raise TypeError("transition payload must be a mapping")
        required_payload_fields = {
            "previous_observation",
            "action",
            "observation",
            "reward",
            "terminated",
            "truncated",
            "info",
        }
        optional_payload_fields = {"simulation_time"}
        supplied_payload_fields = set(payload_value)
        missing_payload = sorted(required_payload_fields - supplied_payload_fields)
        unknown_payload = sorted(
            supplied_payload_fields - required_payload_fields - optional_payload_fields
        )
        if missing_payload:
            raise ValueError(f"transition payload is missing fields: {missing_payload}")
        if unknown_payload:
            raise ValueError(f"transition payload has unknown fields: {unknown_payload}")

        delivery_time = _require_finite("logical_time", value["logical_time"])
        simulation_time = _require_finite(
            "simulation_time", payload_value.get("simulation_time", delivery_time)
        )
        if simulation_time > delivery_time:
            raise ValueError("simulation_time must be no later than envelope logical_time")

        info_value = payload_value["info"]
        if not isinstance(info_value, Mapping):
            raise TypeError("transition payload info must be a mapping")
        reward_value = _require_finite("reward", payload_value["reward"])
        terminated_value = payload_value["terminated"]
        truncated_value = payload_value["truncated"]
        if not isinstance(terminated_value, bool):
            raise TypeError("transition payload terminated must be bool")
        if not isinstance(truncated_value, bool):
            raise TypeError("transition payload truncated must be bool")

        return cls(
            run_id=cast(str, value["run_id"]),
            generation=cast(int, value["generation"]),
            worker_id=cast(str, value["worker_id"]),
            episode_id=cast(str, value["episode_id"]),
            step_id=cast(int, value["step_id"]),
            policy_version=cast(int, value["policy_version"]),
            idempotency_key=cast(str, value["idempotency_key"]),
            logical_time=float(simulation_time),
            previous_observation=payload_value["previous_observation"],
            action=payload_value["action"],
            next_observation=payload_value["observation"],
            reward=float(reward_value),
            terminated=terminated_value,
            truncated=truncated_value,
            info=cast(Mapping[str, object], info_value),
            schema_version=cast(int, value["schema_version"]),
        )

    @classmethod
    def from_envelope(cls, value: object) -> TransitionRecord:
        """Build from a mapping or a duck-typed ``ReceivedEnvelope`` object."""

        if isinstance(value, Mapping):
            return cls.from_dict(cast(Mapping[str, object], value))
        interaction_class = getattr(value, "interaction_class", None)
        if interaction_class is not None and interaction_class != "RLTransition":
            raise ValueError(
                "transition envelope interaction_class must be RLTransition"
            )
        envelope = getattr(value, "envelope", None)
        if not isinstance(envelope, Mapping):
            raise TypeError("transition envelope must be a mapping or expose .envelope")
        return cls.from_dict(cast(Mapping[str, object], envelope))
