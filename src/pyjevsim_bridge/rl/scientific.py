"""Claim-governed experiment identities for RL qualification campaigns.

The types in this module do not run an experiment.  They freeze the inputs
that must remain identical while local and federated rollout backends execute
one, and provide a worker-count-independent seed namespace for assigning
episodes to parallel workers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

SCIENTIFIC_MANIFEST_SCHEMA_VERSION: Final = "1"
GLOBAL_EPISODE_SEED_DERIVATION_ID: Final = "sha256-global-episode-v1"
PROJECTION_CONTRACT_ID: Final = "canonical-semantic-projection-v1"
EXCLUSION_POLICY_ID: Final = "retain-all-v1"
RERUN_POLICY_ID: Final = "infrastructure-only-v1"
DEFAULT_PROJECTION_EXCLUDED_KEYS: Final = frozenset(
    {
        "completion_order",
        "host",
        "pid",
        "wall_time",
        "wall_time_ns",
        "worker_id",
    }
)


class ScientificProtocolError(ValueError):
    """A campaign value cannot support a reproducible scientific claim."""


def _non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScientificProtocolError(f"{name} must be a non-empty string")
    return value


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScientificProtocolError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScientificProtocolError(f"{name} must be a non-negative integer")
    return value


def _sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScientificProtocolError(
            f"{name} must be a lowercase SHA-256 hex digest"
        )
    return value


def _seed_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ScientificProtocolError(f"{name} must be a sequence of seeds")
    result = tuple(
        _non_negative_integer(f"{name}[{index}]", seed)
        for index, seed in enumerate(values)
    )
    if len(result) != len(set(result)):
        raise ScientificProtocolError(f"{name} must not contain duplicate seeds")
    return result


def _string_tuple(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ScientificProtocolError(f"{name} must be a sequence of strings")
    result = tuple(
        _non_empty(f"{name}[{index}]", value)
        for index, value in enumerate(values)
    )
    if not result:
        raise ScientificProtocolError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ScientificProtocolError(f"{name} must not contain duplicates")
    return result


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScientificProtocolError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ScientificProtocolError(f"{name} must be a finite number")
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
        raise ScientificProtocolError(
            f"value is not canonical JSON compatible: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class ScientificExperimentManifest:
    """Immutable measured-campaign input and its content address."""

    experiment_id: str
    model_sha256: str
    plugin_sha256: str
    environment_contract_sha256: str
    projection_contract_sha256: str
    learner_implementation_sha256: str
    algorithm_config_sha256: str
    resource_budget_sha256: str
    tuning_seeds: tuple[int, ...]
    measured_training_master_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    environment_step_budget: int
    evaluation_interval_steps: int
    evaluation_episodes: int
    primary_endpoints: tuple[str, ...]
    quality_threshold: float
    noninferiority_margin: float
    exclusion_policy_id: str
    rerun_policy_id: str
    schema_version: str = SCIENTIFIC_MANIFEST_SCHEMA_VERSION
    seed_derivation_id: str = GLOBAL_EPISODE_SEED_DERIVATION_ID
    manifest_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCIENTIFIC_MANIFEST_SCHEMA_VERSION:
            raise ScientificProtocolError(
                f"unsupported scientific manifest schema_version {self.schema_version!r}"
            )
        _non_empty("experiment_id", self.experiment_id)
        _non_empty("seed_derivation_id", self.seed_derivation_id)
        if self.seed_derivation_id != GLOBAL_EPISODE_SEED_DERIVATION_ID:
            raise ScientificProtocolError(
                "seed_derivation_id must be worker-count independent"
            )
        for name in (
            "model_sha256",
            "plugin_sha256",
            "environment_contract_sha256",
            "projection_contract_sha256",
            "learner_implementation_sha256",
            "algorithm_config_sha256",
            "resource_budget_sha256",
        ):
            _sha256(name, getattr(self, name))

        tuning = _seed_tuple("tuning_seeds", self.tuning_seeds)
        measured = _seed_tuple(
            "measured_training_master_seeds",
            self.measured_training_master_seeds,
        )
        evaluation = _seed_tuple("evaluation_seeds", self.evaluation_seeds)
        if not tuning:
            raise ScientificProtocolError("tuning_seeds must not be empty")
        if not evaluation:
            raise ScientificProtocolError("evaluation_seeds must not be empty")
        if len(measured) < 10:
            raise ScientificProtocolError(
                "measured_training_master_seeds requires at least 10 seeds"
            )
        partitions = {
            "tuning/measured": set(tuning).intersection(measured),
            "tuning/evaluation": set(tuning).intersection(evaluation),
            "measured/evaluation": set(measured).intersection(evaluation),
        }
        overlaps = {
            name: sorted(values) for name, values in partitions.items() if values
        }
        if overlaps:
            raise ScientificProtocolError(f"campaign seed partitions overlap: {overlaps}")
        object.__setattr__(self, "tuning_seeds", tuning)
        object.__setattr__(self, "measured_training_master_seeds", measured)
        object.__setattr__(self, "evaluation_seeds", evaluation)

        _positive_integer("environment_step_budget", self.environment_step_budget)
        _positive_integer(
            "evaluation_interval_steps", self.evaluation_interval_steps
        )
        if self.evaluation_interval_steps > self.environment_step_budget:
            raise ScientificProtocolError(
                "evaluation_interval_steps must not exceed environment_step_budget"
            )
        _positive_integer("evaluation_episodes", self.evaluation_episodes)
        endpoints = _string_tuple("primary_endpoints", self.primary_endpoints)
        object.__setattr__(self, "primary_endpoints", endpoints)
        object.__setattr__(
            self,
            "quality_threshold",
            _finite("quality_threshold", self.quality_threshold),
        )
        margin = _finite("noninferiority_margin", self.noninferiority_margin)
        if margin < 0:
            raise ScientificProtocolError("noninferiority_margin must be non-negative")
        object.__setattr__(self, "noninferiority_margin", margin)
        if self.exclusion_policy_id != EXCLUSION_POLICY_ID:
            raise ScientificProtocolError(
                f"unsupported exclusion_policy_id {self.exclusion_policy_id!r}"
            )
        if self.rerun_policy_id != RERUN_POLICY_ID:
            raise ScientificProtocolError(
                f"unsupported rerun_policy_id {self.rerun_policy_id!r}"
            )
        object.__setattr__(
            self,
            "manifest_sha256",
            hashlib.sha256(_canonical_json(self.content())).hexdigest(),
        )

    def content(self) -> dict[str, object]:
        """Return the complete digest input without the digest itself."""

        return {
            "algorithm_config_sha256": self.algorithm_config_sha256,
            "environment_contract_sha256": self.environment_contract_sha256,
            "environment_step_budget": self.environment_step_budget,
            "evaluation_episodes": self.evaluation_episodes,
            "evaluation_interval_steps": self.evaluation_interval_steps,
            "evaluation_seeds": list(self.evaluation_seeds),
            "exclusion_policy_id": self.exclusion_policy_id,
            "experiment_id": self.experiment_id,
            "learner_implementation_sha256": self.learner_implementation_sha256,
            "measured_training_master_seeds": list(
                self.measured_training_master_seeds
            ),
            "model_sha256": self.model_sha256,
            "noninferiority_margin": self.noninferiority_margin,
            "plugin_sha256": self.plugin_sha256,
            "primary_endpoints": list(self.primary_endpoints),
            "projection_contract_sha256": self.projection_contract_sha256,
            "quality_threshold": self.quality_threshold,
            "rerun_policy_id": self.rerun_policy_id,
            "resource_budget_sha256": self.resource_budget_sha256,
            "schema_version": self.schema_version,
            "seed_derivation_id": self.seed_derivation_id,
            "tuning_seeds": list(self.tuning_seeds),
        }

    def verify(self, expected_sha256: str) -> None:
        """Fail before measurement if a declared content address differs."""

        if self.manifest_sha256 != _sha256(
            "expected manifest SHA-256", expected_sha256
        ):
            raise ScientificProtocolError("scientific manifest digest mismatch")


def derive_global_episode_seed(master_seed: int, global_episode_ordinal: int) -> int:
    """Derive a stable seed without worker identity or completion order."""

    master = _non_negative_integer("master_seed", master_seed)
    ordinal = _non_negative_integer(
        "global_episode_ordinal", global_episode_ordinal
    )
    digest = hashlib.sha256()
    for part in (
        GLOBAL_EPISODE_SEED_DERIVATION_ID.encode("ascii"),
        str(master).encode("ascii"),
        str(ordinal).encode("ascii"),
    ):
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def canonical_semantic_projection(
    value: object,
    *,
    excluded_keys: frozenset[str] = DEFAULT_PROJECTION_EXCLUDED_KEYS,
) -> object:
    """Remove runtime identity and quantize floats for semantic comparison."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScientificProtocolError("semantic projection contains non-finite float")
        rounded = round(value, 9)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        if any(not isinstance(key, str) for key in value):
            raise ScientificProtocolError(
                "semantic projection mapping keys must be strings"
            )
        for key in sorted(cast(str, item) for item in value):
            if key in excluded_keys:
                continue
            result[key] = canonical_semantic_projection(
                value[key], excluded_keys=excluded_keys
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            canonical_semantic_projection(item, excluded_keys=excluded_keys)
            for item in value
        ]
    raise ScientificProtocolError(
        f"semantic projection does not support {type(value).__name__}"
    )


def semantic_projection_sha256(value: object) -> str:
    """Return a content address for the transport-neutral semantic view."""

    projected = canonical_semantic_projection(value)
    return hashlib.sha256(_canonical_json(projected)).hexdigest()


def semantic_projection_contract_sha256() -> str:
    """Bind the declared projection rules to the executing module bytes."""

    implementation = __file__
    if not implementation:
        raise ScientificProtocolError("projection implementation has no source path")
    content = Path(implementation).read_bytes()
    contract = _canonical_json(
        {
            "contract_id": PROJECTION_CONTRACT_ID,
            "excluded_keys": sorted(DEFAULT_PROJECTION_EXCLUDED_KEYS),
            "float_decimal_places": 9,
            "json": "sorted-compact-utf8-no-nan",
            "negative_zero": "normalize",
            "numeric_type_distinction": True,
        }
    )
    digest = hashlib.sha256()
    digest.update(len(contract).to_bytes(8, "big"))
    digest.update(contract)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)
    return digest.hexdigest()


SEMANTIC_PROJECTION_CONTRACT_SHA256: Final = (
    semantic_projection_contract_sha256()
)
