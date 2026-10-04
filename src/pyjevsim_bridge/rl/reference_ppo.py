"""Deterministic model-neutral NumPy CPU reference PPO.

The adapter implements the generic :class:`LearnerAdapter` seam.  It receives
only immutable transition batches and publishes opaque candidates through the
existing learner/policy lifecycle.  It never owns a simulator or rollout
transport.  Floating-point reproducibility is intentionally limited to a
source/runtime/CPU/thread-locked NumPy profile.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import platform
import struct
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Final, NoReturn, Self, cast

import numpy as np
import numpy.typing as npt

from pyjevsim_bridge.rl.inference_metrics import (
    LOADED_INFERENCE_METRICS_SOURCE_SHA256,
    file_sha256,
    inference_metrics_source_sha256,
    metric_function,
    metric_span,
)
from pyjevsim_bridge.rl.learning import (
    EMPTY_TRANSITION_BATCH_SHA256,
    LearnerRecoveryState,
    LearningContractError,
    LoadedPolicy,
    PolicyArtifact,
    PolicyArtifactRef,
    PolicyCandidate,
    PolicyCompatibility,
    PolicyCompatibilityError,
    PolicyIntegrityError,
    ValidatedTransitionBatch,
)
from pyjevsim_bridge.rl.ppo_features import (
    LOADED_PPO_FEATURES_SOURCE_SHA256,
    PPO_INFERENCE_INPUT_SCHEMA_VERSION,
    FeatureContractBinding,
    FeatureContractError,
    PPOFeatureContract,
    _scoped_feature_binding,
    bind_feature_contract,
    ppo_features_source_sha256,
    validate_feature_profile,
)
from pyjevsim_bridge.rl.records import TransitionRecord

REFERENCE_PPO_ALGORITHM_ID: Final = "reference-ppo"
REFERENCE_PPO_ALGORITHM_VERSION: Final = "2"
REFERENCE_PPO_OBJECTIVE_ID: Final = "decision-index-v1"
REFERENCE_PPO_MEDIA_TYPE: Final = "application/vnd.pyjevsim.reference-ppo+json"
REFERENCE_PPO_CHECKPOINT_SCHEMA: Final = "reference-ppo-checkpoint-v2"
INITIALIZATION_DOMAIN: Final = "reference-ppo-xavier-v1"
SAMPLING_DOMAIN: Final = "reference-ppo-categorical-v1"
SHUFFLE_DOMAIN: Final = "reference-ppo-minibatch-order-v1"
TRANSITION_CHAIN_DOMAIN: Final = "reference-ppo-transition-chain-v1"
ZERO_SHA256: Final = "0" * 64

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[Any]
IntArray = npt.NDArray[np.int64]

_POLICY_SHAPES: Final = (
    ("policy.w1", lambda c: (c.hidden_size, c.observation_size)),
    ("policy.b1", lambda c: (c.hidden_size,)),
    ("policy.w2", lambda c: (c.hidden_size, c.hidden_size)),
    ("policy.b2", lambda c: (c.hidden_size,)),
    ("policy.w3", lambda c: (c.action_count, c.hidden_size)),
    ("policy.b3", lambda c: (c.action_count,)),
)
_VALUE_SHAPES: Final = (
    ("value.w1", lambda c: (c.hidden_size, c.observation_size)),
    ("value.b1", lambda c: (c.hidden_size,)),
    ("value.w2", lambda c: (c.hidden_size, c.hidden_size)),
    ("value.b2", lambda c: (c.hidden_size,)),
    ("value.w3", lambda c: (1, c.hidden_size)),
    ("value.b3", lambda c: (1,)),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reference_ppo_source_sha256() -> str:
    return file_sha256(Path(__file__), role="reference_ppo_source")


LOADED_REFERENCE_PPO_SOURCE_SHA256: Final = reference_ppo_source_sha256()


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ReferencePPOConfigV1:
    """Closed PPO defaults; bounded tests may explicitly override dimensions/budgets."""

    observation_size: int = 80
    hidden_size: int = 64
    action_count: int = 6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-5
    max_gradient_norm: float = 0.5
    advantage_epsilon: float = 1e-8
    batch_size: int = 2048
    minibatch_size: int = 256
    epochs: int = 10
    initialization_seed: int = 0
    shuffle_seed: int = 0
    dtype: str = "float32"
    device: str = "cpu"
    schema_version: str = "reference-ppo-config-v1"
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "observation_size",
            "hidden_size",
            "action_count",
            "batch_size",
            "minibatch_size",
            "epochs",
        ):
            _positive_int(name, getattr(self, name))
        _non_negative_int("initialization_seed", self.initialization_seed)
        _non_negative_int("shuffle_seed", self.shuffle_seed)
        bounded = {
            "gamma": (self.gamma, 0.0, 1.0, True),
            "gae_lambda": (self.gae_lambda, 0.0, 1.0, True),
            "policy_clip": (self.policy_clip, 0.0, 1.0, False),
            "value_coefficient": (self.value_coefficient, 0.0, math.inf, False),
            "entropy_coefficient": (self.entropy_coefficient, 0.0, math.inf, True),
            "learning_rate": (self.learning_rate, 0.0, math.inf, False),
            "adam_beta1": (self.adam_beta1, 0.0, 1.0, False),
            "adam_beta2": (self.adam_beta2, 0.0, 1.0, False),
            "adam_epsilon": (self.adam_epsilon, 0.0, math.inf, False),
            "max_gradient_norm": (self.max_gradient_norm, 0.0, math.inf, False),
            "advantage_epsilon": (self.advantage_epsilon, 0.0, math.inf, False),
        }
        for name, (raw, low, high, low_inclusive) in bounded.items():
            value = _finite(name, raw)
            low_ok = value >= low if low_inclusive else value > low
            if not low_ok or value >= high:
                raise ValueError(f"{name} is outside its closed PPO domain")
        if self.minibatch_size > self.batch_size:
            raise ValueError("minibatch_size must not exceed batch_size")
        if self.dtype != "float32" or self.device != "cpu":
            raise ValueError("reference PPO requires NumPy CPU float32")
        object.__setattr__(self, "sha256", _sha256(_canonical_json(self.content())))

    @property
    def is_frozen_production(self) -> bool:
        return self.sha256 == FROZEN_REFERENCE_PPO_CONFIG_SHA256

    def content(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "sha256"
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        expected = {item.name for item in fields(cls) if item.name != "sha256"}
        if set(value) != expected:
            raise PolicyIntegrityError("reference PPO config fields differ")
        try:
            return cls(**cast(Any, dict(value)))
        except (TypeError, ValueError) as exc:
            raise PolicyIntegrityError(f"reference PPO config is invalid: {exc}") from exc


FROZEN_REFERENCE_PPO_CONFIG_SHA256: Final = ReferencePPOConfigV1().sha256


@dataclass(frozen=True, slots=True)
class PPOInferenceInputV1:
    """Observation plus contemporaneous mask and stateless sampling identity."""

    observation: object
    action_mask: tuple[bool, ...]
    sampling_seed: int
    run_id: str
    generation: int
    worker_id: str
    episode_id: str
    step_id: int
    explore: bool = True

    @metric_function("input_validation")
    def __post_init__(self) -> None:
        mask = tuple(self.action_mask)
        if not mask or any(type(item) is not bool for item in mask) or not any(mask):
            raise ValueError("action_mask must contain booleans with one valid action")
        object.__setattr__(self, "action_mask", mask)
        _non_negative_int("sampling_seed", self.sampling_seed)
        _non_negative_int("generation", self.generation)
        _non_negative_int("step_id", self.step_id)
        for name in ("run_id", "worker_id", "episode_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.explore) is not bool:
            raise TypeError("explore must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PPO_INFERENCE_INPUT_SCHEMA_VERSION,
            "observation": self.observation,
            "action_mask": self.action_mask,
            "run_id": self.run_id,
            "generation": self.generation,
            "worker_id": self.worker_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "sampling_seed": self.sampling_seed,
            "explore": self.explore,
        }


@dataclass(frozen=True, slots=True)
class PPOLossMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    gradient_norm: float
    clipped_fraction: float

    def content(self) -> dict[str, float]:
        return {item.name: float(getattr(self, item.name)) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class ReferencePPOCheckpointV1:
    """Legacy Python API spelling; only current wire-schema-v2 is accepted."""

    payload: bytes
    policy_version: int = field(init=False)
    update_count: int = field(init=False)
    environment_steps: int = field(init=False)
    config_sha256: str = field(init=False)
    feature_contract_sha256: str = field(init=False)
    policy_parameter_sha256: str = field(init=False)
    value_parameter_sha256: str = field(init=False)
    optimizer_sha256: str = field(init=False)
    transition_chain_sha256: str = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        decoded = _decode_payload(self.payload)
        _validate_source_runtime(decoded)
        _validate_checkpoint_contents(decoded)
        for name in (
            "config_sha256",
            "feature_contract_sha256",
            "policy_parameter_sha256",
            "value_parameter_sha256",
            "optimizer_sha256",
            "transition_chain_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(name, decoded[name]))
        for name in ("policy_version", "update_count", "environment_steps"):
            object.__setattr__(self, name, _non_negative_int(name, decoded[name]))
        object.__setattr__(self, "payload", bytes(self.payload))
        object.__setattr__(self, "sha256", _sha256(self.payload))


@dataclass(frozen=True, slots=True)
class ReferencePPOCapabilityReceipt:
    """Computed implementation capability; never a learning-quality claim."""

    source_sha256: str
    feature_source_sha256: str
    config_sha256: str
    feature_contract_sha256: str
    numpy_runtime_sha256: str
    initial_policy_sha256: str
    final_policy_sha256: str
    initial_value_sha256: str
    final_value_sha256: str
    initial_optimizer_sha256: str
    final_optimizer_sha256: str
    initial_checkpoint_sha256: str
    final_checkpoint_sha256: str
    update_count: int
    adam_step_count: int
    qualification_adam_step_count: int
    environment_steps: int
    checkpoint_reloaded: bool
    resume_bytes_matched: bool
    mask_violation_count: int
    evaluation_leak_count: int
    raw_ledger_sha256: str
    raw_artifact_count: int
    claim_scope: str = field(init=False, default="reference-ppo-local-qualification-only")
    blockers: tuple[str, ...] = field(init=False)
    admitted: bool = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        digest_names = (
            "source_sha256",
            "feature_source_sha256",
            "config_sha256",
            "feature_contract_sha256",
            "numpy_runtime_sha256",
            "initial_policy_sha256",
            "final_policy_sha256",
            "initial_value_sha256",
            "final_value_sha256",
            "initial_optimizer_sha256",
            "final_optimizer_sha256",
            "initial_checkpoint_sha256",
            "final_checkpoint_sha256",
            "raw_ledger_sha256",
        )
        for name in digest_names:
            _require_sha256(name, getattr(self, name))
        for name in (
            "update_count",
            "adam_step_count",
            "qualification_adam_step_count",
            "environment_steps",
            "mask_violation_count",
            "evaluation_leak_count",
            "raw_artifact_count",
        ):
            _non_negative_int(name, getattr(self, name))
        blockers: list[str] = []
        # This legacy receipt certifies only its original named M1 campaign;
        # generic feature plugins are qualified separately, never via this gate.
        from pyjevsim_bridge.rl.qualification_models.anti_torpedo_features import (
            ANTI_TORPEDO_FEATURE_CONTRACT_SHA256,
            LOADED_ANTI_TORPEDO_FEATURE_SOURCE_SHA256,
        )
        if self.source_sha256 != LOADED_REFERENCE_PPO_SOURCE_SHA256:
            blockers.append("reference-ppo-source-mismatch")
        if self.feature_source_sha256 != LOADED_ANTI_TORPEDO_FEATURE_SOURCE_SHA256:
            blockers.append("feature-source-mismatch")
        if self.config_sha256 != FROZEN_REFERENCE_PPO_CONFIG_SHA256:
            blockers.append("production-config-mismatch")
        if self.feature_contract_sha256 != ANTI_TORPEDO_FEATURE_CONTRACT_SHA256:
            blockers.append("feature-contract-mismatch")
        if self.numpy_runtime_sha256 != numpy_runtime_sha256():
            blockers.append("numpy-runtime-mismatch")
        for prefix in ("policy", "value", "optimizer", "checkpoint"):
            if getattr(self, f"initial_{prefix}_sha256") == getattr(
                self, f"final_{prefix}_sha256"
            ):
                blockers.append(f"{prefix}-digest-unchanged")
        if self.update_count != 2:
            blockers.append("requires-exactly-two-logical-updates")
        if self.adam_step_count != 160:
            blockers.append("uninterrupted-adam-step-count-differs")
        if self.qualification_adam_step_count != 320:
            blockers.append("qualification-adam-step-count-differs")
        if self.environment_steps != 4096:
            blockers.append("qualification-environment-step-count-differs")
        if not self.checkpoint_reloaded:
            blockers.append("checkpoint-not-reloaded")
        if not self.resume_bytes_matched:
            blockers.append("resume-bytes-differ")
        if self.mask_violation_count:
            blockers.append("action-mask-violations")
        if self.evaluation_leak_count:
            blockers.append("evaluation-leakage")
        if self.raw_artifact_count <= 0:
            blockers.append("raw-ledger-artifacts-missing")
        blockers.append("verified-qualification-evidence-required")
        frozen = tuple(blockers)
        object.__setattr__(self, "blockers", frozen)
        object.__setattr__(self, "admitted", not frozen)
        content = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in {"sha256"}
        }
        object.__setattr__(self, "sha256", _sha256(_canonical_json(content)))


def _runtime_platform_identity() -> dict[str, object]:
    """Log platform provenance without creating Windows shell helper processes.

    Windows platform_version is kernel32.dll-derived logging metadata, may
    differ from the actual OS version, and must not be used as a feature gate.
    A changed profile deliberately changes the runtime fingerprint; no legacy
    identity is migrated and acquisition failures have no shell fallback.
    """
    if sys.platform != "win32":
        return {"platform": platform.platform()}
    native = sys.getwindowsversion()
    version = getattr(native, "platform_version", None)
    if (not isinstance(version, tuple) or len(version) != 3
        or any(type(value) is not int or value < 0 for value in version)):
        raise PolicyIntegrityError(
            "native Windows platform_version must contain three nonnegative integers"
        )
    product_type = getattr(native, "product_type", None)
    if type(product_type) is not int or product_type not in (1, 2, 3):
        raise PolicyIntegrityError("native Windows product_type must be integer 1, 2, or 3")
    service_pack = (
        getattr(native, "service_pack_major", None),
        getattr(native, "service_pack_minor", None),
    )
    if any(type(value) is not int or value < 0 for value in service_pack):
        raise PolicyIntegrityError(
            "native Windows service pack must contain two nonnegative integers"
        )
    pointer_bits = struct.calcsize("P") * 8
    if type(pointer_bits) is not int or pointer_bits not in (32, 64):
        raise PolicyIntegrityError("native Windows pointer width must be 32 or 64 bits")
    # Saved runtime identities are decoded from JSON and compared directly.
    # Arrays must therefore retain list shape across publication/reload.
    return {
        "platform": "Windows-native-" + ".".join(str(value) for value in version),
        "platform_query_profile": "windows-native-platform-v1",
        "windows_platform_version": list(version),
        "windows_product_type": product_type,
        "windows_service_pack": list(service_pack),
        "pointer_bits": pointer_bits,
    }


def numpy_runtime_identity() -> Mapping[str, object]:
    """Return qualification-lock material without claiming cross-host identity."""

    helper_sha256 = inference_metrics_source_sha256()
    if helper_sha256 != LOADED_INFERENCE_METRICS_SOURCE_SHA256:
        raise PolicyIntegrityError("inference metrics source changed after import")
    config = np.__config__
    dependencies = config.CONFIG.get("Build Dependencies", {})
    blas = dependencies.get("blas", {})
    lapack = dependencies.get("lapack", {})
    multiarray = np._core._multiarray_umath  # type: ignore[attr-defined]
    binary_path = Path(cast(str, multiarray.__file__))
    cpu_features = {
        str(key): bool(value)
        for key, value in sorted(
            cast(Mapping[str, object], multiarray.__cpu_features__).items()
        )
    }
    thread_limits = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return MappingProxyType(
        {
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            **_runtime_platform_identity(),
            "byteorder": sys.byteorder,
            "blas_name": str(blas.get("name", "unknown")),
            "blas_version": str(blas.get("version", "unknown")),
            "lapack_name": str(lapack.get("name", "unknown")),
            "lapack_version": str(lapack.get("version", "unknown")),
            "cpu_features": cpu_features,
            "numpy_binary_sha256": file_sha256(binary_path, role="numpy_binary"),
            "inference_metrics_source_sha256": helper_sha256,
            "thread_limits": thread_limits,
            "dtype": "float32",
            "device": "cpu",
        }
    )


def numpy_runtime_sha256() -> str:
    return _sha256(_canonical_json(dict(numpy_runtime_identity())))


def _hash_uniform(seed: int, name: str, index: int) -> float:
    digest = hashlib.sha256(
        b"\x00".join(
            (
                INITIALIZATION_DOMAIN.encode(),
                str(seed).encode(),
                name.encode(),
                str(index).encode(),
            )
        )
    ).digest()
    integer = int.from_bytes(digest[:8], "big") >> 11
    return integer / float(1 << 53)


def _initialize_tensor(seed: int, name: str, shape: tuple[int, ...]) -> FloatArray:
    if name.endswith((".b1", ".b2", ".b3")):
        return np.zeros(shape, dtype=np.float32)
    fan_out, fan_in = shape
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    values = np.fromiter(
        (
            (2.0 * _hash_uniform(seed, name, index) - 1.0) * bound
            for index in range(math.prod(shape))
        ),
        dtype=np.float32,
        count=math.prod(shape),
    )
    return values.reshape(shape)


def _new_parameters(config: ReferencePPOConfigV1) -> dict[str, FloatArray]:
    result: dict[str, FloatArray] = {}
    for name, shape_fn in (*_POLICY_SHAPES, *_VALUE_SHAPES):
        result[name] = _initialize_tensor(
            config.initialization_seed, name, shape_fn(config)
        )
    return result


def _parameter_blob(parameters: Mapping[str, FloatArray], names: Sequence[str]) -> bytes:
    return b"".join(
        np.asarray(parameters[name], dtype="<f4", order="C").tobytes(order="C")
        for name in names
    )


def _parameter_digest(parameters: Mapping[str, FloatArray], names: Sequence[str]) -> str:
    return _sha256(_parameter_blob(parameters, names))


def _encode_tensors(parameters: Mapping[str, FloatArray]) -> dict[str, object]:
    return {
        name: {
            "data": base64.b64encode(
                np.asarray(value, dtype="<f4", order="C").tobytes(order="C")
            ).decode("ascii"),
            "shape": list(value.shape),
        }
        for name, value in sorted(parameters.items())
    }


def _decode_tensors(
    value: object,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, FloatArray]:
    if not isinstance(value, dict) or set(value) != set(expected_shapes):
        raise PolicyIntegrityError("checkpoint tensor names differ")
    result: dict[str, FloatArray] = {}
    for name in sorted(expected_shapes):
        row = value[name]
        if not isinstance(row, dict) or set(row) != {"data", "shape"}:
            raise PolicyIntegrityError(f"checkpoint tensor {name} fields differ")
        if row["shape"] != list(expected_shapes[name]) or not isinstance(row["data"], str):
            raise PolicyIntegrityError(f"checkpoint tensor {name} shape differs")
        try:
            raw = base64.b64decode(row["data"], validate=True)
        except ValueError as exc:
            raise PolicyIntegrityError(f"checkpoint tensor {name} base64 is invalid") from exc
        expected_bytes = math.prod(expected_shapes[name]) * 4
        if len(raw) != expected_bytes:
            raise PolicyIntegrityError(f"checkpoint tensor {name} size differs")
        array = np.frombuffer(raw, dtype="<f4").copy().reshape(expected_shapes[name])
        if not bool(np.all(np.isfinite(array))):
            raise PolicyIntegrityError(f"checkpoint tensor {name} is non-finite")
        result[name] = array
    return result


def _decode_payload(payload: bytes) -> dict[str, object]:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PolicyIntegrityError(f"reference PPO checkpoint JSON is invalid: {exc}") from exc
    if not isinstance(decoded, dict):
        raise PolicyIntegrityError("reference PPO checkpoint root must be an object")
    if _canonical_json(decoded) != payload:
        raise PolicyIntegrityError("reference PPO checkpoint is not canonical JSON")
    expected = {
        "schema_version",
        "algorithm_id",
        "algorithm_version",
        "config",
        "config_sha256",
        "feature_contract_sha256",
        "feature_source_sha256",
        "feature_profile",
        "ppo_features_source_sha256",
        "action_schema_sha256",
        "reference_ppo_source_sha256",
        "numpy_runtime_sha256",
        "run_id",
        "generation",
        "policy_version",
        "update_count",
        "environment_steps",
        "adam_step",
        "bound_behavior_artifact_sha256",
        "transition_chain_sha256",
        "transition_digests",
        "applied_batches",
        "applied_records",
        "stream_cursors",
        "closed_streams",
        "objective_id",
        "parameters",
        "adam_first",
        "adam_second",
        "policy_parameter_sha256",
        "value_parameter_sha256",
        "optimizer_sha256",
        "seed_domains",
        "runtime",
        "last_metrics",
    }
    if set(decoded) != expected:
        raise PolicyIntegrityError("reference PPO checkpoint fields differ")
    if (
        decoded["schema_version"] != REFERENCE_PPO_CHECKPOINT_SCHEMA
        or decoded["algorithm_id"] != REFERENCE_PPO_ALGORITHM_ID
        or decoded["algorithm_version"] != REFERENCE_PPO_ALGORITHM_VERSION
        or decoded["objective_id"] != REFERENCE_PPO_OBJECTIVE_ID
    ):
        raise PolicyIntegrityError("reference PPO checkpoint identity differs")
    return cast(dict[str, object], decoded)


def _validate_source_runtime(decoded: Mapping[str, object]) -> None:
    if reference_ppo_source_sha256() != LOADED_REFERENCE_PPO_SOURCE_SHA256:
        raise PolicyIntegrityError("reference PPO source changed after import")
    if decoded["reference_ppo_source_sha256"] != LOADED_REFERENCE_PPO_SOURCE_SHA256:
        raise PolicyCompatibilityError("reference PPO source digest differs")
    if (
        ppo_features_source_sha256() != LOADED_PPO_FEATURES_SOURCE_SHA256
        or decoded["ppo_features_source_sha256"] != LOADED_PPO_FEATURES_SOURCE_SHA256
    ):
        raise PolicyCompatibilityError("PPO feature helper source digest differs")
    current_runtime = dict(numpy_runtime_identity())
    if decoded["runtime"] != current_runtime:
        raise PolicyCompatibilityError("NumPy runtime identity differs")
    if decoded["numpy_runtime_sha256"] != numpy_runtime_sha256():
        raise PolicyCompatibilityError("NumPy runtime digest differs")


@metric_function("integrity")
def _assert_current_implementation(runtime_sha256: str) -> None:
    """Check drift before inference or optimizer mutation, not during publication."""
    if reference_ppo_source_sha256() != LOADED_REFERENCE_PPO_SOURCE_SHA256:
        raise PolicyIntegrityError("reference PPO source changed after import")
    if ppo_features_source_sha256() != LOADED_PPO_FEATURES_SOURCE_SHA256:
        raise PolicyCompatibilityError("PPO feature helper source changed after import")
    if numpy_runtime_sha256() != runtime_sha256:
        raise PolicyCompatibilityError("NumPy runtime changed after binding")


def _checkpoint_batches(value: object) -> list[tuple[str, str, int, int]]:
    if not isinstance(value, list):
        raise PolicyIntegrityError("checkpoint applied_batches must be an array")
    expected = {"batch_sha256", "chain_sha256", "record_count", "update_index"}
    result: list[tuple[str, str, int, int]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != expected:
            raise PolicyIntegrityError("checkpoint applied batch fields differ")
        result.append(
            (
                _require_sha256("batch_sha256", row["batch_sha256"]),
                _require_sha256("chain_sha256", row["chain_sha256"]),
                _positive_int("batch record_count", row["record_count"]),
                _non_negative_int("batch update_index", row["update_index"]),
            )
        )
    if [item[3] for item in result] != list(range(len(result))):
        raise PolicyIntegrityError("checkpoint update indices are not contiguous")
    if len({item[0] for item in result}) != len(result):
        raise PolicyIntegrityError("checkpoint repeats an applied batch")
    return result


def _validate_seed_domains(value: object, config: ReferencePPOConfigV1) -> None:
    expected = {
        "initialization": INITIALIZATION_DOMAIN,
        "initialization_seed": config.initialization_seed,
        "sampling": SAMPLING_DOMAIN,
        "shuffle": SHUFFLE_DOMAIN,
        "shuffle_seed": config.shuffle_seed,
    }
    if value != expected:
        raise PolicyIntegrityError("checkpoint seed domains or counters differ")


def _validate_checkpoint_contents(
    decoded: Mapping[str, object],
    *,
    source_batch_sha256: str | None = None,
) -> tuple[
    ReferencePPOConfigV1,
    dict[str, FloatArray],
    dict[str, FloatArray],
    dict[str, FloatArray],
]:
    config_raw = decoded["config"]
    if not isinstance(config_raw, dict):
        raise PolicyIntegrityError("checkpoint config must be an object")
    config = ReferencePPOConfigV1.from_dict(cast(dict[str, object], config_raw))
    try:
        profile = validate_feature_profile(decoded["feature_profile"])
    except FeatureContractError as exc:
        raise PolicyIntegrityError(f"checkpoint feature profile is invalid: {exc}") from exc
    if (
        profile["feature_size"] != config.observation_size
        or profile["action_count"] != config.action_count
        or profile["dtype"] != config.dtype
        or profile["contract_sha256"] != decoded["feature_contract_sha256"]
        or profile["source_sha256"] != decoded["feature_source_sha256"]
        or profile["action_schema_sha256"] != decoded["action_schema_sha256"]
    ):
        raise PolicyIntegrityError("checkpoint feature profile differs from config or identities")
    if decoded["config_sha256"] != config.sha256:
        raise PolicyIntegrityError("checkpoint config digest differs")
    _validate_seed_domains(decoded["seed_domains"], config)
    shapes = _expected_shapes(config)
    parameters = _decode_tensors(decoded["parameters"], shapes)
    first = _decode_tensors(decoded["adam_first"], shapes)
    second = _decode_tensors(decoded["adam_second"], shapes)
    _validate_parameter_digests(decoded, parameters)
    adam_step = _non_negative_int("adam_step", decoded["adam_step"])
    if decoded["optimizer_sha256"] != _optimizer_digest(first, second, adam_step):
        raise PolicyIntegrityError("checkpoint optimizer digest differs")

    update_count = _non_negative_int("update_count", decoded["update_count"])
    policy_version = _non_negative_int("policy_version", decoded["policy_version"])
    environment_steps = _non_negative_int(
        "environment_steps", decoded["environment_steps"]
    )
    stream_cursors = _decode_cursors(decoded["stream_cursors"])
    _decode_closed_streams(decoded["closed_streams"], stream_cursors)
    if len(stream_cursors) > environment_steps:
        raise PolicyIntegrityError("checkpoint stream count exceeds transition count")
    batches = _checkpoint_batches(decoded["applied_batches"])
    records = _decode_pairs(
        decoded["applied_records"],
        "idempotency_key",
        "record_sha256",
        key_digest=False,
    )
    raw_digests = decoded["transition_digests"]
    if not isinstance(raw_digests, list):
        raise PolicyIntegrityError("checkpoint transition_digests must be an array")
    transition_digests = [
        _require_sha256(f"transition_digests[{index}]", digest)
        for index, digest in enumerate(raw_digests)
    ]
    if policy_version != update_count or len(batches) != update_count:
        raise PolicyIntegrityError(
            "checkpoint policy version, update count, and batch count differ"
        )
    if (
        sum(item[2] for item in batches) != environment_steps
        or len(records) != environment_steps
        or len(transition_digests) != environment_steps
        or sorted(records.values()) != sorted(transition_digests)
    ):
        raise PolicyIntegrityError("checkpoint transition counts do not reconcile")
    chain = ZERO_SHA256
    cursor = 0
    expected_adam = 0
    for _batch_digest, batch_chain, record_count, _update_index in batches:
        if record_count > config.batch_size:
            raise PolicyIntegrityError("checkpoint batch exceeds configured size")
        for digest in transition_digests[cursor : cursor + record_count]:
            chain = _sha256(
                TRANSITION_CHAIN_DOMAIN.encode()
                + bytes.fromhex(chain)
                + bytes.fromhex(digest)
            )
        cursor += record_count
        if batch_chain != chain:
            raise PolicyIntegrityError("checkpoint per-batch transition chain differs")
        expected_adam += config.epochs * math.ceil(record_count / config.minibatch_size)
    if decoded["transition_chain_sha256"] != chain:
        raise PolicyIntegrityError("checkpoint terminal transition chain differs")
    if adam_step != expected_adam:
        raise PolicyIntegrityError("checkpoint Adam step count does not reconcile")
    behavior = decoded["bound_behavior_artifact_sha256"]
    if update_count == 0:
        if behavior is not None:
            raise PolicyIntegrityError("initial checkpoint cannot bind behavior artifact")
    else:
        _require_sha256("bound_behavior_artifact_sha256", behavior)
    if source_batch_sha256 is not None:
        expected_source = EMPTY_TRANSITION_BATCH_SHA256 if not batches else batches[-1][0]
        if source_batch_sha256 != expected_source:
            raise PolicyIntegrityError("artifact source batch differs from checkpoint")
    return config, parameters, first, second


def _validate_published_pair(
    reference: PolicyArtifactRef, artifact: PolicyArtifact
) -> None:
    if not isinstance(reference, PolicyArtifactRef) or not isinstance(
        artifact, PolicyArtifact
    ):
        raise TypeError("published binding requires a reference and artifact")
    if (
        reference.run_id,
        reference.generation,
        reference.policy_version,
        reference.sha256,
        reference.size_bytes,
        reference.media_type,
        reference.compatibility,
        reference.source_batch_sha256,
    ) != (
        artifact.run_id,
        artifact.generation,
        artifact.policy_version,
        artifact.sha256,
        artifact.size_bytes,
        artifact.media_type,
        artifact.compatibility,
        artifact.source_batch_sha256,
    ):
        raise PolicyIntegrityError("published artifact differs from its reference")
    decoded = _decode_payload(artifact.payload)
    _validate_source_runtime(decoded)
    if (decoded["run_id"], decoded["generation"], decoded["policy_version"]) != (
        artifact.run_id,
        artifact.generation,
        artifact.policy_version,
    ):
        raise PolicyIntegrityError("published artifact embedded identity differs")
    _validate_checkpoint_contents(
        decoded, source_batch_sha256=artifact.source_batch_sha256
    )


@metric_function("forward")
def _forward_tower(
    x: FloatArray,
    parameters: Mapping[str, FloatArray],
    prefix: str,
) -> tuple[FloatArray, tuple[FloatArray, FloatArray]]:
    h1 = np.tanh(x @ parameters[f"{prefix}.w1"].T + parameters[f"{prefix}.b1"])
    h1 = np.asarray(h1, dtype=np.float32)
    h2 = np.tanh(h1 @ parameters[f"{prefix}.w2"].T + parameters[f"{prefix}.b2"])
    h2 = np.asarray(h2, dtype=np.float32)
    output = h2 @ parameters[f"{prefix}.w3"].T + parameters[f"{prefix}.b3"]
    return np.asarray(output, dtype=np.float32), (h1, h2)


def _masked_distribution(
    logits: FloatArray, masks: BoolArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if logits.shape != masks.shape or bool(np.any(np.sum(masks, axis=1) == 0)):
        raise LearningContractError("each policy row requires at least one valid action")
    if not bool(np.all(np.isfinite(logits))):
        raise LearningContractError("policy logits must be finite")
    masked = np.where(masks, logits, np.float32(-np.inf))
    maximum = np.max(masked, axis=1, keepdims=True)
    shifted = np.where(masks, masked - maximum, np.float32(0.0))
    exponentials = np.where(masks, np.exp(shifted), np.float32(0.0))
    normalizer = np.sum(exponentials, axis=1, keepdims=True)
    probabilities = np.asarray(
        exponentials / normalizer,
        dtype=np.float32,
    )
    log_probabilities = np.where(
        masks,
        shifted - np.log(normalizer),
        np.float32(0.0),
    )
    entropy = -np.sum(probabilities * log_probabilities, axis=1, dtype=np.float32)
    if not (
        bool(np.all(np.isfinite(probabilities)))
        and bool(np.all(np.isfinite(log_probabilities)))
        and bool(np.all(np.isfinite(entropy)))
    ):
        raise LearningContractError("policy distribution must be finite")
    return (
        probabilities,
        np.asarray(log_probabilities, dtype=np.float32),
        np.asarray(entropy, dtype=np.float32),
    )


def compute_gae(
    rewards: FloatArray,
    values: FloatArray,
    next_values: FloatArray,
    terminated: BoolArray,
    truncated: BoolArray,
    stream_keys: Sequence[tuple[str, str]],
    step_ids: Sequence[int],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    """Decision-index GAE: cutoffs bootstrap but never continue an episode trace.

    ``next_values`` belongs to each transition's actual final next observation,
    not to an auto-reset observation or the next array row. Both flags true
    gives termination precedence. Invalid numeric data is rejected even when
    termination would otherwise mask its contribution.
    """

    size = len(rewards)
    if not (
        values.shape
        == next_values.shape
        == rewards.shape
        == terminated.shape
        == truncated.shape
        == (size,)
        and len(stream_keys) == len(step_ids) == size
    ):
        raise ValueError("GAE arrays and identities must have identical one-dimensional size")
    if size == 0:
        raise ValueError("GAE arrays must not be empty")
    for name, array in (("rewards", rewards), ("values", values), ("next_values", next_values)):
        if array.dtype.kind not in "fiu" or not bool(np.all(np.isfinite(array))):
            raise ValueError(f"GAE {name} must contain finite numbers")
        with np.errstate(over="ignore", invalid="ignore"):
            representable = np.asarray(array, dtype=np.float32)
        if not bool(np.all(np.isfinite(representable))):
            raise ValueError(f"GAE {name} must be representable as finite float32")
    if terminated.dtype.kind != "b" or truncated.dtype.kind != "b":
        raise ValueError("GAE terminal flags must be boolean arrays")
    gamma_value = _finite("gamma", gamma)
    lambda_value = _finite("gae_lambda", gae_lambda)
    if not 0.0 <= gamma_value <= 1.0 or not 0.0 <= lambda_value <= 1.0:
        raise ValueError("GAE gamma and lambda must be in [0, 1]")
    advantages = np.zeros(size, dtype=np.float32)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(stream_keys):
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(item, str) or not item for item in key)
        ):
            raise ValueError("GAE stream identity must contain worker and episode strings")
        _non_negative_int("GAE step_id", step_ids[index])
        groups.setdefault(key, []).append(index)
    gamma32 = np.float32(gamma_value)
    lambda32 = np.float32(lambda_value)
    for indices in groups.values():
        for left, right in zip(indices, indices[1:], strict=False):
            if step_ids[right] != step_ids[left] + 1:
                raise ValueError("GAE stream fragments must be step-contiguous")
            if terminated[left] or truncated[left]:
                raise ValueError("GAE stream cannot continue after a terminal record")
        following_advantage = np.float32(0.0)
        following_step: int | None = None
        for index in reversed(indices):
            done = bool(terminated[index] or truncated[index])
            bootstrap = np.float32(0.0 if terminated[index] else 1.0)
            consecutive = following_step is not None and following_step == step_ids[index] + 1
            recursive = np.float32(1.0 if consecutive and not done else 0.0)
            with np.errstate(over="ignore", invalid="ignore"):
                delta = np.float32(
                    rewards[index] + gamma32 * bootstrap * next_values[index] - values[index]
                )
                following_advantage = np.float32(
                    delta + gamma32 * lambda32 * recursive * following_advantage
                )
            advantages[index] = following_advantage
            following_step = step_ids[index]
    with np.errstate(over="ignore", invalid="ignore"):
        returns = np.asarray(advantages + values, dtype=np.float32)
    if not bool(np.all(np.isfinite(advantages))) or not bool(np.all(np.isfinite(returns))):
        raise ValueError("GAE produced non-finite advantages or returns")
    return advantages, returns


@dataclass(frozen=True, slots=True)
class _PPOLossOutputGradients:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    clipped_fraction: float
    policy: FloatArray
    value: FloatArray


def _loss_and_output_gradients(
    logits: FloatArray,
    masks: BoolArray,
    actions: IntArray,
    old_log_probabilities: FloatArray,
    advantages: FloatArray,
    values: FloatArray,
    returns: FloatArray,
    *,
    policy_clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> _PPOLossOutputGradients:
    """Evaluate the frozen mean loss and its policy/value output gradients."""

    size = len(logits)
    if size <= 0 or logits.ndim != 2:
        raise LearningContractError("PPO loss requires a non-empty logits matrix")
    if (
        masks.shape != logits.shape
        or actions.shape != (size,)
        or old_log_probabilities.shape != (size,)
        or advantages.shape != (size,)
        or values.shape != (size,)
        or returns.shape != (size,)
    ):
        raise LearningContractError("PPO loss input shapes differ")
    if not (
        bool(np.all(np.isfinite(old_log_probabilities)))
        and bool(np.all(np.isfinite(advantages)))
        and bool(np.all(np.isfinite(values)))
        and bool(np.all(np.isfinite(returns)))
    ):
        raise LearningContractError("PPO loss inputs must be finite")
    if bool(np.any(actions < 0)) or bool(np.any(actions >= logits.shape[1])):
        raise LearningContractError("PPO loss action lies outside the action space")
    rows = np.arange(size, dtype=np.int64)
    if not bool(np.all(masks[rows, actions])):
        raise LearningContractError("PPO loss action is masked")

    probabilities, log_probabilities, entropy = _masked_distribution(logits, masks)
    selected_log = log_probabilities[rows, actions]
    ratios = np.asarray(np.exp(selected_log - old_log_probabilities), dtype=np.float32)
    if not bool(np.all(np.isfinite(ratios))):
        raise LearningContractError("PPO probability ratio must be finite")
    clipped = np.clip(
        ratios,
        np.float32(1.0 - policy_clip),
        np.float32(1.0 + policy_clip),
    )
    surrogate = np.minimum(ratios * advantages, clipped * advantages)
    policy_loss = -float(np.mean(surrogate, dtype=np.float32))
    active = np.logical_not(
        np.logical_or(
            np.logical_and(advantages >= 0, ratios > 1.0 + policy_clip),
            np.logical_and(advantages < 0, ratios < 1.0 - policy_clip),
        )
    )
    selected_gradient = np.asarray(
        -advantages * ratios * active.astype(np.float32) / np.float32(size),
        dtype=np.float32,
    )
    policy_gradient = np.asarray(
        -probabilities * selected_gradient[:, None], dtype=np.float32
    )
    policy_gradient[rows, actions] += selected_gradient
    entropy_gradient = np.asarray(
        np.float32(entropy_coefficient)
        * probabilities
        * (log_probabilities + entropy[:, None])
        / np.float32(size),
        dtype=np.float32,
    )
    entropy_gradient[~masks] = np.float32(0.0)
    policy_gradient += entropy_gradient

    errors = np.asarray(values - returns, dtype=np.float32)
    value_loss = 0.5 * float(np.mean(errors * errors, dtype=np.float32))
    value_gradient = np.asarray(
        np.float32(value_coefficient) * errors[:, None] / np.float32(size),
        dtype=np.float32,
    )
    entropy_mean = float(np.mean(entropy, dtype=np.float32))
    total_loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy_mean
    )
    return _PPOLossOutputGradients(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
        total_loss=total_loss,
        clipped_fraction=float(np.mean(np.logical_not(active), dtype=np.float32)),
        policy=policy_gradient,
        value=value_gradient,
    )


def _clip_gradients_by_global_norm(
    gradients: Mapping[str, FloatArray], max_gradient_norm: float
) -> tuple[dict[str, FloatArray], float]:
    """Return float32 gradient copies clipped by one deterministic global norm."""

    if not gradients:
        raise LearningContractError("PPO optimizer requires gradients")
    limit = _finite("max_gradient_norm", max_gradient_norm)
    if limit <= 0.0:
        raise LearningContractError("max_gradient_norm must be positive")
    squared = np.float32(0.0)
    for name in sorted(gradients):
        gradient = gradients[name]
        if not bool(np.all(np.isfinite(gradient))):
            raise LearningContractError("PPO gradient must be finite")
        squared = np.float32(
            squared + np.sum(gradient * gradient, dtype=np.float32)
        )
    norm = np.float32(np.sqrt(squared))
    if not bool(np.isfinite(norm)):
        raise LearningContractError("PPO gradient norm is non-finite")
    scale = np.float32(1.0)
    if norm > limit:
        scale = np.float32(limit) / norm
    return (
        {
            name: np.asarray(gradients[name] * scale, dtype=np.float32)
            for name in gradients
        },
        float(norm),
    )


def _backward_tower(
    x: FloatArray,
    cache: tuple[FloatArray, FloatArray],
    output_gradient: FloatArray,
    parameters: Mapping[str, FloatArray],
    prefix: str,
) -> dict[str, FloatArray]:
    h1, h2 = cache
    result: dict[str, FloatArray] = {}
    result[f"{prefix}.w3"] = np.asarray(output_gradient.T @ h2, dtype=np.float32)
    result[f"{prefix}.b3"] = np.asarray(np.sum(output_gradient, axis=0), dtype=np.float32)
    dh2 = np.asarray(output_gradient @ parameters[f"{prefix}.w3"], dtype=np.float32)
    dz2 = np.asarray(dh2 * (np.float32(1.0) - h2 * h2), dtype=np.float32)
    result[f"{prefix}.w2"] = np.asarray(dz2.T @ h1, dtype=np.float32)
    result[f"{prefix}.b2"] = np.asarray(np.sum(dz2, axis=0), dtype=np.float32)
    dh1 = np.asarray(dz2 @ parameters[f"{prefix}.w2"], dtype=np.float32)
    dz1 = np.asarray(dh1 * (np.float32(1.0) - h1 * h1), dtype=np.float32)
    result[f"{prefix}.w1"] = np.asarray(dz1.T @ x, dtype=np.float32)
    result[f"{prefix}.b1"] = np.asarray(np.sum(dz1, axis=0), dtype=np.float32)
    return result


def _optimizer_digest(
    first: Mapping[str, FloatArray],
    second: Mapping[str, FloatArray],
    step: int,
) -> str:
    names = tuple(sorted(first))
    return _sha256(
        _canonical_json({"step": step, "names": list(names)})
        + _parameter_blob(first, names)
        + _parameter_blob(second, names)
    )


def _sampling_uniform(value: PPOInferenceInputV1, artifact_sha256: str) -> float:
    digest = hashlib.sha256(
        b"\x00".join(
            (
                SAMPLING_DOMAIN.encode(),
                artifact_sha256.encode(),
                str(value.sampling_seed).encode(),
                value.run_id.encode(),
                str(value.generation).encode(),
                value.worker_id.encode(),
                value.episode_id.encode(),
                str(value.step_id).encode(),
            )
        )
    ).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / float(1 << 53)


@dataclass(slots=True)
class _LoadedReferencePPOPolicy:
    parameters: Mapping[str, FloatArray]
    feature_contract: PPOFeatureContract
    feature_binding: FeatureContractBinding
    artifact_sha256: str
    run_id: str
    generation: int
    action_count: int
    runtime_sha256: str
    artifact: PolicyArtifact
    _active_rollout_scope: object | None = field(default=None, init=False, repr=False)

    async def actions(self, observations: Mapping[str, object]) -> Mapping[str, object]:
        with metric_span("actions_total"):
            return self._actions(observations)

    def _actions(self, observations: Mapping[str, object]) -> Mapping[str, object]:
        _assert_current_implementation(self.runtime_sha256)
        self.feature_binding.assert_current()
        return self._action_values(observations, self.feature_binding)

    def _action_values(
        self, observations: Mapping[str, object], binding: FeatureContractBinding,
    ) -> Mapping[str, object]:
        selected: dict[str, object] = {}
        for worker_id, raw in observations.items():
            with metric_span("input_validation"):
                if not isinstance(raw, PPOInferenceInputV1):
                    raise TypeError("reference PPO requires PPOInferenceInputV1 values")
                if worker_id != raw.worker_id:
                    raise ValueError("inference input worker identity differs")
                if (raw.run_id, raw.generation) != (self.run_id, self.generation):
                    raise PolicyCompatibilityError("inference run/generation differs from artifact")
                encoded_input = raw.to_dict()
            derived_mask = binding.action_mask(encoded_input)
            if derived_mask != raw.action_mask:
                raise LearningContractError("contemporaneous action mask differs from state")
            encoded = binding.encode(encoded_input)
            if len(encoded) != self.parameters["policy.w1"].shape[1]:
                raise LearningContractError("feature vector width differs from policy")
            x = np.asarray([encoded], dtype=np.float32)
            logits, _cache = _forward_tower(x, self.parameters, "policy")
            with metric_span("distribution_sampling"):
                mask = np.asarray([raw.action_mask], dtype=np.bool_)
                probabilities, _logs, _entropy = _masked_distribution(logits, mask)
                valid = [index for index, allowed in enumerate(raw.action_mask) if allowed]
                if not raw.explore:
                    choice = max(valid, key=lambda index: (float(logits[0, index]), -index))
                else:
                    uniform = _sampling_uniform(raw, self.artifact_sha256)
                    cumulative = 0.0
                    choice = valid[-1]
                    for index in valid:
                        cumulative += float(probabilities[0, index])
                        if uniform < cumulative:
                            choice = index
                            break
            selected[worker_id] = binding.decode_action(choice)
        return selected


class ReferencePPORolloutScope:
    """One opt-in, queue-only frozen-policy segment with boundary verification.

    This is not strict immediate source-change detection, a file lock, or a
    security sandbox. A file changed and restored between boundaries may escape
    detection. Successful outputs may be admitted only after this context exits.
    No binding or validation profile is inserted into the sampling identity.
    """

    def __init__(self, policy: LoadedPolicy) -> None:
        if type(policy) is not _LoadedReferencePPOPolicy:
            raise PolicyCompatibilityError("rollout scope requires a loaded reference PPO policy")
        self._policy = policy
        self._bound_policy = self._policy
        self._owner = (os.getpid(), threading.get_ident())
        self._entered = self._active = self._closed = False

    def _assert_owner(self) -> None:
        if (os.getpid(), threading.get_ident()) != self._owner:
            raise PolicyCompatibilityError("rollout scope belongs to one process/thread")

    def __enter__(self) -> Self:
        self._assert_owner()
        if self._entered or self._closed:
            raise PolicyCompatibilityError("rollout scope cannot be reused")
        self._entered = True
        try:
            with metric_span("scope_bind"):
                # Lazy opt-in only: importing the generic learner still imports
                # no model. Mutable/subclassed generic plugins are not supported.
                from pyjevsim_bridge.rl.models.queue_features import QueueFeatures

                policy = self._policy
                if type(policy.feature_contract) is not QueueFeatures:
                    raise PolicyCompatibilityError(
                        "rollout scope supports only built-in QueueFeatures"
                    )
                if policy._active_rollout_scope is not None:
                    raise PolicyCompatibilityError("policy already has an active rollout scope")
                self._artifact = policy.artifact
                self._identity = self._policy_identity()
                self._binding = policy.feature_binding
                self._contract = policy.feature_contract
                self._parameters = policy.parameters
                self._arrays = tuple(
                    (name, array, array.shape, array.dtype, array.strides)
                    for name, array in self._parameters.items()
                )
                self._methods = tuple(
                    (name, getattr(self._contract, name).__func__)
                    for name in (
                        "observation_action_mask", "action_mask", "encode",
                        "encode_action", "decode_action",
                    )
                )
                self._active = True
                policy._active_rollout_scope = self
                self._features = _scoped_feature_binding(self._binding, self._assert_identity)
                self._verify_boundary()
        except BaseException:
            self._close()
            raise
        return self

    def _policy_identity(self) -> tuple[object, ...]:
        policy = self._policy
        return (
            policy.artifact_sha256, policy.run_id, policy.generation,
            policy.action_count, policy.runtime_sha256,
        )

    @metric_function("integrity")
    def _assert_identity(self) -> None:
        self._assert_owner()
        if not self._active or self._closed:
            raise PolicyCompatibilityError("rollout scope is not active")
        policy = self._policy
        if (
            policy is not self._bound_policy
            or policy._active_rollout_scope is not self
            or policy.artifact is not self._artifact
            or self._policy_identity() != self._identity
            or policy.feature_binding is not self._binding
            or policy.feature_contract is not self._contract
            or self._binding.contract is not self._contract
            or policy.parameters is not self._parameters
        ):
            raise PolicyCompatibilityError("rollout scope policy or binding identity changed")
        if set(self._parameters) != {row[0] for row in self._arrays}:
            raise PolicyIntegrityError("rollout scope parameter names changed")
        for name, array, shape, dtype, strides in self._arrays:
            if (
                self._parameters[name] is not array or array.shape != shape
                or array.dtype != dtype or array.strides != strides or array.flags.writeable
            ):
                raise PolicyIntegrityError("rollout scope frozen parameter state changed")
        for name, function in self._methods:
            if getattr(getattr(self._contract, name, None), "__func__", None) is not function:
                raise FeatureContractError("rollout scope feature method changed")

    @metric_function("integrity")
    def _verify_boundary(self) -> None:
        self._assert_identity()
        policy, artifact = self._policy, self._artifact
        _assert_current_implementation(policy.runtime_sha256)
        self._binding.assert_current()
        decoded = _decode_payload(artifact.payload)
        if (
            (policy.artifact_sha256, policy.run_id, policy.generation)
            != (artifact.sha256, artifact.run_id, artifact.generation)
            or (decoded["run_id"], decoded["generation"], decoded["policy_version"])
            != (artifact.run_id, artifact.generation, artifact.policy_version)
            or decoded["feature_profile"] != self._binding.profile
            or decoded["numpy_runtime_sha256"] != policy.runtime_sha256
        ):
            raise PolicyIntegrityError("rollout scope artifact identity differs")
        _validate_parameter_digests(decoded, policy.parameters)

    @property
    def features(self) -> FeatureContractBinding:
        """The same per-input binding methods, usable only within this scope."""
        self._assert_identity()
        return self._features

    async def actions(self, observations: Mapping[str, object]) -> Mapping[str, object]:
        with metric_span("actions_total"):
            self._features_if_active().assert_current()
            return self._policy._action_values(observations, self._features)

    def _features_if_active(self) -> FeatureContractBinding:
        self._assert_identity()
        return self._features

    def _close(self) -> None:
        self._active, self._closed = False, True
        if self._bound_policy._active_rollout_scope is self:
            self._bound_policy._active_rollout_scope = None

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            with metric_span("scope_final_verify"):
                self._verify_boundary()
        except BaseException as final_error:
            if exc is not None:
                raise BaseExceptionGroup(
                    "rollout body and final verification both failed", [exc, final_error],
                ) from None
            raise
        finally:
            self._close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("rollout scopes are process-local and cannot be serialized")


def open_rollout_scope(policy: LoadedPolicy) -> ReferencePPORolloutScope:
    """Create a single-use explicit scope; ordinary policy.actions stays strict."""
    return ReferencePPORolloutScope(policy)


class ReferencePPOPolicyLoader:
    def __init__(self, feature_contract: PPOFeatureContract) -> None:
        self.feature_contract = feature_contract
        self.feature_binding = bind_feature_contract(feature_contract)
        self.feature_profile = self.feature_binding.profile
        self.feature_contract_sha256 = str(self.feature_profile["contract_sha256"])
        self.feature_source_sha256 = str(self.feature_profile["source_sha256"])
        self.source_sha256 = LOADED_REFERENCE_PPO_SOURCE_SHA256
        self.runtime_sha256 = numpy_runtime_sha256()

    async def load(self, artifact: PolicyArtifact) -> LoadedPolicy:
        _assert_current_implementation(self.runtime_sha256)
        self.feature_binding.assert_current()
        if artifact.media_type != REFERENCE_PPO_MEDIA_TYPE:
            raise PolicyCompatibilityError("reference PPO media type differs")
        if (
            artifact.compatibility.algorithm_id != REFERENCE_PPO_ALGORITHM_ID
            or artifact.compatibility.algorithm_version != REFERENCE_PPO_ALGORITHM_VERSION
        ):
            raise PolicyCompatibilityError("reference PPO algorithm identity differs")
        decoded = _decode_payload(artifact.payload)
        _validate_source_runtime(decoded)
        _validate_checkpoint_contents(decoded, source_batch_sha256=artifact.source_batch_sha256)
        if (decoded["run_id"], decoded["generation"], decoded["policy_version"]) != (
            artifact.run_id,
            artifact.generation,
            artifact.policy_version,
        ):
            raise PolicyIntegrityError(
                "embedded policy run/generation/version differs from artifact"
            )
        if decoded["feature_contract_sha256"] != self.feature_contract_sha256:
            raise PolicyCompatibilityError("reference PPO feature contract differs")
        if decoded["feature_source_sha256"] != self.feature_source_sha256:
            raise PolicyCompatibilityError("reference PPO feature source differs")
        if artifact.compatibility.observation_schema_sha256 != self.feature_contract_sha256:
            raise PolicyCompatibilityError("artifact observation compatibility differs")
        if (
            artifact.compatibility.action_schema_sha256
            != self.feature_profile["action_schema_sha256"]
        ):
            raise PolicyCompatibilityError("artifact action compatibility differs")
        if decoded["feature_profile"] != self.feature_profile:
            raise PolicyCompatibilityError("artifact feature profile differs")
        config_value = decoded["config"]
        if not isinstance(config_value, dict):
            raise PolicyIntegrityError("reference PPO config must be an object")
        config = ReferencePPOConfigV1.from_dict(cast(dict[str, object], config_value))
        if decoded["config_sha256"] != config.sha256:
            raise PolicyIntegrityError("reference PPO config digest differs")
        shapes = _expected_shapes(config)
        parameters = _decode_tensors(decoded["parameters"], shapes)
        _validate_parameter_digests(decoded, parameters)
        for value in parameters.values():
            value.setflags(write=False)
        return _LoadedReferencePPOPolicy(
            parameters=MappingProxyType(parameters),
            feature_contract=self.feature_contract,
            feature_binding=self.feature_binding,
            artifact_sha256=artifact.sha256,
            run_id=artifact.run_id,
            generation=artifact.generation,
            action_count=config.action_count,
            runtime_sha256=self.runtime_sha256,
            artifact=artifact,
        )


def _expected_shapes(config: ReferencePPOConfigV1) -> dict[str, tuple[int, ...]]:
    return {
        name: shape_fn(config) for name, shape_fn in (*_POLICY_SHAPES, *_VALUE_SHAPES)
    }


def _validate_parameter_digests(
    decoded: Mapping[str, object], parameters: Mapping[str, FloatArray]
) -> None:
    policy_names = tuple(name for name, _shape in _POLICY_SHAPES)
    value_names = tuple(name for name, _shape in _VALUE_SHAPES)
    if decoded["policy_parameter_sha256"] != _parameter_digest(parameters, policy_names):
        raise PolicyIntegrityError("policy parameter digest differs")
    if decoded["value_parameter_sha256"] != _parameter_digest(parameters, value_names):
        raise PolicyIntegrityError("value parameter digest differs")


class ReferencePPOLearnerAdapter:
    """On-policy, action-mask-aware PPO adapter with resumable immutable state."""

    def __init__(
        self,
        feature_contract: PPOFeatureContract,
        *,
        run_id: str,
        generation: int,
        compatibility: PolicyCompatibility,
        config: ReferencePPOConfigV1 | None = None,
    ) -> None:
        self.feature_contract = feature_contract
        self.feature_binding = bind_feature_contract(feature_contract)
        self.feature_profile = self.feature_binding.profile
        self.feature_contract_sha256 = str(self.feature_profile["contract_sha256"])
        self.feature_source_sha256 = str(self.feature_profile["source_sha256"])
        self.source_sha256 = LOADED_REFERENCE_PPO_SOURCE_SHA256
        self.runtime_sha256 = numpy_runtime_sha256()
        self.compatibility = compatibility
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id
        self.generation = _non_negative_int("generation", generation)
        self.config = ReferencePPOConfigV1() if config is None else config
        if (
            self.config.observation_size != self.feature_binding.feature_size
            or self.config.action_count != self.feature_binding.action_count
            or self.config.dtype != self.feature_profile["dtype"]
        ):
            raise PolicyCompatibilityError(
                "PPO config dimensions or dtype differ from feature profile"
            )
        if (
            compatibility.algorithm_id != REFERENCE_PPO_ALGORITHM_ID
            or compatibility.algorithm_version != REFERENCE_PPO_ALGORITHM_VERSION
        ):
            raise PolicyCompatibilityError("reference PPO compatibility identity differs")
        if compatibility.observation_schema_sha256 != self.feature_contract_sha256:
            raise PolicyCompatibilityError(
                "observation compatibility differs from feature contract"
            )
        if compatibility.action_schema_sha256 != self.feature_profile["action_schema_sha256"]:
            raise PolicyCompatibilityError(
                "action compatibility differs from feature action schema"
            )
        _assert_current_implementation(self.runtime_sha256)
        self._parameters = _new_parameters(self.config)
        self._adam_first = {name: np.zeros_like(value) for name, value in self._parameters.items()}
        self._adam_second = {name: np.zeros_like(value) for name, value in self._parameters.items()}
        self._adam_step = 0
        self._policy_version = 0
        self._update_count = 0
        self._environment_steps = 0
        self._bound_artifact_sha256: str | None = None
        self._last_behavior_artifact_sha256: str | None = None
        self._applied_batches: dict[str, str] = {}
        self._applied_batch_sizes: dict[str, int] = {}
        self._applied_batch_order: list[str] = []
        self._applied_records: dict[str, str] = {}
        self._transition_digests: list[str] = []
        self._stream_cursors: dict[tuple[str, str], tuple[int, float]] = {}
        self._closed_streams: set[tuple[str, str]] = set()
        self._transition_chain_sha256 = ZERO_SHA256
        self._last_metrics: PPOLossMetrics | None = None

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def adam_step(self) -> int:
        return self._adam_step

    @property
    def parameter_digests(self) -> tuple[str, str, str]:
        policy_names = tuple(name for name, _shape in _POLICY_SHAPES)
        value_names = tuple(name for name, _shape in _VALUE_SHAPES)
        return (
            _parameter_digest(self._parameters, policy_names),
            _parameter_digest(self._parameters, value_names),
            _optimizer_digest(self._adam_first, self._adam_second, self._adam_step),
        )

    def bind_published(
        self, reference: PolicyArtifactRef, artifact: PolicyArtifact
    ) -> None:
        _validate_published_pair(reference, artifact)
        if (reference.run_id, reference.generation) != (self.run_id, self.generation):
            raise PolicyCompatibilityError(
                "published reference run/generation differs from learner"
            )
        if reference.compatibility != self.compatibility:
            raise PolicyCompatibilityError("published reference compatibility differs")
        if reference.policy_version != self._policy_version:
            raise LearningContractError("published reference version differs from learner state")
        if artifact.payload != self._payload():
            raise PolicyIntegrityError(
                "published artifact payload differs from exact learner checkpoint"
            )
        expected_source = (
            EMPTY_TRANSITION_BATCH_SHA256
            if not self._applied_batch_order
            else self._applied_batch_order[-1]
        )
        if artifact.source_batch_sha256 != expected_source:
            raise PolicyIntegrityError("published artifact source batch differs")
        if self._bound_artifact_sha256 is not None:
            if self._bound_artifact_sha256 != reference.sha256:
                raise LearningContractError("learner policy version is already bound")
            return
        self._bound_artifact_sha256 = reference.sha256

    def validate_recovery_state(self, state: LearnerRecoveryState) -> None:
        if not isinstance(state, LearnerRecoveryState):
            raise TypeError("state must be LearnerRecoveryState")
        if (state.run_id, state.generation) != (self.run_id, self.generation):
            raise LearningContractError("learner recovery identity differs")
        if (
            state.last_published_version != self._policy_version
            or state.next_policy_version != self._policy_version + 1
            or state.last_published_reference is None
            or state.last_published_reference.sha256 != self._bound_artifact_sha256
            or state.failed
        ):
            raise LearningContractError(
                "learner recovery policy boundary differs from adapter checkpoint"
            )
        if set(state.consumed_batches) != set(self._applied_batches):
            raise LearningContractError("learner recovery consumed batches differ")
        if dict(state.consumed_records) != self._applied_records:
            raise LearningContractError("learner recovery consumed records differ")
        if dict(state.stream_positions) != self._stream_cursors:
            raise LearningContractError("learner recovery stream cursors differ")

    def initial_policy(self) -> PolicyCandidate:
        if self._update_count or self._environment_steps or self._policy_version:
            raise LearningContractError("initial policy is available only at fresh state")
        return self._candidate(initial=True)

    async def update(self, batch: ValidatedTransitionBatch) -> PolicyCandidate:
        _assert_current_implementation(self.runtime_sha256)
        self.feature_binding.assert_current()
        if not isinstance(batch, ValidatedTransitionBatch):
            raise TypeError("batch must be ValidatedTransitionBatch")
        if len(batch.records) > self.config.batch_size:
            raise LearningContractError("PPO batch exceeds the frozen rollout batch size")
        if self._bound_artifact_sha256 is None:
            raise LearningContractError("current behavior artifact must be bound before update")
        behavior_artifact_sha256 = self._bound_artifact_sha256
        previous_batch = self._applied_batches.get(batch.sha256)
        if previous_batch is not None:
            raise LearningContractError("PPO batch was already applied")
        self._validate_new_records(batch)
        arrays = self._batch_arrays(batch)
        metrics = self._optimize(batch, *arrays)
        record_digests: dict[str, str] = {}
        chain = self._transition_chain_sha256
        for record in batch.records:
            digest = _sha256(_canonical_json(record.to_dict()))
            record_digests[record.idempotency_key] = digest
            chain = _sha256(
                TRANSITION_CHAIN_DOMAIN.encode()
                + bytes.fromhex(chain)
                + bytes.fromhex(digest)
            )
        self._transition_chain_sha256 = chain
        self._applied_records.update(record_digests)
        self._applied_batches[batch.sha256] = chain
        self._applied_batch_sizes[batch.sha256] = len(batch.records)
        self._applied_batch_order.append(batch.sha256)
        self._transition_digests.extend(record_digests.values())
        for record in batch.records:
            self._stream_cursors[(record.worker_id, record.episode_id)] = (
                record.step_id,
                record.logical_time,
            )
            if record.terminated or record.truncated:
                self._closed_streams.add((record.worker_id, record.episode_id))
        self._environment_steps += len(batch.records)
        self._update_count += 1
        self._policy_version += 1
        self._last_behavior_artifact_sha256 = behavior_artifact_sha256
        self._bound_artifact_sha256 = None
        self._last_metrics = metrics
        return self._candidate(initial=False)

    def _validate_new_records(self, batch: ValidatedTransitionBatch) -> None:
        if (batch.run_id, batch.generation) != (self.run_id, self.generation):
            raise LearningContractError("PPO batch run/generation differs from learner")
        versions = {record.policy_version for record in batch.records}
        if versions != {self._policy_version}:
            raise LearningContractError("PPO batch must have zero lag and one behavior version")
        for record in batch.records:
            digest = _sha256(_canonical_json(record.to_dict()))
            previous = self._applied_records.get(record.idempotency_key)
            if previous is not None:
                if previous != digest:
                    raise LearningContractError("applied transition identity conflicts")
                raise LearningContractError("transition was already applied")
            stream = (record.worker_id, record.episode_id)
            if stream in self._closed_streams:
                raise LearningContractError("PPO cannot reopen a closed episode stream")
            cursor = self._stream_cursors.get(stream)
            if cursor is not None and record.step_id <= cursor[0]:
                raise LearningContractError("transition regresses behind checkpoint cursor")
            info = record.info
            if info.get("evaluation") is not False:
                raise LearningContractError(
                    "training transition must explicitly exclude evaluation"
                )
            if info.get("behavior_artifact_sha256") != self._bound_artifact_sha256:
                raise LearningContractError("transition behavior artifact differs")
            previous_mask = _mask_from_info(info, "previous_action_mask", self.config.action_count)
            next_mask = _mask_from_info(info, "next_action_mask", self.config.action_count)
            previous_input = _inference_mapping(record.previous_observation, "previous_observation")
            next_input = _inference_mapping(record.next_observation, "next_observation")
            _validate_transition_input(previous_input, record, previous=True)
            _validate_transition_input(next_input, record, previous=False)
            if previous_mask != self.feature_binding.action_mask(previous_input):
                raise LearningContractError("previous contemporaneous action mask differs")
            if next_mask != self.feature_binding.action_mask(next_input):
                raise LearningContractError("next contemporaneous action mask differs")
            action_index = self.feature_binding.encode_action(record.action)
            if not previous_mask[action_index]:
                raise LearningContractError("reference PPO action is masked or outside its space")
        # Validate the whole fragment before optimization; preserve the final
        # admitted cursor across batches, not just monotonic ordering.
        cursors = dict(self._stream_cursors)
        closed = set(self._closed_streams)
        for record in batch.records:
            stream = (record.worker_id, record.episode_id)
            if stream in closed:
                raise LearningContractError("PPO cannot continue after a terminal record")
            cursor = cursors.get(stream)
            if cursor is not None and (
                record.step_id != cursor[0] + 1 or record.logical_time < cursor[1]
            ):
                raise LearningContractError(
                    "PPO stream must remain step-contiguous and time-ordered"
                )
            cursors[stream] = (record.step_id, record.logical_time)
            if record.terminated or record.truncated:
                closed.add(stream)

    def _batch_arrays(
        self, batch: ValidatedTransitionBatch
    ) -> tuple[
        FloatArray,
        FloatArray,
        BoolArray,
        IntArray,
        FloatArray,
        BoolArray,
        BoolArray,
        list[tuple[str, str]],
        list[int],
    ]:
        previous = np.asarray(
            [
                self.feature_binding.encode(
                    _inference_mapping(record.previous_observation, "previous_observation")
                )
                for record in batch.records
            ],
            dtype=np.float32,
        )
        following = np.asarray(
            [
                self.feature_binding.encode(
                    _inference_mapping(record.next_observation, "next_observation")
                )
                for record in batch.records
            ],
            dtype=np.float32,
        )
        if (
            previous.shape != (len(batch.records), self.config.observation_size)
            or following.shape != previous.shape
        ):
            raise LearningContractError("feature batch shape differs from PPO config")
        if not bool(np.all(np.isfinite(previous))) or not bool(np.all(np.isfinite(following))):
            raise LearningContractError("PPO feature batches must be finite")
        masks = np.asarray(
            [
                _mask_from_info(
                    record.info, "previous_action_mask", self.config.action_count
                )
                for record in batch.records
            ],
            dtype=np.bool_,
        )
        actions = np.asarray(
            [self.feature_binding.encode_action(record.action) for record in batch.records],
            dtype=np.int64,
        )
        rewards = np.asarray(
            [record.reward for record in batch.records], dtype=np.float32
        )
        terminated = np.asarray(
            [record.terminated for record in batch.records], dtype=np.bool_
        )
        truncated = np.asarray(
            [record.truncated for record in batch.records], dtype=np.bool_
        )
        streams = [(record.worker_id, record.episode_id) for record in batch.records]
        steps = [record.step_id for record in batch.records]
        return previous, following, masks, actions, rewards, terminated, truncated, streams, steps

    def _optimize(
        self,
        batch: ValidatedTransitionBatch,
        observations: FloatArray,
        next_observations: FloatArray,
        masks: BoolArray,
        actions: IntArray,
        rewards: FloatArray,
        terminated: BoolArray,
        truncated: BoolArray,
        streams: list[tuple[str, str]],
        steps: list[int],
    ) -> PPOLossMetrics:
        logits, _ = _forward_tower(observations, self._parameters, "policy")
        probabilities, log_probabilities, _entropy = _masked_distribution(logits, masks)
        action_indices = np.asarray(actions, dtype=np.int64)
        row_indices = np.arange(len(batch.records), dtype=np.int64)
        old_log_probabilities = np.asarray(
            log_probabilities[row_indices, action_indices], dtype=np.float32
        )
        values_raw, _ = _forward_tower(observations, self._parameters, "value")
        next_values_raw, _ = _forward_tower(next_observations, self._parameters, "value")
        old_values = np.asarray(values_raw[:, 0], dtype=np.float32)
        next_values = np.asarray(next_values_raw[:, 0], dtype=np.float32)
        advantages, returns = compute_gae(
            rewards,
            old_values,
            next_values,
            terminated,
            truncated,
            streams,
            steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        mean = np.mean(advantages, dtype=np.float32)
        variance = np.mean((advantages - mean) ** 2, dtype=np.float32)
        normalized = np.asarray(
            (advantages - mean) / np.sqrt(variance + np.float32(self.config.advantage_epsilon)),
            dtype=np.float32,
        )
        metrics: PPOLossMetrics | None = None
        for epoch in range(self.config.epochs):
            order = sorted(
                range(len(batch.records)),
                key=lambda index: (
                    hashlib.sha256(
                        b"\x00".join(
                            (
                                SHUFFLE_DOMAIN.encode(),
                                str(self.config.shuffle_seed).encode(),
                                str(self._update_count).encode(),
                                str(epoch).encode(),
                                batch.records[index].idempotency_key.encode(),
                            )
                        )
                    ).digest(),
                    batch.records[index].idempotency_key,
                ),
            )
            for start in range(0, len(order), self.config.minibatch_size):
                indices = np.asarray(
                    order[start : start + self.config.minibatch_size], dtype=np.int64
                )
                metrics = self._minibatch_step(
                    observations[indices],
                    masks[indices],
                    action_indices[indices],
                    old_log_probabilities[indices],
                    old_values[indices],
                    normalized[indices],
                    returns[indices],
                )
        if metrics is None:
            raise LearningContractError("PPO optimizer performed no minibatch step")
        return metrics

    def _minibatch_step(
        self,
        x: FloatArray,
        masks: BoolArray,
        actions: npt.NDArray[np.int64],
        old_log_probabilities: FloatArray,
        _old_values: FloatArray,
        advantages: FloatArray,
        returns: FloatArray,
    ) -> PPOLossMetrics:
        logits, policy_cache = _forward_tower(x, self._parameters, "policy")
        value_output, value_cache = _forward_tower(x, self._parameters, "value")
        loss = _loss_and_output_gradients(
            logits,
            masks,
            actions,
            old_log_probabilities,
            advantages,
            np.asarray(value_output[:, 0], dtype=np.float32),
            returns,
            policy_clip=self.config.policy_clip,
            value_coefficient=self.config.value_coefficient,
            entropy_coefficient=self.config.entropy_coefficient,
        )
        gradients = _backward_tower(
            x, policy_cache, loss.policy, self._parameters, "policy"
        )
        gradients.update(
            _backward_tower(x, value_cache, loss.value, self._parameters, "value")
        )
        clipped_gradients, norm = _clip_gradients_by_global_norm(
            gradients, self.config.max_gradient_norm
        )
        self._adam_step += 1
        beta1 = np.float32(self.config.adam_beta1)
        beta2 = np.float32(self.config.adam_beta2)
        for name in sorted(self._parameters):
            gradient = clipped_gradients[name]
            first = self._adam_first[name]
            second = self._adam_second[name]
            first[:] = beta1 * first + (np.float32(1.0) - beta1) * gradient
            second[:] = beta2 * second + (np.float32(1.0) - beta2) * gradient * gradient
            first_hat = first / np.float32(1.0 - self.config.adam_beta1**self._adam_step)
            second_hat = second / np.float32(1.0 - self.config.adam_beta2**self._adam_step)
            update = np.float32(self.config.learning_rate) * first_hat / (
                np.sqrt(second_hat) + np.float32(self.config.adam_epsilon)
            )
            self._parameters[name][:] = np.asarray(
                self._parameters[name] - update, dtype=np.float32
            )
            if not bool(np.all(np.isfinite(self._parameters[name]))):
                raise LearningContractError("PPO parameter update is non-finite")
        return PPOLossMetrics(
            policy_loss=loss.policy_loss,
            value_loss=loss.value_loss,
            entropy=loss.entropy,
            total_loss=loss.total_loss,
            gradient_norm=float(norm),
            clipped_fraction=loss.clipped_fraction,
        )

    def _candidate(self, *, initial: bool) -> PolicyCandidate:
        payload = self._payload()
        checkpoint = ReferencePPOCheckpointV1(payload)
        policy_digest, value_digest, optimizer_digest = self.parameter_digests
        return PolicyCandidate(
            payload=payload,
            media_type=REFERENCE_PPO_MEDIA_TYPE,
            compatibility=self.compatibility,
            provenance={
                "adapter": "pyjevsim_bridge.rl.reference_ppo.ReferencePPOLearnerAdapter",
                "initial_policy": initial,
                "config_sha256": self.config.sha256,
                "feature_contract_sha256": self.feature_contract_sha256,
                "policy_parameter_sha256": policy_digest,
                "value_parameter_sha256": value_digest,
                "optimizer_sha256": optimizer_digest,
                "checkpoint_sha256": checkpoint.sha256,
                "update_count": self._update_count,
                "environment_steps": self._environment_steps,
                "adam_step": self._adam_step,
            },
        )

    def _payload(self) -> bytes:
        _assert_current_implementation(self.runtime_sha256)
        self.feature_binding.assert_current()
        policy_digest, value_digest, optimizer_digest = self.parameter_digests
        cursors = [
            {
                "worker_id": key[0],
                "episode_id": key[1],
                "step_id": value[0],
                "logical_time": value[1],
            }
            for key, value in sorted(self._stream_cursors.items())
        ]
        value: dict[str, object] = {
            "schema_version": REFERENCE_PPO_CHECKPOINT_SCHEMA,
            "algorithm_id": REFERENCE_PPO_ALGORITHM_ID,
            "algorithm_version": REFERENCE_PPO_ALGORITHM_VERSION,
            "objective_id": REFERENCE_PPO_OBJECTIVE_ID,
            "config": self.config.content(),
            "config_sha256": self.config.sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "feature_source_sha256": self.feature_source_sha256,
            "feature_profile": self.feature_profile,
            "ppo_features_source_sha256": LOADED_PPO_FEATURES_SOURCE_SHA256,
            "action_schema_sha256": self.feature_profile["action_schema_sha256"],
            "reference_ppo_source_sha256": self.source_sha256,
            "numpy_runtime_sha256": self.runtime_sha256,
            "run_id": self.run_id,
            "generation": self.generation,
            "policy_version": self._policy_version,
            "update_count": self._update_count,
            "environment_steps": self._environment_steps,
            "adam_step": self._adam_step,
            "bound_behavior_artifact_sha256": self._last_behavior_artifact_sha256,
            "transition_chain_sha256": self._transition_chain_sha256,
            "transition_digests": list(self._transition_digests),
            "applied_batches": [
                {
                    "batch_sha256": key,
                    "chain_sha256": self._applied_batches[key],
                    "record_count": self._applied_batch_sizes[key],
                    "update_index": index,
                }
                for index, key in enumerate(self._applied_batch_order)
            ],
            "applied_records": [
                {"idempotency_key": key, "record_sha256": value}
                for key, value in sorted(self._applied_records.items())
            ],
            "stream_cursors": cursors,
            "closed_streams": [
                {"worker_id": worker, "episode_id": episode}
                for worker, episode in sorted(self._closed_streams)
            ],
            "parameters": _encode_tensors(self._parameters),
            "adam_first": _encode_tensors(self._adam_first),
            "adam_second": _encode_tensors(self._adam_second),
            "policy_parameter_sha256": policy_digest,
            "value_parameter_sha256": value_digest,
            "optimizer_sha256": optimizer_digest,
            "seed_domains": {
                "initialization": INITIALIZATION_DOMAIN,
                "initialization_seed": self.config.initialization_seed,
                "sampling": SAMPLING_DOMAIN,
                "shuffle": SHUFFLE_DOMAIN,
                "shuffle_seed": self.config.shuffle_seed,
            },
            "runtime": dict(numpy_runtime_identity()),
            "last_metrics": None if self._last_metrics is None else self._last_metrics.content(),
        }
        return _canonical_json(value)

    @classmethod
    def from_artifact(
        cls,
        artifact: PolicyArtifact,
        feature_contract: PPOFeatureContract,
    ) -> Self:
        if artifact.media_type != REFERENCE_PPO_MEDIA_TYPE:
            raise PolicyCompatibilityError("reference PPO media type differs")
        decoded = _decode_payload(artifact.payload)
        _validate_source_runtime(decoded)
        if (decoded["run_id"], decoded["generation"], decoded["policy_version"]) != (
            artifact.run_id,
            artifact.generation,
            artifact.policy_version,
        ):
            raise PolicyIntegrityError("checkpoint identity differs from artifact")
        config, parameters, first, second = _validate_checkpoint_contents(
            decoded, source_batch_sha256=artifact.source_batch_sha256
        )
        result = cls(
            feature_contract,
            run_id=artifact.run_id,
            generation=artifact.generation,
            compatibility=artifact.compatibility,
            config=config,
        )
        if decoded["feature_contract_sha256"] != result.feature_contract_sha256:
            raise PolicyCompatibilityError("checkpoint feature contract differs")
        if decoded["feature_source_sha256"] != result.feature_source_sha256:
            raise PolicyCompatibilityError("checkpoint feature source differs")
        if decoded["feature_profile"] != result.feature_profile:
            raise PolicyCompatibilityError("checkpoint feature profile differs")
        adam_step = _non_negative_int("adam_step", decoded["adam_step"])
        result._parameters = parameters
        result._adam_first = first
        result._adam_second = second
        result._adam_step = adam_step
        result._policy_version = _non_negative_int("policy_version", decoded["policy_version"])
        result._update_count = _non_negative_int("update_count", decoded["update_count"])
        result._environment_steps = _non_negative_int(
            "environment_steps", decoded["environment_steps"]
        )
        result._bound_artifact_sha256 = artifact.sha256
        behavior = decoded["bound_behavior_artifact_sha256"]
        result._last_behavior_artifact_sha256 = (
            None
            if behavior is None
            else _require_sha256("bound_behavior_artifact_sha256", behavior)
        )
        result._transition_chain_sha256 = _require_sha256(
            "transition_chain_sha256", decoded["transition_chain_sha256"]
        )
        batches = _checkpoint_batches(decoded["applied_batches"])
        result._applied_batches = {item[0]: item[1] for item in batches}
        result._applied_batch_sizes = {item[0]: item[2] for item in batches}
        result._applied_batch_order = [item[0] for item in batches]
        result._applied_records = _decode_pairs(
            decoded["applied_records"], "idempotency_key", "record_sha256", key_digest=False
        )
        raw_digests = cast(list[object], decoded["transition_digests"])
        result._transition_digests = [
            _require_sha256(f"transition_digests[{index}]", digest)
            for index, digest in enumerate(raw_digests)
        ]
        result._stream_cursors = _decode_cursors(decoded["stream_cursors"])
        result._closed_streams = _decode_closed_streams(
            decoded["closed_streams"], result._stream_cursors
        )
        metrics = decoded["last_metrics"]
        if metrics is not None:
            if not isinstance(metrics, dict) or set(metrics) != {
                item.name for item in fields(PPOLossMetrics)
            }:
                raise PolicyIntegrityError("checkpoint loss metrics differ")
            result._last_metrics = PPOLossMetrics(
                **{name: _finite(name, raw) for name, raw in metrics.items()}
            )
        return result


def _validate_transition_input(
    value: Mapping[str, object], record: TransitionRecord, *, previous: bool
) -> None:
    expected = {
        "run_id": record.run_id,
        "generation": record.generation,
        "worker_id": record.worker_id,
        "episode_id": record.episode_id,
        "step_id": record.step_id - 1 if previous else record.step_id,
    }
    if record.step_id < 1 or any(
        type(value.get(name)) is not type(item) or value.get(name) != item
        for name, item in expected.items()
    ):
        name = "previous" if previous else "next"
        raise LearningContractError(f"PPO {name} observation identity differs from transition")


def _mask_from_info(
    info: Mapping[str, object], name: str, action_count: int
) -> tuple[bool, ...]:
    value = info.get(name)
    if not isinstance(value, tuple):
        raise LearningContractError(f"{name} must be a tuple")
    if (
        len(value) != action_count
        or any(type(item) is not bool for item in value)
        or not any(value)
    ):
        raise LearningContractError(f"{name} is not a valid action mask")
    return cast(tuple[bool, ...], value)


def _inference_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LearningContractError(f"{name} must be a PPO inference-input mapping")
    return cast(Mapping[str, object], value)


def _decode_pairs(
    value: object,
    key_name: str,
    value_name: str,
    *,
    key_digest: bool = True,
) -> dict[str, str]:
    if not isinstance(value, list):
        raise PolicyIntegrityError("checkpoint applied identity table must be an array")
    result: dict[str, str] = {}
    for row in value:
        if not isinstance(row, dict) or set(row) != {key_name, value_name}:
            raise PolicyIntegrityError("checkpoint applied identity row differs")
        key = row[key_name]
        if not isinstance(key, str) or not key:
            raise PolicyIntegrityError("checkpoint applied identity key is invalid")
        if key_digest:
            _require_sha256(key_name, key)
        digest = _require_sha256(value_name, row[value_name])
        if key in result:
            raise PolicyIntegrityError("checkpoint applied identity repeats a key")
        result[key] = digest
    return result


def _decode_cursors(value: object) -> dict[tuple[str, str], tuple[int, float]]:
    if not isinstance(value, list):
        raise PolicyIntegrityError("checkpoint stream cursors must be an array")
    result: dict[tuple[str, str], tuple[int, float]] = {}
    expected = {"worker_id", "episode_id", "step_id", "logical_time"}
    for row in value:
        if not isinstance(row, dict) or set(row) != expected:
            raise PolicyIntegrityError("checkpoint stream cursor fields differ")
        worker = row["worker_id"]
        episode = row["episode_id"]
        if not isinstance(worker, str) or not worker or not isinstance(episode, str) or not episode:
            raise PolicyIntegrityError("checkpoint stream cursor identity is invalid")
        key = (worker, episode)
        if key in result:
            raise PolicyIntegrityError("checkpoint stream cursor repeats an identity")
        result[key] = (
            _non_negative_int("cursor step_id", row["step_id"]),
            _finite("cursor logical_time", row["logical_time"]),
        )
    return result


def _decode_closed_streams(
    value: object, cursors: Mapping[tuple[str, str], tuple[int, float]]
) -> set[tuple[str, str]]:
    if not isinstance(value, list):
        raise PolicyIntegrityError("checkpoint closed_streams must be an array")
    keys: list[tuple[str, str]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"worker_id", "episode_id"}:
            raise PolicyIntegrityError("checkpoint closed stream fields differ")
        worker, episode = row["worker_id"], row["episode_id"]
        if not isinstance(worker, str) or not worker or not isinstance(episode, str) or not episode:
            raise PolicyIntegrityError("checkpoint closed stream identity is invalid")
        key = (worker, episode)
        if key not in cursors:
            raise PolicyIntegrityError("checkpoint closed stream has no cursor")
        keys.append(key)
    if keys != sorted(set(keys)):
        raise PolicyIntegrityError("checkpoint closed streams must be sorted and unique")
    return set(keys)


# Versioned public spelling frozen by IF-RL-016; retain the concise source name
# as a compatibility alias.
ReferencePPOCapabilityReceiptV1 = ReferencePPOCapabilityReceipt


def __getattr__(name: str) -> object:
    """Deprecated explicit M1 exports; common imports never load a model."""
    if name in {"REFERENCE_PPO_ACTION_SCHEMA_SHA256", "REFERENCE_PPO_FEATURE_ID"}:
        from pyjevsim_bridge.rl.qualification_models.anti_torpedo_features import (
            ANTI_TORPEDO_ACTION_SCHEMA_SHA256,
            ANTI_TORPEDO_FEATURE_SCHEMA_VERSION,
        )
        if name == "REFERENCE_PPO_ACTION_SCHEMA_SHA256":
            return ANTI_TORPEDO_ACTION_SCHEMA_SHA256
        return ANTI_TORPEDO_FEATURE_SCHEMA_VERSION
    raise AttributeError(name)


__all__ = [
    "FROZEN_REFERENCE_PPO_CONFIG_SHA256",
    "LOADED_REFERENCE_PPO_SOURCE_SHA256",
    "PPOInferenceInputV1",
    "PPOLossMetrics",
    "REFERENCE_PPO_ACTION_SCHEMA_SHA256",  # noqa: F822 - explicit lazy legacy export
    "REFERENCE_PPO_ALGORITHM_ID",
    "REFERENCE_PPO_ALGORITHM_VERSION",
    "REFERENCE_PPO_OBJECTIVE_ID",
    "REFERENCE_PPO_CHECKPOINT_SCHEMA",
    "REFERENCE_PPO_FEATURE_ID",  # noqa: F822 - explicit lazy legacy export
    "REFERENCE_PPO_MEDIA_TYPE",
    "ReferencePPOCapabilityReceipt",
    "ReferencePPOCapabilityReceiptV1",
    "ReferencePPOCheckpointV1",
    "ReferencePPOConfigV1",
    "ReferencePPOLearnerAdapter",
    "ReferencePPOPolicyLoader",
    "ReferencePPORolloutScope",
    "compute_gae",
    "numpy_runtime_identity",
    "numpy_runtime_sha256",
    "open_rollout_scope",
    "reference_ppo_source_sha256",
]
