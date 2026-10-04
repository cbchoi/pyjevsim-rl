"""Model-neutral structural contract for CPU float32 PPO feature plugins.

The model owns semantics; this module validates the closed interface, source
identity, masks, finite features and action codec without importing any model.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from pyjevsim_bridge.rl.inference_metrics import file_sha256, metric_function
from pyjevsim_bridge.rl.learning import LearningContractError

PPO_INFERENCE_INPUT_SCHEMA_VERSION = "ppo-inference-input-v1"
PPO_INFERENCE_INPUT_FIELDS = frozenset({
    "schema_version", "observation", "action_mask", "run_id", "generation",
    "worker_id", "episode_id", "step_id", "sampling_seed", "explore",
})
FEATURE_PROFILE_FIELDS = frozenset({
    "feature_id", "schema_version", "feature_size", "action_count", "dtype",
    "contract_sha256", "source_sha256", "action_schema_sha256", "codec_sha256",
})


class FeatureContractError(LearningContractError, ValueError):
    """A feature plugin violates its declared structural or source contract."""


class PPOFeatureContract(Protocol):
    @property
    def feature_id(self) -> str: ...
    @property
    def schema_version(self) -> str: ...
    @property
    def feature_size(self) -> int: ...
    @property
    def action_count(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    @property
    def contract_sha256(self) -> str: ...
    @property
    def source_sha256(self) -> str: ...
    @property
    def action_schema_sha256(self) -> str: ...

    def observation_action_mask(self, observation: object) -> tuple[bool, ...]: ...
    def encode(self, value: Mapping[str, object]) -> tuple[float, ...]: ...
    def action_mask(self, value: Mapping[str, object]) -> tuple[bool, ...]: ...
    def encode_action(self, action: object) -> int: ...
    def decode_action(self, index: int) -> object: ...


def _json_value(value: object) -> object:
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise FeatureContractError("closed JSON must not contain nonfinite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise FeatureContractError("closed JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise FeatureContractError("value is not closed canonical JSON")


def canonical_feature_json(value: object) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def ppo_features_source_sha256() -> str:
    return file_sha256(Path(__file__), role="ppo_features_source")


LOADED_PPO_FEATURES_SOURCE_SHA256 = ppo_features_source_sha256()


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise FeatureContractError(f"{name} must be a nonempty string")
    return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FeatureContractError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(name: str, value: object) -> str:
    result = _text(name, value)
    if len(result) != 64 or any(item not in "0123456789abcdef" for item in result):
        raise FeatureContractError(f"{name} must be a lowercase SHA256 digest")
    return result


def validate_action_mask(value: object, action_count: int) -> tuple[bool, ...]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != action_count
        or any(type(item) is not bool for item in value)
        or not any(value)
    ):
        raise FeatureContractError("action mask must have exact boolean width and a valid action")
    return tuple(value)


def validate_inference_envelope(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != PPO_INFERENCE_INPUT_FIELDS:
        raise FeatureContractError("inference envelope fields differ from the closed schema")
    if value["schema_version"] != PPO_INFERENCE_INPUT_SCHEMA_VERSION:
        raise FeatureContractError("inference envelope schema differs")
    for name in ("run_id", "worker_id", "episode_id"):
        _text(name, value[name])
    for name in ("generation", "step_id", "sampling_seed"):
        _integer(name, value[name])
    if type(value["explore"]) is not bool:
        raise FeatureContractError("inference explore must be bool")
    canonical_feature_json(value["observation"])
    return cast(Mapping[str, object], value)


def validate_feature_profile(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != FEATURE_PROFILE_FIELDS:
        raise FeatureContractError("feature profile fields differ")
    for name in ("feature_id", "schema_version"):
        _text(name, value[name])
    for name in ("feature_size", "action_count"):
        _integer(name, value[name], minimum=1)
    if value["dtype"] != "float32":
        raise FeatureContractError("feature dtype must be float32")
    for name in ("contract_sha256", "source_sha256", "action_schema_sha256", "codec_sha256"):
        _digest(name, value[name])
    return dict(value)


@dataclass(frozen=True, slots=True)
class FeatureContractBinding:
    """Snapshot of a validated plugin; never trusts subsequent metadata drift."""

    contract: PPOFeatureContract
    source_path: Path
    profile_json: bytes
    action_json: tuple[bytes, ...]

    @property
    def profile(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self.profile_json))

    @property
    def feature_size(self) -> int:
        return cast(int, self.profile["feature_size"])

    @property
    def action_count(self) -> int:
        return len(self.action_json)

    @metric_function("integrity")
    def assert_current(self) -> None:
        profile = self.profile
        try:
            actual_source = file_sha256(self.source_path, role="feature_plugin_source")
        except OSError as exc:
            raise FeatureContractError("feature source is no longer readable") from exc
        if actual_source != profile["source_sha256"]:
            raise FeatureContractError("feature source changed after binding")
        self._assert_metadata_current(profile)

    def _assert_metadata_current(self, profile: Mapping[str, object]) -> None:
        for name in FEATURE_PROFILE_FIELDS - {"codec_sha256"}:
            actual = getattr(self.contract, name, None)
            if type(actual) is not type(profile[name]) or actual != profile[name]:
                raise FeatureContractError(f"feature metadata changed after binding: {name}")

    @metric_function("input_validation")
    def observation_action_mask(self, observation: object) -> tuple[bool, ...]:
        self.assert_current()
        canonical_feature_json(observation)
        return validate_action_mask(
            self.contract.observation_action_mask(observation), self.action_count
        )

    @metric_function("input_validation")
    def action_mask(self, value: Mapping[str, object]) -> tuple[bool, ...]:
        self.assert_current()
        envelope = validate_inference_envelope(value)
        supplied = validate_action_mask(envelope["action_mask"], self.action_count)
        derived = self.observation_action_mask(envelope["observation"])
        full = validate_action_mask(self.contract.action_mask(envelope), self.action_count)
        if supplied != derived or supplied != full:
            raise FeatureContractError("contemporaneous action mask differs from state")
        return supplied

    @metric_function("feature")
    def encode(self, value: Mapping[str, object]) -> tuple[float, ...]:
        self.action_mask(value)
        encoded = self.contract.encode(value)
        if not isinstance(encoded, tuple) or len(encoded) != self.feature_size:
            raise FeatureContractError("feature vector must be a tuple of exact declared width")
        result: list[float] = []
        for item in encoded:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise FeatureContractError("feature values must be finite numbers")
            try:
                converted = struct.unpack("<f", struct.pack("<f", float(item)))[0]
            except (OverflowError, struct.error) as exc:
                raise FeatureContractError("feature values must be finite float32") from exc
            if not math.isfinite(converted):
                raise FeatureContractError("feature values must be finite float32")
            result.append(converted)
        return tuple(result)

    @metric_function("decode")
    def encode_action(self, action: object) -> int:
        self.assert_current()
        encoded = canonical_feature_json(action)
        index = _integer("action codec index", self.contract.encode_action(action))
        if index >= self.action_count or encoded != self.action_json[index]:
            raise FeatureContractError("action codec input differs from declared action set")
        if canonical_feature_json(self.contract.decode_action(index)) != encoded:
            raise FeatureContractError("action codec changed after binding")
        return index

    @metric_function("decode")
    def decode_action(self, index: int) -> object:
        self.assert_current()
        _integer("action codec index", index)
        if index >= self.action_count:
            raise FeatureContractError("action codec index outside declared action set")
        action = self.contract.decode_action(index)
        if canonical_feature_json(action) != self.action_json[index]:
            raise FeatureContractError("action codec changed after binding")
        if self.encode_action(action) != index:
            raise FeatureContractError("action codec is no longer bijective")
        return json.loads(self.action_json[index])


@dataclass(frozen=True, slots=True)
class _ScopedFeatureContractBinding(FeatureContractBinding):
    """Internal queue-only scope view; inherited plugin calls stay unchanged."""

    validate_owner: Callable[[], None]

    @metric_function("integrity")
    def assert_current(self) -> None:
        self.validate_owner()
        self._assert_metadata_current(self.profile)


def _scoped_feature_binding(
    binding: FeatureContractBinding, validate_owner: Callable[[], None],
) -> FeatureContractBinding:
    return _ScopedFeatureContractBinding(
        binding.contract, binding.source_path, binding.profile_json,
        binding.action_json, validate_owner,
    )


def bind_feature_contract(contract: PPOFeatureContract) -> FeatureContractBinding:
    profile = {name: getattr(contract, name, None)
               for name in FEATURE_PROFILE_FIELDS - {"codec_sha256"}}
    profile["codec_sha256"] = "0" * 64
    validate_feature_profile(profile)
    methods = ("observation_action_mask", "encode", "action_mask", "encode_action", "decode_action")
    for method in methods:
        if not callable(getattr(contract, method, None)):
            raise FeatureContractError(f"feature contract is missing {method}")
    source_name = inspect.getsourcefile(type(contract))
    if source_name is None:
        raise FeatureContractError("feature class must have a file-backed Python source module")
    source_path = Path(source_name).resolve()
    try:
        actual_source = file_sha256(source_path, role="feature_plugin_source")
    except OSError as exc:
        raise FeatureContractError("feature source module is not readable") from exc
    if actual_source != profile["source_sha256"]:
        raise FeatureContractError("feature defining-module source digest differs")
    actions: list[bytes] = []
    for index in range(cast(int, profile["action_count"])):
        action = contract.decode_action(index)
        actions.append(canonical_feature_json(action))
        roundtrip = _integer("action codec roundtrip", contract.encode_action(action))
        if roundtrip != index:
            raise FeatureContractError("action codec must be a bijection")
    if len(set(actions)) != len(actions):
        raise FeatureContractError("action codec has canonical JSON collisions")
    profile["codec_sha256"] = hashlib.sha256(
        canonical_feature_json([json.loads(action) for action in actions])
    ).hexdigest()
    result = FeatureContractBinding(
        contract, source_path, canonical_feature_json(profile), tuple(actions)
    )
    result.assert_current()
    return result


__all__ = [
    "LOADED_PPO_FEATURES_SOURCE_SHA256", "PPO_INFERENCE_INPUT_SCHEMA_VERSION",
    "PPOFeatureContract", "FeatureContractBinding", "FeatureContractError",
    "bind_feature_contract", "canonical_feature_json", "ppo_features_source_sha256",
    "validate_action_mask", "validate_feature_profile", "validate_inference_envelope",
]
