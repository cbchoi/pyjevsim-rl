"""Frozen TASK-RL-105 feature and action-mask contract for AT/SIM v2."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from pyjevsim_bridge.rl.ppo_features import PPO_INFERENCE_INPUT_SCHEMA_VERSION
from pyjevsim_bridge.rl.qualification_models.anti_torpedo_profile import (
    build_scenario_bank_v1,
)

V2_OBSERVATION_SCHEMA_VERSION: Final = "anti-torpedo-observation-v2"
ANTI_TORPEDO_FEATURE_SCHEMA_VERSION: Final = "anti-torpedo-feature-v1"
ANTI_TORPEDO_FEATURE_SIZE: Final = 80
ANTI_TORPEDO_ACTION_COUNT: Final = 6

_INPUT_KEYS: Final = frozenset(
    {
        "schema_version",
        "observation",
        "action_mask",
        "run_id",
        "generation",
        "worker_id",
        "episode_id",
        "step_id",
        "sampling_seed",
        "explore",
    }
)
_OBSERVATION_KEYS: Final = frozenset(
    {
        "schema_version",
        "tick",
        "ship",
        "torpedo",
        "relative",
        "pending_target",
        "decoy_launched",
        "decoys",
        "scenario",
        "previous_action",
        "scenario_ordinal",
        "factor_vector",
        "config_sha256",
        "scenario_family_sha256",
    }
)
_PADDING: Final = (False, "", "none", False, 0.0, 0.0, 0.0, 0.0, 0.0, False)
_SCENARIO_BANK: Final = build_scenario_bank_v1()


class AntiTorpedoFeatureError(ValueError):
    """Raised before inference when the frozen feature contract is violated."""


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
        raise AntiTorpedoFeatureError("feature input is not closed JSON") from exc


_CONTRACT_CONTENT: Final = {
    "action_count": ANTI_TORPEDO_ACTION_COUNT,
    "action_mask_rule": "TTT-not-launched-not-launched-not-launched",
    "decoy_order": "sense-id-ascending-four-slots",
    "feature_order": (
        "tick;ship7;torpedo7;relative5;pending4;launched;"
        "decoy0..3x10;factor-b0..b7;previous-action--1..5"
    ),
    "feature_size": ANTI_TORPEDO_FEATURE_SIZE,
    "float": "little-endian-ieee754-float32",
    "normalization": {
        "tick": 30.0,
        "xy": 512.0,
        "z": 64.0,
        "range": 1024.0,
        "closing_and_speed": 16.0,
        "lifespan": 10.0,
        "time_of_flight": 30.0,
        "heading": "sin-cos-degrees",
    },
    "observation_schema": V2_OBSERVATION_SCHEMA_VERSION,
    "schema_version": ANTI_TORPEDO_FEATURE_SCHEMA_VERSION,
    "plugin_contract": "ppo-feature-contract-v1",
    "raw_mask": "closed-observation-decoy-launched-boolean",
    "action_codec": "bijective-int-0-through-5",
}
ANTI_TORPEDO_FEATURE_CONTRACT_SHA256: Final = hashlib.sha256(
    _canonical_json(_CONTRACT_CONTENT)
).hexdigest()
ANTI_TORPEDO_ACTION_SCHEMA_SHA256: Final = hashlib.sha256(
    _canonical_json({
        "action_count": ANTI_TORPEDO_ACTION_COUNT,
        "actions": list(range(ANTI_TORPEDO_ACTION_COUNT)),
        "feature_contract_sha256": ANTI_TORPEDO_FEATURE_CONTRACT_SHA256,
        "mask_rule": "TTT-not-launched-not-launched-not-launched",
        "schema_version": "reference-ppo-action-v1",
    })
).hexdigest()


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


LOADED_ANTI_TORPEDO_FEATURE_SOURCE_SHA256: Final = _source_sha256()


def _mapping(name: str, value: object, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise AntiTorpedoFeatureError(f"{name} fields differ from the closed schema")
    if any(not isinstance(key, str) for key in value):
        raise AntiTorpedoFeatureError(f"{name} keys must be strings")
    return cast(Mapping[str, object], value)


def _sequence(name: str, value: object, length: int) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise AntiTorpedoFeatureError(f"{name} must contain exactly {length} values")
    return tuple(value)


def _integer(name: str, value: object, *, lower: int, upper: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AntiTorpedoFeatureError(f"{name} must be an integer")
    if value < lower or (upper is not None and value > upper):
        raise AntiTorpedoFeatureError(f"{name} is outside the closed domain")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise AntiTorpedoFeatureError(f"{name} must be a non-empty string")
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise AntiTorpedoFeatureError(f"{name} must be bool")
    return value


def _number(name: str, value: object, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AntiTorpedoFeatureError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < lower or number > upper:
        raise AntiTorpedoFeatureError(f"{name} is outside the closed finite domain")
    return number


def _f32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise AntiTorpedoFeatureError("feature is not representable as float32") from exc
    if not math.isfinite(result):
        raise AntiTorpedoFeatureError("feature float32 round-trip is not finite")
    return cast(float, result)


class AntiTorpedoV2FeatureContract:
    """Pure deterministic encoder; identity fields are validated, never learned."""

    schema_version: Final = ANTI_TORPEDO_FEATURE_SCHEMA_VERSION
    feature_id: Final = ANTI_TORPEDO_FEATURE_SCHEMA_VERSION
    dtype: Final = "float32"
    feature_size: Final = ANTI_TORPEDO_FEATURE_SIZE
    action_count: Final = ANTI_TORPEDO_ACTION_COUNT
    contract_sha256: Final = ANTI_TORPEDO_FEATURE_CONTRACT_SHA256
    source_sha256: Final = LOADED_ANTI_TORPEDO_FEATURE_SOURCE_SHA256
    action_schema_sha256: Final = ANTI_TORPEDO_ACTION_SCHEMA_SHA256

    def observation_action_mask(self, observation: object) -> tuple[bool, ...]:
        """Derive initial mask without inventing a prior inference envelope."""
        if _source_sha256() != self.source_sha256:
            raise AntiTorpedoFeatureError("feature source changed after import")
        value = _mapping("observation", observation, _OBSERVATION_KEYS)
        if value["schema_version"] != V2_OBSERVATION_SCHEMA_VERSION:
            raise AntiTorpedoFeatureError("only the physical v2 observation is admitted")
        launched = _boolean("decoy_launched", value["decoy_launched"])
        return (True, True, True, not launched, not launched, not launched)

    def encode_action(self, action: object) -> int:
        return _integer("action", action, lower=0, upper=self.action_count - 1)

    def decode_action(self, index: int) -> object:
        return _integer("action index", index, lower=0, upper=self.action_count - 1)

    def _validated(
        self, value: Mapping[str, object]
    ) -> tuple[Mapping[str, object], tuple[bool, ...]]:
        if _source_sha256() != self.source_sha256:
            raise AntiTorpedoFeatureError("feature source changed after import")
        root = _mapping("inference input", value, _INPUT_KEYS)
        if root["schema_version"] != PPO_INFERENCE_INPUT_SCHEMA_VERSION:
            raise AntiTorpedoFeatureError("inference input schema_version differs")
        _text("run_id", root["run_id"])
        _integer("generation", root["generation"], lower=0)
        _text("worker_id", root["worker_id"])
        _text("episode_id", root["episode_id"])
        _integer("step_id", root["step_id"], lower=0)
        _integer("sampling_seed", root["sampling_seed"], lower=0)
        _boolean("explore", root["explore"])

        observation = _mapping("observation", root["observation"], _OBSERVATION_KEYS)
        if observation["schema_version"] != V2_OBSERVATION_SCHEMA_VERSION:
            raise AntiTorpedoFeatureError("only the physical v2 observation is admitted")
        launched = _boolean("decoy_launched", observation["decoy_launched"])
        raw_mask = _sequence("action_mask", root["action_mask"], self.action_count)
        mask = tuple(_boolean(f"action_mask[{index}]", item) for index, item in enumerate(raw_mask))
        expected_mask = (True, True, True, not launched, not launched, not launched)
        if mask != expected_mask or not any(mask):
            raise AntiTorpedoFeatureError("action_mask differs from the physical v2 state")

        ordinal = _integer("scenario_ordinal", observation["scenario_ordinal"], lower=0, upper=255)
        bank = _SCENARIO_BANK
        config = bank.configs[ordinal]
        bits = _sequence("factor_vector", observation["factor_vector"], 8)
        factor_vector = tuple(
            _integer(f"factor_vector[{i}]", bit, lower=0, upper=1)
            for i, bit in enumerate(bits)
        )
        if factor_vector != config.level_bits:
            raise AntiTorpedoFeatureError("factor_vector differs from config ordinal")
        if observation["config_sha256"] != config.config_sha256:
            raise AntiTorpedoFeatureError("config_sha256 differs from config ordinal")
        if observation["scenario_family_sha256"] != bank.family_sha256:
            raise AntiTorpedoFeatureError("scenario_family_sha256 differs")
        scenario = _sequence("scenario", observation["scenario"], 5)
        if scenario[0] != config.config_id or scenario[1] != config.config_sha256:
            raise AntiTorpedoFeatureError("scenario identity differs from config ordinal")
        return observation, mask

    def action_mask(self, value: Mapping[str, object]) -> tuple[bool, ...]:
        """Validate provenance and return the contemporaneous state-derived mask."""

        _observation, mask = self._validated(value)
        self.encode(value)
        return mask

    def encode(self, value: Mapping[str, object]) -> tuple[float, ...]:
        """Return the exact 80-element little-endian float32 feature vector."""

        observation, _mask = self._validated(value)
        features: list[float] = []
        tick = _integer("tick", observation["tick"], lower=0, upper=30)
        features.append(tick / 30.0)

        def motion(name: str) -> tuple[float, ...]:
            row = _sequence(name, observation[name], 6)
            x = _number(f"{name}.x", row[0], -512.0, 512.0)
            y = _number(f"{name}.y", row[1], -512.0, 512.0)
            z = _number(f"{name}.z", row[2], -64.0, 64.0)
            heading = _number(f"{name}.heading", row[3], 0.0, 360.0)
            if heading == 360.0:
                raise AntiTorpedoFeatureError(f"{name}.heading must be below 360")
            xy_speed = _number(f"{name}.xy_speed", row[4], -16.0, 16.0)
            z_speed = _number(f"{name}.z_speed", row[5], -16.0, 16.0)
            radians = math.radians(heading)
            return (
                x / 512.0,
                y / 512.0,
                z / 64.0,
                math.sin(radians),
                math.cos(radians),
                xy_speed / 16.0,
                z_speed / 16.0,
            )

        features.extend(motion("ship"))
        features.extend(motion("torpedo"))
        ship_row = _sequence("ship", observation["ship"], 6)
        torpedo_row = _sequence("torpedo", observation["torpedo"], 6)
        scenario = _sequence("scenario", observation["scenario"], 5)
        scenario_speeds = (
            _number("scenario.ship_xy_speed", scenario[2], -16.0, 16.0),
            _number("scenario.torpedo_xy_speed", scenario[3], -16.0, 16.0),
            _number("scenario.torpedo_z_speed", scenario[4], -16.0, 16.0),
        )
        observed_speeds = (
            _number("ship.xy_speed", ship_row[4], -16.0, 16.0),
            _number("torpedo.xy_speed", torpedo_row[4], -16.0, 16.0),
            _number("torpedo.z_speed", torpedo_row[5], -16.0, 16.0),
        )
        if scenario_speeds != observed_speeds:
            raise AntiTorpedoFeatureError(
                "scenario speed duplicates differ from physical observation"
            )
        relative = _sequence("relative", observation["relative"], 5)
        features.extend(
            (
                _number("relative.x", relative[0], -512.0, 512.0) / 512.0,
                _number("relative.y", relative[1], -512.0, 512.0) / 512.0,
                _number("relative.z", relative[2], -64.0, 64.0) / 64.0,
                _number("relative.range", relative[3], 0.0, 1024.0) / 1024.0,
                _number("relative.closing_rate", relative[4], -16.0, 16.0) / 16.0,
            )
        )
        pending = _sequence("pending_target", observation["pending_target"], 4)
        pending_present = _boolean("pending_target.present", pending[0])
        pending_xyz = (
            _number("pending_target.x", pending[1], -512.0, 512.0),
            _number("pending_target.y", pending[2], -512.0, 512.0),
            _number("pending_target.z", pending[3], -64.0, 64.0),
        )
        if not pending_present and pending_xyz != (0.0, 0.0, 0.0):
            raise AntiTorpedoFeatureError("absent pending_target must use zero padding")
        features.extend(
            (
                float(pending_present),
                pending_xyz[0] / 512.0,
                pending_xyz[1] / 512.0,
                pending_xyz[2] / 64.0,
            )
        )
        launched = _boolean("decoy_launched", observation["decoy_launched"])
        features.append(float(launched))

        decoys = _sequence("decoys", observation["decoys"], 4)
        factor_bits = _sequence("factor_vector", observation["factor_vector"], 8)
        expected_decoy_type = "self_propelled" if factor_bits[6] == 1 else "stationary"
        present_slots: list[bool] = []
        for index, raw in enumerate(decoys):
            row = _sequence(f"decoys[{index}]", raw, 10)
            present = _boolean(f"decoys[{index}].present", row[0])
            present_slots.append(present)
            if not present:
                if row != _PADDING:
                    raise AntiTorpedoFeatureError("absent decoy must equal the padding sentinel")
                features.extend((0.0,) * 10)
                continue
            sense_id = _text(f"decoys[{index}].sense_id", row[1])
            if sense_id != f"blue_ship_0::decoy_{index}":
                raise AntiTorpedoFeatureError("decoy sense IDs are not in frozen order")
            decoy_type = row[2]
            if decoy_type not in {"stationary", "self_propelled"}:
                raise AntiTorpedoFeatureError("decoy type is not supported")
            if decoy_type != expected_decoy_type:
                raise AntiTorpedoFeatureError("decoy type differs from factor_vector")
            active = _boolean(f"decoys[{index}].active", row[3])
            x = _number(f"decoys[{index}].x", row[4], -512.0, 512.0)
            y = _number(f"decoys[{index}].y", row[5], -512.0, 512.0)
            z = _number(f"decoys[{index}].z", row[6], -64.0, 64.0)
            lifespan = _number(f"decoys[{index}].lifespan", row[7], -30.0, 10.0)
            flight = _number(f"decoys[{index}].time_of_flight", row[8], -30.0, 30.0)
            propelled = _boolean(f"decoys[{index}].propelled", row[9])
            if decoy_type == "stationary" and propelled:
                raise AntiTorpedoFeatureError("stationary decoy cannot be propelled")
            features.extend(
                (
                    1.0,
                    float(decoy_type == "stationary"),
                    float(decoy_type == "self_propelled"),
                    float(active),
                    x / 512.0,
                    y / 512.0,
                    z / 64.0,
                    lifespan / 10.0,
                    flight / 30.0,
                    float(propelled),
                )
            )

        if present_slots != ([True] * 4 if launched else [False] * 4):
            raise AntiTorpedoFeatureError("decoy presence differs from launch state")

        factor_vector = factor_bits
        features.extend(
            float(_integer(f"factor_vector[{i}]", bit, lower=0, upper=1))
            for i, bit in enumerate(factor_vector)
        )
        previous = _integer(
            "previous_action", observation["previous_action"], lower=-1, upper=5
        )
        features.extend(float(previous == action) for action in range(-1, 6))
        if len(features) != self.feature_size:
            raise AntiTorpedoFeatureError("internal feature size differs from contract")
        encoded = tuple(_f32(item) for item in features)
        packed = struct.pack(f"<{self.feature_size}f", *encoded)
        if struct.unpack(f"<{self.feature_size}f", packed) != encoded:
            raise AntiTorpedoFeatureError("feature vector failed float32 round-trip")
        return encoded


__all__ = [
    "ANTI_TORPEDO_ACTION_COUNT",
    "ANTI_TORPEDO_ACTION_SCHEMA_SHA256",
    "ANTI_TORPEDO_FEATURE_CONTRACT_SHA256",
    "ANTI_TORPEDO_FEATURE_SCHEMA_VERSION",
    "ANTI_TORPEDO_FEATURE_SIZE",
    "AntiTorpedoFeatureError",
    "AntiTorpedoV2FeatureContract",
    "LOADED_ANTI_TORPEDO_FEATURE_SOURCE_SHA256",
]
