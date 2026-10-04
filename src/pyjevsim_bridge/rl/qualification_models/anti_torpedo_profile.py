"""Dependency-free, source-locked anti-torpedo scenario profile.

The module freezes the TASK-RL-104 scenario identity without importing
PyJevSim.  Eight binary factors form an exhaustive 256-row design.  Explicit
configuration identity is separate from episode, master, and worker seeds.
The registered GF(2) equations assign the exact 32/160/64 partitions while
preserving factor and pairwise balance within every fold.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, TypeAlias, cast

PROFILE_SCHEMA_VERSION: Final = "anti-torpedo-effective-profile-v1"
SCENARIO_SCHEMA_VERSION: Final = "anti-torpedo-effective-scenario-v1"
SCENARIO_FAMILY_ID: Final = "anti-torpedo-effective-8bit-factorial-v1"
CONFIG_ASSIGNMENT_ID: Final = "explicit-ordinal-digest-not-seed-selected-v1"
GF2_FOLD_ID: Final = "gf2-registered-three-equation-fold-v1"
SEED_PROFILE_ID: Final = "domain-separated-episode-master-worker-seeds-v1"
FACTOR_COUNT: Final = 8
CONFIG_COUNT: Final = 1 << FACTOR_COUNT
TUNING_CONFIG_COUNT: Final = 32
MEASURED_CONFIG_COUNT: Final = 160
EVALUATION_CONFIG_COUNT: Final = 64
MEASURED_COHORT_COUNT: Final = 20
MEASURED_COHORT_SIZE: Final = 8

JsonScalar: TypeAlias = str | int | float | bool | None


class ScenarioProfileError(ValueError):
    """The scenario profile is not the closed TASK-RL-104 design."""


class VariationClass(StrEnum):
    INITIAL_STATE = "initial-state"
    SENSOR = "sensor"
    DYNAMICS = "dynamics"


class SensitivityStage(StrEnum):
    INITIAL_OBSERVATION = "initial-observation"
    RUNTIME_TRACE = "runtime-trace"
    POST_DEPLOY_TRACE = "post-deploy-trace"


class SeedRole(StrEnum):
    QUALIFICATION_EPISODE = "qualification-episode"
    TUNING_MASTER = "tuning-master"
    MEASURED_MASTER = "measured-master"
    WORKER = "worker"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScenarioProfileError(
            f"profile value is not canonical JSON compatible: {exc}"
        ) from exc


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScenarioProfileError(f"{name} must be a non-empty string")
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScenarioProfileError(f"{name} must be a non-negative integer")
    return value


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScenarioProfileError(
            f"{name} must be a lowercase SHA-256 hex digest"
        )
    return value


def _require_level(name: str, value: JsonScalar) -> JsonScalar:
    if isinstance(value, float) and not math.isfinite(value):
        raise ScenarioProfileError(f"{name} must be finite")
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise ScenarioProfileError(f"{name} must be a JSON scalar")
    return value


@dataclass(frozen=True, slots=True)
class ScenarioFactorV1:
    factor_id: str
    bit_index: int
    variation_class: VariationClass
    levels: tuple[JsonScalar, JsonScalar]
    model_json_pointers: tuple[str, ...]
    observation_json_pointers: tuple[str, ...]
    sensitivity_stage: SensitivityStage
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_non_empty("factor_id", self.factor_id)
        bit_index = _require_non_negative_int("factor bit_index", self.bit_index)
        if bit_index >= FACTOR_COUNT:
            raise ScenarioProfileError("factor bit_index is outside the 8-bit design")
        if len(self.levels) != 2:
            raise ScenarioProfileError("factor requires exactly two levels")
        levels = tuple(
            _require_level(f"factor level[{index}]", level)
            for index, level in enumerate(self.levels)
        )
        if levels[0] == levels[1]:
            raise ScenarioProfileError("factor levels must differ")
        object.__setattr__(self, "levels", cast(tuple[JsonScalar, JsonScalar], levels))
        for name in ("model_json_pointers", "observation_json_pointers"):
            pointers = tuple(getattr(self, name))
            if not pointers or len(pointers) != len(set(pointers)):
                raise ScenarioProfileError(f"{name} must be non-empty and unique")
            if any(not pointer.startswith("/") for pointer in pointers):
                raise ScenarioProfileError(f"{name} must contain JSON pointers")
            object.__setattr__(self, name, pointers)
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "bit_index": self.bit_index,
            "factor_id": self.factor_id,
            "levels": list(self.levels),
            "model_json_pointers": list(self.model_json_pointers),
            "observation_json_pointers": list(self.observation_json_pointers),
            "sensitivity_stage": self.sensitivity_stage.value,
            "variation_class": self.variation_class.value,
        }


FACTOR_SPECS: Final = (
    ScenarioFactorV1(
        "approach-side",
        0,
        VariationClass.INITIAL_STATE,
        (-1, 1),
        (
            "/Torpedo/0/ManueverObject/x",
            "/Torpedo/0/ManueverObject/heading",
        ),
        (
            "/torpedo/0",
            "/torpedo/3",
            "/relative/0",
            "/scenario_profile/2",
        ),
        SensitivityStage.INITIAL_OBSERVATION,
    ),
    ScenarioFactorV1(
        "horizontal-standoff",
        1,
        VariationClass.INITIAL_STATE,
        (18.0, 24.0),
        (
            "/Torpedo/0/ManueverObject/x",
            "/Torpedo/0/ManueverObject/y",
        ),
        ("/torpedo/0", "/torpedo/1", "/relative", "/scenario_profile/3"),
        SensitivityStage.INITIAL_OBSERVATION,
    ),
    ScenarioFactorV1(
        "torpedo-depth",
        2,
        VariationClass.INITIAL_STATE,
        (-8.0, -12.0),
        ("/Torpedo/0/ManueverObject/z",),
        ("/torpedo/2", "/relative/2", "/scenario_profile/4"),
        SensitivityStage.INITIAL_OBSERVATION,
    ),
    ScenarioFactorV1(
        "bearing-offset",
        3,
        VariationClass.INITIAL_STATE,
        (-22.5, 22.5),
        ("/Torpedo/0/ManueverObject/heading",),
        ("/torpedo/3", "/scenario_profile/5"),
        SensitivityStage.INITIAL_OBSERVATION,
    ),
    ScenarioFactorV1(
        "torpedo-xy-speed",
        4,
        VariationClass.DYNAMICS,
        (4.0, 6.0),
        ("/Torpedo/0/ManueverObject/xy_speed",),
        ("/torpedo/4", "/scenario_profile/6"),
        SensitivityStage.INITIAL_OBSERVATION,
    ),
    ScenarioFactorV1(
        "detector-range",
        5,
        VariationClass.SENSOR,
        (20.0, 50.0),
        ("/Torpedo/0/DetectorObject/detection_range",),
        ("/scenario_profile/7", "/pending_target"),
        SensitivityStage.RUNTIME_TRACE,
    ),
    ScenarioFactorV1(
        "decoy-family",
        6,
        VariationClass.DYNAMICS,
        ("stationary", "self_propelled"),
        ("/SurfaceShip/0/LauncherObject/DecoyObjects",),
        ("/scenario_profile/8", "/decoys"),
        SensitivityStage.POST_DEPLOY_TRACE,
    ),
    ScenarioFactorV1(
        "decoy-launch-speed",
        7,
        VariationClass.DYNAMICS,
        (8.0, 12.0),
        ("/SurfaceShip/0/LauncherObject/DecoyObjects",),
        ("/scenario_profile/9", "/decoys"),
        SensitivityStage.POST_DEPLOY_TRACE,
    ),
)


def _level_bits(ordinal: int) -> tuple[int, ...]:
    return tuple((ordinal >> bit_index) & 1 for bit_index in range(FACTOR_COUNT))


def _factor_values(level_bits: Sequence[int]) -> dict[str, JsonScalar]:
    return {
        factor.factor_id: factor.levels[level_bits[factor.bit_index]]
        for factor in FACTOR_SPECS
    }


def _decoys(decoy_family: str, launch_speed: float) -> list[dict[str, object]]:
    if decoy_family == "stationary":
        return [
            {
                "azimuth": azimuth,
                "elevation": 45.0,
                "lifespan": 10.0,
                "speed": launch_speed,
                "type": "stationary",
            }
            for azimuth in (45.0, 135.0, 225.0, 315.0)
        ]
    if decoy_family == "self_propelled":
        headings = (270.0, 180.0, 225.0, 315.0)
        return [
            {
                "azimuth": azimuth,
                "elevation": 45.0,
                "heading": headings[index],
                "lifespan": 10.0,
                "speed": launch_speed,
                "type": "self_propelled",
                "xy_speed": 2.0,
            }
            for index, azimuth in enumerate((45.0, 135.0, 225.0, 315.0))
        ]
    raise ScenarioProfileError("unknown decoy family")


def _materialize_payload(ordinal: int) -> dict[str, object]:
    bits = _level_bits(ordinal)
    values = _factor_values(bits)
    decoy_family = cast(str, values["decoy-family"])
    standoff = cast(float, values["horizontal-standoff"])
    approach_side = cast(int, values["approach-side"])
    torpedo_x = approach_side * standoff
    torpedo_y = standoff
    direct_bearing = math.degrees(math.atan2(-torpedo_x, -torpedo_y)) % 360.0
    heading = (
        direct_bearing + cast(float, values["bearing-offset"])
    ) % 360.0
    launch_speed = cast(float, values["decoy-launch-speed"])
    return {
        "SurfaceShip": [
            {
                "LauncherObject": {
                    "DecoyObjects": _decoys(decoy_family, launch_speed)
                },
                "ManueverObject": {
                    "heading": 0.0,
                    "x": 0.0,
                    "xy_speed": 3.0,
                    "y": 0.0,
                    "z": 0.0,
                    "z_speed": 0.0,
                },
            }
        ],
        "Torpedo": [
            {
                "DetectorObject": {
                    "detection_range": values["detector-range"]
                },
                "ManueverObject": {
                    "heading": heading,
                    "x": torpedo_x,
                    "xy_speed": values["torpedo-xy-speed"],
                    "y": torpedo_y,
                    "z": values["torpedo-depth"],
                    "z_speed": 1.0,
                },
                "TorpedoControlObject": {"range": 1.0},
            }
        ],
        "factor_levels": values,
        "level_bits": list(bits),
        "scenario_id": f"effective-{ordinal:03d}-{decoy_family}",
        "schema_version": SCENARIO_SCHEMA_VERSION,
    }


def _model_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "SurfaceShip": payload["SurfaceShip"],
        "Torpedo": payload["Torpedo"],
    }


def _walk_finite(value: object) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ScenarioProfileError("materialized config contains a non-finite value")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _walk_finite(item)
        return
    if isinstance(value, list):
        for item in value:
            _walk_finite(item)
        return
    raise ScenarioProfileError("materialized config contains a non-JSON value")


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioProfileError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ScenarioProfileError(f"{name} must be finite")
    return result


def validate_materialized_config(value: Mapping[str, object]) -> int:
    """Validate the closed config before any executor or model side effect."""

    expected_top = {
        "SurfaceShip",
        "Torpedo",
        "factor_levels",
        "level_bits",
        "scenario_id",
        "schema_version",
    }
    if set(value) != expected_top:
        raise ScenarioProfileError("materialized config top-level keys differ")
    raw_bits = value["level_bits"]
    if not isinstance(raw_bits, list) or len(raw_bits) != FACTOR_COUNT or any(
        isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1)
        for bit in raw_bits
    ):
        raise ScenarioProfileError("materialized config requires eight binary bits")
    bits = cast(list[int], raw_bits)
    ordinal = sum(bit << bit_index for bit_index, bit in enumerate(bits))
    expected = _materialize_payload(ordinal)
    if _canonical_json(value) != _canonical_json(expected):
        raise ScenarioProfileError("materialized config differs from its closed ordinal")
    _walk_finite(value)
    ship = cast(list[dict[str, object]], value["SurfaceShip"])[0]
    torpedo = cast(list[dict[str, object]], value["Torpedo"])[0]
    ship_motion = cast(dict[str, object], ship["ManueverObject"])
    torpedo_motion = cast(dict[str, object], torpedo["ManueverObject"])
    ship_position = tuple(
        _finite_number(f"ship {axis}", ship_motion[axis])
        for axis in ("x", "y", "z")
    )
    torpedo_position = tuple(
        _finite_number(f"torpedo {axis}", torpedo_motion[axis])
        for axis in ("x", "y", "z")
    )
    separation = math.sqrt(
        sum(
            (torpedo_position[index] - ship_position[index]) ** 2
            for index in range(3)
        )
    )
    if separation <= 1.0:
        raise ScenarioProfileError("initial ship/torpedo separation is not safe")
    heading = _finite_number("torpedo heading", torpedo_motion["heading"])
    if not 0.0 <= heading < 360.0:
        raise ScenarioProfileError("torpedo heading is not normalized")
    if _finite_number("torpedo depth", torpedo_motion["z"]) >= 0.0:
        raise ScenarioProfileError("torpedo depth must be below the surface")
    if any(
        _finite_number(f"torpedo {name}", torpedo_motion[name]) <= 0.0
        for name in ("xy_speed", "z_speed")
    ):
        raise ScenarioProfileError("torpedo speeds must be positive")
    detector = cast(dict[str, object], torpedo["DetectorObject"])
    if _finite_number("detector range", detector["detection_range"]) <= 0.0:
        raise ScenarioProfileError("detector range must be positive")
    launcher = cast(dict[str, object], ship["LauncherObject"])
    decoys = cast(list[dict[str, object]], launcher["DecoyObjects"])
    if not 1 <= len(decoys) <= 4:
        raise ScenarioProfileError("materialized config requires one through four decoys")
    if any(
        _finite_number("decoy speed", decoy["speed"]) <= 0.0
        or _finite_number("decoy lifespan", decoy["lifespan"]) <= 0.0
        for decoy in decoys
    ):
        raise ScenarioProfileError("decoy speed and lifespan must be positive")
    return ordinal


@dataclass(frozen=True, slots=True)
class ScenarioConfigV1:
    ordinal: int
    config_id: str = field(init=False)
    level_bits: tuple[int, ...] = field(init=False)
    canonical_json: str = field(init=False, repr=False)
    config_sha256: str = field(init=False)
    model_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        ordinal = _require_non_negative_int("config ordinal", self.ordinal)
        if ordinal >= CONFIG_COUNT:
            raise ScenarioProfileError("config ordinal is outside the 8-bit bank")
        payload = _materialize_payload(ordinal)
        validate_materialized_config(payload)
        canonical = _canonical_json(payload)
        object.__setattr__(self, "config_id", cast(str, payload["scenario_id"]))
        object.__setattr__(self, "level_bits", _level_bits(ordinal))
        object.__setattr__(self, "canonical_json", canonical.decode("utf-8"))
        object.__setattr__(
            self, "config_sha256", hashlib.sha256(canonical).hexdigest()
        )
        object.__setattr__(
            self, "model_sha256", _content_sha256(_model_payload(payload))
        )

    def materialize(self) -> dict[str, object]:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):
            raise ScenarioProfileError("canonical scenario is not an object")
        return cast(dict[str, object], value)

    def content(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "config_sha256": self.config_sha256,
            "level_bits": list(self.level_bits),
            "model_sha256": self.model_sha256,
            "ordinal": self.ordinal,
        }


def profile_generator_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ScenarioBankV1:
    generator_source_sha256: str
    factors: tuple[ScenarioFactorV1, ...]
    configs: tuple[ScenarioConfigV1, ...]
    schema_version: str = PROFILE_SCHEMA_VERSION
    factor_schema_sha256: str = field(init=False)
    scenario_source_sha256: str = field(init=False)
    family_sha256: str = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ScenarioProfileError("scenario bank schema_version differs")
        _require_sha256("generator_source_sha256", self.generator_source_sha256)
        factors = tuple(self.factors)
        if factors != FACTOR_SPECS:
            raise ScenarioProfileError("scenario bank factor schema is not closed")
        configs = tuple(self.configs)
        if len(configs) != CONFIG_COUNT or tuple(
            config.ordinal for config in configs
        ) != tuple(range(CONFIG_COUNT)):
            raise ScenarioProfileError("scenario bank must contain ordinals 0..255")
        if len({config.config_sha256 for config in configs}) != CONFIG_COUNT:
            raise ScenarioProfileError("scenario config digests must be unique")
        if len({config.model_sha256 for config in configs}) != CONFIG_COUNT:
            raise ScenarioProfileError("scenario model digests must be unique")
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "configs", configs)
        factor_digest = _content_sha256([factor.content() for factor in factors])
        source_digest = _content_sha256(
            [config.model_sha256 for config in configs]
        )
        family_digest = _content_sha256(
            {
                "factor_schema_sha256": factor_digest,
                "family_id": SCENARIO_FAMILY_ID,
                "generator_source_sha256": self.generator_source_sha256,
                "scenario_source_sha256": source_digest,
                "config_assignment_id": CONFIG_ASSIGNMENT_ID,
            }
        )
        object.__setattr__(self, "factor_schema_sha256", factor_digest)
        object.__setattr__(self, "scenario_source_sha256", source_digest)
        object.__setattr__(self, "family_sha256", family_digest)
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "configs": [config.content() for config in self.configs],
            "factor_schema_sha256": self.factor_schema_sha256,
            "factors": [factor.content() for factor in self.factors],
            "family_id": SCENARIO_FAMILY_ID,
            "family_sha256": self.family_sha256,
            "generator_source_sha256": self.generator_source_sha256,
            "scenario_source_sha256": self.scenario_source_sha256,
            "schema_version": self.schema_version,
            "config_assignment_id": CONFIG_ASSIGNMENT_ID,
        }


@dataclass(frozen=True, slots=True)
class SeedIdentityV1:
    role: SeedRole
    ordinal: int
    seed: int
    derivation_digest: str
    namespace: str = SEED_PROFILE_ID
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_non_negative_int("seed ordinal", self.ordinal)
        _require_non_negative_int("seed", self.seed)
        _require_sha256("seed derivation_digest", self.derivation_digest)
        if self.namespace != SEED_PROFILE_ID:
            raise ScenarioProfileError("seed namespace differs")
        expected = hashlib.sha256(
            f"{SEED_PROFILE_ID}:{self.role.value}:{self.ordinal}".encode("ascii")
        ).hexdigest()
        if self.derivation_digest != expected:
            raise ScenarioProfileError("seed derivation digest differs")
        if self.seed != int.from_bytes(bytes.fromhex(expected)[:8], "big"):
            raise ScenarioProfileError("seed value differs")
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "derivation_digest": self.derivation_digest,
            "namespace": self.namespace,
            "ordinal": self.ordinal,
            "role": self.role.value,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class SeedIdentityProfileV1:
    qualification_episodes: tuple[SeedIdentityV1, ...]
    tuning_masters: tuple[SeedIdentityV1, ...]
    measured_masters: tuple[SeedIdentityV1, ...]
    workers: tuple[SeedIdentityV1, ...]
    profile_id: str = SEED_PROFILE_ID
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.profile_id != SEED_PROFILE_ID:
            raise ScenarioProfileError("seed profile_id differs")
        if len(self.qualification_episodes) != CONFIG_COUNT or any(
            item.role is not SeedRole.QUALIFICATION_EPISODE
            for item in self.qualification_episodes
        ):
            raise ScenarioProfileError(
                "seed profile requires 256 qualification episode seeds"
            )
        if len(self.tuning_masters) != 5 or any(
            item.role is not SeedRole.TUNING_MASTER
            for item in self.tuning_masters
        ):
            raise ScenarioProfileError("seed profile requires five tuning masters")
        if len(self.measured_masters) != 20 or any(
            item.role is not SeedRole.MEASURED_MASTER
            for item in self.measured_masters
        ):
            raise ScenarioProfileError("seed profile requires twenty measured masters")
        if len(self.workers) != 4 or any(
            item.role is not SeedRole.WORKER for item in self.workers
        ):
            raise ScenarioProfileError("seed profile requires four worker seeds")
        all_values = [
            *(item.seed for item in self.qualification_episodes),
            *(item.seed for item in self.tuning_masters),
            *(item.seed for item in self.measured_masters),
            *(item.seed for item in self.workers),
        ]
        if len(all_values) != len(set(all_values)):
            raise ScenarioProfileError("seed roles overlap")
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "measured_masters": [item.content() for item in self.measured_masters],
            "profile_id": self.profile_id,
            "qualification_episodes": [
                item.content() for item in self.qualification_episodes
            ],
            "tuning_masters": [item.content() for item in self.tuning_masters],
            "workers": [item.content() for item in self.workers],
        }


def gf2_fold(ordinal: int) -> int:
    value = _require_non_negative_int("GF(2) ordinal", ordinal)
    if value >= CONFIG_COUNT:
        raise ScenarioProfileError("GF(2) ordinal is outside the 8-bit space")
    bits = _level_bits(value)
    f0 = bits[0] ^ bits[3] ^ bits[4] ^ bits[6] ^ bits[7]
    f1 = bits[1] ^ bits[3] ^ bits[5] ^ bits[6]
    f2 = bits[2] ^ bits[4] ^ bits[5] ^ bits[6]
    return f0 + 2 * f1 + 4 * f2


def gf2_cohort(ordinal: int) -> int:
    """Split each 32-row fold into four factor-balanced 8-row cohorts."""

    value = _require_non_negative_int("GF(2) ordinal", ordinal)
    if value >= CONFIG_COUNT:
        raise ScenarioProfileError("GF(2) ordinal is outside the 8-bit space")
    bits = _level_bits(value)
    return (bits[0] ^ bits[1]) + 2 * (bits[0] ^ bits[2])


def _balanced(configs: Sequence[ScenarioConfigV1]) -> bool:
    if not configs or len(configs) % 2:
        return False
    half = len(configs) // 2
    return all(
        sum(config.level_bits[bit_index] for config in configs) == half
        for bit_index in range(FACTOR_COUNT)
    )


def _pairwise_balanced(configs: Sequence[ScenarioConfigV1]) -> bool:
    if len(configs) % 4:
        return False
    exact = len(configs) // 4
    for left_bit in range(FACTOR_COUNT):
        for right_bit in range(left_bit + 1, FACTOR_COUNT):
            counts = [0, 0, 0, 0]
            for config in configs:
                level = config.level_bits[left_bit] + 2 * config.level_bits[right_bit]
                counts[level] += 1
            if counts != [exact, exact, exact, exact]:
                return False
    return True


@dataclass(frozen=True, slots=True)
class ScenarioPartitionV1:
    family_sha256: str
    bank_sha256: str
    tuning: tuple[ScenarioConfigV1, ...]
    measured: tuple[ScenarioConfigV1, ...]
    evaluation: tuple[ScenarioConfigV1, ...]
    measured_cohorts: tuple[tuple[ScenarioConfigV1, ...], ...]
    fold_id: str = GF2_FOLD_ID
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256("partition family_sha256", self.family_sha256)
        _require_sha256("partition bank_sha256", self.bank_sha256)
        if self.fold_id != GF2_FOLD_ID:
            raise ScenarioProfileError("partition GF(2) fold differs")
        groups = (
            ("tuning", tuple(self.tuning), TUNING_CONFIG_COUNT),
            ("measured", tuple(self.measured), MEASURED_CONFIG_COUNT),
            ("evaluation", tuple(self.evaluation), EVALUATION_CONFIG_COUNT),
        )
        all_ordinals: list[int] = []
        for name, values, exact in groups:
            if len(values) != exact:
                raise ScenarioProfileError(f"partition {name} requires {exact} configs")
            if not _balanced(values):
                raise ScenarioProfileError(f"partition {name} is not factor-balanced")
            all_ordinals.extend(item.ordinal for item in values)
            object.__setattr__(self, name, values)
        if set(all_ordinals) != set(range(CONFIG_COUNT)) or len(all_ordinals) != len(
            set(all_ordinals)
        ):
            raise ScenarioProfileError("partitions must be disjoint and exhaustive")
        all_configs = (*self.tuning, *self.measured, *self.evaluation)
        cohorts = tuple(tuple(cohort) for cohort in self.measured_cohorts)
        if len(cohorts) != MEASURED_COHORT_COUNT or any(
            len(cohort) != MEASURED_COHORT_SIZE or not _balanced(cohort)
            for cohort in cohorts
        ):
            raise ScenarioProfileError("measured cohorts must be twenty balanced rows")
        cohort_ordinals = [
            item.ordinal for cohort in cohorts for item in cohort
        ]
        if set(cohort_ordinals) != {
            item.ordinal for item in self.measured
        } or len(cohort_ordinals) != len(set(cohort_ordinals)):
            raise ScenarioProfileError("measured cohorts do not partition measured configs")
        folds = tuple(
            tuple(config for config in all_configs if gf2_fold(config.ordinal) == fold)
            for fold in range(8)
        )
        if any(
            len(values) != 32
            or not _balanced(values)
            or not _pairwise_balanced(values)
            for values in folds
        ):
            raise ScenarioProfileError(
                "every GF(2) fold requires 16/16 factors and 8/8/8/8 pairs"
            )
        if {gf2_fold(item.ordinal) for item in self.tuning} != {0}:
            raise ScenarioProfileError("tuning partition must be fold 0")
        if {gf2_fold(item.ordinal) for item in self.measured} != set(range(1, 6)):
            raise ScenarioProfileError("measured partition must be folds 1..5")
        if {gf2_fold(item.ordinal) for item in self.evaluation} != {6, 7}:
            raise ScenarioProfileError("evaluation partition must be folds 6..7")
        object.__setattr__(self, "measured_cohorts", cohorts)
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        def digests(values: Sequence[ScenarioConfigV1]) -> list[str]:
            return [item.config_sha256 for item in values]

        return {
            "evaluation": digests(self.evaluation),
            "family_sha256": self.family_sha256,
            "fold_id": self.fold_id,
            "measured": digests(self.measured),
            "measured_cohorts": [digests(cohort) for cohort in self.measured_cohorts],
            "bank_sha256": self.bank_sha256,
            "tuning": digests(self.tuning),
        }


def _diff_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: set[str] = set()
        keys = set(left).union(right)
        for key in keys:
            pointer = f"{prefix}/{key}"
            if key not in left or key not in right:
                result.add(pointer)
            else:
                result.update(_diff_paths(left[key], right[key], pointer))
        return result
    if isinstance(left, list) and isinstance(right, list):
        result = set()
        for index in range(max(len(left), len(right))):
            pointer = f"{prefix}/{index}"
            if index >= len(left) or index >= len(right):
                result.add(pointer)
            else:
                result.update(_diff_paths(left[index], right[index], pointer))
        return result
    return set() if left == right else {prefix or "/"}


def validate_declared_factor_sensitivity(bank: ScenarioBankV1) -> None:
    """Prove every one-bit model change stays within its declared subtree."""

    for factor in bank.factors:
        mask = 1 << factor.bit_index
        for ordinal in range(CONFIG_COUNT):
            if ordinal & mask:
                continue
            left = _model_payload(bank.configs[ordinal].materialize())
            right = _model_payload(bank.configs[ordinal ^ mask].materialize())
            changes = _diff_paths(left, right)
            if not changes:
                raise ScenarioProfileError(
                    f"factor {factor.factor_id} has no material model sensitivity"
                )
            if any(
                not any(
                    path == declared or path.startswith(f"{declared}/")
                    for declared in factor.model_json_pointers
                )
                for path in changes
            ):
                raise ScenarioProfileError(
                    f"factor {factor.factor_id} changes an undeclared model path"
                )
            if any(
                not any(
                    path == declared or path.startswith(f"{declared}/")
                    for path in changes
                )
                for declared in factor.model_json_pointers
            ):
                raise ScenarioProfileError(
                    f"factor {factor.factor_id} lacks a declared path witness"
                )


@dataclass(frozen=True, slots=True)
class AntiTorpedoScenarioProfileV1:
    bank: ScenarioBankV1
    seeds: SeedIdentityProfileV1
    partition: ScenarioPartitionV1
    schema_version: str = PROFILE_SCHEMA_VERSION
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ScenarioProfileError("profile schema_version differs")
        if self.partition.family_sha256 != self.bank.family_sha256:
            raise ScenarioProfileError("partition family differs from bank")
        if self.partition.bank_sha256 != self.bank.sha256:
            raise ScenarioProfileError("partition bank differs")
        validate_declared_factor_sensitivity(self.bank)
        object.__setattr__(self, "sha256", _content_sha256(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "bank_sha256": self.bank.sha256,
            "family_sha256": self.bank.family_sha256,
            "generator_source_sha256": self.bank.generator_source_sha256,
            "partition_sha256": self.partition.sha256,
            "scenario_source_sha256": self.bank.scenario_source_sha256,
            "schema_version": self.schema_version,
            "seed_profile_sha256": self.seeds.sha256,
        }


def build_scenario_bank() -> ScenarioBankV1:
    current = profile_generator_source_sha256()
    if current != LOADED_PROFILE_GENERATOR_SOURCE_SHA256:
        raise ScenarioProfileError("profile generator source changed after import")
    return ScenarioBankV1(
        generator_source_sha256=current,
        factors=FACTOR_SPECS,
        configs=tuple(ScenarioConfigV1(ordinal) for ordinal in range(CONFIG_COUNT)),
    )


def build_scenario_bank_v1() -> ScenarioBankV1:
    """Stable TASK-RL-104 name for the closed v1 bank builder."""

    return build_scenario_bank()


def materialize_config(
    bank: ScenarioBankV1, ordinal: int
) -> dict[str, object]:
    """Return a fresh, validated config selected explicitly by ordinal."""

    if not isinstance(bank, ScenarioBankV1):
        raise TypeError("bank must be ScenarioBankV1")
    value = _require_non_negative_int("config ordinal", ordinal)
    if value >= CONFIG_COUNT:
        raise ScenarioProfileError("config ordinal is outside the 8-bit bank")
    materialized = bank.configs[value].materialize()
    validate_materialized_config(materialized)
    return materialized


def _seed_identity(role: SeedRole, ordinal: int) -> SeedIdentityV1:
    digest = hashlib.sha256(
        f"{SEED_PROFILE_ID}:{role.value}:{ordinal}".encode("ascii")
    ).hexdigest()
    return SeedIdentityV1(
        role=role,
        ordinal=ordinal,
        seed=int.from_bytes(bytes.fromhex(digest)[:8], "big"),
        derivation_digest=digest,
    )


def build_seed_identity_profile() -> SeedIdentityProfileV1:
    return SeedIdentityProfileV1(
        qualification_episodes=tuple(
            _seed_identity(SeedRole.QUALIFICATION_EPISODE, ordinal)
            for ordinal in range(CONFIG_COUNT)
        ),
        tuning_masters=tuple(
            _seed_identity(SeedRole.TUNING_MASTER, ordinal)
            for ordinal in range(5)
        ),
        measured_masters=tuple(
            _seed_identity(SeedRole.MEASURED_MASTER, ordinal)
            for ordinal in range(20)
        ),
        workers=tuple(
            _seed_identity(SeedRole.WORKER, ordinal) for ordinal in range(4)
        ),
    )


def build_scenario_partition(bank: ScenarioBankV1) -> ScenarioPartitionV1:
    folds = tuple(
        tuple(config for config in bank.configs if gf2_fold(config.ordinal) == fold)
        for fold in range(8)
    )
    tuning = folds[0]
    measured = tuple(config for fold in folds[1:6] for config in fold)
    evaluation = tuple(config for fold in folds[6:8] for config in fold)
    measured_cohorts = tuple(
        tuple(
            config
            for config in folds[fold]
            if gf2_cohort(config.ordinal) == cohort
        )
        for fold in range(1, 6)
        for cohort in range(4)
    )
    return ScenarioPartitionV1(
        family_sha256=bank.family_sha256,
        bank_sha256=bank.sha256,
        tuning=tuning,
        measured=measured,
        evaluation=evaluation,
        measured_cohorts=measured_cohorts,
    )


def build_anti_torpedo_scenario_profile() -> AntiTorpedoScenarioProfileV1:
    bank = build_scenario_bank()
    seeds = build_seed_identity_profile()
    partition = build_scenario_partition(bank)
    return AntiTorpedoScenarioProfileV1(bank, seeds, partition)


LOADED_PROFILE_GENERATOR_SOURCE_SHA256: Final = profile_generator_source_sha256()


__all__ = [
    "CONFIG_COUNT",
    "CONFIG_ASSIGNMENT_ID",
    "EVALUATION_CONFIG_COUNT",
    "FACTOR_COUNT",
    "FACTOR_SPECS",
    "GF2_FOLD_ID",
    "LOADED_PROFILE_GENERATOR_SOURCE_SHA256",
    "MEASURED_COHORT_COUNT",
    "MEASURED_COHORT_SIZE",
    "MEASURED_CONFIG_COUNT",
    "PROFILE_SCHEMA_VERSION",
    "SCENARIO_FAMILY_ID",
    "SCENARIO_SCHEMA_VERSION",
    "SEED_PROFILE_ID",
    "TUNING_CONFIG_COUNT",
    "AntiTorpedoScenarioProfileV1",
    "ScenarioBankV1",
    "ScenarioConfigV1",
    "ScenarioFactorV1",
    "ScenarioPartitionV1",
    "ScenarioProfileError",
    "SeedIdentityV1",
    "SeedIdentityProfileV1",
    "SeedRole",
    "SensitivityStage",
    "VariationClass",
    "build_anti_torpedo_scenario_profile",
    "build_scenario_bank",
    "build_scenario_bank_v1",
    "build_scenario_partition",
    "build_seed_identity_profile",
    "gf2_cohort",
    "gf2_fold",
    "materialize_config",
    "profile_generator_source_sha256",
    "validate_declared_factor_sensitivity",
    "validate_materialized_config",
]
