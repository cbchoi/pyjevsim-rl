"""Nontrivial AT/SIM workload adapter for local RL qualification.

This module deliberately labels the current source combination as a prototype.
The model graph comes from the external PyJevSim AT/SIM example, while the
transport-neutral action, observation, reward, and termination contract lives
here so the same episode can later be hosted by a gorti federation.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from pyjevsim.definition import ExecutionType
from pyjevsim.system_executor import SysExecutor

from pyjevsim_bridge.rl.adapters import FunctionalEpisodeBinding
from pyjevsim_bridge.rl.contracts import EpisodeContext, ExecutorProtocol, StepView
from pyjevsim_bridge.rl.environment import PyJevSimEnv
from pyjevsim_bridge.rl.executor import (
    PINNED_PYJEVSIM_2_1_2_PROFILE,
    ExecutorQualificationPolicy,
    ExecutorSemanticCapability,
)
from pyjevsim_bridge.rl.qualification_models.anti_torpedo_profile import (
    CONFIG_COUNT as V2_CONFIG_COUNT,
)
from pyjevsim_bridge.rl.qualification_models.anti_torpedo_profile import (
    ScenarioBankV1,
    build_scenario_bank_v1,
    materialize_config,
)
from pyjevsim_bridge.rl.scientific import (
    SEMANTIC_PROJECTION_CONTRACT_SHA256,
)

WORKLOAD_VERSION: Final = "AntiTorpedoCountermeasure-v1"
OBSERVATION_SCHEMA_VERSION: Final = "anti-torpedo-observation-v1"
V2_WORKLOAD_VERSION: Final = "AntiTorpedoCountermeasure-v2"
V2_OBSERVATION_SCHEMA_VERSION: Final = "anti-torpedo-observation-v2"
ACTION_PORT: Final = "rl_action"
DECISION_DELTA: Final = 1.0
MAX_STEPS: Final = 30
COLLISION_RADIUS: Final = 1.0
RANGE_DELTA_CLIP: Final = 5.0
TURN_COST: Final = 0.1
LAUNCH_COST: Final = 0.5
SHIP_HIT_REWARD: Final = -100.0
DECOY_CAPTURE_REWARD: Final = 100.0
CLAIM_GRADE: Final = False
AT_SIM_ROOT_ENV: Final = "PYJEVSIM_ATSIM_ROOT"
ACTION_COMMANDS: Final[tuple[tuple[float, bool], ...]] = (
    (0.0, False),
    (-45.0, False),
    (45.0, False),
    (0.0, True),
    (-45.0, True),
    (45.0, True),
)

_MODEL_MODULE_FILES: Final = {
    "project_config": "project_config.py",
    "model.detector": "model/detector.py",
    "model.launcher": "model/launcher.py",
    "model.manuever": "model/manuever.py",
    "model.rl_command_control": "model/rl_command_control.py",
    "model.rl_surfaceship": "model/rl_surfaceship.py",
    "model.self_propelled_decoy": "model/self_propelled_decoy.py",
    "model.stationary_decoy": "model/stationary_decoy.py",
    "model.torpedo": "model/torpedo.py",
    "model.torpedo_controller": "model/torpedo_controller.py",
    "model.tracking_manuever": "model/tracking_manuever.py",
    "mobject.detector_object": "mobject/detector_object.py",
    "mobject.launcher_object": "mobject/launcher_object.py",
    "mobject.manuever_object": "mobject/manuever_object.py",
    "mobject.self_propelled_decoy_object": (
        "mobject/self_propelled_decoy_object.py"
    ),
    "mobject.stationary_decoy_object": "mobject/stationary_decoy_object.py",
    "mobject.torpedo_control_object": "mobject/torpedo_control_object.py",
    "utils.sensing": "utils/sensing.py",
    "utils.sim_context": "utils/sim_context.py",
    "utils.ticking": "utils/ticking.py",
}
_MODEL_SOURCE_FILES: Final = tuple(_MODEL_MODULE_FILES.values())
_IMPORT_LOCK = threading.Lock()
_MANAGED_MODULE_DIGESTS: dict[str, str] = {}

_SCENARIO_BANK: Final[tuple[dict[str, object], ...]] = (
    {
        "scenario_id": "self_propelled",
        "SurfaceShip": [
            {
                "ManueverObject": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "heading": 0.0,
                    "xy_speed": 3.0,
                    "z_speed": 0.0,
                },
                "LauncherObject": {
                    "DecoyObjects": [
                        {
                            "type": "self_propelled",
                            "elevation": 45.0,
                            "azimuth": 45.0,
                            "speed": 7.0,
                            "lifespan": 10.0,
                            "heading": 270.0,
                            "xy_speed": 2.0,
                        },
                        {
                            "type": "self_propelled",
                            "elevation": 45.0,
                            "azimuth": 135.0,
                            "speed": 10.0,
                            "lifespan": 10.0,
                            "heading": 180.0,
                            "xy_speed": 2.0,
                        },
                        {
                            "type": "self_propelled",
                            "elevation": 45.0,
                            "azimuth": 225.0,
                            "speed": 10.0,
                            "lifespan": 10.0,
                            "heading": 225.0,
                            "xy_speed": 2.0,
                        },
                        {
                            "type": "self_propelled",
                            "elevation": 45.0,
                            "azimuth": 315.0,
                            "speed": 10.0,
                            "lifespan": 10.0,
                            "heading": 315.0,
                            "xy_speed": 2.0,
                        },
                    ]
                },
            }
        ],
        "Torpedo": [
            {
                "ManueverObject": {
                    "x": 20.0,
                    "y": 20.0,
                    "z": -10.0,
                    "heading": 270.0,
                    "xy_speed": 5.0,
                    "z_speed": 1.0,
                },
                "DetectorObject": {"detection_range": 35.0},
                "TorpedoControlObject": {"range": 1.0},
            }
        ],
    },
    {
        "scenario_id": "stationary",
        "SurfaceShip": [
            {
                "ManueverObject": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "heading": 0.0,
                    "xy_speed": 3.0,
                    "z_speed": 0.0,
                },
                "LauncherObject": {
                    "DecoyObjects": [
                        {
                            "type": "stationary",
                            "elevation": 45.0,
                            "azimuth": azimuth,
                            "speed": 15.0,
                            "lifespan": 5.0,
                        }
                        for azimuth in (45.0, 135.0, 225.0, 315.0)
                    ]
                },
            }
        ],
        "Torpedo": [
            {
                "ManueverObject": {
                    "x": 20.0,
                    "y": 20.0,
                    "z": -10.0,
                    "heading": 270.0,
                    "xy_speed": 5.0,
                    "z_speed": 1.0,
                },
                "DetectorObject": {"detection_range": 35.0},
                "TorpedoControlObject": {"range": 1.0},
            }
        ],
    },
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


_SCENARIO_BANK_BYTES: Final = _canonical_json(_SCENARIO_BANK)
SCENARIO_BANK_SHA256: Final = hashlib.sha256(_SCENARIO_BANK_BYTES).hexdigest()
ENVIRONMENT_CONTRACT_SHA256: Final = hashlib.sha256(
    _canonical_json(
        {
            "action_port": ACTION_PORT,
            "actions": ACTION_COMMANDS,
            "collision_radius": COLLISION_RADIUS,
            "collision_rule": "swept-relative-segment-active-either-ship-first",
            "decision_delta": DECISION_DELTA,
            "initial_previous_action": -1,
            "launch_cost": LAUNCH_COST,
            "max_steps": MAX_STEPS,
            "observation_schema": OBSERVATION_SCHEMA_VERSION,
            "padding": (
                False,
                "",
                "none",
                False,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                False,
            ),
            "projection_contract_sha256": SEMANTIC_PROJECTION_CONTRACT_SHA256,
            "range_delta_clip": RANGE_DELTA_CLIP,
            "seed_selector": "sha256-prefix-byte-mod-bank-v1",
            "step_bootstrap": "init-start-delay0-step0-snapshot",
            "terminal_rewards": (SHIP_HIT_REWARD, DECOY_CAPTURE_REWARD),
            "turn_cost": TURN_COST,
            "workload_version": WORKLOAD_VERSION,
        }
    )
).hexdigest()


def _require_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("anti-torpedo reset seed must be a non-negative integer")
    return seed


def scenario_for_seed(seed: int) -> tuple[str, dict[str, object]]:
    """Return one fresh literal scenario selected only from the seed digest."""

    validated = _require_seed(seed)
    selector = hashlib.sha256(
        f"anti-torpedo-v1:{validated}".encode("ascii")
    ).digest()[0] % len(_SCENARIO_BANK)
    materialized = json.loads(_SCENARIO_BANK_BYTES.decode("utf-8"))
    if not isinstance(materialized, list):
        raise RuntimeError("frozen scenario bank did not decode to a list")
    scenario = cast(dict[str, object], materialized[selector])
    return cast(str, scenario["scenario_id"]), scenario


def scenario_config_sha256(scenario: Mapping[str, object]) -> str:
    """Return the selected scenario identity, distinct from the bank identity."""

    return hashlib.sha256(_canonical_json(scenario)).hexdigest()


def resolve_atsim_root() -> Path:
    """Resolve the exact external example tree or fail with setup guidance."""

    configured = os.environ.get(AT_SIM_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[5] / "pyjevsim" / "examples" / "hla_atsim"
        root = root.resolve()
    missing = [relative for relative in _MODEL_SOURCE_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"AT/SIM source root {root} is incomplete; set {AT_SIM_ROOT_ENV}; "
            f"missing {missing}"
        )
    return root


def atsim_source_sha256(root: Path | None = None) -> str:
    """Hash every model-side source file that establishes episode semantics."""

    source_root = resolve_atsim_root() if root is None else root.resolve()
    digest = hashlib.sha256()
    for relative in _MODEL_SOURCE_FILES:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"AT/SIM semantic source is missing: {path}")
        name = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def adapter_source_sha256() -> str:
    """Hash the executing workload adapter implementation."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


PYJEVSIM_EXECUTOR_SOURCE_SHA256: Final = hashlib.sha256(
    _canonical_json(PINNED_PYJEVSIM_2_1_2_PROFILE.source_sha256)
).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _verify_expected_digest(
    name: str, actual: str, expected: object | None
) -> None:
    if expected is None:
        return
    if actual != _require_sha256(name, expected):
        raise RuntimeError(f"{name} does not match the admitted source lock")


def _executor_policy() -> ExecutorQualificationPolicy:
    evidence = PINNED_PYJEVSIM_2_1_2_PROFILE.issue(SysExecutor)
    return ExecutorQualificationPolicy(
        semantic_evidence=evidence,
        required_semantics=(
            ExecutorSemanticCapability.CONFLUENT_TRANSITION,
            ExecutorSemanticCapability.ZERO_TIME_CASCADE,
        ),
    )


def _load_atsim_types(
    root: Path, *, expected_source_sha256: str
) -> tuple[type[Any], type[Any], type[Any], Any]:
    """Load the non-packaged example through a locked, collision-checked seam."""

    root_text = str(root)
    with _IMPORT_LOCK:
        preexisting = {
            name for name in _MODEL_MODULE_FILES if name in sys.modules
        }
        unmanaged = preexisting.difference(_MANAGED_MODULE_DIGESTS)
        if unmanaged:
            raise RuntimeError(
                "AT/SIM semantic modules were imported outside the managed "
                f"source-lock seam: {sorted(unmanaged)}"
            )
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        modules = {
            name: importlib.import_module(name) for name in _MODEL_MODULE_FILES
        }
        for name, relative in _MODEL_MODULE_FILES.items():
            module_file = getattr(modules[name], "__file__", None)
            if not isinstance(module_file, str):
                raise RuntimeError(f"AT/SIM module {name} has no source path")
            loaded_file = Path(module_file).resolve()
            expected_file = (root / relative).resolve()
            if loaded_file != expected_file:
                raise RuntimeError(
                    f"AT/SIM module {name} resolved to {loaded_file}, expected "
                    f"{expected_file}"
                )
            observed_digest = hashlib.sha256(loaded_file.read_bytes()).hexdigest()
            managed_digest = _MANAGED_MODULE_DIGESTS.get(name)
            if managed_digest is not None and managed_digest != observed_digest:
                raise RuntimeError(
                    f"AT/SIM module {name} changed after its executable was loaded"
                )
            _MANAGED_MODULE_DIGESTS[name] = observed_digest
        observed_source = atsim_source_sha256(root)
        if observed_source != expected_source_sha256:
            raise RuntimeError("AT/SIM source changed during managed import")
        command_module = modules["model.rl_command_control"]
        if tuple(command_module.ACTION_COMMANDS) != ACTION_COMMANDS:
            raise RuntimeError("AT/SIM action mapping differs from the frozen contract")
    return (
        cast(type[Any], modules["model.rl_surfaceship"].RLSurfaceShip),
        cast(type[Any], modules["model.torpedo"].Torpedo),
        cast(type[Any], modules["utils.sim_context"].SimContext),
        modules["utils.ticking"].commit_tick,
    )


def _position(value: object) -> tuple[float, float, float]:
    getter = getattr(value, "get_position", None)
    if not callable(getter):
        raise TypeError("AT/SIM object must provide get_position()")
    raw = getter()
    if not isinstance(raw, tuple) or len(raw) != 3:
        raise TypeError("AT/SIM position must be a three-tuple")
    result = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("AT/SIM position must be finite")
    return cast(tuple[float, float, float], result)


def _motion_row(value: object) -> tuple[float, float, float, float, float, float]:
    x, y, z = _position(value)
    dynamic = cast(Any, value)
    row = (
        x,
        y,
        z,
        float(dynamic.heading),
        float(dynamic.xy_speed),
        float(dynamic.z_speed),
    )
    if not all(math.isfinite(item) for item in row):
        raise ValueError("AT/SIM motion observation must be finite")
    return row


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def swept_segment_distance(
    source_previous: tuple[float, ...],
    source_current: tuple[float, ...],
    target_previous: tuple[float, ...],
    target_current: tuple[float, ...],
) -> float:
    def vector3(name: str, value: tuple[float, ...]) -> tuple[float, float, float]:
        if not isinstance(value, tuple) or len(value) < 3:
            raise TypeError(f"{name} must be a tuple with at least three numbers")
        result: list[float] = []
        for item in value[:3]:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{name} must contain only numbers")
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(f"{name} must contain only finite numbers")
            result.append(number)
        return result[0], result[1], result[2]

    source_start = vector3("source_previous", source_previous)
    source_end = vector3("source_current", source_current)
    target_start = vector3("target_previous", target_previous)
    target_end = vector3("target_current", target_current)
    relative_previous = tuple(
        source_start[index] - target_start[index] for index in range(3)
    )
    relative_delta = tuple(
        source_end[index]
        - target_end[index]
        - relative_previous[index]
        for index in range(3)
    )
    denominator = sum(item * item for item in relative_delta)
    if denominator == 0:
        fraction = 0.0
    else:
        fraction = max(
            0.0,
            min(
                1.0,
                -sum(
                    relative_previous[index] * relative_delta[index]
                    for index in range(3)
                )
                / denominator,
            ),
        )
    return math.sqrt(
        sum(
            (
                relative_previous[index]
                + fraction * relative_delta[index]
            )
            ** 2
            for index in range(3)
        )
    )


def _observation_rows(observation: object, name: str) -> tuple[object, ...]:
    if not isinstance(observation, Mapping):
        raise TypeError("AT/SIM transition observation must be a mapping")
    row = observation[name]
    if not isinstance(row, tuple):
        raise TypeError(f"AT/SIM observation {name} must be a tuple")
    return row


def swept_collision_outcome(
    previous_observation: object, current_observation: object
) -> str | None:
    """Classify one committed interval with ship-impact priority."""

    previous_ship = cast(
        tuple[float, ...], _observation_rows(previous_observation, "ship")
    )
    current_ship = cast(
        tuple[float, ...], _observation_rows(current_observation, "ship")
    )
    previous_torpedo = cast(
        tuple[float, ...], _observation_rows(previous_observation, "torpedo")
    )
    current_torpedo = cast(
        tuple[float, ...], _observation_rows(current_observation, "torpedo")
    )
    if swept_segment_distance(
        previous_torpedo,
        current_torpedo,
        previous_ship,
        current_ship,
    ) <= COLLISION_RADIUS:
        return "ship_hit"

    previous_rows = cast(
        tuple[tuple[object, ...], ...],
        _observation_rows(previous_observation, "decoys"),
    )
    current_rows = cast(
        tuple[tuple[object, ...], ...],
        _observation_rows(current_observation, "decoys"),
    )
    previous_decoys = {
        cast(str, row[1]): row for row in previous_rows if cast(bool, row[0])
    }
    current_decoys = {
        cast(str, row[1]): row for row in current_rows if cast(bool, row[0])
    }
    for sense_id in sorted(set(previous_decoys).union(current_decoys)):
        previous = previous_decoys.get(sense_id)
        current = current_decoys.get(sense_id)
        if previous is None:
            if current is None:
                raise RuntimeError("decoy identity disappeared from both endpoints")
            previous = current
        if current is None:
            current = previous
        if not (cast(bool, previous[3]) or cast(bool, current[3])):
            continue
        if swept_segment_distance(
            previous_torpedo,
            current_torpedo,
            cast(tuple[float, ...], previous[4:7]),
            cast(tuple[float, ...], current[4:7]),
        ) <= COLLISION_RADIUS:
            return "decoy_capture"
    return None


def anti_torpedo_reward(
    *,
    previous_range: float,
    current_range: float,
    action: int,
    effective_launch: bool,
    outcome: str | None,
) -> float:
    """Apply the exact frozen reward constants to one classified interval."""

    for name, value in (
        ("previous_range", previous_range),
        ("current_range", current_range),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if isinstance(action, bool) or not isinstance(action, int) or action not in range(6):
        raise ValueError("anti-torpedo reward action must be an integer from 0 to 5")
    if type(effective_launch) is not bool:
        raise TypeError("effective_launch must be a bool")
    if outcome not in {None, "ship_hit", "decoy_capture"}:
        raise ValueError("anti-torpedo outcome is not supported")
    distance_gain = max(
        -RANGE_DELTA_CLIP,
        min(RANGE_DELTA_CLIP, float(current_range) - float(previous_range)),
    )
    reward = distance_gain - (TURN_COST if action in {1, 2, 4, 5} else 0.0)
    if effective_launch:
        reward -= LAUNCH_COST
    if outcome == "ship_hit":
        reward += SHIP_HIT_REWARD
    elif outcome == "decoy_capture":
        reward += DECOY_CAPTURE_REWARD
    if not math.isfinite(reward):
        raise ValueError("AT/SIM reward must be finite")
    return reward


class _EpisodeSemantics:
    def __init__(
        self,
        *,
        context: Any,
        ship: Any,
        torpedo: Any,
        scenario_id: str,
        config_sha256: str,
        atsim_model_source_sha256: str,
        commit_tick: Any,
    ) -> None:
        self.context = context
        self.ship = ship
        self.torpedo = torpedo
        self.scenario_id = scenario_id
        self.config_sha256 = config_sha256
        self.atsim_model_source_sha256 = atsim_model_source_sha256
        self.commit_tick = commit_tick
        self.previous_range: float | None = None
        self.prepared_target: float | None = None
        self.outcome_by_step: dict[int, str | None] = {}

    def initialize(self, executor: ExecutorProtocol) -> None:
        initializer = getattr(executor, "init_sim", None)
        if not callable(initializer):
            raise TypeError("AT/SIM executor must provide init_sim()")
        initializer()
        executor.insert_external_event("start", None, scheduled_time=0.0)
        executor.step(0.0)
        self.context.snapshot.refresh(self.context.items)

    def apply_action(self, executor: ExecutorProtocol, action: object) -> None:
        command_module = importlib.import_module("model.rl_command_control")
        command_module.decode_action(action)
        executor.insert_external_event(ACTION_PORT, action, scheduled_time=0.0)

    def decision_time(self, executor: ExecutorProtocol) -> float:
        target = float(executor.get_global_time()) + DECISION_DELTA
        if not math.isfinite(target):
            raise ValueError("AT/SIM decision target must be finite")
        if self.prepared_target == target:
            raise RuntimeError("AT/SIM decision boundary prepared twice")
        self.commit_tick(self.context, int(target))
        self.context.snapshot.refresh(self.context.items)
        self.prepared_target = target
        return target

    def _decoy_rows(self) -> tuple[tuple[object, ...], ...]:
        rows: list[tuple[object, ...]] = []
        for sense_id, decoy in sorted(self.context.decoys, key=lambda item: item[0]):
            x, y, z = _position(decoy)
            self_propelled = hasattr(decoy, "propelled_mode")
            active = bool(decoy.check_active())
            row = (
                True,
                str(sense_id),
                "self_propelled" if self_propelled else "stationary",
                active,
                x,
                y,
                z,
                float(decoy.lifespan),
                float(decoy.time_of_flight),
                bool(decoy.propelled_mode) if self_propelled else False,
            )
            if not all(
                math.isfinite(cast(float, row[index])) for index in range(4, 9)
            ):
                raise ValueError("AT/SIM decoy observation must be finite")
            rows.append(row)
        if len(rows) > 4:
            raise ValueError("AT/SIM workload produced more than four decoys")
        padding = (False, "", "none", False, 0.0, 0.0, 0.0, 0.0, 0.0, False)
        rows.extend(padding for _ in range(4 - len(rows)))
        return tuple(rows)

    def observe(self, _executor: ExecutorProtocol, _events: object) -> dict[str, object]:
        ship = _motion_row(self.ship.mo)
        torpedo = _motion_row(self.torpedo.mo)
        relative_vector = tuple(torpedo[index] - ship[index] for index in range(3))
        current_range = _distance(ship, torpedo)
        closing_rate = (
            0.0
            if self.previous_range is None
            else self.previous_range - current_range
        )
        self.previous_range = current_range
        pending = getattr(self.torpedo.mo, "pending_target", None)
        if pending is None:
            pending_target: tuple[object, ...] = (False, 0.0, 0.0, 0.0)
        else:
            target = tuple(float(item) for item in pending)
            if len(target) != 3 or not all(math.isfinite(item) for item in target):
                raise ValueError("AT/SIM pending target must be a finite three-tuple")
            pending_target = (True, *target)
        previous_action = self.ship.rl_controller.previous_action
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "tick": int(self.context.tick),
            "ship": ship,
            "torpedo": torpedo,
            "relative": (*relative_vector, current_range, closing_rate),
            "pending_target": pending_target,
            "decoy_launched": bool(self.ship.rl_controller.launch_committed),
            "decoys": self._decoy_rows(),
            "scenario": (
                self.scenario_id,
                self.config_sha256,
                ship[4],
                torpedo[4],
                torpedo[5],
            ),
            "previous_action": -1 if previous_action is None else previous_action,
        }

    def _outcome(self, transition: StepView) -> str | None:
        if transition.step_id in self.outcome_by_step:
            return self.outcome_by_step[transition.step_id]
        outcome = swept_collision_outcome(
            transition.previous_observation, transition.observation
        )
        self.outcome_by_step[transition.step_id] = outcome
        return outcome

    def reward(self, transition: StepView) -> float:
        previous_relative = cast(
            tuple[float, ...],
            _observation_rows(transition.previous_observation, "relative"),
        )
        current_relative = cast(
            tuple[float, ...],
            _observation_rows(transition.observation, "relative"),
        )
        action = transition.action
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("AT/SIM transition action must be an integer")
        outcome = self._outcome(transition)
        return anti_torpedo_reward(
            previous_range=previous_relative[3],
            current_range=current_relative[3],
            action=action,
            effective_launch=bool(
                self.ship.rl_controller.last_deploy_effective
            ),
            outcome=outcome,
        )

    def terminated(self, transition: StepView) -> bool:
        return self._outcome(transition) is not None

    def info(self, transition: StepView) -> dict[str, object]:
        return {
            "action_mask": self.ship.rl_controller.action_mask,
            "adapter_source_sha256": LOADED_ADAPTER_SOURCE_SHA256,
            "atsim_model_source_sha256": self.atsim_model_source_sha256,
            "claim_grade": CLAIM_GRADE,
            "config_sha256": self.config_sha256,
            "effective_launch": bool(
                self.ship.rl_controller.last_deploy_effective
            ),
            "environment_contract_sha256": ENVIRONMENT_CONTRACT_SHA256,
            "outcome": self._outcome(transition),
            "projection_contract_sha256": (
                SEMANTIC_PROJECTION_CONTRACT_SHA256
            ),
            "pyjevsim_executor_profile_id": (
                PINNED_PYJEVSIM_2_1_2_PROFILE.profile_id
            ),
            "pyjevsim_executor_revision": (
                PINNED_PYJEVSIM_2_1_2_PROFILE.revision
            ),
            "pyjevsim_executor_source_sha256": (
                PYJEVSIM_EXECUTOR_SOURCE_SHA256
            ),
            "scenario_id": self.scenario_id,
            "scenario_bank_sha256": SCENARIO_BANK_SHA256,
            "workload_version": WORKLOAD_VERSION,
        }


def build_anti_torpedo_episode(context: EpisodeContext) -> FunctionalEpisodeBinding:
    """Build, but do not step, one fresh source-locked AT/SIM model graph."""

    seed = _require_seed(context.seed)
    scenario_id, scenario = scenario_for_seed(seed)
    config_sha256 = scenario_config_sha256(scenario)
    root = resolve_atsim_root()
    atsim_digest = atsim_source_sha256(root)
    expected = context.options
    _verify_expected_digest(
        "expected AT/SIM model source SHA-256",
        atsim_digest,
        expected.get("expected_atsim_model_source_sha256"),
    )
    _verify_expected_digest(
        "expected adapter source SHA-256",
        LOADED_ADAPTER_SOURCE_SHA256,
        expected.get("expected_adapter_source_sha256"),
    )
    _verify_expected_digest(
        "expected environment contract SHA-256",
        ENVIRONMENT_CONTRACT_SHA256,
        expected.get("expected_environment_contract_sha256"),
    )
    _verify_expected_digest(
        "expected projection contract SHA-256",
        SEMANTIC_PROJECTION_CONTRACT_SHA256,
        expected.get("expected_projection_contract_sha256"),
    )
    _verify_expected_digest(
        "expected PyJevSim executor source SHA-256",
        PYJEVSIM_EXECUTOR_SOURCE_SHA256,
        expected.get("expected_pyjevsim_executor_source_sha256"),
    )
    _verify_expected_digest(
        "expected scenario bank SHA-256",
        SCENARIO_BANK_SHA256,
        expected.get("expected_scenario_bank_sha256"),
    )
    surface_type, torpedo_type, context_type, commit_tick = _load_atsim_types(
        root, expected_source_sha256=atsim_digest
    )

    model_context = context_type()
    executor = SysExecutor(1.0, ex_mode=ExecutionType.HLA_TIME, snapshot_manager=None)
    model_context.set_executor(executor)
    ship_data = cast(list[dict[str, object]], scenario["SurfaceShip"])[0]
    torpedo_data = cast(list[dict[str, object]], scenario["Torpedo"])[0]
    ship = surface_type("blue_ship_0", ship_data, model_context)
    torpedo = torpedo_type("red_torpedo_0", torpedo_data, model_context)
    if atsim_source_sha256(root) != atsim_digest:
        raise RuntimeError("AT/SIM source changed during episode construction")

    executor.insert_input_port("start")
    executor.insert_input_port(ACTION_PORT)
    for model in (ship, torpedo):
        executor.register_entity(model)
        executor.coupling_relation(None, "start", model, "start")
    executor.coupling_relation(None, ACTION_PORT, ship, ACTION_PORT)

    semantics = _EpisodeSemantics(
        context=model_context,
        ship=ship,
        torpedo=torpedo,
        scenario_id=scenario_id,
        config_sha256=config_sha256,
        atsim_model_source_sha256=atsim_digest,
        commit_tick=commit_tick,
    )
    return FunctionalEpisodeBinding(
        executor=cast(ExecutorProtocol, executor),
        initialize_fn=semantics.initialize,
        apply_action_fn=semantics.apply_action,
        decision_time_fn=semantics.decision_time,
        observe_fn=semantics.observe,
        reward_fn=semantics.reward,
        terminated_fn=semantics.terminated,
        info_fn=semantics.info,
    )


class AntiTorpedoEnv(PyJevSimEnv):
    """Add workload reset admission and provenance to the generic facade."""

    def __init__(
        self,
        *,
        instance_id: str,
        run_id: str,
        expected_atsim_model_source_sha256: str,
        expected_adapter_source_sha256: str,
        expected_environment_contract_sha256: str,
        expected_projection_contract_sha256: str,
        expected_pyjevsim_executor_source_sha256: str,
        expected_scenario_bank_sha256: str,
    ) -> None:
        self._expected_digests = {
            "expected_atsim_model_source_sha256": _require_sha256(
                "expected AT/SIM model source SHA-256",
                expected_atsim_model_source_sha256,
            ),
            "expected_adapter_source_sha256": _require_sha256(
                "expected adapter source SHA-256", expected_adapter_source_sha256
            ),
            "expected_environment_contract_sha256": _require_sha256(
                "expected environment contract SHA-256",
                expected_environment_contract_sha256,
            ),
            "expected_projection_contract_sha256": _require_sha256(
                "expected projection contract SHA-256",
                expected_projection_contract_sha256,
            ),
            "expected_pyjevsim_executor_source_sha256": _require_sha256(
                "expected PyJevSim executor source SHA-256",
                expected_pyjevsim_executor_source_sha256,
            ),
            "expected_scenario_bank_sha256": _require_sha256(
                "expected scenario bank SHA-256", expected_scenario_bank_sha256
            ),
        }
        super().__init__(
            build_anti_torpedo_episode,
            instance_id=instance_id,
            max_steps=MAX_STEPS,
            run_id=run_id,
            plugin_version=WORKLOAD_VERSION,
            executor_qualification=_executor_policy(),
            require_claim_grade=False,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[object, dict[str, object]]:
        validated = _require_seed(seed)
        if options is not None:
            raise ValueError("anti-torpedo reset options are not supported")
        scenario_id, scenario = scenario_for_seed(validated)
        config_sha256 = scenario_config_sha256(scenario)
        actual_digests = {
            "expected_atsim_model_source_sha256": atsim_source_sha256(),
            "expected_adapter_source_sha256": LOADED_ADAPTER_SOURCE_SHA256,
            "expected_environment_contract_sha256": ENVIRONMENT_CONTRACT_SHA256,
            "expected_projection_contract_sha256": (
                SEMANTIC_PROJECTION_CONTRACT_SHA256
            ),
            "expected_pyjevsim_executor_source_sha256": (
                PYJEVSIM_EXECUTOR_SOURCE_SHA256
            ),
            "expected_scenario_bank_sha256": SCENARIO_BANK_SHA256,
        }
        for name, expected in self._expected_digests.items():
            _verify_expected_digest(name, actual_digests[name], expected)
        observation, info = super().reset(
            seed=validated, options=self._expected_digests
        )
        info.update(
            {
                "action_mask": (True, True, True, True, True, True),
                "adapter_source_sha256": LOADED_ADAPTER_SOURCE_SHA256,
                "atsim_model_source_sha256": actual_digests[
                    "expected_atsim_model_source_sha256"
                ],
                "claim_grade": CLAIM_GRADE,
                "config_sha256": config_sha256,
                "environment_contract_sha256": ENVIRONMENT_CONTRACT_SHA256,
                "projection_contract_sha256": (
                    SEMANTIC_PROJECTION_CONTRACT_SHA256
                ),
                "pyjevsim_executor_profile_id": (
                    PINNED_PYJEVSIM_2_1_2_PROFILE.profile_id
                ),
                "pyjevsim_executor_revision": (
                    PINNED_PYJEVSIM_2_1_2_PROFILE.revision
                ),
                "pyjevsim_executor_source_sha256": (
                    PYJEVSIM_EXECUTOR_SOURCE_SHA256
                ),
                "scenario_bank_sha256": SCENARIO_BANK_SHA256,
                "scenario_id": scenario_id,
                "workload_version": WORKLOAD_VERSION,
            }
        )
        return observation, info


def anti_torpedo_environment_factory(
    *,
    instance_id: str = "env-0",
    run_id: str = "local",
    expected_atsim_model_source_sha256: str | None = None,
    expected_adapter_source_sha256: str | None = None,
    expected_environment_contract_sha256: str | None = None,
    expected_projection_contract_sha256: str | None = None,
    expected_pyjevsim_executor_source_sha256: str | None = None,
    expected_scenario_bank_sha256: str | None = None,
) -> PyJevSimEnv:
    """Return a spawn-resolvable, prototype-grade AT/SIM environment."""

    return AntiTorpedoEnv(
        instance_id=instance_id,
        run_id=run_id,
        expected_atsim_model_source_sha256=(
            atsim_source_sha256()
            if expected_atsim_model_source_sha256 is None
            else expected_atsim_model_source_sha256
        ),
        expected_adapter_source_sha256=(
            LOADED_ADAPTER_SOURCE_SHA256
            if expected_adapter_source_sha256 is None
            else expected_adapter_source_sha256
        ),
        expected_environment_contract_sha256=(
            ENVIRONMENT_CONTRACT_SHA256
            if expected_environment_contract_sha256 is None
            else expected_environment_contract_sha256
        ),
        expected_projection_contract_sha256=(
            SEMANTIC_PROJECTION_CONTRACT_SHA256
            if expected_projection_contract_sha256 is None
            else expected_projection_contract_sha256
        ),
        expected_pyjevsim_executor_source_sha256=(
            PYJEVSIM_EXECUTOR_SOURCE_SHA256
            if expected_pyjevsim_executor_source_sha256 is None
            else expected_pyjevsim_executor_source_sha256
        ),
        expected_scenario_bank_sha256=(
            SCENARIO_BANK_SHA256
            if expected_scenario_bank_sha256 is None
            else expected_scenario_bank_sha256
        ),
    )


def _scenario_ordinal(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("scenario_ordinal must be an integer")
    if value < 0 or value >= V2_CONFIG_COUNT:
        raise ValueError("scenario_ordinal must be between 0 and 255")
    return value


_V2_BANK: Final[ScenarioBankV1] = build_scenario_bank_v1()
V2_SCENARIO_BANK_SHA256: Final = _V2_BANK.sha256
V2_SCENARIO_FAMILY_SHA256: Final = _V2_BANK.family_sha256
V2_SCENARIO_SOURCE_SHA256: Final = _V2_BANK.scenario_source_sha256
V2_FACTOR_SCHEMA_SHA256: Final = _V2_BANK.factor_schema_sha256
V2_GENERATOR_SOURCE_SHA256: Final = _V2_BANK.generator_source_sha256
V2_ENVIRONMENT_CONTRACT_SHA256: Final = hashlib.sha256(
    _canonical_json(
        {
            "base_environment_contract_sha256": ENVIRONMENT_CONTRACT_SHA256,
            "config_selection": "explicit-scenario-ordinal-v1",
            "factor_schema_sha256": V2_FACTOR_SCHEMA_SHA256,
            "observation_schema": V2_OBSERVATION_SCHEMA_VERSION,
            "scenario_bank_sha256": V2_SCENARIO_BANK_SHA256,
            "scenario_family_sha256": V2_SCENARIO_FAMILY_SHA256,
            "scenario_source_sha256": V2_SCENARIO_SOURCE_SHA256,
            "workload_version": V2_WORKLOAD_VERSION,
        }
    )
).hexdigest()


class _V2EpisodeSemantics(_EpisodeSemantics):
    def __init__(
        self,
        *,
        scenario_ordinal: int,
        factor_vector: tuple[int, ...],
        scenario_family_sha256: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.scenario_ordinal = scenario_ordinal
        self.factor_vector = factor_vector
        self.scenario_family_sha256 = scenario_family_sha256

    def observe(self, executor: ExecutorProtocol, events: object) -> dict[str, object]:
        observation = super().observe(executor, events)
        observation.update(
            {
                "schema_version": V2_OBSERVATION_SCHEMA_VERSION,
                "scenario_ordinal": self.scenario_ordinal,
                "factor_vector": self.factor_vector,
                "config_sha256": self.config_sha256,
                "scenario_family_sha256": self.scenario_family_sha256,
            }
        )
        return observation

    def info(self, transition: StepView) -> dict[str, object]:
        info = super().info(transition)
        info.update(
            {
                "environment_contract_sha256": V2_ENVIRONMENT_CONTRACT_SHA256,
                "factor_schema_sha256": V2_FACTOR_SCHEMA_SHA256,
                "factor_vector": self.factor_vector,
                "scenario_bank_sha256": V2_SCENARIO_BANK_SHA256,
                "scenario_family_sha256": self.scenario_family_sha256,
                "scenario_ordinal": self.scenario_ordinal,
                "scenario_source_sha256": V2_SCENARIO_SOURCE_SHA256,
                "workload_version": V2_WORKLOAD_VERSION,
            }
        )
        return info


@dataclass(frozen=True, slots=True)
class AntiTorpedoV2DiagnosticSnapshot:
    """Immutable pre/post-step witness exposed without live model references."""

    canonical_observation: bytes
    logical_time: float
    action_mask: tuple[bool, ...]


def _v2_logical_time(info: Mapping[str, object]) -> float:
    value = info.get("logical_time")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("anti-torpedo-v2 logical_time must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("anti-torpedo-v2 logical_time must be finite")
    return result


_V2_EXPECTED_OPTION_KEYS: Final = frozenset(
    {
        "expected_adapter_source_sha256",
        "expected_atsim_model_source_sha256",
        "expected_environment_contract_sha256",
        "expected_factor_schema_sha256",
        "expected_profile_generator_source_sha256",
        "expected_projection_contract_sha256",
        "expected_pyjevsim_executor_source_sha256",
        "expected_scenario_bank_sha256",
        "expected_scenario_family_sha256",
        "expected_scenario_source_sha256",
        "scenario_ordinal",
    }
)


def _verify_v2_profile_locks(options: Mapping[str, object], bank: ScenarioBankV1) -> None:
    if set(options) != set(_V2_EXPECTED_OPTION_KEYS):
        raise ValueError("anti-torpedo-v2 episode options differ from the closed schema")
    expected = {
        "expected_factor_schema_sha256": bank.factor_schema_sha256,
        "expected_profile_generator_source_sha256": bank.generator_source_sha256,
        "expected_scenario_bank_sha256": bank.sha256,
        "expected_scenario_family_sha256": bank.family_sha256,
        "expected_scenario_source_sha256": bank.scenario_source_sha256,
    }
    for name, actual in expected.items():
        _verify_expected_digest(name, actual, options.get(name))


def build_anti_torpedo_v2_episode(
    context: EpisodeContext,
) -> FunctionalEpisodeBinding:
    """Build one explicit-ordinal v2 profile episode without changing DEVS timing."""

    _require_seed(context.seed)
    options = context.options
    bank = build_scenario_bank_v1()
    _verify_v2_profile_locks(options, bank)
    ordinal = _scenario_ordinal(options["scenario_ordinal"])
    config = bank.configs[ordinal]
    scenario = materialize_config(bank, ordinal)
    if scenario_config_sha256(scenario) != config.config_sha256:
        raise RuntimeError("materialized v2 scenario digest differs from the profile")

    root = resolve_atsim_root()
    atsim_digest = atsim_source_sha256(root)
    for name, actual in (
        ("expected_atsim_model_source_sha256", atsim_digest),
        ("expected_adapter_source_sha256", LOADED_ADAPTER_SOURCE_SHA256),
        ("expected_environment_contract_sha256", V2_ENVIRONMENT_CONTRACT_SHA256),
        ("expected_projection_contract_sha256", SEMANTIC_PROJECTION_CONTRACT_SHA256),
        ("expected_pyjevsim_executor_source_sha256", PYJEVSIM_EXECUTOR_SOURCE_SHA256),
    ):
        _verify_expected_digest(name, actual, options.get(name))
    surface_type, torpedo_type, context_type, commit_tick = _load_atsim_types(
        root, expected_source_sha256=atsim_digest
    )
    model_context = context_type()
    executor = SysExecutor(1.0, ex_mode=ExecutionType.HLA_TIME, snapshot_manager=None)
    model_context.set_executor(executor)
    ship_data = cast(list[dict[str, object]], scenario["SurfaceShip"])[0]
    torpedo_data = cast(list[dict[str, object]], scenario["Torpedo"])[0]
    ship = surface_type("blue_ship_0", ship_data, model_context)
    torpedo = torpedo_type("red_torpedo_0", torpedo_data, model_context)
    if atsim_source_sha256(root) != atsim_digest:
        raise RuntimeError("AT/SIM source changed during v2 episode construction")
    executor.insert_input_port("start")
    executor.insert_input_port(ACTION_PORT)
    for model in (ship, torpedo):
        executor.register_entity(model)
        executor.coupling_relation(None, "start", model, "start")
    executor.coupling_relation(None, ACTION_PORT, ship, ACTION_PORT)
    semantics = _V2EpisodeSemantics(
        context=model_context,
        ship=ship,
        torpedo=torpedo,
        scenario_id=config.config_id,
        config_sha256=config.config_sha256,
        atsim_model_source_sha256=atsim_digest,
        commit_tick=commit_tick,
        scenario_ordinal=ordinal,
        factor_vector=config.level_bits,
        scenario_family_sha256=bank.family_sha256,
    )
    return FunctionalEpisodeBinding(
        executor=cast(ExecutorProtocol, executor),
        initialize_fn=semantics.initialize,
        apply_action_fn=semantics.apply_action,
        decision_time_fn=semantics.decision_time,
        observe_fn=semantics.observe,
        reward_fn=semantics.reward,
        terminated_fn=semantics.terminated,
        info_fn=semantics.info,
    )


class AntiTorpedoV2Env(PyJevSimEnv):
    def __init__(self, *, instance_id: str, run_id: str, expected: Mapping[str, str]) -> None:
        self._expected_v2 = dict(expected)
        self._v2_observation: object | None = None
        self._v2_logical_time: float | None = None
        self._v2_action_mask: tuple[bool, ...] | None = None
        super().__init__(
            build_anti_torpedo_v2_episode,
            instance_id=instance_id,
            max_steps=MAX_STEPS,
            run_id=run_id,
            plugin_version=V2_WORKLOAD_VERSION,
            executor_qualification=_executor_policy(),
            require_claim_grade=False,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[object, dict[str, object]]:
        validated_seed = _require_seed(seed)
        if options is None or set(options) != {"scenario_ordinal"}:
            raise ValueError(
                "anti-torpedo-v2 reset options must contain only scenario_ordinal"
            )
        ordinal = _scenario_ordinal(options["scenario_ordinal"])
        bank = build_scenario_bank_v1()
        expected_actual = {
            "expected_factor_schema_sha256": bank.factor_schema_sha256,
            "expected_profile_generator_source_sha256": bank.generator_source_sha256,
            "expected_scenario_bank_sha256": bank.sha256,
            "expected_scenario_family_sha256": bank.family_sha256,
            "expected_scenario_source_sha256": bank.scenario_source_sha256,
        }
        for name, actual in expected_actual.items():
            _verify_expected_digest(name, actual, self._expected_v2[name])
        internal_options: dict[str, object] = dict(self._expected_v2)
        internal_options["scenario_ordinal"] = ordinal
        observation, info = super().reset(
            seed=validated_seed, options=internal_options
        )
        config = bank.configs[ordinal]
        info.update(
            {
                "action_mask": (True, True, True, True, True, True),
                "config_sha256": config.config_sha256,
                "environment_contract_sha256": V2_ENVIRONMENT_CONTRACT_SHA256,
                "factor_schema_sha256": bank.factor_schema_sha256,
                "factor_vector": config.level_bits,
                "scenario_bank_sha256": bank.sha256,
                "scenario_family_sha256": bank.family_sha256,
                "scenario_id": config.config_id,
                "scenario_ordinal": ordinal,
                "scenario_source_sha256": bank.scenario_source_sha256,
                "workload_version": V2_WORKLOAD_VERSION,
            }
        )
        self._v2_observation = observation
        self._v2_logical_time = _v2_logical_time(info)
        self._v2_action_mask = cast(tuple[bool, ...], info["action_mask"])
        return observation, info

    def step(
        self, action: object
    ) -> tuple[object, float, bool, bool, dict[str, object]]:
        """Reject invalid or masked actions before any model/executor mutation."""

        with self._lifecycle_lock:
            self._ensure_lifecycle_admission("step")
            if isinstance(action, bool) or not isinstance(action, int):
                raise TypeError("anti-torpedo-v2 action must be an integer")
            if action < 0 or action >= len(ACTION_COMMANDS):
                raise ValueError("anti-torpedo-v2 action must be between 0 and 5")
            if self._v2_action_mask is not None and not self._v2_action_mask[action]:
                raise ValueError(f"anti-torpedo-v2 action {action} is masked")
            result = self._step_unlocked(action)
            observation, _reward, _terminated, _truncated, info = result
            self._v2_observation = observation
            self._v2_logical_time = _v2_logical_time(info)
            self._v2_action_mask = cast(tuple[bool, ...], info["action_mask"])
            return result

    def diagnostic_snapshot(self) -> AntiTorpedoV2DiagnosticSnapshot:
        """Return an immutable qualification witness for the current v2 state."""

        with self._lifecycle_lock:
            self._ensure_open()
            if (
                self._v2_observation is None
                or self._v2_logical_time is None
                or self._v2_action_mask is None
            ):
                raise RuntimeError("anti-torpedo-v2 reset is required before snapshot")
            return AntiTorpedoV2DiagnosticSnapshot(
                canonical_observation=_canonical_json(self._v2_observation),
                logical_time=self._v2_logical_time,
                action_mask=self._v2_action_mask,
            )


def anti_torpedo_v2_environment_factory(
    *,
    instance_id: str = "env-v2-0",
    run_id: str = "local-v2",
    expected_atsim_model_source_sha256: str | None = None,
    expected_adapter_source_sha256: str | None = None,
    expected_environment_contract_sha256: str | None = None,
    expected_projection_contract_sha256: str | None = None,
    expected_pyjevsim_executor_source_sha256: str | None = None,
    expected_profile_generator_source_sha256: str | None = None,
    expected_scenario_bank_sha256: str | None = None,
    expected_scenario_family_sha256: str | None = None,
    expected_scenario_source_sha256: str | None = None,
    expected_factor_schema_sha256: str | None = None,
) -> AntiTorpedoV2Env:
    """Return a top-level pickle-safe explicit-profile v2 environment."""

    defaults = {
        "expected_atsim_model_source_sha256": atsim_source_sha256(),
        "expected_adapter_source_sha256": LOADED_ADAPTER_SOURCE_SHA256,
        "expected_environment_contract_sha256": V2_ENVIRONMENT_CONTRACT_SHA256,
        "expected_projection_contract_sha256": SEMANTIC_PROJECTION_CONTRACT_SHA256,
        "expected_pyjevsim_executor_source_sha256": PYJEVSIM_EXECUTOR_SOURCE_SHA256,
        "expected_profile_generator_source_sha256": V2_GENERATOR_SOURCE_SHA256,
        "expected_scenario_bank_sha256": V2_SCENARIO_BANK_SHA256,
        "expected_scenario_family_sha256": V2_SCENARIO_FAMILY_SHA256,
        "expected_scenario_source_sha256": V2_SCENARIO_SOURCE_SHA256,
        "expected_factor_schema_sha256": V2_FACTOR_SCHEMA_SHA256,
    }
    provided = {
        "expected_atsim_model_source_sha256": expected_atsim_model_source_sha256,
        "expected_adapter_source_sha256": expected_adapter_source_sha256,
        "expected_environment_contract_sha256": expected_environment_contract_sha256,
        "expected_projection_contract_sha256": expected_projection_contract_sha256,
        "expected_pyjevsim_executor_source_sha256": expected_pyjevsim_executor_source_sha256,
        "expected_profile_generator_source_sha256": expected_profile_generator_source_sha256,
        "expected_scenario_bank_sha256": expected_scenario_bank_sha256,
        "expected_scenario_family_sha256": expected_scenario_family_sha256,
        "expected_scenario_source_sha256": expected_scenario_source_sha256,
        "expected_factor_schema_sha256": expected_factor_schema_sha256,
    }
    expected = {
        name: _require_sha256(name, defaults[name] if value is None else value)
        for name, value in provided.items()
    }
    return AntiTorpedoV2Env(instance_id=instance_id, run_id=run_id, expected=expected)


LOADED_ADAPTER_SOURCE_SHA256: Final = adapter_source_sha256()
