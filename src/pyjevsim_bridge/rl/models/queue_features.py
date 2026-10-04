"""Model-owned7-feature/3-action projection; no learner/update implementation."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pyjevsim_bridge.rl.ppo_features import (
    FeatureContractError,
    canonical_feature_json,
    validate_action_mask,
    validate_inference_envelope,
)

from .queue_control import (
    MODES,
    OBSERVATION_FIELDS,
    OBSERVATION_SCHEMA,
    QueueConfigurationError,
    action_mode,
)

FEATURE_SCHEMA = "queue-control-feature-v1"
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_ACTIONS = tuple({"mode": mode} for mode in MODES)
_ACTION_SHA256 = hashlib.sha256(canonical_feature_json(_ACTIONS)).hexdigest()
_CONTRACT_SHA256 = hashlib.sha256(
    canonical_feature_json(
        {
            "schema": FEATURE_SCHEMA,
            "observation_schema": OBSERVATION_SCHEMA,
            "dtype": "float32",
            "action_schema_sha256": _ACTION_SHA256,
            "features": [
                "waiting/capacity",
                "busy",
                "remaining_work",
                "mode_index/2",
                "min(time/source_end,1)",
                "drops/(initial+source_arrivals+1)",
                "source_exhausted",
            ],
        }
    )
).hexdigest()


def _observation(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != OBSERVATION_FIELDS:
        raise FeatureContractError("queue observation fields differ")
    raw = dict(value)
    if raw["schema_version"] != OBSERVATION_SCHEMA:
        raise FeatureContractError("queue observation schema differs")
    for name in (
        "waiting_capacity",
        "initial_job_count",
        "source_arrivals",
        "admitted",
        "completed",
        "dropped",
    ):
        if type(raw[name]) is not int or raw[name] < 0:
            raise FeatureContractError(f"queue {name} must be a nonnegative integer")
    if not 1 <= raw["waiting_capacity"] <= 16:
        raise FeatureContractError("queue capacity outside1..16")
    for name in (
        "logical_time",
        "source_end_time",
        "remaining_work",
        "backlog_integral",
        "energy_integral",
        "cumulative_cost",
    ):
        number = raw[name]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise FeatureContractError(f"queue {name} must be finite nonnegative numeric")
    if raw["source_end_time"] == 0 or raw["remaining_work"] > 1:
        raise FeatureContractError("queue end time or remaining work outside domain")
    waiting, current = raw["waiting_job_ids"], raw["in_service_id"]
    if not isinstance(waiting, (tuple, list)) or any(
        not isinstance(job, str) or not job for job in waiting
    ):
        raise FeatureContractError("queue waiting IDs must be a sequence of nonempty strings")
    if current is not None and (not isinstance(current, str) or not current):
        raise FeatureContractError("queue service ID must be a string or null")
    jobs = list(waiting) + ([current] if current is not None else [])
    if len(set(jobs)) != len(jobs) or len(waiting) > raw["waiting_capacity"]:
        raise FeatureContractError("queue inventory duplicates or overflow")
    if (
        not isinstance(raw["mode"], str)
        or raw["mode"] not in MODES
        or type(raw["source_exhausted"]) is not bool
    ):
        raise FeatureContractError("queue mode/exhaustion type invalid")
    if current is None and (waiting or raw["remaining_work"] != 0 or raw["mode"] != "idle"):
        raise FeatureContractError("empty queue has inconsistent state")
    if (
        raw["admitted"] != len(jobs) + raw["completed"]
        or raw["initial_job_count"] + raw["source_arrivals"] != raw["admitted"] + raw["dropped"]
    ):
        raise FeatureContractError("queue conservation fields disagree")
    return raw


class QueueFeatures:
    feature_id = FEATURE_SCHEMA
    schema_version = FEATURE_SCHEMA
    feature_size = 7
    action_count = 3
    dtype = "float32"
    contract_sha256 = _CONTRACT_SHA256
    action_schema_sha256 = _ACTION_SHA256
    source_sha256 = _SOURCE_SHA256

    def observation_action_mask(self, observation: object) -> tuple[bool, ...]:
        raw = _observation(observation)
        return (True, True, True) if raw["in_service_id"] is not None else (True, False, False)

    def action_mask(self, value: Mapping[str, object]) -> tuple[bool, ...]:
        envelope = validate_inference_envelope(value)
        expected = self.observation_action_mask(envelope["observation"])
        if validate_action_mask(envelope["action_mask"], 3) != expected:
            raise FeatureContractError("queue envelope mask differs from physical state")
        return expected

    def encode(self, value: Mapping[str, object]) -> tuple[float, ...]:
        self.action_mask(value)
        raw = _observation(value["observation"])
        vector = (
            len(raw["waiting_job_ids"]) / raw["waiting_capacity"],
            float(raw["in_service_id"] is not None),
            raw["remaining_work"],
            MODES.index(raw["mode"]) / 2,
            min(raw["logical_time"] / raw["source_end_time"], 1.0),
            raw["dropped"] / (raw["initial_job_count"] + raw["source_arrivals"] + 1),
            float(raw["source_exhausted"]),
        )
        return tuple(struct.unpack("!f", struct.pack("!f", item))[0] for item in vector)

    def encode_action(self, action: object) -> int:
        try:
            return MODES.index(action_mode(action))
        except QueueConfigurationError as exc:
            raise FeatureContractError(str(exc)) from exc

    def decode_action(self, index: int) -> object:
        if type(index) is not int or not 0 <= index < 3:
            raise FeatureContractError("queue action index outside0..2")
        return {"mode": MODES[index]}


def make_queue_features() -> QueueFeatures:
    return QueueFeatures()
