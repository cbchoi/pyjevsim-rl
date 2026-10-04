"""Closed local runner inputs and immutable, relocatable evidence primitives."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .ppo_features import canonical_feature_json
from .reference_ppo import ReferencePPOConfigV1

CONFIG_SCHEMA = "local-training-config-v1"
CHECKPOINT_SCHEMA = "local-runner-checkpoint-v1"
RECEIPT_SCHEMA = "local-runner-receipt-v1"
CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "generation",
        "model_id",
        "model_version",
        "model_factory",
        "feature_factory",
        "model_config",
        "source_modules",
        "executor_profile_id",
        "boundary_delta",
        "max_steps",
        "objective_id",
        "ppo_config",
        "backend",
        "worker_count",
        "training",
        "evaluation",
    }
)
CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "config_sha256",
        "source_identity",
        "runtime_identity",
        "objective_id",
        "run_id",
        "generation",
        "completed_updates",
        "completed_transitions",
        "next_job_index",
        "seed_schedule_sha256",
        "active_policy_reference",
        "learner_recovery_state",
        "artifacts",
        "episode_dispositions_sha256",
        "checkpoint_sha256",
    }
)
_IMPORT = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*\Z")


class RunnerContractError(ValueError):
    """A config, source, job or checkpoint fails closed validation."""


def canonical(value: object) -> bytes:
    return canonical_feature_json(value)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def byte_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_fields(value: object, fields: set[str] | frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RunnerContractError(f"{name} fields differ from closed schema")
    canonical(value)
    return dict(value)


def integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RunnerContractError(f"{name} must be an integer >= {minimum}")
    return value


def text_value(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunnerContractError(f"{name} must be nonempty text")
    return value


def sha_value(value: object, name: str) -> str:
    result = text_value(value, name)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise RunnerContractError(f"{name} must be lowercase SHA256")
    return result


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RunnerContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> object:
    raise RunnerContractError(f"nonfinite JSON token: {value}")


def decode_json(payload: bytes) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RunnerContractError("malformed JSON") from exc


@dataclass(frozen=True)
class RunnerConfig:
    """Immutable canonical config; content() returns a detached copy."""

    payload: bytes

    def __post_init__(self) -> None:
        raw = object_fields(decode_json(self.payload), CONFIG_FIELDS, "runner config")
        if raw["schema_version"] != CONFIG_SCHEMA or raw["objective_id"] != "decision-index-v1":
            raise RunnerContractError("runner schema/objective differs")
        for key in ("run_id", "model_id", "model_version"):
            text_value(raw[key], key)
        for key, minimum in (("generation", 0), ("worker_count", 1), ("max_steps", 1)):
            integer(raw[key], key, minimum)
        if raw["backend"] not in ("serial", "thread", "process"):
            raise RunnerContractError("unknown local backend")
        delta = raw["boundary_delta"]
        if type(delta) not in (int, float) or not math.isfinite(delta) or delta <= 0:
            raise RunnerContractError("boundary_delta must be finite and positive")
        modules = raw["source_modules"]
        if not isinstance(modules, dict) or not modules:
            raise RunnerContractError("source_modules must be an explicit nonempty inventory")
        for module, expected in modules.items():
            if not isinstance(module, str) or not _IMPORT.fullmatch(module + ":source"):
                raise RunnerContractError("source module must be importable module name")
            sha_value(expected, "source module digest")
        for key in ("model_factory", "feature_factory"):
            reference = text_value(raw[key], key)
            if not _IMPORT.fullmatch(reference) or reference.split(":")[0] not in modules:
                raise RunnerContractError("factory reference missing from source inventory")
        if raw["executor_profile_id"] is not None:
            text_value(raw["executor_profile_id"], "executor_profile_id")
        if not isinstance(raw["model_config"], dict):
            raise RunnerContractError("model_config must be an object")
        if not isinstance(raw["ppo_config"], dict):
            raise RunnerContractError("ppo_config must be an object")
        ppo = ReferencePPOConfigV1.from_dict(raw["ppo_config"])
        if ppo.schema_version != "reference-ppo-config-v1":
            raise RunnerContractError("PPO config schema differs")
        train = object_fields(
            raw["training"],
            {
                "total_updates",
                "seed_schedule_domain",
                "master_seed",
                "max_episode_jobs",
            },
            "training",
        )
        integer(train["total_updates"], "total_updates", 1)
        integer(train["master_seed"], "master_seed")
        integer(train["max_episode_jobs"], "max_episode_jobs", raw["worker_count"])
        if train["seed_schedule_domain"] != "local-runner-job-seed-v1":
            raise RunnerContractError("seed schedule domain differs")
        evaluation = object_fields(
            raw["evaluation"],
            {
                "episode_count",
                "master_seed",
                "explore",
                "model_configs",
            },
            "evaluation",
        )
        count = integer(evaluation["episode_count"], "evaluation episode_count", 1)
        integer(evaluation["master_seed"], "evaluation master_seed")
        bank = evaluation["model_configs"]
        if not isinstance(bank, list) or not bank or any(not isinstance(row, dict) for row in bank):
            raise RunnerContractError("evaluation.model_configs must be a nonempty object bank")
        if evaluation["explore"] is not False:
            raise RunnerContractError("evaluation explore must be false")
        if ppo.batch_size % raw["worker_count"] or count % raw["worker_count"] or count % len(bank):
            raise RunnerContractError("batch/evaluation counts violate worker/case divisibility")
        object.__setattr__(self, "payload", canonical(raw))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RunnerConfig:
        return cls(canonical(value))

    def content(self) -> dict[str, Any]:
        return cast(dict[str, Any], decode_json(self.payload))

    @property
    def sha256(self) -> str:
        return byte_digest(self.payload)


def load_config(path: str | Path) -> RunnerConfig:
    return RunnerConfig(Path(path).read_bytes())


def ensure_config(value: RunnerConfig | Mapping[str, object]) -> RunnerConfig:
    return value if isinstance(value, RunnerConfig) else RunnerConfig.from_dict(value)


def import_callable(reference: str) -> Any:
    module_name, name = reference.split(":")
    result = getattr(importlib.import_module(module_name), name)
    if not callable(result):
        raise RunnerContractError("factory reference is not callable")
    return result


def verify_sources(config: RunnerConfig) -> dict[str, str]:
    sources: dict[str, str] = {}
    for name, expected in config.content()["source_modules"].items():
        module = importlib.import_module(name)
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise RunnerContractError("source inventory requires file-backed modules")
        actual = byte_digest(Path(filename).read_bytes())
        if actual != expected:
            raise RunnerContractError(f"source inventory drift: {name}")
        sources[name] = actual
    return sources


def build_schedule(config: RunnerConfig) -> dict[str, Any]:
    data = config.content()
    schedule: dict[str, Any] = {
        "domain": data["training"]["seed_schedule_domain"],
        "evaluation_case_bank_sha256": digest(data["evaluation"]["model_configs"]),
        "training": [],
        "evaluation": [],
    }
    seeds: set[int] = set()
    for phase, count in (
        ("training", data["training"]["max_episode_jobs"]),
        ("evaluation", data["evaluation"]["episode_count"]),
    ):
        for index in range(count):
            master = data[phase]["master_seed"]
            seed = int.from_bytes(
                hashlib.sha256(
                    canonical(
                        [
                            schedule["domain"],
                            phase,
                            master,
                            index,
                        ]
                    )
                ).digest()[:8],
                "big",
            ) & ((1 << 63) - 1)
            if seed in seeds:
                raise RunnerContractError("scheduled episode seed collision")
            seeds.add(seed)
            case = (
                index % len(data["evaluation"]["model_configs"]) if phase == "evaluation" else None
            )
            model = (
                data["model_config"] if case is None else data["evaluation"]["model_configs"][case]
            )
            schedule[phase].append(
                {
                    "job_index": index,
                    "episode_id": f"{data['run_id']}:{phase}:job-{index:08d}",
                    "worker_id": f"worker-{index % data['worker_count']:04d}",
                    "seed": seed,
                    "case_index": case,
                    "config_sha256": digest(model),
                    "model_config": model,
                }
            )
    return schedule


def atomic_write(path: Path, body: bytes) -> None:
    """Publish a new immutable file; interrupted pending bytes remain diagnostic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".pending-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    if temporary.read_bytes() != body:
        raise RunnerContractError("written artifact differs before publication")
    os.link(temporary, path)  # Exclusive atomic publication: never replaces an old checkpoint.
    temporary.unlink()


def contained_file(root: Path, relative: object) -> Path:
    value = text_value(relative, "artifact path")
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root.resolve()):
        raise RunnerContractError("artifact path escapes checkpoint bundle")
    return path
