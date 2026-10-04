"""Generate model-specific installed examples, never instantiate/train a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from pyjevsim_bridge.rl.executor import PINNED_PYJEVSIM_2_1_2_PROFILE
from pyjevsim_bridge.rl.reference_ppo import ReferencePPOConfigV1

from .queue_control import CONFIG_SCHEMA, MODEL_ID, MODEL_VERSION

# Explicitly reviewed import closure, not automatic transitive discovery.
# Additional dependency identities do not expand the8-module semantic profile.
_DEPENDENCY_LF_SHA256 = dict(PINNED_PYJEVSIM_2_1_2_PROFILE.source_sha256) | {
    "pyjevsim": "23d8d1d67318ca9161846eff8ff2d9d06faf3a0faedc79d378856ee7e973642c",
    "pyjevsim.core_model": "abfb74a4cd783aecfcfe96fb2abaf3febb5f1517e34ad66ec418abe0c35c2d9c",
    "pyjevsim.default_message_catcher": (
        "6d34bdeb2148fdac5263943efb8106647c69b3c71fce53fce17289a2cc7a6483"
    ),
    "pyjevsim.executor": "14afe3c1ce16632ac805474a8cf3f8ea18558fee13dd5c09b5b26af3737a2502",
    "pyjevsim.executor_factory": "508593d17e958e0a4fafbe0348201923c5a7a2a41a4d3d858e8b1e2c1d271625",
    "pyjevsim.restore_handler": "a40d763d2f53eeb19bf9ddd44b7ec468016c91f4fc021e9b15c264aa946137e5",
    "pyjevsim.snapshot_executor": (
        "c2927308ed8430dc7f04b89af984d42c7280c8860a347c4683724b9dfa4320a9"
    ),
    "pyjevsim.snapshot_factory": "b065ab138123e19f1467b0ccb013617b10851542829facdf5be151ffab42ebb2",
    "pyjevsim.snapshot_manager": "99e756963b922ed2bbd5e48413f9dc612f7e68ac006cda8d59c8dca41a95a3b5",
    "pyjevsim.structural_executor": (
        "1b40803d60dff2c351beffef09e1d8f5e8a3398a7d41c918f934811fe01031fb"
    ),
    "pyjevsim.structural_model": "8642faddde2ef598ed281fe4298b1dce59ee8d40be4b928356986f80d2509185",
    "pyjevsim.termination_manager": (
        "d7c01d85de63463560581d76a61dc743a18397e6e02ca3b87a3ff0f3b0f96a47"
    ),
}
_MODEL_MODULES = (
    "pyjevsim_bridge.rl.models",
    "pyjevsim_bridge.rl.models.queue_control",
    "pyjevsim_bridge.rl.models.queue_features",
    "pyjevsim_bridge.rl.models.queue_config",
)


def _module_bytes(name: str) -> bytes:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        raise ValueError(f"source module must have a Python source file: {name}")
    return Path(spec.origin).read_bytes()


def source_inventory() -> dict[str, str]:
    if importlib.metadata.version("pyjevsim") != "2.1.2":
        raise ValueError("queue example requires reviewed PyJevSim2.1.2 revision9893099")
    result = {}
    for name, expected in sorted(_DEPENDENCY_LF_SHA256.items()):
        data = _module_bytes(name)
        if hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"reviewed PyJevSim dependency source differs: {name}")
        result[name] = hashlib.sha256(data).hexdigest()
    for name in _MODEL_MODULES:
        result[name] = hashlib.sha256(_module_bytes(name)).hexdigest()
    return result


def build_config(
    *, profile: str = "smoke", backend: str = "serial", workers: int = 1, run_id: str | None = None
) -> dict[str, Any]:
    """Return the closed example configuration and actual installed source hashes."""
    if profile not in ("smoke", "qualification") or backend not in ("serial", "thread", "process"):
        raise ValueError("unknown example profile or backend")
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError("example workers must be1,2or4")
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ValueError("run_id must be nonempty")
    qualification = profile == "qualification"
    config: dict[str, Any] = {
        "schema_version": CONFIG_SCHEMA,
        "waiting_capacity": 3 if qualification else 2,
        "initial_service": None,
        "initial_waiting": [],
        "initial_mode": "idle",
        "arrival_spec": {
            "kind": "bernoulli-slots",
            "slot_count": 64,
            "slot_width": 0.5,
            "probability": 0.5 if qualification else 0.25,
        },
        "cost_weights": {"backlog": 1.0, "energy": 1.0, "drop": 10.0},
    }
    ppo = ReferencePPOConfigV1(
        observation_size=7,
        action_count=3,
        hidden_size=16,
        batch_size=256 if qualification else 64,
        minibatch_size=64 if qualification else 32,
        epochs=4 if qualification else 2,
        initialization_seed=0,
        shuffle_seed=1,
    )
    return {
        "schema_version": "local-training-config-v1",
        "run_id": run_id or f"queue-local-{profile}-v1",
        "generation": 0,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_factory": "pyjevsim_bridge.rl.models.queue_control:make_queue_episode",
        "feature_factory": "pyjevsim_bridge.rl.models.queue_features:make_queue_features",
        "model_config": config,
        "source_modules": source_inventory(),
        "executor_profile_id": PINNED_PYJEVSIM_2_1_2_PROFILE.profile_id,
        "boundary_delta": 0.5,
        "max_steps": 64,
        "objective_id": "decision-index-v1",
        "ppo_config": ppo.content(),
        "backend": backend,
        "worker_count": workers,
        "training": {
            "total_updates": 16 if qualification else 2,
            "seed_schedule_domain": "local-runner-job-seed-v1",
            "master_seed": 20260919,
            "max_episode_jobs": 128,
        },
        "evaluation": {
            "episode_count": 4,
            "master_seed": 20260920,
            "explore": False,
            "model_configs": [json.loads(json.dumps(config))],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "qualification"), default="smoke")
    parser.add_argument("--backend", choices=("serial", "thread", "process"), default="serial")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    content = build_config(
        profile=args.profile, backend=args.backend, workers=args.workers, run_id=args.run_id
    )
    payload = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    with args.output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"Created {args.output} ({args.profile}; not a utility/performance result)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
