"""Bounded P0 fixed-delta RL state, separate from engine and domain state.

Only fresh/restore environments constructed here are admitted. This provider
does not adopt arbitrary closures, recurrent policies or variable boundaries.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..adapters import FunctionalEpisodeBinding
from ..environment import PyJevSimEnv
from ..executor import ExecutorDriver, FixedDeltaBoundary
from ..models import _queue_snapshot_state as legacy_values
from .contracts import (
    BoundaryStateHandle, ContinuationError, Violation, canonical_bytes,
    decode_json, exact_fields, fail, thaw_value,
)
from .registry import SourceBinding

PROFILE_ID = "CC-P0-queue-compat"
BOUNDARY_SCHEMA = "queue-fixed-delta-boundary-v1"
_DESCRIPTOR = {"schema_id", "kind", "delta", "max_steps", "instance_id", "run_id"}
_FLAGS = (
    "_failed", "_closed", "_closing", "_disposing", "_constructing",
    "_construction_cleanup_started", "_stepping", "_close_requested",
)
_IMPLEMENTATIONS = {
    cls: {
        name: (getattr(cls, name), getattr(getattr(cls, name), "__code__", None))
        for name, member in vars(cls).items()
        if callable(member) and (not name.startswith("__") or name == "__init__")
    }
    for cls in (PyJevSimEnv, ExecutorDriver, FixedDeltaBoundary, FunctionalEpisodeBinding)
}


def _detached(value: dict) -> dict:
    return decode_json(canonical_bytes(value))


def _check_descriptor(value: dict) -> dict:
    exact_fields(value, _DESCRIPTOR, "P0 boundary descriptor")
    if value["schema_id"] != BOUNDARY_SCHEMA or value["kind"] != "fixed-delta":
        fail("only the P0 fixed-delta boundary is implemented", "CC_UNSUPPORTED_PROFILE")
    try:
        if legacy_values.number(value["delta"], "delta") <= 0:
            fail("delta must be positive")
        legacy_values.integer(value["max_steps"], "max_steps", 1, legacy_values.MAX_STEPS)
        legacy_values.text(value["instance_id"], "instance_id")
        legacy_values.text(value["run_id"], "run_id")
    except legacy_values.QueueSnapshotError as exc:
        raise ContinuationError("CC_INVALID_PAYLOAD", str(exc)) from exc
    return value


def _stamp(env: PyJevSimEnv, state: BoundaryStateHandle) -> None:
    env._continuation_boundary_state = state
    env._continuation_binding_values = dict(vars(env._binding))
    env._continuation_factory_identity = env._factory
    env._continuation_shape = frozenset((*vars(env), "_continuation_shape"))


class FixedDeltaBoundaryProvider:
    """The empty reward handle is explicit; prior observation owns the baseline."""

    def __init__(self) -> None:
        from .. import adapters, environment, executor
        files = {
            "continuation.boundary": Path(__file__),
            "rl.environment": Path(environment.__file__),
            "rl.executor": Path(executor.__file__),
            "rl.adapters": Path(adapters.__file__),
        }
        self.source_bindings = tuple(
            SourceBinding(name, str(path), hashlib.sha256(path.read_bytes()).hexdigest())
            for name, path in files.items()
        )

    def verify_identity(self) -> None:
        for cls, methods in _IMPLEMENTATIONS.items():
            if any(getattr(cls, name) is not method or getattr(method, "__code__", None) is not code
                   for name, (method, code) in methods.items()):
                fail("installed boundary implementation changed", "CC_INCOMPATIBLE_IDENTITY")

    def cursor(self, runtime: Any) -> int:
        return runtime.env._step_id

    def validate_branch(self, saved_logical: dict, branch: Any) -> None:
        """Validate the declared branch's sampling lineage before allocation.

        Sampling context is not model RNG state. The model seed and stored tape
        remain unchanged; this hook does not introduce a model reseed path.
        """
        try:
            previous = legacy_values.fields(
                thaw_value(saved_logical["sampling_context"]), legacy_values.SAMPLING_FIELDS,
                "saved sampling context",
            )
            requested = legacy_values.fields(
                thaw_value(branch.sampling_context), legacy_values.SAMPLING_FIELDS,
                "branch sampling context",
            )
            previous = legacy_values.sampling_context(previous, previous["run_id"])
            requested = legacy_values.sampling_context(requested, requested["run_id"])
        except legacy_values.QueueSnapshotError as exc:
            raise ContinuationError("CC_INVALID_PAYLOAD", str(exc)) from exc
        fixed = ("domain", "phase", "master", "run_id", "generation")
        if any(previous[key] != requested[key] for key in fixed):
            fail("branch sampling changes its fixed family lineage", "CC_INCOMPATIBLE_IDENTITY")
        if requested["segment"] != "suffix" or requested["logical_branch_id"] != branch.branch_id:
            fail("branch sampling does not identify the requested suffix", "CC_INCOMPATIBLE_IDENTITY")

    def descriptor_from_request(self, request: Any) -> dict:
        return _check_descriptor({
            "schema_id": BOUNDARY_SCHEMA, "kind": "fixed-delta",
            "delta": request.delta, "max_steps": request.max_steps,
            "instance_id": request.instance_id, "run_id": request.run_id,
        })

    def allocate_state(self, descriptor: dict, cleanup: Any) -> BoundaryStateHandle:
        _check_descriptor(descriptor)
        return BoundaryStateHandle({})

    def create_fresh(self, binding: Any, descriptor: dict, seed: int,
                     cleanup: Any, boundary_state: BoundaryStateHandle) -> PyJevSimEnv:
        descriptor = _check_descriptor(descriptor)
        consumed = False

        def fresh_factory(_context: Any) -> Any:
            nonlocal consumed
            if consumed:
                fail("P0 continuation runtime does not permit a second reset", "CC_INVALID_BOUNDARY")
            consumed = True
            return binding

        env = PyJevSimEnv(
            fresh_factory, instance_id=descriptor["instance_id"],
            run_id=descriptor["run_id"], boundary=FixedDeltaBoundary(descriptor["delta"]),
            max_steps=descriptor["max_steps"], plugin_version=PROFILE_ID,
        )
        cleanup.add("boundary-env", env.close)
        env.reset(seed=seed)
        _stamp(env, boundary_state)
        return env

    def bind_uninitialized(self, binding: Any, engine: Any,
                           boundary_state: BoundaryStateHandle, descriptor: dict,
                           cleanup: Any, *, branch: Any = None) -> PyJevSimEnv:
        descriptor = _check_descriptor(descriptor)
        instance_id = descriptor["instance_id"] if branch is None else branch.instance_id

        def no_reset(_context: Any) -> Any:
            fail("restored P0 runtime cannot reset", "CC_INVALID_BOUNDARY")

        env = PyJevSimEnv(
            no_reset, instance_id=instance_id, run_id=descriptor["run_id"],
            boundary=FixedDeltaBoundary(descriptor["delta"]),
            max_steps=descriptor["max_steps"], plugin_version=PROFILE_ID,
        )
        cleanup.add("boundary-env", env.close)
        env._binding = binding
        env._driver = ExecutorDriver(engine)
        _stamp(env, boundary_state)
        return env

    def export_state(self, runtime: Any) -> dict:
        env = runtime.env
        return _detached({
            "descriptor": {
                "schema_id": BOUNDARY_SCHEMA, "kind": "fixed-delta",
                "delta": env._boundary.delta, "max_steps": env._max_steps,
                "instance_id": env._instance_id, "run_id": env._run_id,
            },
            "environment": {
                "instance_id": env._instance_id, "run_id": env._run_id,
                "episode_number": env._episode_number, "step_id": env._step_id,
                "observation": env._observation, "seed": env._seed,
                "done": env._done, "failed": env._failed,
                "driver_has_advanced": env._driver._has_advanced,
            },
            "reward_state": runtime.boundary_state.value,
            "phase": "reset-committed" if env._step_id == 0 else "decision-committed",
        })

    def validate_payload(self, state: dict, policy_context: dict, profile: Any) -> None:
        exact_fields(state, {"descriptor", "environment", "reward_state", "phase"},
                     "P0 boundary state")
        descriptor = _check_descriptor(state["descriptor"])
        env = exact_fields(state["environment"], legacy_values.ENV_FIELDS, "P0 environment")
        exact_fields(state["reward_state"], set(), "P0 empty reward state")
        try:
            legacy_values.policy_context(policy_context)
            legacy_values.integer(env["episode_number"], "episode_number", 1)
            legacy_values.integer(env["seed"], "seed")
            step = legacy_values.integer(env["step_id"], "step_id", 0,
                                         descriptor["max_steps"] - 1)
            legacy_values.fields(env["observation"], set(legacy_values.OBSERVATION_FIELDS),
                                 "committed observation")
        except legacy_values.QueueSnapshotError as exc:
            raise ContinuationError("CC_INVALID_PAYLOAD", str(exc)) from exc
        if env["instance_id"] != descriptor["instance_id"] or env["run_id"] != descriptor["run_id"]:
            fail("boundary descriptor and environment identity differ")
        if env["done"] is not False or env["failed"] is not False:
            fail("P0 capture requires a nonterminal, nonfailed state", "CC_INVALID_BOUNDARY")
        if type(env["driver_has_advanced"]) is not bool or env["driver_has_advanced"] != (step > 0):
            fail("driver flag and completed step cursor differ")
        phase = "reset-committed" if step == 0 else "decision-committed"
        if state["phase"] != phase:
            fail("RL phase differs from completed step cursor", "CC_INVALID_BOUNDARY")

    def inspect(self, runtime: Any, engine_view: dict) -> tuple[Violation, ...]:
        try:
            env = runtime.env
            if type(env) is not PyJevSimEnv or any(getattr(env, flag) for flag in _FLAGS) or env._done:
                fail("RL reset/step/close is incomplete or episode is not live", "CC_INVALID_BOUNDARY")
            if type(env._boundary) is not FixedDeltaBoundary or type(env._driver) is not ExecutorDriver:
                fail("P0 supports only its fixed-delta environment", "CC_UNSUPPORTED_PROFILE")
            if (env._driver.executor is not runtime.engine or env._binding is not runtime.binding
                    or runtime.binding.executor is not runtime.engine
                    or env._continuation_boundary_state is not runtime.boundary_state):
                fail("boundary ownership differs", "CC_INVALID_BOUNDARY")
            if (frozenset(vars(env)) != env._continuation_shape
                    or vars(runtime.binding) != env._continuation_binding_values
                    or env._factory is not env._continuation_factory_identity
                    or env._plugin_version != PROFILE_ID or env._executor_qualification is not None
                    or env._require_claim_grade or env._driver._closed):
                fail("P0 environment/binding configuration changed", "CC_UNSUPPORTED_PROFILE")
            state = self.export_state(runtime)
            self.validate_payload(state, runtime.metadata.get("policy_context", {}), None)
            if env._observation["logical_time"] != engine_view["executor"]["global_time"]:
                fail("RL cache and engine clock differ", "CC_INVALID_BOUNDARY")
        except ContinuationError as exc:
            return (Violation(exc.code, "boundary.inspect", "boundary", "OB-Q-08", str(exc)),)
        return ()

    def restore_into(self, env: PyJevSimEnv, boundary_state: BoundaryStateHandle,
                     state: dict) -> None:
        descriptor = _check_descriptor(state["descriptor"])
        if (env._boundary.delta != descriptor["delta"] or env._max_steps != descriptor["max_steps"]
                or env._run_id != descriptor["run_id"]):
            fail("allocated boundary differs from payload")
        values = _detached(state["environment"])
        # Instance override was explicitly supplied to bind_uninitialized. All
        # logical state, including run/seed/cursor/reward baseline, is preserved.
        for source, destination in (
            ("episode_number", "_episode_number"), ("step_id", "_step_id"),
            ("observation", "_observation"), ("seed", "_seed"),
            ("done", "_done"), ("failed", "_failed"),
        ):
            setattr(env, destination, values[source])
        env._driver._has_advanced = values["driver_has_advanced"]
        boundary_state.value.clear()
        boundary_state.value.update(_detached(state["reward_state"]))

    def validate_restored(self, env: PyJevSimEnv, model_view: dict, engine_view: dict,
                          *, expected_state: dict, branch: Any = None) -> None:
        if env._observation["logical_time"] != engine_view["executor"]["global_time"]:
            fail("restored RL clock differs from engine", "CC_CONFORMANCE_FAILED")
        if any(getattr(env, flag) for flag in _FLAGS) or env._done:
            fail("restored RL state is not a completed live boundary", "CC_INVALID_BOUNDARY")
        if env._continuation_boundary_state.value != {}:
            fail("P0 has no hidden reward accumulator", "CC_UNSUPPORTED_PROFILE")
        if (env._driver.executor is not env._binding.executor
                or type(env._boundary) is not FixedDeltaBoundary
                or env._driver._has_advanced != (env._step_id > 0)
                or not 0 <= env._step_id < env._max_steps
                or vars(env._binding) != env._continuation_binding_values):
            fail("restored boundary ownership/cursor differs", "CC_CONFORMANCE_FAILED")
        expected = _detached(expected_state)
        if branch is not None:
            # These two physical provenance fields are the only allowed delta.
            # Logical run/seed/step/reward/phase/horizon must remain exact.
            expected["descriptor"]["instance_id"] = branch.instance_id
            expected["environment"]["instance_id"] = branch.instance_id
        actual = self.export_state(SimpleNamespace(
            env=env, boundary_state=env._continuation_boundary_state,
        ))
        if canonical_bytes(actual) != canonical_bytes(expected):
            fail("restored boundary differs from the saved committed state", "CC_CONFORMANCE_FAILED")
        # ModelAdapter.validate_composition owns the pure observation/domain
        # cross-check. It runs again on exported values before publication.
