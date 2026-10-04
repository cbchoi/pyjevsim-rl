"""Declared fixed-delta RL state, independent of any domain model.

Validators are trusted installed, source-bound pure functions. Their registration
does not discover hidden state or certify a model's physical observation/reward
relation; that relation belongs to ModelStateAdapter.validate_composition.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, SimpleNamespace

from .. import adapters, environment, executor
from ..adapters import FunctionalEpisodeBinding
from ..environment import PyJevSimEnv
from ..executor import ExecutorDriver, FixedDeltaBoundary
from .contracts import (
    BoundaryStateHandle, ContinuationError, Violation, canonical_bytes, checked_id,
    checked_sha, exact_fields, fail, freeze_value, normalize_owned, thaw_value,
)
from .registry import SourceBinding

PROVIDER_ID = "declared-fixed-delta-boundary-v1"
VERSION = "1"
MAX_STEPS = 100_000
_DESCRIPTOR = {"schema_id", "kind", "delta", "max_steps", "instance_id", "run_id"}
_ENVIRONMENT = {"instance_id", "run_id", "episode_number", "step_id", "observation",
                "seed", "done", "failed", "driver_has_advanced"}
_SAMPLING = {"domain", "phase", "master", "segment", "logical_branch_id", "run_id",
             "generation", "worker_id", "episode_id", "sampling_seed"}
_FLAGS = ("_failed", "_closed", "_closing", "_disposing", "_constructing",
          "_construction_cleanup_started", "_stepping", "_close_requested")
_IMPLEMENTATIONS = {
    cls: {name: (getattr(cls, name), getattr(getattr(cls, name), "__code__", None))
          for name, member in vars(cls).items()
          if callable(member) and (not name.startswith("__") or name == "__init__")}
    for cls in (PyJevSimEnv, ExecutorDriver, FixedDeltaBoundary, FunctionalEpisodeBinding)
}


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# Pin this module at import as well as registration: a long-lived interpreter
# must not pair its old loaded implementation with newly edited file contents.
_IMPORT_SHA = _sha(__file__)


def _detached(value):
    return normalize_owned(value).to_plain()


def _integer(value, label, minimum=0, maximum=2**256 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        fail(f"{label} is outside the declared integer range")
    return value


def _number(value, label, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        fail(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def validate_policy_context(value):
    exact_fields(value, {"policy_sha256", "policy_version", "feature_contract_sha256"}, "policy context")
    checked_sha(value["policy_sha256"], "policy sha")
    checked_sha(value["feature_contract_sha256"], "feature contract sha")
    _integer(value["policy_version"], "policy version")


def validate_sampling_context(value, run_id=None):
    exact_fields(value, _SAMPLING, "sampling context")
    for name in ("domain", "phase", "segment", "logical_branch_id", "run_id", "worker_id", "episode_id"):
        checked_id(value[name], name)
    for name in ("master", "generation", "sampling_seed"):
        _integer(value[name], name)
    if value["domain"] != "pyjevsim-live-branch-v1" or value["segment"] not in ("prefix", "suffix"):
        fail("unsupported declared sampling domain/segment")
    if run_id is not None and value["run_id"] != run_id:
        fail("sampling and environment run IDs differ")
    if value["segment"] == "prefix" and value["logical_branch_id"] != "prefix":
        fail("prefix sampling requires prefix identity")


def _validator_fingerprint(function):
    # Stateful callable objects, bound methods and lexical cells require a
    # separately declared state owner; this first generic profile rejects them.
    if type(function) is not FunctionType or function.__closure__:
        fail("validator must be a source-backed function without closure cells", "CC_UNSUPPORTED_PROFILE")
    path = inspect.getsourcefile(function)
    if path is None or not Path(path).is_file():
        fail("validator source is unavailable", "CC_UNSUPPORTED_PROFILE")
    return (function, function.__code__, canonical_bytes({
        "defaults": function.__defaults__, "keyword_defaults": function.__kwdefaults__,
    }))


def _validate_with(function, value, label):
    if type(value) is not dict:
        fail(f"{label} must be a declared object")
    owned = normalize_owned(value)
    detached = owned.to_plain()
    before = owned.canonical_bytes
    try:
        result = function(detached)
    except ContinuationError:
        raise
    except Exception as exc:
        raise ContinuationError("CC_INVALID_PAYLOAD", f"{label} validator rejected its value") from exc
    if result is not None:
        fail(f"{label} validator must raise on failure and return None")
    if canonical_bytes(detached) != before:
        fail(f"{label} validator mutated its input")


@dataclass(frozen=True, slots=True)
class _Declaration:
    profile_id: str
    schema_id: str
    observation_validator: object
    reward_validator: object
    initial_reward_state: object


def _stamp(env, state):
    env._continuation_boundary_state = state
    env._continuation_binding_values = dict(vars(env._binding))
    env._continuation_factory_identity = env._factory
    env._continuation_driver_shape = frozenset(vars(env._driver))
    env._continuation_shape = frozenset((*vars(env), "_continuation_shape"))


class DeclaredFixedDeltaBoundaryProvider:
    def __init__(self, *, profile_id, schema_id, observation_validator, reward_validator,
                 initial_reward_state, source_bindings=()):
        checked_id(profile_id, "profile ID")
        checked_id(schema_id, "boundary schema")
        self._validator_identities = (
            _validator_fingerprint(observation_validator), _validator_fingerprint(reward_validator),
        )
        _validate_with(reward_validator, initial_reward_state, "initial reward state")
        self._declaration = _Declaration(profile_id, schema_id, observation_validator,
                                          reward_validator, freeze_value(initial_reward_state))
        self._declared_identity = self._declaration
        self._configuration_bytes = canonical_bytes(self.configuration_identity())
        files = {
            "continuation.generic_boundary": (Path(__file__), _IMPORT_SHA),
            "rl.environment": (Path(environment.__file__), _sha(environment.__file__)),
            "rl.executor": (Path(executor.__file__), _sha(executor.__file__)),
            "rl.adapters": (Path(adapters.__file__), _sha(adapters.__file__)),
        }
        for label, function in (("observation", observation_validator), ("reward", reward_validator)):
            path = Path(inspect.getsourcefile(function)).resolve()
            files[f"boundary.validator.{label}"] = (path, _sha(path))
        bindings = [SourceBinding(name, str(path.resolve()), sha) for name, (path, sha) in files.items()]
        for source in source_bindings:
            if type(source) is not SourceBinding or source.logical_id in files:
                fail("invalid or overlapping declared source binding")
            bindings.append(source)
        if len({source.logical_id for source in bindings}) != len(bindings):
            fail("duplicate boundary source identity")
        self.source_bindings = tuple(bindings)
        self.verify_identity()

    @property
    def profile_id(self):
        return self._declaration.profile_id

    @property
    def schema_id(self):
        return self._declaration.schema_id

    @property
    def initial_reward_state(self):
        """Detached configuration value; each runtime receives a fresh container."""
        return thaw_value(self._declaration.initial_reward_state)

    def configuration_identity(self):
        """Portable declaration identity; no process addresses or callback repr."""
        def identity(function):
            return {"module": function.__module__, "qualname": function.__qualname__,
                    "defaults": function.__defaults__, "keyword_defaults": function.__kwdefaults__}
        return _detached({"profile_id": self.profile_id, "schema_id": self.schema_id,
            "observation_validator": identity(self._declaration.observation_validator),
            "reward_validator": identity(self._declaration.reward_validator),
            "initial_reward_state": self.initial_reward_state})

    def verify_source_bytes(self):
        for source in self.source_bindings:
            if _sha(source.path) != source.sha256:
                fail(f"boundary source changed: {source.logical_id}", "CC_INCOMPATIBLE_IDENTITY")

    def verify_loaded_identity(self, *, part="all"):
        # The two parts preserve the existing source-read boundary inside the
        # legacy shim. Calling this public helper normally checks both parts.
        if part not in ("all", "declaration", "implementation"):
            fail("unknown loaded identity part", "CC_INCOMPATIBLE_IDENTITY")
        if part in ("all", "declaration"):
            if self._declaration is not self._declared_identity:
                fail("declared boundary configuration replaced", "CC_INCOMPATIBLE_IDENTITY")
            if canonical_bytes(self.configuration_identity()) != self._configuration_bytes:
                fail("declared boundary configuration identity changed", "CC_INCOMPATIBLE_IDENTITY")
            functions = (self._declaration.observation_validator, self._declaration.reward_validator)
            if tuple(_validator_fingerprint(function) for function in functions) != self._validator_identities:
                fail("declared validator implementation changed", "CC_INCOMPATIBLE_IDENTITY")
        if part in ("all", "implementation"):
            for cls, methods in _IMPLEMENTATIONS.items():
                if any(getattr(cls, name) is not function or getattr(function, "__code__", None) is not code
                       for name, (function, code) in methods.items()):
                    fail("installed RL implementation changed", "CC_INCOMPATIBLE_IDENTITY")

    def verify_identity(self):
        self.verify_loaded_identity(part="declaration")
        self.verify_source_bytes()
        self.verify_loaded_identity(part="implementation")

    def _descriptor(self, value):
        exact_fields(value, _DESCRIPTOR, "declared fixed-delta descriptor")
        if value["schema_id"] != self.schema_id or value["kind"] != "fixed-delta":
            fail("boundary descriptor differs from registration", "CC_UNSUPPORTED_PROFILE")
        _number(value["delta"], "delta", positive=True)
        _integer(value["max_steps"], "max_steps", 1, MAX_STEPS)
        checked_id(value["instance_id"], "instance ID")
        checked_id(value["run_id"], "run ID")
        return value

    def cursor(self, runtime):
        return runtime.env._step_id

    def validate_branch(self, saved_logical, branch):
        previous = thaw_value(saved_logical["sampling_context"])
        requested = thaw_value(branch.sampling_context)
        validate_sampling_context(previous)
        validate_sampling_context(requested)
        fixed = ("domain", "phase", "master", "run_id", "generation")
        if any(previous[key] != requested[key] for key in fixed):
            fail("branch changes fixed sampling lineage", "CC_INCOMPATIBLE_IDENTITY")
        if requested["segment"] != "suffix" or requested["logical_branch_id"] != branch.branch_id:
            fail("branch sampling does not identify the requested suffix", "CC_INCOMPATIBLE_IDENTITY")

    def descriptor_from_request(self, request):
        if request.profile_id != self.profile_id:
            fail("reset profile differs from boundary registration", "CC_UNSUPPORTED_PROFILE")
        return self._descriptor({"schema_id": self.schema_id, "kind": "fixed-delta",
            "delta": request.delta, "max_steps": request.max_steps,
            "instance_id": request.instance_id, "run_id": request.run_id})

    def allocate_state(self, descriptor, cleanup):
        self._descriptor(descriptor)
        return BoundaryStateHandle(self.initial_reward_state)

    def create_fresh(self, binding, descriptor, seed, cleanup, boundary_state):
        descriptor = self._descriptor(descriptor)
        consumed = False
        def fresh_factory(_context):
            nonlocal consumed
            if consumed:
                fail("declared continuation runtime cannot reset twice", "CC_INVALID_BOUNDARY")
            consumed = True
            return binding
        env = PyJevSimEnv(fresh_factory, instance_id=descriptor["instance_id"],
            run_id=descriptor["run_id"], boundary=FixedDeltaBoundary(descriptor["delta"]),
            max_steps=descriptor["max_steps"], plugin_version=self.profile_id)
        cleanup.add("boundary-env", env.close)
        env.reset(seed=seed)
        _stamp(env, boundary_state)
        return env

    def bind_uninitialized(self, binding, engine, boundary_state, descriptor, cleanup, *, branch=None):
        descriptor = self._descriptor(descriptor)
        def no_reset(_context):
            fail("restored declared runtime cannot reset", "CC_INVALID_BOUNDARY")
        env = PyJevSimEnv(no_reset,
            instance_id=descriptor["instance_id"] if branch is None else branch.instance_id,
            run_id=descriptor["run_id"], boundary=FixedDeltaBoundary(descriptor["delta"]),
            max_steps=descriptor["max_steps"], plugin_version=self.profile_id)
        cleanup.add("boundary-env", env.close)
        env._binding = binding
        env._driver = ExecutorDriver(engine)
        _stamp(env, boundary_state)
        return env

    def export_state(self, runtime):
        env = runtime.env
        return _detached({
            "descriptor": {"schema_id": self.schema_id, "kind": "fixed-delta",
                "delta": env._boundary.delta, "max_steps": env._max_steps,
                "instance_id": env._instance_id, "run_id": env._run_id},
            "environment": {"instance_id": env._instance_id, "run_id": env._run_id,
                "episode_number": env._episode_number, "step_id": env._step_id,
                "observation": env._observation, "seed": env._seed, "done": env._done,
                "failed": env._failed, "driver_has_advanced": env._driver._has_advanced},
            "reward_state": runtime.boundary_state.value,
            "phase": "reset-committed" if env._step_id == 0 else "decision-committed",
        })

    def validate_payload(self, state, policy_context, profile):
        exact_fields(state, {"descriptor", "environment", "reward_state", "phase"}, "boundary state")
        descriptor = self._descriptor(state["descriptor"])
        if profile is not None and profile.profile_id != self.profile_id:
            fail("registered boundary profile differs", "CC_INCOMPATIBLE_IDENTITY")
        env = exact_fields(state["environment"], _ENVIRONMENT, "environment state")
        validate_policy_context(policy_context)
        _integer(env["episode_number"], "episode number", 1)
        _integer(env["seed"], "environment seed")
        step = _integer(env["step_id"], "step cursor", 0, descriptor["max_steps"] - 1)
        _validate_with(self._declaration.observation_validator, env["observation"], "observation")
        if "logical_time" not in env["observation"]:
            fail("observation must declare logical_time")
        _number(env["observation"]["logical_time"], "observation clock")
        _validate_with(self._declaration.reward_validator, state["reward_state"], "reward state")
        if env["instance_id"] != descriptor["instance_id"] or env["run_id"] != descriptor["run_id"]:
            fail("environment and descriptor identities differ")
        if env["done"] is not False or env["failed"] is not False:
            fail("boundary must be live and nonfailed", "CC_INVALID_BOUNDARY")
        if type(env["driver_has_advanced"]) is not bool or env["driver_has_advanced"] != (step > 0):
            fail("driver flag and completed cursor differ")
        if state["phase"] != ("reset-committed" if step == 0 else "decision-committed"):
            fail("phase and committed cursor differ", "CC_INVALID_BOUNDARY")

    def inspect(self, runtime, engine_view):
        try:
            env = runtime.env
            if type(env) is not PyJevSimEnv or any(getattr(env, flag) for flag in _FLAGS) or env._done:
                fail("RL lifecycle is not a live completed boundary", "CC_INVALID_BOUNDARY")
            if type(env._boundary) is not FixedDeltaBoundary or type(env._driver) is not ExecutorDriver:
                fail("declared profile requires fixed-delta PyJevSimEnv", "CC_UNSUPPORTED_PROFILE")
            if (env._driver.executor is not runtime.engine or env._binding is not runtime.binding
                    or runtime.binding.executor is not runtime.engine
                    or env._continuation_boundary_state is not runtime.boundary_state):
                fail("candidate boundary ownership differs", "CC_INVALID_BOUNDARY")
            if (frozenset(vars(env)) != env._continuation_shape
                    or frozenset(vars(env._driver)) != env._continuation_driver_shape
                    or vars(runtime.binding) != env._continuation_binding_values
                    or env._factory is not env._continuation_factory_identity
                    or env._plugin_version != self.profile_id or env._executor_qualification is not None
                    or env._require_claim_grade or env._driver._closed
                    or env._driver._qualification_policy is not None or env._driver._semantic_evidence is not None):
                fail("declared boundary configuration changed", "CC_UNSUPPORTED_PROFILE")
            state = self.export_state(runtime)
            self.validate_payload(state, thaw_value(runtime.metadata.get("policy_context", {})), None)
            if env._observation["logical_time"] != engine_view["executor"]["global_time"]:
                fail("committed observation and engine clock differ", "CC_INVALID_BOUNDARY")
        except (ContinuationError, AttributeError, KeyError, TypeError, ValueError) as exc:
            return (Violation(getattr(exc, "code", "CC_UNSUPPORTED_PROFILE"), "boundary.inspect",
                              "boundary", message=str(exc)),)
        return ()

    def restore_into(self, env, boundary_state, state):
        descriptor = self._descriptor(state["descriptor"])
        if (env._boundary.delta != descriptor["delta"] or env._max_steps != descriptor["max_steps"]
                or env._run_id != descriptor["run_id"]
                or env._continuation_boundary_state is not boundary_state):
            fail("allocated boundary differs from declared payload")
        values = _detached(state["environment"])
        reward = _detached(state["reward_state"])
        _validate_with(self._declaration.reward_validator, reward, "reward state")
        for source, destination in (("episode_number", "_episode_number"), ("step_id", "_step_id"),
                ("observation", "_observation"), ("seed", "_seed"), ("done", "_done"), ("failed", "_failed")):
            setattr(env, destination, values[source])
        env._driver._has_advanced = values["driver_has_advanced"]
        boundary_state.value.clear()
        boundary_state.value.update(reward)

    def validate_restored(self, env, model_view, engine_view, *, expected_state, branch=None):
        if env._observation["logical_time"] != engine_view["executor"]["global_time"]:
            fail("restored RL/engine clock mismatch", "CC_CONFORMANCE_FAILED")
        if any(getattr(env, flag) for flag in _FLAGS) or env._done:
            fail("restored RL cut is not live/committed", "CC_INVALID_BOUNDARY")
        if (env._driver.executor is not env._binding.executor
                or type(env._boundary) is not FixedDeltaBoundary
                or env._driver._has_advanced != (env._step_id > 0)
                or not 0 <= env._step_id < env._max_steps
                or vars(env._binding) != env._continuation_binding_values):
            fail("restored boundary ownership/cursor mismatch", "CC_CONFORMANCE_FAILED")
        expected = _detached(expected_state)
        if branch is not None:
            expected["descriptor"]["instance_id"] = branch.instance_id
            expected["environment"]["instance_id"] = branch.instance_id
        actual = self.export_state(SimpleNamespace(env=env, boundary_state=env._continuation_boundary_state))
        if canonical_bytes(actual) != canonical_bytes(expected):
            fail("restored boundary differs from committed state", "CC_CONFORMANCE_FAILED")
