"""Dependency-free contracts for the pyjevsim reinforcement-learning bridge.

The real :mod:`pyjevsim` package is deliberately not imported here.  Runtime
objects are accepted structurally, which keeps model contract tests usable in
environments where the optional pyjevsim dependency is not installed.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable


class RLEnvironmentError(RuntimeError):
    """Base class for deterministic environment lifecycle failures."""


class StepViewSnapshotError(RLEnvironmentError):
    """A value cannot be represented by the immutable ``StepView`` contract."""

    def __init__(self, message: str, *, path: str, value: object) -> None:
        self.path = path
        self.value_type = type(value).__qualname__
        super().__init__(message)


_IMMUTABLE_ATOMS = (type(None), bool, int, float, complex, str, bytes)
_NOT_IMMUTABLE = object()


@dataclass(frozen=True, slots=True)
class ArraySnapshot:
    """Canonical immutable C-order bytes for a numeric array-like value.

    Callbacks receive this dependency-free representation instead of a
    mutable ndarray alias. Object arrays are intentionally outside the
    supported value-semantic domain.
    """

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        if type(self.dtype) is not str or not self.dtype:
            raise ValueError("ArraySnapshot dtype must be a non-empty str")
        if type(self.shape) is not tuple or any(
            type(dimension) is not int or dimension < 0
            for dimension in self.shape
        ):
            raise ValueError(
                "ArraySnapshot shape must be a tuple of non-negative ints"
            )
        if type(self.data) is not bytes:
            raise ValueError("ArraySnapshot data must be bytes")


@dataclass(frozen=True, slots=True)
class EnumSnapshot:
    """Value-semantic enum identity detached from mutable member sidecars."""

    enum_type: str
    name: str
    value: object

    def __post_init__(self) -> None:
        if type(self.enum_type) is not str or not self.enum_type:
            raise ValueError("EnumSnapshot enum_type must be a non-empty str")
        if type(self.name) is not str or not self.name:
            raise ValueError("EnumSnapshot name must be a non-empty str")
        object.__setattr__(
            self,
            "value",
            _snapshot(self.value, "EnumSnapshot.value"),
        )


def _canonical_immutable(value: object, path: str) -> object:
    """Detach supported immutable subclasses from identity and sidecars."""

    value_type = type(value)
    if value_type in _IMMUTABLE_ATOMS:
        return value
    if isinstance(value, ArraySnapshot):
        try:
            return ArraySnapshot(
                dtype=value.dtype,
                shape=value.shape,
                data=value.data,
            )
        except Exception as exc:
            raise StepViewSnapshotError(
                f"array snapshot reconstruction failed at {path}: {exc}",
                path=path,
                value=value,
            ) from exc
    if isinstance(value, EnumSnapshot):
        try:
            return EnumSnapshot(
                enum_type=value.enum_type,
                name=value.name,
                value=value.value,
            )
        except Exception as exc:
            raise StepViewSnapshotError(
                f"enum snapshot reconstruction failed at {path}: {exc}",
                path=path,
                value=value,
            ) from exc
    canonical: object
    atom_type: type[object]
    try:
        if isinstance(value, bool):
            canonical, atom_type = bool(value), bool
        elif isinstance(value, int):
            canonical, atom_type = int(value), int
        elif isinstance(value, float):
            canonical, atom_type = float(value), float
        elif isinstance(value, complex):
            canonical, atom_type = complex(value), complex
        elif isinstance(value, str):
            canonical, atom_type = str(value), str
        elif isinstance(value, bytes):
            canonical, atom_type = bytes(value), bytes
        else:
            return _NOT_IMMUTABLE
    except Exception as exc:
        raise StepViewSnapshotError(
            f"immutable subclass conversion failed at {path}: {exc}",
            path=path,
            value=value,
        ) from exc
    if type(canonical) is not atom_type:
        raise StepViewSnapshotError(
            f"immutable subclass retained identity at {path}",
            path=path,
            value=value,
        )
    return canonical


def _snapshot_enum(value: Enum, path: str) -> EnumSnapshot:
    enum_type = type(value)
    try:
        type_name = f"{enum_type.__module__}.{enum_type.__qualname__}"
        name = value.name
        enum_value = value.value
    except Exception as exc:
        raise StepViewSnapshotError(
            f"enum inspection failed at {path}: {exc}",
            path=path,
            value=value,
        ) from exc
    try:
        return EnumSnapshot(
            enum_type=type_name,
            name=name,
            value=_snapshot(enum_value, f"{path}.value"),
        )
    except StepViewSnapshotError:
        raise
    except Exception as exc:
        raise StepViewSnapshotError(
            f"enum snapshot failed at {path}: {exc}",
            path=path,
            value=value,
        ) from exc


def _snapshot_key(key: object, path: str) -> object:
    """Freeze a value-semantic mapping key or reject identity/mutable keys."""

    if isinstance(key, Enum):
        frozen_enum = _snapshot_enum(key, path)
        try:
            hash(frozen_enum)
        except TypeError as exc:
            raise StepViewSnapshotError(
                f"enum mapping key is not value-hashable at {path}",
                path=path,
                value=key,
            ) from exc
        return frozen_enum
    frozen_immutable = _canonical_immutable(key, path)
    if frozen_immutable is not _NOT_IMMUTABLE:
        try:
            hash(frozen_immutable)
        except TypeError as exc:
            raise StepViewSnapshotError(
                f"canonical mapping key is not hashable at {path}",
                path=path,
                value=key,
            ) from exc
        return frozen_immutable
    if isinstance(key, tuple):
        return tuple(
            _snapshot_key(item, f"{path}[{index}]")
            for index, item in enumerate(key)
        )
    if isinstance(key, frozenset):
        frozen_items: set[object] = set()
        for index, item in enumerate(key):
            item_path = f"{path}{{item[{index}]}}"
            frozen_item = _snapshot_key(item, item_path)
            if frozen_item in frozen_items:
                raise StepViewSnapshotError(
                    f"canonical frozenset key collision at {item_path}",
                    path=item_path,
                    value=item,
                )
            frozen_items.add(frozen_item)
        return frozenset(frozen_items)
    raise StepViewSnapshotError(
        f"unsupported mutable or identity-based mapping key at {path}: "
        f"{type(key).__qualname__}",
        path=path,
        value=key,
    )


def _snapshot_array(value: object, path: str) -> ArraySnapshot:
    """Canonicalize a numeric ndarray-like object without importing NumPy."""

    try:
        dtype_object = value.dtype  # type: ignore[attr-defined]
        dtype = str(dtype_object)
        kind = dtype_object.kind
        has_object = bool(dtype_object.hasobject)
        raw_itemsize = dtype_object.itemsize
        itemsize = operator.index(raw_itemsize)
        raw_shape = value.shape  # type: ignore[attr-defined]
        tobytes = value.tobytes  # type: ignore[attr-defined]
    except Exception as exc:
        raise StepViewSnapshotError(
            f"array-like metadata inspection failed at {path}: {exc}",
            path=path,
            value=value,
        ) from exc
    if not dtype or not isinstance(kind, str) or has_object or kind == "O":
        raise StepViewSnapshotError(
            f"object or invalid dtype is unsupported at {path}: {dtype!r}",
            path=path,
            value=value,
        )
    if isinstance(raw_itemsize, bool) or itemsize <= 0:
        raise StepViewSnapshotError(
            f"invalid dtype itemsize at {path}: {itemsize!r}",
            path=path,
            value=value,
        )
    try:
        shape = tuple(
            operator.index(dimension)
            if not isinstance(dimension, bool)
            else -1
            for dimension in raw_shape
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise StepViewSnapshotError(
            f"invalid array shape at {path}: {raw_shape!r}",
            path=path,
            value=value,
        ) from exc
    if any(isinstance(dimension, bool) or dimension < 0 for dimension in shape):
        raise StepViewSnapshotError(
            f"invalid array shape at {path}: {shape!r}",
            path=path,
            value=value,
        )
    if not callable(tobytes):
        raise StepViewSnapshotError(
            f"array-like value at {path} does not provide tobytes()",
            path=path,
            value=value,
        )
    try:
        raw_data = tobytes(order="C")
        if not isinstance(raw_data, (bytes, bytearray, memoryview)):
            raise TypeError("tobytes() did not return a bytes-like value")
        data = bytes(raw_data)
    except Exception as exc:
        raise StepViewSnapshotError(
            f"array-like byte snapshot failed at {path}: {exc}",
            path=path,
            value=value,
        ) from exc
    expected_size = math.prod(shape) * itemsize
    if len(data) != expected_size:
        raise StepViewSnapshotError(
            f"array byte size mismatch at {path}: "
            f"expected {expected_size}, got {len(data)}",
            path=path,
            value=value,
        )
    return ArraySnapshot(dtype=dtype, shape=shape, data=data)


def _snapshot(value: object, path: str = "value") -> object:
    """Build a deep immutable snapshot for plugin callbacks.

    The accepted domain is deliberately value-semantic: scalar atoms, enums,
    mappings with value-semantic keys, sequences, sets, byte buffers, and
    numeric NumPy-compatible arrays canonicalized as dtype/shape/C-order bytes.
    Arbitrary custom objects are rejected instead of being presented as
    immutable when ``deepcopy`` would still leave a mutable object graph.
    """

    if isinstance(value, Mapping):
        frozen: dict[object, object] = {}
        for index, (key, item) in enumerate(value.items()):
            key_path = f"{path}.key[{index}]"
            frozen_key = _snapshot_key(key, key_path)
            if frozen_key in frozen:
                raise StepViewSnapshotError(
                    f"canonical mapping key collision at {key_path}: "
                    f"{frozen_key!r}",
                    path=key_path,
                    value=key,
                )
            frozen[frozen_key] = _snapshot(item, f"{path}[{frozen_key!r}]")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _snapshot(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        frozen_items: set[object] = set()
        for index, item in enumerate(value):
            item_path = f"{path}{{item[{index}]}}"
            frozen_item = _snapshot(item, item_path)
            try:
                duplicate = frozen_item in frozen_items
            except TypeError as exc:
                raise StepViewSnapshotError(
                    f"canonical set element is not hashable at {item_path}",
                    path=item_path,
                    value=item,
                ) from exc
            if duplicate:
                raise StepViewSnapshotError(
                    f"canonical set element collision at {item_path}",
                    path=item_path,
                    value=item,
                )
            frozen_items.add(frozen_item)
        return frozenset(frozen_items)
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Enum):
        return _snapshot_enum(value, path)
    frozen_immutable = _canonical_immutable(value, path)
    if frozen_immutable is not _NOT_IMMUTABLE:
        return frozen_immutable
    try:
        is_array_like = all(
            hasattr(value, name)
            for name in ("__array_interface__", "dtype", "shape")
        )
    except Exception as exc:
        raise StepViewSnapshotError(
            f"array-like inspection failed at {path}: {exc}",
            path=path,
            value=value,
        ) from exc
    if is_array_like:
        return _snapshot_array(value, path)
    raise StepViewSnapshotError(
        f"unsupported mutable value at {path}: {type(value).__qualname__}",
        path=path,
        value=value,
    )


@runtime_checkable
class ExecutorProtocol(Protocol):
    """Canonical subset of ``pyjevsim.SysExecutor`` used by the RL bridge."""

    def get_global_time(self) -> float:
        """Return the last committed simulation time."""
        ...

    def get_next_event_time(self) -> float:
        """Return the next scheduled simulation-event time."""
        ...

    def insert_external_event(
        self,
        port: str,
        payload: object,
        scheduled_time: float = 0,
    ) -> None:
        """Schedule an external event for a model-facing executor port."""
        ...

    def step(self, granted_time: float) -> object:
        """Commit every event at or before ``granted_time`` and return outputs."""
        ...

    def is_terminated(self) -> bool:
        """Report whether the simulation has reached its domain terminal state."""
        ...

    def terminate_simulation(self) -> None:
        """Release executor resources."""
        ...


@dataclass(frozen=True)
class EpisodeContext:
    """Immutable inputs supplied to the model factory for one reset."""

    episode_id: str
    instance_id: str
    seed: int | None
    options: Mapping[str, object]


@dataclass(frozen=True)
class StepView:
    """Immutable committed simulation view passed to reward and info bindings."""

    episode_id: str
    instance_id: str
    step_id: int
    previous_logical_time: float
    logical_time: float
    action: object
    previous_observation: object
    observation: object
    external_events: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _snapshot(self.action, "action"))
        object.__setattr__(
            self,
            "previous_observation",
            _snapshot(self.previous_observation, "previous_observation"),
        )
        object.__setattr__(
            self, "observation", _snapshot(self.observation, "observation")
        )
        object.__setattr__(
            self,
            "external_events",
            _snapshot(self.external_events, "external_events"),
        )


@runtime_checkable
class EpisodeBinding(Protocol):
    """Connect one freshly built executor graph to the generic environment."""

    executor: ExecutorProtocol

    def apply_action(self, action: object) -> None:
        """Validate and translate an action into executor external events."""
        ...

    def next_decision_time(self) -> float:
        """Return a plugin-selected decision boundary when requested."""
        ...

    def observe(self, external_events: object) -> object:
        """Construct an observation from committed executor outputs."""
        ...

    def reward(self, transition: StepView) -> float:
        """Compute the reward for a committed step."""
        ...

    def terminated(self, transition: StepView) -> bool:
        """Evaluate the model-domain terminal condition."""
        ...

    def info(self, transition: StepView) -> Mapping[str, object]:
        """Return model-specific, non-authoritative diagnostic information."""
        ...

    def close(self) -> None:
        """Release binding-owned resources."""
        ...


class EpisodeFactory(Protocol):
    """Build a new binding, executor, and model graph for every episode."""

    def __call__(self, context: EpisodeContext) -> EpisodeBinding:
        """Create a fresh episode binding."""
        ...


class EnvironmentClosedError(RLEnvironmentError):
    """An operation was attempted after the environment was closed."""


class EpisodeStateError(RLEnvironmentError):
    """An operation is invalid for the current episode lifecycle state."""


class EpisodeLifecycleError(RLEnvironmentError):
    """A typed reset/step/close failure with an explicit lifecycle boundary."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str,
        phase: str,
        completion_boundary: str,
        instance_id: str,
        episode_id: str | None,
        step_id: int,
        logical_time: float | None,
        recoverable: bool,
        plugin_version: str | None = None,
        causes: tuple[BaseException, ...] = (),
        failed_resources: tuple[str, ...] = (),
    ) -> None:
        self.run_id = run_id
        self.phase = phase
        self.completion_boundary = completion_boundary
        self.instance_id = instance_id
        self.episode_id = episode_id
        self.step_id = step_id
        self.logical_time = logical_time
        self.recoverable = recoverable
        self.plugin_version = plugin_version
        self.causes = causes
        self.failed_resources = failed_resources
        super().__init__(message)


class EpisodeCleanupError(EpisodeLifecycleError):
    """Episode cleanup is incomplete and retained resources may be retried."""


class EpisodeStepError(EpisodeLifecycleError):
    """A step failed with an explicit simulation completion boundary."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str,
        phase: str,
        completion_boundary: str,
        instance_id: str,
        episode_id: str,
        step_id: int,
        logical_time: float | None,
        plugin_version: str | None = None,
        causes: tuple[BaseException, ...] = (),
        failed_resources: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            message,
            run_id=run_id,
            phase=phase,
            completion_boundary=completion_boundary,
            instance_id=instance_id,
            episode_id=episode_id,
            step_id=step_id,
            logical_time=logical_time,
            recoverable=False,
            plugin_version=plugin_version,
            causes=causes,
            failed_resources=failed_resources,
        )


class ExecutorContractError(RLEnvironmentError):
    """An executor violated the monotonic-time or finite-time contract."""
