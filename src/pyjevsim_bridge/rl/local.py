"""Deterministic, transport-neutral local rollout orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Protocol, TypeVar, cast

from .local_process import (
    LocalProcessWorkerError,
    ProcessBackendOptions,
    ProcessCleanupReceipt,
    ProcessWorkerSupervisor,
)
from .records import ResetResult, TransitionRecord


class LocalEnvironment(Protocol):
    """The minimal environment surface consumed by :class:`LocalRolloutPool`."""

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, object] | None = None
    ) -> tuple[object, Mapping[str, object]] | Awaitable[tuple[object, Mapping[str, object]]]: ...

    def step(
        self, action: object
    ) -> (
        tuple[object, float, bool, bool, Mapping[str, object]]
        | Awaitable[tuple[object, float, bool, bool, Mapping[str, object]]]
    ): ...

    def close(self) -> None | Awaitable[None]: ...


EnvironmentProvider = LocalEnvironment | Callable[[], LocalEnvironment]
T = TypeVar("T")


class LocalRolloutBatchError(RuntimeError):
    """One or more workers advanced inconsistently; reset is required."""

    def __init__(
        self,
        phase: str,
        errors: Mapping[str, BaseException],
        *,
        returned_results: Mapping[str, object] | None = None,
    ) -> None:
        self.phase = phase
        self.errors = dict(errors)
        # These are unvalidated raw returns, never learner-admissible records.
        # Only the container is copied/frozen: arbitrary model-owned nested
        # objects are not guaranteed to be immutable or serializable.
        self.returned_results: Mapping[str, object] = MappingProxyType(
            dict(returned_results) if returned_results is not None else {}
        )
        details = ", ".join(
            f"{worker_id}: {type(error).__name__}: {error}"
            for worker_id, error in sorted(self.errors.items())
        )
        super().__init__(f"local rollout {phase} failed; reset required ({details})")


class LocalRolloutPoolState(StrEnum):
    """Observable lifecycle state of a local rollout pool."""

    OPEN = "open"
    CLOSING = "closing"
    CLOSE_FAILED = "close-failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class WorkerCloseReceipt:
    """Last terminal close outcome for one worker environment."""

    worker_id: str
    attempt: int
    succeeded: bool
    error: BaseException | None = None


class LocalRolloutCloseError(RuntimeError):
    """Worker-attributed close failure that may be retried explicitly."""

    def __init__(self, receipts: Mapping[str, WorkerCloseReceipt]) -> None:
        self.receipts = dict(receipts)
        self.errors = {
            worker_id: receipt.error
            for worker_id, receipt in self.receipts.items()
            if not receipt.succeeded and receipt.error is not None
        }
        details = ", ".join(
            f"{worker_id}: {type(error).__name__}: {error}"
            for worker_id, error in sorted(self.errors.items())
        )
        super().__init__(f"local rollout close failed ({details})")


class LocalRolloutRecoveryError(RuntimeError):
    """Dirty worker runtimes could not be safely replaced."""

    def __init__(self, phase: str, errors: Mapping[str, BaseException]) -> None:
        self.phase = phase
        self.errors = dict(errors)
        details = ", ".join(
            f"{worker_id}: {type(error).__name__}: {error}"
            for worker_id, error in sorted(self.errors.items())
        )
        super().__init__(f"local rollout recovery {phase} failed ({details})")


@dataclass(slots=True)
class _LocalExecutionContext:
    pool: object
    phase: str
    active: bool = True
    caller_cancelled: bool = False


_LOCAL_EXECUTION: ContextVar[_LocalExecutionContext | None] = ContextVar(
    "pyjevsim_rl_local_execution",
    default=None,
)


def derive_episode_seed(run_seed: int | None, worker_id: str, episode_number: int) -> int | None:
    """Derive a stable 63-bit episode seed.

    Python's built-in ``hash`` is deliberately randomized per process, so the
    derivation uses a versioned, length-framed SHA-256 input.  The result is
    stable across processes, operating systems, Python versions and worker
    completion orders.  A missing run seed remains missing rather than
    silently inventing non-reproducible entropy.
    """

    if run_seed is None:
        return None
    if isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed < 0:
        raise ValueError("run_seed must be a non-negative integer or None")
    if not isinstance(worker_id, str) or not worker_id:
        raise ValueError("worker_id must be a non-empty string")
    if (
        isinstance(episode_number, bool)
        or not isinstance(episode_number, int)
        or episode_number < 0
    ):
        raise ValueError("episode_number must be a non-negative integer")

    parts = (
        b"pyjevsim-rl-seed-v1",
        str(run_seed).encode("ascii"),
        worker_id.encode("utf-8"),
        str(episode_number).encode("ascii"),
    )
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def _idempotency_key(
    run_id: str,
    generation: int,
    worker_id: str,
    episode_id: str,
    step_id: int,
    policy_version: int,
) -> str:
    digest = hashlib.sha256()
    for value in (
        "pyjevsim-rl-transition-v1",
        run_id,
        str(generation),
        worker_id,
        episode_id,
        str(step_id),
        str(policy_version),
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


async def _invoke(method: Callable[..., T], /, *args: object, **kwargs: object) -> T:
    """Call sync model code off-loop while accepting native async test doubles."""

    if inspect.iscoroutinefunction(method):
        return cast(T, await method(*args, **kwargs))
    result = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(result):
        return cast(T, await result)
    return result


def _materialize_environment(provider: EnvironmentProvider) -> LocalEnvironment:
    if all(hasattr(provider, name) for name in ("reset", "step", "close")):
        return cast(LocalEnvironment, provider)
    if not callable(provider):
        raise TypeError("each worker requires an environment or environment factory")
    environment = provider()
    if not all(hasattr(environment, name) for name in ("reset", "step", "close")):
        raise TypeError("environment factory must return reset/step/close methods")
    return environment


class _ProcessEnvironmentProxy:
    """Adapt one supervisor worker to the existing environment call surface."""

    def __init__(self, supervisor: ProcessWorkerSupervisor, worker_id: str) -> None:
        self._supervisor = supervisor
        self._worker_id = worker_id

    async def reset(
        self, *, seed: int | None = None, options: Mapping[str, object] | None = None
    ) -> object:
        if options is not None:
            raise ValueError("process backend reset options are not supported")
        return await self._supervisor.reset(self._worker_id, seed=seed)

    async def step(self, action: object) -> object:
        return await self._supervisor.step(self._worker_id, action)

    async def close(self) -> None:
        await self._supervisor.close_worker(self._worker_id)


class LocalRolloutPool:
    """Run isolated environments concurrently and return canonical worker order.

    ``workers`` maps stable worker IDs to either an already-created environment
    or a zero-argument environment factory.  A distinct environment object is
    required for every worker.  Factories are materialized once; each
    ``PyJevSimEnv.reset`` is responsible for rebuilding its episode binding.
    """

    def __init__(
        self,
        workers: Mapping[str, EnvironmentProvider],
        *,
        run_id: str = "local",
        generation: int = 0,
        run_seed: int | None = None,
        backend: Literal["serial", "thread", "process"] = "thread",
        process_options: ProcessBackendOptions | None = None,
    ) -> None:
        if not isinstance(workers, Mapping) or not workers:
            raise ValueError("workers must be a non-empty mapping")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if run_seed is not None and (
            isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed < 0
        ):
            raise ValueError("run_seed must be a non-negative integer or None")
        if backend not in {"serial", "thread", "process"}:
            raise ValueError("backend must be 'serial', 'thread' or 'process'")
        if backend != "process" and process_options is not None:
            raise ValueError("process_options requires backend='process'")

        worker_ids = tuple(sorted(workers))
        if any(not isinstance(worker_id, str) or not worker_id for worker_id in worker_ids):
            raise ValueError("worker IDs must be non-empty strings")
        environments: dict[str, LocalEnvironment] = {}
        factories: dict[str, Callable[[], LocalEnvironment] | None] = {}
        process_supervisor: ProcessWorkerSupervisor | None = None
        if backend == "process":
            process_factories: dict[str, Callable[[], object]] = {}
            for worker_id in worker_ids:
                provider = workers[worker_id]
                if not callable(provider) or all(
                    hasattr(provider, name) for name in ("reset", "step", "close")
                ):
                    raise TypeError(
                        "process backend requires a zero-argument environment "
                        "factory for every worker"
                    )
                process_factories[worker_id] = provider
                factories[worker_id] = provider
            process_supervisor = ProcessWorkerSupervisor(process_factories, options=process_options)
            environments = {
                worker_id: cast(
                    LocalEnvironment,
                    _ProcessEnvironmentProxy(process_supervisor, worker_id),
                )
                for worker_id in worker_ids
            }
        else:
            for worker_id in worker_ids:
                provider = workers[worker_id]
                if all(hasattr(provider, name) for name in ("reset", "step", "close")):
                    environments[worker_id] = cast(LocalEnvironment, provider)
                    factories[worker_id] = None
                else:
                    if not callable(provider):
                        raise TypeError(
                            "each worker requires an environment or environment factory"
                        )
                    factory = provider
                    environments[worker_id] = _materialize_environment(factory)
                    factories[worker_id] = factory
        identities = [id(environment) for environment in environments.values()]
        if len(identities) != len(set(identities)):
            raise ValueError("each worker must own a distinct environment instance")

        self._worker_ids = worker_ids
        self._backend = backend
        self._process_supervisor = process_supervisor
        self._environments = environments
        self._factories = factories
        self._run_id = run_id
        self._generation = generation
        self._run_seed = run_seed
        self._episode_numbers = dict.fromkeys(worker_ids, -1)
        self._reset_attempt_numbers = dict.fromkeys(worker_ids, -1)
        self._episode_ids: dict[str, str] = {}
        self._committed_episode_ids: dict[str, set[str]] = {
            worker_id: set() for worker_id in worker_ids
        }
        self._step_ids: dict[str, int] = {}
        self._logical_times: dict[str, float] = {}
        self._observations: dict[str, object] = {}
        self._done_workers: set[str] = set()
        self._requires_reset = True
        self._state = LocalRolloutPoolState.OPEN
        self._lifecycle = asyncio.Condition()
        self._active_operations = 0
        self._close_task: asyncio.Task[None] | None = None
        self._close_attempts = dict.fromkeys(worker_ids, 0)
        self._close_receipts: dict[str, WorkerCloseReceipt] = {}
        self._dirty_workers: set[str] = set()
        self._recovery_closed_workers: set[str] = set()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return self._worker_ids

    @property
    def state(self) -> LocalRolloutPoolState:
        return self._state

    @property
    def close_receipts(self) -> Mapping[str, WorkerCloseReceipt]:
        return dict(self._close_receipts)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def process_pids(self) -> Mapping[str, int]:
        if self._process_supervisor is None:
            return {}
        return self._process_supervisor.pids

    @property
    def process_incarnations(self) -> Mapping[str, int]:
        if self._process_supervisor is None:
            return {}
        return self._process_supervisor.incarnations

    @property
    def process_cleanup_receipts(self) -> Mapping[str, ProcessCleanupReceipt]:
        """Latest terminal process receipt per worker; older entries remain in history."""

        if self._process_supervisor is None:
            return {}
        return self._process_supervisor.cleanup_receipts

    @property
    def process_cleanup_history(self) -> tuple[ProcessCleanupReceipt, ...]:
        """Immutable append-only retirement history for process incarnations."""

        if self._process_supervisor is None:
            return ()
        return self._process_supervisor.cleanup_history

    async def _invoke_environment(
        self, method: Callable[..., T], /, *args: object, **kwargs: object
    ) -> T:
        if self._backend != "serial":
            return await _invoke(method, *args, **kwargs)
        # The serial profile must not become a one-worker thread benchmark.
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return cast(T, await result)
        return result

    async def _collect(self, calls: Iterable[Awaitable[T]]) -> list[T | BaseException]:
        if self._backend != "serial":
            return list(await asyncio.gather(*calls, return_exceptions=True))
        results: list[T | BaseException] = []
        # Sequential awaiting also prevents async test doubles from overlapping.
        # Keep every outcome, including cleanup after an earlier worker failed.
        for call in calls:
            try:
                results.append(await call)
            except BaseException as exc:
                results.append(exc)
        return results

    async def _admit_operation(self) -> None:
        async with self._lifecycle:
            if self._state is not LocalRolloutPoolState.OPEN:
                raise RuntimeError(
                    f"local rollout pool does not admit operations while {self._state.value}"
                )
            if self._active_operations:
                raise RuntimeError(
                    "local rollout pool admits only one reset or step operation at a time"
                )
            self._active_operations = 1

    async def _release_operation(self) -> None:
        async with self._lifecycle:
            self._active_operations -= 1
            if self._active_operations == 0:
                self._lifecycle.notify_all()

    async def reset(
        self,
        *,
        seed: int | None = None,
        episode_seeds: Mapping[str, int] | None = None,
    ) -> list[ResetResult]:
        """Reset every worker with derived or explicitly scheduled episode seeds.

        An explicit map overrides only this reset. If ``seed`` is also supplied,
        it remains the run seed for future resets that omit ``episode_seeds``.
        The map is copied and validated before operation admission or mutation.
        """
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None")
        scheduled_seeds: dict[str, int] | None = None
        if episode_seeds is not None:
            if not isinstance(episode_seeds, Mapping):
                raise TypeError("episode_seeds must be a mapping keyed by every worker ID")
            scheduled_seeds = dict(episode_seeds)
            if set(scheduled_seeds) != set(self._worker_ids):
                raise ValueError("episode_seeds must contain exactly every worker ID")
            if any(type(value) is not int or value < 0 for value in scheduled_seeds.values()):
                raise ValueError(
                    "episode_seeds values must be non-negative integers excluding bool"
                )
        await self._admit_operation()
        if seed is not None:
            self._run_seed = seed
        attempt_numbers = {
            worker_id: self._reset_attempt_numbers[worker_id] + 1 for worker_id in self._worker_ids
        }
        self._reset_attempt_numbers.update(attempt_numbers)
        return await self._execute_owned_operation(
            "reset",
            lambda: self._reset(
                attempt_numbers=attempt_numbers, episode_seeds=scheduled_seeds
            ),
        )

    async def _reset(
        self,
        *,
        attempt_numbers: Mapping[str, int],
        episode_seeds: Mapping[str, int] | None = None,
    ) -> list[ResetResult]:
        await self._recover_dirty_workers()

        episode_numbers = {
            worker_id: self._episode_numbers[worker_id] + 1 for worker_id in self._worker_ids
        }
        seeds: dict[str, int | None] = (
            dict(episode_seeds)
            if episode_seeds is not None
            else {
                worker_id: derive_episode_seed(
                    self._run_seed, worker_id, attempt_numbers[worker_id]
                )
                for worker_id in self._worker_ids
            }
        )

        if (
            self._process_supervisor is not None
            and self._process_supervisor.options.dispatch_mode == "batched"
        ):
            raw_results = await self._process_supervisor.reset_batch(seeds)
        else:
            calls = [
                self._invoke_environment(self._environments[worker_id].reset, seed=seeds[worker_id])
                for worker_id in self._worker_ids
            ]
            raw_results = await self._collect(calls)
        returned_results = {
            worker_id: result
            for worker_id, result in zip(self._worker_ids, raw_results, strict=True)
            if not isinstance(result, BaseException)
        }
        call_errors = {
            worker_id: result
            for worker_id, result in zip(self._worker_ids, raw_results, strict=True)
            if isinstance(result, BaseException)
        }
        if call_errors:
            self._requires_reset = True
            self._mark_all_workers_dirty()
            raise LocalRolloutBatchError(
                "reset", call_errors, returned_results=returned_results
            )

        results: list[ResetResult] = []
        reset_logical_times: dict[str, float] = {}
        active_worker = "unknown"
        try:
            for worker_id, raw in zip(self._worker_ids, raw_results, strict=True):
                active_worker = worker_id
                if not isinstance(raw, tuple) or len(raw) != 2:
                    raise TypeError("environment reset must return (observation, info)")
                observation, raw_info = raw
                if not isinstance(raw_info, Mapping):
                    raise TypeError("environment reset info must be a mapping")
                info = dict(raw_info)
                raw_logical_time = info.get("logical_time")
                if isinstance(raw_logical_time, bool) or not isinstance(
                    raw_logical_time, (int, float)
                ):
                    raise TypeError(f"worker {worker_id} reset info logical_time must be numeric")
                logical_time = float(raw_logical_time)
                if not math.isfinite(logical_time):
                    raise ValueError(f"worker {worker_id} reset info logical_time must be finite")
                info["logical_time"] = logical_time
                reset_logical_times[worker_id] = logical_time
                episode_number = episode_numbers[worker_id]
                episode_id = info.get("episode_id")
                if episode_id is None:
                    episode_id = f"{self._run_id}:{worker_id}:{episode_number}"
                if not isinstance(episode_id, str) or not episode_id:
                    raise ValueError("environment reset info episode_id must be non-empty")
                if episode_id in self._committed_episode_ids[worker_id]:
                    raise ValueError(
                        f"worker {worker_id} reused committed episode_id {episode_id!r}"
                    )
                expected = {
                    "run_id": self._run_id,
                    "instance_id": worker_id,
                    "step_id": 0,
                    "seed": seeds[worker_id],
                    "reset_attempt": attempt_numbers[worker_id],
                }
                for key, value in expected.items():
                    if key in info and info[key] != value:
                        raise ValueError(f"worker {worker_id} reset info {key} conflicts with pool")
                    info[key] = value
                info["episode_id"] = episode_id
                results.append(
                    ResetResult(
                        run_id=self._run_id,
                        generation=self._generation,
                        worker_id=worker_id,
                        episode_id=episode_id,
                        seed=seeds[worker_id],
                        observation=observation,
                        info=info,
                    )
                )
        except BaseException as exc:
            self._requires_reset = True
            self._mark_all_workers_dirty()
            raise LocalRolloutBatchError(
                "reset-validation", {active_worker: exc}, returned_results=returned_results
            ) from exc

        for result in results:
            worker_id = result.worker_id
            self._episode_numbers[worker_id] = episode_numbers[worker_id]
            self._episode_ids[worker_id] = result.episode_id
            self._committed_episode_ids[worker_id].add(result.episode_id)
            self._step_ids[worker_id] = 0
            self._logical_times[worker_id] = reset_logical_times[worker_id]
            self._observations[worker_id] = result.observation
        self._done_workers.clear()
        self._dirty_workers.clear()
        self._recovery_closed_workers.clear()
        self._requires_reset = False
        return results

    async def step(
        self, actions: Mapping[str, object], *, policy_version: int
    ) -> list[TransitionRecord]:
        await self._admit_operation()
        return await self._execute_owned_operation(
            "step", lambda: self._step(actions, policy_version=policy_version)
        )

    async def _execute_owned_operation(
        self,
        phase: str,
        body: Callable[[], Awaitable[T]],
    ) -> T:
        execution = _LocalExecutionContext(self, phase)
        token = _LOCAL_EXECUTION.set(execution)
        try:
            owned = asyncio.create_task(self._finish_owned_operation(body(), execution))
        finally:
            _LOCAL_EXECUTION.reset(token)

        first_cancellation: asyncio.CancelledError | None = None
        while not owned.done():
            try:
                await asyncio.shield(owned)
            except asyncio.CancelledError as cancellation:
                self._requires_reset = True
                self._mark_all_workers_dirty()
                execution.caller_cancelled = True
                if first_cancellation is None:
                    first_cancellation = cancellation
            except BaseException:
                # The retained task owns the exception; inspect it below so an
                # earlier caller cancellation can be preserved alongside it.
                break

        try:
            result = owned.result()
        except BaseException as owned_error:
            if first_cancellation is not None:
                self._requires_reset = True
                raise BaseExceptionGroup(
                    f"local rollout {phase} was cancelled and its owned operation also failed",
                    [first_cancellation, owned_error],
                ) from None
            raise
        if first_cancellation is not None:
            self._requires_reset = True
            raise first_cancellation
        return result

    async def _finish_owned_operation(
        self,
        body: Awaitable[T],
        execution: _LocalExecutionContext,
    ) -> T:
        try:
            return await body
        finally:
            if execution.caller_cancelled:
                self._requires_reset = True
                self._mark_all_workers_dirty()
            execution.active = False
            await self._release_operation()

    def _mark_all_workers_dirty(self) -> None:
        self._dirty_workers.update(self._worker_ids)

    async def _recover_dirty_workers(self) -> None:
        if not self._dirty_workers:
            return

        if self._process_supervisor is not None:
            targets = tuple(sorted(self._dirty_workers))
            process_replacement_results = await asyncio.gather(
                *(self._process_supervisor.replace_worker(worker_id) for worker_id in targets),
                return_exceptions=True,
            )
            process_replacement_errors = {
                worker_id: result
                for worker_id, result in zip(targets, process_replacement_results, strict=True)
                if isinstance(result, BaseException)
            }
            if process_replacement_errors:
                self._requires_reset = True
                raise LocalRolloutRecoveryError("replacement", process_replacement_errors)
            self._dirty_workers.clear()
            self._recovery_closed_workers.clear()
            return

        non_recreatable = {
            worker_id: RuntimeError(
                "worker uses a direct environment instance; recovery requires "
                "a zero-argument environment factory"
            )
            for worker_id in sorted(self._dirty_workers)
            if self._factories[worker_id] is None
        }
        if non_recreatable:
            self._requires_reset = True
            raise LocalRolloutRecoveryError("provider", non_recreatable)

        # Snapshot the complete dirty batch before closing any member. A
        # factory must not revive its own retired object or swap in a dirty
        # sibling whose close already succeeded during this recovery attempt.
        retired_ids = {id(self._environments[worker_id]) for worker_id in self._dirty_workers}
        close_targets = tuple(
            worker_id
            for worker_id in sorted(self._dirty_workers)
            if worker_id not in self._recovery_closed_workers
        )
        close_results = await self._collect(
            self._invoke_environment(self._environments[worker_id].close)
            for worker_id in close_targets
        )
        close_errors: dict[str, BaseException] = {}
        for worker_id, close_result in zip(close_targets, close_results, strict=True):
            if isinstance(close_result, BaseException):
                close_errors[worker_id] = close_result
            else:
                self._recovery_closed_workers.add(worker_id)
        if close_errors:
            self._requires_reset = True
            raise LocalRolloutRecoveryError("cleanup", close_errors)

        replacement_targets = tuple(sorted(self._dirty_workers))
        replacement_results = await self._collect(
            self._invoke_environment(
                _materialize_environment,
                cast(Callable[[], LocalEnvironment], self._factories[worker_id]),
            )
            for worker_id in replacement_targets
        )
        replacement_errors: dict[str, BaseException] = {}
        occupied_ids = {
            id(environment)
            for worker_id, environment in self._environments.items()
            if worker_id not in self._dirty_workers
        }
        for worker_id, replacement_result in zip(
            replacement_targets, replacement_results, strict=True
        ):
            if isinstance(replacement_result, BaseException):
                replacement_errors[worker_id] = replacement_result
                continue
            if id(replacement_result) in retired_ids:
                replacement_errors[worker_id] = ValueError(
                    "environment factory returned an instance retired by the dirty recovery batch"
                )
                continue
            if id(replacement_result) in occupied_ids:
                replacement_errors[worker_id] = ValueError(
                    "environment factory reused another worker instance"
                )
                continue
            occupied_ids.add(id(replacement_result))
            self._environments[worker_id] = replacement_result
            self._dirty_workers.discard(worker_id)
            self._recovery_closed_workers.discard(worker_id)
        if replacement_errors:
            self._requires_reset = True
            raise LocalRolloutRecoveryError("replacement", replacement_errors)

    async def _step(
        self, actions: Mapping[str, object], *, policy_version: int
    ) -> list[TransitionRecord]:
        if not isinstance(actions, Mapping):
            raise TypeError("actions must be a mapping keyed by worker ID")
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 0
        ):
            raise ValueError("policy_version must be a non-negative integer")

        expected = set(self._worker_ids)
        supplied = set(actions)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing actions: {missing}")
            if extra:
                details.append(f"extra actions: {extra}")
            raise ValueError("; ".join(details))
        not_reset = [
            worker_id for worker_id in self._worker_ids if worker_id not in self._episode_ids
        ]
        if not_reset:
            raise RuntimeError(f"workers must be reset before step: {not_reset}")
        if self._requires_reset:
            raise RuntimeError("local rollout pool requires reset before step")
        if self._done_workers:
            raise RuntimeError(
                f"terminal workers require reset before step: {sorted(self._done_workers)}"
            )

        calls: list[Awaitable[object]]
        if self._process_supervisor is None:
            calls = [
                self._invoke_environment(self._environments[worker_id].step, actions[worker_id])
                for worker_id in self._worker_ids
            ]
            raw_results = await self._collect(calls)
        else:
            action_payloads: dict[str, bytes] = {}
            preflight_errors: dict[str, BaseException] = {}
            for worker_id in self._worker_ids:
                try:
                    action_payloads[worker_id] = self._process_supervisor.preflight_action(
                        worker_id, actions[worker_id]
                    )
                except LocalProcessWorkerError as exc:
                    preflight_errors[worker_id] = exc
            if preflight_errors:
                raise LocalRolloutBatchError("action-preflight", preflight_errors)
            if self._process_supervisor.options.dispatch_mode == "batched":
                raw_results = await self._process_supervisor.step_batch_preflighted(action_payloads)
            else:
                calls = [
                    self._process_supervisor.step_preflighted(worker_id, action_payloads[worker_id])
                    for worker_id in self._worker_ids
                ]
                raw_results = await self._collect(calls)
        returned_results = {
            worker_id: result
            for worker_id, result in zip(self._worker_ids, raw_results, strict=True)
            if not isinstance(result, BaseException)
        }
        call_errors = {
            worker_id: result
            for worker_id, result in zip(self._worker_ids, raw_results, strict=True)
            if isinstance(result, BaseException)
        }
        if call_errors:
            self._requires_reset = True
            self._mark_all_workers_dirty()
            raise LocalRolloutBatchError(
                "step", call_errors, returned_results=returned_results
            )

        results: list[TransitionRecord] = []
        active_worker = "unknown"
        try:
            for worker_id, raw in zip(self._worker_ids, raw_results, strict=True):
                active_worker = worker_id
                if not isinstance(raw, tuple) or len(raw) != 5:
                    raise TypeError(
                        "environment step must return "
                        "(observation, reward, terminated, truncated, info)"
                    )
                observation, reward, terminated, truncated, raw_info = raw
                if not isinstance(raw_info, Mapping):
                    raise TypeError("environment step info must be a mapping")
                next_step_id = self._step_ids[worker_id] + 1
                episode_id = self._episode_ids[worker_id]
                info = dict(raw_info)
                if "logical_time" not in info:
                    raise ValueError(f"worker {worker_id} step info must contain logical_time")
                raw_logical_time = info["logical_time"]
                if isinstance(raw_logical_time, bool) or not isinstance(
                    raw_logical_time, (int, float)
                ):
                    raise TypeError(f"worker {worker_id} step info logical_time must be numeric")
                logical_time = float(raw_logical_time)
                if not math.isfinite(logical_time):
                    raise ValueError(f"worker {worker_id} step info logical_time must be finite")
                if logical_time < self._logical_times[worker_id]:
                    raise ValueError(
                        f"worker {worker_id} logical_time regressed from "
                        f"{self._logical_times[worker_id]} to {logical_time}"
                    )
                expected_info = {
                    "run_id": self._run_id,
                    "instance_id": worker_id,
                    "episode_id": episode_id,
                    "step_id": next_step_id,
                }
                for key, value in expected_info.items():
                    if key in info and info[key] != value:
                        raise ValueError(f"worker {worker_id} step info {key} conflicts with pool")
                    info[key] = value
                results.append(
                    TransitionRecord(
                        run_id=self._run_id,
                        generation=self._generation,
                        worker_id=worker_id,
                        episode_id=episode_id,
                        step_id=next_step_id,
                        policy_version=policy_version,
                        idempotency_key=_idempotency_key(
                            self._run_id,
                            self._generation,
                            worker_id,
                            episode_id,
                            next_step_id,
                            policy_version,
                        ),
                        logical_time=logical_time,
                        previous_observation=self._observations[worker_id],
                        action=actions[worker_id],
                        next_observation=observation,
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                )
        except BaseException as exc:
            self._requires_reset = True
            self._mark_all_workers_dirty()
            raise LocalRolloutBatchError(
                "step-validation", {active_worker: exc}, returned_results=returned_results
            ) from exc

        for record in results:
            worker_id = record.worker_id
            self._step_ids[worker_id] = record.step_id
            self._logical_times[worker_id] = record.logical_time
            self._observations[worker_id] = record.next_observation
            if record.terminated or record.truncated:
                self._done_workers.add(worker_id)
        return results

    async def close(self) -> None:
        execution = _LOCAL_EXECUTION.get()
        if execution is not None and execution.active and execution.pool is self:
            raise RuntimeError(
                "local rollout pool close cannot be awaited from its active "
                f"{execution.phase} execution context"
            )
        async with self._lifecycle:
            if self._state is LocalRolloutPoolState.CLOSED:
                return
            task = self._close_task
            if task is None or task.done():
                self._state = LocalRolloutPoolState.CLOSING
                task = asyncio.create_task(self._run_close_attempt())
                self._close_task = task
        # One caller cancelling must not cancel the shared close completion.
        await asyncio.shield(task)

    async def _run_close_attempt(self) -> None:
        async with self._lifecycle:
            await self._lifecycle.wait_for(lambda: self._active_operations == 0)
            already_recovered_closed = tuple(
                worker_id
                for worker_id in self._worker_ids
                if worker_id in self._dirty_workers
                and worker_id in self._recovery_closed_workers
                and not self._close_receipts.get(
                    worker_id,
                    WorkerCloseReceipt(worker_id, 0, False),
                ).succeeded
            )
            pending = tuple(
                worker_id
                for worker_id in self._worker_ids
                if worker_id not in already_recovered_closed
                if not self._close_receipts.get(
                    worker_id,
                    WorkerCloseReceipt(worker_id, 0, False),
                ).succeeded
            )

        results = await self._collect(
            self._close_worker(worker_id) for worker_id in pending
        )
        receipts: dict[str, WorkerCloseReceipt] = {}
        for worker_id in already_recovered_closed:
            self._close_attempts[worker_id] += 1
            receipts[worker_id] = WorkerCloseReceipt(
                worker_id=worker_id,
                attempt=self._close_attempts[worker_id],
                succeeded=True,
            )
        for worker_id, result in zip(pending, results, strict=True):
            self._close_attempts[worker_id] += 1
            error = result if isinstance(result, BaseException) else None
            receipts[worker_id] = WorkerCloseReceipt(
                worker_id=worker_id,
                attempt=self._close_attempts[worker_id],
                succeeded=error is None,
                error=error,
            )

        async with self._lifecycle:
            self._close_receipts.update(receipts)
            failures = {
                worker_id: receipt
                for worker_id, receipt in self._close_receipts.items()
                if not receipt.succeeded
            }
            self._state = (
                LocalRolloutPoolState.CLOSE_FAILED if failures else LocalRolloutPoolState.CLOSED
            )
        if failures:
            raise LocalRolloutCloseError(failures)

    async def _close_worker(self, worker_id: str) -> None:
        execution = _LocalExecutionContext(self, f"worker close ({worker_id})")
        token: Token[_LocalExecutionContext | None] = _LOCAL_EXECUTION.set(execution)
        try:
            await self._invoke_environment(self._environments[worker_id].close)
        finally:
            execution.active = False
            _LOCAL_EXECUTION.reset(token)
