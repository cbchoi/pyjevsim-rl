"""A small Gym-like environment over a freshly built pyjevsim executor."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn, cast

from pyjevsim_bridge.rl.contracts import (
    EnvironmentClosedError,
    EpisodeBinding,
    EpisodeCleanupError,
    EpisodeContext,
    EpisodeFactory,
    EpisodeLifecycleError,
    EpisodeStateError,
    EpisodeStepError,
    ExecutorProtocol,
    StepView,
)
from pyjevsim_bridge.rl.executor import (
    BindingDecisionBoundary,
    DecisionBoundary,
    ExecutorDriver,
    ExecutorQualificationPolicy,
)


@dataclass(frozen=True)
class _GuardedDecisionBoundary:
    boundary: DecisionBoundary
    guard: Callable[[], None]

    def resolve(self, executor: ExecutorProtocol) -> float:
        target = self.boundary.resolve(executor)
        self.guard()
        return target


class _StepCloseRequestedError(EnvironmentClosedError):
    def __init__(self, *, phase: str, completion_boundary: str) -> None:
        self.phase = phase
        self.completion_boundary = completion_boundary
        super().__init__("close requested during step callback")


class PyJevSimEnv:
    """Dependency-free Gymnasium-shaped facade for one environment instance."""

    def __init__(
        self,
        factory: EpisodeFactory,
        *,
        instance_id: str = "env-0",
        boundary: DecisionBoundary | None = None,
        max_steps: int | None = None,
        run_id: str = "local",
        plugin_version: str | None = None,
        executor_qualification: ExecutorQualificationPolicy | None = None,
        require_claim_grade: bool = False,
    ) -> None:
        if not instance_id:
            raise ValueError("instance_id must be non-empty")
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if plugin_version == "":
            raise ValueError("plugin_version must be non-empty when provided")
        if executor_qualification is not None and not isinstance(
            executor_qualification, ExecutorQualificationPolicy
        ):
            raise TypeError(
                "executor_qualification must use ExecutorQualificationPolicy"
            )
        if type(require_claim_grade) is not bool:
            raise TypeError("require_claim_grade must be a bool")
        if require_claim_grade and (
            executor_qualification is None
            or not executor_qualification.claim_grade
        ):
            raise ValueError(
                "claim-grade execution requires the controlled pinned "
                "executor qualification policy"
            )
        self._factory = factory
        self._instance_id = instance_id
        self._boundary = boundary
        self._max_steps = max_steps
        self._run_id = run_id
        self._plugin_version = plugin_version
        self._executor_qualification = executor_qualification
        self._require_claim_grade = require_claim_grade
        self._episode_number = 0
        self._binding: EpisodeBinding | None = None
        self._driver: ExecutorDriver | None = None
        self._observation: object = None
        self._seed: int | None = None
        self._step_id = 0
        self._done = False
        self._failed = False
        self._closed = False
        self._closing = False
        self._disposing = False
        self._constructing = False
        self._construction_cleanup_started = False
        self._stepping = False
        self._close_requested = False
        self._lifecycle_lock = threading.RLock()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[object, dict[str, object]]:
        """Dispose the previous graph and build a fresh episode binding."""

        with self._lifecycle_lock:
            self._ensure_lifecycle_admission("reset")
            return self._reset_unlocked(seed=seed, options=options)

    def _reset_unlocked(
        self,
        *,
        seed: int | None,
        options: Mapping[str, object] | None,
    ) -> tuple[object, dict[str, object]]:
        self._ensure_open()
        self._dispose_episode(
            phase="reset.cleanup_previous",
            completion_boundary="pre-reset",
        )
        if self._close_requested:
            # A resource cleanup callback may synchronously re-enter close().
            # The current disposal owns both resources until it finishes, so
            # the nested call records intent instead of disposing them twice.
            # Honour that intent before constructing a replacement graph.
            self._closed = True
            raise EnvironmentClosedError(
                "environment was closed during reset cleanup"
            )
        self._episode_number += 1
        episode_id = f"{self._instance_id}:episode-{self._episode_number}"

        binding: EpisodeBinding | None = None
        driver: ExecutorDriver | None = None
        phase = "options_materialization"
        self._constructing = True
        try:
            try:
                materialized_options = {} if options is None else dict(options)
                if self._close_requested:
                    self._abort_closed_construction(
                        binding=None,
                        driver=None,
                        episode_id=episode_id,
                        phase=phase,
                        logical_time=None,
                    )
                context = EpisodeContext(
                    episode_id=episode_id,
                    instance_id=self._instance_id,
                    seed=seed,
                    options=MappingProxyType(materialized_options),
                )
                if self._executor_qualification is not None:
                    phase = "executor_qualification_preflight"
                    self._executor_qualification.preflight()
                phase = "factory"
                binding = self._factory(context)
                phase = "executor_driver"
                driver = ExecutorDriver(binding.executor)
                if self._executor_qualification is not None:
                    phase = "executor_qualification"
                    driver.apply_qualification_policy(
                        self._executor_qualification
                    )
                if self._close_requested:
                    self._abort_closed_construction(
                        binding=binding,
                        driver=driver,
                        episode_id=episode_id,
                        phase=phase,
                        logical_time=None,
                    )
                phase = "binding_initialization"
                initializer = getattr(binding, "initialize", None)
                if callable(initializer):
                    initializer()
                if self._close_requested:
                    self._abort_closed_construction(
                        binding=binding,
                        driver=driver,
                        episode_id=episode_id,
                        phase=phase,
                        logical_time=None,
                    )
                phase = "initial_observation"
                observation = binding.observe(())
                if self._close_requested:
                    self._abort_closed_construction(
                        binding=binding,
                        driver=driver,
                        episode_id=episode_id,
                        phase=phase,
                        logical_time=None,
                    )
                phase = "initial_time"
                logical_time = driver.logical_time
                if self._close_requested:
                    self._abort_closed_construction(
                        binding=binding,
                        driver=driver,
                        episode_id=episode_id,
                        phase=phase,
                        logical_time=logical_time,
                    )
            except BaseException as original:
                if self._construction_cleanup_started:
                    raise
                errors: list[BaseException] = [original]
                failed_resources: list[str] = []
                if phase == "initial_time":
                    logical_time = None
                else:
                    logical_time, clock_error = self._safe_logical_time(driver)
                    if clock_error is not None:
                        errors.append(clock_error)
                if driver is not None:
                    try:
                        driver.close()
                    except BaseException as cleanup_error:
                        errors.append(cleanup_error)
                        # Keep the failed resource reachable so reset() or close()
                        # can retry cleanup. Dropping the only reference here
                        # would turn a reported cleanup failure into a leak.
                        self._driver = driver
                        failed_resources.append("executor")
                if binding is not None:
                    try:
                        binding.close()
                    except BaseException as cleanup_error:
                        errors.append(cleanup_error)
                        self._binding = binding
                        failed_resources.append("binding")
                self._failed = True
                self._done = True
                if failed_resources:
                    raise EpisodeCleanupError(
                        "episode reset failed and cleanup remains incomplete",
                        run_id=self._run_id,
                        phase="reset.cleanup_failed_initialization",
                        completion_boundary=f"{phase}:cleanup-incomplete",
                        instance_id=self._instance_id,
                        episode_id=episode_id,
                        step_id=0,
                        logical_time=logical_time,
                        recoverable=True,
                        plugin_version=self._plugin_version,
                        causes=tuple(errors),
                        failed_resources=tuple(failed_resources),
                    ) from original
                if self._close_requested:
                    self._closed = True
                raise EpisodeLifecycleError(
                    f"episode reset failed during {phase}: {original}",
                    run_id=self._run_id,
                    phase=f"reset.{phase}",
                    completion_boundary="pre-episode",
                    instance_id=self._instance_id,
                    episode_id=episode_id,
                    step_id=0,
                    logical_time=logical_time,
                    recoverable=True,
                    plugin_version=self._plugin_version,
                    causes=tuple(errors),
                ) from original

        finally:
            self._constructing = False
            self._construction_cleanup_started = False

        self._binding = binding
        self._driver = driver
        self._observation = observation
        self._seed = seed
        self._step_id = 0
        self._done = False
        self._failed = False
        return observation, self._provenance(cast(float, logical_time))

    def _abort_closed_construction(
        self,
        *,
        binding: EpisodeBinding | None,
        driver: ExecutorDriver | None,
        episode_id: str,
        phase: str,
        logical_time: float | None,
    ) -> NoReturn:
        self._construction_cleanup_started = True
        cleanup_errors: list[BaseException] = []
        failed_resources: list[str] = []
        if driver is not None:
            try:
                driver.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                self._driver = driver
                failed_resources.append("executor")
        if binding is not None:
            try:
                binding.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                self._binding = binding
                failed_resources.append("binding")
        self._failed = True
        self._done = True
        if failed_resources:
            raise EpisodeCleanupError(
                "close requested during reset construction and cleanup failed",
                run_id=self._run_id,
                phase="reset.cleanup_closed_construction",
                completion_boundary=f"{phase}:cleanup-incomplete",
                instance_id=self._instance_id,
                episode_id=episode_id,
                step_id=0,
                logical_time=logical_time,
                recoverable=True,
                plugin_version=self._plugin_version,
                causes=tuple(cleanup_errors),
                failed_resources=tuple(failed_resources),
            ) from cleanup_errors[0]
        self._closed = True
        raise EnvironmentClosedError(
            "environment was closed during reset construction"
        )

    def step(
        self,
        action: object,
    ) -> tuple[object, float, bool, bool, dict[str, object]]:
        """Apply one action and return the Gymnasium-ordered five-tuple."""

        with self._lifecycle_lock:
            self._ensure_lifecycle_admission("step")
            return self._step_unlocked(action)

    def _step_unlocked(
        self,
        action: object,
    ) -> tuple[object, float, bool, bool, dict[str, object]]:
        self._ensure_open()
        if self._failed:
            raise EpisodeStateError("episode failed; reset or close required")
        if self._binding is None or self._driver is None:
            raise EpisodeStateError("reset must be called before step")
        if self._done:
            raise EpisodeStateError("step called after terminated or truncated; reset required")

        binding = self._binding
        driver = self._driver
        previous_observation = self._observation
        phase = "pre_step_clock"
        completion_boundary = "pre-commit"
        failure_time: float | None = None
        self._stepping = True
        try:
            failure_time = driver.logical_time
            self._raise_if_step_close_requested()
            phase = "apply_action"
            binding.apply_action(action)
            self._raise_if_step_close_requested()
            phase = "resolve_boundary"
            boundary = self._boundary or BindingDecisionBoundary(
                binding.next_decision_time
            )
            phase = "executor_advance"
            completion_boundary = "commit-unknown"
            committed = driver.advance(
                _GuardedDecisionBoundary(
                    boundary=boundary,
                    guard=lambda: self._raise_if_step_close_requested(
                        phase="resolve_boundary",
                        completion_boundary="pre-commit",
                    ),
                )
            )
            completion_boundary = "committed"
            failure_time = committed.logical_time
            self._raise_if_step_close_requested()
            phase = "observe"
            observation = binding.observe(committed.external_events)
            self._raise_if_step_close_requested()
            next_step_id = self._step_id + 1
            phase = "step_view_snapshot"
            transition = StepView(
                episode_id=f"{self._instance_id}:episode-{self._episode_number}",
                instance_id=self._instance_id,
                step_id=next_step_id,
                previous_logical_time=committed.previous_time,
                logical_time=committed.logical_time,
                action=action,
                previous_observation=previous_observation,
                observation=observation,
                external_events=committed.external_events,
            )
            self._raise_if_step_close_requested()
            phase = "reward"
            reward = float(binding.reward(transition))
            self._raise_if_step_close_requested()
            if not math.isfinite(reward):
                raise EpisodeStateError(f"reward must be finite, got {reward!r}")
            phase = "termination"
            terminated = bool(binding.terminated(transition))
            self._raise_if_step_close_requested()
            if not terminated:
                terminated = driver.is_terminated()
                self._raise_if_step_close_requested()
            truncated = self._max_steps is not None and next_step_id >= self._max_steps
            phase = "info"
            plugin_info = dict(binding.info(transition))
            self._raise_if_step_close_requested()
        except BaseException as exc:
            # Action injection and executor advancement are not generally
            # reversible. Fail closed so stale observation/step identity can
            # never be paired with an already-advanced model.
            self._failed = True
            if isinstance(exc, _StepCloseRequestedError):
                phase = exc.phase
                completion_boundary = exc.completion_boundary
            causes: list[BaseException] = [exc]
            failed_resources: tuple[str, ...] = ()
            if phase == "executor_advance":
                committed_time, clock_error = self._safe_logical_time(driver)
                if committed_time is not None:
                    failure_time = committed_time
                if clock_error is not None:
                    causes.append(clock_error)
            if self._close_requested:
                try:
                    self._dispose_episode(
                        phase="step.close_requested_cleanup",
                        completion_boundary=completion_boundary,
                    )
                except EpisodeCleanupError as cleanup_error:
                    causes.extend(cleanup_error.causes)
                    failed_resources = cleanup_error.failed_resources
                    self._failed = True
                    self._done = True
                    raise EpisodeCleanupError(
                        "step close requested and cleanup remains incomplete",
                        run_id=self._run_id,
                        phase=f"step.{phase}.close_requested_cleanup",
                        completion_boundary=(
                            f"{completion_boundary}:cleanup-incomplete"
                        ),
                        instance_id=self._instance_id,
                        episode_id=(
                            f"{self._instance_id}:episode-"
                            f"{self._episode_number}"
                        ),
                        step_id=self._step_id + 1,
                        logical_time=failure_time,
                        recoverable=True,
                        plugin_version=self._plugin_version,
                        causes=tuple(causes),
                        failed_resources=failed_resources,
                    ) from exc
                else:
                    self._closed = True
                self._failed = True
                self._done = True
            raise EpisodeStepError(
                f"episode step failed during {phase}: {exc}",
                run_id=self._run_id,
                phase=phase,
                completion_boundary=completion_boundary,
                instance_id=self._instance_id,
                episode_id=f"{self._instance_id}:episode-{self._episode_number}",
                step_id=self._step_id + 1,
                logical_time=failure_time,
                plugin_version=self._plugin_version,
                causes=tuple(causes),
                failed_resources=failed_resources,
            ) from exc
        finally:
            self._stepping = False

        self._step_id = next_step_id
        self._observation = observation
        self._done = terminated or truncated
        info = plugin_info
        info.update(self._provenance(committed.logical_time))
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the active episode; an incomplete cleanup remains retryable."""

        with self._lifecycle_lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        if self._closed or self._closing:
            return
        if self._disposing or self._constructing or self._stepping:
            self._close_requested = True
            return
        self._closing = True
        try:
            self._dispose_episode(
                phase="close.cleanup",
                completion_boundary="close-requested",
            )
            self._closed = True
        finally:
            self._closing = False

    def _dispose_episode(self, *, phase: str, completion_boundary: str) -> None:
        if self._disposing:
            raise EpisodeStateError("episode cleanup is already in progress")
        self._disposing = True
        try:
            self._dispose_episode_once(
                phase=phase,
                completion_boundary=completion_boundary,
            )
        finally:
            self._disposing = False

    def _dispose_episode_once(
        self, *, phase: str, completion_boundary: str
    ) -> None:
        binding, driver = self._binding, self._driver
        errors: list[BaseException] = []
        failed_resources: list[str] = []
        logical_time, clock_error = self._safe_logical_time(driver)
        if driver is not None:
            try:
                driver.close()
            except BaseException as exc:
                errors.append(exc)
                failed_resources.append("executor")
            else:
                self._driver = None
        if binding is not None:
            try:
                binding.close()
            except BaseException as exc:
                errors.append(exc)
                failed_resources.append("binding")
            else:
                self._binding = None
        if errors:
            # Cleanup is part of the episode lifecycle boundary. Until every
            # resource confirms closure, the environment must neither step
            # the partially disposed graph nor claim to be closed. Retained
            # references make a subsequent reset()/close() a cleanup retry.
            self._failed = True
            self._done = True
            raise EpisodeCleanupError(
                "episode cleanup failed",
                run_id=self._run_id,
                phase=phase,
                completion_boundary=f"{completion_boundary}:cleanup-incomplete",
                instance_id=self._instance_id,
                episode_id=self._current_episode_id(),
                step_id=self._step_id,
                logical_time=logical_time,
                recoverable=True,
                plugin_version=self._plugin_version,
                causes=(
                    ((clock_error,) if clock_error is not None else ())
                    + tuple(errors)
                ),
                failed_resources=tuple(failed_resources),
            ) from errors[0]
        self._observation = None
        self._done = False
        self._failed = False

    def _current_episode_id(self) -> str | None:
        if self._episode_number <= 0:
            return None
        return f"{self._instance_id}:episode-{self._episode_number}"

    @staticmethod
    def _safe_logical_time(
        driver: ExecutorDriver | None,
    ) -> tuple[float | None, BaseException | None]:
        if driver is None:
            return None, None
        try:
            return driver.logical_time, None
        except BaseException as exc:
            return None, exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise EnvironmentClosedError("environment is closed")
        if self._closing:
            raise EpisodeStateError("environment is closing")

    def _ensure_lifecycle_admission(self, operation: str) -> None:
        if self._constructing or self._disposing or self._stepping:
            raise EpisodeStateError(
                f"{operation} cannot re-enter an active environment lifecycle operation"
            )

    def _raise_if_step_close_requested(
        self,
        *,
        phase: str | None = None,
        completion_boundary: str | None = None,
    ) -> None:
        if self._close_requested:
            if phase is None or completion_boundary is None:
                raise EnvironmentClosedError(
                    "close requested during step callback"
                )
            raise _StepCloseRequestedError(
                phase=phase,
                completion_boundary=completion_boundary,
            )

    def _provenance(self, logical_time: float) -> dict[str, object]:
        return {
            "run_id": self._run_id,
            "episode_id": f"{self._instance_id}:episode-{self._episode_number}",
            "instance_id": self._instance_id,
            "step_id": self._step_id,
            "logical_time": logical_time,
            "seed": self._seed,
            "executor_qualified": (
                self._executor_qualification is not None
                and self._executor_qualification.claim_grade
            ),
            "executor_qualification_policy_id": (
                None
                if self._executor_qualification is None
                else self._executor_qualification.policy_id
            ),
        }
