"""Callable-based adapters for adding RL semantics to existing models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pyjevsim_bridge.rl.contracts import ExecutorProtocol, StepView


@dataclass
class FunctionalEpisodeBinding:
    """Build an episode binding from small model-specific callables.

    This is the shortest integration path for an existing pyjevsim model: the
    model factory constructs a fresh ``SysExecutor`` and supplies action,
    observation, reward, and termination functions here. An optional
    initializer performs semantic bootstrap only after executor qualification.
    No subclass of framework internals is required.
    """

    executor: ExecutorProtocol
    apply_action_fn: Callable[[ExecutorProtocol, object], None]
    observe_fn: Callable[[ExecutorProtocol, object], object]
    reward_fn: Callable[[StepView], float]
    terminated_fn: Callable[[StepView], bool] | None = None
    info_fn: Callable[[StepView], Mapping[str, object]] | None = None
    decision_time_fn: Callable[[ExecutorProtocol], float] | None = None
    initialize_fn: Callable[[ExecutorProtocol], None] | None = None
    close_fn: Callable[[], None] | None = None

    def initialize(self) -> None:
        """Run an optional semantic bootstrap after executor qualification."""

        if self.initialize_fn is not None:
            self.initialize_fn(self.executor)

    def apply_action(self, action: object) -> None:
        self.apply_action_fn(self.executor, action)

    def next_decision_time(self) -> float:
        if self.decision_time_fn is not None:
            return float(self.decision_time_fn(self.executor))
        return float(self.executor.get_next_event_time())

    def observe(self, external_events: object) -> object:
        return self.observe_fn(self.executor, external_events)

    def reward(self, transition: StepView) -> float:
        return float(self.reward_fn(transition))

    def terminated(self, transition: StepView) -> bool:
        if self.terminated_fn is None:
            return False
        return bool(self.terminated_fn(transition))

    def info(self, transition: StepView) -> Mapping[str, object]:
        if self.info_fn is None:
            return {}
        return self.info_fn(transition)

    def close(self) -> None:
        if self.close_fn is not None:
            self.close_fn()
