"""Small production-controlled PyJevSim model for migration qualification."""

from __future__ import annotations

from typing import Any

from pyjevsim.behavior_model import BehaviorModel
from pyjevsim.definition import ExecutionType
from pyjevsim.system_executor import SysExecutor

from pyjevsim_bridge.rl.adapters import FunctionalEpisodeBinding
from pyjevsim_bridge.rl.contracts import EpisodeContext, ExecutorProtocol, StepView


class ControlledCounter(BehaviorModel):  # type: ignore[misc]
    """One deterministic confluent counter with an executor-facing action port."""

    def __init__(self) -> None:
        super().__init__("controlled-counter")
        self.insert_state("active", deadline=1.0)
        self.insert_input_port("action")
        self.insert_output_port("unused")
        self.init_state("active")
        self.value = 0

    def output(self, deliverer: Any) -> None:
        del deliverer

    def int_trans(self) -> None:
        self.value += 1

    def ext_trans(self, port: str, message: Any) -> None:
        if port != "action":
            raise ValueError(f"unexpected input port: {port}")
        self.value += sum(int(item) for item in message.retrieve())


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _reward(step: StepView) -> float:
    return float(
        _integer(step.observation, "observation")
        - _integer(step.previous_observation, "previous_observation")
    )


def build_episode(context: EpisodeContext) -> FunctionalEpisodeBinding:
    """Build but do not semantically step one fresh qualified executor graph."""

    del context
    model = ControlledCounter()
    executor = SysExecutor(1.0, ex_mode=ExecutionType.HLA_TIME)
    executor.register_entity(model)
    executor.insert_input_port("action")
    executor.coupling_relation(None, "action", model, "action")

    def initialize(value: ExecutorProtocol) -> None:
        value.step(0.0)

    return FunctionalEpisodeBinding(
        executor=executor,
        initialize_fn=initialize,
        apply_action_fn=lambda value, action: value.insert_external_event(
            "action",
            action,
            scheduled_time=1.0,
        ),
        observe_fn=lambda _value, _events: model.value,
        reward_fn=_reward,
        info_fn=lambda _step: {"model_profile": "controlled-counter-v1"},
    )
