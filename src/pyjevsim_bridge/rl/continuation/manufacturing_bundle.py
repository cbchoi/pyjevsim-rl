"""Explicit registration of the bounded manufacturing development profile."""
from .adapters.manufacturing import ManufacturingModelAdapter
from .generic_boundary import DeclaredFixedDeltaBoundaryProvider
from .generic_bundle import make_declared_bundle
from ..models.manufacturing import validate_observation, validate_reward_state

PROFILE_ID = "CC-M1-manufacturing"


def make_manufacturing_bundle():
    model = ManufacturingModelAdapter()
    boundary = DeclaredFixedDeltaBoundaryProvider(
        profile_id=PROFILE_ID, schema_id="manufacturing-fixed-delta-v1",
        observation_validator=validate_observation, reward_validator=validate_reward_state,
        initial_reward_state={"last_cost": 0.0, "last_completed": 0},
        source_bindings=model.source_bindings)
    return make_declared_bundle(profile_id=PROFILE_ID, model=model, boundary=boundary,
        model_provider_id="manufacturing-model-adapter-v1", projection_id="manufacturing-committed-v1",
        capabilities=("static-flat", "fixed-delta", "inherited-confluence", "shared-tool-ledger",
                      "nonpreemptive-maintenance", "model-mt19937", "explicit-reward-baseline"))


def register_manufacturing_bundle(registry):
    bundle = make_manufacturing_bundle()
    registry.register(bundle)
    return bundle
