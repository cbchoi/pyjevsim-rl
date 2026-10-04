"""Explicit held-out packet profile over the unchanged declared common core."""
from .adapters.packet_network import PacketNetworkModelAdapter
from .generic_boundary import DeclaredFixedDeltaBoundaryProvider
from .generic_bundle import make_declared_bundle
from ..models.packet_network import validate_observation, validate_reward_state

PROFILE_ID = "CC-N1-packet"


def make_packet_bundle():
    model = PacketNetworkModelAdapter()
    boundary = DeclaredFixedDeltaBoundaryProvider(profile_id=PROFILE_ID,
        schema_id="packet-network-fixed-delta-v1", observation_validator=validate_observation,
        reward_validator=validate_reward_state, initial_reward_state={"last_delivered": 0, "last_dropped": 0},
        source_bindings=model.source_bindings)
    return make_declared_bundle(profile_id=PROFILE_ID, model=model, boundary=boundary,
        model_provider_id="packet-network-model-adapter-v1", projection_id="packet-network-committed-v1",
        capabilities=("static-flat", "fixed-delta", "inherited-confluence", "two-path-routing",
                      "finite-fifo", "absolute-deadline", "deterministic-input", "explicit-reward-baseline"))


def register_packet_bundle(registry):
    bundle = make_packet_bundle()
    registry.register(bundle)
    return bundle
