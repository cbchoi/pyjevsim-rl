"""Explicit installation of the bounded queue continuation provider bundle.

The engine, domain adapter and RL boundary own separate state.  This wiring is
not a generic-model qualification and never imports the legacy live codec.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from .contracts import (
    ContinuationBundle, ParserLimits, ProfileDescriptor, ProviderIdentity,
    ResetRequest, RuntimeParts, digest, fail, thaw_value,
)
from .references import ReferenceRegistry
from .registry import ContinuationRegistry, SourceBinding

PROFILE_ID = "CC-P0-queue-compat"
PROFILE_VERSION = "1"


def _core_bindings() -> tuple[SourceBinding, ...]:
    root = Path(__file__).parent
    names = (
        "contracts.py", "registry.py", "envelope.py", "references.py",
        "lifecycle.py", "coordinator.py", "queue_bundle.py",
    )
    return tuple(
        SourceBinding(f"continuation.{name[:-3]}", str(root / name),
                      hashlib.sha256((root / name).read_bytes()).hexdigest())
        for name in names
    )


def _binding_identity(provider) -> str:
    sources = tuple(provider.source_bindings)
    if not sources:
        fail("queue installation has no implementation source identity")
    return digest({source.logical_id: source.sha256 for source in sources})


class QueueFreshFactory:
    """Normal initialization only; restore never invokes this factory."""

    def __init__(self, engine, model, boundary) -> None:
        self.engine_provider = engine
        self.model_adapter = model
        self.boundary_provider = boundary
        self.source_bindings = _core_bindings()

    def create_episode(self, request: ResetRequest, cleanup) -> RuntimeParts:
        if type(request) is not ResetRequest or request.profile_id != PROFILE_ID:
            fail("queue fresh factory requires its registered ResetRequest",
                 "CC_UNSUPPORTED_PROFILE")
        model = self.model_adapter
        boundary = self.boundary_provider
        engine_provider = self.engine_provider
        # Input sampling here is legitimate fresh initialization, never restore.
        construction = model.construction_from_request(request)
        boundary_descriptor = boundary.descriptor_from_request(request)
        engine = engine_provider.allocate_empty({
            "provider_id": "pyjevsim-flat-hla-v1", "mode": "HLA_TIME",
            "time_resolution": 1.0, "name": "default",
        }, cleanup)
        services = engine_provider.services(engine)
        graph = model.allocate_shell(construction, services, cleanup)
        boundary_state = boundary.allocate_state(boundary_descriptor, cleanup)
        refs = ReferenceRegistry(model.topology(), model.reference_objects(graph))
        engine_provider.attach(engine, graph, refs)
        model.rebind(graph, refs, services)
        binding = model.make_binding(graph, services, boundary_state)
        env = boundary.create_fresh(binding, boundary_descriptor, request.seed,
                                    cleanup, boundary_state)
        refs.identity_snapshot()
        return RuntimeParts(
            engine, graph, binding, env, boundary_state, refs,
            metadata={
                "policy_context": thaw_value(request.policy_context),
                "sampling_context": thaw_value(request.sampling_context),
                "config_sha256": digest(construction["config"]),
            },
        )


def make_queue_bundle() -> ContinuationBundle:
    """Create a fresh exact-build bundle; registration remains explicit."""
    from .adapters.queue import QueueModelAdapter
    from .boundary import FixedDeltaBoundaryProvider
    from .engine_pyjevsim import PyJevSimEngineProvider, source_identity

    engine = PyJevSimEngineProvider()
    model = QueueModelAdapter()
    boundary = FixedDeltaBoundaryProvider()
    fresh = QueueFreshFactory(engine, model, boundary)
    obligations = tuple(model.obligations())
    runtime_identity = {
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "byteorder": sys.byteorder,
        "platform": sys.platform,
        "native_and_engine_sources": source_identity(),
        "continuation_sources": {item.logical_id: item.sha256 for item in fresh.source_bindings},
    }
    profile = ProfileDescriptor(
        profile_id=PROFILE_ID, version=PROFILE_VERSION,
        engine=ProviderIdentity("pyjevsim-flat-hla-v1", "1", digest(source_identity())),
        model=ProviderIdentity("queue-model-adapter-v1", "1", _binding_identity(model)),
        boundary=ProviderIdentity("queue-fixed-delta-v1", "1", _binding_identity(boundary)),
        runtime_sha256=digest(runtime_identity),
        obligation_manifest_sha256=digest([item.to_payload() for item in obligations]),
        scheduler_semantics_id="pyjevsim-native-flat-set-calendar-p0",
        projection_id="queue-physical-and-rl-transition-v1",
        capabilities=("static-flat", "fixed-delta", "inline-input-tape",
                      "fixed-feedforward-policy", "queue-tie-projection"),
        limits=ParserLimits(), obligations=obligations,
    )
    return ContinuationBundle(profile, engine, model, boundary, fresh)


def register_queue_bundle(registry: ContinuationRegistry) -> ContinuationBundle:
    """Register one new bundle; same profile ID cannot replace an installation."""
    bundle = make_queue_bundle()
    registry.register(bundle)
    return bundle
