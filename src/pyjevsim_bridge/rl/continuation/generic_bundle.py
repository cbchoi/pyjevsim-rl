"""Explicit construction of a source-bound, model-declared continuation bundle.

No model names or domain state fields enter this builder. Installation is a
trusted allowlist decision, not automatic qualification of arbitrary Python.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from .contracts import (ContinuationBundle, ParserLimits, ProfileDescriptor,
    ProviderIdentity, ResetRequest, RuntimeParts, checked_id, digest, fail, thaw_value)
from .engine_pyjevsim import PyJevSimEngineProvider, source_identity
from .generic_boundary import DeclaredFixedDeltaBoundaryProvider, PROVIDER_ID as BOUNDARY_ID
from .obligations import (boundary_obligations, compose_obligations,
    coordinator_obligations, engine_obligations)
from .references import ReferenceRegistry
from .registry import SourceBinding


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


_IMPORT_SHA = _sha(__file__)


def _core_bindings():
    root = Path(__file__).parent
    names = ("contracts.py", "registry.py", "envelope.py", "references.py", "lifecycle.py",
             "coordinator.py", "obligations.py", "generic_bundle.py")
    return tuple(SourceBinding(f"continuation.{name[:-3]}", str(root / name),
        _IMPORT_SHA if name == "generic_bundle.py" else _sha(root / name)) for name in names)


def _implementation(provider):
    sources = tuple(getattr(provider, "source_bindings", ()))
    if not sources or any(type(source) is not SourceBinding for source in sources):
        fail("declared providers require explicit SourceBinding values", "CC_UNSUPPORTED_PROFILE")
    if len({source.logical_id for source in sources}) != len(sources):
        fail("duplicate declared source identity")
    return digest({source.logical_id: source.sha256 for source in sources})


def _callable_identity(value):
    if not callable(value):
        fail("fresh factory extension method is missing", "CC_UNSUPPORTED_PROFILE")
    function = getattr(value, "__func__", value)
    return function, getattr(function, "__code__", None), getattr(value, "__self__", None)


class DeclaredFreshFactory:
    """Calls declared initializers once on fresh create, never during restore."""
    def __init__(self, *, profile_id, engine, model, boundary):
        self.profile_id = checked_id(profile_id, "profile ID")
        self.engine_provider, self.model_adapter, self.boundary_provider = engine, model, boundary
        self.source_bindings = _core_bindings()
        self._owners = (engine, model, boundary)
        self._extensions = tuple((obj, name, _callable_identity(getattr(obj, name, None)))
            for obj, name in ((model, "construction_from_request"),
                (boundary, "descriptor_from_request"), (boundary, "create_fresh")))

    def verify_loaded_identity(self):
        if (any(current is not expected for current, expected in zip(
                (self.engine_provider, self.model_adapter, self.boundary_provider), self._owners, strict=True))
                or self.profile_id != self.boundary_provider.profile_id):
            fail("declared factory provider/configuration changed", "CC_INCOMPATIBLE_IDENTITY")
        for obj, name, identity in self._extensions:
            if _callable_identity(getattr(obj, name, None)) != identity:
                fail("declared factory extension changed", "CC_INCOMPATIBLE_IDENTITY")

    def verify_source_bytes(self):
        for source in self.source_bindings:
            if _sha(source.path) != source.sha256:
                fail("declared factory source changed", "CC_INCOMPATIBLE_IDENTITY")

    def verify_identity(self):
        # Factory loaded checks preceded source reads in the strict predecessor.
        self.verify_loaded_identity()
        self.verify_source_bytes()

    def create_episode(self, request, cleanup):
        if type(request) is not ResetRequest or request.profile_id != self.profile_id:
            fail("fresh request does not match registered profile", "CC_UNSUPPORTED_PROFILE")
        model, boundary, provider = self.model_adapter, self.boundary_provider, self.engine_provider
        construction = model.construction_from_request(request)
        descriptor = boundary.descriptor_from_request(request)
        engine = provider.allocate_empty({"provider_id": "pyjevsim-flat-hla-v1", "mode": "HLA_TIME",
            "time_resolution": 1.0, "name": "default"}, cleanup)
        services = provider.services(engine)
        graph = model.allocate_shell(construction, services, cleanup)
        state = boundary.allocate_state(descriptor, cleanup)
        refs = ReferenceRegistry(model.topology(), model.reference_objects(graph))
        provider.attach(engine, graph, refs)
        model.rebind(graph, refs, services)
        binding = model.make_binding(graph, services, state)
        env = boundary.create_fresh(binding, descriptor, request.seed, cleanup, state)
        refs.identity_snapshot()
        return RuntimeParts(engine, graph, binding, env, state, refs, metadata={
            "policy_context": thaw_value(request.policy_context),
            "sampling_context": thaw_value(request.sampling_context),
            "config_sha256": digest(construction["config"]),
        })


def make_declared_bundle(*, profile_id, model, boundary, model_provider_id, projection_id,
                         profile_version="1", model_version="1", capabilities=(), limits=None):
    """Build only; callers explicitly register the resulting trusted bundle."""
    checked_id(profile_id, "profile ID")
    if type(boundary) is not DeclaredFixedDeltaBoundaryProvider or boundary.profile_id != profile_id:
        fail("declared boundary/profile registration differs", "CC_UNSUPPORTED_PROFILE")
    engine = PyJevSimEngineProvider()
    fresh = DeclaredFreshFactory(profile_id=profile_id, engine=engine, model=model, boundary=boundary)
    obligations = compose_obligations(engine=engine_obligations(), model=model.obligations(),
        boundary=boundary_obligations(), coordinator=coordinator_obligations())
    runtime_identity = {"python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]), "byteorder": sys.byteorder, "platform": sys.platform,
        "native_and_engine_sources": source_identity(),
        "continuation_sources": {source.logical_id: source.sha256 for source in fresh.source_bindings}}
    profile = ProfileDescriptor(profile_id=profile_id, version=profile_version,
        engine=ProviderIdentity("pyjevsim-flat-hla-v1", "1", digest(source_identity())),
        model=ProviderIdentity(model_provider_id, model_version, _implementation(model)),
        boundary=ProviderIdentity(BOUNDARY_ID, "1", digest({
            "sources": _implementation(boundary), "configuration": boundary.configuration_identity(),
        })),
        runtime_sha256=digest(runtime_identity),
        obligation_manifest_sha256=digest([obligation.to_payload() for obligation in obligations]),
        scheduler_semantics_id="pyjevsim-native-flat-set-calendar-p0", projection_id=projection_id,
        capabilities=tuple(capabilities), limits=ParserLimits() if limits is None else limits,
        obligations=obligations)
    return ContinuationBundle(profile, engine, model, boundary, fresh)
