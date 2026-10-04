"""Explicit installation allowlist; never imports code named by a snapshot."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ContinuationBundle, ContinuationError, ProfileDescriptor,
    checked_id, checked_sha, fail,
)


@dataclass(frozen=True, slots=True)
class SourceBinding:
    """Trusted installation metadata, never decoded from snapshot input."""
    logical_id: str
    path: str
    sha256: str

    def __post_init__(self):
        checked_id(self.logical_id)
        checked_id(self.path, "source path")
        checked_sha(self.sha256)


_METHODS = {
    "engine": ("services", "inspect", "export_state", "validate_payload", "allocate_empty", "attach",
               "restore_into", "validate_restored"),
    "model": ("reference_objects", "descriptor", "topology", "obligations", "inspect", "export_state",
              "validate_payload", "validate_composition", "allocate_shell", "restore_into",
              "rebind", "make_binding", "validate_restored"),
    "boundary": ("cursor", "validate_branch", "inspect", "export_state", "validate_payload", "allocate_state",
                 "bind_uninitialized", "restore_into", "validate_restored"),
    "fresh_factory": ("create_episode",),
}


def _method_identity(provider: Any, name: str) -> tuple:
    method = getattr(provider, name, None)
    if not callable(method):
        fail(f"installed provider lacks {name}", "CC_UNSUPPORTED_PROFILE")
    function = getattr(method, "__func__", method)
    return (function, getattr(function, "__code__", None), getattr(method, "__self__", None))


def _identity(bundle: ContinuationBundle) -> tuple:
    result = []
    for role, required in _METHODS.items():
        provider = getattr(bundle, role)
        methods = required + (("verify_identity",) if hasattr(provider, "verify_identity") else ())
        # The split hooks are optional for legacy installed providers, but a
        # callable used by the built-in shim is pinned just like that shim.
        methods += tuple(name for name in ("verify_source_bytes", "verify_loaded_identity")
                         if callable(getattr(provider, name, None)))
        sources = tuple(getattr(provider, "source_bindings", ()))
        if any(type(source) is not SourceBinding for source in sources):
            fail(f"{role} source bindings are not trusted SourceBinding values")
        if len({source.logical_id for source in sources}) != len(sources):
            fail(f"{role} source bindings contain duplicate IDs")
        result.append((role, id(provider), type(provider), sources,
                       tuple((name, _method_identity(provider, name)) for name in methods)))
    return tuple(result)


def _verify_sources(bundle: ContinuationBundle) -> None:
    for role in _METHODS:
        provider = getattr(bundle, role)
        for source in tuple(getattr(provider, "source_bindings", ())):
            try:
                with Path(source.path).open("rb") as stream:
                    measured = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError as exc:
                raise ContinuationError("CC_INCOMPATIBLE_IDENTITY", "installed source is unreadable") from exc
            if measured != source.sha256:
                fail(f"installed source changed: {source.logical_id}", "CC_INCOMPATIBLE_IDENTITY")
        verify = getattr(provider, "verify_identity", None)
        if verify is not None:
            # Trusted installed code may additionally validate engine runtime and
            # callbacks. This is not an import or a snapshot-supplied callback.
            verify()


class ContinuationRegistry:
    def __init__(self) -> None:
        self._bundles: dict[str, ContinuationBundle] = {}
        self._identities: dict[str, tuple] = {}
        self._profiles: dict[str, dict] = {}
        self._generations: dict[str, int] = {}
        self._next_generation = 0
        # No provider/property/validator/source I/O runs inside this lock. It
        # publishes the related entry maps atomically; it is not a runtime lock
        # or a lock on Python objects and installed files.
        self._entry_lock = threading.RLock()

    def _entry(self, profile_id: str) -> tuple:
        with self._entry_lock:
            bundle = self._bundles.get(profile_id)
            if bundle is not None:
                return (bundle, self._generations[profile_id],
                        self._identities[profile_id], self._profiles[profile_id])
        fail(f"profile {profile_id} is not installed", "CC_UNSUPPORTED_PROFILE")

    def _require_entry(self, profile_id: str, witness: tuple) -> None:
        current = self._entry(profile_id)
        if (current[0] is not witness[0] or current[1] != witness[1]
                or current[2] is not witness[2] or current[3] is not witness[3]):
            fail("registered entry changed during operation", "CC_INCOMPATIBLE_IDENTITY")

    def validation_context(self, profile: ProfileDescriptor, *, operation: str, runtime=None):
        """Create a phase witness, not a source/identity validation checkpoint."""
        return ValidationContext(self, profile, operation=operation, runtime=runtime)

    def register(self, bundle: ContinuationBundle) -> None:
        if type(bundle) is not ContinuationBundle or type(bundle.profile) is not ProfileDescriptor:
            fail("registration requires a ContinuationBundle", "CC_UNSUPPORTED_PROFILE")
        key = bundle.profile.profile_id
        with self._entry_lock:
            previous = self._bundles.get(key)
        if previous is not None:
            if previous is not bundle:
                fail(f"profile {key} is already registered", "CC_INCOMPATIBLE_IDENTITY")
            self.get(key)
            return
        identity = _identity(bundle)
        _verify_sources(bundle)
        if _identity(bundle) != identity:
            fail("provider changed during registration", "CC_INCOMPATIBLE_IDENTITY")
        profile = bundle.profile.to_payload()
        with self._entry_lock:
            previous = self._bundles.get(key)
            if previous is None:
                self._next_generation += 1
                self._bundles[key] = bundle
                self._identities[key] = identity
                self._profiles[key] = profile
                self._generations[key] = self._next_generation
                return
        # A callback or another registration may have published this key while
        # its trusted verifiers ran. Do not hold the entry lock across get().
        if previous is not bundle:
            fail(f"profile {key} is already registered", "CC_INCOMPATIBLE_IDENTITY")
        self.get(key)

    def get(self, profile_id: str) -> ContinuationBundle:
        checked_id(profile_id, "profile ID")
        witness = self._entry(profile_id)
        bundle, _, identity, profile = witness
        if bundle.profile.to_payload() != profile or _identity(bundle) != identity:
            fail("registered provider/profile was replaced", "CC_INCOMPATIBLE_IDENTITY")
        _verify_sources(bundle)
        if _identity(bundle) != identity:
            fail("provider changed during identity verification", "CC_INCOMPATIBLE_IDENTITY")
        self._require_entry(profile_id, witness)
        return bundle

    def resolve(self, profile: ProfileDescriptor) -> ContinuationBundle:
        if type(profile) is not ProfileDescriptor:
            fail("resolve requires a ProfileDescriptor")
        bundle = self.get(profile.profile_id)
        if profile.to_payload() != self._entry(profile.profile_id)[3]:
            fail("snapshot and installed profile identities differ", "CC_INCOMPATIBLE_IDENTITY")
        return bundle


_UNSPECIFIED = object()


@dataclass(frozen=True, slots=True, eq=False)
class ValidationStamp:
    """Internal same-operation witness; unsuitable for wire serialization."""

    _context_token: object
    registry: object
    generation: int
    operation: str
    phase: int
    owner_thread: object
    runtime: object

    def __reduce_ex__(self, protocol):
        raise TypeError("validation stamps are process-local")


class ValidationContext:
    """Process-local lifetime witness, never cached validation authority.

    Creating, stamping or requiring this context checks no source bytes or
    loaded code. Only ``resolve`` is a full registry checkpoint. Trusted callers
    advance the phase across callbacks/runtime mutation. This is not a sandbox
    against arbitrary Python field modification. Implementation stays in this
    already source-pinned module without adding source reads to strict paths.
    """

    __slots__ = ("_registry", "_profile", "_entry_witness", "_operation",
                 "_runtime", "_owner_thread", "_token", "_phase", "_closed",
                 "_last_checkpoint")

    def __init__(self, registry, profile: ProfileDescriptor, *, operation: str, runtime=None):
        if type(registry) is not ContinuationRegistry or type(profile) is not ProfileDescriptor:
            fail("validation context requires an installed registry/profile", "CC_INCOMPATIBLE_IDENTITY")
        self._operation = checked_id(operation, "validation operation")
        self._registry = registry
        self._profile = profile
        self._entry_witness = registry._entry(profile.profile_id)
        if profile.to_payload() != self._entry_witness[3]:
            fail("validation context profile differs", "CC_INCOMPATIBLE_IDENTITY")
        self._runtime = runtime
        self._owner_thread = threading.current_thread()
        self._token = object()
        self._phase = 0
        self._closed = False
        self._last_checkpoint = None

    @property
    def registry(self):
        return self._registry

    @property
    def profile(self):
        return self._profile

    @property
    def operation(self):
        return self._operation

    @property
    def runtime(self):
        return self._runtime

    @property
    def generation(self):
        return self._entry_witness[1]

    @property
    def phase(self):
        return self._phase

    @property
    def last_checkpoint(self):
        return self._last_checkpoint

    def _require_active(self, *, runtime=_UNSPECIFIED):
        if threading.current_thread() is not self._owner_thread:
            fail("validation context belongs to another thread", "CC_BUSY")
        if self._closed:
            fail("validation context is closed", "CC_INCOMPATIBLE_IDENTITY")
        if runtime is not _UNSPECIFIED and runtime is not self._runtime:
            fail("validation context belongs to another runtime", "CC_INCOMPATIBLE_IDENTITY")
        self._registry._require_entry(self._profile.profile_id, self._entry_witness)

    def resolve(self, checkpoint: str):
        """Run one full legacy resolve; no memo or reduced checking frequency."""
        self._require_active()
        checked_id(checkpoint, "validation checkpoint")
        bundle = self._registry.resolve(self._profile)
        self._require_active()
        self._last_checkpoint = checkpoint
        return bundle

    def stamp(self) -> ValidationStamp:
        self._require_active()
        return ValidationStamp(self._token, self._registry, self.generation,
                               self._operation, self._phase, self._owner_thread, self._runtime)

    def require(self, stamp: ValidationStamp, *, runtime=_UNSPECIFIED) -> None:
        self._require_active(runtime=runtime)
        if (type(stamp) is not ValidationStamp or stamp._context_token is not self._token
                or stamp.registry is not self._registry or stamp.generation != self.generation
                or stamp.operation != self._operation or stamp.phase != self._phase
                or stamp.owner_thread is not self._owner_thread or stamp.runtime is not self._runtime):
            fail("validation stamp is stale or belongs to another operation", "CC_INCOMPATIBLE_IDENTITY")

    def advance_phase(self, reason: str) -> int:
        self._require_active()
        checked_id(reason, "validation phase reason")
        self._phase += 1
        return self._phase

    def close(self) -> None:
        if threading.current_thread() is not self._owner_thread:
            fail("validation context belongs to another thread", "CC_BUSY")
        self._closed = True

    def __enter__(self):
        self._require_active()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __reduce_ex__(self, protocol):
        raise TypeError("validation contexts are process-local")
