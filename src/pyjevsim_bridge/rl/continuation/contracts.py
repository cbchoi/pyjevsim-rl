"""Dependency-free value contracts for opt-in continuation.

These types do not establish simulator correctness.  Providers and their
declared profiles retain the semantic proof and conformance obligations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

SNAPSHOT_SCHEMA = "pyjevsim-continuation-snapshot-v1"


class ContinuationError(RuntimeError):
    def __init__(self, code: str, message: str, *, phase: str = "",
                 state_disposition: str = "", cleanup_errors: tuple = ()) -> None:
        self.code = code
        self.phase = phase
        self.state_disposition = state_disposition
        self.cleanup_errors = tuple(cleanup_errors)
        super().__init__(f"{code}: {str(message)[:4096]}")


def fail(message: str, code: str = "CC_INVALID_PAYLOAD") -> None:
    raise ContinuationError(code, message)


def exact_fields(value: Any, names: set[str], name: str) -> dict:
    if type(value) is not dict or set(value) != names:
        fail(f"{name} fields differ")
    return value


def checked_id(value: Any, name: str = "identity", limit: int = 4096) -> str:
    if type(value) is not str or not value:
        fail(f"{name} must be a nonempty string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        fail(f"{name} is not UTF-8")
    if size > limit:
        fail(f"{name} exceeds its byte limit", "CC_LIMIT")
    return value


def checked_sha(value: Any, name: str = "sha256") -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        fail(f"{name} must be lowercase SHA256")
    return value


@dataclass(frozen=True, slots=True)
class ParserLimits:
    max_bytes: int = 32 * 1024 * 1024
    max_depth: int = 64
    max_models: int = 10_000
    max_couplings: int = 100_000
    max_nodes: int = 1_000_000
    max_id_bytes: int = 4096

    def __post_init__(self) -> None:
        maximums = (32 * 1024 * 1024, 64, 10_000, 100_000, 1_000_000, 4096)
        for name, maximum in zip(self.__dataclass_fields__, maximums):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= maximum:
                fail(f"invalid parser limit {name}", "CC_LIMIT")

    def to_payload(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, value: dict) -> ParserLimits:
        return cls(**exact_fields(value, set(cls.__dataclass_fields__), "limits"))


DEFAULT_LIMITS = ParserLimits()


def freeze_value(value: Any, *, limits: ParserLimits = DEFAULT_LIMITS) -> Any:
    """Detach the closed JSON value domain; never retain mutable input aliases."""
    count = 0
    active: set[int] = set()

    def visit(item: Any, depth: int) -> Any:
        nonlocal count
        count += 1
        if count > limits.max_nodes or depth > limits.max_depth:
            fail("value depth/node limit exceeded", "CC_LIMIT")
        kind = type(item)
        if item is None or kind is bool:
            return item
        if kind is int:
            if item.bit_length() > 256:
                fail("integer exceeds 256-bit value domain", "CC_LIMIT")
            return item
        if kind is float:
            if not math.isfinite(item):
                fail("nonfinite number is outside the value domain")
            return item
        if kind is str:
            try:
                item.encode("utf-8")
            except UnicodeError:
                fail("value is not UTF-8")
            return item
        if kind not in (dict, MappingProxyType, list, tuple):
            fail(f"unsupported value type {kind.__name__}")
        marker = id(item)
        if marker in active:
            fail("cyclic value payload; use declared semantic references")
        active.add(marker)
        try:
            if kind in (dict, MappingProxyType):
                result = {}
                for key, entry in item.items():
                    checked_id(key, "object key", limits.max_id_bytes)
                    result[key] = visit(entry, depth + 1)
                return MappingProxyType(result)
            return tuple(visit(entry, depth + 1) for entry in item)
        finally:
            active.remove(marker)

    return visit(value, 0)


def thaw_value(value: Any) -> Any:
    if type(value) in (dict, MappingProxyType):
        return {key: thaw_value(entry) for key, entry in value.items()}
    if type(value) in (list, tuple):
        return [thaw_value(entry) for entry in value]
    return value


def canonical_bytes(value: Any, *, limits: ParserLimits = DEFAULT_LIMITS) -> bytes:
    plain = thaw_value(freeze_value(value, limits=limits))
    try:
        data = json.dumps(plain, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ContinuationError("CC_INVALID_PAYLOAD", "invalid canonical value") from exc
    if len(data) > limits.max_bytes:
        fail("serialized value exceeds byte limit", "CC_LIMIT")
    return data


def _owned_plain(value: Any) -> Any:
    """Copy a validated frozen tree in canonical object-key iteration order."""
    if type(value) is MappingProxyType:
        return {key: _owned_plain(value[key]) for key in sorted(value)}
    if type(value) is tuple:
        return [_owned_plain(entry) for entry in value]
    return value


@dataclass(frozen=True, slots=True, init=False)
class OwnedValue:
    """Owned canonical JSON object, not a validation or lifecycle capability.

    Construction validates the same object-root domain as
    ``decode_json(canonical_bytes(value))``. No mutable input or returned plain
    copy is retained. The frozen view uses mapping proxies and tuples, while
    ``to_plain`` returns fresh dictionaries/lists with canonical key order.
    Arbitrary cached bytes cannot be supplied through this constructor.
    """

    _value: Any = field(repr=False)
    _canonical_bytes: bytes = field(repr=False)

    def __init__(self, value: Any, *, limits: ParserLimits = DEFAULT_LIMITS) -> None:
        frozen = freeze_value(value, limits=limits)
        plain = _owned_plain(frozen)
        try:
            data = json.dumps(plain, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
            raise ContinuationError("CC_INVALID_PAYLOAD", "invalid canonical value") from exc
        if len(data) > limits.max_bytes:
            fail("serialized value exceeds byte limit", "CC_LIMIT")
        # Check after domain/size validation, matching the legacy round trip's
        # failure precedence for malformed or oversized non-object roots.
        if type(frozen) is not MappingProxyType:
            fail("payload is not a canonical JSON object")
        object.__setattr__(self, "_value", frozen)
        object.__setattr__(self, "_canonical_bytes", data)

    @property
    def value(self) -> Mapping:
        return self._value

    @property
    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def to_plain(self) -> dict:
        return _owned_plain(self._value)


def normalize_owned(value: Any, *, limits: ParserLimits = DEFAULT_LIMITS) -> OwnedValue:
    """Normalize an object-root value without parsing its generated JSON again.

    This is not a decoder for untrusted wire bytes and does not skip source,
    profile, state-owner, callback, or runtime admission checks. Scalar/list
    roots remain unsupported, as in ``decode_json(canonical_bytes(value))``.
    """
    return OwnedValue(value, limits=limits)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def decode_json(data: bytes, *, limits: ParserLimits = DEFAULT_LIMITS) -> dict:
    if type(data) is not bytes or not data:
        fail("payload must be nonempty bytes")
    if len(data) > limits.max_bytes:
        fail("payload exceeds byte limit", "CC_LIMIT")
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise ContinuationError("CC_INVALID_PAYLOAD", "payload is not UTF-8") from exc
    # Bound nesting before the recursive standard-library parser allocates it.
    quoted = escaped = False
    depth = 0
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > limits.max_depth + 1:
                fail("JSON nesting exceeds depth limit", "CC_LIMIT")
        elif char in "]}":
            depth -= 1

    def pairs(entries):
        result = {}
        for key, entry in entries:
            if key in result:
                fail(f"duplicate JSON key {key!r}")
            result[key] = entry
        return result

    def integer(token):
        if len(token.lstrip("-")) > 78:
            fail("integer token exceeds value domain", "CC_LIMIT")
        return int(token)

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_int=integer,
                            parse_constant=lambda _: fail("nonfinite JSON constant"))
        if type(result) is not dict or canonical_bytes(result, limits=limits) != data:
            fail("payload is not a canonical JSON object")
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise ContinuationError("CC_INVALID_PAYLOAD", "invalid JSON payload") from exc
    return result


class _PayloadMixin:
    @classmethod
    def from_payload(cls, value: dict):
        return cls(**exact_fields(value, set(cls.__dataclass_fields__), cls.__name__))

    def to_payload(self) -> dict:
        result = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if hasattr(value, "to_payload"):
                value = value.to_payload()
            elif type(value) is tuple:
                value = [v.to_payload() if hasattr(v, "to_payload") else thaw_value(v) for v in value]
            result[name] = thaw_value(value)
        return result


@dataclass(frozen=True, slots=True)
class ProviderIdentity(_PayloadMixin):
    provider_id: str
    version: str
    implementation_sha256: str

    def __post_init__(self):
        checked_id(self.provider_id)
        checked_id(self.version)
        checked_sha(self.implementation_sha256)

    @classmethod
    def from_payload(cls, value):
        return cls(**exact_fields(value, set(cls.__dataclass_fields__), "provider identity"))


@dataclass(frozen=True, slots=True)
class StateObligation(_PayloadMixin):
    obligation_id: str
    semantic_path: str
    owner: str
    read_sites: tuple[str, ...] = ()
    write_sites: tuple[str, ...] = ()
    disposition: str = "preserve"
    encoding: str = "json-values"
    restore_rule: str = "declared"
    invariant: str = "declared"
    counterexample: str = "unverified"
    test_ids: tuple[str, ...] = ()
    evidence_status: str = "unverified"

    def __post_init__(self):
        for name in ("obligation_id", "semantic_path", "encoding", "restore_rule", "invariant", "counterexample"):
            checked_id(getattr(self, name), name)
        if self.owner not in ("engine", "model", "boundary", "coordinator"):
            fail("unknown state owner")
        if self.disposition not in ("preserve", "constant", "reconstruct", "omit"):
            fail("unknown obligation disposition")
        if self.evidence_status not in ("unverified", "reviewed", "tested", "verified", "unsupported"):
            fail("unknown obligation evidence status")
        for name in ("read_sites", "write_sites", "test_ids"):
            entries = tuple(getattr(self, name))
            for entry in entries:
                checked_id(entry, name)
            object.__setattr__(self, name, entries)

    @classmethod
    def from_payload(cls, value):
        return cls(**exact_fields(value, set(cls.__dataclass_fields__), "state obligation"))


@dataclass(frozen=True, slots=True)
class ProfileDescriptor(_PayloadMixin):
    profile_id: str
    version: str
    engine: ProviderIdentity
    model: ProviderIdentity
    boundary: ProviderIdentity
    runtime_sha256: str
    obligation_manifest_sha256: str
    scheduler_semantics_id: str
    projection_id: str
    capabilities: tuple[str, ...] = ()
    limits: ParserLimits = field(default_factory=ParserLimits)
    schema_id: str = SNAPSHOT_SCHEMA
    obligations: tuple[StateObligation, ...] = ()

    def __post_init__(self):
        if type(self.limits) is not ParserLimits:
            fail("profile limits must be ParserLimits")
        for name in ("profile_id", "version", "scheduler_semantics_id", "projection_id"):
            checked_id(getattr(self, name), name, self.limits.max_id_bytes)
        if self.schema_id != SNAPSHOT_SCHEMA:
            fail("unknown continuation schema", "CC_INCOMPATIBLE_IDENTITY")
        for name in ("engine", "model", "boundary"):
            if type(getattr(self, name)) is not ProviderIdentity:
                fail(f"{name} identity must be ProviderIdentity")
        checked_sha(self.runtime_sha256)
        checked_sha(self.obligation_manifest_sha256)
        capabilities = tuple(self.capabilities)
        for item in capabilities:
            checked_id(item, "capability", self.limits.max_id_bytes)
        if len(set(capabilities)) != len(capabilities):
            fail("duplicate capability")
        obligations = tuple(self.obligations)
        if any(type(item) is not StateObligation for item in obligations):
            fail("obligations must be StateObligation values")
        if len({item.obligation_id for item in obligations}) != len(obligations):
            fail("duplicate obligation ID")
        if len({item.semantic_path for item in obligations}) != len(obligations):
            fail("a semantic state path has multiple owners")
        if digest([item.to_payload() for item in obligations]) != self.obligation_manifest_sha256:
            fail("obligation manifest digest differs", "CC_INCOMPATIBLE_IDENTITY")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "obligations", obligations)

    @classmethod
    def from_payload(cls, value):
        row = dict(exact_fields(value, set(cls.__dataclass_fields__), "profile"))
        for name in ("engine", "model", "boundary"):
            row[name] = ProviderIdentity.from_payload(row[name])
        row["limits"] = ParserLimits.from_payload(row["limits"])
        if type(row["obligations"]) is not list or type(row["capabilities"]) is not list:
            fail("profile collections must be lists")
        row["obligations"] = tuple(StateObligation.from_payload(v) for v in row["obligations"])
        return cls(**row)


@dataclass(frozen=True, slots=True)
class Violation(_PayloadMixin):
    code: str
    phase: str = ""
    semantic_path: str = ""
    obligation_id: str = ""
    message: str = ""


@dataclass(frozen=True, slots=True)
class CapabilityReport(_PayloadMixin):
    status: str
    reasons: tuple[Violation, ...] = ()
    profile_id: str = ""
    boundary_kind: str = ""
    obligation_status: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in ("supported", "unsupported", "unverified", "busy"):
            fail("unknown capability status")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "obligation_status", freeze_value(self.obligation_status))


def _freeze_mappings(instance, names):
    for name in names:
        value = getattr(instance, name)
        if type(value) not in (dict, MappingProxyType):
            fail(f"{name} must be a value mapping")
        object.__setattr__(instance, name, freeze_value(value))


@dataclass(frozen=True, slots=True)
class ResetRequest(_PayloadMixin):
    profile_id: str
    model_config: Mapping
    seed: int
    instance_id: str
    run_id: str
    delta: float
    max_steps: int
    policy_context: Mapping
    sampling_context: Mapping

    def __post_init__(self):
        for name in ("profile_id", "instance_id", "run_id"):
            checked_id(getattr(self, name), name)
        if type(self.seed) is not int or not 0 <= self.seed < 2**256:
            fail("seed must be a bounded nonnegative integer")
        if type(self.max_steps) is not int or self.max_steps <= 0:
            fail("max_steps must be positive")
        if type(self.delta) not in (int, float) or not math.isfinite(self.delta) or self.delta <= 0:
            fail("delta must be finite and positive")
        _freeze_mappings(self, ("model_config", "policy_context", "sampling_context"))


@dataclass(frozen=True, slots=True)
class CaptureRequest(_PayloadMixin):
    expected_profile_id: str
    expected_boundary_cursor: int
    family_id: str
    prefix_id: str
    policy_context: Mapping

    def __post_init__(self):
        for name in ("expected_profile_id", "family_id", "prefix_id"):
            checked_id(getattr(self, name), name)
        if type(self.expected_boundary_cursor) is not int or self.expected_boundary_cursor < 0:
            fail("expected_boundary_cursor must be nonnegative integer")
        _freeze_mappings(self, ("policy_context",))


@dataclass(frozen=True, slots=True)
class BranchContext(_PayloadMixin):
    family_id: str
    prefix_id: str
    branch_id: str
    policy_context: Mapping
    sampling_context: Mapping
    instance_id: str

    def __post_init__(self):
        for name in ("family_id", "prefix_id", "branch_id", "instance_id"):
            checked_id(getattr(self, name), name)
        _freeze_mappings(self, ("policy_context", "sampling_context"))


@dataclass(frozen=True, slots=True)
class CleanupReceipt(_PayloadMixin):
    success: bool
    failed_resources: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.success) is not bool:
            fail("cleanup success must be bool")
        object.__setattr__(self, "failed_resources", tuple(self.failed_resources))
        object.__setattr__(self, "errors", tuple(self.errors))
        if self.success and (self.failed_resources or self.errors):
            fail("successful cleanup cannot have unresolved failures")


@dataclass(slots=True)
class BoundaryStateHandle:
    value: dict = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeParts:
    engine: Any
    graph: Any
    binding: Any
    env: Any
    boundary_state: BoundaryStateHandle
    refs: Any
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContinuationSnapshot:
    data: bytes

    def __post_init__(self):
        decode_json(self.data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def size_bytes(self) -> int:
        return len(self.data)


class EngineStateProvider(Protocol):
    def services(self, engine): ...
    def inspect(self, runtime, topology) -> tuple[Violation, ...]: ...
    def export_state(self, runtime, refs) -> dict: ...
    def validate_payload(self, state, topology, profile) -> None: ...
    def allocate_empty(self, descriptor, cleanup): ...
    def attach(self, engine, graph, refs) -> None: ...
    def restore_into(self, engine, state, refs) -> None: ...
    def validate_restored(self, engine, state, refs) -> None: ...


class ModelStateAdapter(Protocol):
    def reference_objects(self, graph) -> dict: ...
    def descriptor(self) -> dict: ...
    def topology(self) -> dict: ...
    def obligations(self) -> tuple[StateObligation, ...]: ...
    def inspect(self, graph) -> tuple[Violation, ...]: ...
    def export_state(self, graph, refs) -> dict: ...
    def validate_payload(self, state, topology, profile) -> None: ...
    def validate_composition(self, engine_state, model_state, boundary_state, logical_context) -> None: ...
    def allocate_shell(self, descriptor, services, cleanup): ...
    def restore_into(self, graph, state) -> None: ...
    def rebind(self, graph, refs, services) -> None: ...
    def make_binding(self, graph, services, boundary_state): ...
    def validate_restored(self, graph, engine_view) -> None: ...


class RLBoundaryProvider(Protocol):
    def cursor(self, runtime: RuntimeParts) -> int: ...
    def validate_branch(self, saved_logical, branch: BranchContext) -> None: ...
    def inspect(self, runtime, engine_view) -> tuple[Violation, ...]: ...
    def export_state(self, runtime) -> dict: ...
    def validate_payload(self, state, policy_context, profile) -> None: ...
    def allocate_state(self, descriptor, cleanup) -> BoundaryStateHandle: ...
    def bind_uninitialized(self, binding, engine, boundary_state, descriptor, cleanup,
                           *, branch: BranchContext | None = None): ...
    def restore_into(self, env, boundary_state, state) -> None: ...
    def validate_restored(self, env, model_view, engine_view, *, expected_state,
                          branch: BranchContext | None = None) -> None: ...


class FreshFactory(Protocol):
    def create_episode(self, request: ResetRequest, cleanup) -> RuntimeParts: ...


@dataclass(frozen=True, slots=True)
class ContinuationBundle:
    profile: ProfileDescriptor
    engine: EngineStateProvider
    model: ModelStateAdapter
    boundary: RLBoundaryProvider
    fresh_factory: FreshFactory
