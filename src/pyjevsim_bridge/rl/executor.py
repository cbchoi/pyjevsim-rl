"""Decision-boundary control for duck-typed pyjevsim ``SysExecutor`` objects."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Protocol, cast

from pyjevsim_bridge.rl.contracts import ExecutorContractError, ExecutorProtocol


class DecisionBoundary(Protocol):
    """Resolve the logical time to which an executor should advance."""

    def resolve(self, executor: ExecutorProtocol) -> float:
        """Return a finite target time no earlier than the executor clock."""
        ...


@dataclass(frozen=True)
class FixedDeltaBoundary:
    """Advance by the same positive simulation-time interval on every step."""

    delta: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("fixed decision delta must be finite and positive")

    def resolve(self, executor: ExecutorProtocol) -> float:
        return float(executor.get_global_time()) + self.delta


@dataclass(frozen=True)
class NextEventBoundary:
    """Advance through the next scheduled pyjevsim event time."""

    def resolve(self, executor: ExecutorProtocol) -> float:
        return float(executor.get_next_event_time())


@dataclass(frozen=True)
class BindingDecisionBoundary:
    """Use the decision time supplied by an episode binding callback."""

    decision_time: object

    def resolve(self, executor: ExecutorProtocol) -> float:
        del executor
        callback = self.decision_time
        if not callable(callback):
            raise TypeError("decision_time must be callable")
        return float(callback())


class ExecutorSemanticCapability(StrEnum):
    """Independently qualified semantics at the ``SysExecutor`` boundary.

    A runtime version alone is insufficient evidence for these behaviours:
    scheduler and message-delivery implementations can change without changing
    the public ``step`` signature.  Callers that depend on a capability must
    supply qualification evidence and request it explicitly.
    """

    CONFLUENT_TRANSITION = "confluent_transition"
    ZERO_TIME_CASCADE = "zero_time_cascade"
    MULTI_OUTPUT_BAG = "multi_output_bag"
    SAME_TIME_MESSAGE_ORDERING = "same_time_message_ordering"
    SAME_TIME_MODEL_ORDERING = "same_time_model_ordering"
    MESSAGE_MUTATION_ISOLATION = "message_mutation_isolation"
    EXECUTOR_CALLBACK_TIME = "executor_callback_time"
    MODEL_CALLBACK_TIME = "model_callback_time"


@dataclass(frozen=True)
class ExecutorSemanticRequirement:
    """One requested capability with its exact qualified scope, if any.

    A bare :class:`ExecutorSemanticCapability` remains accepted for unscoped
    claims.  It deliberately cannot consume evidence whose positive claim is
    narrower: the caller must name that scope exactly so a per-port ordering
    observation cannot be mistaken for a global ordering guarantee.
    """

    capability: ExecutorSemanticCapability
    scope: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.capability, ExecutorSemanticCapability):
            raise TypeError("executor semantic requirement must use the capability enum")
        if self.scope is not None and (
            not isinstance(self.scope, str) or not self.scope.strip()
        ):
            raise ValueError("executor semantic requirement scope must be non-empty")


_MISSING_ACCESSOR = object()


@dataclass(frozen=True)
class ExecutorQualificationProfile:
    """Controlled semantic claims for one exact executor implementation.

    ``supported`` records observed semantics. ``limitations`` records negative
    observations with a concise reason. ``scopes`` narrows otherwise ambiguous
    positive observations. ``capability_modules`` binds every claim to the
    implementation modules that establish it. The union must exactly match the
    normalized source fingerprints, including the executor class's own module.

    A driver trusts the built-in reviewed profile by default. Additional
    profiles require an explicit trust decision at driver construction; merely
    constructing evidence with an arbitrary revision or source digest is not a
    qualification decision.
    """

    profile_id: str
    issuer: str
    executor_module: str
    executor_qualname: str
    implementation: str
    version: str
    revision: str
    qualification_source: str
    source_sha256: tuple[tuple[str, str], ...]
    supported: frozenset[ExecutorSemanticCapability]
    limitations: tuple[tuple[ExecutorSemanticCapability, str], ...] = ()
    scopes: tuple[tuple[ExecutorSemanticCapability, str], ...] = ()
    capability_modules: tuple[
        tuple[ExecutorSemanticCapability, tuple[str, ...]], ...
    ] = ()
    distribution: str | None = None

    def __post_init__(self) -> None:
        supported = frozenset(self.supported)
        object.__setattr__(self, "supported", supported)
        limitations = tuple(self.limitations)
        object.__setattr__(self, "limitations", limitations)
        scopes = tuple(self.scopes)
        object.__setattr__(self, "scopes", scopes)
        source_sha256 = tuple(self.source_sha256)
        object.__setattr__(self, "source_sha256", source_sha256)
        capability_modules = tuple(
            (capability, tuple(modules))
            for capability, modules in self.capability_modules
        )
        object.__setattr__(self, "capability_modules", capability_modules)
        for label, value in (
            ("profile ID", self.profile_id),
            ("issuer", self.issuer),
            ("executor module", self.executor_module),
            ("executor qualified name", self.executor_qualname),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"executor qualification {label} must be non-empty")
        if not isinstance(self.implementation, str) or not self.implementation.strip():
            raise ValueError("executor qualification implementation must be non-empty")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("executor qualification version must be non-empty")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("executor qualification revision must be non-empty")
        if (
            not isinstance(self.qualification_source, str)
            or not self.qualification_source.strip()
        ):
            raise ValueError("executor qualification source must be non-empty")
        if self.distribution is not None and (
            not isinstance(self.distribution, str) or not self.distribution.strip()
        ):
            raise ValueError("executor qualification distribution must be non-empty")
        if not source_sha256:
            raise ValueError("executor qualification requires source fingerprints")
        seen_modules: set[str] = set()
        for module_name, digest in source_sha256:
            if not isinstance(module_name, str) or not module_name.strip():
                raise ValueError("executor qualification source module must be non-empty")
            if module_name in seen_modules:
                raise ValueError(
                    f"duplicate executor qualification source module: {module_name!r}"
                )
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"executor qualification source digest for {module_name!r} "
                    "must be lowercase SHA-256"
                )
            seen_modules.add(module_name)
        if self.executor_module not in seen_modules:
            raise ValueError(
                "executor qualification must fingerprint its executor module "
                f"{self.executor_module!r}"
            )
        if any(
            not isinstance(capability, ExecutorSemanticCapability)
            for capability in supported
        ):
            raise TypeError("supported executor capabilities must use the enum")

        limited: set[ExecutorSemanticCapability] = set()
        for capability, reason in limitations:
            if not isinstance(capability, ExecutorSemanticCapability):
                raise TypeError("limited executor capabilities must use the enum")
            if capability in supported:
                raise ValueError(
                    f"executor capability {capability.value!r} cannot be both "
                    "supported and limited"
                )
            if capability in limited:
                raise ValueError(
                    f"duplicate executor capability limitation: {capability.value!r}"
                )
            if not reason.strip():
                raise ValueError("executor capability limitation reason must be non-empty")
            limited.add(capability)

        scoped: set[ExecutorSemanticCapability] = set()
        for capability, scope in scopes:
            if capability not in supported:
                raise ValueError(
                    f"scope for unsupported executor capability: {capability.value!r}"
                )
            if capability in scoped:
                raise ValueError(
                    f"duplicate executor capability scope: {capability.value!r}"
                )
            if not isinstance(scope, str) or not scope.strip():
                raise ValueError("executor capability scope must be non-empty")
            scoped.add(capability)

        claimed = supported | limited
        mapped: set[ExecutorSemanticCapability] = set()
        mapped_modules: set[str] = set()
        for capability, modules in capability_modules:
            if not isinstance(capability, ExecutorSemanticCapability):
                raise TypeError("executor module claims must use the capability enum")
            if capability not in claimed:
                raise ValueError(
                    f"module binding for unclaimed capability: {capability.value!r}"
                )
            if capability in mapped:
                raise ValueError(
                    f"duplicate capability module binding: {capability.value!r}"
                )
            if not modules:
                raise ValueError(
                    f"capability {capability.value!r} requires source modules"
                )
            for module_name in modules:
                if module_name not in seen_modules:
                    raise ValueError(
                        f"capability {capability.value!r} references unqualified "
                        f"module {module_name!r}"
                    )
                mapped_modules.add(module_name)
            mapped.add(capability)
        if mapped != claimed:
            missing = ", ".join(
                sorted(capability.value for capability in claimed - mapped)
            )
            raise ValueError(
                "every executor capability claim requires a source-module "
                f"binding; missing: {missing}"
            )
        if mapped_modules != seen_modules:
            unrelated = ", ".join(sorted(seen_modules - mapped_modules))
            raise ValueError(
                "executor qualification contains source fingerprints unrelated "
                f"to its claims: {unrelated}"
            )

    def limitation(self, capability: ExecutorSemanticCapability) -> str | None:
        """Return the recorded limitation reason, if one exists."""

        return dict(self.limitations).get(capability)

    def scope(self, capability: ExecutorSemanticCapability) -> str | None:
        """Return the qualified scope of a supported capability, if narrowed."""

        return dict(self.scopes).get(capability)

    def issue(self, executor_type: type[object]) -> ExecutorSemanticEvidence:
        """Issue evidence for the exact runtime class named by this profile."""

        actual = f"{executor_type.__module__}.{executor_type.__qualname__}"
        expected = f"{self.executor_module}.{self.executor_qualname}"
        if actual != expected:
            raise ValueError(
                "executor qualification profile/type mismatch: "
                f"profile names {expected}, runtime type is {actual}"
            )
        return ExecutorSemanticEvidence(executor_type=executor_type, profile=self)

    def verify_runtime(self, executor: object) -> None:
        """Verify that this profile fingerprints the exact loaded runtime."""

        actual_type = type(executor)
        actual = f"{actual_type.__module__}.{actual_type.__qualname__}"
        expected = f"{self.executor_module}.{self.executor_qualname}"
        if actual != expected:
            raise ExecutorContractError(
                "executor qualification profile/type mismatch: "
                f"profile qualifies {expected}, runtime is {actual}"
            )

        self.verify_loaded_sources()

    def verify_loaded_sources(self) -> None:
        """Verify distribution and source identity before factory execution."""

        if self.distribution is not None:
            try:
                observed_version = metadata.version(self.distribution)
            except metadata.PackageNotFoundError as exc:
                raise ExecutorContractError(
                    "executor semantic evidence cannot verify distribution "
                    f"{self.distribution!r}"
                ) from exc
            if observed_version != self.version:
                raise ExecutorContractError(
                    "executor semantic evidence version mismatch: "
                    f"expected {self.version!r}, observed {observed_version!r}"
                )

        for module_name, expected_digest in self.source_sha256:
            try:
                module = sys.modules.get(module_name) or importlib.import_module(
                    module_name
                )
                source_path = Path(module.__file__ or "")
                source = source_path.read_bytes().replace(b"\r\n", b"\n")
            except (AttributeError, ImportError, OSError) as exc:
                raise ExecutorContractError(
                    "executor semantic evidence cannot read qualified source "
                    f"module {module_name!r}: {exc}"
                ) from exc
            observed_digest = hashlib.sha256(source).hexdigest()
            if observed_digest != expected_digest:
                raise ExecutorContractError(
                    "executor semantic evidence source mismatch: "
                    f"module {module_name!r} expected {expected_digest}, "
                    f"observed {observed_digest}"
                )


@dataclass(frozen=True)
class ExecutorSemanticEvidence:
    """An exact runtime class bound to one controlled qualification profile."""

    executor_type: type[object]
    profile: ExecutorQualificationProfile

    @property
    def supported(self) -> frozenset[ExecutorSemanticCapability]:
        return self.profile.supported

    def limitation(self, capability: ExecutorSemanticCapability) -> str | None:
        return self.profile.limitation(capability)

    def scope(self, capability: ExecutorSemanticCapability) -> str | None:
        return self.profile.scope(capability)

    def verify_runtime(
        self,
        executor: object,
        trusted_profiles: dict[str, ExecutorQualificationProfile],
    ) -> None:
        """Require an independently trusted profile before checking runtime."""

        trusted = self._trusted_profile(trusted_profiles)
        if self.executor_type is not type(executor):
            raise ExecutorContractError(
                "executor semantic evidence type mismatch: evidence was issued "
                f"for {self.executor_type.__module__}."
                f"{self.executor_type.__qualname__}"
            )
        trusted.verify_runtime(executor)

    def verify_pre_factory(
        self,
        trusted_profiles: dict[str, ExecutorQualificationProfile],
    ) -> None:
        """Verify declared runtime identity before model factory callbacks."""

        trusted = self._trusted_profile(trusted_profiles)
        actual = f"{self.executor_type.__module__}.{self.executor_type.__qualname__}"
        expected = f"{trusted.executor_module}.{trusted.executor_qualname}"
        if actual != expected:
            raise ExecutorContractError(
                "executor semantic evidence declared type mismatch: "
                f"profile qualifies {expected}, evidence declares {actual}"
            )
        trusted.verify_loaded_sources()

    def _trusted_profile(
        self,
        trusted_profiles: dict[str, ExecutorQualificationProfile],
    ) -> ExecutorQualificationProfile:
        trusted = trusted_profiles.get(self.profile.profile_id)
        if trusted is None:
            raise ExecutorContractError(
                "executor semantic evidence uses an untrusted qualification "
                f"profile {self.profile.profile_id!r} from {self.profile.issuer!r}"
            )
        if trusted != self.profile:
            raise ExecutorContractError(
                "executor semantic evidence does not match trusted profile "
                f"{self.profile.profile_id!r} from issuer {trusted.issuer!r}"
            )
        return trusted


@dataclass(frozen=True)
class ExecutorQualificationPolicy:
    """Environment-owned admission policy for an executor runtime.

    Model bindings supply an executor, but they do not select which semantic
    evidence or trust roots admit it.  An orchestrator constructs this frozen
    policy and injects it into :class:`PyJevSimEnv`; ``None`` at that boundary
    is an explicit development/non-claim mode.
    """

    semantic_evidence: ExecutorSemanticEvidence
    required_semantics: tuple[
        ExecutorSemanticCapability | ExecutorSemanticRequirement, ...
    ]
    additional_trusted_profiles: tuple[ExecutorQualificationProfile, ...] = ()
    policy_id: str = field(init=False)
    claim_grade: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_evidence, ExecutorSemanticEvidence):
            raise TypeError(
                "executor qualification policy evidence must use "
                "ExecutorSemanticEvidence"
            )
        requirements = tuple(self.required_semantics)
        profiles = tuple(self.additional_trusted_profiles)
        object.__setattr__(self, "required_semantics", requirements)
        object.__setattr__(self, "additional_trusted_profiles", profiles)
        if not requirements:
            raise ValueError(
                "executor qualification policy requires at least one semantic"
            )
        for requirement in requirements:
            if not isinstance(
                requirement,
                (ExecutorSemanticCapability, ExecutorSemanticRequirement),
            ):
                raise TypeError(
                    "executor qualification policy requirements must use a "
                    "capability or ExecutorSemanticRequirement"
                )
        seen_requirements: dict[ExecutorSemanticCapability, str | None] = {}
        for requirement in requirements:
            capability = (
                requirement
                if isinstance(requirement, ExecutorSemanticCapability)
                else requirement.capability
            )
            scope = (
                None
                if isinstance(requirement, ExecutorSemanticCapability)
                else requirement.scope
            )
            if capability in seen_requirements:
                if seen_requirements[capability] == scope:
                    raise ValueError(
                        "duplicate executor qualification policy requirement: "
                        f"{capability.value!r} with scope {scope!r}"
                    )
                raise ValueError(
                    "conflicting scopes requested for executor capability "
                    f"{capability.value!r}"
                )
            seen_requirements[capability] = scope
        seen_profile_ids: set[str] = set()
        for profile in profiles:
            if not isinstance(profile, ExecutorQualificationProfile):
                raise TypeError(
                    "executor qualification policy trust roots must use "
                    "ExecutorQualificationProfile"
                )
            if profile.profile_id in seen_profile_ids:
                raise ValueError(
                    "duplicate executor qualification policy trust root: "
                    f"{profile.profile_id!r}"
                )
            seen_profile_ids.add(profile.profile_id)
        canonical = {
            "evidence_profile": self._profile_identity(
                self.semantic_evidence.profile
            ),
            "executor_type": (
                f"{self.semantic_evidence.executor_type.__module__}."
                f"{self.semantic_evidence.executor_type.__qualname__}"
            ),
            "required_semantics": sorted(
                (
                    self._requirement_identity(requirement)
                    for requirement in requirements
                ),
                key=lambda item: json.dumps(item, sort_keys=True),
            ),
            "additional_trust": [
                self._profile_identity(profile)
                for profile in sorted(profiles, key=lambda item: item.profile_id)
            ],
        }
        encoded = json.dumps(
            canonical, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        object.__setattr__(
            self,
            "policy_id",
            f"executor-policy/sha256:{hashlib.sha256(encoded).hexdigest()}",
        )
        controlled = globals().get("PINNED_PYJEVSIM_2_1_2_PROFILE")
        object.__setattr__(
            self,
            "claim_grade",
            controlled is not None
            and self.semantic_evidence.profile is controlled
            and not profiles,
        )

    def preflight(self) -> None:
        """Verify trust, sources, and required claims before factory entry."""

        trusted = self._trusted_profiles()
        self.semantic_evidence.verify_pre_factory(trusted)
        _verify_semantic_requirements(
            self.semantic_evidence, self.required_semantics
        )

    def _trusted_profiles(self) -> dict[str, ExecutorQualificationProfile]:
        trusted = {
            PINNED_PYJEVSIM_2_1_2_PROFILE.profile_id:
                PINNED_PYJEVSIM_2_1_2_PROFILE
        }
        trusted.update(
            {profile.profile_id: profile for profile in self.additional_trusted_profiles}
        )
        return trusted

    @staticmethod
    def _profile_identity(profile: ExecutorQualificationProfile) -> object:
        return {
            "profile_id": profile.profile_id,
            "issuer": profile.issuer,
            "executor_module": profile.executor_module,
            "executor_qualname": profile.executor_qualname,
            "implementation": profile.implementation,
            "version": profile.version,
            "revision": profile.revision,
            "qualification_source": profile.qualification_source,
            "source_sha256": profile.source_sha256,
            "supported": sorted(item.value for item in profile.supported),
            "limitations": sorted(
                (capability.value, reason)
                for capability, reason in profile.limitations
            ),
            "scopes": sorted(
                (capability.value, scope)
                for capability, scope in profile.scopes
            ),
            "capability_modules": sorted(
                (capability.value, modules)
                for capability, modules in profile.capability_modules
            ),
            "distribution": profile.distribution,
        }

    @staticmethod
    def _requirement_identity(
        requirement: ExecutorSemanticCapability | ExecutorSemanticRequirement,
    ) -> object:
        if isinstance(requirement, ExecutorSemanticCapability):
            return {"capability": requirement.value, "scope": None}
        return {
            "capability": requirement.capability.value,
            "scope": requirement.scope,
        }


PINNED_PYJEVSIM_2_1_2_PROFILE = ExecutorQualificationProfile(
    profile_id="gorti.pyjevsim-rl/2.1.2/9893099",
    issuer="gorti-controlled-qualification/v1",
    executor_module="pyjevsim.system_executor",
    executor_qualname="SysExecutor",
    implementation="pyjevsim",
    version="2.1.2",
    revision="9893099b47aae89a3432c668b18ec4b6b15c043e",
    qualification_source="TEST-RL-EXEC pinned real-runtime suite",
    source_sha256=(
        (
            "pyjevsim.system_executor",
            "12e6e2e0b2906c00fbc9a5097c747af6f81f37138b50837a341003275ab505ee",
        ),
        (
            "pyjevsim.schedule_queue",
            "e6c06a2e5489e05da16cbe35a4d79c6a84b6fca14582db4e604be1960f15d004",
        ),
        (
            "pyjevsim.behavior_executor",
            "cbe2d9ff4b73161c27f14fa7bfea4552c5a79250c5cf2234b6a5c425e3457f2a",
        ),
        (
            "pyjevsim.behavior_model",
            "282d647a7fc0bf5624d25dcffd4250e5f690f3e6d9893fb2e0699da0073f9f53",
        ),
        (
            "pyjevsim.system_message",
            "5ed622d08ad4848a0822d02a2c12577cc8188d51077749facaa56ec9952392aa",
        ),
        (
            "pyjevsim.message_deliverer",
            "e966f1ada63fda40bf7ae4b61742c59d607c70e677d0611f0d56af2efed9ad50",
        ),
        (
            "pyjevsim.system_object",
            "447b72e53517028da6ba7ad61604ba480b56cb016f33e0963396863bb7b19d5f",
        ),
        (
            "pyjevsim.definition",
            "f07937da39558f0bf0c8831ac848ee59fbf62b47e4f4053ecfb7ef22f13d9a4f",
        ),
    ),
    supported=frozenset(
        {
            ExecutorSemanticCapability.CONFLUENT_TRANSITION,
            ExecutorSemanticCapability.ZERO_TIME_CASCADE,
            ExecutorSemanticCapability.MULTI_OUTPUT_BAG,
            ExecutorSemanticCapability.SAME_TIME_MESSAGE_ORDERING,
            ExecutorSemanticCapability.EXECUTOR_CALLBACK_TIME,
        }
    ),
    limitations=(
        (
            ExecutorSemanticCapability.SAME_TIME_MODEL_ORDERING,
            "ScheduleQueue uses unordered set buckets",
        ),
        (
            ExecutorSemanticCapability.MESSAGE_MUTATION_ISOLATION,
            "fan-out shares one SysMessage reference",
        ),
        (
            ExecutorSemanticCapability.MODEL_CALLBACK_TIME,
            "BehaviorModel.global_time updates after callbacks",
        ),
    ),
    scopes=(
        (
            ExecutorSemanticCapability.SAME_TIME_MESSAGE_ORDERING,
            "insertion order for one receiver/input port at one timestamp",
        ),
    ),
    capability_modules=(
        (
            ExecutorSemanticCapability.CONFLUENT_TRANSITION,
            (
                "pyjevsim.system_executor",
                "pyjevsim.behavior_executor",
                "pyjevsim.behavior_model",
                "pyjevsim.message_deliverer",
                "pyjevsim.definition",
            ),
        ),
        (
            ExecutorSemanticCapability.ZERO_TIME_CASCADE,
            (
                "pyjevsim.system_executor",
                "pyjevsim.schedule_queue",
                "pyjevsim.behavior_executor",
                "pyjevsim.definition",
            ),
        ),
        (
            ExecutorSemanticCapability.MULTI_OUTPUT_BAG,
            (
                "pyjevsim.system_executor",
                "pyjevsim.message_deliverer",
                "pyjevsim.system_message",
                "pyjevsim.definition",
            ),
        ),
        (
            ExecutorSemanticCapability.SAME_TIME_MESSAGE_ORDERING,
            (
                "pyjevsim.system_executor",
                "pyjevsim.system_message",
                "pyjevsim.system_object",
                "pyjevsim.definition",
            ),
        ),
        (
            ExecutorSemanticCapability.EXECUTOR_CALLBACK_TIME,
            ("pyjevsim.system_executor", "pyjevsim.definition"),
        ),
        (
            ExecutorSemanticCapability.SAME_TIME_MODEL_ORDERING,
            (
                "pyjevsim.system_executor",
                "pyjevsim.schedule_queue",
                "pyjevsim.definition",
            ),
        ),
        (
            ExecutorSemanticCapability.MESSAGE_MUTATION_ISOLATION,
            (
                "pyjevsim.system_executor",
                "pyjevsim.message_deliverer",
                "pyjevsim.system_message",
                "pyjevsim.definition",
            ),
        ),
        (
            ExecutorSemanticCapability.MODEL_CALLBACK_TIME,
            (
                "pyjevsim.system_executor",
                "pyjevsim.behavior_executor",
                "pyjevsim.behavior_model",
                "pyjevsim.definition",
            ),
        ),
    ),
    distribution="pyjevsim",
)


def _verify_semantic_requirements(
    evidence: ExecutorSemanticEvidence,
    requirements: Iterable[
        ExecutorSemanticCapability | ExecutorSemanticRequirement
    ],
) -> None:
    normalized: dict[
        ExecutorSemanticCapability, ExecutorSemanticRequirement
    ] = {}
    for value in requirements:
        requirement = (
            ExecutorSemanticRequirement(value)
            if isinstance(value, ExecutorSemanticCapability)
            else value
        )
        existing = normalized.get(requirement.capability)
        if existing is not None and existing != requirement:
            raise ValueError(
                "conflicting scopes requested for executor capability "
                f"{requirement.capability.value!r}"
            )
        normalized[requirement.capability] = requirement

    requested = frozenset(normalized)
    unsupported = requested - evidence.supported
    if unsupported:
        details = []
        for capability in sorted(unsupported, key=lambda item: item.value):
            reason = evidence.limitation(capability) or "not qualified"
            details.append(f"{capability.value}: {reason}")
        raise ExecutorContractError(
            "executor does not satisfy required semantics: " + "; ".join(details)
        )

    scope_mismatches: list[str] = []
    for capability in sorted(requested, key=lambda item: item.value):
        requested_scope = normalized[capability].scope
        qualified_scope = evidence.scope(capability)
        if requested_scope == qualified_scope:
            continue
        if qualified_scope is not None and requested_scope is None:
            scope_mismatches.append(
                f"{capability.value}: evidence is qualified only for scope "
                f"{qualified_scope!r}; the exact scope must be requested"
            )
        elif qualified_scope is None:
            scope_mismatches.append(
                f"{capability.value}: requested scope {requested_scope!r} "
                "is not qualified"
            )
        else:
            scope_mismatches.append(
                f"{capability.value}: requested scope {requested_scope!r} "
                f"does not match qualified scope {qualified_scope!r}"
            )
    if scope_mismatches:
        raise ExecutorContractError(
            "executor does not satisfy required semantic scopes: "
            + "; ".join(scope_mismatches)
        )


@dataclass(frozen=True)
class ExecutorStep:
    """Outputs and time interval committed by one executor advancement."""

    previous_time: float
    logical_time: float
    external_events: object


class ExecutorDriver:
    """Validate monotonic time around the canonical executor ``step`` seam.

    Only the repository-controlled pinned profile is trusted by default.
    ``additional_trusted_profiles`` is a caller-local escape hatch for tests or
    an externally governed qualification workflow; supplying one is not, by
    itself, production admission or repository trust.
    """

    def __init__(
        self,
        executor: ExecutorProtocol,
        *,
        semantic_evidence: ExecutorSemanticEvidence | None = None,
        required_semantics: Iterable[
            ExecutorSemanticCapability | ExecutorSemanticRequirement
        ] = (),
        additional_trusted_profiles: Iterable[ExecutorQualificationProfile] = (),
    ) -> None:
        self._executor = executor
        self._closed = False
        self._has_advanced = False
        self._semantic_evidence = semantic_evidence
        self._qualification_policy: ExecutorQualificationPolicy | None = None
        self._trusted_profiles = {
            PINNED_PYJEVSIM_2_1_2_PROFILE.profile_id: PINNED_PYJEVSIM_2_1_2_PROFILE
        }
        for profile in additional_trusted_profiles:
            existing = self._trusted_profiles.get(profile.profile_id)
            if existing is not None and existing != profile:
                raise ValueError(
                    "conflicting executor qualification profile ID: "
                    f"{profile.profile_id!r}"
                )
            self._trusted_profiles[profile.profile_id] = profile
        self.require_semantics(*required_semantics)

    @property
    def executor(self) -> ExecutorProtocol:
        return self._executor

    @property
    def semantic_evidence(self) -> ExecutorSemanticEvidence | None:
        """Return qualification evidence supplied for this executor."""

        return self._semantic_evidence

    @property
    def qualification_policy(self) -> ExecutorQualificationPolicy | None:
        """Return the environment-owned policy applied before execution."""

        return self._qualification_policy

    def apply_qualification_policy(
        self, policy: ExecutorQualificationPolicy
    ) -> None:
        """Apply one immutable production admission policy before execution.

        The method is separate from construction so a lifecycle owner can
        retain and close the driver when qualification fails.  It cannot be
        used to replace legacy caller evidence or to swap policy after
        admission.
        """

        if not isinstance(policy, ExecutorQualificationPolicy):
            raise TypeError("executor qualification policy has an invalid type")
        if self._closed:
            raise ExecutorContractError("executor driver is closed")
        if self._has_advanced:
            raise ExecutorContractError(
                "executor qualification policy must be fixed before advance"
            )
        if self._qualification_policy is not None:
            if self._qualification_policy == policy:
                return
            raise ExecutorContractError(
                "executor qualification policy is already fixed for this driver"
            )
        if self._semantic_evidence is not None:
            raise ExecutorContractError(
                "executor qualification policy cannot replace caller evidence"
            )

        previous_evidence = self._semantic_evidence
        previous_profiles = self._trusted_profiles.copy()
        try:
            self._semantic_evidence = policy.semantic_evidence
            for profile in policy.additional_trusted_profiles:
                existing = self._trusted_profiles.get(profile.profile_id)
                if existing is not None and existing != profile:
                    raise ExecutorContractError(
                        "executor qualification policy conflicts with trusted "
                        f"profile ID {profile.profile_id!r}"
                    )
                self._trusted_profiles[profile.profile_id] = profile
            self.require_semantics(*policy.required_semantics)
        except BaseException:
            self._semantic_evidence = previous_evidence
            self._trusted_profiles = previous_profiles
            raise
        self._qualification_policy = policy

    @property
    def logical_time(self) -> float:
        return self._finite_time(self._executor.get_global_time(), "global time")

    def advance(self, boundary: DecisionBoundary) -> ExecutorStep:
        if self._closed:
            raise ExecutorContractError("executor driver is closed")
        self._has_advanced = True

        previous = self.logical_time
        target = self._finite_time(boundary.resolve(self._executor), "decision time")
        if target < previous:
            raise ExecutorContractError(
                f"decision time regressed from {previous!r} to {target!r}"
            )

        external_events = self._normalize_external_events(self._executor.step(target))
        committed = self.logical_time
        if committed < previous:
            raise ExecutorContractError(
                f"executor clock regressed from {previous!r} to {committed!r}"
            )
        if committed > target:
            raise ExecutorContractError(
                f"executor advanced past decision time {target!r} to {committed!r}"
            )
        if committed != target:
            raise ExecutorContractError(
                f"executor stopped before decision time {target!r} at {committed!r}"
            )
        return ExecutorStep(previous, committed, external_events)

    def require_semantics(
        self,
        *requirements: ExecutorSemanticCapability | ExecutorSemanticRequirement,
    ) -> None:
        """Fail closed unless exact-runtime evidence supports every requirement."""

        if not requirements:
            return
        for value in requirements:
            if not isinstance(
                value,
                (ExecutorSemanticCapability, ExecutorSemanticRequirement),
            ):
                raise TypeError(
                    "required executor semantics must use a capability or "
                    "ExecutorSemanticRequirement"
                )

        evidence = self._semantic_evidence
        if evidence is None:
            names = ", ".join(
                sorted(
                    (
                        item.value
                        if isinstance(item, ExecutorSemanticCapability)
                        else item.capability.value
                    )
                    for item in requirements
                )
            )
            raise ExecutorContractError(
                f"executor semantic evidence is required for: {names}"
            )
        evidence.verify_runtime(self._executor, self._trusted_profiles)
        _verify_semantic_requirements(evidence, requirements)

    def is_terminated(self) -> bool:
        return bool(self._executor.is_terminated())

    def close(self) -> None:
        if self._closed:
            return
        self._executor.terminate_simulation()
        self._closed = True

    @staticmethod
    def _normalize_external_events(events: object) -> object:
        """Preserve a pyjevsim output bag as a value-oriented ordered sequence.

        Pinned ``SysExecutor.step`` returns a deep-copied ``deque``.  The RL
        callback contract accepts value-semantic sequences and rejects mutable
        container implementations. Its output items contain mutable custom
        ``SysMessage`` objects, so those are projected through their public
        accessors into value-semantic metadata and payload tuples. Item order,
        timestamps, duplicate messages, and values inside the already isolated
        copy are otherwise left unchanged.
        """

        if isinstance(events, deque):
            return tuple(
                ExecutorDriver._normalize_external_event(event, index)
                for index, event in enumerate(events)
            )
        return events

    @staticmethod
    def _normalize_external_event(event: object, index: int) -> object:
        if not isinstance(event, tuple) or len(event) != 2:
            return event
        instant, message = event
        accessor_names = ("get_src", "get_dst", "get_msg_time", "retrieve")
        try:
            accessors = {
                name: getattr(message, name, _MISSING_ACCESSOR)
                for name in accessor_names
            }
        except BaseException as exc:
            raise ExecutorContractError(
                f"executor output event {index} accessor discovery failed: {exc}"
            ) from exc

        present = {
            name for name, accessor in accessors.items()
            if accessor is not _MISSING_ACCESSOR
        }
        message_type = type(message)
        message_module = getattr(message_type, "__module__", "")
        sys_message_identity = (
            message_type.__name__ == "SysMessage"
            or (
                isinstance(message_module, str)
                and message_module.startswith("pyjevsim.")
            )
        )
        if not present and not sys_message_identity:
            return event
        invalid = [
            name
            for name in accessor_names
            if accessors[name] is _MISSING_ACCESSOR
            or not callable(accessors[name])
        ]
        if invalid:
            raise ExecutorContractError(
                f"executor output event {index} has an incomplete SysMessage "
                f"surface: {', '.join(invalid)}"
            )
        get_src = cast("Callable[[], object]", accessors["get_src"])
        get_dst = cast("Callable[[], object]", accessors["get_dst"])
        get_msg_time = cast("Callable[[], object]", accessors["get_msg_time"])
        retrieve = cast("Callable[[], Iterable[object]]", accessors["retrieve"])
        try:
            payload = tuple(retrieve())
            projected = {
                "source": get_src(),
                "destination": get_dst(),
                "message_time": get_msg_time(),
                "payload": payload,
            }
        except BaseException as exc:
            raise ExecutorContractError(
                f"executor output event {index} cannot be projected safely: {exc}"
            ) from exc
        return instant, projected

    @staticmethod
    def _finite_time(value: float, label: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ExecutorContractError(f"{label} must be finite, got {result!r}")
        return result
