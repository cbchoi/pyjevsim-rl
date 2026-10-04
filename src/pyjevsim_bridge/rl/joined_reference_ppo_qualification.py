"""Fail-closed joined-gorti reference-PPO qualification evidence boundary."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, Protocol

SCHEMA_VERSION: Final = "joined-reference-ppo-qualification-v1"
CAPABILITY_SCOPE: Final = "reference-ppo-joined-rollout-qualification-only"
EXACT_TRANSITIONS: Final = 4096
EXACT_BATCH_SIZE: Final = 2048
EXACT_BATCH_COUNT: Final = 2
EXACT_ACTORS: Final = 4
EXACT_EVALUATIONS: Final = 16
EXACT_QUALIFICATION_ADAM_STEPS: Final = 320
NEGATIVE_PROBES: Final = frozenset(
    {
        "emulation",
        "missing-participant",
        "wrong-declaration",
        "missing-time-enable",
        "missing-sync-callback",
        "stale-generation",
        "wrong-policy-activation",
        "time-regression",
        "delivery-duplicate",
        "idempotency-conflict",
        "parity-mismatch",
        "cleanup-incomplete",
        "ledger-tamper",
    }
)

RAW_ROLES: Final = frozenset(
    {
        "manifest",
        "runtime-lock",
        "membership-activation",
        "time-grants",
        "delivery",
        "local-parity",
        "policy-recovery",
        "evaluation-isolation",
        "cleanup",
        "negative-probe",
        "prerequisite-ledger",
    }
)


class JoinedQualificationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class JoinedVerificationContext:
    """Filesystem authority used by the evidence verifier.

    Production uses the repository containing this module.  Tests may provide a
    sealed temporary repository, but cannot make evidence paths absolute or
    escape that root.
    """

    repository_root: Path
    prerequisite_file_sha256: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        root = self.repository_root.resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise JoinedQualificationError("verification repository root is invalid")
        object.__setattr__(self, "repository_root", root)
        if self.prerequisite_file_sha256 is not None:
            if set(self.prerequisite_file_sha256) != {"TASK-RL-104", "TASK-RL-105"}:
                raise JoinedQualificationError("test prerequisite lock set differs")
            for value in self.prerequisite_file_sha256.values():
                _digest("test prerequisite digest", value)


def _production_context() -> JoinedVerificationContext:
    return JoinedVerificationContext(Path(__file__).resolve().parents[3])


def _read_bound_file(
    context: JoinedVerificationContext,
    relative: object,
    *,
    size: object,
    digest: object,
) -> bytes:
    if not isinstance(relative, str):
        raise JoinedQualificationError("bound file path must be relative")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise JoinedQualificationError("bound file path escapes repository")
    target = (context.repository_root / Path(*pure.parts)).resolve(strict=True)
    try:
        target.relative_to(context.repository_root)
    except ValueError as exc:
        raise JoinedQualificationError("bound file escapes repository") from exc
    if target.is_symlink() or not target.is_file():
        raise JoinedQualificationError("bound file is not a regular file")
    body = target.read_bytes()
    if len(body) != _positive("bound file size", size) or _sha(body) != _digest(
        "bound file digest", digest
    ):
        raise JoinedQualificationError("bound file failed independent re-hash")
    return body


def _verify_indexed_ledger(body: bytes, root: Path) -> None:
    try:
        ledger = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JoinedQualificationError("prerequisite ledger is invalid JSON") from exc
    entries = ledger.get("entries") if isinstance(ledger, dict) else None
    if not isinstance(entries, list) or not entries:
        raise JoinedQualificationError("prerequisite ledger entries are missing")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not {
            "path", "size_bytes", "sha256"
        }.issubset(entry):
            raise JoinedQualificationError("prerequisite ledger entry is malformed")
        path = entry["path"]
        if not isinstance(path, str) or path in seen:
            raise JoinedQualificationError("prerequisite ledger path is duplicated")
        seen.add(path)
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path:
            raise JoinedQualificationError("prerequisite artifact path escapes ledger")
        target = (root / Path(*pure.parts)).resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise JoinedQualificationError("prerequisite artifact escapes ledger") from exc
        payload = target.read_bytes()
        if target.is_symlink() or len(payload) != _positive(
            "prerequisite artifact size", entry["size_bytes"]
        ) or _sha(payload) != _digest("prerequisite artifact digest", entry["sha256"]):
            raise JoinedQualificationError("prerequisite indexed artifact failed re-hash")


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise JoinedQualificationError("value is not canonical JSON") from exc


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _positive(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise JoinedQualificationError(f"{name} must be a positive integer")
    return value


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise JoinedQualificationError(f"{name} must be SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise JoinedQualificationError(f"{name} must be SHA-256") from exc
    return value


@dataclass(frozen=True, slots=True)
class JoinedReferencePPOPlan:
    actors: int = EXACT_ACTORS
    batch_size: int = EXACT_BATCH_SIZE
    batch_count: int = EXACT_BATCH_COUNT
    transitions: int = EXACT_TRANSITIONS
    evaluations: int = EXACT_EVALUATIONS
    qualification_adam_steps: int = EXACT_QUALIFICATION_ADAM_STEPS
    production_admission: bool = True
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "actors",
            "batch_size",
            "batch_count",
            "transitions",
            "evaluations",
            "qualification_adam_steps",
        ):
            _positive(name, getattr(self, name))
        if type(self.production_admission) is not bool:
            raise JoinedQualificationError("production_admission must be bool")
        object.__setattr__(self, "sha256", _sha(_canonical(self.content())))

    @property
    def exact(self) -> bool:
        return self.content() == {
            "actors": 4,
            "batch_count": 2,
            "batch_size": 2048,
            "evaluations": 16,
            "production_admission": True,
            "qualification_adam_steps": 320,
            "transitions": 4096,
        }

    def content(self) -> dict[str, object]:
        return {
            "actors": self.actors,
            "batch_count": self.batch_count,
            "batch_size": self.batch_size,
            "evaluations": self.evaluations,
            "production_admission": self.production_admission,
            "qualification_adam_steps": self.qualification_adam_steps,
            "transitions": self.transitions,
        }


@dataclass(frozen=True, slots=True)
class JoinedArtifactEntry:
    role: str
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in RAW_ROLES:
            raise JoinedQualificationError("unknown joined evidence role")
        pure = PurePosixPath(self.path)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != self.path:
            raise JoinedQualificationError("artifact path must be normalized and relative")
        _positive("size_bytes", self.size_bytes)
        _digest("sha256", self.sha256)

    def content(self) -> dict[str, object]:
        return {
            "media_type": "application/json",
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class JoinedReferencePPOExecution:
    runner_kind: str
    artifacts: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if self.runner_kind not in {"internal-actual-gorti-v1", "injected-test-double"}:
            raise JoinedQualificationError("runner kind differs from closed schema")
        if set(self.artifacts) != RAW_ROLES:
            raise JoinedQualificationError("execution artifact roles are incomplete")
        if any(not isinstance(value, bytes) or not value for value in self.artifacts.values()):
            raise JoinedQualificationError("execution artifacts must be non-empty bytes")
        object.__setattr__(self, "artifacts", dict(self.artifacts))


class JoinedReferencePPORunner(Protocol):
    def __call__(self, plan: JoinedReferencePPOPlan) -> JoinedReferencePPOExecution: ...


@dataclass(frozen=True, slots=True)
class JoinedReferencePPOCapability:
    plan_sha256: str
    raw_ledger_sha256: str
    evidence_sha256: str
    scope: str = field(init=False, default=CAPABILITY_SCOPE)
    admitted: bool = field(init=False, default=True)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("plan_sha256", "raw_ledger_sha256", "evidence_sha256"):
            _digest(name, getattr(self, name))
        object.__setattr__(
            self,
            "sha256",
            _sha(
                _canonical(
                    {
                        "admitted": True,
                        "evidence_sha256": self.evidence_sha256,
                        "plan_sha256": self.plan_sha256,
                        "raw_ledger_sha256": self.raw_ledger_sha256,
                        "scope": self.scope,
                    }
                )
            ),
        )

    def content(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "evidence_sha256": self.evidence_sha256,
            "plan_sha256": self.plan_sha256,
            "raw_ledger_sha256": self.raw_ledger_sha256,
            "scope": self.scope,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class JoinedReferencePPOBundle:
    plan_sha256: str
    entries: tuple[JoinedArtifactEntry, ...]
    blockers: tuple[str, ...]
    capability: JoinedReferencePPOCapability | None
    admitted: bool = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        entries = tuple(sorted(self.entries, key=lambda item: item.path))
        if {item.role for item in entries} != RAW_ROLES or len(entries) != len(RAW_ROLES):
            raise JoinedQualificationError("bundle evidence roles are incomplete or duplicated")
        blockers = tuple(dict.fromkeys(self.blockers))
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "admitted", not blockers and self.capability is not None)
        object.__setattr__(self, "sha256", _sha(_canonical(self.content(include_sha=False))))

    def content(self, *, include_sha: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "plan_sha256": self.plan_sha256,
            "entries": [item.content() for item in self.entries],
            "blockers": list(self.blockers),
            "capability": None if self.capability is None else self.capability.content(),
        }
        if include_sha:
            value["bundle_sha256"] = self.sha256
        return value


def _object(payload: bytes, schema: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JoinedQualificationError("evidence is not JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise JoinedQualificationError(f"evidence differs from {schema}")
    return value


def _exact_set(name: str, value: object, expected: set[str]) -> None:
    if not isinstance(value, list) or set(value) != expected or len(value) != len(expected):
        raise JoinedQualificationError(f"{name} differs")


def _verify_typed_evidence(
    payloads: Mapping[str, bytes],
    plan: JoinedReferencePPOPlan,
    context: JoinedVerificationContext | None = None,
) -> str:
    verification = _production_context() if context is None else context
    manifest = _object(payloads["manifest"], "joined-manifest-v1")
    if manifest.get("plan_sha256") != plan.sha256 or manifest.get("plan") != plan.content():
        raise JoinedQualificationError("manifest plan differs")
    runtime = _object(payloads["runtime-lock"], "joined-runtime-lock-v1")
    for name in (
        "rtid_sha256",
        "fom_sha256",
        "sdk_sha256",
        "harness_sha256",
        "pyjevsim_sha256",
    ):
        _digest(name, runtime.get(name))
    if runtime.get("execution_mode") != "actual-federation" or runtime.get(
        "diagnostic_emulation"
    ) is not False or runtime.get("rtid_ready") is not True:
        raise JoinedQualificationError("runtime is not actual joined gorti")
    _positive("rtid_pid", runtime.get("rtid_pid"))
    if not isinstance(runtime.get("rtid_path"), str) or not runtime["rtid_path"]:
        raise JoinedQualificationError("rtid path is missing")
    origins = runtime.get("import_origins")
    if not isinstance(origins, dict) or set(origins) != {
        "gorti",
        "pyjevsim",
        "qualification",
    } or any(not isinstance(value, str) or not value for value in origins.values()):
        raise JoinedQualificationError("runtime import origins differ")
    files = runtime.get("files")
    required_file_roles = {"rtid", "fom", "sdk", "harness", "pyjevsim", "qualification"}
    if not isinstance(files, list) or len(files) != len(required_file_roles):
        raise JoinedQualificationError("runtime file lock set differs")
    observed_file_roles: set[str] = set()
    for row in files:
        if not isinstance(row, dict) or set(row) != {
            "role", "path", "size_bytes", "sha256", "rehash_sha256"
        }:
            raise JoinedQualificationError("runtime file lock row differs")
        role = row["role"]
        if role not in required_file_roles or role in observed_file_roles:
            raise JoinedQualificationError("runtime file lock role differs")
        observed_file_roles.add(role)
        digest = _digest("runtime file sha256", row["sha256"])
        if digest != _digest("runtime file rehash", row["rehash_sha256"]):
            raise JoinedQualificationError("runtime supplied rehash differs")
        _read_bound_file(
            verification, row["path"], size=row["size_bytes"], digest=digest
        )
    if observed_file_roles != required_file_roles:
        raise JoinedQualificationError("runtime file lock roles are incomplete")
    children = runtime.get("children")
    expected_children = {"coordinator", *(f"worker-{index}" for index in range(4))}
    if not isinstance(children, list) or len(children) != 5:
        raise JoinedQualificationError("runtime child identities differ")
    child_ids: set[str] = set()
    for child in children:
        if not isinstance(child, dict) or set(child) != {
            "participant_id", "import_origin", "distribution", "version", "source_sha256"
        }:
            raise JoinedQualificationError("runtime child row differs")
        participant = child["participant_id"]
        if participant not in expected_children or participant in child_ids:
            raise JoinedQualificationError("runtime child participant differs")
        child_ids.add(participant)
        if any(not isinstance(child[name], str) or not child[name] for name in (
            "import_origin", "distribution", "version"
        )):
            raise JoinedQualificationError("runtime child provenance differs")
        _digest("runtime child source", child["source_sha256"])

    membership = _object(payloads["membership-activation"], "joined-membership-v1")
    workers = {f"worker-{index}" for index in range(4)}
    expected_participants = workers | {"coordinator"}
    participants = membership.get("participants")
    if not isinstance(participants, list) or {
        row.get("participant_id") for row in participants if isinstance(row, dict)
    } != expected_participants or len(participants) != 5:
        raise JoinedQualificationError("participants differ")
    handles: set[int] = set()
    pids: set[int] = set()
    for row in participants:
        if not isinstance(row, dict):
            raise JoinedQualificationError("participant row differs")
        handles.add(_positive("participant handle", row.get("handle")))
        pids.add(_positive("participant pid", row.get("pid")))
        expected_role = "coordinator" if row.get("participant_id") == "coordinator" else "actor"
        if row.get("role") != expected_role:
            raise JoinedQualificationError("participant role differs")
    if len(handles) != 5 or len(pids) != 5:
        raise JoinedQualificationError("participant handles/PIDs are not unique")
    declarations = membership.get("declarations")
    if not isinstance(declarations, dict) or set(declarations) != expected_participants:
        raise JoinedQualificationError("declaration participant set differs")
    for participant, declaration in declarations.items():
        expected_publish = ["RLAction", "RLControl", "RLPolicyAnnouncement"] if participant == "coordinator" else ["RLReceipt", "RLTransition"]
        expected_subscribe = ["RLReceipt", "RLTransition"] if participant == "coordinator" else ["RLAction", "RLControl", "RLPolicyAnnouncement"]
        if not isinstance(declaration, dict) or declaration.get("publish") != expected_publish or declaration.get("subscribe") != expected_subscribe:
            raise JoinedQualificationError("declaration matrix differs")
    time_enabled = membership.get("time_enabled")
    if not isinstance(time_enabled, dict) or set(time_enabled) != expected_participants or any(
        value != {"regulating": True, "constrained": True} for value in time_enabled.values()
    ):
        raise JoinedQualificationError("time enablement differs")
    sync = membership.get("synchronization")
    expected_phases = {
        "ready", "training-0", "training-1", "evaluation-0", "evaluation-2", "terminal"
    }
    _exact_set("registered synchronization labels", membership.get("registered_labels"), expected_phases)
    if not isinstance(sync, dict) or set(sync) != expected_phases:
        raise JoinedQualificationError("synchronization phases differ")
    for phase in expected_phases:
        value = sync[phase]
        if not isinstance(value, dict):
            raise JoinedQualificationError("synchronization row differs")
        for key in ("announced", "achieved", "synchronized"):
            _exact_set(f"{phase} {key}", value.get(key), expected_participants)
    activations = membership.get("policy_activations")
    activation_phases = {"training-0", "training-1", "evaluation-0", "evaluation-2"}
    if not isinstance(activations, dict) or set(activations) != activation_phases:
        raise JoinedQualificationError("policy activation phases differ")
    for phase, value in activations.items():
        if not isinstance(value, dict):
            raise JoinedQualificationError("policy activation row differs")
        _exact_set(f"{phase} ACK", value.get("acknowledged"), workers)
        _digest(f"{phase} artifact", value.get("artifact_sha256"))
    if membership.get("federation_joined") is not True:
        raise JoinedQualificationError("federation membership is incomplete")

    time = _object(payloads["time-grants"], "joined-time-grants-v1")
    mappings = time.get("model_delivery_pairs")
    if not isinstance(mappings, list) or len(mappings) != EXACT_TRANSITIONS:
        raise JoinedQualificationError("time/grant evidence differs")
    stream_positions: dict[tuple[str, str], tuple[int, float, float]] = {}
    episode_mappings: dict[tuple[str, str, str], tuple[int, float, float]] = {}
    phase_counts = {"train-1": 0, "train-2": 0}
    for row in mappings:
        if not isinstance(row, dict) or set(row) != {
            "phase",
            "worker_id",
            "episode_id",
            "episode_sequence",
            "inner_time",
            "terminal_time",
            "mapped_delivery_time",
            "request_time",
            "grant_time",
        }:
            raise JoinedQualificationError("model/delivery time row fields differ")
        phase = row["phase"]
        worker = row["worker_id"]
        episode = row["episode_id"]
        sequence = row["episode_sequence"]
        values = (
            row["inner_time"],
            row["terminal_time"],
            row["mapped_delivery_time"],
            row["request_time"],
            row["grant_time"],
        )
        if phase not in phase_counts or worker not in workers or not isinstance(episode, str) or type(sequence) is not int or sequence < 1 or any(type(item) not in {int, float} for item in values):
            raise JoinedQualificationError("model/delivery time mapping differs")
        inner, terminal, mapped, request, grant = (float(item) for item in values)
        base = 0.0 if phase == "train-1" else 32768.0
        phase_limit = base + 32768.0
        expected_mapped = base + sequence * 32.0 + terminal
        if not (1.0 <= inner <= terminal <= 30.0) or mapped != expected_mapped or not (mapped < phase_limit) or grant != request or grant < mapped:
            raise JoinedQualificationError("frozen time mapping formula differs")
        stream_key = (phase, worker)
        previous = stream_positions.get(stream_key)
        if previous is not None and ((sequence, inner) <= (previous[0], previous[1]) or grant < previous[2]):
            raise JoinedQualificationError("time stream regressed")
        stream_positions[stream_key] = (sequence, inner, grant)
        episode_key = (phase, worker, episode)
        episode_mapping = (sequence, terminal, mapped)
        prior_episode_mapping = episode_mappings.setdefault(episode_key, episode_mapping)
        if prior_episode_mapping != episode_mapping:
            raise JoinedQualificationError("episode mapping is inconsistent")
        phase_counts[phase] += 1
    if phase_counts != {"train-1": 2048, "train-2": 2048}:
        raise JoinedQualificationError("time phase bounds differ")

    delivery = _object(payloads["delivery"], "joined-delivery-v1")
    for name in ("raw", "semantic", "unique"):
        if delivery.get(name) != EXACT_TRANSITIONS:
            raise JoinedQualificationError("delivery count differs")
    if any(delivery.get(name) != 0 for name in ("duplicate", "reject", "conflict")):
        raise JoinedQualificationError("delivery failure counters are nonzero")
    shards = delivery.get("shards")
    if not isinstance(shards, list) or len(shards) != 8:
        raise JoinedQualificationError("delivery shards differ")
    expected_shards = {(batch, worker): 512 for batch in (1, 2) for worker in workers}
    observed_shards: dict[tuple[int, str], int] = {}
    for row in shards:
        if not isinstance(row, dict) or set(row) != {"batch", "actor", "raw", "semantic", "unique"}:
            raise JoinedQualificationError("delivery shard row differs")
        shard_key = (row["batch"], row["actor"])
        if shard_key in observed_shards or row["raw"] != row["semantic"] or row["raw"] != row["unique"]:
            raise JoinedQualificationError("delivery shard counts differ")
        observed_shards[shard_key] = row["raw"]
    if observed_shards != expected_shards:
        raise JoinedQualificationError("per-batch/per-actor delivery differs")

    parity = _object(payloads["local-parity"], "joined-local-parity-v1")
    pairs = parity.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXACT_TRANSITIONS:
        raise JoinedQualificationError("local semantic parity pairs differ")
    identities: set[str] = set()
    for row in pairs:
        if not isinstance(row, dict) or set(row) != {"identity", "local_sha256", "joined_sha256"}:
            raise JoinedQualificationError("parity pair fields differ")
        identity = row["identity"]
        if not isinstance(identity, str) or identity in identities:
            raise JoinedQualificationError("parity identity is missing or duplicated")
        identities.add(identity)
        local = _digest("local parity digest", row["local_sha256"])
        if local != _digest("joined parity digest", row["joined_sha256"]):
            raise JoinedQualificationError("local/joined digest differs")
    if any(parity.get(name) != 0 for name in ("mismatch", "missing", "extra")):
        raise JoinedQualificationError("parity failure counters are nonzero")

    policy = _object(payloads["policy-recovery"], "joined-policy-recovery-v1")
    if (
        policy.get("versions") != [0, 1, 2]
        or policy.get("batch_sizes") != [2048, 2048]
        or policy.get("uninterrupted_adam_steps") != 160
        or policy.get("fresh_adam_steps") != 80
        or policy.get("reload_adam_steps") != 80
        or policy.get("qualification_adam_steps") != 320
    ):
        raise JoinedQualificationError("policy/checkpoint recovery evidence differs")
    artifact_pairs = policy.get("artifact_pairs")
    if not isinstance(artifact_pairs, list) or len(artifact_pairs) != 5:
        raise JoinedQualificationError("policy artifact pairs differ")
    expected_policy_roles = {"policy-0", "policy-1", "policy-2", "checkpoint-1", "checkpoint-2"}
    observed_policy_roles: set[str] = set()
    for row in artifact_pairs:
        if not isinstance(row, dict) or set(row) != {"role", "local_sha256", "joined_sha256"}:
            raise JoinedQualificationError("policy artifact pair fields differ")
        if _digest("local policy artifact", row["local_sha256"]) != _digest(
            "joined policy artifact", row["joined_sha256"]
        ):
            raise JoinedQualificationError("local/joined policy bytes differ")
        role = row["role"]
        if role not in expected_policy_roles or role in observed_policy_roles:
            raise JoinedQualificationError("policy artifact role differs")
        observed_policy_roles.add(role)
    branches = policy.get("branches")
    if not isinstance(branches, list) or {row.get("branch") for row in branches if isinstance(row, dict)} != {"uninterrupted", "fresh", "reload"}:
        raise JoinedQualificationError("policy recovery branches differ")
    expected_branch_steps = {"uninterrupted": 160, "fresh": 80, "reload": 80}
    for row in branches:
        if not isinstance(row, dict) or set(row) != {"branch", "batch_sha256", "cursor", "adam_steps", "payload_sha256"}:
            raise JoinedQualificationError("policy recovery branch row differs")
        if row["adam_steps"] != expected_branch_steps.get(row["branch"]):
            raise JoinedQualificationError("policy recovery branch Adam count differs")
        _digest("policy branch batch", row["batch_sha256"])
        _digest("policy branch payload", row["payload_sha256"])
        if not isinstance(row["cursor"], dict):
            raise JoinedQualificationError("policy recovery branch cursor differs")
    branch_by_name = {row["branch"]: row for row in branches}
    if branch_by_name["uninterrupted"]["payload_sha256"] != branch_by_name["reload"]["payload_sha256"]:
        raise JoinedQualificationError("reloaded policy payload differs")
    cursor = policy.get("recovery_cursor")
    if not isinstance(cursor, dict) or cursor.get("policy_version") != 1 or cursor.get("batch_index") != 1 or cursor.get("restored") is not True:
        raise JoinedQualificationError("recovery cursor differs")

    evaluation = _object(payloads["evaluation-isolation"], "joined-evaluation-v1")
    evaluation_pairs = evaluation.get("pairs")
    if (
        not isinstance(evaluation_pairs, list)
        or len(evaluation_pairs) != 16
        or evaluation.get("training_writes") != 0
        or evaluation.get("state_before_sha256") != evaluation.get("state_after_sha256")
    ):
        raise JoinedQualificationError("evaluation isolation differs")
    eval_ids: set[str] = set()
    total_eval_steps = 0
    observed_eval_keys: set[tuple[int, int]] = set()
    for row in evaluation_pairs:
        if not isinstance(row, dict) or set(row) != {"identity", "policy_version", "ordinal", "steps", "terminated", "truncated", "action_trace_sha256", "final_observation_sha256", "return_sha256", "local_sha256", "joined_sha256"}:
            raise JoinedQualificationError("evaluation pair fields differ")
        identity = row["identity"]
        if not isinstance(identity, str) or identity in eval_ids:
            raise JoinedQualificationError("evaluation identity differs")
        eval_ids.add(identity)
        if _digest("local evaluation", row["local_sha256"]) != _digest(
            "joined evaluation", row["joined_sha256"]
        ):
            raise JoinedQualificationError("evaluation digest differs")
        policy_version = row["policy_version"]
        ordinal = row["ordinal"]
        steps = row["steps"]
        if policy_version not in (0, 2) or type(ordinal) is not int or type(steps) is not int or not (1 <= steps <= 30):
            raise JoinedQualificationError("evaluation episode provenance differs")
        if row["terminated"] is not True or row["truncated"] is not False:
            raise JoinedQualificationError("evaluation terminal state differs")
        evaluation_key = (policy_version, ordinal)
        if evaluation_key in observed_eval_keys:
            raise JoinedQualificationError("evaluation policy/ordinal is duplicated")
        observed_eval_keys.add(evaluation_key)
        total_eval_steps += steps
        for name in ("action_trace_sha256", "final_observation_sha256", "return_sha256"):
            _digest(name, row[name])
    if len({ordinal for _, ordinal in observed_eval_keys}) != 8 or total_eval_steps > 480:
        raise JoinedQualificationError("evaluation assignment or budget differs")

    cleanup = _object(payloads["cleanup"], "joined-cleanup-v1")
    cleanup_rows = cleanup.get("rows")
    if not isinstance(cleanup_rows, list) or len(cleanup_rows) != 6:
        raise JoinedQualificationError("joined cleanup rows differ")
    expected_cleanup = expected_participants | {"rtid"}
    if {row.get("participant_id") for row in cleanup_rows if isinstance(row, dict)} != expected_cleanup:
        raise JoinedQualificationError("joined cleanup participants differ")
    for row in cleanup_rows:
        if not isinstance(row, dict) or row.get("transport_closed") is not True or row.get("process_clean") is not True:
            raise JoinedQualificationError("joined cleanup row differs")
        if row.get("participant_id") == "rtid":
            if row.get("supervised_termination") is not True:
                raise JoinedQualificationError("RTI termination is not supervised")
        elif row.get("resigned") is not True:
            raise JoinedQualificationError("federate did not resign")
    negative = _object(payloads["negative-probe"], "joined-negative-probe-v1")
    probes = negative.get("probes")
    if not isinstance(probes, list) or {row.get("probe_id") for row in probes if isinstance(row, dict)} != NEGATIVE_PROBES or len(probes) != len(NEGATIVE_PROBES) or any(not isinstance(row, dict) or row.get("rejected") is not True or row.get("state_unchanged") is not True for row in probes):
        raise JoinedQualificationError("negative probe differs")
    prerequisite = _object(payloads["prerequisite-ledger"], "joined-prerequisite-v1")
    expected_prerequisites = {
        "task104_commit": "c6b3ad1be7ce2af4a72d6efd9f793ee455f5ff0c",
        "task104_ledger_sha256": "ceda4d3b592cb5c7abbb95cd2ee0a9336766ccf91b2c5da8f935b8eedf3c7ad7",
        "task104_ledger_file_sha256": "ac0773a681c31d58a3d964832a29bc9351ad069a4f619a9e714dcbd8559ededb",
        "task105_implementation_commit": "eef274dde3ac833eee94f0726a0781f402229e4d",
        "task105_acceptance_commit": "654a7d8b9c11426da495b5a5a9df795846aba76c",
        "task105_raw_ledger_sha256": "4c8830b553d692e42bfcba784dc39a391071b733de81b63cf845f905cdb04b3a",
        "task105_bundle_sha256": "c8a2a42929340e7defd7b749d8b4697efc98b2efc2923de8ab155309d35c8792",
        "task105_ledger_file_sha256": "3a108498efcb3740f5357e3d52db7bdb8dc208e21370c38aef4e12d84903a50f",
    }
    if any(prerequisite.get(key) != value for key, value in expected_prerequisites.items()) or prerequisite.get("all_rehashed") is not True:
        raise JoinedQualificationError("local prerequisite ledger differs")
    prerequisite_digests = prerequisite.get("artifact_sha256")
    if not isinstance(prerequisite_digests, list) or not prerequisite_digests:
        raise JoinedQualificationError("prerequisite artifact ledger is empty")
    if len(prerequisite_digests) < 2:
        raise JoinedQualificationError("TASK104/TASK105 prerequisite roots are missing")
    observed_tasks: set[str] = set()
    for row in prerequisite_digests:
        if not isinstance(row, dict) or set(row) != {"task", "path", "size_bytes", "sha256", "rehash_sha256"} or row["task"] not in {"TASK-RL-104", "TASK-RL-105"} or not isinstance(row["path"], str) or _positive("prerequisite size", row["size_bytes"]) <= 0 or _digest("prerequisite artifact", row["sha256"]) != _digest("prerequisite rehash", row["rehash_sha256"]):
            raise JoinedQualificationError("prerequisite artifact row differs")
        task = row["task"]
        if task in observed_tasks:
            raise JoinedQualificationError("prerequisite task is duplicated")
        observed_tasks.add(task)
        body = _read_bound_file(
            verification, row["path"], size=row["size_bytes"], digest=row["sha256"]
        )
        expected_file_digest = (
            verification.prerequisite_file_sha256 or {
                "TASK-RL-104": expected_prerequisites["task104_ledger_file_sha256"],
                "TASK-RL-105": expected_prerequisites["task105_ledger_file_sha256"],
            }
        )[task]
        if _sha(body) != expected_file_digest:
            raise JoinedQualificationError("prerequisite ledger fixed digest differs")
        ledger_path = (
            verification.repository_root / Path(*PurePosixPath(row["path"]).parts)
        ).resolve(strict=True)
        _verify_indexed_ledger(body, ledger_path.parent)
    if observed_tasks != {"TASK-RL-104", "TASK-RL-105"}:
        raise JoinedQualificationError("prerequisite task set differs")
    return _sha(_canonical({role: _sha(payloads[role]) for role in sorted(RAW_ROLES)}))


def _write(root: Path, role: str, payload: bytes) -> JoinedArtifactEntry:
    relative = f"artifacts/{role}.json"
    target = root / Path(*PurePosixPath(relative).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return JoinedArtifactEntry(role, relative, len(payload), _sha(payload))


def verify_joined_reference_ppo_bundle(
    root: Path,
    bundle: JoinedReferencePPOBundle,
    *,
    context: JoinedVerificationContext | None = None,
) -> None:
    resolved = root.resolve(strict=True)
    payloads: dict[str, bytes] = {}
    for entry in bundle.entries:
        target = (resolved / Path(*PurePosixPath(entry.path).parts)).resolve(strict=True)
        try:
            target.relative_to(resolved)
        except ValueError as exc:
            raise JoinedQualificationError("artifact escapes root") from exc
        payload = target.read_bytes()
        if target.is_symlink() or len(payload) != entry.size_bytes or _sha(payload) != entry.sha256:
            raise JoinedQualificationError("artifact failed re-hash")
        payloads[entry.role] = payload
    ledger = resolved / "ledger.json"
    if ledger.read_bytes() != _canonical(bundle.content()):
        raise JoinedQualificationError("durable ledger differs")
    if bundle.capability is not None:
        plan = JoinedReferencePPOPlan()
        evidence_sha256 = _verify_typed_evidence(payloads, plan, context)
        raw_sha256 = _sha(_canonical([entry.content() for entry in bundle.entries]))
        if (
            bundle.capability.plan_sha256 != plan.sha256
            or bundle.capability.evidence_sha256 != evidence_sha256
            or bundle.capability.raw_ledger_sha256 != raw_sha256
        ):
            raise JoinedQualificationError("capability differs from verified facts")


def _actual_gorti_runner(_plan: JoinedReferencePPOPlan) -> JoinedReferencePPOExecution:
    if not _plan.exact:
        raise JoinedQualificationError("actual joined-gorti runner requires the exact plan")
    repository = Path(__file__).resolve().parents[3]
    benchmark = repository / (
        "engineering/specifications/pyjevsim-rl/benchmarks/gorti_same_host/"
        "joined_reference_ppo_joined.py"
    )
    rtid = repository / "bin/rtid.exe"
    python = repository / "pysdk/.venv/Scripts/python.exe"
    oracle = repository / (
        "engineering/specifications/pyjevsim-rl/testcases/reference_ppo_qualification"
    )
    for required in (benchmark, rtid, python, oracle / "ledger.json"):
        if not required.is_file():
            raise JoinedQualificationError(f"actual joined prerequisite is missing: {required}")
    module_name = "_task_rl_106_joined_launcher"
    spec = importlib.util.spec_from_file_location(module_name, benchmark)
    if spec is None or spec.loader is None:
        raise JoinedQualificationError("joined launcher cannot be source-loaded")
    benchmark_dir = str(benchmark.parent)
    inserted = benchmark_dir not in sys.path
    if inserted:
        sys.path.insert(0, benchmark_dir)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if inserted and sys.path and sys.path[0] == benchmark_dir:
            sys.path.pop(0)
    output_parent = Path(
        tempfile.mkdtemp(prefix="task-rl-106-", dir=repository / "engineering")
    )
    summary = module.run_joined(
        oracle_root=oracle,
        output_root=output_parent,
        run_name="actual",
        rtid_path=rtid,
        python_executable=python,
    )
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        raise JoinedQualificationError("joined launcher did not complete; blocked evidence cannot be relabelled")
    if summary.get("blockers") not in (None, []):
        raise JoinedQualificationError("joined launcher retained admission blockers")
    run_root = output_parent / "actual"
    projection_path = run_root / "qualification-projection.json"
    try:
        projection = json.loads(projection_path.read_bytes())
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JoinedQualificationError("complete joined evidence projection is missing") from exc
    if not isinstance(projection, dict) or set(projection) != {
        "schema_version", "artifacts"
    } or projection["schema_version"] != "joined-reference-ppo-evidence-projection-v1":
        raise JoinedQualificationError("joined evidence projection schema differs")
    rows = projection["artifacts"]
    if not isinstance(rows, list) or len(rows) != len(RAW_ROLES):
        raise JoinedQualificationError("joined evidence projection roles differ")
    artifacts: dict[str, bytes] = {}
    resolved_root = run_root.resolve(strict=True)
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "role", "path", "size_bytes", "sha256"
        }:
            raise JoinedQualificationError("joined evidence projection row differs")
        role = row["role"]
        if role not in RAW_ROLES or role in artifacts:
            raise JoinedQualificationError("joined evidence projection role differs")
        relative = row["path"]
        if not isinstance(relative, str):
            raise JoinedQualificationError("joined evidence projection path differs")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise JoinedQualificationError("joined evidence projection path escapes")
        target = (resolved_root / Path(*pure.parts)).resolve(strict=True)
        try:
            target.relative_to(resolved_root)
        except ValueError as exc:
            raise JoinedQualificationError("joined evidence projection artifact escapes") from exc
        body = target.read_bytes()
        if target.is_symlink() or len(body) != _positive(
            "projected artifact size", row["size_bytes"]
        ) or _sha(body) != _digest("projected artifact digest", row["sha256"]):
            raise JoinedQualificationError("joined evidence projection artifact failed re-hash")
        artifacts[role] = body
    return JoinedReferencePPOExecution("internal-actual-gorti-v1", artifacts)


def run_joined_reference_ppo_qualification(
    root: Path,
    *,
    plan: JoinedReferencePPOPlan | None = None,
    runner: JoinedReferencePPORunner | None = None,
    context: JoinedVerificationContext | None = None,
) -> JoinedReferencePPOBundle:
    selected = JoinedReferencePPOPlan() if plan is None else plan
    if root.exists():
        raise JoinedQualificationError("qualification root must not exist")
    root.mkdir(parents=True, exist_ok=False)
    injected = runner is not None
    execution = _actual_gorti_runner(selected) if runner is None else runner(selected)
    blockers: list[str] = []
    if injected or execution.runner_kind != "internal-actual-gorti-v1":
        blockers.append("nonactual-or-injected-runner")
    if not selected.exact:
        blockers.append("bounded-or-nonexact-plan")
    entries = tuple(
        _write(root, role, execution.artifacts[role]) for role in sorted(RAW_ROLES)
    )
    capability: JoinedReferencePPOCapability | None = None
    if not blockers:
        evidence_sha256 = _verify_typed_evidence(execution.artifacts, selected, context)
        raw_sha256 = _sha(_canonical([entry.content() for entry in entries]))
        capability = JoinedReferencePPOCapability(selected.sha256, raw_sha256, evidence_sha256)
    bundle = JoinedReferencePPOBundle(selected.sha256, entries, tuple(blockers), capability)
    ledger_path = root / "ledger.json"
    descriptor = os.open(ledger_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical(bundle.content()))
        stream.flush()
        os.fsync(stream.fileno())
    verify_joined_reference_ppo_bundle(root, bundle, context=context)
    return bundle


__all__ = [
    "CAPABILITY_SCOPE",
    "JoinedArtifactEntry",
    "JoinedQualificationError",
    "JoinedReferencePPOBundle",
    "JoinedReferencePPOCapability",
    "JoinedReferencePPOExecution",
    "JoinedReferencePPOPlan",
    "JoinedReferencePPORunner",
    "JoinedVerificationContext",
    "run_joined_reference_ppo_qualification",
    "verify_joined_reference_ppo_bundle",
]
