"""Deterministic gorti federation channel for pyjevsim RL rollouts.

The module deliberately depends only on the duck-typed ``Federate`` surface.
It can therefore be tested without an RTI and used with the in-process or gRPC
SDK transports without changing the rollout record types.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import hashlib
import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from rti1516e.events import (
    FederationHalted,
    FederationSynchronized,
    ReceiveInteraction,
    SynchronizationPointAnnounced,
    SynchronizationPointRegistrationFailed,
    TimeAdvanceGrant,
)

SCHEMA_VERSION = 1
MAX_PAYLOAD_BYTES = 1024 * 1024
ACTION_CLASS = "RLAction"
TRANSITION_CLASS = "RLTransition"
RECEIPT_CLASS = "RLReceipt"
CONTROL_CLASS = "RLControl"
POLICY_CLASS = "RLPolicyAnnouncement"
PAYLOAD_FIELD = "payload"
TRANSITION_BATCH_FIELD = "batch_records"

_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "generation",
    "worker_id",
    "episode_id",
    "step_id",
    "policy_version",
    "idempotency_key",
    "logical_time",
    "payload",
)


class FederationProtocolError(RuntimeError):
    """Base class for fail-closed rollout-channel failures."""


class EnvelopeValidationError(FederationProtocolError, ValueError):
    """An envelope is malformed, unsupported, stale, or non-canonical."""


class FederationHaltedError(FederationProtocolError):
    """The RTI reported that the federation can no longer make progress."""


class EventStreamExhaustedError(FederationProtocolError):
    """The callback stream ended before the required completion event."""


class IdempotencyConflictError(FederationProtocolError):
    """An idempotency key was reused for different content."""


class Role(enum.StrEnum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    ACTOR = "actor"
    LEARNER = "learner"
    EVALUATOR = "evaluator"


_INTERACTION_CLASSES = (
    ACTION_CLASS,
    TRANSITION_CLASS,
    RECEIPT_CLASS,
    CONTROL_CLASS,
    POLICY_CLASS,
)

_INTERACTION_SENDERS: dict[str, frozenset[Role]] = {
    ACTION_CLASS: frozenset({Role.COORDINATOR, Role.LEARNER, Role.EVALUATOR}),
    TRANSITION_CLASS: frozenset({Role.WORKER, Role.ACTOR}),
    RECEIPT_CLASS: frozenset({Role.WORKER, Role.ACTOR}),
    CONTROL_CLASS: frozenset({Role.COORDINATOR, Role.EVALUATOR}),
    POLICY_CLASS: frozenset({Role.COORDINATOR, Role.LEARNER}),
}

_ROLE_SUBSCRIPTIONS: dict[Role, frozenset[str]] = {
    Role.COORDINATOR: frozenset({TRANSITION_CLASS, RECEIPT_CLASS}),
    Role.WORKER: frozenset({ACTION_CLASS, CONTROL_CLASS, POLICY_CLASS}),
    Role.ACTOR: frozenset({ACTION_CLASS, CONTROL_CLASS, POLICY_CLASS}),
    Role.LEARNER: frozenset({TRANSITION_CLASS}),
    Role.EVALUATOR: frozenset({TRANSITION_CLASS, POLICY_CLASS}),
}


@dataclass(frozen=True)
class ReceivedEnvelope:
    interaction_class: str
    envelope: dict[str, Any]
    timestamp: float


@dataclass(frozen=True)
class GrantedBatch:
    granted_time: float
    interactions: tuple[ReceivedEnvelope, ...]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {key: val for key, val in attributes.items() if not key.startswith("_")}
    raise EnvelopeValidationError("envelope must be a mapping, dataclass, or to_dict object")


def canonical_json(value: Any) -> bytes:
    """Return canonical UTF-8 JSON (stable keys, separators, and no NaN)."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EnvelopeValidationError(f"value is not canonical JSON compatible: {exc}") from exc


def validate_envelope(value: Any, *, generation: int | None = None) -> dict[str, Any]:
    envelope = _plain(value)
    missing = [field for field in _REQUIRED_FIELDS if field not in envelope]
    if missing:
        raise EnvelopeValidationError(f"missing envelope fields: {', '.join(missing)}")

    version = envelope["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise EnvelopeValidationError(f"unsupported schema_version: {version!r}")
    for field in ("generation", "step_id", "policy_version"):
        item = envelope[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise EnvelopeValidationError(f"{field} must be a non-negative integer")
    for field in ("run_id", "worker_id", "episode_id", "idempotency_key"):
        item = envelope[field]
        if not isinstance(item, str) or not item:
            raise EnvelopeValidationError(f"{field} must be a non-empty string")
    logical_time = envelope["logical_time"]
    if isinstance(logical_time, bool) or not isinstance(logical_time, (int, float)):
        raise EnvelopeValidationError("logical_time must be numeric")
    envelope["logical_time"] = float(logical_time)
    if not math.isfinite(envelope["logical_time"]):
        raise EnvelopeValidationError("logical_time must be finite")
    if generation is not None and envelope["generation"] != generation:
        raise EnvelopeValidationError(
            f"stale federation generation {envelope['generation']}; expected {generation}"
        )
    canonical_json(envelope)
    return cast(dict[str, Any], envelope)


def encode_envelope(value: Any, *, generation: int | None = None) -> bytes:
    encoded = canonical_json(validate_envelope(value, generation=generation))
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise EnvelopeValidationError(f"encoded envelope exceeds {MAX_PAYLOAD_BYTES} bytes")
    return encoded


def decode_envelope(
    value: bytes | bytearray | memoryview | str,
    *,
    generation: int | None = None,
) -> dict[str, Any]:
    try:
        raw = bytes(value).decode("utf-8") if not isinstance(value, str) else value
    except (UnicodeDecodeError, TypeError) as exc:
        raise EnvelopeValidationError(f"invalid UTF-8 JSON envelope: {exc}") from exc
    if len(raw.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise EnvelopeValidationError(f"encoded envelope exceeds {MAX_PAYLOAD_BYTES} bytes")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EnvelopeValidationError(f"invalid UTF-8 JSON envelope: {exc}") from exc
    return validate_envelope(decoded, generation=generation)


def _transition_simulation_time(value: Any) -> float:
    """Validate the model-committed time carried inside a transition payload."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnvelopeValidationError("RLTransition payload simulation_time must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise EnvelopeValidationError(
            "RLTransition payload simulation_time must be finite and non-negative"
        )
    return result


class GortiRolloutChannel:
    """Role-aware TSO channel over a joined, duck-typed gorti Federate."""

    def __init__(
        self,
        federate: Any,
        role: Role | str,
        *,
        generation: int,
        dedup_window: int = 100_000,
    ) -> None:
        self.federate = federate
        try:
            self.role = role if isinstance(role, Role) else Role(str(role).lower())
        except ValueError as exc:
            raise ValueError(f"unsupported rollout role: {role!r}") from exc
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if isinstance(dedup_window, bool) or not isinstance(dedup_window, int) or dedup_window <= 0:
            raise ValueError("dedup_window must be a positive integer")
        self.generation = generation
        self.dedup_window = dedup_window
        self.logical_time = 0.0
        self.lookahead: float | None = None
        self._pending: deque[ReceivedEnvelope] = deque()
        self._deferred: deque[Any] = deque()
        self._seen: dict[str, bytes] = {}
        self._seen_order: deque[str] = deque()
        self._failed = False
        self._callback_lock = asyncio.Lock()

    def _ensure_healthy(self) -> None:
        if self._failed:
            raise FederationProtocolError("rollout channel is failed; create a new joined channel")

    async def declare(self, role: Role | str | None = None) -> None:
        self._ensure_healthy()
        selected = self.role
        if role is not None:
            selected = role if isinstance(role, Role) else Role(str(role).lower())
            self.role = selected
        publications = tuple(
            class_name
            for class_name in _INTERACTION_CLASSES
            if selected in _INTERACTION_SENDERS[class_name]
        )
        subscriptions = tuple(
            class_name
            for class_name in _INTERACTION_CLASSES
            if class_name in _ROLE_SUBSCRIPTIONS[selected]
        )
        for class_name in publications:
            await self.federate.publish_interaction_class(class_name)
        for class_name in subscriptions:
            await self.federate.subscribe_interaction_class(class_name)

    async def enable_time(self, *, lookahead: float) -> None:
        self._ensure_healthy()
        if isinstance(lookahead, bool) or not isinstance(lookahead, (int, float)):
            raise ValueError("lookahead must be a finite positive number")
        lookahead = float(lookahead)
        if not math.isfinite(lookahead) or lookahead <= 0:
            raise ValueError("lookahead must be a finite positive number")
        await self.federate.enable_time_regulation(lookahead)
        await self.federate.enable_time_constrained()
        self.lookahead = lookahead

    async def send_action(self, command: Any) -> None:
        self._ensure_sender(ACTION_CLASS)
        await self._send(ACTION_CLASS, command)

    async def send_transition(self, record: Any) -> None:
        self._ensure_sender(TRANSITION_CLASS)
        await self._send(TRANSITION_CLASS, record)

    async def send_transition_batch(
        self,
        records: Sequence[Any],
        *,
        delivery_time: float | None = None,
    ) -> None:
        """Send one atomic wire interaction containing multiple transitions.

        Records retain their model-committed logical times while the outer
        interaction uses one TSO delivery timestamp.  The receiver expands the
        batch before deduplication so existing consumers still observe ordinary
        transition envelopes.
        """

        self._ensure_sender(TRANSITION_CLASS)
        self._ensure_healthy()
        if self.lookahead is None:
            raise FederationProtocolError("enable_time() must complete before a TSO send")
        if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
            raise EnvelopeValidationError("transition batch must be a non-string sequence")
        if not records:
            raise EnvelopeValidationError("transition batch must not be empty")
        prepared: list[dict[str, Any]] = []
        identity: tuple[object, ...] | None = None
        for value in records:
            envelope = validate_envelope(value, generation=self.generation)
            current_identity = tuple(
                envelope[field]
                for field in ("run_id", "generation", "worker_id", "episode_id", "policy_version")
            )
            if identity is None:
                identity = current_identity
            elif current_identity != identity:
                raise EnvelopeValidationError(
                    "transition batch records must share run, generation, worker, episode, "
                    "and policy identity"
                )
            payload = envelope["payload"]
            if not isinstance(payload, Mapping):
                raise EnvelopeValidationError("RLTransition payload must be a mapping")
            normalized_payload = dict(payload)
            if TRANSITION_BATCH_FIELD in normalized_payload:
                raise EnvelopeValidationError("nested transition batches are not supported")
            supplied = normalized_payload.get("simulation_time", envelope["logical_time"])
            simulation_time = _transition_simulation_time(supplied)
            if simulation_time != envelope["logical_time"]:
                raise EnvelopeValidationError(
                    "RLTransition payload simulation_time conflicts with record logical_time"
                )
            normalized_payload["simulation_time"] = simulation_time
            envelope["payload"] = normalized_payload
            prepared.append(envelope)
        step_ids = [int(item["step_id"]) for item in prepared]
        logical_times = [float(item["logical_time"]) for item in prepared]
        if step_ids != sorted(step_ids) or len(set(step_ids)) != len(step_ids):
            raise EnvelopeValidationError(
                "transition batch step IDs must be strictly increasing and unique"
            )
        if logical_times != sorted(logical_times):
            raise EnvelopeValidationError("transition batch logical times must be monotonic")
        minimum_delivery = max(max(logical_times), self.logical_time + self.lookahead)
        if delivery_time is None:
            selected_delivery_time = minimum_delivery
        else:
            if isinstance(delivery_time, bool) or not isinstance(delivery_time, (int, float)):
                raise ValueError("delivery_time must be a finite numeric value")
            selected_delivery_time = float(delivery_time)
            if not math.isfinite(selected_delivery_time) or selected_delivery_time < 0:
                raise ValueError("delivery_time must be finite and non-negative")
            if selected_delivery_time < minimum_delivery:
                raise FederationProtocolError(
                    "explicit transition-batch delivery_time violates model time or lookahead"
                )
        digest = hashlib.sha256(canonical_json(prepared)).hexdigest()
        first = prepared[0]
        outer = {
            "schema_version": SCHEMA_VERSION,
            "run_id": first["run_id"],
            "generation": self.generation,
            "worker_id": first["worker_id"],
            "episode_id": first["episode_id"],
            "step_id": step_ids[-1],
            "policy_version": first["policy_version"],
            "idempotency_key": f"transition-batch:{digest}",
            "logical_time": selected_delivery_time,
            "payload": {
                TRANSITION_BATCH_FIELD: prepared,
                "batch_count": len(prepared),
                "simulation_time": selected_delivery_time,
            },
        }
        await self._send(TRANSITION_CLASS, outer)

    async def send_planned_transition_batch(self, batch: Any) -> None:
        """Send one batch emitted by :class:`AdaptiveTransitionBatcher`."""

        from pyjevsim_bridge.rl.batching import PlannedTransitionBatch

        if not isinstance(batch, PlannedTransitionBatch):
            raise TypeError("batch must be a PlannedTransitionBatch")
        if len(batch.records) == 1:
            await self.send_transition(batch.records[0])
            return
        await self.send_transition_batch(batch.records)

    async def send_control(self, command: Any) -> None:
        self._ensure_sender(CONTROL_CLASS)
        await self._send(CONTROL_CLASS, command)

    async def send_policy_announcement(self, announcement: Any) -> None:
        self._ensure_sender(POLICY_CLASS)
        await self._send(POLICY_CLASS, announcement)

    async def send_receipt(self, receipt: Any) -> None:
        """Send a worker/actor receipt to the coordinator under TSO."""

        self._ensure_sender(RECEIPT_CLASS)
        await self._send(RECEIPT_CLASS, receipt)

    def _ensure_sender(self, class_name: str) -> None:
        if self.role in _INTERACTION_SENDERS[class_name]:
            return
        allowed = ", ".join(role.value for role in Role if role in _INTERACTION_SENDERS[class_name])
        raise FederationProtocolError(
            f"role {self.role.value} may not send {class_name}; allowed roles: {allowed}"
        )

    async def _send(self, class_name: str, value: Any) -> None:
        self._ensure_healthy()
        envelope = validate_envelope(value, generation=self.generation)
        if self.lookahead is None:
            raise FederationProtocolError("enable_time() must complete before a TSO send")
        minimum = self.logical_time + self.lookahead
        timestamp = envelope["logical_time"]
        if class_name == TRANSITION_CLASS:
            payload = envelope["payload"]
            if not isinstance(payload, Mapping):
                raise EnvelopeValidationError("RLTransition payload must be a mapping")
            envelope["payload"] = dict(payload)
            if "simulation_time" in envelope["payload"]:
                supplied_simulation_time = _transition_simulation_time(
                    envelope["payload"]["simulation_time"]
                )
                if supplied_simulation_time != timestamp:
                    raise EnvelopeValidationError(
                        "RLTransition payload simulation_time conflicts with record logical_time"
                    )
            envelope["payload"]["simulation_time"] = timestamp
            timestamp = max(timestamp, minimum)
            envelope["logical_time"] = timestamp
        elif timestamp < minimum:
            raise FederationProtocolError(
                f"TSO timestamp {timestamp} violates lookahead; minimum is {minimum}"
            )
        await self.federate.send_interaction(
            class_name,
            parameters={PAYLOAD_FIELD: encode_envelope(envelope, generation=self.generation)},
            timestamp=timestamp,
        )

    async def advance_to(self, logical_time: float) -> GrantedBatch:
        async with self._callback_lock:
            return await self._advance_to_locked(logical_time)

    async def _advance_to_locked(self, logical_time: float) -> GrantedBatch:
        self._ensure_healthy()
        if isinstance(logical_time, bool) or not isinstance(logical_time, (int, float)):
            raise ValueError("advance target must be numeric and non-boolean")
        target = float(logical_time)
        if not math.isfinite(target) or target < self.logical_time:
            raise ValueError("advance target must be finite and monotonic")
        previous_grant = self.logical_time
        # Once NMR reaches the federate, an exception (including cancellation)
        # cannot prove that the RTI rejected it. A retry could therefore create
        # two outstanding requests, so the channel must be replaced.
        try:
            await self.federate.next_message_request(target)
        except BaseException:
            self._failed = True
            raise
        deferred = deque(self._deferred)
        self._deferred.clear()
        preserved: list[Any] = []
        grant: float | None = None
        failure: BaseException | None = None
        try:
            stream = self.federate.events().__aiter__()
            while True:
                if deferred:
                    event = deferred.popleft()
                else:
                    try:
                        event = await anext(stream)
                    except StopAsyncIteration as exc:
                        raise EventStreamExhaustedError(
                            "event stream ended before TimeAdvanceGrant"
                        ) from exc
                if isinstance(event, ReceiveInteraction):
                    self._pending.extend(self._decode_interactions(event))
                    continue
                if isinstance(event, FederationHalted):
                    raise FederationHaltedError(
                        f"federation halted: {event.cause}; stalled={event.stalled_federate_handle}"
                    )
                if isinstance(event, TimeAdvanceGrant):
                    raw_grant = event.time
                    if isinstance(raw_grant, bool) or not isinstance(raw_grant, (int, float)):
                        raise FederationProtocolError(
                            f"invalid time grant {raw_grant!r}; "
                            f"current={previous_grant}, target={target}"
                        )
                    grant = float(raw_grant)
                    if not math.isfinite(grant) or grant < previous_grant or grant > target:
                        raise FederationProtocolError(
                            f"invalid time grant {grant}; current={previous_grant}, target={target}"
                        )
                    # The RTI has committed this grant even if later decoding or
                    # deduplication fails. Preserve it before post-processing.
                    self.logical_time = grant
                    break
                preserved.append(event)
        except BaseException as exc:
            failure = exc
        finally:
            self._deferred.extend(preserved)
            self._deferred.extend(deferred)
        if failure is not None:
            self._failed = True
            raise failure
        if grant is None:  # defensive invariant; every successful loop exits on a grant
            raise EventStreamExhaustedError("event stream ended before TimeAdvanceGrant")
        try:
            for item in self._pending:
                if not math.isfinite(item.timestamp) or item.timestamp < 0:
                    raise FederationProtocolError(
                        f"invalid delivery timestamp {item.timestamp}; "
                        "expected a finite, non-negative value"
                    )
                if not previous_grant <= item.timestamp <= grant:
                    raise FederationProtocolError(
                        f"delivery timestamp {item.timestamp} is outside grant window "
                        f"[{previous_grant}, {grant}]"
                    )
            received = self._deduplicate(list(self._pending))
        except BaseException:
            self._failed = True
            raise
        self._pending.clear()
        return GrantedBatch(grant, tuple(received))

    async def rendezvous(self, label: str, *, register: bool = False) -> None:
        async with self._callback_lock:
            await self._rendezvous_locked(label, register=register)

    async def _rendezvous_locked(self, label: str, *, register: bool = False) -> None:
        self._ensure_healthy()
        if not isinstance(label, str) or not label:
            raise ValueError("synchronization label must be non-empty")
        if register:
            await self.federate.sync.register_synchronization_point(label)
        deferred = deque(self._deferred)
        self._deferred.clear()
        preserved: list[Any] = []
        phase = "announcement"
        stream = self.federate.events().__aiter__()
        try:
            while True:
                if deferred:
                    event = deferred.popleft()
                else:
                    try:
                        event = await anext(stream)
                    except StopAsyncIteration:
                        if phase == "synchronized":
                            stream = self.federate.events().__aiter__()
                            try:
                                event = await anext(stream)
                            except StopAsyncIteration as exc:
                                raise EventStreamExhaustedError(
                                    f"event stream ended before federation synchronized: {label}"
                                ) from exc
                        else:
                            raise EventStreamExhaustedError(
                                f"event stream ended before sync announcement: {label}"
                            ) from None
                if isinstance(event, ReceiveInteraction):
                    self._pending.extend(self._decode_interactions(event))
                elif isinstance(event, FederationHalted):
                    raise FederationHaltedError(f"federation halted during {label}: {event.cause}")
                elif (
                    isinstance(event, SynchronizationPointRegistrationFailed)
                    and event.label == label
                ):
                    raise FederationProtocolError(f"synchronization registration failed: {label}")
                elif (
                    phase == "announcement"
                    and isinstance(event, SynchronizationPointAnnounced)
                    and event.label == label
                ):
                    await self.federate.sync.synchronization_point_achieved(label)
                    phase = "synchronized"
                elif (
                    phase == "synchronized"
                    and isinstance(event, FederationSynchronized)
                    and event.label == label
                ):
                    if event.failed_to_sync:
                        raise FederationProtocolError(
                            "synchronization completed with failed federates: "
                            f"{event.failed_to_sync}"
                        )
                    return
                else:
                    preserved.append(event)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._deferred.extend(preserved)
            self._deferred.extend(deferred)

    def _decode_interactions(self, event: ReceiveInteraction) -> tuple[ReceivedEnvelope, ...]:
        if event.class_name not in _INTERACTION_CLASSES:
            raise EnvelopeValidationError(
                f"unsupported rollout interaction class: {event.class_name}"
            )
        if event.class_name not in _ROLE_SUBSCRIPTIONS[self.role]:
            raise FederationProtocolError(
                f"role {self.role.value} may not receive {event.class_name}"
            )
        if PAYLOAD_FIELD not in event.parameters:
            raise EnvelopeValidationError(f"{event.class_name} is missing payload")
        envelope = decode_envelope(event.parameters[PAYLOAD_FIELD], generation=self.generation)
        if event.timestamp is None:
            raise EnvelopeValidationError(f"{event.class_name} must use timestamp order")
        if isinstance(event.timestamp, bool) or not isinstance(event.timestamp, (int, float)):
            raise EnvelopeValidationError(f"{event.class_name} delivery timestamp must be numeric")
        timestamp = float(event.timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise EnvelopeValidationError(
                f"{event.class_name} delivery timestamp must be finite and non-negative"
            )
        if envelope["logical_time"] != timestamp:
            raise EnvelopeValidationError(
                f"envelope logical_time {envelope['logical_time']} != TSO timestamp {timestamp}"
            )
        if event.class_name == TRANSITION_CLASS:
            payload = envelope["payload"]
            if not isinstance(payload, Mapping):
                raise EnvelopeValidationError("RLTransition payload must be a mapping")
            if TRANSITION_BATCH_FIELD in payload:
                return self._decode_transition_batch(envelope, timestamp)
            if "simulation_time" not in payload:
                raise EnvelopeValidationError("RLTransition payload is missing simulation_time")
            simulation_time = _transition_simulation_time(payload["simulation_time"])
            if simulation_time > timestamp:
                raise EnvelopeValidationError(
                    "RLTransition simulation_time must be no later than delivery"
                )
        return (ReceivedEnvelope(event.class_name, envelope, timestamp),)

    def _decode_transition_batch(
        self, outer: Mapping[str, Any], timestamp: float
    ) -> tuple[ReceivedEnvelope, ...]:
        payload = outer["payload"]
        if not isinstance(payload, Mapping) or set(payload) != {
            TRANSITION_BATCH_FIELD,
            "batch_count",
            "simulation_time",
        }:
            raise EnvelopeValidationError("transition batch payload fields differ")
        raw_records = payload[TRANSITION_BATCH_FIELD]
        count = payload["batch_count"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(raw_records, list)
            or len(raw_records) != count
        ):
            raise EnvelopeValidationError("transition batch count differs from records")
        if _transition_simulation_time(payload["simulation_time"]) != timestamp:
            raise EnvelopeValidationError("transition batch delivery time differs")
        identity = tuple(
            outer[field]
            for field in ("run_id", "generation", "worker_id", "episode_id", "policy_version")
        )
        decoded: list[ReceivedEnvelope] = []
        previous_step = -1
        previous_time = -math.inf
        for raw in raw_records:
            inner = validate_envelope(raw, generation=self.generation)
            inner_identity = tuple(
                inner[field]
                for field in ("run_id", "generation", "worker_id", "episode_id", "policy_version")
            )
            if inner_identity != identity:
                raise EnvelopeValidationError("transition batch inner identity differs")
            step_id = int(inner["step_id"])
            logical_time = float(inner["logical_time"])
            if step_id <= previous_step or logical_time < previous_time or logical_time > timestamp:
                raise EnvelopeValidationError("transition batch inner order/time differs")
            inner_payload = inner["payload"]
            if not isinstance(inner_payload, Mapping) or TRANSITION_BATCH_FIELD in inner_payload:
                raise EnvelopeValidationError("transition batch inner payload is invalid")
            if _transition_simulation_time(inner_payload.get("simulation_time")) != logical_time:
                raise EnvelopeValidationError("transition batch inner simulation time differs")
            previous_step = step_id
            previous_time = logical_time
            decoded.append(ReceivedEnvelope(TRANSITION_CLASS, inner, timestamp))
        return tuple(decoded)

    def _deduplicate(self, values: list[ReceivedEnvelope]) -> list[ReceivedEnvelope]:
        accepted: list[ReceivedEnvelope] = []
        candidate = dict(self._seen)
        new_keys: list[str] = []
        for received in values:
            key = received.envelope["idempotency_key"]
            digest = hashlib.sha256(canonical_json(received.envelope)).digest()
            previous = candidate.get(key)
            if previous is None:
                candidate[key] = digest
                new_keys.append(key)
                accepted.append(received)
            elif previous != digest:
                raise IdempotencyConflictError(
                    f"idempotency key reused with different content: {key}"
                )
        self._seen = candidate
        self._seen_order.extend(new_keys)
        while len(self._seen_order) > self.dedup_window:
            expired = self._seen_order.popleft()
            self._seen.pop(expired, None)
        return accepted
