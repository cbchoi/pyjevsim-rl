"""Closed, value-only state codec for the opt-in queue branch profile.

No object deserialization, simulation, file writes, or process launch occurs here.
Infinity is admitted only in explicitly identified scheduling fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, NoReturn, cast

from .queue_control import MODES, OBSERVATION_FIELDS, RATES, validate_queue_config

PROFILE = "queue-live-branch-v1"
SCHEMA = "queue-simulator-snapshot-v1"
MAX_BYTES = 32 * 1024 * 1024
MAX_STEPS = 100_000
NAMES = ("dc", "arrival-source", "buffer-server", "completion-sink")
POLICY_FIELDS = {"policy_sha256", "policy_version", "feature_contract_sha256"}
SAMPLING_FIELDS = {
    "domain",
    "phase",
    "master",
    "segment",
    "logical_branch_id",
    "run_id",
    "generation",
    "worker_id",
    "episode_id",
    "sampling_seed",
}
ENV_FIELDS = {
    "instance_id",
    "run_id",
    "episode_number",
    "step_id",
    "observation",
    "seed",
    "done",
    "failed",
    "driver_has_advanced",
}
BEHAVIOR_FIELDS = {"states", "cur_state", "global_time", "cancel_reschedule"}
EXECUTOR_FIELDS = {
    "global_time",
    "next_event_time",
    "cur_state",
    "request_time",
    "cancel_reschedule",
    "instance_time",
    "destruct_time",
    "engine_name",
}
SERVER_FIELDS = {
    "capacity",
    "current",
    "waiting",
    "remaining",
    "mode",
    "initial_count",
    "admitted",
    "source_arrivals",
    "completed",
    "dropped",
    "last_event_time",
    "backlog",
    "energy",
    "trace",
}


class QueueSnapshotError(RuntimeError):
    """A typed, fail-closed snapshot/profile admission failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def fail(message: str, code: str = "incompatible_snapshot") -> NoReturn:
    raise QueueSnapshotError(code, message)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            fail(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def decode(data: bytes) -> dict[str, Any]:
    if type(data) is not bytes or not 0 < len(data) <= MAX_BYTES:
        fail("snapshot bytes are absent or exceed the bounded profile")
    try:
        value = json.loads(
            data,
            object_pairs_hook=_pairs,
            parse_constant=lambda value: fail(f"nonfinite JSON constant {value}"),
        )
        if type(value) is not dict or canonical(value) != data:
            fail("snapshot is not canonical JSON object")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        raise QueueSnapshotError("incompatible_snapshot", "invalid snapshot JSON") from exc
    return value


def fields(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        fail(f"{name} fields differ")
    return value


def integer(value: object, name: str, minimum: int = 0, maximum: int = 2**256 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        fail(f"{name} must be an integer in the admitted range")
    return value


def number(value: object, name: str, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        fail(f"{name} must be finite numeric")
    try:
        result = float(cast(int | float, value))
    except (OverflowError, ValueError) as exc:
        raise QueueSnapshotError("incompatible_snapshot", f"invalid {name}") from exc
    if not math.isfinite(result) or result < minimum:
        fail(f"{name} must be finite and at least {minimum}")
    return result


def text(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        fail(f"{name} must be bounded nonempty text")
    return value


def sha(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        fail(f"{name} must be a lowercase SHA256")
    return value


def time_out(value: float) -> float | str:
    return "+inf" if value == math.inf else value


def time_in(value: object, name: str) -> float:
    return math.inf if value == "+inf" else number(value, name)


def policy_context(value: Mapping[str, object] | None) -> dict[str, Any]:
    raw = (
        dict(value)
        if value is not None
        else {
            "policy_sha256": "0" * 64,
            "policy_version": 0,
            "feature_contract_sha256": "0" * 64,
        }
    )
    fields(raw, POLICY_FIELDS, "policy_context")
    sha(raw["policy_sha256"], "policy_sha256")
    sha(raw["feature_contract_sha256"], "feature_contract_sha256")
    integer(raw["policy_version"], "policy_version")
    return raw


def sampling_context(value: Mapping[str, object] | None, run_id: str) -> dict[str, Any]:
    raw = (
        dict(value)
        if value is not None
        else {
            "domain": "pyjevsim-live-branch-v1",
            "phase": "correctness",
            "master": 0,
            "segment": "prefix",
            "logical_branch_id": "prefix",
            "run_id": run_id,
            "generation": 0,
            "worker_id": "prefix",
            "episode_id": "prefix",
            "sampling_seed": 0,
        }
    )
    fields(raw, SAMPLING_FIELDS, "sampling_context")
    for key in (
        "domain",
        "phase",
        "segment",
        "logical_branch_id",
        "run_id",
        "worker_id",
        "episode_id",
    ):
        text(raw[key], key)
    for key in ("master", "generation", "sampling_seed"):
        integer(raw[key], key)
    if raw["domain"] != "pyjevsim-live-branch-v1" or raw["segment"] not in ("prefix", "suffix"):
        fail("unknown sampling domain/segment")
    if raw["run_id"] != run_id:
        fail("sampling and environment run IDs differ")
    if raw["segment"] == "prefix" and raw["logical_branch_id"] != "prefix":
        fail("prefix sampling requires the shared prefix identity")
    return raw


def _behavior(raw: object, name: str, now: float) -> None:
    row = fields(raw, BEHAVIOR_FIELDS, f"{name}.behavior")
    state = "IDLE" if name == "dc" else "active"
    fields(row["states"], {state}, "behavior states")
    time_in(row["states"][state], "state deadline")
    if row["cur_state"] != state or row["cancel_reschedule"] is not False:
        fail("unsupported behavior state/cancel flag")
    if number(row["global_time"], "model clock") > now:
        fail("model clock exceeds executor clock")


def _executor(raw: object, now: float) -> None:
    row = fields(raw, EXECUTOR_FIELDS, "behavior executor")
    if number(row["global_time"], "executor model clock") > now:
        fail("executor model clock exceeds global clock")
    request = time_in(row["request_time"], "request time")
    if request <= now or time_in(row["next_event_time"], "next event time") != request:
        fail("undrained or inconsistent future event")
    if (
        row["cur_state"] != ""
        or row["cancel_reschedule"] is not False
        or row["instance_time"] != 0
        or row["destruct_time"] != "+inf"
        or row["engine_name"] != "default"
    ):
        fail("unsupported executor lifecycle/state")


def validate_state(raw: object) -> dict[str, Any]:
    state = fields(
        raw,
        {
            "config",
            "config_sha256",
            "seed",
            "delta",
            "max_steps",
            "tape",
            "models",
            "executor",
            "environment",
        },
        "state",
    )
    try:
        checked = validate_queue_config(state["config"])
    except (ValueError, TypeError, OverflowError) as exc:
        raise QueueSnapshotError("incompatible_snapshot", "invalid queue config") from exc
    if (
        canonical(checked) != canonical(state["config"])
        or digest(checked) != state["config_sha256"]
    ):
        fail("queue config identity differs")
    integer(state["seed"], "seed")
    delta = number(state["delta"], "delta")
    if delta == 0:
        fail("delta must be positive")
    maximum = integer(state["max_steps"], "max_steps", 1, MAX_STEPS)
    tape = fields(state["tape"], {"seed", "end_time", "events"}, "arrival tape")
    if tape["seed"] != state["seed"]:
        fail("arrival tape seed differs")
    end = number(tape["end_time"], "tape end time")
    arrival = checked["arrival_spec"]
    configured_end = (
        arrival["end_time"]
        if arrival["kind"] == "explicit"
        else arrival["slot_count"] * arrival["slot_width"]
    )
    if end != configured_end:
        fail("arrival tape horizon differs from config")
    if type(tape["events"]) is not list or len(tape["events"]) > MAX_STEPS:
        fail("arrival tape exceeds bounded profile")
    previous = 0.0
    identifiers: set[str] = set()
    for item in tape["events"]:
        if type(item) is not list or len(item) != 2:
            fail("arrival tape event differs")
        instant = number(item[0], "arrival time")
        name = text(item[1], "arrival job ID")
        if not 0 < instant <= end or instant < previous or name in identifiers:
            fail("arrival tape event order/identity differs")
        previous = instant
        identifiers.add(name)
    if arrival["kind"] == "explicit" and tape["events"] != [
        [item["time"], item["job_id"]] for item in arrival["events"]
    ]:
        fail("explicit input tape differs from config")
    if arrival["kind"] == "bernoulli-slots":
        last_slot = 0
        for instant, name in tape["events"]:
            if not name.startswith("arrival-") or not name[8:].isdigit():
                fail("unknown generated arrival identity")
            slot = int(name[8:])
            if (
                not last_slot < slot <= arrival["slot_count"]
                or instant != slot * arrival["slot_width"]
            ):
                fail("generated arrival slot differs")
            last_slot = slot
    ex = fields(
        state["executor"],
        {
            "global_time",
            "target_time",
            "time_resolution",
            "simulation_mode",
            "calendar",
            "models",
        },
        "executor",
    )
    now = number(ex["global_time"], "global time")
    if (
        ex["target_time"] != 0
        or ex["time_resolution"] != 1.0
        or ex["simulation_mode"] != "SIMULATION_IDLE"
    ):
        fail("unsupported HLA executor control state")
    models = fields(state["models"], set(NAMES), "models")
    wrappers = fields(ex["models"], set(NAMES), "executor models")
    calendar = fields(ex["calendar"], set(NAMES), "event calendar")
    for name in NAMES:
        row = fields(models[name], {"behavior", "values"}, f"model {name}")
        _behavior(row["behavior"], name, now)
        _executor(wrappers[name], now)
        behavior = row["behavior"]
        wrapper = wrappers[name]
        if wrapper["global_time"] != behavior["global_time"]:
            fail("model and executor callback clocks differ")
        deadline = time_in(behavior["states"][behavior["cur_state"]], "deadline")
        if time_in(wrapper["request_time"], "request time") != wrapper["global_time"] + deadline:
            fail("model deadline and absolute request time differ")
        if calendar[name] != wrappers[name]["request_time"]:
            fail("calendar/model request time differs")
    source = fields(models["arrival-source"]["values"], {"index", "exhausted"}, "source")
    index = integer(source["index"], "source cursor", 0, len(tape["events"]))
    if type(source["exhausted"]) is not bool or source["exhausted"] != (now >= end):
        fail("source exhaustion differs from horizon")
    if any(item[0] > now for item in tape["events"][:index]) or any(
        item[0] <= now for item in tape["events"][index:]
    ):
        fail("source cursor differs from committed boundary")
    expected_source = (
        "+inf"
        if source["exhausted"]
        else (tape["events"][index][0] if index < len(tape["events"]) else end)
    )
    if calendar["arrival-source"] != expected_source:
        fail("source next event differs from tape cursor")
    server = fields(models["buffer-server"]["values"], SERVER_FIELDS, "server")
    if server["capacity"] != checked["waiting_capacity"] or server["mode"] not in MODES:
        fail("server configuration differs")
    if server["current"] is not None:
        text(server["current"], "current job")
    if type(server["waiting"]) is not list or len(server["waiting"]) > server["capacity"]:
        fail("waiting queue differs")
    for item in server["waiting"]:
        text(item, "waiting job")
    live = server["waiting"] + ([] if server["current"] is None else [server["current"]])
    if len(set(live)) != len(live) or (
        server["current"] is None and (server["waiting"] or server["mode"] != "idle")
    ):
        fail("duplicate/orphaned server inventory")
    for key in ("initial_count", "admitted", "source_arrivals", "completed", "dropped"):
        integer(server[key], key)
    for key in ("remaining", "last_event_time", "backlog", "energy"):
        number(server[key], key)
    if server["last_event_time"] > now or server["remaining"] > 1:
        fail("invalid server timing anchor")
    if server["initial_count"] != len(checked["initial_waiting"]) + int(
        checked["initial_service"] is not None
    ):
        fail("initial job count differs from config")
    rate = RATES[MODES.index(server["mode"])]
    expected_deadline = (
        server["remaining"] / rate if server["current"] is not None and rate else math.inf
    )
    if (
        time_in(models["buffer-server"]["behavior"]["states"]["active"], "server deadline")
        != expected_deadline
    ):
        fail("server schedule differs from original residual work")
    if wrappers["buffer-server"]["global_time"] != server["last_event_time"]:
        fail("server model clock differs from its arithmetic anchor")
    if server["source_arrivals"] != index or server["admitted"] != len(live) + server["completed"]:
        fail("server arrival/admission conservation differs")
    if (
        server["initial_count"] + server["source_arrivals"]
        != server["admitted"] + server["dropped"]
    ):
        fail("server job conservation differs")
    if type(server["trace"]) is not list or len(server["trace"]) > MAX_STEPS * 16:
        fail("server trace exceeds bounded profile")
    previous = 0.0
    for event in server["trace"]:
        if (
            type(event) is not list
            or len(event) != 3
            or event[0] not in {"output", "internal", "external-arrival", "external-action"}
        ):
            fail("unknown server trace event")
        instant = number(event[1], "trace instant")
        if instant < previous or instant > now:
            fail("trace time order differs")
        previous = instant
        if event[0] == "external-action":
            if fields(event[2], {"mode"}, "trace action")["mode"] not in MODES:
                fail("unknown trace action")
        else:
            text(event[2], "trace job")
    sink = fields(models["completion-sink"]["values"], {"ledger", "pending"}, "sink")
    if (
        sink["pending"] != []
        or type(sink["ledger"]) is not list
        or len(sink["ledger"]) != server["completed"]
    ):
        fail("undrained/inconsistent sink")
    previous = 0.0
    completed: set[str] = set()
    for item in sink["ledger"]:
        fields(item, {"job_id", "completion_time"}, "completion")
        job = text(item["job_id"], "completed job")
        instant = number(item["completion_time"], "completion time")
        if instant < previous or instant > now or job in completed or job in live:
            fail("completion order/inventory differs")
        previous = instant
        completed.add(job)
    fields(models["dc"]["values"], set(), "default catcher")
    if calendar["dc"] != "+inf" or calendar["completion-sink"] != "+inf":
        fail("nonpassive catcher/sink")
    env = fields(state["environment"], ENV_FIELDS, "environment")
    text(env["instance_id"], "instance ID")
    text(env["run_id"], "run ID")
    integer(env["episode_number"], "episode number", 1)
    steps = integer(env["step_id"], "step cursor", 0, maximum)
    if (
        env["seed"] != state["seed"]
        or env["done"] is not False
        or env["failed"] is not False
        or steps == maximum
    ):
        fail("capture requires a live nonterminal environment")
    if type(env["driver_has_advanced"]) is not bool or env["driver_has_advanced"] != (steps > 0):
        fail("driver advancement differs from step cursor")
    observation = fields(env["observation"], set(OBSERVATION_FIELDS), "observation")
    if observation["logical_time"] != now:
        fail("environment observation/clock differs")
    return state
