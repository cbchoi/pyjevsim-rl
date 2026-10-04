"""Bounded closed envelope, with provider validation before allocation.

Round-trip validation is structural evidence only, not semantic conformance.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass

from .contracts import (
    SNAPSHOT_SCHEMA, ContinuationSnapshot, ProfileDescriptor, canonical_bytes,
    checked_id, checked_sha, decode_json, digest, exact_fields, fail, normalize_owned,
)
from .registry import ContinuationRegistry

_FIELDS = {
    "schema_id", "profile", "identities", "logical_context", "topology",
    "engine_state", "model_state", "boundary_state", "integrity",
}
_IDENTITIES = {
    "engine_source_sha256", "model_source_sha256", "boundary_source_sha256",
    "runtime_sha256", "config_sha256", "obligation_manifest_sha256",
}


def _topology(value: dict, profile: ProfileDescriptor) -> None:
    topology = exact_fields(value, {"nodes", "couplings", "shared_resources", "aliases"}, "topology")
    nodes, resources = topology["nodes"], topology["shared_resources"]
    if type(nodes) is not dict or type(resources) is not dict or not nodes:
        fail("topology nodes/resources must be objects and nodes must be nonempty")
    if len(nodes) > profile.limits.max_models or len(nodes) + len(resources) > profile.limits.max_nodes:
        fail("topology inventory exceeds limits", "CC_LIMIT")
    if set(nodes) & set(resources):
        fail("model/resource semantic IDs overlap")
    for semantic_id, node in nodes.items():
        checked_id(semantic_id, "semantic model ID", profile.limits.max_id_bytes)
        exact_fields(node, {"type_id", "schema_id", "inputs", "outputs"}, "model node")
        checked_id(node["type_id"], "model type", profile.limits.max_id_bytes)
        checked_id(node["schema_id"], "model schema", profile.limits.max_id_bytes)
        for role in ("inputs", "outputs"):
            ports = node[role]
            if type(ports) is not list:
                fail("ports must be lists")
            for port in ports:
                checked_id(port, "port", profile.limits.max_id_bytes)
            if len(set(ports)) != len(ports):
                fail("duplicate port")
    for semantic_id, node in resources.items():
        checked_id(semantic_id, "resource ID", profile.limits.max_id_bytes)
        exact_fields(node, {"type_id", "schema_id"}, "resource node")
        checked_id(node["type_id"], "resource type", profile.limits.max_id_bytes)
        checked_id(node["schema_id"], "resource schema", profile.limits.max_id_bytes)
    couplings = topology["couplings"]
    if type(couplings) is not list:
        fail("couplings must be a list")
    if len(couplings) > profile.limits.max_couplings:
        fail("coupling limit exceeded", "CC_LIMIT")
    for item in couplings:
        exact_fields(item, {"source_node", "source_port", "target_node", "target_port"}, "coupling")
        for endpoint in ("source", "target"):
            node, port = item[endpoint + "_node"], item[endpoint + "_port"]
            checked_id(node, "coupling node", profile.limits.max_id_bytes)
            checked_id(port, "coupling port", profile.limits.max_id_bytes)
            if node not in nodes or port not in nodes[node]["inputs"] + nodes[node]["outputs"]:
                fail("coupling references undeclared node/port")
        # Coupling order and multiplicity are not normalized: a provider must
        # define them, and hierarchical external ports can reverse directions.
    aliases = topology["aliases"]
    if type(aliases) is not list:
        fail("aliases must be a list")
    destinations = set(nodes) | set(resources)
    paths = set()
    for alias in aliases:
        exact_fields(alias, {"owner", "path", "target"}, "alias")
        for name in ("owner", "path", "target"):
            checked_id(alias[name], "alias " + name, profile.limits.max_id_bytes)
        if alias["owner"] not in destinations or alias["target"] not in destinations:
            fail("alias references an undeclared semantic ID")
        if (alias["owner"], alias["path"]) in paths:
            fail("alias path is assigned more than once")
        paths.add((alias["owner"], alias["path"]))


def _validate(payload: dict, registry: ContinuationRegistry) -> tuple[dict, ProfileDescriptor, bytes]:
    exact_fields(payload, _FIELDS, "snapshot envelope")
    if payload["schema_id"] != SNAPSHOT_SCHEMA:
        fail("unsupported snapshot schema", "CC_INCOMPATIBLE_IDENTITY")
    profile = ProfileDescriptor.from_payload(payload["profile"])
    bundle = registry.resolve(profile)
    # Apply the narrower installed limits before provider payload callbacks.
    before = canonical_bytes(payload, limits=profile.limits)
    identities = exact_fields(payload["identities"], _IDENTITIES, "identities")
    for key in _IDENTITIES:
        checked_sha(identities[key], key)
    expected = {
        "engine_source_sha256": profile.engine.implementation_sha256,
        "model_source_sha256": profile.model.implementation_sha256,
        "boundary_source_sha256": profile.boundary.implementation_sha256,
        "runtime_sha256": profile.runtime_sha256,
        "obligation_manifest_sha256": profile.obligation_manifest_sha256,
    }
    if any(identities[key] != value for key, value in expected.items()):
        fail("envelope identities differ from installed profile", "CC_INCOMPATIBLE_IDENTITY")
    logical = exact_fields(payload["logical_context"],
                           {"family_id", "prefix_id", "policy_context", "sampling_context"}, "logical context")
    checked_id(logical["family_id"], "family ID", profile.limits.max_id_bytes)
    checked_id(logical["prefix_id"], "prefix ID", profile.limits.max_id_bytes)
    if type(logical["policy_context"]) is not dict or type(logical["sampling_context"]) is not dict:
        fail("policy and sampling contexts must be objects")
    _topology(payload["topology"], profile)
    for name in ("engine_state", "model_state", "boundary_state"):
        if type(payload[name]) is not dict:
            fail(f"{name} must be an object")
    model_construction = payload["model_state"].get("construction")
    if type(model_construction) is not dict or type(model_construction.get("config")) is not dict:
        fail("model construction must include its complete configuration")
    if digest(model_construction["config"]) != identities["config_sha256"]:
        fail("model construction config hash differs", "CC_INCOMPATIBLE_IDENTITY")
    if type(payload["engine_state"].get("construction")) is not dict:
        fail("engine construction descriptor missing")
    if type(payload["boundary_state"].get("descriptor")) is not dict:
        fail("RL boundary descriptor missing")
    checked_sha(payload["integrity"], "integrity")
    if digest({key: value for key, value in payload.items() if key != "integrity"}) != payload["integrity"]:
        fail("snapshot integrity differs", "CC_INCOMPATIBLE_IDENTITY")
    # No extension callbacks or writes occur between the bounded canonical
    # snapshot above and these validators. Reuse only this immutable byte value.
    validations = (
        (bundle.engine.validate_payload, (payload["engine_state"], payload["topology"], profile)),
        (bundle.model.validate_payload, (payload["model_state"], payload["topology"], profile)),
        (bundle.boundary.validate_payload, (payload["boundary_state"], logical["policy_context"], profile)),
        (bundle.model.validate_composition, (payload["engine_state"], payload["model_state"],
                                             payload["boundary_state"], logical)),
    )
    for validate, arguments in validations:
        if validate(*arguments) is not None:
            fail("installed validator must raise on failure and return None on success")
    if canonical_bytes(payload) != before:
        fail("installed payload validator mutated its input", "CC_INVALID_PAYLOAD")
    registry.resolve(profile)
    # Identity verifiers are trusted callbacks too. A validator may have kept a
    # reference to its input; the final verifier must not rewrite the anchor.
    after_identity = canonical_bytes(payload)
    if after_identity != before:
        fail("installed identity verifier mutated payload", "CC_INVALID_PAYLOAD")
    return payload, profile, after_identity


def encode_snapshot(payload: dict, *, registry: ContinuationRegistry) -> ContinuationSnapshot:
    exact_fields(payload, _FIELDS - {"integrity"}, "unsigned snapshot envelope")
    # Detach before passing values to trusted validators.
    owned = normalize_owned(payload)
    detached = owned.to_plain()
    detached["integrity"] = hashlib.sha256(owned.canonical_bytes).hexdigest()
    _, _, encoded = _validate(detached, registry)
    return ContinuationSnapshot(encoded)


def _snapshot_bytes(snapshot: ContinuationSnapshot | bytes) -> bytes:
    if type(snapshot) is ContinuationSnapshot:
        data = snapshot.data
    elif type(snapshot) is bytes:
        data = snapshot
    else:
        fail("decode requires bytes or ContinuationSnapshot")
    return data


def decode_snapshot(snapshot: ContinuationSnapshot | bytes, *, registry: ContinuationRegistry) -> dict:
    payload, _, _ = _validate(decode_json(_snapshot_bytes(snapshot)), registry)
    return payload


@dataclass(frozen=True, slots=True)
class _CheckedPayload:
    """Operation-local input anchor, never a cached candidate/source admission.

    Working copies may only be acquired in the initial validation phase. The
    immutable bytes remain a comparison anchor after callbacks, not proof that
    any current runtime state or installed source is still valid.
    """

    profile: ProfileDescriptor
    context: object
    _stamp: object
    _owned: object

    def working_copy(self) -> dict:
        self.context.require(self._stamp)
        return self._owned.to_plain()

    def assert_unchanged(self, working: dict) -> None:
        # stamp() checks this operation is live, on its owner thread, and bound
        # to the same registration. It does not skip any source checkpoint.
        self.context.stamp()
        if canonical_bytes(working) != self._owned.canonical_bytes:
            fail("installed restore callback mutated its input", "CC_CONFORMANCE_FAILED")


@contextmanager
def decode_checked(snapshot: ContinuationSnapshot | bytes, *, registry: ContinuationRegistry):
    """Keep input ownership and operation lifetime private to restore."""
    payload, profile, _ = _validate(decode_json(_snapshot_bytes(snapshot)), registry)
    with registry.validation_context(profile, operation="restore-input") as context:
        yield _CheckedPayload(profile, context, context.stamp(), normalize_owned(payload))
