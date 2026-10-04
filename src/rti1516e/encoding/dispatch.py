"""Resolve HLA type specifications to configured codecs.

The conformance test in pysdk/tests/spec/m4/test_spec_m4_encoding_conformance.py
iterates tests/conformance/encoding_vectors.json and calls codec_for(vec.type)
for each vector. The test passes when 100% of vectors round-trip
byte-identical to Go.

Two entry points:

- ``codec_for(spec)`` — accepts either a primitive type-name string
  (e.g. "HLAinteger32BE") or a parsed type spec dict
  (e.g. {"kind": "HLAfixedArray", "element": "HLAfloat64BE",
  "cardinality": 3}). Returns a configured Codec instance.

- ``parse_type_spec(raw)`` — normalize raw input (string OR dict) into a
  canonical type-spec dict. Useful for tests that need to introspect
  the resolved spec separately from constructing the codec.

JSON descriptor shapes (mirrors rti/pkg/encoding/dispatch.go on the Go side):

- Primitive: ``"HLAinteger32BE"`` (bare string) — auto-wrapped to
  ``{"kind": "HLAinteger32BE"}`` by parse_type_spec.
- HLAfixedArray: ``{"kind": "HLAfixedArray", "element": <type>,
  "cardinality": <int>}``
- HLAvariableArray: ``{"kind": "HLAvariableArray", "element": <type>}``
- HLAfixedRecord: ``{"kind": "HLAfixedRecord", "fields":
  [{"name": <str>, "type": <type>}, ...]}``
- HLAvariantRecord: ``{"kind": "HLAvariantRecord", "discriminator":
  <primitive name>, "alternatives": [{"discriminant": <value>,
  "type": <type>}, ...]}``. The Python value-shape uses the literal
  field names ``"discriminator"`` and ``"value"`` (matching Go's
  rti/pkg/encoding/variant_record.go); see the conformance vectors.
- HLAopaqueData: ``{"kind": "HLAopaqueData"}``
"""

from __future__ import annotations

from typing import Any

from rti1516e.encoding._base import Codec, OctetBoundary
from rti1516e.encoding.byte_codec import (
    HLAASCIIchar,
    HLAboolean,
    HLAoctet,
    HLAoctetPairBE,
    HLAoctetPairLE,
    HLAunicodeChar,
)
from rti1516e.encoding.fixed_array import HLAfixedArray
from rti1516e.encoding.fixed_record import HLAfixedRecord
from rti1516e.encoding.float_codec import (
    HLAfloat32BE,
    HLAfloat32LE,
    HLAfloat64BE,
    HLAfloat64LE,
)
from rti1516e.encoding.integer import (
    HLAinteger16BE,
    HLAinteger16LE,
    HLAinteger32BE,
    HLAinteger32LE,
    HLAinteger64BE,
    HLAinteger64LE,
)
from rti1516e.encoding.opaque import HLAopaqueData
from rti1516e.encoding.string_codec import HLAASCIIstring, HLAunicodeString
from rti1516e.encoding.variable_array import HLAvariableArray
from rti1516e.encoding.variant_record import HLAvariantRecord
from rti1516e.errors import EncodingTypeMismatch

# Registry of primitive codec classes keyed by their HLA name. Mirrors the
# Go side's PrimitiveByName (rti/pkg/encoding/registry.go). Each class is
# zero-arg constructible.
_PRIMITIVES: dict[str, type[Codec]] = {
    # Integers
    "HLAinteger16BE": HLAinteger16BE,
    "HLAinteger16LE": HLAinteger16LE,
    "HLAinteger32BE": HLAinteger32BE,
    "HLAinteger32LE": HLAinteger32LE,
    "HLAinteger64BE": HLAinteger64BE,
    "HLAinteger64LE": HLAinteger64LE,
    # Floats
    "HLAfloat32BE": HLAfloat32BE,
    "HLAfloat32LE": HLAfloat32LE,
    "HLAfloat64BE": HLAfloat64BE,
    "HLAfloat64LE": HLAfloat64LE,
    # Bytes / chars / boolean
    "HLAoctet": HLAoctet,
    "HLAoctetPairBE": HLAoctetPairBE,
    "HLAoctetPairLE": HLAoctetPairLE,
    "HLAboolean": HLAboolean,
    "HLAASCIIchar": HLAASCIIchar,
    "HLAunicodeChar": HLAunicodeChar,
    # Strings (length-prefixed)
    "HLAASCIIstring": HLAASCIIstring,
    "HLAunicodeString": HLAunicodeString,
}

# Names of primitive codecs whose canonical decoded value is a Python int.
# Used by variant-record dispatch to coerce JSON-string discriminant keys
# (when present) to the matching int type — although the conformance
# vectors deliver discriminants as JSON numbers (int after parsing), the
# coercion is defensive against future vectors keyed by stringified ints.
_INTEGER_PRIMITIVES: frozenset[str] = frozenset(
    {
        "HLAinteger16BE",
        "HLAinteger16LE",
        "HLAinteger32BE",
        "HLAinteger32LE",
        "HLAinteger64BE",
        "HLAinteger64LE",
        "HLAoctet",
    }
)


def parse_type_spec(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Normalize a type spec (string or dict) into the canonical dict form.

    Strings normalize to ``{"kind": <name>}``; dicts pass through after a
    presence check on ``"kind"``. Anything else raises
    :class:`EncodingTypeMismatch`.
    """
    if isinstance(raw, str):
        return {"kind": raw}
    if isinstance(raw, dict):
        if "kind" not in raw:
            raise EncodingTypeMismatch(
                f"type spec missing 'kind': {raw!r}"
            )
        return raw
    raise EncodingTypeMismatch(
        f"type spec must be str or dict, got {type(raw).__name__}"
    )


def codec_for(spec: str | dict[str, Any]) -> Codec:
    """Return a configured Codec for the given type spec.

    Examples:
        codec_for("HLAinteger32BE")
        codec_for({"kind": "HLAfixedArray", "element": "HLAoctet",
                   "cardinality": 4})
        codec_for({"kind": "HLAvariantRecord",
                   "discriminator": "HLAinteger32BE",
                   "alternatives": [
                       {"discriminant": 1, "type": "HLAoctet"},
                       {"discriminant": 2, "type": "HLAfloat64BE"},
                   ]})
    """
    parsed = parse_type_spec(spec)
    kind = parsed["kind"]

    # Primitive: identified by name, zero-arg construction.
    primitive_cls = _PRIMITIVES.get(kind)
    if primitive_cls is not None:
        return primitive_cls()

    # Composite kinds. Each branch recurses through codec_for for nested
    # element/field/alternative types.
    if kind == "HLAopaqueData":
        # The conformance vectors deliver opaque payloads as hex strings
        # (JSON has no bytes type). Per the spec test contract
        # ``decoded == vec.value``, the codec returned via dispatch must
        # decode opaque blobs back to the same hex-string shape the JSON
        # delivers. The wrapper below preserves byte-identical wire
        # output (delegated to the real HLAopaqueData) while presenting
        # the JSON-friendly Python value-shape on encode and decode.
        return _OpaqueHexAdapter()

    if kind == "HLAfixedArray":
        element_spec = parsed.get("element")
        if element_spec is None:
            raise EncodingTypeMismatch(
                "HLAfixedArray descriptor missing 'element'"
            )
        if "cardinality" not in parsed:
            raise EncodingTypeMismatch(
                "HLAfixedArray descriptor missing 'cardinality'"
            )
        cardinality = _coerce_cardinality(parsed["cardinality"])
        element_codec = codec_for(element_spec)
        return HLAfixedArray(element_codec, cardinality)

    if kind == "HLAvariableArray":
        element_spec = parsed.get("element")
        if element_spec is None:
            raise EncodingTypeMismatch(
                "HLAvariableArray descriptor missing 'element'"
            )
        element_codec = codec_for(element_spec)
        return HLAvariableArray(element_codec)

    if kind == "HLAfixedRecord":
        raw_fields = parsed.get("fields")
        if not isinstance(raw_fields, list):
            raise EncodingTypeMismatch(
                "HLAfixedRecord descriptor missing or non-list 'fields'"
            )
        fields: list[tuple[str, Codec]] = []
        for i, raw_field in enumerate(raw_fields):
            if not isinstance(raw_field, dict):
                raise EncodingTypeMismatch(
                    f"HLAfixedRecord field {i}: not an object"
                )
            name = raw_field.get("name")
            if not isinstance(name, str):
                raise EncodingTypeMismatch(
                    f"HLAfixedRecord field {i}: missing or non-string 'name'"
                )
            type_spec = raw_field.get("type")
            if type_spec is None:
                raise EncodingTypeMismatch(
                    f"HLAfixedRecord field {name!r}: missing 'type'"
                )
            fields.append((name, codec_for(type_spec)))
        return HLAfixedRecord(fields)

    if kind == "HLAvariantRecord":
        return _build_variant_record(parsed)

    raise EncodingTypeMismatch(f"unknown type kind: {kind!r}")


class _OpaqueHexAdapter(Codec):
    """Vector-friendly facade over HLAopaqueData.

    Wire bytes are produced by the real HLAopaqueData codec (W1D), so the
    cross-language byte-identicality contract still flows through the
    canonical implementation. The adapter only reshapes the Python value
    side: encode accepts bytes-like OR a hex string (delegated unchanged),
    and decode returns a lowercase hex string instead of ``bytes`` so the
    JSON-loaded vector value compares equal in
    ``test_spec_m4_encoding_conformance.py``.

    Lives in dispatch.py (not opaque.py) because the underlying codec is
    Wave-1-frozen — wrapping at dispatch time keeps all per-codec modules
    untouched while giving the spec test the value-shape it expects.
    """

    def __init__(self) -> None:
        self._inner = HLAopaqueData()

    @property
    def octet_boundary(self) -> OctetBoundary:
        return self._inner.octet_boundary

    def encode(self, value: Any) -> bytes:
        # HLAopaqueData.encode already accepts both bytes-like and a
        # hex-encoded str (per its docstring), so a plain delegation
        # round-trips every conformance vector input shape.
        return self._inner.encode(value)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        payload, consumed = self._inner.decode(data, offset)
        return payload.hex(), consumed


def _coerce_cardinality(raw: Any) -> int:
    """Coerce the JSON-loaded cardinality value to a non-negative int.

    Mirrors the Go side's ``jsonCardinality`` (rti/pkg/encoding/dispatch.go):
    accepts int or a float that's an exact non-negative integer.
    """
    if isinstance(raw, bool):
        # Reject bool (an int subclass) — clearly a category error in
        # a cardinality slot.
        raise EncodingTypeMismatch(
            f"cardinality must be a non-negative integer, got bool {raw!r}"
        )
    if isinstance(raw, int):
        if raw < 0:
            raise EncodingTypeMismatch(
                f"cardinality {raw} is negative"
            )
        return raw
    if isinstance(raw, float):
        if not raw.is_integer() or raw < 0:
            raise EncodingTypeMismatch(
                f"cardinality {raw!r} is not a non-negative integer"
            )
        return int(raw)
    raise EncodingTypeMismatch(
        f"cardinality has unsupported type {type(raw).__name__}"
    )


def _build_variant_record(parsed: dict[str, Any]) -> HLAvariantRecord:
    """Construct an HLAvariantRecord from the JSON descriptor shape.

    JSON shape (mirrors rti/pkg/encoding/dispatch.go ``jsonVariantRecord``):

        {"kind": "HLAvariantRecord",
         "discriminator": "<primitive name>",
         "alternatives": [
             {"discriminant": <value>, "type": <type spec>},
             ...
         ]}

    The conformance vectors' value-shape uses the literal field names
    ``"discriminator"`` (for the discriminant) and ``"value"`` (for the
    active alternative); we pass those names to the HLAvariantRecord
    constructor so encode/decode see the keys the vectors specify.

    Discriminant key normalization: JSON delivers numeric discriminants
    as Python ``int`` (or ``float`` for non-integral, which the variant
    codec's ``_canonical_discriminant`` then round-trips through the
    discriminant codec). For integer-typed discriminants we additionally
    accept stringified ints (``"1"`` -> ``1``) to match the example shape
    documented in the M4 dispatch plan; non-integer keys for integer
    discriminants are a contract error.
    """
    disc_name = parsed.get("discriminator")
    if not isinstance(disc_name, str):
        raise EncodingTypeMismatch(
            "HLAvariantRecord descriptor missing or non-string 'discriminator'"
        )
    if disc_name not in _PRIMITIVES:
        raise EncodingTypeMismatch(
            f"HLAvariantRecord discriminator {disc_name!r} is not a primitive"
        )
    disc_codec = _PRIMITIVES[disc_name]()

    raw_alts = parsed.get("alternatives")
    if not isinstance(raw_alts, list):
        raise EncodingTypeMismatch(
            "HLAvariantRecord descriptor missing or non-list 'alternatives'"
        )

    variants: dict[Any, tuple[str, Codec]] = {}
    for i, raw_alt in enumerate(raw_alts):
        if not isinstance(raw_alt, dict):
            raise EncodingTypeMismatch(
                f"HLAvariantRecord alternative {i}: not an object"
            )
        if "discriminant" not in raw_alt:
            raise EncodingTypeMismatch(
                f"HLAvariantRecord alternative {i}: missing 'discriminant'"
            )
        type_spec = raw_alt.get("type")
        if type_spec is None:
            raise EncodingTypeMismatch(
                f"HLAvariantRecord alternative {i}: missing 'type'"
            )
        key = _canonicalize_discriminant_key(
            raw_alt["discriminant"], disc_name, disc_codec
        )
        alt_codec = codec_for(type_spec)
        # Variant-name in the value-shape is hardcoded to "value" — see
        # the conformance vectors and rti/pkg/encoding/variant_record.go.
        variants[key] = ("value", alt_codec)

    return HLAvariantRecord(
        discriminant=("discriminator", disc_codec),
        variants=variants,
    )


def _canonicalize_discriminant_key(
    raw_key: Any, disc_name: str, disc_codec: Codec
) -> Any:
    """Normalize a JSON-loaded discriminant key to the canonical Python
    value the discriminant codec emits on decode.

    Round-trips ``raw_key`` through ``disc_codec.encode`` then
    ``disc_codec.decode`` so the variants-map key compares equal to the
    value HLAvariantRecord computes at decode time. Mirrors the Go side's
    ``parseDiscriminantValue`` (rti/pkg/encoding/dispatch.go).

    Defensive accepts: stringified ints for integer-typed discriminants
    (``"1"`` -> ``1``), in case a future JSON descriptor uses dict-keyed
    variants where keys are always strings.
    """
    key = raw_key
    if isinstance(key, str) and disc_name in _INTEGER_PRIMITIVES:
        try:
            key = int(key)
        except ValueError as exc:
            raise EncodingTypeMismatch(
                f"discriminator {disc_name}: cannot parse {raw_key!r} as int: {exc}"
            ) from exc
    encoded = disc_codec.encode(key)
    canonical, _ = disc_codec.decode(encoded)
    return canonical
