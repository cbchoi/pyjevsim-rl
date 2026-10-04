"""HLAfixedRecord codec for ordered named fields.

Construction: HLAfixedRecord(fields=[(name, codec), ...]).
Wire format: each field encoded in order, each padded to its codec's
OctetBoundary BEFORE encoding. OctetBoundary of the record = the max
of all field boundaries.

Python value mapping: dict[str, Any] keyed by field name. Mirrors
rti/pkg/encoding/fixed_record.go (Go side), byte-identical.
"""

from __future__ import annotations

from typing import Any

from rti1516e.encoding._base import (
    Codec,
    OctetBoundary,
    aligned_offset,
    pad_to_boundary,
)
from rti1516e.errors import EncodingInsufficientBytes, EncodingTypeMismatch


class HLAfixedRecord(Codec):
    """Record with a fixed list of named fields.

    Wire layout per IEEE 1516.2-2010 §4: fields are concatenated in
    declared order. Before each field's bytes the encoder inserts zero-or-
    more padding octets so the field begins at an offset that is a
    multiple of the field's OctetBoundary, measured from the start of
    the record. The record's OctetBoundary is the max of its fields'
    boundaries (1 for an empty record).
    """

    def __init__(self, fields: list[tuple[str, Codec]]) -> None:
        # Defensive copy — a caller could otherwise mutate the field list
        # after construction and break determinism.
        self.fields: list[tuple[str, Codec]] = list(fields)

    @property
    def octet_boundary(self) -> OctetBoundary:
        if not self.fields:
            return 1
        return max(c.octet_boundary for _, c in self.fields)

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, dict):
            raise EncodingTypeMismatch(
                f"HLAfixedRecord cannot encode {type(value).__name__}; expected dict"
            )
        out = bytearray()
        for name, codec in self.fields:
            pad_to_boundary(out, codec.octet_boundary)
            if name not in value:
                raise EncodingTypeMismatch(
                    f"HLAfixedRecord field {name!r} missing from input"
                )
            out.extend(codec.encode(value[name]))
        return bytes(out)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        # Mirror Go's fixed_record.go decode: advance the cursor by the
        # padding required to align to each field's OctetBoundary, then
        # let the field codec decode at the aligned position.
        out: dict[str, Any] = {}
        pos = offset
        for name, codec in self.fields:
            aligned = aligned_offset(pos, codec.octet_boundary)
            if aligned > len(data):
                raise EncodingInsufficientBytes(
                    f"HLAfixedRecord: padding before field {name!r} "
                    f"requires offset {aligned}, have {len(data)}"
                )
            value, consumed = codec.decode(data, aligned)
            out[name] = value
            pos = aligned + consumed
        return out, pos - offset
