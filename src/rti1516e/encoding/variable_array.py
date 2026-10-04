"""HLAvariableArray dynamically-sized array codec.

Wire format: 4-byte HLAinteger32BE length prefix, then N elements
concatenated (with per-element padding to element's OctetBoundary).

OctetBoundary: max(4, element.octet_boundary) — the length prefix is
4-byte aligned but the first element pads to its own boundary after.
"""

from __future__ import annotations

import struct
from typing import Any

from rti1516e.encoding._base import (
    Codec,
    OctetBoundary,
    aligned_offset,
    pad_to_boundary,
)
from rti1516e.errors import EncodingInsufficientBytes, EncodingTypeMismatch

# 4-byte big-endian unsigned int for the length prefix. Mirrors Go's
# binary.BigEndian.PutUint32/Uint32 in rti/pkg/encoding/variable_array.go.
_S_LEN = struct.Struct(">I")


class HLAvariableArray(Codec):
    """Variable-length array of elements encoded by ``element``.

    Mirrors ``rti/pkg/encoding/variable_array.go``: 4-byte big-endian
    element count followed by N concatenated element encodings with
    inter-element padding to ``element.octet_boundary``. The composite's
    OctetBoundary is ``max(4, element.octet_boundary)`` because the
    length prefix forces at least 4-byte alignment regardless of the
    element type.
    """

    def __init__(self, element: Codec) -> None:
        self.element = element

    @property
    def octet_boundary(self) -> OctetBoundary:
        return max(4, self.element.octet_boundary)

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, (list, tuple)):
            raise EncodingTypeMismatch(
                f"HLAvariableArray cannot encode {type(value).__name__} "
                "(want list or tuple)"
            )
        count = len(value)
        if count > 0xFFFFFFFF:
            raise EncodingTypeMismatch(
                f"HLAvariableArray: count {count} exceeds uint32 max"
            )
        boundary = self.element.octet_boundary
        out = bytearray(_S_LEN.pack(count))
        for item in value:
            pad_to_boundary(out, boundary)
            out.extend(self.element.encode(item))
        return bytes(out)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 4:
            raise EncodingInsufficientBytes(
                f"HLAvariableArray: need 4 bytes for length prefix at "
                f"offset {offset}, have {max(0, len(data) - offset)}"
            )
        (count,) = _S_LEN.unpack_from(data, offset)
        boundary = self.element.octet_boundary
        out: list[Any] = []
        # Alignment within the composite is relative to its own start
        # (mirrors Go's slice-based decoder in rti/pkg/encoding/variable_array.go).
        pos = 4
        for _ in range(count):
            pos = aligned_offset(pos, boundary)
            if offset + pos > len(data):
                raise EncodingInsufficientBytes(
                    f"HLAvariableArray: short buffer at offset {offset + pos} "
                    f"(buffer length {len(data)})"
                )
            element_value, consumed = self.element.decode(data, offset + pos)
            out.append(element_value)
            pos += consumed
        return out, pos
