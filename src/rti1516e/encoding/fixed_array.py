"""HLAfixedArray fixed-cardinality array codec.

Construction: HLAfixedArray(element_codec=<Codec>, cardinality=N).
Wire format: N elements concatenated. OctetBoundary = element's
OctetBoundary. Padding inserted between elements when the element's
encoded length isn't a multiple of its OctetBoundary.

The element_codec is resolved at construction time by the dispatcher
(rti1516e.encoding.dispatch.codec_for); HLAfixedArray itself is type-
erased over the element type.
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


class HLAfixedArray(Codec):
    """Array of ``cardinality`` elements, each encoded by ``element``.

    Mirrors ``rti/pkg/encoding/fixed_array.go``: byte-identical output,
    same OctetBoundary semantics. The composite's OctetBoundary equals
    the element's OctetBoundary; inter-element padding is applied only
    when the running buffer length is not already a multiple of that
    boundary (so primitive elements whose encoded width matches their
    boundary emit no padding).
    """

    def __init__(self, element: Codec, cardinality: int) -> None:
        if cardinality < 0:
            raise ValueError(
                f"HLAfixedArray: negative cardinality {cardinality}"
            )
        self.element = element
        self.cardinality = cardinality

    @property
    def octet_boundary(self) -> OctetBoundary:
        return self.element.octet_boundary

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, (list, tuple)):
            raise EncodingTypeMismatch(
                f"HLAfixedArray cannot encode {type(value).__name__} "
                "(want list or tuple)"
            )
        if len(value) != self.cardinality:
            raise EncodingTypeMismatch(
                f"HLAfixedArray expected {self.cardinality} elements, "
                f"got {len(value)}"
            )
        boundary = self.element.octet_boundary
        out = bytearray()
        for item in value:
            pad_to_boundary(out, boundary)
            out.extend(self.element.encode(item))
        return bytes(out)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        boundary = self.element.octet_boundary
        out: list[Any] = []
        # Alignment within the composite is relative to its own start, so
        # ``offset`` is treated as position 0 of the composite's frame.
        # This matches the Go reference, which slices the buffer at the
        # composite's start before decoding (rti/pkg/encoding/fixed_array.go).
        pos = 0
        for _ in range(self.cardinality):
            pos = aligned_offset(pos, boundary)
            if offset + pos > len(data):
                raise EncodingInsufficientBytes(
                    f"HLAfixedArray: short buffer at offset {offset + pos} "
                    f"(buffer length {len(data)})"
                )
            element_value, consumed = self.element.decode(data, offset + pos)
            out.append(element_value)
            pos += consumed
        return out, pos
