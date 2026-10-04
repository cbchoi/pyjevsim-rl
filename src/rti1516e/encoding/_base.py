"""Codec ABC and shared IEEE 1516.2 encoding primitives.

The public contract mirrors the Go ``encoding.Codec`` interface in
``rti/pkg/encoding/codec.go``. Encoders must produce byte-identical output
across both implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# OctetBoundary is the alignment, in bytes, that a value of this codec's
# type must be padded to before encoding within a composite. IEEE 1516.2
# Annex A "OctetBoundary" semantics. Mirrors Go's encoding.Codec.OctetBoundary.
OctetBoundary = int


class Codec(ABC):
    """Encoder/decoder for one HLA encoding type.

    Implementations live under rti1516e.encoding.* (one module per type
    family). Each class is instantiated once via dispatch.codec_for(type)
    and reused — implementations MUST be stateless after construction.

    Determinism: encode(value) MUST be a pure function of value. Two calls
    with the same input MUST produce identical bytes. No reliance on dict
    iteration order, no time-of-day, no global state.
    """

    @property
    @abstractmethod
    def octet_boundary(self) -> OctetBoundary:
        """The alignment requirement for values of this codec's type.

        Composite codecs use this when laying out fields/elements to insert
        the right number of padding bytes per IEEE 1516.2 §4.
        """

    @abstractmethod
    def encode(self, value: Any) -> bytes:
        """Encode ``value`` to wire bytes per the IEEE 1516.2 rules.

        MUST be byte-identical to the Go side's encode for the same value.
        """

    @abstractmethod
    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        """Decode wire bytes to ``(value, bytes_consumed)``.

        ``offset`` is the starting byte index. Returns a tuple of the
        decoded value and the number of bytes consumed (so callers can
        advance their cursor in composite types).

        Raises ``EncodingInsufficientBytes`` when ``data[offset:]`` is too
        short for the type. Raises ``EncodingPaddingViolation`` when
        required padding bytes are non-zero.
        """


def pad_to_boundary(buf: bytearray, boundary: OctetBoundary) -> int:
    """Append zero padding bytes to ``buf`` so its length is a multiple of
    ``boundary``. Returns the number of padding bytes appended.

    Used by composite codecs (fixed_array, fixed_record, variant_record,
    variable_array) to align nested values per IEEE 1516.2 §4.

    The function signature is shared by all composite codec implementations.
    """
    if boundary <= 1:
        return 0
    remainder = len(buf) % boundary
    if remainder == 0:
        return 0
    pad = boundary - remainder
    buf.extend(b"\x00" * pad)
    return pad


def aligned_offset(offset: int, boundary: OctetBoundary) -> int:
    """Return the smallest offset >= ``offset`` that's a multiple of
    ``boundary``. Used by decode paths to skip padding bytes.
    """
    if boundary <= 1:
        return offset
    remainder = offset % boundary
    if remainder == 0:
        return offset
    return offset + (boundary - remainder)
