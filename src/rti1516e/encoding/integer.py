"""HLAinteger16/32/64 big- and little-endian codecs.

Six concrete classes total. All MUST round-trip every integer vector in
tests/conformance/encoding_vectors.json byte-identical to the Go side.

OctetBoundary: 2 for 16-bit, 4 for 32-bit, 8 for 64-bit.
Encoding: two's complement, big- or little-endian per the type name.
"""

from __future__ import annotations

import struct
from typing import Any

from rti1516e.encoding._base import Codec, OctetBoundary
from rti1516e.errors import EncodingInsufficientBytes, EncodingTypeMismatch

# Module-level struct.Struct instances. Pre-compiling once is both faster
# (no format-string reparse per call) and clearer about width/endianness.
_S_I16_BE = struct.Struct(">h")
_S_I16_LE = struct.Struct("<h")
_S_I32_BE = struct.Struct(">i")
_S_I32_LE = struct.Struct("<i")
_S_I64_BE = struct.Struct(">q")
_S_I64_LE = struct.Struct("<q")


def _coerce_int(value: Any, low: int, high: int, type_name: str) -> int:
    """Normalize ``value`` to a Python int suitable for two's complement
    encoding into the given inclusive [low, high] range.

    Mirrors the Go side's ``toIntNN`` helpers (rti/pkg/encoding/integer.go).
    Accepts:

    - ``int`` (Python's unbounded signed int) — accepted directly. ``bool``
      is rejected even though it's an int subclass; HLA encoding has a
      distinct HLAboolean type and conflating them would be a category
      error.
    - ``float`` — accepted only when finite, in range, and exactly
      integral (no fractional part). This matches Go's float64 path,
      which is how JSON-loaded vector values arrive.

    Out-of-range values raise ``OverflowError``. Non-numeric or fractional
    inputs raise ``EncodingTypeMismatch``.
    """
    if isinstance(value, bool):
        raise EncodingTypeMismatch(f"{type_name}: bool is not a valid integer value")
    if isinstance(value, int):
        ivalue = value
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise EncodingTypeMismatch(
                f"{type_name}: float value {value!r} is not representable as an integer"
            )
        if not value.is_integer():
            raise EncodingTypeMismatch(
                f"{type_name}: float value {value!r} has a fractional part"
            )
        ivalue = int(value)
    else:
        raise EncodingTypeMismatch(
            f"{type_name}: expected int or integral float, got {type(value).__name__}"
        )
    if ivalue < low or ivalue > high:
        raise OverflowError(f"{type_name}: value {ivalue} out of range [{low}, {high}]")
    return ivalue


class HLAinteger16BE(Codec):
    """16-bit signed big-endian integer. OctetBoundary 2."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 2

    def encode(self, value: Any) -> bytes:
        ivalue = _coerce_int(value, -32768, 32767, "HLAinteger16BE")
        return _S_I16_BE.pack(ivalue)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 2:
            raise EncodingInsufficientBytes(
                f"HLAinteger16BE: need 2 bytes at offset {offset}, have {len(data) - offset}"
            )
        (value,) = _S_I16_BE.unpack_from(data, offset)
        return value, 2


class HLAinteger16LE(Codec):
    """16-bit signed little-endian integer. OctetBoundary 2."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 2

    def encode(self, value: Any) -> bytes:
        ivalue = _coerce_int(value, -32768, 32767, "HLAinteger16LE")
        return _S_I16_LE.pack(ivalue)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 2:
            raise EncodingInsufficientBytes(
                f"HLAinteger16LE: need 2 bytes at offset {offset}, have {len(data) - offset}"
            )
        (value,) = _S_I16_LE.unpack_from(data, offset)
        return value, 2


class HLAinteger32BE(Codec):
    """32-bit signed big-endian integer. OctetBoundary 4."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 4

    def encode(self, value: Any) -> bytes:
        ivalue = _coerce_int(value, -2147483648, 2147483647, "HLAinteger32BE")
        return _S_I32_BE.pack(ivalue)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 4:
            raise EncodingInsufficientBytes(
                f"HLAinteger32BE: need 4 bytes at offset {offset}, have {len(data) - offset}"
            )
        (value,) = _S_I32_BE.unpack_from(data, offset)
        return value, 4


class HLAinteger32LE(Codec):
    """32-bit signed little-endian integer. OctetBoundary 4."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 4

    def encode(self, value: Any) -> bytes:
        ivalue = _coerce_int(value, -2147483648, 2147483647, "HLAinteger32LE")
        return _S_I32_LE.pack(ivalue)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 4:
            raise EncodingInsufficientBytes(
                f"HLAinteger32LE: need 4 bytes at offset {offset}, have {len(data) - offset}"
            )
        (value,) = _S_I32_LE.unpack_from(data, offset)
        return value, 4


class HLAinteger64BE(Codec):
    """64-bit signed big-endian integer. OctetBoundary 8."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 8

    def encode(self, value: Any) -> bytes:
        ivalue = _coerce_int(
            value, -9223372036854775808, 9223372036854775807, "HLAinteger64BE"
        )
        return _S_I64_BE.pack(ivalue)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 8:
            raise EncodingInsufficientBytes(
                f"HLAinteger64BE: need 8 bytes at offset {offset}, have {len(data) - offset}"
            )
        (value,) = _S_I64_BE.unpack_from(data, offset)
        return value, 8


class HLAinteger64LE(Codec):
    """64-bit signed little-endian integer. OctetBoundary 8."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 8

    def encode(self, value: Any) -> bytes:
        ivalue = _coerce_int(
            value, -9223372036854775808, 9223372036854775807, "HLAinteger64LE"
        )
        return _S_I64_LE.pack(ivalue)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 8:
            raise EncodingInsufficientBytes(
                f"HLAinteger64LE: need 8 bytes at offset {offset}, have {len(data) - offset}"
            )
        (value,) = _S_I64_LE.unpack_from(data, offset)
        return value, 8
