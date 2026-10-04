"""HLAfloat32/64 big- and little-endian codecs (IEEE 754).

Module is named float_codec to avoid shadowing the float built-in.
OctetBoundary: 4 for 32-bit, 8 for 64-bit.

Wire format mirrors ``rti/pkg/encoding/float.go``: ``struct`` format codes
``>f`` / ``<f`` for single-precision and ``>d`` / ``<d`` for double-precision
yield byte-identical output to the Go side's ``binary.{Big,Little}Endian``
plus ``math.Float{32,64}bits``.

Type-acceptance policy mirrors the Go side: ``encode`` accepts any value
``float`` or ``int`` accepts (Python's ``float()`` widens ``int`` and
``bool``). For HLAfloat32* a Python ``float`` (double precision) is
narrowed via ``struct.pack(">f", ...)`` — the caller is responsible for
ensuring the value is exactly representable in single precision when
round-trip equality matters.

Decode always returns a Python ``float`` (which is double-precision), so
HLAfloat32* decode widens the single-precision wire value to double — same
shape as Go's ``float64(math.Float32frombits(...))``.
"""

from __future__ import annotations

import struct
from typing import Any

from rti1516e.encoding._base import Codec, OctetBoundary
from rti1516e.errors import EncodingInsufficientBytes

__all__ = [
    "HLAfloat32BE",
    "HLAfloat32LE",
    "HLAfloat64BE",
    "HLAfloat64LE",
]


def _coerce_float(value: Any, type_name: str) -> float:
    """Accept a Python ``float`` / ``int`` / ``bool`` (the latter two via
    ``float()`` widening) and return a ``float``. Raises ``TypeError``
    for any other input — mirrors the Go side's ``asFloat32/64`` ok-check.
    """
    # bool is a subclass of int; ``float(True) == 1.0`` so it's accepted.
    if isinstance(value, (float, int)):
        return float(value)
    raise TypeError(f"encoding: {type_name}.encode: want float or int, got {type(value).__name__}")


class HLAfloat32BE(Codec):
    """IEEE-754 single-precision big-endian. OctetBoundary 4."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 4

    def encode(self, value: Any) -> bytes:
        f = _coerce_float(value, "HLAfloat32BE")
        return struct.pack(">f", f)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 4:
            raise EncodingInsufficientBytes(
                f"HLAfloat32BE.decode: need 4 bytes at offset {offset}, "
                f"have {max(0, len(data) - offset)}"
            )
        (value,) = struct.unpack(">f", data[offset : offset + 4])
        return float(value), 4


class HLAfloat32LE(Codec):
    """IEEE-754 single-precision little-endian. OctetBoundary 4."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 4

    def encode(self, value: Any) -> bytes:
        f = _coerce_float(value, "HLAfloat32LE")
        return struct.pack("<f", f)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 4:
            raise EncodingInsufficientBytes(
                f"HLAfloat32LE.decode: need 4 bytes at offset {offset}, "
                f"have {max(0, len(data) - offset)}"
            )
        (value,) = struct.unpack("<f", data[offset : offset + 4])
        return float(value), 4


class HLAfloat64BE(Codec):
    """IEEE-754 double-precision big-endian. OctetBoundary 8."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 8

    def encode(self, value: Any) -> bytes:
        f = _coerce_float(value, "HLAfloat64BE")
        return struct.pack(">d", f)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 8:
            raise EncodingInsufficientBytes(
                f"HLAfloat64BE.decode: need 8 bytes at offset {offset}, "
                f"have {max(0, len(data) - offset)}"
            )
        (value,) = struct.unpack(">d", data[offset : offset + 8])
        return float(value), 8


class HLAfloat64LE(Codec):
    """IEEE-754 double-precision little-endian. OctetBoundary 8."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 8

    def encode(self, value: Any) -> bytes:
        f = _coerce_float(value, "HLAfloat64LE")
        return struct.pack("<d", f)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        if len(data) - offset < 8:
            raise EncodingInsufficientBytes(
                f"HLAfloat64LE.decode: need 8 bytes at offset {offset}, "
                f"have {max(0, len(data) - offset)}"
            )
        (value,) = struct.unpack("<d", data[offset : offset + 8])
        return float(value), 8
