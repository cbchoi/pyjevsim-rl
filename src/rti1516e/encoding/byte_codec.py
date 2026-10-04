"""HLAoctet, HLAoctetPairBE/LE, HLAboolean, HLAASCIIchar, HLAunicodeChar.

- HLAoctet: 1 byte, OctetBoundary 1
- HLAoctetPairBE/LE: 2 bytes, OctetBoundary 2
- HLAboolean: encoded as HLAinteger32BE (per IEEE 1516.2 §4.12), OctetBoundary 4
- HLAASCIIchar: 1 byte, OctetBoundary 1
- HLAunicodeChar: UTF-16BE 2 bytes, OctetBoundary 2

Byte-identical to rti/pkg/encoding/byte.go on the Go side. Decoded Python
values match the JSON-vector value shapes in tests/conformance/encoding_vectors.json:

  HLAoctet         -> int in [0, 255]
  HLAoctetPairBE/LE -> list[int] of length 2 (logical order; LE swaps on the wire)
  HLAboolean       -> bool
  HLAASCIIchar     -> str of length 1 (codepoint <= 0x7F)
  HLAunicodeChar   -> str of length 1 (BMP only; non-surrogate)
"""

from __future__ import annotations

from typing import Any

from rti1516e.encoding._base import Codec, OctetBoundary, aligned_offset
from rti1516e.errors import EncodingInsufficientBytes, EncodingTypeMismatch


def _coerce_octet(value: Any, type_name: str) -> int:
    """Coerce a Python value to an int in [0, 255]. Accepts int and bool
    (bool is a subclass of int in Python, mirrored from the Go float64
    JSON-decoding tolerance).

    Raises EncodingTypeMismatch on out-of-range or wrong-typed input.
    """
    if isinstance(value, bool):
        # Treat True/False as 1/0 — same tolerance as Go's int path.
        return int(value)
    if isinstance(value, int):
        if value < 0 or value > 0xFF:
            raise EncodingTypeMismatch(
                f"{type_name} value {value} out of range [0, 255]"
            )
        return value
    raise EncodingTypeMismatch(
        f"{type_name} cannot encode {type(value).__name__}"
    )


class HLAoctet(Codec):
    """Single octet (uint8). OctetBoundary 1."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 1

    def encode(self, value: Any) -> bytes:
        b = _coerce_octet(value, "HLAoctet")
        return bytes([b])

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        start = aligned_offset(offset, self.octet_boundary)
        if len(data) < start + 1:
            raise EncodingInsufficientBytes(
                f"HLAoctet: need 1 byte at offset {start}, have {max(0, len(data) - start)}"
            )
        return data[start], (start - offset) + 1


def _coerce_octet_pair(value: Any, type_name: str) -> tuple[int, int]:
    """Coerce a Python value to a (a, b) pair of ints in [0, 255]."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise EncodingTypeMismatch(
                f"{type_name} requires exactly 2 elements, got {len(value)}"
            )
        a = _coerce_octet(value[0], type_name)
        b = _coerce_octet(value[1], type_name)
        return a, b
    if isinstance(value, (bytes, bytearray)):
        if len(value) != 2:
            raise EncodingTypeMismatch(
                f"{type_name} requires exactly 2 bytes, got {len(value)}"
            )
        return value[0], value[1]
    raise EncodingTypeMismatch(
        f"{type_name} cannot encode {type(value).__name__}"
    )


class HLAoctetPairBE(Codec):
    """Pair of octets in big-endian order. OctetBoundary 2."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 2

    def encode(self, value: Any) -> bytes:
        a, b = _coerce_octet_pair(value, "HLAoctetPairBE")
        return bytes([a, b])

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        start = aligned_offset(offset, self.octet_boundary)
        if len(data) < start + 2:
            raise EncodingInsufficientBytes(
                f"HLAoctetPairBE: need 2 bytes at offset {start}, have {max(0, len(data) - start)}"
            )
        return [data[start], data[start + 1]], (start - offset) + 2


class HLAoctetPairLE(Codec):
    """Pair of octets in little-endian order. OctetBoundary 2.

    The pair is logically [a, b] but emitted on the wire as [b, a]
    (least-significant octet first). Decode reverses the swap so the
    returned Python value preserves the logical pair.
    """

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 2

    def encode(self, value: Any) -> bytes:
        a, b = _coerce_octet_pair(value, "HLAoctetPairLE")
        return bytes([b, a])

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        start = aligned_offset(offset, self.octet_boundary)
        if len(data) < start + 2:
            raise EncodingInsufficientBytes(
                f"HLAoctetPairLE: need 2 bytes at offset {start}, have {max(0, len(data) - start)}"
            )
        return [data[start + 1], data[start]], (start - offset) + 2


class HLAboolean(Codec):
    """Boolean encoded as HLAinteger32BE: 1 = true, 0 = false. OctetBoundary 4.

    Per IEEE 1516.2-2010 §4.12. Decode treats any non-zero 32-bit value
    as true (matches Go's HLAboolean.Decode).
    """

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 4

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, bool):
            raise EncodingTypeMismatch(
                f"HLAboolean cannot encode {type(value).__name__}"
            )
        if value:
            return b"\x00\x00\x00\x01"
        return b"\x00\x00\x00\x00"

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        start = aligned_offset(offset, self.octet_boundary)
        if len(data) < start + 4:
            raise EncodingInsufficientBytes(
                f"HLAboolean: need 4 bytes at offset {start}, have {max(0, len(data) - start)}"
            )
        word = (
            (data[start] << 24)
            | (data[start + 1] << 16)
            | (data[start + 2] << 8)
            | data[start + 3]
        )
        return word != 0, (start - offset) + 4


class HLAASCIIchar(Codec):
    """Single ASCII char (1 byte, code point 0..127). OctetBoundary 1."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 1

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, str):
            raise EncodingTypeMismatch(
                f"HLAASCIIchar cannot encode {type(value).__name__}"
            )
        if len(value) != 1:
            raise EncodingTypeMismatch(
                f"HLAASCIIchar requires single-character string, got len {len(value)}"
            )
        cp = ord(value)
        if cp > 0x7F:
            raise EncodingTypeMismatch(
                f"HLAASCIIchar value 0x{cp:02X} out of ASCII range"
            )
        return bytes([cp])

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        start = aligned_offset(offset, self.octet_boundary)
        if len(data) < start + 1:
            raise EncodingInsufficientBytes(
                f"HLAASCIIchar: need 1 byte at offset {start}, have {max(0, len(data) - start)}"
            )
        b = data[start]
        if b > 0x7F:
            raise EncodingTypeMismatch(
                f"HLAASCIIchar byte 0x{b:02X} out of ASCII range"
            )
        return chr(b), (start - offset) + 1


class HLAunicodeChar(Codec):
    """Single Unicode char as UTF-16BE (2 bytes, BMP only). OctetBoundary 2.

    Surrogate-pair handling for non-BMP code points is out of scope per
    IEEE 1516.2-2010 §4 — supplementary code points belong in
    HLAunicodeString.
    """

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 2

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, str):
            raise EncodingTypeMismatch(
                f"HLAunicodeChar cannot encode {type(value).__name__}"
            )
        if len(value) != 1:
            raise EncodingTypeMismatch(
                f"HLAunicodeChar requires single-character string, got len {len(value)}"
            )
        cp = ord(value)
        if cp > 0xFFFF:
            raise EncodingTypeMismatch(
                f"HLAunicodeChar code point U+{cp:04X} outside BMP "
                "(surrogate pairs out of scope)"
            )
        if 0xD800 <= cp <= 0xDFFF:
            raise EncodingTypeMismatch(
                f"HLAunicodeChar code point U+{cp:04X} is a UTF-16 surrogate"
            )
        return bytes([(cp >> 8) & 0xFF, cp & 0xFF])

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        start = aligned_offset(offset, self.octet_boundary)
        if len(data) < start + 2:
            raise EncodingInsufficientBytes(
                f"HLAunicodeChar: need 2 bytes at offset {start}, have {max(0, len(data) - start)}"
            )
        cp = (data[start] << 8) | data[start + 1]
        if 0xD800 <= cp <= 0xDFFF:
            raise EncodingTypeMismatch(
                f"HLAunicodeChar code unit U+{cp:04X} is a UTF-16 surrogate (BMP only)"
            )
        return chr(cp), (start - offset) + 2
