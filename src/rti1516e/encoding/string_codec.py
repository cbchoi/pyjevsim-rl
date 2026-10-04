"""HLAASCIIstring and HLAunicodeString length-prefixed codecs.

Both codecs follow IEEE 1516.2-2010 §4.

Both share the same envelope shape:

  HLAASCIIstring   4-byte HLAinteger32BE length prefix (count of code units
                   == bytes) + N bytes of ASCII payload (each <= 0x7F).
                   OctetBoundary == 4.

  HLAunicodeString 4-byte HLAinteger32BE length prefix (count of UTF-16
                   code units, NOT bytes) + 2*N bytes of UTF-16BE payload.
                   OctetBoundary == 4. Surrogate-pair (non-BMP) input is
                   tolerated by Python's ``utf-16-be`` codec, which emits a
                   surrogate pair (two code units) for one supplementary
                   code point — the count prefix counts code units, so this
                   stays consistent with the Go side as long as the prefix
                   is computed from the byte length.

The 4-byte length prefix is itself 4-byte aligned, which is why the
OctetBoundary of both types is 4 even though the per-character widths
are 1 and 2 respectively. Mirrors rti/pkg/encoding/string.go on the Go
side; output MUST be byte-identical.
"""

from __future__ import annotations

import struct
from typing import Any

from rti1516e.encoding._base import Codec, OctetBoundary, aligned_offset
from rti1516e.errors import EncodingInsufficientBytes, EncodingTypeMismatch

# Pre-compiled struct for the shared 4-byte BE length prefix.
_S_U32_BE = struct.Struct(">I")


class HLAASCIIstring(Codec):
    """ASCII-encoded string with HLAinteger32BE length prefix. OctetBoundary 4."""

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 4

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, str):
            raise EncodingTypeMismatch(
                f"HLAASCIIstring cannot encode {type(value).__name__}"
            )
        for i, ch in enumerate(value):
            cp = ord(ch)
            if cp > 0x7F:
                raise EncodingTypeMismatch(
                    f"HLAASCIIstring code point U+{cp:04X} at index {i} out of ASCII range"
                )
        payload = value.encode("ascii")
        return _S_U32_BE.pack(len(payload)) + payload

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        start = aligned_offset(offset, self.octet_boundary)
        if len(data) < start + 4:
            raise EncodingInsufficientBytes(
                f"HLAASCIIstring: need 4 bytes for length prefix at offset "
                f"{start}, have {max(0, len(data) - start)}"
            )
        (length,) = _S_U32_BE.unpack_from(data, start)
        payload_start = start + 4
        if len(data) < payload_start + length:
            raise EncodingInsufficientBytes(
                f"HLAASCIIstring: need {length} payload bytes at offset "
                f"{payload_start}, have {max(0, len(data) - payload_start)}"
            )
        payload = data[payload_start : payload_start + length]
        for i, b in enumerate(payload):
            if b > 0x7F:
                raise EncodingTypeMismatch(
                    f"HLAASCIIstring byte 0x{b:02X} at offset {i} out of ASCII range"
                )
        return payload.decode("ascii"), (start - offset) + 4 + length


class HLAunicodeString(Codec):
    """UTF-16BE-encoded string with HLAinteger32BE length prefix.

    Length is the number of UTF-16 code units (NOT bytes); the byte length
    on the wire is 4 + 2*length. OctetBoundary 4.
    """

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 4

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, str):
            raise EncodingTypeMismatch(
                f"HLAunicodeString cannot encode {type(value).__name__}"
            )
        # Python's "utf-16-be" codec emits no BOM (unlike "utf-16"), so we
        # can use it directly. Surrogate pairs for non-BMP code points come
        # for free; the count prefix is bytes // 2 so it counts code units.
        payload = value.encode("utf-16-be")
        count = len(payload) // 2
        return _S_U32_BE.pack(count) + payload

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        start = aligned_offset(offset, self.octet_boundary)
        if len(data) < start + 4:
            raise EncodingInsufficientBytes(
                f"HLAunicodeString: need 4 bytes for length prefix at offset "
                f"{start}, have {max(0, len(data) - start)}"
            )
        (count,) = _S_U32_BE.unpack_from(data, start)
        payload_start = start + 4
        payload_len = count * 2
        if len(data) < payload_start + payload_len:
            raise EncodingInsufficientBytes(
                f"HLAunicodeString: need {payload_len} payload bytes at offset "
                f"{payload_start}, have {max(0, len(data) - payload_start)}"
            )
        payload = data[payload_start : payload_start + payload_len]
        return payload.decode("utf-16-be"), (start - offset) + 4 + payload_len
