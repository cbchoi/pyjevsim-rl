"""HLAopaqueData length-prefixed byte-blob codec.

Wire format: 4-byte HLAinteger32BE length, then ``length`` raw bytes.
No element typing. OctetBoundary 4.
"""

from __future__ import annotations

import struct
from typing import Any

from rti1516e.encoding._base import Codec, OctetBoundary
from rti1516e.errors import EncodingInsufficientBytes


class HLAopaqueData(Codec):
    """Variable-length byte blob with a 4-byte length prefix.

    Matches ``rti/pkg/encoding/opaque.go``:
      - Encode accepts ``bytes`` / ``bytearray`` / ``memoryview`` (None
        treated as empty), OR a hex-encoded ``str`` (the JSON
        conformance-vector convention; ``"deadbeef"`` -> 4 bytes).
      - Decode returns the payload as a freshly-allocated ``bytes``
        instance plus the number of bytes consumed (``4 + length``).
    """

    @property
    def octet_boundary(self) -> OctetBoundary:
        return 4

    def encode(self, value: Any) -> bytes:
        payload = self._coerce_payload(value)
        return struct.pack(">I", len(payload)) + payload

    def decode(self, data: bytes, offset: int = 0) -> tuple[bytes, int]:
        # Length prefix must fit.
        if len(data) - offset < 4:
            raise EncodingInsufficientBytes(
                "HLAopaqueData decode short buffer (length prefix)"
            )
        length = struct.unpack_from(">I", data, offset)[0]
        # Payload must fit.
        if len(data) - offset - 4 < length:
            raise EncodingInsufficientBytes(
                "HLAopaqueData decode short buffer (payload)"
            )
        start = offset + 4
        end = start + length
        # Copy out so callers can drop the source buffer without aliasing.
        return bytes(data[start:end]), 4 + length

    @staticmethod
    def _coerce_payload(value: Any) -> bytes:
        """Normalize ``value`` to a ``bytes`` payload.

        Accepts None (empty), bytes-like, or a hex-encoded string. Raises
        ``TypeError`` for anything else (mirrors Go's typed error path).
        """
        if value is None:
            return b""
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, str):
            if value == "":
                return b""
            try:
                return bytes.fromhex(value)
            except ValueError as exc:
                raise TypeError(
                    f"HLAopaqueData cannot decode string as hex: {exc}"
                ) from exc
        raise TypeError(
            f"HLAopaqueData cannot encode {type(value).__name__} "
            "(want bytes-like or hex string)"
        )
