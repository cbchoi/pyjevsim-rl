"""IEEE 1516.2-2010 encoding rules compatible with the Go encoder.

Public API:

    from rti1516e.encoding import Codec, codec_for
    from rti1516e.encoding.dispatch import parse_type_spec

Per-codec modules expose the concrete classes (one per HLA encoding type).
The full byte-identical contract against Go is tested by
pysdk/tests/spec/m4/test_spec_m4_encoding_conformance.py iterating
tests/conformance/encoding_vectors.json.

The names exported here form the stable public encoding API. Additional
exports may be added without renaming existing entries.
"""

from __future__ import annotations

from rti1516e.encoding._base import Codec, OctetBoundary
from rti1516e.encoding.dispatch import codec_for, parse_type_spec

__all__ = [
    "Codec",
    "OctetBoundary",
    "codec_for",
    "parse_type_spec",
]
