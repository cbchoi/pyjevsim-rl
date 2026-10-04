"""HLAvariantRecord discriminated-union codec.

Construction: HLAvariantRecord(
    discriminant=("name", discriminant_codec),
    variants={discriminant_value: ("variant_name", variant_codec)},
).

Wire format: discriminant encoded first, padded to its boundary, then
the active variant. OctetBoundary = max(discriminant.octet_boundary,
max(v.octet_boundary for v in variants.values())).

Python value mapping: dict keyed by the discriminant_name and the
active variant's variant_name (e.g. {"discriminator": 1, "value": ...}
to mirror rti/pkg/encoding/variant_record.go's literal "discriminator"
/ "value" keys when constructed with those names).
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


class HLAvariantRecord(Codec):
    """Discriminated union of variants keyed by discriminant value.

    Wire layout per IEEE 1516.2-2010 §4: the discriminant is encoded at
    offset 0, followed by zero-or-more padding octets so the active
    variant begins at an offset that is a multiple of the variant's
    OctetBoundary, then the variant value. The record's overall
    OctetBoundary = max(discriminant boundary, max variant boundary).
    """

    def __init__(
        self,
        discriminant: tuple[str, Codec],
        variants: dict[Any, tuple[str, Codec]],
    ) -> None:
        self.discriminant_name, self.discriminant_codec = discriminant
        # Defensive copy — caller mutation after construction would break
        # determinism (and could swap a variant codec under us).
        self.variants: dict[Any, tuple[str, Codec]] = dict(variants)

    @property
    def octet_boundary(self) -> OctetBoundary:
        boundaries = [self.discriminant_codec.octet_boundary]
        boundaries.extend(c.octet_boundary for _, c in self.variants.values())
        return max(boundaries) if boundaries else 1

    def _canonical_discriminant(self, raw: Any) -> Any:
        """Round-trip the raw discriminant value through the discriminant
        codec to yield its canonical form for variant-map lookup.

        Mirrors the Go side (rti/pkg/encoding/variant_record.go): JSON-
        loaded vectors deliver numeric discriminators as Python ``float``
        even when the codec's canonical type is ``int``; the variants map
        is keyed on the canonical form, so we normalize before lookup.
        """
        bytes_ = self.discriminant_codec.encode(raw)
        canonical, _ = self.discriminant_codec.decode(bytes_)
        return canonical

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, dict):
            raise EncodingTypeMismatch(
                f"HLAvariantRecord cannot encode {type(value).__name__}; "
                "expected dict"
            )
        if self.discriminant_name not in value:
            raise EncodingTypeMismatch(
                f"HLAvariantRecord missing discriminant key "
                f"{self.discriminant_name!r}"
            )
        raw_disc = value[self.discriminant_name]
        canonical = self._canonical_discriminant(raw_disc)
        if canonical not in self.variants:
            raise EncodingTypeMismatch(
                f"HLAvariantRecord no alternative for discriminant value "
                f"{raw_disc!r} (canonical: {canonical!r})"
            )
        variant_name, variant_codec = self.variants[canonical]
        if variant_name not in value:
            raise EncodingTypeMismatch(
                f"HLAvariantRecord missing variant key {variant_name!r} for "
                f"discriminant {canonical!r}"
            )

        out = bytearray(self.discriminant_codec.encode(raw_disc))
        pad_to_boundary(out, variant_codec.octet_boundary)
        out.extend(variant_codec.encode(value[variant_name]))
        return bytes(out)

    def decode(self, data: bytes, offset: int = 0) -> tuple[Any, int]:
        # Discriminant decodes at the record's starting offset (already
        # aligned by the parent composite, or zero on top-level entry).
        disc_aligned = aligned_offset(offset, self.discriminant_codec.octet_boundary)
        if disc_aligned > len(data):
            raise EncodingInsufficientBytes(
                "HLAvariantRecord: padding before discriminant requires "
                f"offset {disc_aligned}, have {len(data)}"
            )
        disc_value, disc_consumed = self.discriminant_codec.decode(data, disc_aligned)
        if disc_value not in self.variants:
            raise EncodingTypeMismatch(
                f"HLAvariantRecord no alternative for discriminant value "
                f"{disc_value!r}"
            )
        variant_name, variant_codec = self.variants[disc_value]
        variant_pos = aligned_offset(
            disc_aligned + disc_consumed, variant_codec.octet_boundary
        )
        if variant_pos > len(data):
            raise EncodingInsufficientBytes(
                "HLAvariantRecord: padding to variant boundary requires "
                f"offset {variant_pos}, have {len(data)}"
            )
        variant_value, variant_consumed = variant_codec.decode(data, variant_pos)
        out: dict[str, Any] = {
            self.discriminant_name: disc_value,
            variant_name: variant_value,
        }
        return out, (variant_pos + variant_consumed) - offset
