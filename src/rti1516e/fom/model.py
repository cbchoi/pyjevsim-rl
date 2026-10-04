"""FOM dataclasses aligned with the Go model in ``rti/pkg/fom/model``.

Strictly mirrors rti/pkg/fom/model/ — same field names, same hierarchy.
This is the cross-language shared mental model: a developer reading the Go
code and the Python code should see identical structure.

Mutability: dataclasses are frozen by default to enforce the post-parse
immutability invariant (FOMs are read-only after parsing).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TypeVar

from rti1516e.fom.modules import MIMModule

# --- DataType sum type ------------------------------------------------------
# Mirrors rti/pkg/fom/model/dataclass.go's DataType variants.


@dataclass(frozen=True)
class DataType:
    """Base class for the DataType sum. Concrete variants below."""

    name: str


@dataclass(frozen=True)
class BasicData(DataType):
    """Built-in primitive (HLAinteger32BE, HLAfloat64BE, etc.)."""

    size: int = 0  # Size in bytes; zero when the FOM does not specify it.
    endianness: str = ""  # "BE" | "LE" | "" for endian-agnostic types


@dataclass(frozen=True)
class SimpleData(DataType):
    """Aliased basic type (e.g. ``Speed`` aliasing ``HLAfloat64BE``)."""

    representation: str = ""


@dataclass(frozen=True)
class EnumeratedData(DataType):
    """Discrete values keyed by name."""

    representation: str = ""
    enumerators: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ArrayData(DataType):
    """Fixed or variable array. ``cardinality`` = -1 ⇒ variable."""

    element_type: str = ""
    cardinality: int = -1


@dataclass(frozen=True)
class FixedRecordData(DataType):
    """Ordered named fields."""

    fields: tuple[tuple[str, str], ...] = ()  # (field_name, type_name)


@dataclass(frozen=True)
class VariantRecordData(DataType):
    """Discriminated union."""

    discriminant_name: str = ""
    discriminant_type: str = ""
    variants: tuple[tuple[str, str, str], ...] = ()  # (enum_value, name, type)


# --- Class hierarchy --------------------------------------------------------


@dataclass(frozen=True)
class Attribute:
    """Attribute on an ObjectClass."""

    name: str
    data_type: str  # name resolved against FOM.data_types
    order: str = "TimeStamp"  # "TimeStamp" or "Receive"
    transportation: str = "HLAreliable"  # or "HLAbestEffort"


@dataclass(frozen=True)
class ObjectClass:
    """ObjectClass node in the inheritance tree."""

    name: str
    parent: str | None = None  # None for HLAobjectRoot
    attributes: tuple[Attribute, ...] = ()
    qualified_name: str = ""

    @property
    def key(self) -> str:
        return self.qualified_name or self.name

    @property
    def parent_key(self) -> str | None:
        return _parent_key(self.parent, self.qualified_name)


@dataclass(frozen=True)
class Parameter:
    """Parameter on an InteractionClass."""

    name: str
    data_type: str


@dataclass(frozen=True)
class InteractionClass:
    """InteractionClass node in the inheritance tree."""

    name: str
    parent: str | None = None  # None for HLAinteractionRoot
    parameters: tuple[Parameter, ...] = ()
    order: str = "TimeStamp"
    transportation: str = "HLAreliable"
    qualified_name: str = ""

    @property
    def key(self) -> str:
        return self.qualified_name or self.name

    @property
    def parent_key(self) -> str | None:
        return _parent_key(self.parent, self.qualified_name)


def _parent_key(parent: str | None, qualified_name: str) -> str | None:
    if qualified_name:
        return qualified_name.rpartition(".")[0] or None
    return parent


_Class = TypeVar("_Class", ObjectClass, InteractionClass)


def _find_class(classes: tuple[_Class, ...], name: str, root: str) -> _Class | None:
    for cls in classes:
        if cls.key == name:
            return cls
    matches = [cls for cls in classes if name in (cls.name, cls.key.removeprefix(root + "."))]
    return matches[0] if len(matches) == 1 else None


_DECIMAL_RATE = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)")


@dataclass(frozen=True)
class UpdateRate:
    """Exact positive maximum rate; None semantics means absent metadata."""

    name: str
    rate: Decimal
    semantics: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("update rate requires a name")
        if not isinstance(self.rate, Decimal):
            raise TypeError("update rate requires an exact Decimal")
        if not self.rate.is_finite():
            raise ValueError(f"update rate {self.name!r} is not finite")
        # IEEE 1516.2-2010 4.11.2 / 6.2.11 applies before runtime rounding.
        if self.rate <= 0:
            raise ValueError(f"update rate {self.name!r} must be greater than zero")
        runtime = float(self.rate)
        if not math.isfinite(runtime) or runtime == 0:
            raise ValueError(f"update rate {self.name!r} is outside the supported float64 range")

    @classmethod
    def from_decimal(cls, name: str, text: str, semantics: str | None = None) -> UpdateRate:
        """Decode xs:decimal without rounding through the decimal context."""
        name, text = name.strip(), text.strip()
        if not _DECIMAL_RATE.fullmatch(text):
            raise ValueError(f"update rate {name!r} requires an xs:decimal value")
        return cls(name=name, rate=Decimal(text), semantics=semantics)


# --- Root FOM ---------------------------------------------------------------


@dataclass(frozen=True)
class FOM:
    """Root node — the parsed FOM as an immutable value.

    Iteration over object_classes / interaction_classes / data_types is
    deterministic (sorted by name) per IEEE 1516.2 Annex A and the Go
    implementation's ordering convention.
    """

    object_classes: tuple[ObjectClass, ...] = ()
    interaction_classes: tuple[InteractionClass, ...] = ()
    data_types: tuple[DataType, ...] = ()
    mim_module: MIMModule | None = None
    update_rates: tuple[UpdateRate, ...] = ()

    def find_object_class(self, name: str) -> ObjectClass | None:
        """Resolve a full path, rootless path, or unambiguous local name."""
        return _find_class(self.object_classes, name, "HLAobjectRoot")

    def find_interaction_class(self, name: str) -> InteractionClass | None:
        """Resolve a full path, rootless path, or unambiguous local name."""
        return _find_class(self.interaction_classes, name, "HLAinteractionRoot")

    def find_attribute(self, class_name: str, name: str) -> Attribute | None:
        """Look up a declared or inherited attribute on the selected branch."""
        cls = self.find_object_class(class_name)
        by_key = {c.key: c for c in self.object_classes}
        visited: set[str] = set()
        while cls is not None and cls.key not in visited:
            visited.add(cls.key)
            for attribute in cls.attributes:
                if attribute.name == name:
                    return attribute
            cls = by_key.get(cls.parent_key) if cls.parent_key else None
        return None

    def find_parameter(self, class_name: str, name: str) -> Parameter | None:
        """Look up a declared or inherited parameter on the selected branch."""
        cls = self.find_interaction_class(class_name)
        by_key = {c.key: c for c in self.interaction_classes}
        visited: set[str] = set()
        while cls is not None and cls.key not in visited:
            visited.add(cls.key)
            for parameter in cls.parameters:
                if parameter.name == name:
                    return parameter
            cls = by_key.get(cls.parent_key) if cls.parent_key else None
        return None

    def find_data_type(self, name: str) -> DataType | None:
        """Lookup helper. Linear scan over self.data_types."""
        for dt in self.data_types:
            if dt.name == name:
                return dt
        return None
