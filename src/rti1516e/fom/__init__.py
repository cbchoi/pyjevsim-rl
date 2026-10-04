"""Python FOM model and parser compatible with ``rti/pkg/fom``.

The parser uses the same FOM-NNN diagnostic codes as the Go side; users
running Python and Go-side code see identical error messages for the
same malformed FOM. Spec test:
pysdk/tests/spec/m4/test_spec_m4_fom_diagnostics.py.

Public API:

    from rti1516e.fom import (
        # Model dataclasses
        FOM, ObjectClass, Attribute, InteractionClass, Parameter,
        DataType, BasicData, SimpleData, EnumeratedData,
        ArrayData, FixedRecordData, VariantRecordData,
        # Parser
        parse, ParseResult, Diagnostic,
    )
"""

from __future__ import annotations

from rti1516e.fom.model import (
    FOM,
    ArrayData,
    Attribute,
    BasicData,
    DataType,
    EnumeratedData,
    FixedRecordData,
    InteractionClass,
    ObjectClass,
    Parameter,
    SimpleData,
    VariantRecordData,
)
from rti1516e.fom.parser import Diagnostic, ParseResult, parse

__all__ = [
    "ArrayData",
    "Attribute",
    "BasicData",
    "DataType",
    "Diagnostic",
    "EnumeratedData",
    "FOM",
    "FixedRecordData",
    "InteractionClass",
    "ObjectClass",
    "Parameter",
    "ParseResult",
    "SimpleData",
    "VariantRecordData",
    "parse",
]
