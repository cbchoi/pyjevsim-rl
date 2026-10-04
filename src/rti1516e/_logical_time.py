"""Selected logical-time representation and lossless wire conversion."""

import math
from typing import Any, Literal, overload

from .errors import CouldNotCreateLogicalTimeFactory, InvalidLogicalTime

FLOAT64_TIME = "HLAfloat64Time"
INTEGER64_TIME = "HLAinteger64Time"


def logical_time_name(name: str) -> str:
    """Empty response metadata is the legacy Float64 contract only."""
    if not isinstance(name, str):
        raise TypeError("logical_time_implementation_name must be a string")
    return name or FLOAT64_TIME


def require_supported_logical_time(name: str) -> str:
    selected = logical_time_name(name)
    if selected not in (FLOAT64_TIME, INTEGER64_TIME):
        raise CouldNotCreateLogicalTimeFactory(
            f"unsupported logical time implementation {name!r}; "
            "expected '', 'HLAfloat64Time', or 'HLAinteger64Time'"
        )
    return selected


def validate_time(value: Any, selected: str, *, allow_legacy_final: bool = False) -> int | float:
    selected = require_supported_logical_time(selected)
    if selected == INTEGER64_TIME:
        if type(value) is not int or not 0 <= value <= (1 << 63) - 1:
            raise InvalidLogicalTime(
                "Integer64 time requires a nonnegative signed int64, not a float"
            )
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidLogicalTime("Float64 time requires a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise InvalidLogicalTime("value exceeds Float64 time range") from exc
    if allow_legacy_final and type(value) is float and result == math.inf:
        return result
    if isinstance(value, int) and (not math.isfinite(result) or int(result) != value):
        raise InvalidLogicalTime("integer cannot be represented exactly by Float64 time")
    if not math.isfinite(result) or result < 0:
        raise InvalidLogicalTime("Float64 time requires a finite nonnegative number")
    return result


def write_time(message: Any, field: str, value: Any, selected: str) -> None:
    value = validate_time(value, selected, allow_legacy_final=field != "lookahead")
    exact_field = "logical_time_value" if field == "value" else field + "_value"
    descriptor = message.DESCRIPTOR.fields_by_name
    if selected == INTEGER64_TIME:
        if exact_field not in descriptor:
            raise InvalidLogicalTime("wire schema does not support exact Integer64 time")
        getattr(message, exact_field).integer64_value = value
        return
    setattr(message, field, value)
    if value == math.inf:
        return
    if exact_field in descriptor:
        getattr(message, exact_field).float64_value = value


@overload
def read_time(
    message: Any, field: str, selected: str, *, optional: Literal[False] = False
) -> int | float: ...


@overload
def read_time(
    message: Any, field: str, selected: str, *, optional: bool
) -> int | float | None: ...


def read_time(
    message: Any, field: str, selected: str, *, optional: bool = False
) -> int | float | None:
    selected = require_supported_logical_time(selected)
    descriptor = message.DESCRIPTOR.fields_by_name
    exact_field = "logical_time_value" if field == "value" else field + "_value"
    exact = exact_field in descriptor and message.HasField(exact_field)
    legacy_field = descriptor[field]
    legacy_present = (
        message.HasField(field)
        if legacy_field.has_presence
        else getattr(message, field) != 0 or math.copysign(1, getattr(message, field)) < 0
    )
    if exact:
        wire = getattr(message, exact_field)
        selector = wire.WhichOneof("value")
        expected = "integer64_value" if selected == INTEGER64_TIME else "float64_value"
        if selector != expected:
            raise InvalidLogicalTime("wire logical-time representation does not match federation")
        value = validate_time(getattr(wire, expected), selected)
        if legacy_present and (selected == INTEGER64_TIME or getattr(message, field) != value):
            raise InvalidLogicalTime("conflicting legacy and exact logical-time values")
        return value
    if optional and not legacy_present:
        return None
    if selected == INTEGER64_TIME:
        raise InvalidLogicalTime("Integer64 response is missing its exact wire value")
    return validate_time(getattr(message, field), selected, allow_legacy_final=field != "lookahead")
