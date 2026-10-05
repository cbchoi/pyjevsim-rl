"""Small declarations for exclusively owned value fields, not a serializer.

Callers must establish state completeness and exclusive ownership. Shared roots,
RNG objects, clocks, scheduler state, topology and reference rebinding remain
explicit provider responsibilities. Validators are trusted, source-bound code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .contracts import canonical_bytes, exact_fields, fail, freeze_value, thaw_value


@dataclass(frozen=True, slots=True)
class OwnedValueField:
    name: str
    validate: Callable[[object], None]

    def __post_init__(self):
        if (type(self.name) is not str or not self.name.isidentifier()
                or self.name.startswith("_")):
            fail("owned field requires a public direct attribute name")
        if not callable(self.validate):
            fail("owned field requires an explicit value validator")


@dataclass(frozen=True, slots=True)
class DeclaredValueFields:
    fields: tuple[OwnedValueField, ...]

    def __post_init__(self):
        if (type(self.fields) is not tuple or not self.fields
                or any(type(field) is not OwnedValueField for field in self.fields)):
            fail("owned fields require a nonempty tuple of declarations")
        if len({field.name for field in self.fields}) != len(self.fields):
            fail("owned field names must be unique")

    def _attributes(self, owner):
        try:
            attributes = object.__getattribute__(owner, "__dict__")
        except AttributeError:
            fail("owned field target requires a direct attribute dictionary")
        if type(attributes) is not dict or any(field.name not in attributes for field in self.fields):
            fail("declared owned field is absent from target attributes")
        return attributes

    def _validated(self, payload):
        exact_fields(payload, {field.name for field in self.fields}, "owned value fields")
        # Validators see a detached closed value tree, never source or target.
        original = freeze_value(payload)
        candidate = thaw_value(original)
        for field in self.fields:
            if field.validate(candidate[field.name]) is not None:
                fail("owned field validator must return None on success")
        if canonical_bytes(candidate) != canonical_bytes(original):
            fail("owned field validator mutated its input")
        return candidate

    def capture(self, owner) -> dict:
        attributes = self._attributes(owner)
        return self._validated({field.name: attributes[field.name] for field in self.fields})

    def validate(self, payload) -> None:
        self._validated(payload)

    def restore_into(self, owner, payload) -> None:
        """Replace declared exclusive values only after whole-record validation.

        No property/setattr hooks, transitions or constructors are invoked.
        Container identity is deliberately not retained: aliases are not values.
        """
        attributes = self._attributes(owner)
        candidate = self._validated(payload)
        attributes.update(candidate)
