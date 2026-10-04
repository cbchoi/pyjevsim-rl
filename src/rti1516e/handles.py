"""IEEE 1516 service-style typed handles.

Wrappers subclass ``int`` so existing gorti code that compares handles
to ints, hashes them in dicts, or passes them to ``int(...)`` keeps
working. The wrapper class disambiguates ``ObjectClassHandle`` from
``AttributeHandle`` at the type level via ``isinstance()`` — which is
the only distinction porting users from reference_rti need.

NOTE: Two handles of different types but the same underlying value
will compare equal (``ObjectClassHandle(5) == AttributeHandle(5)``
returns True). This is a deliberate dual-accept concession: it lets
mixed-typed and bare-int callers interoperate. Code that needs strict
type-distinction must use ``isinstance()``.
"""

from __future__ import annotations


class _StrongHandle(int):
    __slots__ = ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({int(self)})"


class ObjectClassHandle(_StrongHandle):
    __slots__ = ()


class AttributeHandle(_StrongHandle):
    __slots__ = ()


class InteractionClassHandle(_StrongHandle):
    __slots__ = ()


class ParameterHandle(_StrongHandle):
    __slots__ = ()


class ObjectInstanceHandle(_StrongHandle):
    __slots__ = ()


class FederateHandle(_StrongHandle):
    __slots__ = ()


class DimensionHandle(_StrongHandle):
    __slots__ = ()


class RegionHandle(_StrongHandle):
    __slots__ = ()


class MessageRetractionHandle(_StrongHandle):
    __slots__ = ()
