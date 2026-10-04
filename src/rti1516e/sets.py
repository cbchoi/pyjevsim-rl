"""IEEE 1516 service-style typed handle collections.

Set subclasses coerce ``int`` to the typed handle on insert so mixed
callers work. Map subclasses coerce keys on ``__setitem__``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .handles import (
    AttributeHandle,
    DimensionHandle,
    FederateHandle,
    ParameterHandle,
    RegionHandle,
)


class AttributeHandleSet(set[AttributeHandle]):
    def __init__(self, iterable: Iterable[Any] = ()) -> None:
        super().__init__(
            h if isinstance(h, AttributeHandle) else AttributeHandle(int(h))
            for h in iterable
        )

    def add(self, h: Any) -> None:
        super().add(h if isinstance(h, AttributeHandle) else AttributeHandle(int(h)))

    def update(self, *iterables: Iterable[Any]) -> None:
        for it in iterables:
            for h in it:
                self.add(h)


class ParameterHandleSet(set[ParameterHandle]):
    def __init__(self, iterable: Iterable[Any] = ()) -> None:
        super().__init__(
            h if isinstance(h, ParameterHandle) else ParameterHandle(int(h))
            for h in iterable
        )

    def add(self, h: Any) -> None:
        super().add(h if isinstance(h, ParameterHandle) else ParameterHandle(int(h)))

    def update(self, *iterables: Iterable[Any]) -> None:
        for it in iterables:
            for h in it:
                self.add(h)


class FederateHandleSet(set[FederateHandle]):
    def __init__(self, iterable: Iterable[Any] = ()) -> None:
        super().__init__(
            h if isinstance(h, FederateHandle) else FederateHandle(int(h))
            for h in iterable
        )

    def add(self, h: Any) -> None:
        super().add(h if isinstance(h, FederateHandle) else FederateHandle(int(h)))

    def update(self, *iterables: Iterable[Any]) -> None:
        for it in iterables:
            for h in it:
                self.add(h)


class DimensionHandleSet(set[DimensionHandle]):
    def __init__(self, iterable: Iterable[Any] = ()) -> None:
        super().__init__(
            h if isinstance(h, DimensionHandle) else DimensionHandle(int(h))
            for h in iterable
        )

    def add(self, h: Any) -> None:
        super().add(h if isinstance(h, DimensionHandle) else DimensionHandle(int(h)))

    def update(self, *iterables: Iterable[Any]) -> None:
        for it in iterables:
            for h in it:
                self.add(h)


class RegionHandleSet(set[RegionHandle]):
    def __init__(self, iterable: Iterable[Any] = ()) -> None:
        super().__init__(
            h if isinstance(h, RegionHandle) else RegionHandle(int(h))
            for h in iterable
        )

    def add(self, h: Any) -> None:
        super().add(h if isinstance(h, RegionHandle) else RegionHandle(int(h)))

    def update(self, *iterables: Iterable[Any]) -> None:
        for it in iterables:
            for h in it:
                self.add(h)


class AttributeHandleValueMap(dict[AttributeHandle, bytes]):
    def __setitem__(self, k: Any, v: bytes) -> None:
        super().__setitem__(
            k if isinstance(k, AttributeHandle) else AttributeHandle(int(k)),
            v,
        )

    def __getitem__(self, k: Any) -> bytes:
        return super().__getitem__(
            k if isinstance(k, AttributeHandle) else AttributeHandle(int(k))
        )


class ParameterHandleValueMap(dict[ParameterHandle, bytes]):
    def __setitem__(self, k: Any, v: bytes) -> None:
        super().__setitem__(
            k if isinstance(k, ParameterHandle) else ParameterHandle(int(k)),
            v,
        )

    def __getitem__(self, k: Any) -> bytes:
        return super().__getitem__(
            k if isinstance(k, ParameterHandle) else ParameterHandle(int(k))
        )


class AttributeRegionMap(dict[AttributeHandle, RegionHandleSet]):
    """AttributeHandle -> RegionHandleSet."""

    def __setitem__(self, k: Any, v: Any) -> None:
        super().__setitem__(
            k if isinstance(k, AttributeHandle) else AttributeHandle(int(k)),
            v if isinstance(v, RegionHandleSet) else RegionHandleSet(v),
        )

    def __getitem__(self, k: Any) -> RegionHandleSet:
        return super().__getitem__(
            k if isinstance(k, AttributeHandle) else AttributeHandle(int(k))
        )
