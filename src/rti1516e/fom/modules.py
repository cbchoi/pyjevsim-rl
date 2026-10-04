"""Explicit MIM input, kept separate from the user FOM module sequence."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MIMModule:
    """A supplied MIM designator and exact XML bytes, not an ordinary FOM.

    The server validates its supported MIM profile. Both the designator and
    XML must be supplied; omission of the whole module selects the default.
    """

    designator: str
    xml: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.designator, str):
            raise TypeError("MIM designator must be a string")
        if not self.designator.strip():
            raise ValueError("explicit MIM designator must not be empty")
        if not isinstance(self.xml, bytes):
            raise TypeError("MIM XML must be bytes")
        if not self.xml:
            raise ValueError("explicit MIM XML must not be empty")


MIMInput = str | Path | MIMModule


def validate_mim_input(value: MIMInput | None) -> None:
    if value is not None and not isinstance(value, (str, Path, MIMModule)):
        raise TypeError("mim_module must be a path or MIMModule")
    if isinstance(value, str) and not value.strip():
        raise ValueError("mim_module path must not be empty; omit it with None")


def load_mim_module(value: MIMInput | None) -> MIMModule | None:
    validate_mim_input(value)
    if value is None or isinstance(value, MIMModule):
        return value
    return MIMModule(str(value), Path(value).read_bytes())


def module_paths(values: Iterable[str | Path], name: str) -> list[str | Path]:
    if isinstance(values, (str, bytes, Path)):
        raise TypeError(f"{name} must be a sequence of module paths, not a single path")
    try:
        paths = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of module paths") from exc
    if any(not isinstance(path, (str, Path)) for path in paths):
        raise TypeError(f"{name} entries must be module paths")
    if any(isinstance(path, str) and not path for path in paths):
        raise ValueError(f"{name} entries must not be empty")
    return paths
