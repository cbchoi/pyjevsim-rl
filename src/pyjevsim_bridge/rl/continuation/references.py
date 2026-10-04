"""Candidate-local semantic references; no dynamic import or graph discovery."""
from __future__ import annotations

from .contracts import ContinuationError, freeze_value, thaw_value


class ReferenceRegistry:
    def __init__(self, topology: dict, objects: dict | None = None) -> None:
        self._topology = freeze_value(topology)
        self._objects: dict[str, object] = {}
        for name, value in (objects or {}).items():
            self.bind(name, value)

    @property
    def topology(self) -> dict:
        return thaw_value(self._topology)

    def bind(self, name: str, value: object) -> None:
        allowed = set(self._topology['nodes']) | set(self._topology['shared_resources'])
        if name not in allowed or value is None:
            raise ContinuationError('CC_INVALID_PAYLOAD', 'undeclared or empty semantic reference')
        if name in self._objects:
            if self._objects[name] is value:
                return
            raise ContinuationError('CC_CONFORMANCE_FAILED', 'semantic reference replaced')
        if any(item is value for item in self._objects.values()):
            raise ContinuationError('CC_CONFORMANCE_FAILED', 'object has two semantic node identities; use aliases')
        self._objects[name] = value

    def get(self, name: str) -> object:
        try:
            return self._objects[name]
        except KeyError as exc:
            raise ContinuationError('CC_CONFORMANCE_FAILED', f'unbound semantic reference: {name}') from exc

    def items(self):
        return tuple(self._objects.items())

    def identity_snapshot(self) -> dict:
        expected = set(self._topology['nodes']) | set(self._topology['shared_resources'])
        if set(self._objects) != expected:
            raise ContinuationError('CC_CONFORMANCE_FAILED', 'semantic registry is incomplete')
        return dict(self._objects)

    def assert_stable(self, before: dict) -> None:
        current = self.identity_snapshot()
        if set(current) != set(before) or any(current[key] is not before[key] for key in before):
            raise ContinuationError('CC_CONFORMANCE_FAILED', 'semantic object identity changed while loading')
