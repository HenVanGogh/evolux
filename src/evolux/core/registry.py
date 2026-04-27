"""Generic registry — used for plug-in lookup of brains, genomes, evolution
strategies, etc.

Modules register their concrete implementations under a string key, and
configs reference them by name. This is what makes the framework
config-driven and extensible without touching the orchestrator.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Typed string→class registry."""

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._items: dict[str, type[T]] = {}

    def register(self, key: str) -> RegisterDecorator[T]:
        """Decorator: ``@registry.register("foo")``."""

        def deco(cls: type[T]) -> type[T]:
            if key in self._items:
                raise KeyError(f"{self._name}: '{key}' already registered.")
            self._items[key] = cls
            return cls

        return deco  # type: ignore[return-value]

    def add(self, key: str, cls: type[T]) -> None:
        if key in self._items:
            raise KeyError(f"{self._name}: '{key}' already registered.")
        self._items[key] = cls

    def get(self, key: str) -> type[T]:
        if key not in self._items:
            raise KeyError(f"{self._name}: unknown key '{key}'. Available: {sorted(self._items)}")
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __repr__(self) -> str:
        return f"Registry({self._name}, items={list(self._items)})"


# A type alias for the common pattern; concrete typed registries are created
# inside each module (e.g. evolux.brain.BRAIN_REGISTRY).
RegisterDecorator = type
