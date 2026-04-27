"""Component registry — lightweight service locator for simulation components."""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    """Simple key-value store for named simulation components.

    Avoids passing a growing list of objects through every constructor; 
    components can look up their peers by name instead.
    """

    def __init__(self) -> None:
        self._components: dict[str, Any] = {}

    def register(self, name: str, component: Any) -> None:
        if name in self._components:
            raise KeyError(f"Component '{name}' is already registered.")
        self._components[name] = component

    def get(self, name: str, expected_type: type[T] | None = None) -> T:
        if name not in self._components:
            raise KeyError(f"Component '{name}' not found in registry.")
        comp = self._components[name]
        if expected_type is not None and not isinstance(comp, expected_type):
            raise TypeError(
                f"Component '{name}' is {type(comp).__name__}, expected {expected_type.__name__}."
            )
        return comp  # type: ignore[return-value]

    def has(self, name: str) -> bool:
        return name in self._components

    def __repr__(self) -> str:
        return f"ComponentRegistry({list(self._components.keys())})"
