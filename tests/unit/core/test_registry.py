"""Acceptance tests for evolux.core.registry."""

from __future__ import annotations

import pytest

from evolux.core.registry import Registry


def test_register_and_lookup() -> None:
    reg: Registry = Registry("widgets")

    @reg.register("foo")
    class Foo:
        pass

    assert "foo" in reg
    assert reg.get("foo") is Foo


def test_add_and_lookup() -> None:
    reg: Registry = Registry("widgets")
    reg.add("bar", int)
    assert "bar" in reg
    assert reg.get("bar") is int


def test_duplicate_key_raises_on_register() -> None:
    reg: Registry = Registry("widgets")

    @reg.register("dup")
    class First:
        pass

    with pytest.raises(KeyError):

        @reg.register("dup")
        class Second:
            pass


def test_duplicate_key_raises_on_add() -> None:
    reg: Registry = Registry("widgets")
    reg.add("x", int)
    with pytest.raises(KeyError):
        reg.add("x", str)


def test_missing_key_raises() -> None:
    reg: Registry = Registry("widgets")
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_keys_sorted() -> None:
    reg: Registry = Registry("widgets")
    reg.add("b", int)
    reg.add("a", str)
    assert reg.keys() == ["a", "b"]


def test_contains() -> None:
    reg: Registry = Registry("widgets")
    reg.add("k", float)
    assert "k" in reg
    assert "missing" not in reg


def test_repr() -> None:
    reg: Registry = Registry("test_repr")
    assert "test_repr" in repr(reg)
