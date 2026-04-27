"""Core RNG / config / device / registry tests."""

from __future__ import annotations

import pytest
import torch

from evolux.core import RNG, BaseConfig, ConfigError, Registry, get_device


def test_rng_deterministic() -> None:
    a = RNG(42).split("brain")
    b = RNG(42).split("brain")
    assert torch.rand(4, generator=a).tolist() == torch.rand(4, generator=b).tolist()


def test_rng_different_names_independent() -> None:
    root = RNG(42)
    g1 = root.split("alpha")
    g2 = root.split("beta")
    s1 = torch.rand(4, generator=g1)
    s2 = torch.rand(4, generator=g2)
    assert not torch.allclose(s1, s2)


def test_get_device_auto() -> None:
    d = get_device("auto")
    assert d.type in {"cpu", "cuda", "mps"}


def test_registry_basic() -> None:
    reg: Registry = Registry("test")

    @reg.register("foo")
    class Foo:
        pass

    assert "foo" in reg
    assert reg.get("foo") is Foo


def test_registry_duplicate_raises() -> None:
    reg: Registry = Registry("test")
    reg.add("x", int)
    with pytest.raises(KeyError):
        reg.add("x", str)


def test_base_config_frozen_and_strict() -> None:
    class C(BaseConfig):
        x: int = 1

    c = C()
    with pytest.raises(Exception):
        c.x = 2  # type: ignore[misc]
    with pytest.raises(Exception):
        C(unknown_field=5)  # type: ignore[call-arg]


def test_load_config_round_trip(tmp_path) -> None:
    from evolux.core.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("simulation:\n  seed: 7\n  batch_size: 4\n")
    cfg = load_config(p)
    assert cfg.simulation.seed == 7
    assert cfg.simulation.batch_size == 4


def test_config_error_on_invalid(tmp_path) -> None:
    from evolux.core.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("simulation:\n  bogus_key: 1\n")
    with pytest.raises((ConfigError, Exception)):
        load_config(p)
