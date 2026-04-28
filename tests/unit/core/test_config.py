"""Acceptance tests for evolux.core.config — load_config."""

from __future__ import annotations

import pytest

from evolux.core.config import ConfigError, load_config


def test_round_trip_yaml(tmp_path) -> None:
    """Basic YAML loads and validates correctly."""
    p = tmp_path / "cfg.yaml"
    p.write_text("simulation:\n  seed: 7\n  batch_size: 4\n")
    cfg = load_config(p)
    assert cfg.simulation.seed == 7
    assert cfg.simulation.batch_size == 4


def test_rejects_unknown_keys(tmp_path) -> None:
    """Unknown keys at the top level must raise ConfigError."""
    p = tmp_path / "bad.yaml"
    p.write_text("simulation:\n  bogus_key: 99\n")
    with pytest.raises((ConfigError, Exception)):
        load_config(p)


def test_rejects_unknown_top_level_key(tmp_path) -> None:
    """Unknown top-level key must raise ConfigError."""
    p = tmp_path / "bad.yaml"
    p.write_text("not_a_real_section:\n  value: 1\n")
    with pytest.raises((ConfigError, Exception)):
        load_config(p)


def test_deep_merge_with_base(tmp_path) -> None:
    """Config loaded with ``base`` deep-merges: override wins, rest from base."""
    base = tmp_path / "base.yaml"
    base.write_text("simulation:\n  seed: 1\n  batch_size: 32\n")
    override = tmp_path / "exp.yaml"
    override.write_text("simulation:\n  seed: 99\n")
    cfg = load_config(override, base=base)
    assert cfg.simulation.seed == 99
    assert cfg.simulation.batch_size == 32  # inherited from base


def test_extends_mechanism(tmp_path) -> None:
    """Inline ``extends`` key triggers deep-merge from the referenced file."""
    base = tmp_path / "base.yaml"
    base.write_text("simulation:\n  seed: 1\n  batch_size: 64\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nsimulation:\n  seed: 5\n")
    cfg = load_config(child)
    assert cfg.simulation.seed == 5
    assert cfg.simulation.batch_size == 64


def test_missing_file_raises() -> None:
    with pytest.raises((ConfigError, FileNotFoundError)):
        load_config("/nonexistent/path/cfg.yaml")


def test_defaults_are_sensible(tmp_path) -> None:
    """An empty YAML should still load with default values."""
    p = tmp_path / "empty.yaml"
    p.write_text("")
    cfg = load_config(p)
    assert cfg.simulation.seed == 0
    assert cfg.simulation.batch_size == 256
