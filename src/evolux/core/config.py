"""Typed config loading via pydantic.

Configs are YAML files matching one of the dataclasses defined here (or in
module-local config files). Loading produces validated, immutable objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConfigError(ValueError):
    """Raised on malformed or missing configuration."""


class BaseConfig(BaseModel):
    """Base for every typed config. Frozen + extra-forbid by default."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ── Top-level config schema (skeletal — modules extend in their own files) ──


class SimulationConfig(BaseConfig):
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:0"
    batch_size: int = 256
    max_generations: int = 1000
    steps_per_generation: int = 500
    log_interval: int = 10
    checkpoint_interval: int = 50
    output_dir: str = "runs"


class PopulationConfig(BaseConfig):
    size: int = 256
    initial_species: int = 1


class EvoluxConfig(BaseConfig):
    """Top-level config; module configs are added as additional fields."""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    # Module-specific configs are loaded as plain dicts here and validated by
    # each module's own Config class. This keeps `core` independent of layers
    # above it.
    modules: dict[str, Any] = Field(default_factory=dict)


# ── Loader ──────────────────────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins for scalars."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    path: str | Path,
    *,
    base: str | Path | None = None,
) -> EvoluxConfig:
    """Load a YAML config file.

    If ``base`` is given, that file is loaded first and the main config is
    deep-merged on top. This supports the common pattern of an experiment
    config that overrides a default.

    Notes
    -----
    Design rationale: ``docs/ARCHITECTURE.md`` §6 — *Configs are typed*.
    Config loading uses pydantic with ``extra="forbid"`` so any unknown key
    raises a ``ConfigError`` at process start rather than silently being
    ignored until step 50,000.  The ``extends`` key enables a lightweight
    inheritance chain (experiment overrides defaults) without duplicating
    large YAML files.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    # Inline `extends: <relative-path>` support — resolved relative to *this* file.
    extends = raw.pop("extends", None)
    if extends is not None:
        ext_path = (p.parent / extends).resolve()
        if not ext_path.exists():
            raise ConfigError(f"`extends` target not found: {ext_path} (from {p})")
        ext_raw = yaml.safe_load(ext_path.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(ext_raw, raw)

    if base is not None:
        bp = Path(base)
        if not bp.exists():
            raise ConfigError(f"Base config not found: {bp}")
        base_raw = yaml.safe_load(bp.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(base_raw, raw)

    try:
        return EvoluxConfig(**raw)
    except Exception as e:
        raise ConfigError(f"Failed to validate config {p}: {e}") from e
