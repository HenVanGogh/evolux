"""Smoke-import test: every module package must import cleanly."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "evolux",
    "evolux.core",
    "evolux.tensors",
    "evolux.perception",
    "evolux.memory",
    "evolux.genome",
    "evolux.brain",
    "evolux.morphology",
    "evolux.physics",
    "evolux.world",
    "evolux.environment",
    "evolux.fitness",
    "evolux.evolution",
    "evolux.distributed",
    "evolux.viz",
    "evolux.orchestrator",
    "evolux.orchestrator.cli",
]


@pytest.mark.parametrize("modname", MODULES)
def test_import(modname: str) -> None:
    importlib.import_module(modname)
