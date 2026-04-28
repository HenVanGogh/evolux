"""Tests for evolux.viz.loggers — Phase 1 acceptance tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from evolux.viz.loggers import JsonlLogger, TensorboardLogger, WandbLogger

# ── JsonlLogger ─────────────────────────────────────────────────────────────


def test_jsonl_log_scalar_writes_valid_jsonl(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "run.jsonl")
    logger.log_scalar("loss", 0.42, step=1)
    logger.close()

    lines = (tmp_path / "run.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "scalar"
    assert record["key"] == "loss"
    assert record["value"] == pytest.approx(0.42)
    assert record["step"] == 1


def test_jsonl_log_dict_writes_valid_jsonl(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "run.jsonl")
    logger.log_dict({"a": 1.0, "b": 2.0}, step=5)
    logger.close()

    lines = (tmp_path / "run.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["type"] == "dict"
    assert record["data"]["a"] == 1.0


def test_jsonl_log_hist_no_crash(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "run.jsonl")
    values = torch.randn(100)
    logger.log_hist("weights", values, step=0)
    logger.close()

    lines = (tmp_path / "run.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["type"] == "hist"
    assert "mean" in record
    assert "std" in record


def test_jsonl_log_image_no_crash(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "run.jsonl")
    image = torch.zeros(3, 32, 32)
    logger.log_image("frame", image, step=0)
    logger.close()

    lines = (tmp_path / "run.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["type"] == "image"
    assert record["shape"] == [3, 32, 32]


def test_jsonl_close_idempotent(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "run.jsonl")
    logger.log_scalar("x", 1.0, step=0)
    logger.close()
    logger.close()  # second close must not raise


def test_jsonl_write_after_close_is_noop(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "run.jsonl")
    logger.log_scalar("x", 1.0, step=0)
    logger.close()
    logger.log_scalar("x", 2.0, step=1)  # must not raise

    lines = (tmp_path / "run.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1  # second write was no-op


def test_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "run.jsonl"
    logger = JsonlLogger(nested)
    logger.log_scalar("x", 1.0, step=0)
    logger.close()
    assert nested.exists()


# ── TensorboardLogger — fallback path ───────────────────────────────────────


def _hide_module(name: str) -> None:
    """Insert a sentinel in sys.modules so the import fails."""
    sys.modules[name] = None  # type: ignore[assignment]


def _restore_module(name: str, original: object) -> None:
    if original is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = original  # type: ignore[assignment]


def test_tensorboard_fallback_no_crash(tmp_path: Path) -> None:
    """TensorboardLogger must not raise when tensorboard is absent."""
    original = sys.modules.get("torch.utils.tensorboard")
    _hide_module("torch.utils.tensorboard")
    try:
        logger = TensorboardLogger(tmp_path / "tb")
        logger.log_scalar("loss", 0.1, step=0)
        logger.log_dict({"a": 1.0}, step=0)
        logger.log_hist("w", torch.randn(50), step=0)
        logger.log_image("frame", torch.zeros(3, 8, 8), step=0)
        logger.close()
        logger.close()  # idempotent
    finally:
        _restore_module("torch.utils.tensorboard", original)


def test_tensorboard_fallback_writes_jsonl(tmp_path: Path) -> None:
    original = sys.modules.get("torch.utils.tensorboard")
    _hide_module("torch.utils.tensorboard")
    try:
        logger = TensorboardLogger(tmp_path / "tb")
        logger.log_scalar("loss", 0.5, step=3)
        logger.close()
    finally:
        _restore_module("torch.utils.tensorboard", original)

    jsonl_path = tmp_path / "tb" / "events.jsonl"
    lines = jsonl_path.read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["key"] == "loss"


# ── WandbLogger — fallback path ─────────────────────────────────────────────


def test_wandb_fallback_no_crash(tmp_path: Path) -> None:
    """WandbLogger must not raise when wandb is absent."""
    original = sys.modules.get("wandb")
    _hide_module("wandb")
    try:
        logger = WandbLogger(tmp_path / "wandb_fallback")
        logger.log_scalar("reward", 1.0, step=0)
        logger.log_dict({"a": 1.0}, step=0)
        logger.log_hist("w", torch.randn(50), step=0)
        logger.log_image("frame", torch.zeros(3, 8, 8), step=0)
        logger.close()
        logger.close()  # idempotent
    finally:
        _restore_module("wandb", original)


def test_wandb_fallback_writes_jsonl(tmp_path: Path) -> None:
    original = sys.modules.get("wandb")
    _hide_module("wandb")
    try:
        logger = WandbLogger(tmp_path / "wandb_fallback")
        logger.log_scalar("reward", 2.5, step=7)
        logger.close()
    finally:
        _restore_module("wandb", original)

    jsonl_path = tmp_path / "wandb_fallback" / "events.jsonl"
    lines = jsonl_path.read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["key"] == "reward"
    assert record["value"] == pytest.approx(2.5)


# ── StatsLogger protocol compliance ─────────────────────────────────────────


def test_jsonl_implements_stats_logger_protocol(tmp_path: Path) -> None:
    from evolux.core.protocols import StatsLogger

    logger = JsonlLogger(tmp_path / "protocol_check.jsonl")
    # runtime_checkable Protocol: isinstance check verifies method presence.
    assert isinstance(logger, StatsLogger)
    logger.close()
