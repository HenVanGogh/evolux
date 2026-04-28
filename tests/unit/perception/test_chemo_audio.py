"""Tests for the perception module — Phase 2 encoders (ChemoEncoder, AudioEncoder)."""

from __future__ import annotations

import pytest
import torch

from evolux.core.rng import RNG
from evolux.perception import PERCEPTION_REGISTRY, AudioEncoder, ChemoEncoder

# ---------------------------------------------------------------------------
# ChemoEncoder
# ---------------------------------------------------------------------------

_N_CHANNELS = 8
_N_RBF = 16
_HIDDEN_C = 32
_OUT_DIM_C = 48
_B = 4
_B_LARGE = 16


def test_chemo_output_shape() -> None:
    enc = ChemoEncoder(
        n_channels=_N_CHANNELS, output_dim=_OUT_DIM_C, n_rbf=_N_RBF, hidden_dim=_HIDDEN_C
    )
    x = torch.randn(_B, _N_CHANNELS)
    out = enc(x)
    assert out.shape == (_B, _OUT_DIM_C), f"expected ({_B}, {_OUT_DIM_C}), got {out.shape}"


def test_chemo_output_dim_attr() -> None:
    enc = ChemoEncoder(n_channels=_N_CHANNELS, output_dim=_OUT_DIM_C)
    assert enc.output_dim == _OUT_DIM_C


def test_chemo_gradient_flow() -> None:
    enc = ChemoEncoder(
        n_channels=_N_CHANNELS, output_dim=_OUT_DIM_C, n_rbf=_N_RBF, hidden_dim=_HIDDEN_C
    )
    x = torch.randn(_B, _N_CHANNELS, requires_grad=True)
    out = enc(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0.0


def test_chemo_deterministic() -> None:
    enc = ChemoEncoder(
        n_channels=_N_CHANNELS, output_dim=_OUT_DIM_C, n_rbf=_N_RBF, hidden_dim=_HIDDEN_C
    )
    x = torch.randn(_B, _N_CHANNELS)
    assert torch.equal(enc(x), enc(x))


def test_chemo_deterministic_given_rng(rng: RNG) -> None:
    gen = rng.split("chemo_test")
    enc = ChemoEncoder(
        n_channels=_N_CHANNELS, output_dim=_OUT_DIM_C, n_rbf=_N_RBF, hidden_dim=_HIDDEN_C
    )
    x = torch.randn(_B, _N_CHANNELS, generator=gen)
    assert torch.equal(enc(x), enc(x))


def test_chemo_batch_large() -> None:
    enc = ChemoEncoder(
        n_channels=_N_CHANNELS, output_dim=_OUT_DIM_C, n_rbf=_N_RBF, hidden_dim=_HIDDEN_C
    )
    x = torch.randn(_B_LARGE, _N_CHANNELS)
    out = enc(x)
    assert out.shape == (_B_LARGE, _OUT_DIM_C)


def test_chemo_registered() -> None:
    assert "chemo_rbf_v1" in PERCEPTION_REGISTRY
    assert PERCEPTION_REGISTRY.get("chemo_rbf_v1") is ChemoEncoder


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------

_IN_C_A = 2
_T = 64
_OUT_DIM_A = 32


def test_audio_output_shape() -> None:
    enc = AudioEncoder(in_channels=_IN_C_A, output_dim=_OUT_DIM_A)
    x = torch.randn(_B, _IN_C_A, _T)
    out = enc(x)
    assert out.shape == (_B, _OUT_DIM_A), f"expected ({_B}, {_OUT_DIM_A}), got {out.shape}"


def test_audio_output_dim_attr() -> None:
    enc = AudioEncoder(in_channels=_IN_C_A, output_dim=_OUT_DIM_A)
    assert enc.output_dim == _OUT_DIM_A


def test_audio_gradient_flow() -> None:
    enc = AudioEncoder(in_channels=_IN_C_A, output_dim=_OUT_DIM_A)
    x = torch.randn(_B, _IN_C_A, _T, requires_grad=True)
    out = enc(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0.0


def test_audio_deterministic() -> None:
    enc = AudioEncoder(in_channels=_IN_C_A, output_dim=_OUT_DIM_A)
    x = torch.randn(_B, _IN_C_A, _T)
    assert torch.equal(enc(x), enc(x))


def test_audio_deterministic_given_rng(rng: RNG) -> None:
    gen = rng.split("audio_test")
    enc = AudioEncoder(in_channels=_IN_C_A, output_dim=_OUT_DIM_A)
    x = torch.randn(_B, _IN_C_A, _T, generator=gen)
    assert torch.equal(enc(x), enc(x))


def test_audio_batch_large() -> None:
    enc = AudioEncoder(in_channels=_IN_C_A, output_dim=_OUT_DIM_A)
    x = torch.randn(_B_LARGE, _IN_C_A, _T)
    out = enc(x)
    assert out.shape == (_B_LARGE, _OUT_DIM_A)


def test_audio_registered() -> None:
    assert "audio_conv_v1" in PERCEPTION_REGISTRY
    assert PERCEPTION_REGISTRY.get("audio_conv_v1") is AudioEncoder


# ---------------------------------------------------------------------------
# Performance smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_chemo_perf_cpu_smoke() -> None:
    """ChemoEncoder forward pass completes in < 500 ms on any CPU."""
    import time

    enc = ChemoEncoder(n_channels=16, output_dim=64, n_rbf=32, hidden_dim=64)
    enc.eval()
    x = torch.randn(256, 16)

    with torch.no_grad():
        enc(x)  # warm-up

    start = time.perf_counter()
    with torch.no_grad():
        enc(x)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 500.0, f"ChemoEncoder catastrophically slow: {elapsed_ms:.1f} ms"


@pytest.mark.benchmark
def test_audio_perf_cpu_smoke() -> None:
    """AudioEncoder forward pass completes in < 500 ms on any CPU."""
    import time

    enc = AudioEncoder(in_channels=4, output_dim=64)
    enc.eval()
    x = torch.randn(64, 4, 256)

    with torch.no_grad():
        enc(x)  # warm-up

    start = time.perf_counter()
    with torch.no_grad():
        enc(x)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 500.0, f"AudioEncoder catastrophically slow: {elapsed_ms:.1f} ms"
