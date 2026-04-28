"""Tests for the perception module — Phase 1 encoders."""

from __future__ import annotations

import pytest
import torch

from evolux.core.rng import RNG
from evolux.perception import PERCEPTION_REGISTRY, ConcatEncoder, ProprioMLP, VisionCNN

# ---------------------------------------------------------------------------
# VisionCNN
# ---------------------------------------------------------------------------

_IMG_SIZE = (32, 32)
_IN_C = 3
_B = 4
_OUT_DIM = 64


def test_vision_cnn_output_shape() -> None:
    enc = VisionCNN(in_channels=_IN_C, output_dim=_OUT_DIM, img_size=_IMG_SIZE)
    x = torch.randn(_B, _IN_C, *_IMG_SIZE)
    out = enc(x)
    assert out.shape == (_B, _OUT_DIM), f"expected ({_B}, {_OUT_DIM}), got {out.shape}"


def test_vision_cnn_output_dim_attr() -> None:
    enc = VisionCNN(in_channels=_IN_C, output_dim=_OUT_DIM, img_size=_IMG_SIZE)
    assert enc.output_dim == _OUT_DIM


def test_vision_cnn_gradient_flow() -> None:
    enc = VisionCNN(in_channels=_IN_C, output_dim=_OUT_DIM, img_size=_IMG_SIZE)
    x = torch.randn(_B, _IN_C, *_IMG_SIZE, requires_grad=True)
    out = enc(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0.0


def test_vision_cnn_deterministic() -> None:
    enc = VisionCNN(in_channels=_IN_C, output_dim=_OUT_DIM, img_size=_IMG_SIZE)
    x = torch.randn(_B, _IN_C, *_IMG_SIZE)
    assert torch.equal(enc(x), enc(x))


def test_vision_cnn_deterministic_given_rng(rng: RNG) -> None:
    gen = rng.split("vision_cnn_test")
    enc = VisionCNN(in_channels=_IN_C, output_dim=_OUT_DIM, img_size=_IMG_SIZE)
    x = torch.randn(_B, _IN_C, *_IMG_SIZE, generator=gen)
    assert torch.equal(enc(x), enc(x))


def test_vision_cnn_registered() -> None:
    assert "vision_cnn_v1" in PERCEPTION_REGISTRY
    assert PERCEPTION_REGISTRY.get("vision_cnn_v1") is VisionCNN


# ---------------------------------------------------------------------------
# ProprioMLP
# ---------------------------------------------------------------------------

_P_DIM = 16
_HIDDEN = 32
_OUT_DIM_P = 48


def test_proprio_mlp_output_shape() -> None:
    enc = ProprioMLP(in_dim=_P_DIM, output_dim=_OUT_DIM_P, hidden_dim=_HIDDEN)
    x = torch.randn(_B, _P_DIM)
    out = enc(x)
    assert out.shape == (_B, _OUT_DIM_P), f"expected ({_B}, {_OUT_DIM_P}), got {out.shape}"


def test_proprio_mlp_output_dim_attr() -> None:
    enc = ProprioMLP(in_dim=_P_DIM, output_dim=_OUT_DIM_P, hidden_dim=_HIDDEN)
    assert enc.output_dim == _OUT_DIM_P


def test_proprio_mlp_gradient_flow() -> None:
    enc = ProprioMLP(in_dim=_P_DIM, output_dim=_OUT_DIM_P, hidden_dim=_HIDDEN)
    x = torch.randn(_B, _P_DIM, requires_grad=True)
    out = enc(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0.0


def test_proprio_mlp_deterministic() -> None:
    enc = ProprioMLP(in_dim=_P_DIM, output_dim=_OUT_DIM_P, hidden_dim=_HIDDEN)
    x = torch.randn(_B, _P_DIM)
    assert torch.equal(enc(x), enc(x))


def test_proprio_mlp_deterministic_given_rng(rng: RNG) -> None:
    gen = rng.split("proprio_mlp_test")
    enc = ProprioMLP(in_dim=_P_DIM, output_dim=_OUT_DIM_P, hidden_dim=_HIDDEN)
    x = torch.randn(_B, _P_DIM, generator=gen)
    assert torch.equal(enc(x), enc(x))


def test_proprio_mlp_registered() -> None:
    assert "proprio_mlp_v1" in PERCEPTION_REGISTRY
    assert PERCEPTION_REGISTRY.get("proprio_mlp_v1") is ProprioMLP


# ---------------------------------------------------------------------------
# ConcatEncoder
# ---------------------------------------------------------------------------

_VIS_OUT = 32
_PRO_OUT = 24
_CONCAT_OUT = _VIS_OUT + _PRO_OUT


def _make_concat_encoder() -> ConcatEncoder:
    vis = VisionCNN(in_channels=_IN_C, output_dim=_VIS_OUT, img_size=_IMG_SIZE)
    pro = ProprioMLP(in_dim=_P_DIM, output_dim=_PRO_OUT, hidden_dim=_HIDDEN)
    return ConcatEncoder(encoders={"vision": vis, "proprio": pro})


def test_concat_encoder_output_dim_attr() -> None:
    enc = _make_concat_encoder()
    assert enc.output_dim == _CONCAT_OUT


def test_concat_encoder_output_shape() -> None:
    enc = _make_concat_encoder()
    obs = {
        "vision": torch.randn(_B, _IN_C, *_IMG_SIZE),
        "proprio": torch.randn(_B, _P_DIM),
    }
    out = enc(obs)
    assert out.shape == (_B, _CONCAT_OUT), f"expected ({_B}, {_CONCAT_OUT}), got {out.shape}"


def test_concat_encoder_gradient_flow() -> None:
    enc = _make_concat_encoder()
    vis_x = torch.randn(_B, _IN_C, *_IMG_SIZE, requires_grad=True)
    pro_x = torch.randn(_B, _P_DIM, requires_grad=True)
    obs = {"vision": vis_x, "proprio": pro_x}
    out = enc(obs)
    out.sum().backward()
    assert vis_x.grad is not None and vis_x.grad.abs().sum().item() > 0.0
    assert pro_x.grad is not None and pro_x.grad.abs().sum().item() > 0.0


def test_concat_encoder_deterministic() -> None:
    enc = _make_concat_encoder()
    obs = {
        "vision": torch.randn(_B, _IN_C, *_IMG_SIZE),
        "proprio": torch.randn(_B, _P_DIM),
    }
    assert torch.equal(enc(obs), enc(obs))


def test_concat_encoder_registered() -> None:
    assert "concat" in PERCEPTION_REGISTRY
    assert PERCEPTION_REGISTRY.get("concat") is ConcatEncoder


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_vision_cnn_perf_cpu_smoke() -> None:
    """VisionCNN forward pass completes in < 500 ms on any CPU (regression guard)."""
    import time

    enc = VisionCNN(in_channels=3, output_dim=256, img_size=(32, 32))
    enc.eval()
    x = torch.randn(64, 3, 32, 32)

    with torch.no_grad():
        enc(x)  # warm-up

    start = time.perf_counter()
    with torch.no_grad():
        enc(x)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 500.0, f"VisionCNN catastrophically slow: {elapsed_ms:.1f} ms"


@pytest.mark.gpu
def test_vision_cnn_perf_gpu() -> None:
    """VisionCNN B=256, 64x64 RGB should be <= 3 ms on GPU (SPEC budget)."""
    import time

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    enc = VisionCNN(in_channels=3, output_dim=256, img_size=(64, 64)).to(device)
    enc.eval()
    x = torch.randn(256, 3, 64, 64, device=device)

    with torch.no_grad():
        enc(x)  # warm-up
    torch.cuda.synchronize()

    n_iters = 20
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            enc(x)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / n_iters * 1000

    assert elapsed_ms <= 3.0, f"VisionCNN too slow on GPU: {elapsed_ms:.2f} ms > 3 ms budget"
