"""GPU-only kernel correctness tests.

Run automatically by pytest when tilelang + CUDA are available (e.g. on the
CMP 50HX box); skipped otherwise.  Validates all three TileLang kernels and
both custom autograd Functions against the PyTorch references, forward and
backward.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilechat.kernels import (
    TiledAttention,
    TiledGRUStep,
    reference_attention,
    reference_gru_step,
    tilelang_available,
)

def _cuda_actually_works() -> bool:
    """torch.cuda.is_available() can be true on GPUs the installed torch
    build cannot run kernels on (e.g. sm_52 with cu130 wheels)."""
    if not torch.cuda.is_available():
        return False
    try:
        _ = (torch.zeros(1, device="cuda") + 1).item()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (tilelang_available() and _cuda_actually_works()),
    reason="requires tilelang (TileScale) and a *usable* CUDA device",
)

DEVICE = torch.device("cuda")


def _cuda_rand(*shape, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g).to(DEVICE)


SHAPES = [(64, 10, 500), (7, 4, 32), (5, 10, 100)]  # (batch, seq_len, hidden)


@pytest.mark.parametrize("B,L,H", SHAPES)
def test_attention_kernel_forward(B, L, H):
    from tilechat.kernels import tilescale_attention

    q = _cuda_rand(B, H, seed=B + H)
    enc = _cuda_rand(B, L, H, seed=L + H)
    out = tilescale_attention(q, enc)
    ref = reference_attention(q, enc)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), (out - ref).abs().max()


@pytest.mark.parametrize("B,L,H", SHAPES)
def test_gru_kernels_forward(B, L, H):
    from tilechat.kernels import tilescale_gru_gate, tilescale_linear

    x = _cuda_rand(B, H, seed=1)
    h = _cuda_rand(B, H, seed=2)
    w_ih = _cuda_rand(3 * H, H, seed=3)
    w_hh = _cuda_rand(3 * H, H, seed=4)
    b_ih = _cuda_rand(3 * H, seed=5)
    b_hh = _cuda_rand(3 * H, seed=6)

    gi = tilescale_linear(x, w_ih, b_ih)
    gh = tilescale_linear(h, w_hh, b_hh)
    gi_ref = torch.addmm(b_ih, x, w_ih.t())
    gh_ref = torch.addmm(b_hh, h, w_hh.t())
    assert torch.allclose(gi, gi_ref, atol=1e-3, rtol=1e-3)
    assert torch.allclose(gh, gh_ref, atol=1e-3, rtol=1e-3)

    h_new = tilescale_gru_gate(gi, gh, h)
    h_ref = reference_gru_step(x, h, w_ih, w_hh, b_ih, b_hh)[0]
    assert torch.allclose(h_new, h_ref, atol=1e-3, rtol=1e-3)


def test_autograd_functions_kernel_path_gradients():
    torch.manual_seed(99)
    B, L, H = 16, 10, 500
    x = _cuda_rand(B, H, seed=1).requires_grad_(True)
    h = _cuda_rand(B, H, seed=2).requires_grad_(True)
    w_ih = _cuda_rand(3 * H, H, seed=3).requires_grad_(True)
    w_hh = _cuda_rand(3 * H, H, seed=4).requires_grad_(True)
    b_ih = _cuda_rand(3 * H, seed=5).requires_grad_(True)
    b_hh = _cuda_rand(3 * H, seed=6).requires_grad_(True)
    enc = _cuda_rand(B, L, H, seed=7)
    g_h = _cuda_rand(B, H, seed=8)

    h_new = TiledGRUStep.apply(x, h, w_ih, w_hh, b_ih, b_hh, True)
    ctx = TiledAttention.apply(h_new.detach().requires_grad_(True), enc, True)
    (ctx * 1.0).sum().backward()
    loss = (h_new * g_h).sum()
    loss.backward()

    x2 = x.detach().clone().requires_grad_(True)
    h2 = h.detach().clone().requires_grad_(True)
    w_ih2 = w_ih.detach().clone().requires_grad_(True)
    w_hh2 = w_hh.detach().clone().requires_grad_(True)
    b_ih2 = b_ih.detach().clone().requires_grad_(True)
    b_hh2 = b_hh.detach().clone().requires_grad_(True)
    h_ref, *_ = reference_gru_step(x2, h2, w_ih2, w_hh2, b_ih2, b_hh2)
    ctx_ref = reference_attention(h_ref.detach().requires_grad_(True), enc)
    ctx_ref.sum().backward()
    (h_ref * g_h).sum().backward()

    assert torch.allclose(h_new, h_ref, atol=1e-3, rtol=1e-3)
    for a, b in zip(
        [x.grad, h.grad, w_ih.grad, w_hh.grad, b_ih.grad, b_hh.grad],
        [x2.grad, h2.grad, w_ih2.grad, w_hh2.grad, b_ih2.grad, b_hh2.grad],
    ):
        assert a is not None and b is not None
        assert torch.allclose(a, b, atol=1e-2, rtol=1e-2), (a - b).abs().max()
