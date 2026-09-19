"""Backend-independent math tests (run everywhere, CPU included).

* the pure-PyTorch GRU-step reference must match ``nn.GRUCell`` exactly,
* the custom autograd Functions (PyTorch path) must produce gradients equal
  to autograd through the references,
* kernel-selection policy behaves as documented.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilechat.kernels import (
    TiledAttention,
    TiledGRUStep,
    _gate_torch,
    reference_attention,
    reference_gru_step,
    reference_linear,
)


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_reference_gru_step_matches_nn_grucell():
    torch.manual_seed(1)
    B, H = 4, 7
    cell = torch.nn.GRUCell(H, H)
    x = _rand(B, H, seed=2)
    h = _rand(B, H, seed=3)
    ref_out, *_ = reference_gru_step(
        x, h, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh
    )
    torch_out = cell(x, h)
    assert torch.allclose(ref_out, torch_out, atol=1e-6, rtol=1e-5)


def test_gru_step_function_pytorch_path_gradients_match_autograd():
    torch.manual_seed(4)
    B, H = 3, 5
    x1 = _rand(B, H, seed=5).requires_grad_(True)
    h1 = _rand(B, H, seed=6).requires_grad_(True)
    wih = _rand(3 * H, H, seed=7).requires_grad_(True)
    whh = _rand(3 * H, H, seed=8).requires_grad_(True)
    bih = _rand(3 * H, seed=9).requires_grad_(True)
    bhh = _rand(3 * H, seed=10).requires_grad_(True)
    gseed = _rand(B, H, seed=11)

    # custom Function, PyTorch (fallback) forward
    out_fn = TiledGRUStep.apply(x1, h1, wih, whh, bih, bhh, False)
    out_fn.backward(gseed)
    grads_fn = [x1.grad, h1.grad, wih.grad, whh.grad, bih.grad, bhh.grad]

    # autograd through the reference implementation
    x2 = x1.detach().clone().requires_grad_(True)
    h2 = h1.detach().clone().requires_grad_(True)
    wih2 = wih.detach().clone().requires_grad_(True)
    whh2 = whh.detach().clone().requires_grad_(True)
    bih2 = bih.detach().clone().requires_grad_(True)
    bhh2 = bhh.detach().clone().requires_grad_(True)
    out_ref, *_ = reference_gru_step(x2, h2, wih2, whh2, bih2, bhh2)
    out_ref.backward(gseed)
    grads_ref = [x2.grad, h2.grad, wih2.grad, whh2.grad, bih2.grad, bhh2.grad]

    assert torch.allclose(out_fn.detach(), out_ref.detach(), atol=1e-6, rtol=1e-5)
    for a, b in zip(grads_fn, grads_ref):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_attention_function_pytorch_path_gradients_match_autograd():
    torch.manual_seed(12)
    B, L, H = 3, 4, 5
    q1 = _rand(B, H, seed=13).requires_grad_(True)
    enc1 = _rand(B, L, H, seed=14).requires_grad_(True)
    gseed = _rand(B, H, seed=15)

    out_fn = TiledAttention.apply(q1, enc1, False)
    out_fn.backward(gseed)

    q2 = q1.detach().clone().requires_grad_(True)
    enc2 = enc1.detach().clone().requires_grad_(True)
    out_ref = reference_attention(q2, enc2)
    out_ref.backward(gseed)

    assert torch.allclose(out_fn.detach(), out_ref.detach(), atol=1e-6, rtol=1e-5)
    assert torch.allclose(q1.grad, q2.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(enc1.grad, enc2.grad, atol=1e-5, rtol=1e-4)


def test_gate_math_matches_chunks():
    torch.manual_seed(16)
    B, H = 2, 6
    x, h = _rand(B, H, seed=17), _rand(B, H, seed=18)
    wih, whh = _rand(3 * H, H, seed=19), _rand(3 * H, H, seed=20)
    bih, bhh = _rand(3 * H, seed=21), _rand(3 * H, seed=22)
    out_ref, gi, gh, r, z, n = reference_gru_step(x, h, wih, whh, bih, bhh)
    assert torch.allclose(_gate_torch(gi, gh, h), out_ref, atol=1e-6)
    assert torch.allclose(r, torch.sigmoid(gi[:, :H] + gh[:, :H]), atol=1e-6)
    assert torch.allclose(z, torch.sigmoid(gi[:, H : 2 * H] + gh[:, H : 2 * H]), atol=1e-6)


def test_reference_linear_matches_addmm():
    torch.manual_seed(23)
    B, K, N = 4, 5, 7
    a, w, b = _rand(B, K, seed=24), _rand(N, K, seed=25), _rand(N, seed=26)
    assert torch.allclose(reference_linear(a, w, b), torch.addmm(b, a, w.t()), atol=1e-6)


def test_backend_policy(monkeypatch):
    from tilechat.kernels import kernels_usable

    cpu = torch.zeros(2, 2)

    monkeypatch.setenv("TILECHAT_BACKEND", "pytorch")
    assert kernels_usable(cpu) is False

    monkeypatch.setenv("TILECHAT_BACKEND", "auto")
    # CPU tensors always take the reference path (kernels JIT for CUDA).
    assert kernels_usable(cpu) is False

    monkeypatch.setenv("TILECHAT_BACKEND", "tilescale")
    with pytest.raises(RuntimeError):
        kernels_usable(cpu)
