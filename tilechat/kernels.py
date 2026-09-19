"""TileScale / TileLang kernels for the seq2seq chatbot decoder hot path.

This module re-implements the decoder's per-token compute as three TileLang
kernels.  TileScale (https://github.com/tile-ai/tilescale) keeps TileLang's
Python kernel language / compiler / JIT / cache interfaces, so these kernels
work with the ``tilelang`` package shipped by TileScale (and with upstream
TileLang wheels).

Kernels (see the README tutorial for a walkthrough):

1. ``_linear_kernel``           -- tiled SIMT GEMM:   C = A @ W.T + bias
2. ``_gru_gate_kernel``         -- fused GRU cell gates: sigmoid / tanh / blend
3. ``_luong_attention_kernel``  -- fused Luong dot-attention:
                                   softmax(q @ enc^T) @ enc in one launch

The original tutorial computes this with a cuDNN GRU plus PyTorch bmm and
softmax.  Here the *forward* pass of each piece is a single hand-written
TileLang kernel, wrapped in ``torch.autograd.Function`` so gradients still
flow: the backward passes are expressed with standard PyTorch matmul /
elementwise ops and are verified against pure-autograd references in the
tests.

Everything falls back to equivalent pure-PyTorch code when tilelang is not
installed or no CUDA device is available, so the whole tutorial also runs on
CPU laptops.  Select the backend with ``TILECHAT_BACKEND=auto|tilescale|pytorch``.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

try:  # TileScale ships the TileLang python package (``import tilelang``).
    import tilelang
    import tilelang.language as T

    _HAS_TILELANG = True
except Exception:  # pragma: no cover - tilelang is optional at runtime
    tilelang = None
    T = None
    _HAS_TILELANG = False


def tilelang_available() -> bool:
    """True if the TileScale/TileLang python package imported successfully."""
    return _HAS_TILELANG


# ---------------------------------------------------------------------------
# Reference (pure PyTorch) implementations -- the fallback backend and the
# correctness oracle used by the tests.
# ---------------------------------------------------------------------------


def reference_linear(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """C = A @ W.T + bias with A (B, K), W (N, K), bias (N,) -> (B, N)."""
    return torch.addmm(bias, a, w.t())


def reference_attention(q: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
    """Fused Luong dot-attention.

    q   (B, H)    -- decoder GRU output for the current step
    enc (B, L, H) -- encoder outputs, batch-first
    returns the context vectors (B, H)

    Matches the tutorial's ``Attn`` ("dot") + the ``context = attn_weights
    .bmm(encoder_outputs.transpose(0, 1))`` step of the decoder.
    """
    scores = torch.bmm(q.unsqueeze(1), enc.transpose(1, 2)).squeeze(1)  # (B, L)
    weights = F.softmax(scores, dim=1)
    return torch.bmm(weights.unsqueeze(1), enc).squeeze(1)  # (B, H)


def reference_gru_step(x, h, weight_ih, weight_hh, bias_ih, bias_hh):
    """One cuDNN-style GRU cell step (same equations as nn.GRU, 1 layer).

    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh (W_in x + b_in + r * (W_hn h + b_hn))
    h' = (1 - z) * n + z * h
    """
    gi = torch.addmm(bias_ih, x, weight_ih.t())  # (B, 3H)
    gh = torch.addmm(bias_hh, h, weight_hh.t())  # (B, 3H)
    i_r, i_z, i_n = gi.chunk(3, dim=1)
    h_r, h_z, h_n = gh.chunk(3, dim=1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h, gi, gh, r, z, n


def _gate_torch(gi, gh, h_prev):
    """PyTorch implementation of the fused gate math done by the gate kernel."""
    h_dim = h_prev.shape[1]
    i_r, i_z, i_n = gi[:, :h_dim], gi[:, h_dim : 2 * h_dim], gi[:, 2 * h_dim :]
    h_r, h_z, h_n = gh[:, :h_dim], gh[:, h_dim : 2 * h_dim], gh[:, 2 * h_dim :]
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h_prev


# ---------------------------------------------------------------------------
# TileLang (TileScale) kernels.
#
# Only language constructs proven by the official examples are used:
#   @tilelang.jit + fn.compile(...), T.const dims, T.Tensor annotations,
#   T.empty outputs, T.Kernel(grid), T.get_thread_binding, T.alloc_shared,
#   T.alloc_local, T.alloc_fragment, T.clear, T.serial, T.Parallel, T.copy,
#   T.ceildiv, T.exp, T.max.  All tensors are float32 (the tutorial's dtype).
# ---------------------------------------------------------------------------

if _HAS_TILELANG:

    @tilelang.jit
    def _linear_kernel(
        A,
        W,
        bias,
        BLOCK_M: int,
        BLOCK_N: int,
        BLOCK_K: int,
        dtype: T.dtype = T.float32,
        accum_dtype: T.dtype = T.float32,
    ):
        """C = A @ W.T + bias for A (B, K), W (N, K) -> C (B, N).

        A tiled SIMT GEMM: shared-memory tiles for A and W, a register
        fragment accumulator, boundary-free because the wrapper only ever
        compiles block sizes that divide B, K and N.
        """
        B, K, N = T.const("B, K, N")
        A: T.Tensor((B, K), dtype)
        W: T.Tensor((N, K), dtype)
        bias: T.Tensor((N,), dtype)
        C = T.empty((B, N), dtype)

        with T.Kernel(T.ceildiv(B, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (bm, bn):
            m0 = bm * BLOCK_M
            n0 = bn * BLOCK_N
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            W_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(C_local)
            for ko in T.serial(T.ceildiv(K, BLOCK_K)):
                k0 = ko * BLOCK_K
                T.copy(A[m0, k0], A_shared)
                T.copy(W[n0, k0], W_shared)
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    for k in T.serial(BLOCK_K):
                        C_local[i, j] += A_shared[i, k] * W_shared[j, k]
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                C[m0 + i, n0 + j] = C_local[i, j] + bias[n0 + j]
        return C

    @tilelang.jit
    def _gru_gate_kernel(
        GI,
        GH,
        Hprev,
        THREADS: int,
        dtype: T.dtype = T.float32,
    ):
        """Fused GRU gate evaluation for a whole batch.

        GI, GH (B, 3H): input / hidden projections (r, z, n sections).
        Hprev (B, H): previous hidden state. Returns h' (B, H):

            r  = sigmoid(GI[:,0:H]    + GH[:,0:H])
            z  = sigmoid(GI[:,H:2H]   + GH[:,H:2H])
            n  = tanh  (GI[:,2H:3H] + r * GH[:,2H:3H])   # via exp()
            h' = (1 - z) * n + z * Hprev
        """
        B, H, N3 = T.const("B, H, N3")
        GI: T.Tensor((B, N3), dtype)
        GH: T.Tensor((B, N3), dtype)
        Hprev: T.Tensor((B, H), dtype)
        Hout = T.empty((B, H), dtype)

        with T.Kernel(T.ceildiv(B * H, THREADS), threads=THREADS) as bx:
            tn = T.get_thread_binding(0)
            p = bx * THREADS + tn
            if p < B * H:
                b = p // H
                hh = p - b * H
                r = 1.0 / (1.0 + T.exp(0.0 - (GI[b, hh] + GH[b, hh])))
                z = 1.0 / (1.0 + T.exp(0.0 - (GI[b, H + hh] + GH[b, H + hh])))
                n = 2.0 / (1.0 + T.exp(0.0 - 2.0 * (GI[b, 2 * H + hh] + r * GH[b, 2 * H + hh]))) - 1.0
                hprev = Hprev[b, hh]
                Hout[b, hh] = (1.0 - z) * n + z * hprev
        return Hout

    @tilelang.jit
    def _luong_attention_kernel(
        Q,
        S,
        THREADS: int,
        dtype: T.dtype = T.float32,
    ):
        """Fused Luong dot-attention: softmax(Q @ S^T) @ S in a single kernel.

        Q (B, H): decoder GRU output for the current step.
        S (B, L, H): encoder outputs, batch-first.
        Returns the context vectors (B, H).

        One CUDA block per batch element.  Block-local stages: (1) the first
        L threads dot q against each encoder output row, (2) thread 0 takes a
        numerically-stable softmax over the L scores, (3) the first H threads
        form the weighted sum of encoder outputs.  TileLang inserts the
        shared-memory barriers between stages automatically.
        """
        B, L, H = T.const("B, L, H")
        Q: T.Tensor((B, H), dtype)
        S: T.Tensor((B, L, H), dtype)
        O = T.empty((B, H), dtype)

        with T.Kernel(B, threads=THREADS) as b:
            tn = T.get_thread_binding(0)
            s_shared = T.alloc_shared((L,), dtype)
            w_shared = T.alloc_shared((L,), dtype)

            # Stage 1: scores[l] = dot(q, S[b, l, :]) -- first L threads.
            if tn < L:
                acc = T.alloc_local((1,), dtype)
                T.clear(acc)
                for h in T.serial(H):
                    acc[0] += Q[b, h] * S[b, tn, h]
                s_shared[tn] = acc[0]

            # Stage 2: numerically-stable softmax over the L scores.
            if tn == 0:
                mx = T.alloc_local((1,), dtype)
                sm = T.alloc_local((1,), dtype)
                T.clear(mx)
                T.clear(sm)
                mx[0] = s_shared[0]
                for l in T.serial(L):
                    mx[0] = T.max(mx[0], s_shared[l])
                for l in T.serial(L):
                    e = T.exp(s_shared[l] - mx[0])
                    w_shared[l] = e
                    sm[0] += e
                for l in T.serial(L):
                    w_shared[l] = w_shared[l] / sm[0]

            # Stage 3: context[h] = sum_l w[l] * S[b, l, h] -- first H threads.
            if tn < H:
                acc2 = T.alloc_local((1,), dtype)
                T.clear(acc2)
                for l in T.serial(L):
                    acc2[0] += w_shared[l] * S[b, l, tn]
                O[b, tn] = acc2[0]
        return O


# ---------------------------------------------------------------------------
# Kernel compile cache + python wrappers.
# ---------------------------------------------------------------------------

_CACHE: Dict[Tuple, object] = {}
_LOCK = threading.Lock()
_WARNED: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[tilechat] {msg}")


def _get_kernel(fn, params: dict):
    """Compile-once cache: kernels are shape-specialized by TileLang's JIT."""
    # @tilelang.jit objects may not expose __name__; fall back to repr.
    fn_key = getattr(fn, "__name__", repr(fn))
    key = (fn_key, tuple(sorted((k, repr(v)) for k, v in params.items())))
    with _LOCK:
        kern = _CACHE.get(key)
        if kern is None:
            kern = fn.compile(**params)
            _CACHE[key] = kern
    return kern


def _threads_for(hidden: int) -> int:
    """Threads-per-block covering the largest kernel dimension (<= 1024)."""
    t = max(32, ((hidden + 31) // 32) * 32)
    if t > 1024:
        raise ValueError(
            f"hidden size {hidden} too large for the fused attention kernel "
            "(max 1024); use TILECHAT_BACKEND=pytorch"
        )
    return t


def kernels_usable(t: torch.Tensor) -> bool:
    """Decide whether the TileScale kernels should handle this tensor.

    Honours ``TILECHAT_BACKEND``: ``pytorch`` forces the fallback,
    ``tilescale`` forces the kernels (error if unusable), ``auto`` (default)
    uses the kernels whenever tilelang is importable and the tensor is CUDA
    float32.
    """
    pref = os.environ.get("TILECHAT_BACKEND", "auto").strip().lower()
    if pref == "pytorch":
        return False
    if not _HAS_TILELANG:
        if pref == "tilescale":
            raise RuntimeError("TILECHAT_BACKEND=tilescale but the tilelang package is not installed")
        return False
    if t.device.type == "cpu":
        # The kernels JIT for CUDA; on CPU tensors always take the reference path.
        if pref == "tilescale":
            raise RuntimeError("TILECHAT_BACKEND=tilescale requires CUDA tensors")
        return False
    if t.dtype != torch.float32:
        if pref == "tilescale":
            raise RuntimeError("TileScale chatbot kernels require float32 tensors")
        return False
    return True


def tilescale_linear(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """C = A @ W.T + bias computed by the ``_linear_kernel`` TileLang kernel.

    Inputs are zero-padded to multiples of the fixed tile (16, 64, 32) --
    tile sizes chosen so every shared tile divides evenly across the 128
    threads (required by tilelang's layout inference).  Zero padding changes
    nothing mathematically and the padding rim is sliced away.
    """
    a = a.contiguous()
    w = w.contiguous()
    bias = bias.contiguous()
    B, K = a.shape
    N = w.shape[0]
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 32
    Bp = ((B + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    Kp = ((K + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    Np = ((N + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

    kern = _get_kernel(_linear_kernel, dict(B=Bp, K=Kp, N=Np, BLOCK_M=BLOCK_M,
                                            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K))

    def _pad2(x, rows, cols):
        if x.shape == (rows, cols):
            return x
        out = torch.zeros(rows, cols, device=x.device, dtype=x.dtype)
        out[: x.shape[0], : x.shape[1]] = x
        return out

    bias_p = bias
    if Np != N:
        bias_p = torch.zeros(Np, device=bias.device, dtype=bias.dtype)
        bias_p[:N] = bias

    c = kern(_pad2(a, Bp, Kp), _pad2(w, Np, Kp), bias_p)
    return c[:B, :N]


def tilescale_gru_gate(gi: torch.Tensor, gh: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    """h' = GRU gates(GI, GH) applied to h_prev via ``_gru_gate_kernel``."""
    gi = gi.contiguous()
    gh = gh.contiguous()
    h_prev = h_prev.contiguous()
    B, N3 = gi.shape
    H = h_prev.shape[1]
    kern = _get_kernel(_gru_gate_kernel, dict(B=B, H=H, N3=N3, THREADS=256))
    return kern(gi, gh, h_prev)


def tilescale_attention(q: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
    """Fused attention context computed by ``_luong_attention_kernel``."""
    q = q.contiguous()
    enc = enc.contiguous()
    B, L, H = enc.shape
    kern = _get_kernel(_luong_attention_kernel, dict(B=B, L=L, H=H, THREADS=_threads_for(max(H, L))))
    return kern(q, enc)


# ---------------------------------------------------------------------------
# torch.autograd.Function wrappers: TileLang kernel forward, PyTorch backward.
# The backward math is the analytic GRU-cell / softmax-attention gradient and
# is checked against autograd in tests/test_math.py and, on CUDA machines,
# tests/test_kernels.py.
# ---------------------------------------------------------------------------


class TiledGRUStep(torch.autograd.Function):
    """One GRU-cell step; forward runs the TileScale kernels when available."""

    @staticmethod
    def forward(ctx, x, h, weight_ih, weight_hh, bias_ih, bias_hh, use_kernel):
        use_kernel = bool(use_kernel)
        if use_kernel:
            gi = tilescale_linear(x, weight_ih, bias_ih)
            gh = tilescale_linear(h, weight_hh, bias_hh)
            h_new = tilescale_gru_gate(gi, gh, h)
        else:
            gi = reference_linear(x, weight_ih, bias_ih)
            gh = reference_linear(h, weight_hh, bias_hh)
            h_new = _gate_torch(gi, gh, h)
        ctx.save_for_backward(x, h, weight_ih, weight_hh, gi, gh)
        return h_new

    @staticmethod
    def backward(ctx, grad_out):
        x, h, weight_ih, weight_hh, gi, gh = ctx.saved_tensors
        g = grad_out
        h_dim = h.shape[1]
        i_r, i_z, i_n = gi[:, :h_dim], gi[:, h_dim : 2 * h_dim], gi[:, 2 * h_dim :]
        h_r, h_z, h_n = gh[:, :h_dim], gh[:, h_dim : 2 * h_dim], gh[:, 2 * h_dim :]
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        # h' = (1 - z) * n + z * h
        dn = g * (1 - z)
        dz = g * (h - n)
        # n = tanh(i_n + r * h_n)
        du = dn * (1 - n * n)
        da = du * h_n * r * (1 - r)  # through r
        db = dz * z * (1 - z)
        grad_gi = torch.cat([da, db, du], dim=1)
        grad_gh = torch.cat([da, db, du * r], dim=1)
        grad_x = grad_gi @ weight_ih
        grad_h = grad_gh @ weight_hh + g * z
        grad_wih = grad_gi.t() @ x
        grad_whh = grad_gh.t() @ h
        grad_bih = grad_gi.sum(0)
        grad_bhh = grad_gh.sum(0)
        return grad_x, grad_h, grad_wih, grad_whh, grad_bih, grad_bhh, None


class TiledAttention(torch.autograd.Function):
    """Fused Luong attention; forward runs the TileScale kernel when available."""

    @staticmethod
    def forward(ctx, q, enc, use_kernel):
        if bool(use_kernel):
            context = tilescale_attention(q, enc)
        else:
            context = reference_attention(q, enc)
        ctx.save_for_backward(q, enc)
        return context

    @staticmethod
    def backward(ctx, grad_out):
        q, enc = ctx.saved_tensors
        scores = torch.bmm(q.unsqueeze(1), enc.transpose(1, 2)).squeeze(1)  # (B, L)
        w = torch.softmax(scores, dim=1)
        # dw[b, l] = sum_h grad_out[b, h] * enc[b, l, h]
        dw = torch.bmm(grad_out.unsqueeze(1), enc.transpose(1, 2)).squeeze(1)
        # softmax backward
        ds = w * (dw - (dw * w).sum(dim=1, keepdim=True))
        dq = torch.bmm(ds.unsqueeze(1), enc).squeeze(1)
        # enc receives gradient both through the weighted sum (direct) and
        # through the scores s = q @ enc^T that produced the softmax weights.
        denc = w.unsqueeze(2) * grad_out.unsqueeze(1) + ds.unsqueeze(2) * q.unsqueeze(1)
        return dq, denc, None
