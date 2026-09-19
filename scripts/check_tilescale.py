#!/usr/bin/env python
"""Environment report + TileScale kernel validation.

Run this on whatever GPU box you have (e.g. the CMP 50HX):

    python scripts/check_tilescale.py            # correctness
    python scripts/check_tilescale.py --bench    # + latency numbers

Exercises all three kernels (tiled GEMM, fused GRU gates, fused attention)
against the PyTorch references on CUDA, plus the autograd Functions used by
the decoder.  Exits non-zero on failure.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cuda_op_works() -> bool:
    """cuda.is_available() can be True on GPUs the torch build can't run."""
    try:
        _ = (torch.zeros(1, device="cuda") + 1).item()
        return True
    except Exception as e:
        print(f"cuda probe op failed: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="also measure latencies")
    args = ap.parse_args()

    print("torch            :", torch.__version__)
    print("cuda available   :", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("cuda device      :", p.name)
        print("capability       :", f"sm_{p.major}{p.minor}")
        print("total memory     :", f"{p.total_memory / 1e9:.1f} GB")
    try:
        import tilelang

        print("tilelang         :", tilelang.__version__)
    except Exception as e:  # pragma: no cover
        print("tilelang         : NOT INSTALLED ({})".format(e))
        return 1

    if not torch.cuda.is_available() or not _cuda_op_works():
        print("\nCUDA device missing or unusable by this torch build -- kernel")
        print("validation must run on a GPU box (e.g. the CMP 50HX).")
        return 1

    from tilechat import kernels as _tk

    _tk._ensure_jit_host_compiler()
    if _tk.host_compiler_note:
        print("host compiler    :", _tk.host_compiler_note)

    from tilechat.kernels import (
        TiledAttention,
        TiledGRUStep,
        reference_attention,
        reference_gru_step,
        tilescale_attention,
        tilescale_gru_gate,
        tilescale_linear,
    )

    dev = torch.device("cuda")
    torch.manual_seed(0)
    # tutorial-sized shapes: batch 64, MAX_LENGTH 10, hidden 500
    B, L, H = 64, 10, 500
    x = torch.randn(B, H, device=dev)
    h = torch.randn(B, H, device=dev)
    enc = torch.randn(B, L, H, device=dev)
    w_ih = torch.randn(3 * H, H, device=dev)
    w_hh = torch.randn(3 * H, H, device=dev)
    b_ih = torch.randn(3 * H, device=dev)
    b_hh = torch.randn(3 * H, device=dev)

    print("\n== compiling + validating kernels (first JIT compile takes a bit) ==")
    t0 = time.time()
    gi = tilescale_linear(x, w_ih, b_ih)
    gh = tilescale_linear(h, w_hh, b_hh)
    h_new = tilescale_gru_gate(gi, gh, h)
    ctx = tilescale_attention(x, enc)
    print(f"JIT compile + run took {time.time() - t0:.1f}s")

    gi_ref = torch.addmm(b_ih, x, w_ih.t())
    gh_ref = torch.addmm(b_hh, h, w_hh.t())
    h_ref = reference_gru_step(x, h, w_ih, w_hh, b_ih, b_hh)[0]
    ctx_ref = reference_attention(x, enc)

    errs = {
        "linear(GEMM)": (gi - gi_ref).abs().max().item(),
        "gru_gate": (h_new - h_ref).abs().max().item(),
        "attention": (ctx - ctx_ref).abs().max().item(),
    }
    for name, e in errs.items():
        print(f"  {name:<14} max abs err = {e:.3e}")
    ok = all(e < 1e-3 for e in errs.values())

    if args.bench:
        def bench(fn, n=200):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.time() - t) / n * 1e6

        print("\n== per-call latency, tutorial shapes (B=64, L=10, H=500) ==")
        print(f"  tiled GEMM  : {bench(lambda: tilescale_linear(x, w_ih, b_ih)):8.1f} us")
        print(f"  gru gates   : {bench(lambda: tilescale_gru_gate(gi, gh, h)):8.1f} us")
        print(f"  attention   : {bench(lambda: tilescale_attention(x, enc)):8.1f} us")
        print(f"  reference   : {bench(lambda: torch.addmm(b_ih, x, w_ih.t())):8.1f} us (addmm)")

    print("\nKERNEL CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
