"""Command line entry points: train / chat / check.

Examples
--------
Train with TileScale kernels on CUDA (falls back to the PyTorch reference
backend automatically elsewhere)::

    python -m tilechat train --iterations 4000

Chat with a trained checkpoint::

    python -m tilechat chat --checkpoint save/cb_model/final_checkpoint.tar

Sanity-check the kernels against the PyTorch references on whatever device
is available::

    python -m tilechat check
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from .data import Voc, loadPrepareData, trimRareWords
from .kernels import tilelang_available
from .models import (
    ATTN_MODEL,
    DECODER_N_LAYERS,
    DROPOUT,
    ENCODER_N_LAYERS,
    HIDDEN_SIZE,
    MODEL_NAME,
    EncoderRNN,
    GreedySearchDecoder,
    LuongAttnDecoderRNN,
    device,
)
from .training import (
    CLIP,
    DECODER_LEARNING_RATIO,
    LEARNING_RATE,
    PRINT_EVERY,
    SAVE_EVERY,
    evaluateInput,
    trainIters,
)

DEFAULT_SAVE_DIR = os.environ.get(
    "TILECHAT_SAVE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "save"),
)


def _make_models(voc, args):
    embedding = nn.Embedding(voc.num_words, args.hidden_size)
    encoder = EncoderRNN(voc.num_words, args.hidden_size, args.encoder_layers, dropout=args.dropout)
    decoder = LuongAttnDecoderRNN(
        args.attn_model, embedding, args.hidden_size, voc.num_words,
        n_layers=args.decoder_layers, dropout=args.dropout, backend=args.backend,
    )
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    return embedding, encoder, decoder


def cmd_train(args):
    voc, pairs = loadPrepareData()
    pairs = trimRareWords(voc, pairs)
    embedding, encoder, decoder = _make_models(voc, args)

    # Initialize optimizers (the tutorial uses ASGD)
    print("Building optimizers ...")
    encoder_optimizer = torch.optim.ASGD(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = torch.optim.ASGD(
        decoder.parameters(), lr=args.learning_rate * args.decoder_learning_ratio
    )

    print("Models built! (device: {}, backend policy: {})".format(device, args.backend))

    trainIters(
        MODEL_NAME, voc, pairs, encoder, decoder,
        encoder_optimizer, decoder_optimizer, embedding,
        args.encoder_layers, args.decoder_layers, args.save_dir,
        args.iterations, args.batch_size, args.print_every, args.save_every, args.clip,
        "cornell movie-dialogs corpus",
    )

    # Stash a checkpoint under the save dir root for easy chatting
    final_path = os.path.join(args.save_dir, MODEL_NAME, "final_checkpoint.tar")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(
        {
            "iteration": args.iterations,
            "en": encoder.state_dict(),
            "de": decoder.state_dict(),
            "emb": embedding.state_dict(),
            "voc": voc.__dict__,
            "config": {
                "hidden_size": args.hidden_size,
                "encoder_layers": args.encoder_layers,
                "decoder_layers": args.decoder_layers,
                "dropout": args.dropout,
                "attn_model": args.attn_model,
            },
        },
        final_path,
    )
    print("Saved final checkpoint to", final_path)


def cmd_chat(args):
    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"checkpoint not found: {args.checkpoint}\n"
            "train first: python -m tilechat train"
        )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    voc_dict = checkpoint["voc"]
    config = checkpoint.get("config", {})

    # Rebuild the vocabulary exactly as stored
    voc = Voc("cornell movie-dialogs corpus")
    voc.__dict__.update(voc_dict)

    args.hidden_size = config.get("hidden_size", HIDDEN_SIZE)
    args.encoder_layers = config.get("encoder_layers", ENCODER_N_LAYERS)
    args.decoder_layers = config.get("decoder_layers", DECODER_N_LAYERS)
    args.dropout = config.get("dropout", DROPOUT)
    args.attn_model = config.get("attn_model", ATTN_MODEL)

    embedding, encoder, decoder = _make_models(voc, args)
    embedding.load_state_dict(checkpoint["emb"])
    encoder.load_state_dict(checkpoint["en"])
    decoder.load_state_dict(checkpoint["de"])
    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)
    print("Chat away! ('q' or 'quit' to exit)")
    evaluateInput(encoder, decoder, searcher, voc)


def cmd_check(args):
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
    print("tilelang (TileScale):", "installed" if tilelang_available() else "NOT installed")
    print("resolved device:", device)

    from .kernels import reference_attention, reference_gru_step

    gen = torch.Generator().manual_seed(7)
    B, L, H = args.batch_size, args.seq_len, args.hidden_size
    x = torch.randn(B, H, generator=gen)
    h = torch.randn(B, H, generator=gen)
    enc = torch.randn(B, L, H, generator=gen)

    if device.type != "cuda":
        print("No usable CUDA device -> running reference backend only (CPU).")
        ref_h, gi, gh, r, z, n = reference_gru_step(x, h, torch.randn(3 * H, H), torch.randn(3 * H, H),
                                                    torch.randn(3 * H), torch.randn(3 * H))
        ref_ctx = reference_attention(x, enc)
        print("reference GRU step:", tuple(ref_h.shape), "| reference context:", tuple(ref_ctx.shape))
        print("OK (CPU path). Run on a CUDA machine (e.g. the CMP 50HX) to exercise the kernels.")
        return

    # Kernel vs reference on CUDA
    dev = torch.device("cuda")
    x, h, enc = x.to(dev), h.to(dev), enc.to(dev)
    w_ih = torch.randn(3 * H, H, generator=gen).to(dev)
    w_hh = torch.randn(3 * H, H, generator=gen).to(dev)
    b_ih = torch.randn(3 * H, generator=gen).to(dev)
    b_hh = torch.randn(3 * H, generator=gen).to(dev)

    from .kernels import TiledAttention, TiledGRUStep

    h_kernel = TiledGRUStep.apply(x, h, w_ih, w_hh, b_ih, b_hh, True)
    h_ref = reference_gru_step(x, h, w_ih, w_hh, b_ih, b_hh)[0]
    err_gru = (h_kernel - h_ref).abs().max().item()

    ctx_kernel = TiledAttention.apply(x, enc, True)
    ctx_ref = reference_attention(x, enc)
    err_attn = (ctx_kernel - ctx_ref).abs().max().item()

    print(f"GRU step   max|kernel - reference| = {err_gru:.3e}")
    print(f"attention  max|kernel - reference| = {err_attn:.3e}")
    ok = err_gru < 1e-3 and err_attn < 1e-3
    print("KERNEL CHECK", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


def build_parser():
    p = argparse.ArgumentParser(prog="tilechat", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", default="auto", choices=["auto", "tilescale", "pytorch"],
                        help="kernel policy (auto = kernels on CUDA fp32, PyTorch elsewhere)")
    common.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    common.add_argument("--encoder-layers", type=int, default=ENCODER_N_LAYERS)
    common.add_argument("--decoder-layers", type=int, default=DECODER_N_LAYERS)
    common.add_argument("--dropout", type=float, default=DROPOUT)
    common.add_argument("--attn-model", default=ATTN_MODEL, choices=["dot"])

    p_train = sub.add_parser("train", parents=[common], help="train the chatbot")
    p_train.add_argument("--iterations", type=int, default=4000)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--clip", type=float, default=CLIP)
    p_train.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p_train.add_argument("--decoder-learning-ratio", type=float, default=DECODER_LEARNING_RATIO)
    p_train.add_argument("--print-every", type=int, default=PRINT_EVERY)
    p_train.add_argument("--save-every", type=int, default=SAVE_EVERY)
    p_train.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    p_train.set_defaults(func=cmd_train)

    p_chat = sub.add_parser("chat", parents=[common], help="chat with a trained checkpoint")
    p_chat.add_argument("--checkpoint", required=True)
    p_chat.set_defaults(func=cmd_chat)

    p_check = sub.add_parser("check", parents=[common], help="verify TileScale kernels vs references")
    p_check.add_argument("--batch-size", type=int, default=64)
    p_check.add_argument("--seq-len", type=int, default=10)
    p_check.set_defaults(func=cmd_check)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
