# tilechat — PyTorch Chatbot Tutorial, rewritten with TileScale (TileLang) kernels

This is a rewrite of the official
[PyTorch chatbot tutorial](https://docs.pytorch.org/tutorials/beginner/chatbot_tutorial.html)
(EncoderRNN + LuongAttnDecoderRNN over the Cornell Movie-Dialogs corpus) in
which the decoder's per-token hot path is implemented with hand-written
**TileLang** kernels — the kernel language shipped by
[tile-ai/tilescale](https://github.com/tile-ai/tilescale) (`import tilelang`).

The tutorial's structure, data pipeline, loss, teacher forcing and greedy
decoding are all preserved. What changed is *where the decoder math runs*:

| tutorial (original)                        | this repo                                                            |
| ------------------------------------------ | -------------------------------------------------------------------- |
| `nn.GRU(H, H)` cell step in the decoder     | 2 tiled GEMM kernels + 1 fused gate kernel (`TiledGRUStep`)          |
| `Attn("dot")` scores → `F.softmax` → `bmm`  | 1 fused attention kernel (`TiledAttention`)                          |
| cuDNN/cuBLAS ops behind autograd            | explicit kernels behind a custom `torch.autograd.Function` (analytic backward, verified against autograd in the tests) |

Everything keeps a **pure-PyTorch fallback backend**, so the code also runs
on CPU-only machines (and on GPUs where the kernels are unavailable).

## Contents

1. [Install](#install)
2. [Data](#1-data--the-cornell-movie-dialogs-corpus)
3. [The kernels](#2-the-three-tilelang-kernels)
4. [Models](#3-models-encoder--kernel-decoder)
5. [Training](#4-training)
6. [Evaluation / chat](#5-evaluation--chatting)
7. [Running](#running)
8. [Testing](#testing)
9. [Running on a CMP 50HX (10 GB)](#running-on-a-cmp-50hx-10-gb)

---

## Install

Requires Python ≥ 3.10, PyTorch ≥ 2.0, and a CUDA toolkit (`nvcc`) on the
GPU box — TileLang JIT-compiles kernels at runtime.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch tilelang pytest      # or: bash scripts/install_tilescale.sh
```

Two ways to get the kernel runtime:

* **Upstream TileLang wheel** (`pip install tilelang`, default). TileScale
  keeps TileLang's Python kernel language, compiler, JIT and cache
  interfaces — ordinary TileLang programs (these kernels included) run
  unchanged on either.
* **TileScale from source** (`INSTALL_TILESCALE_SOURCE=1 bash scripts/install_tilescale.sh`)
  adds the single-host multi-GPU distributed runtime; needs a C++ toolchain
  and `git clone --recursive`.

Verify the environment and (on a GPU box) the kernels:

```bash
python -m tilechat check            # env report + kernel-vs-reference check
python scripts/check_tilescale.py --bench   # + per-kernel latencies
```

---

## 1) Data — the Cornell Movie-Dialogs corpus

(`tilechat/data.py`, faithful port of the tutorial's data sections.)

The corpus is downloaded and extracted automatically on first use (~62 MB,
cached under `data/`):

1. **Wrangle** — parse `movie_lines.txt` and `movie_conversations.txt`
   (`+++ $ +++` separated), and extract consecutive utterance pairs into
   `data/formatted_movie_lines.txt` (`extractSentencePairs`).
2. **Normalize** — `unicodeToAscii` + lowercase + isolate `.!?` + drop
   non-letters (`normalizeString`).
3. **Vocabulary** — the tutorial's `Voc` class maps words ↔ indexes
   (`PAD=0`, `SOS=1`, `EOS=2`, words from 3).
4. **Filter + trim** — keep pairs with both sentences shorter than
   `MAX_LENGTH = 10` words, then `trimRareWords(MIN_COUNT=3)` drops words
   seen < 3 times and any pair containing them.
5. **Batch** — `batch2TrainData` sorts a random pair-batch by input length
   (descending, required by `pack_padded_sequence`), zero-pads to a
   `(L, B)` LongTensor and produces a boolean padding mask + target tensor.

Result on the stock corpus: ~64k training pairs, ~18k vocabulary words
(after trimming) — same as the tutorial.

## 2) The three TileLang kernels

(`tilechat/kernels.py`.) All float32, all JIT-compiled per shape via
`@tilelang.jit` + `fn.compile(...)` with a compile-once cache. Only core
TileLang constructs are used: `T.const` dims, `T.Tensor` annotations,
`T.Kernel`, `T.get_thread_binding`, `T.alloc_shared/local/fragment`,
`T.clear`, `T.serial`, `T.Parallel`, `T.copy`, `T.exp`, `T.max`.

### a) Tiled GEMM — `_linear_kernel`

`C = A @ W.T + bias` for `A (B, K)`, `W (N, K)` — the decoder GRU's input
and hidden projections. Classic SIMT tiling: `A`- and `W`-tiles stream
through shared memory, a register `T.alloc_fragment` accumulator runs the
FMA loop per output element, bias is added on the store.

Shapes are **zero-padded** to multiples of the fixed tile `16×64×32`
(tilelang's layout inference requires every shared tile to divide evenly
across the 128 threads; padding is exact and sliced away afterwards).

### b) Fused GRU gates — `_gru_gate_kernel`

Given the input/hidden projections `GI, GH (B, 3H)` and `h_prev (B, H)`,
one flat thread mapping computes, for every hidden unit:

```
r  = sigmoid(GI_r + GH_r)
z  = sigmoid(GI_z + GH_z)
n  = tanh(GI_n + r ⊙ GH_n)        # written via exp()
h' = (1 − z) ⊙ n + z ⊙ h_prev
```

i.e. the whole non-GEMM part of a GRU cell collapses into a single
elementwise kernel launch.

### c) Fused Luong dot-attention — `_luong_attention_kernel`

The tutorial computes attention scores with a bmm, softmaxes them, and
bmms the weights against the encoder outputs — three kernel launches plus
intermediates. This kernel does it in **one launch**: one CUDA block per
batch element,

1. the first `L` threads dot `q` against each encoder output row
   (scores in shared memory),
2. thread 0 runs a numerically-stable softmax over the `L` scores
   (max-subtraction like any production softmax),
3. the first `H` threads accumulate `Σ_l w[l] · enc[l, h]` into the output.

TileLang auto-inserts the `__syncthreads()` barriers between the
shared-memory stages (visible in the generated CUDA).

### Autograd

The kernels cover the **forward** pass; the backward passes are expressed
analytically with standard PyTorch ops inside two `torch.autograd.Function`s
(`TiledGRUStep`, `TiledAttention`) — including the subtle attention term
many hand-written backwards miss: `encoder_outputs` receives gradient both
through the weighted sum **and** through the softmax scores
(`denc += ds ⊗ q`). Both Functions are tested against autograd exactly.

## 3) Models — encoder + kernel decoder

(`tilechat/models.py`, structure preserved from the tutorial.)

* `EncoderRNN` — unchanged: embedding → bidirectional GRU (cuDNN handles
  sequence-level work efficiently) with the summed-directions reduction.
* `LuongAttnDecoderRNN` — same interface/params as the tutorial, but the
  GRU cell is a `_GRUStep` module (two `_linear_kernel` GEMMs + the gate
  kernel) and attention runs in the fused kernel. Weight init matches
  `nn.GRU` defaults and `decoder(input_step, last_hidden, encoder_outputs)`
  → `(softmax over vocab, next_hidden)` exactly like before.
* `GreedySearchDecoder` — the tutorial's greedy loop, unchanged.

Backend selection (`--backend auto|tilescale|pytorch`, or env
`TILECHAT_BACKEND`): `auto` uses the kernels for CUDA float32 tensors and
the PyTorch reference everywhere else; `pytorch` always uses the reference;
`tilescale` demands them. The multi-layer decoder falls back to the
reference path (the kernels implement the single-layer cell).

## 4) Training

(`tilechat/training.py`, verbatim port of the tutorial's procedure.)

* `maskNLLLoss` — negative log-likelihood averaged over non-PAD positions.
* `train(...)` — zero grads → encoder forward → SOS-seeded decoder loop for
  `max_target_len` steps (teacher forcing on by default, matching
  `TEACHER_FORCING_RATIO = 1.0`) → accumulate masked NLL → backward →
  `clip_grad_norm_(50.0)` on both models → ASGD steps (the tutorial's
  optimizer; decoder LR × 5).
* `trainIters(...)` — random-pair batches, progress prints, periodic
  checkpoints under `save/cb_model/...`.

Tutorial hyper-parameters are the defaults: `hidden_size=500`,
`encoder_n_layers=2`, `decoder_n_layers=1`, `dropout=0.1`, `batch_size=64`,
`clip=50`, `lr=1e-4`.

## 5) Evaluation — chatting

* `evaluate(encoder, decoder, searcher, voc, sentence)` — normalizes,
  indexes, and greedy-decodes one sentence.
* `evaluateInput(...)` — the tutorial's REPL: type a sentence, get a
  reply, `q`/`quit` to exit.

## Running

```bash
# train (kernels used automatically on CUDA; reference path on CPU)
python -m tilechat train --iterations 4000

# chat with the trained model
python -m tilechat chat --checkpoint save/cb_model/final_checkpoint.tar

# force a backend
TILECHAT_BACKEND=pytorch python -m tilechat train
python -m tilechat train --backend tilescale
```

Useful flags: `--hidden-size`, `--encoder-layers`, `--decoder-layers`,
`--batch-size`, `--clip`, `--learning-rate`, `--print-every`,
`--save-every`, `--save-dir`. Env: `TILECHAT_DEVICE=cpu|cuda` forces the
device (handy when a GPU exists that the torch build can't use),
`TILECHAT_DATA_DIR`, `TILECHAT_SAVE_DIR`, `TILECHAT_CORPUS_URL`.

## Testing

```bash
python -m pytest tests/ -q
```

* `tests/test_math.py` — CPU-safe: the PyTorch references match
  `nn.GRUCell` exactly; both autograd Functions produce gradients equal to
  autograd (this caught a real attention-backward bug during development);
  backend policy behaves as documented.
* `tests/test_kernels.py` — GPU-only: all three kernels vs references,
  forward **and** backward, on multiple shapes; auto-skipped without a
  usable CUDA device + tilelang.
* `tests/test_pipeline.py` — end-to-end: corpus → vocab → 3 training
  iterations (finite loss) → greedy decode.

## Running on a CMP 50HX (10 GB)

The CMP 50HX (TU102, Turing, sm_75, 10 GB) is comfortably within reach for
this workload: the tutorial model is tiny (hidden 500, batch 64,
`MAX_LENGTH` 10) and fp32 training fits in a few GB of device memory.

The quickest path is the one-shot installer, which is safe to re-run:

```bash
bash scripts/install_50hx.sh
```

It checks the driver and compute capability (torch cu130 wheels need
sm ≥ 7.5), checks `nvcc`, sets up the venv and installs
`requirements.txt` (reinstalling torch from the cu130 index if the PyPI
build somehow lacks CUDA), then probes the toolchain the same way the JIT
does — `nvcc -ccbin=<c++>` — and reports which host compiler was picked
(tilechat auto-switches to an older `/usr/bin/g++-N` when nvcc rejects the
default one; override with `CXX`/`CC`). Finally it runs the full
validation chain below including the strict `TILECHAT_BACKEND=tilescale`
check.

Manual equivalent:

1. Install CUDA-capable torch + `nvcc` (TileLang needs `nvcc` in `PATH`
   at first JIT).
2. `pip install -r requirements.txt`
3. `python scripts/check_tilescale.py --bench` — compiles all kernels for
   sm_75, validates them against the PyTorch references and prints
   per-kernel latencies.
4. `pytest tests/ -q` — the GPU kernel tests now run (not skip).
5. `python -m tilechat train` — trains with the TileLang kernels on the
   decoder hot path.

Notes for the 50HX run:

* **First iterations are slow** (JIT compiles a handful of
  shape-specialized kernels; cached afterwards).
* tilelang compiles with `nvcc -ccbin=<host c++>`; nvcc rejects host
  compilers newer than it supports (seen in the wild: nvcc 12.4 vs
  g++ 15). Before the first JIT, tilechat probes that exact command and
  auto-switches `CXX` to the newest compatible `/usr/bin/g++-N`
  (`sudo apt install g++-13` provides one); setting `CXX`/`CC` yourself
  overrides the probe entirely.
* Multi-GPU boxes: cuDNN ≥ 9.11 refuses to run in a process that can see
  *any* GPU older than sm_75 — the tutorial encoder's cuDNN GRU then dies
  with "cuDNN version … is not compatible with devices with SM < 7.5".
  Importing `tilechat` therefore auto-hides sub-sm_75 GPUs from
  `CUDA_VISIBLE_DEVICES` before CUDA initializes (`tilechat/cuda_guard.py`,
  prints a notice when it hides something). An explicit
  `CUDA_VISIBLE_DEVICES` in the environment wins; set
  `TILECHAT_DISABLE_GPU_FILTER=1` to opt out. The install script exports
  the same filtered value for interactive shells.
* `TILECHAT_BACKEND=tilescale python -m tilechat check` is a strict check
  (errors instead of silently falling back).
* The 10 GB leaves headroom for `--batch-size 128` or
  `--hidden-size 768` experiments.

## Notes & credits

* Original tutorial: PyTorch team — *Chatbot Tutorial* (Sung Kim, Jong
  Wook Kim et al.).
* Kernel language: [tile-ai/tilelang](https://github.com/tile-ai/tilelang)
  / [tile-ai/tilescale](https://github.com/tile-ai/tilescale) (Apache-2.0).
* Related: [tile-ai/TileOPs](https://github.com/tile-ai/TileOPs) (MIT) — a
  spec-driven LLM operator library built on TileLang (GEMM, attention, …;
  auto-tuned, CUDA-Graph compatible, fp16/bf16). Worth reading for how
  production TileLang kernels are structured (manifest/spec discipline,
  roofline-scored benchmarks). **Not a dependency here**: it installs from
  source, targets compute-capability **9.0 (Hopper)** GPUs with CUDA
  Toolkit 13.2, so its ops cannot execute on the CMP 50HX (sm_75) — and
  this repo's kernels are fp32 for tutorial fidelity, whereas TileOPs is
  fp16/bf16. Revisit it when porting these kernels to an SM90 GPU or when
  switching the pipeline to half precision.
* Deviations from the tutorial: `ast.literal_eval` instead of `eval` in
  the conversation parser; field lists passed as ordered lists; training
  batches drawn lazily per iteration (identical distribution); checkpoints
  add a `config` block; ASGD optimizers kept from the tutorial.
