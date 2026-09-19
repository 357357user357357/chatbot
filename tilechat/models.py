"""Models: EncoderRNN, Luong-attention decoder with TileScale kernels, greedy search.

The encoder is the tutorial's bidirectional GRU unchanged (sequence-level
work, already efficient in cuDNN).  The *decoder* runs one GRU cell step and
one Luong dot-attention per generated token -- that inner loop is rewritten
here with the TileLang kernels from ``tilechat.kernels``:

    nn.GRU(H, H) step     ->  TiledGRUStep  (2x _linear_kernel + _gru_gate_kernel)
    Attn("dot") + context ->  TiledAttention (_luong_attention_kernel)

The decoder keeps the tutorial's interface, loss path and number of
parameters, so training code is a drop-in.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MAX_LENGTH, SOS_token
from .kernels import TiledAttention, TiledGRUStep, kernels_usable

# The tutorial's config (model/config section)
MODEL_NAME = "cb_model"
ATTN_MODEL = "dot"
HIDDEN_SIZE = 500
ENCODER_N_LAYERS = 2
DECODER_N_LAYERS = 1
DROPOUT = 0.1


def _resolve_use_kernel(tensor, backend: str) -> bool:
    """Map --backend {auto,tilescale,pytorch} to the kernel switch."""
    pref = (backend or "auto").strip().lower()
    if pref == "pytorch":
        return False
    if pref == "tilescale":
        return kernels_usable(tensor)  # raises if it cannot be honoured
    return kernels_usable(tensor)  # auto


class _GRUStep(nn.Module):
    """The decoder's single-layer GRU cell, evaluated by TiledGRUStep.

    Holds the same parameters as ``nn.GRU(H, H)`` (weight_ih, weight_hh,
    bias_ih, bias_hh with r/z/n gate ordering) with PyTorch's default
    initialization, so the module is a drop-in replacement for the tutorial's
    ``self.gru`` decoder GRU.
    """

    def __init__(self, hidden_size: int, backend: str = "auto"):
        super().__init__()
        self.backend = backend
        stdv = 1.0 / (hidden_size ** 0.5)
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        for param in (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh):
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, embedded: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # embedded, hidden: (1, B, H) -- seq-first, one time step
        use_kernel = _resolve_use_kernel(embedded, self.backend)
        h_new = TiledGRUStep.apply(
            embedded.squeeze(0),
            hidden.squeeze(0),
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            use_kernel,
        )
        return h_new.unsqueeze(0)


class _DotAttn(nn.Module):
    """Luong dot attention, fused into one TileLang kernel by TiledAttention.

    Equivalent to the tutorial's ``Attn(method='dot')`` scores
    (sum(hidden * encoder_outputs, dim=2)) followed by softmax and the
    context bmm -- but computed in a single kernel launch.
    """

    def __init__(self, backend: str = "auto"):
        super().__init__()
        self.backend = backend

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        # hidden (1, B, H) -> q (B, H); encoder_outputs (L, B, H) -> (B, L, H)
        q = hidden.squeeze(0)
        enc_bf = encoder_outputs.transpose(0, 1)
        use_kernel = _resolve_use_kernel(q, self.backend)
        context = TiledAttention.apply(q, enc_bf, use_kernel)  # (B, H)
        return context.unsqueeze(1)  # (B, 1, H), tutorial decoder expects this


class EncoderRNN(nn.Module):
    """Tutorial encoder: embedding + bidirectional GRU, halves summed."""

    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Initialize GRU; the input_size and hidden_size params are both equal
        # to hidden_size because our input size is a word embedding with
        # num_features = hidden_size
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True,
        )

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.to("cpu"))
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        # Return output and final hidden state
        return outputs, hidden


class LuongAttnDecoderRNN(nn.Module):
    """Tutorial decoder with the per-token work done by TileScale kernels.

    Same interface, parameter set and math as the tutorial's
    ``LuongAttnDecoderRNN`` (embedding dropout -> GRU step -> dot attention ->
    concat(tanh) -> softmax output), except:

    * the GRU cell is a ``_GRUStep`` running two tiled GEMM kernels plus the
      fused gate kernel,
    * attention scores + softmax + context run in the single fused
      ``_luong_attention_kernel`` launch.

    ``backend`` selects the kernel policy: ``"auto"`` (default) uses the
    kernels on CUDA float32 tensors and the PyTorch reference everywhere
    else; ``"tilescale"`` forces them; ``"pytorch"`` disables them.
    """

    def __init__(self, attn_model, embedding, hidden_size, output_size,
                 n_layers=1, dropout=0.1, backend="auto"):
        super(LuongAttnDecoderRNN, self).__init__()
        # Keep for reference (e.g. GreedySearchDecoder slicing)
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        # The TileScale cell implements the single-layer GRU; deeper stacks
        # fall back to the tutorial path.
        if n_layers != 1:
            backend = "pytorch"
        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = _GRUStep(hidden_size, backend)
        self.attn = _DotAttn(backend)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Initialize word embeddings?  No: the tutorial initializes the
        # embedding once at the top level and shares it, keep as passed in.

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time.
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        # Forward through unidirectional GRU
        rnn_output = self.gru(embedded, last_hidden)  # (1, B, H)
        # Calculate attention weights from the current GRU output
        # (fused kernel: scores -> softmax -> context)
        context = self.attn(rnn_output, encoder_outputs)  # (B, 1, H)
        # Attention weighting using bmm is done inside the kernel above; the
        # tutorial additionally returned attn_weights for debugging:
        #   attn_weights = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Forward attention-weighted GRU output through linear layers
        rnn_output = rnn_output.squeeze(0)  # (B, H)
        context = context.squeeze(1)        # (B, H)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # GRU hidden state for the next step equals the cell output
        hidden = rnn_output.unsqueeze(0)
        # Return output and final hidden state
        return output, hidden


class GreedySearchDecoder(nn.Module):
    """Greedy decoding loop from the tutorial (uses the decoder above)."""

    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length=MAX_LENGTH):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be the first hidden input to
        # the decoder
        decoder_hidden = encoder_hidden[: self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input.view(1, -1)), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores.view(1, -1)), dim=0)
            # Prepare current token to be next decoder input
            decoder_input = decoder_input.view(1, -1)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


# Global device used by the greedy decoder and the training code (the
# tutorial defines this at module level).  models.py is imported by
# training.py, so this is the single definition.
#
# Honours ``TILECHAT_DEVICE`` (e.g. TILECHAT_DEVICE=cpu) and probes that the
# CUDA device actually works before selecting it -- some boxes expose a GPU
# whose compute capability the installed torch build does not support.
def _pick_device() -> torch.device:
    override = os.environ.get("TILECHAT_DEVICE", "").strip().lower()
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return torch.device("cuda")
        except Exception:
            print("[tilechat] CUDA device present but unusable by this torch build; using CPU")
    return torch.device("cpu")


device = _pick_device()

