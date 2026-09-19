"""End-to-end pipeline test: corpus -> train a few steps -> greedy decode.

Downloads the Cornell corpus once into ./data (shared with the CLI, ~62 MB).
Set TILECHAT_SKIP_DOWNLOAD=1 to skip this test on offline machines.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(
    os.environ.get("TILECHAT_SKIP_DOWNLOAD") == "1",
    reason="TILECHAT_SKIP_DOWNLOAD=1 (offline)",
)


def test_end_to_end_tiny_training(tmp_path):
    from tilechat.data import batch2TrainData, loadPrepareData, trimRareWords
    from tilechat.models import EncoderRNN, GreedySearchDecoder, LuongAttnDecoderRNN, device
    from tilechat.training import maskNLLLoss, train

    # 1) data pipeline (downloads + wrangles the corpus on first run)
    voc, pairs = loadPrepareData()
    assert voc.num_words > 1000
    pairs = trimRareWords(voc, pairs)
    assert len(pairs) > 10000

    # 2) tiny models, PyTorch fallback backend (this test must run on CPU)
    hidden = 100
    embedding = torch.nn.Embedding(voc.num_words, hidden)
    encoder = EncoderRNN(voc.num_words, hidden, 1, dropout=0.1).to(device)
    decoder = LuongAttnDecoderRNN(
        "dot", embedding, hidden, voc.num_words, n_layers=1, dropout=0.1, backend="pytorch"
    ).to(device)
    enc_opt = torch.optim.ASGD(encoder.parameters(), lr=0.0001)
    dec_opt = torch.optim.ASGD(decoder.parameters(), lr=0.0005)

    # 3) a few teacher-forced iterations, loss must be finite and decrease
    batch = batch2TrainData(voc, [pairs[i] for i in range(32)])
    inp, lengths, target, mask, max_target_len = batch
    losses = []
    for _ in range(3):
        loss, _, _ = train(
            inp, lengths, target, mask, max_target_len,
            encoder, decoder, embedding, enc_opt, dec_opt, 32, 50.0,
        )
        assert torch.isfinite(loss)
        losses.append(loss.item())

    # 4) greedy decode smoke test (teacher-forced path above vs search path)
    searcher = GreedySearchDecoder(encoder, decoder)
    from tilechat.data import indexesFromSentence

    sentence = pairs[0][0]
    indexes = [indexesFromSentence(voc, sentence)]
    lengths_t = torch.tensor([len(indexes[0])])
    input_batch = torch.LongTensor(indexes).transpose(0, 1).to(device)
    tokens, scores = searcher(input_batch, lengths_t, max_length=10)
    assert tokens.shape[0] == 10
    words = [voc.index2word[int(t)] for t in tokens]
    print("input:", sentence, "-> decoded:", " ".join(w for w in words if w not in ("EOS", "PAD")))

    # 5) held-out validation loss must be finite and positive
    from tilechat.training import validate

    val_loss = validate(pairs[:128], voc, encoder, decoder, embedding, batch_size=32)
    assert torch.isfinite(torch.tensor(val_loss)) and val_loss > 0
    print("validation loss (untrained tiny model):", round(val_loss, 3))


def test_greedy_decoder_temperature_modes():
    """temperature=0 must stay exact argmax; temperature>0 must sample."""
    from tilechat.models import EncoderRNN, GreedySearchDecoder, LuongAttnDecoderRNN, device

    torch.manual_seed(0)
    V, H, L = 50, 16, 7
    embedding = torch.nn.Embedding(V, H)
    encoder = EncoderRNN(V, H, 1, dropout=0.0).to(device).eval()
    decoder = LuongAttnDecoderRNN(
        "dot", embedding, H, V, n_layers=1, dropout=0.0, backend="pytorch"
    ).to(device).eval()
    seq = torch.randint(3, V, (L, 1), device=device)
    lengths = torch.tensor([L])

    # 1) default and temperature ~0 both reproduce the tutorial's argmax walk
    greedy_a, _ = GreedySearchDecoder(encoder, decoder)(seq, lengths, max_length=6)
    greedy_b, _ = GreedySearchDecoder(encoder, decoder, temperature=1e-6)(seq, lengths, max_length=6)
    assert torch.equal(greedy_a, greedy_b)

    # 2) seeded sampling is reproducible ...
    s1a, _ = GreedySearchDecoder(encoder, decoder, temperature=1.0, seed=123)(seq, lengths, max_length=6)
    s1b, _ = GreedySearchDecoder(encoder, decoder, temperature=1.0, seed=123)(seq, lengths, max_length=6)
    assert torch.equal(s1a, s1b)

    # 3) ... and a different seed explores a different path (6 draws from a
    #    near-uniform 50-way distribution colliding is vanishingly unlikely)
    s2, _ = GreedySearchDecoder(encoder, decoder, temperature=1.0, seed=124)(seq, lengths, max_length=6)
    assert not torch.equal(s1a, s2)


def test_beam_search_decoder_matches_greedy_at_width_1():
    """Beam width 1 must equal greedy; width 5 must be deterministic."""
    from tilechat.models import (
        BeamSearchDecoder,
        EncoderRNN,
        GreedySearchDecoder,
        LuongAttnDecoderRNN,
        device,
    )

    torch.manual_seed(0)
    V, H, L = 50, 16, 7
    embedding = torch.nn.Embedding(V, H)
    encoder = EncoderRNN(V, H, 1, dropout=0.0).to(device).eval()
    decoder = LuongAttnDecoderRNN(
        "dot", embedding, H, V, n_layers=1, dropout=0.0, backend="pytorch"
    ).to(device).eval()
    seq = torch.randint(3, V, (L, 1), device=device)
    lengths = torch.tensor([L])

    greedy, _ = GreedySearchDecoder(encoder, decoder)(seq, lengths, max_length=6)
    # width 1 without blocking must reproduce the greedy walk exactly
    beam1, _ = BeamSearchDecoder(encoder, decoder, beam_width=1, no_repeat_trigram=False)(
        seq, lengths, max_length=6
    )
    assert torch.equal(greedy, beam1)

    # with blocking on, the output must not contain any repeated trigram
    # (even when greedy itself loops, e.g. 22 22 22 ...)
    blocked, _ = BeamSearchDecoder(encoder, decoder, beam_width=1)(seq, lengths, max_length=6)
    words = blocked.view(-1).tolist()
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    assert len(trigrams) == len(set(trigrams))

    beam5a, scores = BeamSearchDecoder(encoder, decoder, beam_width=5)(seq, lengths, max_length=6)
    beam5b, _ = BeamSearchDecoder(encoder, decoder, beam_width=5)(seq, lengths, max_length=6)
    assert torch.equal(beam5a, beam5b)  # deterministic (no sampling involved)
    assert beam5a.shape[1] == 1
    assert beam5a.shape[0] == scores.shape[0] and beam5a.shape[0] <= 6
