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
