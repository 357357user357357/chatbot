"""Training / evaluation loop, ported from the tutorial's "Define Training
Procedure", "Training Run" and "Evaluate" sections.

Only changes vs the original:
* ``device`` is imported from ``tilechat.models`` instead of a global.
* the decoder is the TileScale-kernel one (the loop itself is untouched --
  it just calls ``decoder(input_step, hidden, encoder_outputs)``).
"""

from __future__ import annotations

import os
import random

import torch
import torch.nn as nn

from .data import (
    MAX_LENGTH,
    SOS_token,
    batch2TrainData,
    indexesFromSentence,
    normalizeString,
)
from .models import (
    DECODER_N_LAYERS,
    ENCODER_N_LAYERS,
    HIDDEN_SIZE,
    MODEL_NAME,
    EncoderRNN,
    GreedySearchDecoder,
    LuongAttnDecoderRNN,
    device,
)

# Tutorial hyper-parameters
CLIP = 50.0
TEACHER_FORCING_RATIO = 1.0
LEARNING_RATE = 0.0001
DECODER_LEARNING_RATIO = 5.0
PRINT_EVERY = 100
SAVE_EVERY = 1000


def maskNLLLoss(inp, target, mask):
    """Element-wise negative log likelihood over non-PAD positions."""
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len,
          encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip,
          max_length=MAX_LENGTH, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to("cpu")
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[: decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropagation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss, print_losses, n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
               decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers,
               save_dir, n_iteration, batch_size, print_every, save_every, clip,
               corpus_name, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
    # Load batches for each iteration (built lazily here -- the tutorial
    # pre-materialises all n_iteration batches; the distribution is identical
    # because every batch is drawn with random.choice).
    print("Training for {} iterations...".format(n_iteration))
    start_iteration = 1
    print_loss = 0

    # Training loop
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = batch2TrainData(
            voc, [random.choice(pairs) for _ in range(batch_size)]
        )
        # Unpack padding
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss, print_losses, n_totals = train(
            input_variable, lengths, target_variable, mask, max_target_len,
            encoder, decoder, embedding,
            encoder_optimizer, decoder_optimizer, batch_size, clip,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(
                "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                    iteration, iteration / n_iteration * 100, print_loss_avg
                )
            )
            print_loss = 0

        # Save checkpoint
        if iteration % save_every == 0:
            directory = os.path.join(
                save_dir, model_name, corpus_name,
                "{}-{}_{}".format(encoder_n_layers, decoder_n_layers, encoder.hidden_size),
            )
            os.makedirs(directory, exist_ok=True)
            torch.save(
                {
                    "iteration": iteration,
                    "en": encoder.state_dict(),
                    "de": decoder.state_dict(),
                    "emb": embedding.state_dict(),
                    "voc": voc.__dict__,
                    "optimizer_en": encoder_optimizer.state_dict(),
                    "optimizer_de": decoder_optimizer.state_dict(),
                },
                os.path.join(directory, "{}_{}.tar".format(iteration, "checkpoint")),
            )


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc, max_length=MAX_LENGTH):
    ### Chat loop ###
    while True:
        try:
            # Get input sentence
            input_sentence = input("> ")
            # Check if it is quit case
            if input_sentence == "q" or input_sentence == "quit":
                break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, max_length)
            # Format and print response sentence
            output_words[:] = [
                x for x in output_words if not (x == "EOS" or x == "PAD")
            ]
            print("Bot:", " ".join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
        except (EOFError, KeyboardInterrupt):
            # Piped input (`printf 'hi\nq\n' | tilechat chat ...`) hits EOF
            # instead of the "q" sentinel; exit cleanly instead of a traceback.
            break
