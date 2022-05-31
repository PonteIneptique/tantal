from typing import List, Tuple, Optional
import torch
from torch import nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import torch.functional as F
from tokenizers import Tokenizer

from tantal.modules import initialization
from tantal.modules.pie.attention import Attention

TINY = 1e-8


class AttentionalDecoder(nn.Module):
    """
    Decoder using attention over the entire input sequence

    Parameters
    ===========
    label_encoder : LabelEncoder of the task
    in_dim : int, embedding dimension of the task.
        It should be the same as the corresponding encoder to ensure that weights
        can be shared.
    hidden_size : int, hidden size of the encoder, decoder and attention modules.
    context_dim : int (optional), dimensionality of additional context vectors
    """
    def __init__(
            self,
            tokenizer: Tokenizer,
            in_dim: int,
            hidden_size: int,
            context_dim: int = 0,
            dropout: float = .3,
            # rnn
            num_layers=1,
            cell_type='LSTM',
            init_rnn='default'
    ):

        self.tokenizer = tokenizer
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.init_rnn = init_rnn
        super().__init__()

        self.eos = tokenizer.token_to_id("[EOS]")
        self.bos = tokenizer.token_to_id("[BOS]")

        # nll weight
        nll_weight = torch.ones(tokenizer.get_vocab_size())
        nll_weight[tokenizer.token_to_id("[PAD]")] = 0.
        self.register_buffer('nll_weight', nll_weight)

        # emb
        self.embs = nn.Embedding(self.tokenizer.get_vocab_size(), in_dim)

        # rnn
        self.rnn = getattr(nn, cell_type)(in_dim + context_dim, hidden_size,
                                          num_layers=num_layers,
                                          dropout=dropout if num_layers > 1 else 0)
        self.attn = Attention(hidden_size)
        self.proj = nn.Linear(hidden_size, len(tokenizer.get_vocab_size()))

        self.init()

    def init(self):
        # embeddings
        initialization.init_embeddings(self.embs)
        # rnn
        initialization.init_rnn(self.rnn, scheme=self.init_rnn)
        # linear
        initialization.init_linear(self.proj)

    def forward(
            self,
            enc_outs,
            lengths,
            max_seq_len=20,
            bos=None,
            eos=None,
            context=None) -> Tuple[Optional[torch.Tensor], List[Tuple[int]], List[float]]:
        """
         Decoding routine for inference with step-wise argmax procedure
         Parameters
         ===========
         enc_outs : tensor(src_seq_len x batch x hidden_size)
         context : tensor(batch x hidden_size), optional

         :returns: Raw linear output per char, ?, ?
         """
        eos = eos or self.label_encoder.get_eos()
        bos = bos or self.label_encoder.get_bos()
        hidden, batch, device = None, enc_outs.size(1), enc_outs.device
        inp = torch.zeros(batch, dtype=torch.int64, device=device) + bos
        hyps: List[Tuple[int]] = []
        final_scores = torch.tensor([0 for _ in range(batch)], dtype=torch.float64, device="cpu")

        # To make a "character" level loss, we'll append to a loss matrix each probs
        #  It necessarily starts with <BOS>
        loss_matrix_probs: Optional[torch.Tensor] = None

        # As we go, we'll reduce the tensor size by popping finished prediction
        #  To keep adding new characters to the right words, we
        #  store and keep updated a Tensor where Tensor Index -> Batch Original ID
        #  where Batch Original ID is the Word ID (batch_size = number of words)
        tensor_to_original_batch_indexes = torch.tensor(
            list(range(batch)),
            dtype=torch.int64,
            device=device
        )  # Tensor(batch_size)

        for _ in range(max_seq_len):

            # Prepare input
            #    Context is NEVER changed after the method has been called
            emb = self.embs(inp)  # Tensor(batch_size x emb_size)
            if context is not None:
                emb = torch.cat([emb, context], dim=1)  # Tensor(batch_size x (emb_size+context_size))

            # Run rnn
            emb = emb.unsqueeze(0)  # Tensor(1 x batch_size x emb size(+context))

            # hidden is gonna be reused by the next iteration
            #   outs is specific to the current run
            outs, hidden = self.rnn(emb, hidden)
            # Hidden : Tensor(1 x batch_size x emb_size)

            outs, _ = self.attn(outs, enc_outs, lengths)
            outs = self.proj(outs).squeeze(0)

            # Get logits
            probs = F.log_softmax(outs, dim=1)

            # Sample and accumulate
            #  Score are the probabilities
            #  Inp are the new characters (as int) we are adding to our predictions
            score, inp = probs.max(1)  # (Tensor(batch_size, dtype=float), Tensor(batch_size, dtype=int))

            # We create a mask of value that are not ending the string
            non_eos = (inp != eos)  # Tensor(batch_size, dtype=bool)

            # Using this mask, we retrieve the Indexes of items that are not EOS
            #  nonzero() returns a 2D Tensor where each row is an index
            #  not equal to 0. It can be use as a (mask) selector for other tensors (see below)
            keep = torch.nonzero(non_eos).squeeze(1)  # Tensor(dtype=int)

            # We prepare a sequence output made of EOS which we'll fill with predictions
            #   torch.full() takes size as tuple for first argument, filling value as second
            seq_output = torch.full((batch,), eos, device=device, dtype=torch.int64)

            # We replace the value at indexes *tensor_to_original_batch_indexes* by the prediction
            #   of current sequence output
            seq_output[tensor_to_original_batch_indexes] = inp

            # If we are training, we also set-up the same thing for the loss_matrix_probs
            if self.training:
                in_loop_loss = torch.full(
                    (batch, self.tokenizer.get_vocab_size()),
                    .0,
                    device=device
                )
                in_loop_loss[tensor_to_original_batch_indexes] = outs
                if loss_matrix_probs is None:
                    # We add a dimension at the "Word" level
                    loss_matrix_probs = in_loop_loss.unsqueeze(1)
                else:
                    loss_matrix_probs = torch.cat([loss_matrix_probs, in_loop_loss])

            # We set the score where we have EOS predictions as 0
            score[inp == eos] = 0
            # So that we can add the score to finale scores
            final_scores[tensor_to_original_batch_indexes] += score.cpu()

            # We add this new output to the final hypothesis
            hyps.append(seq_output.tolist())

            # If there nothing else than EOS, it's the end of the prediction time
            if non_eos.sum() == 0:
                break

            # Otherwise, we update the tensor_to_batch_indexes by transferring
            #   the current associated index with the new indexes
            tensor_to_original_batch_indexes = tensor_to_original_batch_indexes[keep]

            # We use the Tensor of indexes that are not EOS to filter out
            #   Elements of the batch that are EOS.
            #   inp, context, lengths are all Tensor(batch_size x ....)
            #   so we filter them at the first dimension
            inp = inp[keep]
            context = context[keep]
            lengths = lengths[keep]

            # However, hidden is 3D (Tensor(1 x batch_size x _)
            #   So we filter at the second dimension directly
            if isinstance(hidden, tuple):  # LSTM
                hidden = tuple([hid[:, keep, :] for hid in hidden])
            else:  # GRU
                hidden = hidden[:, keep, :]

            # enc_outs is Tensor(max_seq_len x batch x hidden_size)
            #   Seq_len is supposed to be equal to max(lengths),
            #     but if the maximum length is popped, it is not in sync anymore.
            #   In order to keep wording, we remove extra dimension if lengths.max() has changed.
            # We then update the first (max_seq_len) and second (batch_size) dimensions accordingly.
            max_seq_len = lengths.max()
            enc_outs = enc_outs[:max_seq_len, keep, :]

        hyps = [hyp for hyp in zip(*hyps)]
        final_scores = [s / (len(hyp) + TINY) for s, hyp in zip(final_scores, hyps)]

        return loss_matrix_probs, hyps, final_scores


def sequential_dropout(inp: torch.Tensor, p: float, training: bool):
    if not training or not p:
        return inp

    mask = inp.new(1, inp.size(1), inp.size(2)).bernoulli_(1 - p)
    mask = mask / (1 - p)

    return inp * mask.expand_as(inp)


class LinearDecoder(nn.Module):
    """
    Simple Linear Decoder that outputs a probability distribution
    over the vocabulary

    Parameters
    ===========
    vocab_size : LabelEncoder
    in_features : int, input dimension
    """
    def __init__(self, vocab_size: int, in_features: int, padding_index: int):
        self.out_dim = vocab_size
        super().__init__()

        # nll weight
        nll_weight = torch.ones(self.out_dim)
        nll_weight[padding_index] = 0.
        self.register_buffer('nll_weight', nll_weight)
        # decoder output
        self.decoder = nn.Linear(in_features, self.out_dim)
        self.init()

    def init(self):
        # linear
        initialization.init_linear(self.decoder)

    def forward(self, enc_outs):
        linear_out = self.decoder(enc_outs)
        return linear_out


class RNNEncoder(nn.Module):
    def __init__(self,
                 in_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 cell: str ='GRU',
                 dropout: float =0.0,
                 init_rnn: str ='default'
                 ):

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = cell
        self.dropout = dropout
        self.init_rnn = init_rnn
        super().__init__()

        rnn = []
        for layer in range(num_layers):
            rnn_inp = in_size if layer == 0 else hidden_size * 2
            rnn.append(getattr(nn, cell)(rnn_inp, hidden_size, bidirectional=True))
        self.rnn = nn.ModuleList(rnn)

        self.init()

    def init(self):
        for rnn in self.rnn:
            initialization.init_rnn(rnn, scheme=self.init_rnn)

    def forward(self, inp, lengths, hidden=None, only_last=False, return_hidden=False):
        if hidden is None:
            hidden = [
                initialization.init_hidden_for(
                    inp, 2, 1, self.hidden_size, self.cell, add_init_jitter=False)
                for _ in range(len(self.rnn))]

        _, sort = torch.sort(lengths, descending=True)
        _, unsort = sort.sort()
        inp = nn.utils.rnn.pack_padded_sequence(inp[:, sort], lengths[sort].cpu())
        outs, hiddens = [], []

        for layer, rnn in enumerate(self.rnn):
            # apply dropout only in between layers (not on the output)
            if layer > 0:
                inp, lengths = nn.utils.rnn.pad_packed_sequence(inp)
                inp = sequential_dropout(inp, self.dropout, self.training)
                inp = nn.utils.rnn.pack_padded_sequence(inp, lengths.cpu())

            # run layer
            louts, lhidden = rnn(inp, hidden[layer])

            # unpack
            louts_, _ = nn.utils.rnn.pad_packed_sequence(louts)
            outs.append(louts_[:, unsort])

            if isinstance(lhidden, tuple):
                lhidden = lhidden[0][:, unsort, :], lhidden[1][:, unsort, :]
            else:
                lhidden = lhidden[:, unsort, :]
            hiddens.append(lhidden)

            # recur
            inp = louts

        if only_last:
            outs, hiddens = outs[-1], hiddens[-1]

        if return_hidden:
            return outs, hiddens
        else:
            return outs
