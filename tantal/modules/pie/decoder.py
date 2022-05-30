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

    def forward(self, targets, lengths, enc_outs, src_lengths, context=None):
        """
        Decoding routine for training. Returns the logits corresponding to
        the targets for the `loss` method. Takes care of padding.
        """
        targets, lengths = targets[:-1], lengths - 1
        embs = self.embs(targets)

        if self.context_dim > 0:
            if context is None:
                raise ValueError("Contextual Decoder needs `context`")
            # (seq_len x batch x emb_dim) + (batch x context_dim)
            embs = torch.cat(
                [embs, context.unsqueeze(0).repeat(embs.size(0), 1, 1)],
                dim=2)

        embs, unsort = pack_padded_sequence(embs, lengths, enforce_sorted = False)

        outs, _ = self.rnn(embs)
        outs, _ = pad_packed_sequence(outs)
        outs = outs[:, unsort]

        context, _ = self.attn(outs, enc_outs, src_lengths)

        return self.proj(context)

    def loss(self, logits, targets):
        """
        Compute loss from logits (output of forward)

        Parameters
        ===========
        logits : tensor(seq_len x batch x vocab)
        targets : tensor(seq_len x batch)
        """
        targets = targets[1:]  # remove <bos> from targets

        loss = F.cross_entropy(
            logits.view(-1, len(self.label_encoder)), targets.view(-1),
            weight=self.nll_weight, reduction="mean",
            ignore_index=self.label_encoder.get_pad())

        print(logits.view(-1, len(self.label_encoder)).size())
        print(targets.view(-1).size())
        raise Exception
        # FIXME: normalize loss to be word-level

        return loss

    def predict_max(self, enc_outs, lengths,
                    max_seq_len=20, bos=None, eos=None,
                    context=None):
        """
        Decoding routine for inference with step-wise argmax procedure

        Parameters
        ===========
        enc_outs : tensor(src_seq_len x batch x hidden_size)
        context : tensor(batch x hidden_size), optional
        """
        eos = eos or self.eos
        bos = bos or self.bos
        hidden, batch, device = None, enc_outs.size(1), enc_outs.device
        mask = torch.ones(batch, dtype=torch.int64, device=device)
        inp = torch.zeros(batch, dtype=torch.int64, device=device) + bos
        hyps, scores = [], 0

        for _ in range(max_seq_len):
            if mask.sum().item() == 0:
                break

            # prepare input
            emb = self.embs(inp)
            if context is not None:
                emb = torch.cat([emb, context], dim=1)
            # run rnn
            emb = emb.unsqueeze(0)
            outs, hidden = self.rnn(emb, hidden)
            outs, _ = self.attn(outs, enc_outs, lengths)
            outs = self.proj(outs).squeeze(0)
            # get logits
            probs = F.log_softmax(outs, dim=1)
            # sample and accumulate
            score, inp = probs.max(1)
            hyps.append(inp.tolist())
            mask = mask * (inp != eos).long()
            score = score.cpu()
            score[mask == 0] = 0
            scores += score

        hyps = [self.label_encoder.stringify(hyp) for hyp in zip(*hyps)]
        scores = [s/(len(hyp) + TINY) for s, hyp in zip(scores.tolist(), hyps)]

        return hyps, scores


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
    label_encoder : LabelEncoder
    in_features : int, input dimension
    """
    def __init__(self, label_encoder, in_features):
        self.label_encoder = label_encoder
        super().__init__()

        # nll weight
        nll_weight = torch.ones(len(label_encoder))
        nll_weight[label_encoder.get_pad()] = 0.
        self.register_buffer('nll_weight', nll_weight)
        # decoder output
        self.decoder = nn.Linear(in_features, len(label_encoder))
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
