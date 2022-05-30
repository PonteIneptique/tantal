import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import initialization


def make_length_mask(lengths):
    """
    Compute binary length mask.

    lengths: torch.Tensor(batch, dtype=int) should be on the desired
        output device.

    Returns
    =======

    mask: torch.ByteTensor(batch x seq_len)
    """
    maxlen, batch = lengths.detach().max(), len(lengths)
    return torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device) \
                .repeat(batch, 1) \
                .lt(lengths.unsqueeze(1))


def DotScorer(dec_out, enc_outs, **kwargs):
    """
    Score for query decoder state and the ith encoder state is given
    by their dot product.

    dec_outs: (trg_seq_len x batch x hid_dim)
    enc_outs: (src_seq_len x batch x hid_dim)

    output: ((trg_seq_len x) batch x src_seq_len)
    """
    score = torch.bmm(
        # (batch x src_seq_len x hid_dim)
        enc_outs.transpose(0, 1),
        # (batch x hid_dim x trg_seq_len)
        dec_out.transpose(0, 1).transpose(1, 2))
    # (batch x src_seq_len x trg_seq_len) => (trg_seq_len x batch x src_seq_len)
    return score.transpose(0, 1).transpose(0, 2)


class GeneralScorer(nn.Module):
    """
    Inserts a linear projection to the query state before the dot product
    """
    def __init__(self, dim):
        super(GeneralScorer, self).__init__()
        self.W_a = nn.Linear(dim, dim, bias=False)
        self.init()

    def init(self):
        initialization.init_linear(self.W_a)

    def forward(self, dec_out, enc_outs, **kwargs):
        return DotScorer(self.W_a(dec_out), enc_outs)


class Attention(nn.Module):
    """
    Global attention implementing the three scorer modules from Luong 15.

    Parameters:
    -----------
    - hid_dim: int, dimensionality of the query vector
    - att_dim: (optional) int, dimensionality of the attention space (only
        used by the bahdanau scorer). If not given it will default to hid_dim.
    - scorer: str, one of ('dot', 'general', 'bahdanau')
    - hid_dim2: (optional), int, dimensionality of the key vectors (optionally
        used by the bahdanau scorer if given)
    """
    def __init__(self, hid_dim, att_dim=None, scorer='general'):
        super(Attention, self).__init__()

        # Scorer
        self.scorer = GeneralScorer(hid_dim)

        # Output layer (Luong 15. eq (5))
        self.linear_out = nn.Linear(
            hid_dim * 2, hid_dim, bias=False)

        self.init()

    def init(self):
        initialization.init_linear(self.linear_out)

    def forward(self, dec_out, enc_outs, lengths):
        """
        Parameters:
        -----------

        - dec_outs: torch.Tensor(trg_seq_len x batch_size x hid_dim)
        - enc_outs: torch.Tensor(seq_len x batch_size x hid_dim)
        - lengths: torch.LongTensor(batch), source lengths

        Returns:
        --------
        - context: (trg_seq_len x batch x hid_dim)
        - weights: (trg_seq_len x batch x seq_len)
        """
        # get scores
        # (trg_seq_len x batch x seq_len)
        weights = self.scorer(dec_out, enc_outs)

        # apply source length mask
        mask = make_length_mask(lengths)
        # (batch x src_seq_len) => (trg_seq_len x batch x src_seq_len)
        mask = mask.unsqueeze(0).expand_as(weights)
        # weights = weights * mask.float()
        # Torch 1.1 -> 1.2: (1 - mask) becomes ~(mask)
        weights.masked_fill_(~mask, -float('inf'))

        # normalize
        weights = F.softmax(weights, dim=2)

        # (eq 7) (batch x trg_seq_len x seq_len) * (batch x seq_len x hid_dim)
        # => (batch x trg_seq_len x hid_dim) => (trg_seq_len x batch x hid_dim)
        context = torch.bmm(
            weights.transpose(0, 1), enc_outs.transpose(0, 1)
        ).transpose(0, 1)
        # (eq 5) linear out combining context and hidden
        context = torch.tanh(self.linear_out(torch.cat([context, dec_out], 2)))

        return context, weights
