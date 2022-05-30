import logging
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def init_embeddings(embeddings):
    embeddings.reset_parameters()
    nn.init.constant_(embeddings.weight, 0.01)


def init_linear(linear):
    linear.reset_parameters()
    nn.init.xavier_uniform_(linear.weight)


def init_rnn(rnn, forget_bias=1.0, scheme='default'):
    logger.info("Initializing {} with scheme: {}".format(type(rnn).__name__, scheme))
    for pname, p in rnn.named_parameters():
        if 'bias' in pname:
            nn.init.constant_(p, 0.)
            # forget_bias
            if 'LSTM' in type(rnn).__name__:
                nn.init.constant_(p[rnn.hidden_size:rnn.hidden_size*2], forget_bias)

        elif pname.startswith('weight_ih'):
            if scheme == 'xavier_uniform':
                nn.init.xavier_uniform_(p)
            elif scheme == 'orthogonal':
                nn.init.orthogonal_(p)

        elif pname.startswith('weight_hh'):
            if scheme == 'xavier_uniform':
                nn.init.xavier_uniform_(p)
            elif scheme == 'orthogonal':
                gates = 1
                if 'LSTM' in type(rnn).__name__:
                    gates = 4
                elif 'GRU' in type(rnn).__name__:
                    gates = 3
                for i in range(gates):
                    nn.init.eye_(p[i * rnn.hidden_size: (i+1) * rnn.hidden_size])


def init_hidden_for(inp, num_dirs, num_layers, hid_dim, cell,
                    h_0=None, add_init_jitter=False):
    """
    General function for initializing RNN hidden states

    Parameters:
    - inp: torch.Tensor(seq_len, batch_size, dim)
    """
    size = (num_dirs * num_layers, inp.size(1), hid_dim)

    # create h_0
    if h_0 is not None:
        h_0 = h_0.repeat(1, inp.size(1), 1)
    else:
        h_0 = torch.zeros(*size, device=inp.device)

    # eventualy add jitter
    if add_init_jitter:
        h_0 = h_0 + torch.normal(torch.zeros_like(h_0), 0.3)

    if cell.startswith('LSTM'):
        # compute memory cell
        return h_0, torch.zeros_like(h_0)
    else:
        return h_0


def init_conv(conv):
    conv.reset_parameters()
    nn.init.xavier_uniform_(conv.weight)
    nn.init.constant_(conv.bias, 0.)
    pass


def init_pretrained_embeddings(path, encoder, embedding):
    inits = 0
    with open(path) as f:
        # maybe validate
        header = next(f)
        if len(header.split()) > 2:
            logger.info("Skipping header validation for embedding file: {}".format(path))
            f = (line for it in [[header], f] for line in it)
        else:
            nemb, dim = next(f).split()
            if int(dim) != embedding.weight.data.size(1):
                raise ValueError("Unexpected embeddings size: {}".format(dim))

        for line in f:
            word, *vec = line.split()
            if word in encoder.table:
                embedding.weight.data[encoder.table[word], :].copy_(
                    torch.tensor([float(v) for v in vec]))
                inits += 1

    if embedding.padding_idx is not None:
        embedding.weight.data[embedding.padding_idx].zero_()

    logger.info("Initialized {}/{} embeddings".format(inits, embedding.num_embeddings))
