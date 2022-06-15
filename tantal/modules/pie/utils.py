import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def flatten_padded_batch(batch, nwords):
    """
    Inverse of pad_flat_batch

    Parameters
    ===========
    batch : tensor(seq_len, batch, encoding_size), output of the encoder
    nwords : tensor(batch), lengths of the sequence (without padding)

    Returns
    ========
    tensor(nwords, encoding_size)

    >>> batch = [[[0], [3], [4]], [[1], [0], [5]], [[2], [0], [0]]]
    >>> nwords = [3, 1, 2]
    >>> flatten_padded_batch(torch.tensor(batch), torch.tensor(nwords)).tolist()
    [[0], [1], [2], [3], [4], [5]]
    """
    with torch.no_grad():
        output = []
        for sent, sentlen in zip(batch.transpose(0, 1), nwords):
            output.extend(list(sent[:sentlen].chunk(sentlen)))  # remove <eos>

        return torch.cat(output, dim=0)


def pad(batch, pad_value=0, pos='pre'):
    """
    >>> batch = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]])
    >>> pad(batch, pad=-1, pos='pre').tolist()
    [[-1, -1], [1, 1], [2, 2], [3, 3], [4, 4]]
    >>> pad(batch, pad=5, pos='post').tolist()
    [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    """
    if pos.lower() == 'pre':
        padding = (0, 0) * (batch.dim() - 1) + (1, 0)
    elif pos.lower() == 'post':
        padding = (0, 0) * (batch.dim() - 1) + (0, 1)
    else:
        raise ValueError("Unknown value for pos: {}".format(pos))

    return F.pad(batch, padding, value=pad_value)


def pad_flat_batch(emb, nwords, maxlen=None):
    """
    Transform a 2D flat batch (batch of words in multiple sentences) into a 3D
    padded batch where words have been allocated to their respective sentence
    according to user passed sentence lengths `nwords`

    Parameters
    ===========
    emb : torch.Tensor(total_words x emb_dim), flattened tensor of word embeddings
    nwords : torch.Tensor(batch), number of words per sentence

    Returns
    =======
    torch.Tensor(max_seq_len x batch x emb_dim) where:
        - max_seq_len = max(nwords)
        - batch = len(nwords)

    >>> emb = [[0], [1], [2], [3], [4], [5]]
    >>> nwords = [3, 1, 2]
    >>> pad_flat_batch(torch.tensor(emb), torch.tensor(nwords)).tolist()
    [[[0], [3], [4]], [[1], [0], [5]], [[2], [0], [0]]]
    """
    if len(emb) != sum(nwords):
        raise ValueError("Got {} items but was asked to pad {}"
                         .format(len(emb), sum(nwords).item()))

    output, last = [], 0
    maxlen = maxlen or max(nwords).item()

    for sentlen in nwords.tolist():
        padding = (0, 0, 0, maxlen - sentlen)
        output.append(F.pad(emb[last:last+sentlen], padding))
        last = last + sentlen

    # (seq_len x batch x emb_dim)
    output = torch.stack(output, dim=1)

    return output


def pack_sort(inp, lengths, batch_first=False):
    """
    Transform input into PaddedSequence sorting batch by length (as required).
    Also return an index variable that unsorts the output back to the original
    order.

    Parameters:
    -----------
    inp: torch.Tensor(seq_len x batch x dim)
    lengths: LongTensor of length ``batch``

    >>> from torch.nn.utils.rnn import pad_packed_sequence as unpack
    >>> inp = torch.tensor([[1, 3], [2, 4], [0, 5]], dtype=torch.float)
    >>> lengths = torch.tensor([2, 3]) # unsorted order
    >>> sorted_inp, unsort = pack_sort(inp, lengths)
    >>> sorted_inp, _ = unpack(sorted_inp)
    >>> sorted_inp[:, unsort].tolist()  # original order
    [[1.0, 3.0], [2.0, 4.0], [0.0, 5.0]]
    >>> sorted_inp.tolist()  # sorted by length
    [[3.0, 1.0], [4.0, 2.0], [5.0, 0.0]]
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths)  # no need to use gpu

    lengths, sort = torch.sort(lengths, descending=True)
    _, unsort = sort.sort()

    if batch_first:
        inp = pack_padded_sequence(inp[sort], lengths.tolist())
    else:
        inp = pack_padded_sequence(inp[:, sort], lengths.tolist())

    return inp, unsort
