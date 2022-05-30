import torch
from torch.nn import functional as F


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