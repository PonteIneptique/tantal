import torch
import torch.nn as nn

from tantal.data.vocabulary import Vocabulary
from tantal.modules import initialization
from tantal.modules.pie.utils import pad_flat_batch


class PieEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, cemb_dim: int, padding_int: int, cell: str,
                 dropout: float, num_layers: int, init: str):
        super(PieEmbeddings, self).__init__()

        self._vocab_size = vocab_size
        self._cemb_dim = cemb_dim
        self._padding_int = padding_int
        self._cell = cell
        self._num_layers = num_layers

        # Embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            cemb_dim,
            padding_idx=padding_int
        )
        # init embeddings
        initialization.init_embeddings(self.embedding)

        self.embedding_ngram_encoder = getattr(nn, cell)(
            cemb_dim, cemb_dim, bidirectional=True,
            num_layers=num_layers, dropout=dropout
        )
        initialization.init_rnn(self.embedding_ngram_encoder, scheme=init)

    def forward(self, tokens, tokens_length, sequence_length):
        """ A mix of embedding.RNNEmbedding and Word Embedding

        :returns: Embedding projections and Inside-Word level Encoded Embeddings
        """
        tokens = self.embedding(tokens)
        # rnn
        hidden = None
        _, sort = torch.sort(tokens_length, descending=True)
        _, unsort = sort.sort()
        tokens, tokens_length = tokens[:, sort], tokens_length[sort]

        if isinstance(self.embedding_ngram_encoder, nn.RNNBase):
            outs, emb = self.embedding_ngram_encoder(
                nn.utils.rnn.pack_padded_sequence(tokens, tokens_length.cpu()), hidden)
            outs, _ = nn.utils.rnn.pad_packed_sequence(outs)
            if isinstance(emb, tuple):
                emb, _ = emb
        else:
            outs, (emb, _) = self.embedding_ngram_encoder(tokens, hidden, tokens_length)

        # (max_seq_len x batch * nwords x emb_dim)
        outs, emb = outs[:, unsort], emb[:, unsort]
        # (layers * 2 x batch x hidden) -> (layers x 2 x batch x hidden)
        emb = emb.view(self._num_layers, 2, len(tokens_length), -1)
        # use only last layer
        emb = emb[-1]
        # (2 x batch x hidden) - > (batch x 2 * hidden)
        emb = emb.transpose(0, 1).contiguous().view(len(tokens_length), -1)
        # (batch x 2 * hidden) -> (nwords x batch x 2 * hidden)
        emb = pad_flat_batch(emb, sequence_length, maxlen=max(sequence_length).item())

        return emb, outs
