import pytorch_lightning as pl
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
import torch.nn as nn
from typing import Tuple, Any

from tantal.modules.pie.decoder import LinearDecoder, RNNEncoder, AttentionalDecoder
from tantal.modules import initialization
from tantal.modules.pie.utils import pad_flat_batch, flatten_padded_batch, pad


class Pie(pl.LightningModule):
    """
    Parameters
    ==========
    cemb_dim : int, embedding dimension for char-level embedding layer
    hidden_size : int, hidden_size for all hidden layers
    dropout : float
    """
    def __init__(
            self,
            tokenizer: Tokenizer,
            cemb_dim: int,
            cemb_layers: int,
            hidden_size: int,
            num_layers: int,
            cell: str = 'LSTM',
            init_rnn: str = 'xavier_uniform',
            # dropout
            dropout: float = .3,
            categorical: bool = False
    ):
        super(Pie, self).__init__()
        # args
        self.cemb_dim = cemb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # kwargs
        self.cell = cell
        self.dropout = dropout
        self.cemb_layers = cemb_layers
        # only during training
        self.init_rnn = init_rnn
        self._tokenizer = tokenizer

        # Embeddings
        self.embedding = nn.Embedding(
            tokenizer.get_vocab_size(),
            cemb_dim,
            padding_idx=tokenizer.token_to_id("[PAD]")
        )
        # init embeddings
        initialization.init_embeddings(self.ngram_emb)

        self.embedding_ngram_encoder = getattr(nn, cell)(
            cemb_dim, cemb_dim, bidirectional=True,
            num_layers=num_layers, dropout=dropout
        )
        initialization.init_rnn(self.inword_emb, scheme=init_rnn)

        # Encoder
        self.encoder = RNNEncoder(
            cemb_dim * 2, hidden_size,
            num_layers=num_layers,
            cell=cell,
            dropout=dropout,
            init_rnn=init_rnn
        )

        # Decoders
        self.decoder = None
        if categorical:
            self.decoder = AttentionalDecoder(
                tokenizer.get_vocab_size(),
                cemb_dim,
                self.cemb.embedding_dim,
                context_dim=hidden_size * 2,  # Bi-directional
                num_layers=cemb_layers,
                cell_type=cell,
                dropout=dropout,
                init_rnn=init_rnn
            )
        else:
            self.decoder = LinearDecoder(
                tokenizer.get_vocab_size(),
                self.cemb.embedding_dim
            )

        self.lm_fwd_decoder = LinearDecoder(tokenizer.get_vocab_size(), hidden_size)
        self.lm_bwd_decoder = self.lm_fwd_decoder

        self._weights = {
            "annotation": 1.0,
            "lm_fwd": 1.0,
            "lm_bwd": 1.0
        }

    def _embedding(self, char, clen, wlen):
        """ A mix of embedding.RNNEmbedding and Word Embedding

        :returns: Embedding projections and Inside-Word level Encoded Embeddings
        """
        char = self.emb(char)
        # rnn
        hidden = None
        _, sort = torch.sort(clen, descending=True)
        _, unsort = sort.sort()
        char, nchars = char[:, sort], clen[sort]
        if isinstance(self.rnn, nn.RNNBase):
            outs, emb = self.rnn(
                nn.utils.rnn.pack_padded_sequence(char, nchars.cpu()), hidden)
            outs, _ = nn.utils.rnn.pad_packed_sequence(outs)
            if isinstance(emb, tuple):
                emb, _ = emb
        else:
            outs, (emb, _) = self.rnn(char, hidden, nchars)

        # (max_seq_len x batch * nwords x emb_dim)
        outs, emb = outs[:, unsort], emb[:, unsort]
        # (layers * 2 x batch x hidden) -> (layers x 2 x batch x hidden)
        emb = emb.view(self.num_layers, 2, len(nchars), -1)
        # use only last layer
        emb = emb[-1]
        # (2 x batch x hidden) - > (batch x 2 * hidden)
        emb = emb.transpose(0, 1).contiguous().view(len(nchars), -1)
        # (batch x 2 * hidden) -> (nwords x batch x 2 * hidden)
        emb = pad_flat_batch(emb, wlen, maxlen=max(wlen).item())

        return emb, outs

    def proj(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        # tensor(length, batch_size * words)
        # tensor(length, batch_size)
        flat_subwords, fsw_len, grouped_subwords, gsw_len = x

        # Embedding
        emb, cemb_outs = self._embedding(grouped_subwords, gsw_len, fsw_len)

        # Encoder
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        enc_outs = self.encoder(emb, fsw_len)

        if isinstance(self.decoder, AttentionalDecoder):
            cemb_outs = F.dropout(cemb_outs, p=self.dropout, training=self.training)
            context = flatten_padded_batch(cemb_outs, fsw_len)
            logits = self.decoder(enc_outs=enc_outs, length=gsw_len, context=context)
        else:
            logits = self.decoder(encoded=enc_outs)

        return logits, emb, enc_outs

    def common_train_val_step(self, batch, batch_idx):
        x, (flat_subwords, fsw_len, grouped_subwords, gsw_len) = batch

        emb, enc_outs, dec_out = self.proj(x)

        # Decoder loss
        losses = {
            "loss_annotation": F.cross_entropy(
                dec_out,
                grouped_subwords,
                ignore_index=self._tokenizer.token_to_id("[PAD]"),
                reduction=self.reduction,
                label_smoothing=self.label_smoothing
            )
        }

        # (LM)
        if len(emb) > 1:  # can't compute loss for 1-length batches
            # always at first layer
            fwd, bwd = F.dropout(
                enc_outs[0], p=0, training=self.training
            ).chunk(2, dim=2)
            # forward logits
            lm_fwd = self.lm_fwd_decoder(pad(fwd[:-1], pos='pre'))
            losses["loss_lm_fwd"] = F.cross_entropy(
                lm_fwd.view(-1, self._tokenizer.get_vocab_size()),
                flat_subwords.view(-1),
                weight=self.nll_weight,
                reduction="mean",
                ignore_index=self._tokenizer.token_to_id("[PAD]")
            )
            # backward logits
            lm_bwd = self.lm_bwd_decoder(pad(bwd[1:], pos='post'))
            losses["loss_lm_bwd"] = F.cross_entropy(
                lm_bwd.view(-1, self._tokenizer.get_vocab_size()),
                flat_subwords.view(-1),
                weight=self.nll_weight,
                reduction="mean",
                ignore_index=self._tokenizer.token_to_id("[PAD]")
            )

        loss = sum([
            self._weights.get(k[5:], 1) * losses[k]
            for k in losses
        ])
        return dec_out, loss

    def forward(self, batch, batch_idx):
        return self.proj(batch)[-1]

    def training_step(self, batch, batch_idx):
        _, loss = self.common_train_val_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        dec_out, loss = self.common_train_val_step(batch, batch_idx)
        self.log("dev_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
