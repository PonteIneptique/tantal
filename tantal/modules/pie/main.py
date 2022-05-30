import pytorch_lightning as pl
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
import torch.nn as nn
from typing import Tuple

from tantal.modules.pie.decoder import LinearDecoder, RNNEncoder, AttentionalDecoder
from tantal.modules import initialization
from tantal.modules.pie.utils import pad_flat_batch, flatten_padded_batch


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

    def init_from_encoder(self, encoder):
        # wemb
        total = 0
        for w, idx in encoder.label_encoder.word.table.items():
            if w in self.label_encoder.word.table:
                self.wemb.weight.data[self.label_encoder.word.table[w]].copy_(
                    encoder.wemb.weight.data[idx])
                total += 1
        print("Initialized {}/{} word embs".format(total, len(self.wemb.weight)))
        # cemb
        total = 0
        for w, idx in encoder.label_encoder.char.table.items():
            if w in self.label_encoder.char.table:
                self.cemb.emb.weight.data[self.label_encoder.char.table[w]].copy_(
                    encoder.cemb.emb.weight.data[idx])
                total += 1
        print("Initialized {}/{} char embs".format(total, len(self.cemb.emb.weight)))
        # cemb rnn
        self.cemb.rnn.load_state_dict(encoder.cemb.rnn.state_dict())
        # sentence rnn
        self.encoder.load_state_dict(encoder.encoder.state_dict())

        if self.include_lm:
            pass

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        char, clen, wlen = x

        # Embedding
        emb, cemb_outs = self._embedding(char, clen, wlen)

        # Encoder
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        enc_outs = self.encoder(emb, wlen)

        if isinstance(self.decoder, AttentionalDecoder):
            cemb_outs = F.dropout(cemb_outs, p=self.dropout, training=self.training)
            context = flatten_padded_batch(cemb_outs, wlen)
            logits = self.decoder(enc_outs=enc_outs, length=clen, context=context)
        else:
            logits = self.decoder(encoded=enc_outs)

        # (LM)
        lm_fwd, lm_bwd = None, None
        if self.training:
            if len(emb) > 1:  # can't compute loss for 1-length batches
                # always at first layer
                fwd, bwd = F.dropout(
                    enc_outs[0], p=0, training=self.training
                ).chunk(2, dim=2)
                # forward logits
                lm_fwd = self.lm_fwd_decoder(pad(fwd[:-1], pos='pre'))
                # backward logits
                lm_bwd = self.lm_bwd_decoder.loss(logits, word)

        return {"decoded": logits, "lm_fwd": lm_fwd, "lm_bwd": lm_bwd}

    def predict(self, inp, *tasks, return_probs=False,
                use_beam=False, beam_width=10, **kwargs):
        """
        inp : (word, wlen), (char, clen), text input
        tasks : list of str, target tasks
        """
        tasks = set(self.tasks if not len(tasks) else tasks)
        preds, probs = {}, {}
        (word, wlen), (char, clen) = inp

        # Embedding
        emb, (wemb, cemb, cemb_outs) = self.embedding(word, wlen, char, clen)

        # Encoder
        enc_outs = None
        if self.encoder is not None:
            # TODO: check if we need encoder for this particular batch
            enc_outs = self.encoder(emb, wlen)

        # Decoders
        for task in tasks:

            decoder, at_layer = self.decoders[task], self.tasks[task]['layer']
            outs = None
            if enc_outs is not None:
                outs = enc_outs[at_layer]

            if self.label_encoder.tasks[task].level.lower() == 'char':
                if isinstance(decoder, LinearDecoder):
                    hyps, prob = decoder.predict(cemb_outs, clen)
                elif isinstance(decoder, CRFDecoder):
                    hyps, prob = decoder.predict(cemb_outs, clen)
                else:
                    context = get_context(outs, wemb, wlen, self.tasks[task]['context'])
                    if use_beam:
                        hyps, prob = decoder.predict_beam(
                            cemb_outs, clen, width=beam_width, context=context)
                    else:
                        hyps, prob = decoder.predict_max(
                            cemb_outs, clen, context=context)
                    if self.label_encoder.tasks[task].preprocessor_fn is None:
                        hyps = [''.join(hyp) for hyp in hyps]
            else:
                if isinstance(decoder, LinearDecoder):
                    hyps, prob = decoder.predict(outs, wlen)
                elif isinstance(decoder, CRFDecoder):
                    hyps, prob = decoder.predict(outs, wlen)

            preds[task] = hyps
            probs[task] = prob

        if return_probs:
            return preds, probs

        return preds