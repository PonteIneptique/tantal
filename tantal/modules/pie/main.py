import pytorch_lightning as pl
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Any

from tantal.modules.pie.decoder import LinearDecoder, RNNEncoder, AttentionalDecoder
from tantal.modules.pie.embeddings import PieEmbeddings
from tantal.data.vocabulary import Vocabulary
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
            vocabulary: Vocabulary,
            main_task: str,
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

        # Vocabulary and stuff
        self._vocabulary = vocabulary
        self.tasks = self._vocabulary.tasks
        self.main_task = main_task

        self.embedding = PieEmbeddings(
            vocab_size=self._vocabulary.tokenizer_size,
            cemb_dim=cemb_dim,
            padding_int=self._vocabulary.token_pad_index,
            cell=cell,
            dropout=dropout,
            num_layers=num_layers,
            init=init_rnn
        )
        # Bidirectional so embedding_encoder_out = 2*cemb_dim
        embedding_out_dim = 2 * cemb_dim

        # Encoder
        self.encoder = RNNEncoder(
            in_size=embedding_out_dim, hidden_size=hidden_size,
            num_layers=num_layers,
            cell=cell,
            dropout=dropout,
            init_rnn=init_rnn
        )
        encoder_out_dim = 2 * hidden_size

        self.linear_secondary_tasks = nn.ModuleDict({
            task.name: LinearDecoder(
                vocab_size=vocabulary.get_task_size(task.name),
                in_features=cemb_dim * 2,
                padding_index=vocabulary.categorical_pad_token_index)
            for task in self.tasks.values()
            if task.categorical and task.name not in {main_task, "token_lm"}
        })

        # Decoders
        self.decoder = None
        if not categorical:
            self.decoder = AttentionalDecoder(
                vocabulary=vocabulary,
                cemb_dim=cemb_dim,
                cemb_encoding_dim=embedding_out_dim,
                context_dim=encoder_out_dim,  # Bi-directional
                num_layers=cemb_layers,
                cell_type=cell,
                dropout=dropout,
                init_rnn=init_rnn
            )
        else:
            self.decoder = LinearDecoder(
                vocab_size=vocabulary.get_task_size(main_task),
                in_features=cemb_dim * 2,
                padding_index=vocabulary.categorical_pad_token_index
            )

        self.lm_fwd_decoder = LinearDecoder(
                vocab_size=vocabulary.get_task_size("lm_token"),
                in_features=hidden_size,
                padding_index=vocabulary.categorical_pad_token_index)
        self.lm_bwd_decoder = self.lm_fwd_decoder

        # nll weight
        nll_weight = torch.ones(vocabulary.get_task_size("lm_token"))
        nll_weight[vocabulary.categorical_pad_token_index] = 0.
        self.register_buffer('lm_nll_weight', nll_weight)

        self._weights = {
            "annotation": 1.0,
            "lm_fwd": 1.0,
            "lm_bwd": 1.0,
            **{
                task: 1.0
                for task in self.linear_secondary_tasks
            }
        }

    def proj(
            self,
            tokens: torch.Tensor,
            tokens_length: torch.Tensor,
            sequence_length: torch.Tensor,
            train_or_eval: bool = False,
            max_seq_len: int = 20
    ):
        """

        :param tokens: Unrolled tokens (each token is one dim, outside of any sentence question)
        """
        # tensor(length, batch_size)
        # tensor(length, batch_size * words)
        # Embedding
        # emb: torch.Size([Max(NBWords), BatchSize, 2*Character Embedding Dim])
        # cemb_outs: torch.Size([MaxSubWordCount(Words), Sum(NBWords), 2*Character Embedding Dim])
        emb, encoded_words = self.embedding(tokens, tokens_length, sequence_length)

        # Encoder
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        # enc_outs: torch.Size([Max(NBWords), BatchSize, NBLayers*HiddenSize])
        encoded_sentences = self.encoder(emb, sequence_length)
        # get_context(outs, wemb, wlen, self.tasks[task]['context'])
        if isinstance(self.decoder, AttentionalDecoder):
            # cemb_outs = F.dropout(cemb_outs, p=self.dropout, training=self.training)
            # context: torch.Size([Batch Size * Max Length, 2 * Hidden Size]) <--- PROBLEM
            single_encoded_sentence = flatten_padded_batch(encoded_sentences[-1], sequence_length)
            # Use last layer enc_outs[-1]
            logits = self.decoder(encoded_words=encoded_words, lengths=tokens_length,
                                  encoded_sentence=single_encoded_sentence, train_or_eval=train_or_eval,
                                  max_seq_len=max_seq_len)
        else:
            logits = self.decoder(encoded=encoded_sentences)

        return logits, emb, encoded_sentences

    def common_train_val_step(self, batch, batch_idx):
        x, gt = batch
        tokens, tokens_length, sequence_length = x["token"], x["token__length"], x["token__sequence__length"]

        assert sum(sequence_length).item() == tokens.shape[1], "Number of words accross sentence should match " \
                                                               "unrolled tokens"

        (loss_matrix_probs, hyps, final_scores), emb, enc_outs = self.proj(
            tokens, tokens_length, sequence_length,
            train_or_eval=True,
            max_seq_len=tokens_length.max()
        )

        # out_subwords = pad_sequence(loss_matrix_probs, padding_value=self._tokenizer.token_to_id("[PAD]"))
        # Decoder loss
        losses = {
            "loss_annotation": F.cross_entropy(
                input=loss_matrix_probs.view(-1, loss_matrix_probs.shape[-1]),
                target=tokens.view(-1),
                ignore_index=self._vocabulary.token_pad_index
            )
        }

        # (LM)
        if len(emb) > 1:
            # Divide the two direction of enc_outs[0]
            fwd, bwd = enc_outs[0].chunk(2, dim=2)

            # Remove the first last token and try to predict each next token (WordLevel)
            # Issue: we are not at the word level
            # Solution:
            #   1. Use grouped subwords ? But wouldn't that be weird in terms of efficiency ? #
            #          Not even sure it's possible (same problem ?)
            #   2. Use RNN and predict flat_subwords. Need to share everything though.
            lm_fwd = self.lm_fwd_decoder(pad(fwd[:-1], pos='pre'))
            losses["loss_lm_fwd"] = F.cross_entropy(
                lm_fwd.view(-1, self._vocabulary.get_task_size("lm_token")),
                gt["categoricals"]["lm_token"].view(-1),
                weight=self.lm_nll_weight,
                reduction="mean",
                ignore_index=self._vocabulary.categorical_pad_token_index
            )
            # Same but previous token is the target
            lm_bwd = self.lm_bwd_decoder(pad(bwd[1:], pos='post'))
            losses["loss_lm_bwd"] = F.cross_entropy(
                lm_bwd.view(-1, self._vocabulary.get_task_size("lm_token")),
                gt["categoricals"]["lm_token"].view(-1),
                weight=self.lm_nll_weight,
                reduction="mean",
                ignore_index=self._vocabulary.categorical_pad_token_index
            )

        loss = sum([
            self._weights.get(k[5:], 1) * losses[k]
            for k in losses
        ])
        return hyps, loss

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
