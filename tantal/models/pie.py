from typing import List, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics

from tantal.modules.pie.decoder import LinearDecoder, RNNEncoder, AttentionalDecoder
from tantal.modules.pie.embeddings import PieEmbeddings
from tantal.data.vocabulary import Vocabulary, Task
from tantal.modules.pie.utils import flatten_padded_batch, pad


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
            categorical: bool = False,
            use_secondary_tasks_decision: bool = True
    ):
        super(Pie, self).__init__()
        # args
        self.cemb_dim: int = cemb_dim
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        # kwargs
        self.cell: str = cell
        self.dropout: float = dropout
        self.cemb_layers: int = cemb_layers
        self.use_secondary_tasks_decision: bool = use_secondary_tasks_decision
        # only during training
        self.init_rnn: str = init_rnn

        # Vocabulary and stuff
        self._vocabulary: Vocabulary = vocabulary
        self.tasks: Dict[str, Task] = self._vocabulary.tasks
        self.main_task: str = main_task

        # Bidirectional so embedding_encoder_out = 2*cemb_dim
        embedding_out_dim = 2 * cemb_dim
        encoder_out_dim = 2 * hidden_size
        self._context_dim: int = encoder_out_dim
        if self.use_secondary_tasks_decision:
            self._context_dim += sum([
                vocabulary.get_task_size(task.name)
                for task in self.tasks.values()
                if task.categorical and task.name not in {main_task, "lm_token"}
            ])

        self.embedding = PieEmbeddings(
            vocab_size=self._vocabulary.tokenizer_size,
            cemb_dim=cemb_dim,
            padding_int=self._vocabulary.token_pad_index,
            cell=cell,
            dropout=dropout,
            num_layers=num_layers,
            init=init_rnn
        )

        # Encoder
        self.encoder = RNNEncoder(
            in_size=embedding_out_dim, hidden_size=hidden_size,
            num_layers=num_layers,
            cell=cell,
            dropout=dropout,
            init_rnn=init_rnn
        )

        self.linear_secondary_tasks = nn.ModuleDict({
            task.name: LinearDecoder(
                vocab_size=vocabulary.get_task_size(task.name),
                in_features=encoder_out_dim,
                padding_index=vocabulary.categorical_pad_token_index)
            for task in self.tasks.values()
            if task.categorical and task.name not in {main_task, "lm_token"}
        })

        # Decoders
        self.decoder = None
        if not categorical:
            self.decoder = AttentionalDecoder(
                vocabulary=vocabulary,
                cemb_dim=cemb_dim,
                cemb_encoding_dim=embedding_out_dim,
                context_dim=self._context_dim,  # Bi-directional
                num_layers=cemb_layers,
                cell_type=cell,
                dropout=dropout,
                init_rnn=init_rnn
            )
        else:
            self.decoder = LinearDecoder(
                vocab_size=vocabulary.get_task_size(main_task),
                in_features=self._context_dim,
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
            "lm_fwd": .2,
            "lm_bwd": .2,
            **{
                task: 1.0
                for task in self.linear_secondary_tasks
            }
        }

        for task in self.tasks:
            if task != "lm_token":
                setattr(self, f"acc_{task}", torchmetrics.Accuracy(
                    ignore_index=self._vocabulary.categorical_pad_token_index
                    if task != main_task else self._vocabulary.token_pad_index
                ))
                if task == main_task and not self.tasks[task].categorical:
                    # Add token level accuracy
                    setattr(self, f"acc_{task}_token_level", torchmetrics.Accuracy())

    def get_main_task_gt(self, gt: Dict[str, Dict[str, torch.Tensor]], length: bool = False):
        if self.tasks[self.main_task].categorical:
            return gt["categoricals"][self.main_task]
        else:
            return gt["non_categoricals"][self.main_task]

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
        encoded_sentences: List[torch.Tensor] = self.encoder(emb, sequence_length)

        # Compute secondary tasks
        secondary_tasks = {
            task: module(encoded_sentences[-1])  # Compute a last layer
            for task, module in self.linear_secondary_tasks.items()
        }

        # get_context(outs, wemb, wlen, self.tasks[task]['context'])
        if isinstance(self.decoder, AttentionalDecoder):
            # cemb_outs = F.dropout(cemb_outs, p=self.dropout, training=self.training)
            # context: torch.Size([Batch Size * Max Length, 2 * Hidden Size]) <--- PROBLEM
            if self.use_secondary_tasks_decision:
                single_encoded_sentence = flatten_padded_batch(
                    torch.cat([
                        encoded_sentences[-1],
                        *secondary_tasks.values()
                    ], dim=-1),
                    sequence_length
                )
            else:
                single_encoded_sentence = flatten_padded_batch(
                    encoded_sentences[-1],
                    sequence_length
                )
            # Use last layer enc_outs[-1]
            logits = self.decoder(encoded_words=encoded_words, lengths=tokens_length,
                                  encoded_sentence=single_encoded_sentence, train_or_eval=train_or_eval,
                                  max_seq_len=max_seq_len)
        else:
            logits = self.decoder(encoded=encoded_sentences)

        return logits, emb, encoded_sentences, secondary_tasks

    def common_train_val_step(self, batch, batch_idx):
        x, gt = batch
        tokens, tokens_length, sequence_length = x["token"], x["token__length"], x["token__sequence__length"]

        assert sum(sequence_length).item() == tokens.shape[1], "Number of words accross sentence should match " \
                                                               "unrolled tokens"

        (loss_matrix_probs, preds), emb, enc_outs, secondary_tasks = self.proj(
            tokens, tokens_length, sequence_length,
            train_or_eval=True,
            max_seq_len=self.get_main_task_gt(gt).shape[0]
        )

        # out_subwords = pad_sequence(loss_matrix_probs, padding_value=self._tokenizer.token_to_id("[PAD]"))
        # Decoder loss
        losses = {
            "loss_main_task": F.cross_entropy(
                input=loss_matrix_probs.view(-1, loss_matrix_probs.shape[-1]),
                target=self.get_main_task_gt(gt).view(-1),
                ignore_index=self._vocabulary.token_pad_index
            ),
            **{
                f"loss_{task}": F.cross_entropy(
                    input=secondary_tasks[task].view(-1, secondary_tasks[task].shape[-1]),
                    target=gt["categoricals"][task].view(-1),
                    ignore_index=self._vocabulary.token_pad_index
                )
                for task in gt["categoricals"]  # Quickfix as LEMMA is main task now
                if task not in {self.main_task, "lm_token"}
            }
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
        return preds, loss, losses, {
            task: F.softmax(out, dim=-1).max(-1)
            for task, out in secondary_tasks.items()
        }

    def forward(self, batch, batch_idx):
        (_, hyps, _), _, _, secondary_tasks = self.proj(**batch)
        return hyps, secondary_tasks

    def training_step(self, batch, batch_idx):
        preds, loss, loss_dict, secondary_tasks = self.common_train_val_step(batch, batch_idx)
        self.log("train_loss", loss, batch_size=batch[0]["token__sequence__length"].shape[0])
        for task in loss_dict:
            self.log(
                "train_" + task,
                loss_dict[task],
                batch_size=batch[0]["token__sequence__length"].shape[0],
                prog_bar=True
            )
        return loss

    # def _compute_metrics(self, preds, secondary_task, ground_truth):

    def validation_step(self, batch, batch_idx):
        preds, loss, loss_dict, secondary_tasks = self.common_train_val_step(batch, batch_idx)
        self.log("dev_loss", loss, batch_size=batch[0]["token__sequence__length"].shape[0])
        for task in loss_dict:
            if task != "loss_main_task":
                self.log(
                    "val_" + task,
                    loss_dict[task],
                    batch_size=batch[0]["token__sequence__length"].shape[0],
                    prog_bar=True
                )
        # Batch = x, y
        for task in self.tasks.values():
            if task.name == "lm_token":
                continue
            if task.name == self.main_task:
                out = preds.transpose(1, 0)
            else:
                _, out = secondary_tasks[task.name]  # First is probability

            if task.categorical:
                gt = batch[1]["categoricals"][task.name]
            else:
                gt = batch[1]["non_categoricals"][task.name].transpose(1, 0)  # [:, 1:]  # We remove the first dimension

            attribute = getattr(self, f'acc_{task.name}')
            attribute(out.cpu(), gt.cpu())
            self.log(f'acc_{task.name}', attribute, on_epoch=True, prog_bar=True)

            if task.name == self.main_task and not task.categorical:
                # For non-categorical, specifically for lemma, let's make sure this computes the token level
                #   accuracy
                attribute = getattr(self, f'acc_{task.name}_token_level')
                out, gt = self._vocabulary.tokenizer.decode_batch(out.cpu().tolist()), \
                          self._vocabulary.tokenizer.decode_batch(gt.cpu().tolist())
                local_encoder = list(set(out + gt))
                attribute(
                    torch.tensor([local_encoder.index(val) for val in out]),
                    torch.tensor([local_encoder.index(val) for val in gt]),
                )
                self.log(f'acc_{task.name}_token_level', attribute, on_epoch=True, prog_bar=True)

        return loss

    # def validation_epoch_end(self, outputs=None) -> None:
    #    for task in self.accuracy:
    #        self.log(f'{task}_acc', self.accuracy[task])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
