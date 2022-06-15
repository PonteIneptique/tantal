<<<<<<< HEAD
from typing import List, Dict, Optional, Tuple, Union, IO, Callable
=======
from typing import List, Dict, Optional, Tuple, Union, IO, Callable, Any
>>>>>>> 3f29304c7fc9b68c1697e6726389c4136f93c1ee
from collections import namedtuple

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
import torchmetrics

from tantal.modules.pie.decoder import LinearDecoder, RNNEncoder, AttentionalDecoder
from tantal.modules.pie.embeddings import PieEmbeddings
from tantal.data.vocabulary import Vocabulary, Task
from tantal.utils import ScoreWatcher, compute_scores
from tantal.modules.pie.utils import flatten_padded_batch, pad
from ranger import Ranger

Score = namedtuple("Score", ["score", "steps"])


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
            char_cell: Optional[str] = None,
            cell: str = 'LSTM',
            init_rnn: str = 'xavier_uniform',
            lr: float = .001,
            # dropout
            dropout: float = .3,
            categorical: bool = False,
            use_secondary_tasks_decision: bool = True
    ):
        super(Pie, self).__init__()
        self.save_hyperparameters(ignore=["vocabulary"])

        # args
        self.cemb_dim: int = cemb_dim
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        # kwargs
        self.cell: str = cell
        self.char_cell: str = char_cell or cell
        self.lr: float = lr
        self.dropout: float = dropout
        self.cemb_layers: int = cemb_layers
        self.use_secondary_tasks_decision: bool = use_secondary_tasks_decision
        # only during training
        self.init_rnn: str = init_rnn

        # Vocabulary and stuff
        self._vocabulary: Vocabulary = vocabulary
        self.tasks: Dict[str, Task] = self._vocabulary.tasks
        self.main_task: str = main_task

        # Bidirectional so embedding_encoder_out = 2 * cemb_dim
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
            cell=self.char_cell,
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
                cell_type=self.cell,
                dropout=dropout,
                init_rnn=init_rnn
            )
        else:
            self.decoder = LinearDecoder(
                vocab_size=vocabulary.get_task_size(main_task),
                in_features=self._context_dim,
                padding_index=vocabulary.categorical_pad_token_index
            )

        self.lm = LinearDecoder(
            vocab_size=vocabulary.get_task_size("lm_token"),
            in_features=hidden_size,
            padding_index=vocabulary.categorical_pad_token_index)

        self._weights = {
            "main_task": 1.0,
            "lm_fwd": .2,
            "lm_bwd": .2,
            **{
                task: 1.0
                for task in self.linear_secondary_tasks
            }
        }
        self._watchers: Dict[str, ScoreWatcher] = {
            key: ScoreWatcher(float('inf'), main=key == "main_task")
            for key in self._weights
        }
        self.metrics = {}

    def get_main_task_gt(self, gt: Dict[str, Dict[str, torch.Tensor]], length: bool = False):
        if self.tasks[self.main_task].categorical:
            return gt["categoricals"][self.main_task]
        else:
            return gt["non_categoricals"][self.main_task]

    def _encode(
            self,
            tokens: torch.Tensor,
            tokens_length: torch.Tensor,
            sequence_length: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """

        :param tokens: Unrolled tokens (each token is one dim, outside of any sentence question)

        :returns: Encoded words, encoded sentences (Context), Secondary tasks projection (no max applied)
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

        return encoded_words, encoded_sentences, secondary_tasks

    def _compute_context(
            self,
            encoded_sentences: torch.Tensor,
            sequence_length: torch.Tensor,
            secondary_tasks: Dict[str, torch.Tensor],
    ):
        if self.use_secondary_tasks_decision:
            return flatten_padded_batch(
                torch.cat([
                    encoded_sentences,
                    *secondary_tasks.values()
                ], dim=-1),
                sequence_length
            )
        else:
            return flatten_padded_batch(
                encoded_sentences,
                sequence_length
            )

    def _decode(
            self,
            encoded_words: torch.Tensor,
            encoded_sentences: torch.Tensor,
            tokens_length: torch.Tensor,
            max_seq_len: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get_context(outs, wemb, wlen, self.tasks[task]['context'])
        if isinstance(self.decoder, AttentionalDecoder):
            # cemb_outs = F.dropout(cemb_outs, p=self.dropout, training=self.training)
            # context: torch.Size([Batch Size * Max Length, 2 * Hidden Size]) <--- PROBLEM
            # single_encoded_sentence = self._compute_context(encoded_sentences, sequence_length, secondary_tasks)
            # Use last layer enc_outs[-1]
            hyps, scores = self.decoder(
                encoded_words=encoded_words,
                lengths=tokens_length,
                encoded_sentence=encoded_sentences,
                max_seq_len=max_seq_len
            )
        else:
            logits = self.decoder(encoded=encoded_sentences)
            hyps = logits
            scores = 0
            # ToDo: Transform into hypothesis with probs ?

        return hyps, scores

    def _compute_lm(
            self,
            encoded_sentences: torch.Tensor,
            gt: torch.Tensor,
            batch_size: int
    ):
        if batch_size == 1:
            return {"loss_lm_bwd": 0, "loss_lm_fwd": 0}
        losses = {}
        # Divide the two direction of enc_outs[0]
        fwd, bwd = encoded_sentences.chunk(2, dim=2)

        # Remove the first last token and try to predict each next token (WordLevel)
        # Issue: we are not at the word level
        # Solution:
        #   1. Use grouped subwords ? But wouldn't that be weird in terms of efficiency ? #
        #          Not even sure it's possible (same problem ?)
        #   2. Use RNN and predict flat_subwords. Need to share everything though.
        losses["loss_lm_fwd"] = F.cross_entropy(
            self.lm(pad(fwd[:-1], pos='pre')).view(-1, self._vocabulary.get_task_size("lm_token")),
            gt.view(-1),
            weight=self.lm.nll_weight,
            reduction="mean",
            ignore_index=self._vocabulary.categorical_pad_token_index
        )
        # Same but previous token is the target
        losses["loss_lm_bwd"] = F.cross_entropy(
            self.lm(pad(bwd[1:], pos='post')).view(-1, self._vocabulary.get_task_size("lm_token")),
            gt.view(-1),
            weight=self.lm.nll_weight,
            reduction="mean",
            ignore_index=self._vocabulary.categorical_pad_token_index
        )
        return losses

    def _softmax_secondary_tasks(self, secondary_tasks: Dict[str, torch.Tensor]):
        return {
            task: F.softmax(out, dim=-1).max(-1)
            for task, out in secondary_tasks.items()
        }

    def _compute_secondary_loss(self, secondary_tasks: Dict[str, torch.Tensor], gt: Dict[str, Dict[str, torch.Tensor]]):
        return {
            f"loss_{task}": F.cross_entropy(
                input=secondary_tasks[task].view(-1, secondary_tasks[task].shape[-1]),
                target=gt["categoricals"][task].view(-1),
                weight=self.linear_secondary_tasks[task].nll_weight, reduction="mean",
                ignore_index=self._vocabulary.token_pad_index
            )
            for task in gt["categoricals"]  # Quickfix as LEMMA is main task now
            if task not in {self.main_task, "lm_token"}
        }

    def forward(self, batch):
        tokens, tokens_length, sequence_length = batch["token"], \
                                                 batch["token__length"], \
                                                 batch["token__sequence__length"]
        encoded_words, encoded_sentences, secondary_tasks = self._encode(tokens, tokens_length, sequence_length)
        enhanced_context = self._compute_context(encoded_sentences[-1], sequence_length, secondary_tasks)
        if isinstance(self.decoder, AttentionalDecoder):
            out = self.decoder(
                encoded_tokens=encoded_words,
                tokens_length=tokens_length,
                encoded_sentence=enhanced_context
            )
        else:
            raise NotImplementedError("ToDo: Implement linear as main task")
            main_loss = self.decoder.loss()

        (_, preds), _, _, secondary_tasks = self._encode(tokens, tokens_length, sequence_length)
        preds = preds.transpose(1, 0)
        return preds, {task: prediction.transpose(1, 0) for task, prediction in secondary_tasks.items()}

    def compute_accuracy(self, preds, secondary_tasks, targets, sequences_lengths: torch.Tensor):
        preds = pad_sequence(
            [torch.tensor(pred, device="cpu", requires_grad=False) for pred in preds],
            padding_value=self._vocabulary.token_pad_index,
            batch_first=True
        )
        returns: Dict[str, List[Tuple[str, str]]] = {
            task: []
            for task in self.tasks
            if "lm_token" != task
        }
        for task in self.tasks.values():
            if task.name == "lm_token":
                continue

            if task.categorical:
                gt = targets["categoricals"][task.name].cpu()
            else:
                gt = targets["non_categoricals"][task.name].transpose(1, 0).cpu()

            if task.name == self.main_task:
                out = preds.cpu()
                # If the length of the targets was not reached, create an empty tensor of the size of gt and add preds
                if not task.categorical and preds.shape[1] < gt.shape[1]:
                    out = torch.full(gt.shape, self._vocabulary.token_pad_index, device="cpu", requires_grad=False)
                    out[:, :preds.shape[1]] = preds  # Replace value with pad value
            # If we have a a non main task
            else:
                _, out = secondary_tasks[task.name]  # First is probability
                out = out.cpu()

            if task.categorical:
                out, gt = flatten_padded_batch(out, sequences_lengths), \
                          flatten_padded_batch(gt.cpu(), sequences_lengths)

                returns[task.name].extend(list(zip(self._vocabulary.decode(out.tolist(), task=task.name),
                                                   self._vocabulary.decode(gt.tolist(), task=task.name))))
            else:
                # For non-categorical, specifically for lemma, let's make sure this computes the token level
                #   accuracy
                decoded_out, decoded_gt = self._vocabulary.tokenizer.decode_batch(out.tolist()), \
                          self._vocabulary.tokenizer.decode_batch(gt.tolist())
                returns[task.name].extend(list(zip(decoded_out, decoded_gt)))
                local_encoder = list(set(decoded_out + decoded_gt))
                out, gt = torch.tensor([local_encoder.index(val) for val in decoded_out],
                                       device="cpu", requires_grad=False), \
                          torch.tensor([local_encoder.index(val) for val in decoded_gt],
                                       device="cpu", requires_grad=False)

            if task.name in self.metrics:
                for key in self.metrics[task.name]:
                    self.metrics[task.name][key](out, gt)

        return returns

    def _finalize_metrics(self, pred_dicts: Dict[str, Tuple[List[str], List[str]]],
                          log_kwargs: Optional[Dict[str, Any]] = None):
        if not log_kwargs:
            log_kwargs = {}
        for task in self.tasks.values():

            if task.name == "lm_token":
                continue

            # Compute Score "à la Pie"
            preds, trues = pred_dicts[task.name]
            scores = compute_scores(np.array(trues, dtype="object"), np.array(preds, dtype="object"))
            for key in scores:
                if key == "sup":
                    continue
                self.log(f"{task.name}_{key}", scores[key], **log_kwargs)
            if task.name in self.metrics:
                for key in self.metrics[task.name]:
                    if key == "stats":
                        continue
                    self.log(f'{task.name}_{key}', self.metrics[task.name][key].compute())
                    self.metrics[task.name][key].reset()

    def training_step(self, batch, batch_idx):
        x, gt = batch
        tokens, tokens_length, sequence_length = x["token"], x["token__length"], x["token__sequence__length"]

        assert sum(sequence_length).item() == tokens.shape[1], "Number of words accross sentence should match " \
                                                               "unrolled tokens"

        encoded_words, encoded_sentences, secondary_tasks = self._encode(tokens, tokens_length, sequence_length)
        enhanced_context = self._compute_context(encoded_sentences[-1], sequence_length, secondary_tasks)

        if isinstance(self.decoder, AttentionalDecoder):
            gt_tokens, gt_tokens_length = gt["non_categoricals"][self.main_task], \
                                          gt["non_categoricals"][self.main_task + "__length"]
            main_loss = self.decoder.loss(
                ground_truth=gt_tokens,
                ground_truth_lengths=gt_tokens_length,
                encoded_tokens=encoded_words,
                tokens_length=tokens_length,
                encoded_sentence=enhanced_context
            )
        else:
            raise NotImplementedError("ToDo: Implement linear as main task")
            main_loss = self.decoder.loss()

        batch_size = batch[0]["token__sequence__length"].shape[0]

        losses = {
            "loss_main_task": main_loss
        }
        secondary_tasks = {
            **self._compute_secondary_loss(secondary_tasks, gt),
            **self._compute_lm(encoded_sentences[0], gt["categoricals"]["lm_token"], batch_size=batch_size)
        }

        self.log("train_main_loss", main_loss, batch_size=batch_size)
        for task in secondary_tasks:
            self.log("train_" + task, secondary_tasks[task], batch_size=batch_size)

        loss = sum([
            self._weights.get(loss_name[5:], 1) * loss_value  # Remove "loss_" to get task name
            for loss_name, loss_value in {**losses, **secondary_tasks}.items()
        ])

        return loss

    def validation_step(self, batch, batch_idx) -> Tuple[Dict[str, torch.Tensor], int]:
        """

        :returns: Dictionary of losses per task and batch size
        """
        x, gt = batch
        tokens, tokens_length, sequence_length = x["token"], x["token__length"], x["token__sequence__length"]

        assert sum(sequence_length).item() == tokens.shape[1], "Number of words accross sentence should match " \
                                                               "unrolled tokens"

        encoded_words, encoded_sentences, secondary_tasks = self._encode(tokens, tokens_length, sequence_length)
        enhanced_context = self._compute_context(encoded_sentences[-1], sequence_length, secondary_tasks)

        if isinstance(self.decoder, AttentionalDecoder):
            gt_tokens, gt_tokens_length = gt["non_categoricals"][self.main_task], \
                                          gt["non_categoricals"][self.main_task + "__length"]
            main_loss = self.decoder.loss(
                ground_truth=gt_tokens,
                ground_truth_lengths=gt_tokens_length,
                encoded_tokens=encoded_words,
                tokens_length=tokens_length,
                encoded_sentence=enhanced_context
            )
            hyps, scores = self.decoder(
                encoded_tokens=encoded_words,
                tokens_length=tokens_length,
                encoded_sentence=enhanced_context,
                max_seq_len=gt_tokens_length.max()
            )
        else:
            raise NotImplementedError("ToDo: Implement linear as main task")
            main_loss = self.decoder.loss()

        batch_size = batch[0]["token__sequence__length"].shape[0]

        losses = {
            "loss_main_task": main_loss
        }
        secondary_losses = {
            **self._compute_secondary_loss(secondary_tasks, gt),
            **self._compute_lm(encoded_sentences[0], gt["categoricals"]["lm_token"], batch_size=batch_size)
        }

        self.log("dev_main_loss", main_loss, batch_size=batch_size)
        for task in secondary_losses:
            self.log("dev_" + task, secondary_losses[task], batch_size=batch_size,
                     prog_bar=True if "lm_" not in task else False)

        preds = self.compute_accuracy(hyps, self._softmax_secondary_tasks(secondary_tasks), gt, sequence_length)

        return {**losses, **secondary_losses}, batch_size, preds

    def validation_epoch_end(self, outputs=None) -> Dict[str, torch.Tensor]:
        # outputs: List of (losses, batch_size, preds)

        #sortir preds de outputs, ainsi que batch_size et losses :
        losses, batch_size, preds = zip(*outputs)
        if not isinstance(outputs, List):
            outputs = [outputs]

        self._finalize_metrics(self._collate_steps_pred_gt(preds), log_kwargs={"prog_bar": True})

        nb_batch = sum(batch_size)

        avg_loss = {
            task: sum([loss_dict[task]*b_size for loss_dict, b_size in zip(losses, batch_size)]) / nb_batch
            for task in losses[0].keys()
        }

        print(avg_loss)

        for key in avg_loss:
            self_key = key[5:]
            _, self._weights[self_key] = self._watchers[self_key].update_steps_on_mode(
                avg_loss[key],
                self._weights[self_key],
                task=self_key
            )

        print()
        for task in self._watchers:
            print(self._watchers[task].repr(task, self._weights[task]))

        return avg_loss

    def test_step(self, batch, batch_idx):
        x, gt = batch
        tokens, tokens_length, sequence_length = x["token"], x["token__length"], x["token__sequence__length"]

        assert sum(sequence_length).item() == tokens.shape[1], "Number of words accross sentence should match " \
                                                               "unrolled tokens"

        encoded_words, encoded_sentences, secondary_tasks = self._encode(tokens, tokens_length, sequence_length)
        enhanced_context = self._compute_context(encoded_sentences[-1], sequence_length, secondary_tasks)

        if isinstance(self.decoder, AttentionalDecoder):
            gt_tokens, gt_tokens_length = gt["non_categoricals"][self.main_task], \
                                          gt["non_categoricals"][self.main_task + "__length"]
            hyps, scores = self.decoder(
                encoded_tokens=encoded_words,
                tokens_length=tokens_length,
                encoded_sentence=enhanced_context,
                max_seq_len=gt_tokens_length.max()
            )
        else:
            raise NotImplementedError("ToDo: Implement linear as main task")
            main_loss = self.decoder.loss()

        preds = self.compute_accuracy(hyps, self._softmax_secondary_tasks(secondary_tasks), gt, sequence_length)
        return preds

    def _collate_steps_pred_gt(
            self, data: List[Dict[str, List[Tuple[str, str]]]]) -> Dict[str, Tuple[List[str], List[str]]]:
        returns = {
            key: ([], [])
            for key in data[0].keys()
        }
        for step in data:
            for task_name in step:
                out, gt = zip(*step[task_name])
                returns[task_name][0].extend(out)
                returns[task_name][1].extend(gt)
        return returns

    def test_epoch_end(self, outputs) -> None:
        if not isinstance(outputs, list):
            outputs = [outputs]

        self._finalize_metrics(self._collate_steps_pred_gt(outputs))

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            threshold=1e-3,
            verbose=True,
            factor=.6,
            patience=2,
            min_lr=1e-6
        )
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "lemma_pre",
                    "frequency": 1
                }
            ]
        )
