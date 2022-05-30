from typing import List, Tuple
from operator import itemgetter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tokenizers import Tokenizer, Encoding

from tantal.tokens.align import get_word_groups


class GroundTruthDataset(Dataset):
    def __init__(self, annotations_file: str, task: str, tokenizer: Tokenizer, categorical: bool = False):
        self.annotation_file: str = annotations_file
        self.task: str = task
        self.tokenizer: Tokenizer = tokenizer
        self.categorical: bool = categorical
        self.annotations: List[List[Tuple[str, str]]] = list(self._read())
        self.pad_index = self.tokenizer.token_to_id("[PAD]")
        self.bos_index = self.tokenizer.token_to_id("[BOS]")
        self.eos_index = self.tokenizer.token_to_id("[EOS]")

    def _read(self):
        if self.annotation_file.endswith(".tsv"):
            return self._read_tsv(self.annotation_file, self.task)
        raise ValueError("Unsupported data format")

    def tokenized_to_output(self, data: Encoding):
        grouped_subwords = get_word_groups(data, bos=self.bos_index, eos=self.eos_index)
        return (
            (data.ids, len(data.ids)),
            # ToDo: Should we add EOS everytime ?
            (grouped_subwords, [len(grouped) for grouped in grouped_subwords])
        )

    @staticmethod
    def _read_tsv(filepath, task):
        with open(filepath) as f:
            sentence = []
            headers = []
            for line_idx, line in enumerate(f.read()):
                line = line.strip().split("\t")
                if line_idx == 0:
                    headers = line
                elif not line and sentence:
                    yield sentence
                    sentence = []
                else:
                    line = dict(zip(headers, line))
                    try:
                        sentence.append((line["token"], line[task]))
                    except KeyError:
                        raise  # Custom one day ?

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        forms, tasks = zip(self.annotations[idx])
        forms = self.tokenizer.encode(forms, is_pretokenized=True)
        if not self.categorical:
            raise NotImplementedError
        else:
            annots = self.tokenizer.encode(tasks, is_pretokenized=True)
            return self.tokenized_to_output(forms), self.tokenized_to_output(annots)

    def train_collate_fn(self, batch):
        """
        DataLoaderBatch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        # ToDo
        x, x_length, y, y_length, _ = list(zip(*sorted(batch, key=itemgetter(1), reverse=True)))
        return (
            pad_sequence(
                [torch.tensor(x_i) for x_i in x],
                padding_value=self.pad_index,
                batch_first=True
            ),
            torch.tensor(x_length),
            pad_sequence(
                [torch.tensor(y_i) for y_i in y],
                padding_value=self.pad_index,
                batch_first=True
            )
        )
