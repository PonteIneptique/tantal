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
            (grouped_subwords, [len(grouped) for grouped in grouped_subwords])
        )

    def pad_batch(self, x):
        """

        :info: Padded with value first, batch last
        """
        flat_subwords, fsw_len, grouped_subwords, gsw_len = zip(x)
        # torch.Tensor(Sequence Length * Batch Size)
        flat_subwords = pad_sequence(
            sequences=[torch.tensor(sw, dtype=torch.int8) for sw in flat_subwords],
            padding_value=self.pad_index
        )
        # torch.Tensor(Max word length, Sum(Word Count in Batch))
        # -> First dimension is the maximum word length
        # -> Second dimension is how many word there are in total in the batch
        grouped_subwords = pad_sequence(
            sequences=[
                torch.tensor(sw, dtype=torch.int8)
                for sentence_subwords in grouped_subwords
                for sw in sentence_subwords
            ],
            padding_value=self.pad_index
        )
        return flat_subwords, torch.tensor(fsw_len, dtype=torch.int8), grouped_subwords, torch.tensor([
            token_length for sentence_level in gsw_len for token_length in sentence_level
        ])

    @staticmethod
    def _read_tsv(filepath, task):
        with open(filepath) as f:
            sentence = []
            headers = []
            for line_idx, line in enumerate(f):
                line = line.strip().split("\t")
                if line_idx == 0:
                    headers = line
                elif not [col for col in line if col]:
                    if sentence:
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
        x, y = batch
        return self.pad_batch(x), self.pad_batch(y)
