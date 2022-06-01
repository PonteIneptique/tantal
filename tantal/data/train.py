from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tokenizers import Tokenizer, Encoding

from tantal.data.vocabulary import Task, Vocabulary, get_word_groups


class GroundTruthDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        vocabulary: Vocabulary
    ):
        self.annotation_file: str = annotations_file
        self.vocabulary: Vocabulary = vocabulary
        self.tasks: Dict[str, Task] = vocabulary.tasks
        self.tokenizer: Tokenizer = vocabulary.tokenizer

        self.annotations: List[Dict[str, List[str]]] = list(self._read())
        self.pad_index = self.tokenizer.token_to_id("[PAD]")
        self.bos_index = self.tokenizer.token_to_id("[BOS]")
        self.eos_index = self.tokenizer.token_to_id("[EOS]")

    def _read(self):
        if self.annotation_file.endswith(".tsv"):
            return self._read_tsv(self.annotation_file, list(self.tasks.keys()))
        raise ValueError("Unsupported data format")

    def tokenized_to_output(self, data: Encoding):
        grouped_subwords = get_word_groups(data, bos=self.bos_index, eos=self.eos_index)
        return (
            (data.ids, len(data.ids)),
            (grouped_subwords, [len(grouped) for grouped in grouped_subwords], len(grouped_subwords))
        )

    def pad_batch(self, x):
        """

        :info: Padded with value first, batch last
        """
        flat_subwords, grouped_subwords = zip(*x)
        (flat_subwords, fsw_len) = zip(*flat_subwords)
        (grouped_subwords, gsw_len, nb_words) = zip(*grouped_subwords)
        # torch.Tensor(Sequence Length * Batch Size)
        flat_subwords = pad_sequence(
            sequences=[torch.tensor(sw, dtype=torch.long) for sw in flat_subwords],
            padding_value=self.pad_index
        )
        # torch.Tensor(Max word length, Sum(Word Count in Batch))
        # -> First dimension is the maximum word length
        # -> Second dimension is how many word there are in total in the batch
        grouped_subwords = pad_sequence(
            sequences=[
                torch.tensor(sw, dtype=torch.long)
                for sentence_subwords in grouped_subwords
                for sw in sentence_subwords
            ],
            padding_value=self.pad_index
        )
        return (
            (flat_subwords, torch.tensor(fsw_len, dtype=torch.int)),
            (
                grouped_subwords,
                torch.tensor([
                    token_length for sentence_level in gsw_len for token_length in sentence_level
                ], dtype=torch.int),
                torch.tensor(nb_words, dtype=torch.int)
            )
        )

    def fit_vocab(self):
        self.vocabulary.build_from_sentences(self.annotations)

    @staticmethod
    def _read_tsv(filepath, tasks):
        with open(filepath) as f:
            sentence = []
            headers = []
            for line_idx, line in enumerate(f):
                line = line.strip().split("\t")
                if line_idx == 0:
                    headers = line
                elif not [col for col in line if col]:
                    if sentence:
                        yield {
                            task: [token[task] for token in sentence]
                            for task in sentence[0]
                        }
                        sentence = []
                else:
                    line = dict(zip(headers, line))
                    try:
                        sentence.append({
                            "token": line["token"],
                            "lm_token": line["token"],
                            **{
                                task: line[task]
                                for task in tasks
                                if task != "lm_token"
                            }
                        })
                    except KeyError:
                        raise  # Custom one day ?

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sentence: Dict[str, List[str]] = self.annotations[idx]

        tokens, token_length = self.vocabulary.encode_input(sentence["token"])

        categoricals = {
            task: self.vocabulary.encode(sentence[task], task=task)
            for task in self.vocabulary.tasks
            if task != "token" and self.tasks[task].categorical
        }

        non_categoricals = [
            [
                [task, f"{task}__length"],
                *self.vocabulary.encode(sentence[task], task=task)
            ]
            for task in self.vocabulary.tasks
            if task != "token" and not self.tasks[task].categorical
        ]

        return {"token": tokens, "wordtoken__length": token_length}, categoricals, non_categoricals

    def collate_fn(self, batch):
        """
        DataLoaderBatch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        x, y = zip(*batch)
        return self.pad_batch(x), self.pad_batch(y)
